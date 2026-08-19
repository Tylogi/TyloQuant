"""Torch GPU backend for NINT dequantization plus matmul.

Following llama.cpp's memory strategy, weights remain compressed on the GPU (q plus scales, about 4.5 bpw).
Each forward pass temporarily dequantizes the current layer to fp16 for cuBLAS GEMM instead of keeping the full model
resident in fp16 (a 27B model needs 54 GB in fp16 and does not fit on a 24 GB card, while ~15 GB compressed does).

Matmul uses **llama.cpp-style decomposition** with grouped-sum correction for the affine zero point:

    y[b,o] = Σ_g d_eff[o,g]·(Σ_{i∈g} q[o,i]·x[b,i]) − Σ_g m_eff[o,g]·(Σ_{i∈g} x[b,i])
           = (Wq · xᵀ)[o,b] − (m_eff · xsᵀ)[o,b]

Here ``Wq = d_eff*q`` is the dequantized weight without zp, and ``xs`` is the per-group sum of x.
The current basic implementation materializes Wq and uses cuBLAS. A true INT-fused-GEMM that dequantizes during GEMM
without materializing Wq remains for a future CUDA extension using torch.utils.cpp_extension, with no upper-layer changes.

This module imports torch, a heavy dependency, so ``mfq.kernels.__init__`` does not import it automatically; import it explicitly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from mfq.formats.nint import NintTensor


def _pack_q4(q: torch.Tensor) -> torch.Tensor:
    return (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()


def _pack_qbits(q, bits: int, device: str | torch.device) -> torch.Tensor:
    bits = int(bits)
    if bits == 4:
        return _pack_q4(torch.as_tensor(q, device=device))
    q_np = np.ascontiguousarray(np.asarray(q), dtype=np.uint8)
    shape = q_np.shape
    qbytes = (int(shape[-1]) * bits + 7) // 8
    if bits == 8:
        packed = q_np
    elif 1 <= bits < 8:
        rows = q_np.reshape(-1, shape[-1])
        bit_rows = np.unpackbits(rows[..., None], axis=-1, bitorder="little")[..., :bits]
        packed = np.packbits(bit_rows.reshape(rows.shape[0], -1), axis=1, bitorder="little")
    else:
        raise ValueError(f"unsupported NINT bits for CUDA runtime: {bits}")
    packed = np.ascontiguousarray(packed.reshape(*shape[:-1], qbytes), dtype=np.uint8)
    return torch.as_tensor(packed, device=device)


def _eff_pair_h(tensor: NintTensor, device: str | torch.device) -> torch.Tensor:
    sub_scale = torch.as_tensor(tensor.sub_scale, device=device)
    sub_min = torch.as_tensor(tensor.sub_min, device=device)
    neuron_scale = torch.as_tensor(tensor.neuron_scale, device=device)
    neuron_min = torch.as_tensor(tensor.neuron_min, device=device)
    d_eff = neuron_scale[:, None] * sub_scale.to(torch.float32)
    m_eff = neuron_min[:, None] * sub_min.to(torch.float32)
    return torch.stack((d_eff, m_eff), dim=-1).to(torch.float16).contiguous()


def _unpack_q4(q_packed: torch.Tensor, gs: int) -> torch.Tensor:
    q = torch.empty((*q_packed.shape[:-1], gs), device=q_packed.device, dtype=torch.uint8)
    q[..., 0::2] = q_packed & 0x0F
    q[..., 1::2] = q_packed >> 4
    return q


def _unpack_qbits(q_packed: torch.Tensor, gs: int, bits: int) -> torch.Tensor:
    """Unpack one little-endian NINT group on any Torch device.

    CUDA keeps its fused unpack path.  This portable implementation is used by
    Metal/MPS calibration, where packed candidates are decoded only while a
    layer is resident.
    """

    bits = int(bits)
    gs = int(gs)
    if bits == 4:
        return _unpack_q4(q_packed, gs)
    if bits == 8:
        return q_packed[..., :gs].contiguous()
    if bits <= 0 or bits >= 8:
        raise ValueError(f"unsupported NINT bit width: {bits}")
    bit_offsets = torch.arange(gs, device=q_packed.device, dtype=torch.int64) * bits
    byte_offsets = torch.div(bit_offsets, 8, rounding_mode="floor")
    shifts = (bit_offsets % 8).to(torch.int32)
    padded = torch.nn.functional.pad(q_packed, (0, 1))
    low = padded.index_select(-1, byte_offsets).to(torch.int32)
    high = padded.index_select(-1, byte_offsets + 1).to(torch.int32)
    words = low | (high << 8)
    return ((words >> shifts) & ((1 << bits) - 1)).to(torch.uint8)


def nint_deploy_arrays(tensor: NintTensor) -> dict[str, np.ndarray]:
    """Build execution-ready CPU arrays without the unpacked quantized values."""

    q_packed = _pack_qbits(tensor.q, tensor.spec.bits, "cpu")
    return {
        "q_packed": np.ascontiguousarray(q_packed.numpy(), dtype=np.uint8),
        "sub_scale": np.ascontiguousarray(tensor.sub_scale),
        "sub_min": np.ascontiguousarray(tensor.sub_min),
        "neuron_scale": np.ascontiguousarray(tensor.neuron_scale, dtype=np.float32),
        "neuron_min": np.ascontiguousarray(tensor.neuron_min, dtype=np.float32),
    }


def nint_deploy_to_gpu(
    arrays: Mapping[str, np.ndarray],
    *,
    bits: int,
    groupsize: int,
    neuron_len: int,
    shape: Sequence[int],
    axis: int,
    device: str | torch.device,
) -> dict:
    """Upload execution-ready NINT arrays without CPU bit unpack/repack."""

    required = {"q_packed", "sub_scale", "sub_min", "neuron_scale", "neuron_min"}
    if set(arrays) != required:
        raise ValueError("NINT deploy arrays do not contain the required fields")
    q_packed = np.asarray(arrays["q_packed"])
    sub_scale = np.asarray(arrays["sub_scale"])
    sub_min = np.asarray(arrays["sub_min"])
    neuron_scale = np.asarray(arrays["neuron_scale"])
    neuron_min = np.asarray(arrays["neuron_min"])
    if q_packed.ndim != 3:
        raise ValueError("packed NINT values must have [out, groups, bytes] shape")
    out, ng, qbytes = (int(value) for value in q_packed.shape)
    expected_qbytes = (int(groupsize) * int(bits) + 7) // 8
    if qbytes != expected_qbytes:
        raise ValueError(f"packed NINT group has {qbytes} bytes; expected {expected_qbytes}")
    if sub_scale.shape != (out, ng) or sub_min.shape != (out, ng):
        raise ValueError("packed NINT sub-scale shape mismatch")
    if neuron_scale.shape != (out,) or neuron_min.shape != (out,):
        raise ValueError("packed NINT neuron-scale shape mismatch")
    return {
        "q_packed": torch.as_tensor(q_packed, device=device),
        "sub_scale": torch.as_tensor(sub_scale, device=device),
        "sub_min": torch.as_tensor(sub_min, device=device),
        "neuron_scale": torch.as_tensor(neuron_scale, device=device),
        "neuron_min": torch.as_tensor(neuron_min, device=device),
        "bits": int(bits),
        "out": out,
        "ng": ng,
        "gs": int(groupsize),
        "neuron_len": int(neuron_len),
        "shape": tuple(int(value) for value in shape),
        "axis": int(axis),
        "device": device,
    }


def _d_eff(g: dict) -> torch.Tensor:
    if g.get("d_eff") is not None:
        return g["d_eff"]
    if g.get("eff_pair_h") is not None:
        return g["eff_pair_h"][..., 0].to(torch.float32)
    return g["neuron_scale"][:, None] * g["sub_scale"].to(torch.float32)


def _m_eff(g: dict) -> torch.Tensor:
    if g.get("m_eff") is not None:
        return g["m_eff"]
    if g.get("eff_pair_h") is not None:
        return g["eff_pair_h"][..., 1].to(torch.float32)
    return g["neuron_min"][:, None] * g["sub_min"].to(torch.float32)


def to_gpu(
    tensor: NintTensor,
    device: str | torch.device = "cuda",
    *,
    layout: str | None = None,
) -> dict:
    """Move NINT execution metadata to GPU.

    ``layout="deploy"`` is the default runtime path and keeps compact NINT
    metadata resident. ``layout="experimental"`` builds the older extra
    layouts for kernel experiments.
    """

    if layout is None:
        layout = os.environ.get("MFQ_NINT_LAYOUT", "deploy")
    q = torch.as_tensor(tensor.q, device=device)
    bits = int(tensor.spec.bits)
    q_packed = _pack_qbits(tensor.q, bits, device)
    sub_scale = torch.as_tensor(tensor.sub_scale, device=device)
    sub_min = torch.as_tensor(tensor.sub_min, device=device)
    neuron_scale = torch.as_tensor(tensor.neuron_scale, device=device)
    neuron_min = torch.as_tensor(tensor.neuron_min, device=device)

    g = {
        "q_packed": q_packed,                                                # [out,ng,gs/2] uint8 for INT4
        "sub_scale": sub_scale,                                              # [out,ng] uint8
        "sub_min": sub_min,                                                  # [out,ng] uint8
        "neuron_scale": neuron_scale,                                        # [out] f32
        "neuron_min": neuron_min,                                            # [out] f32
        "bits": bits,
        "out": tensor.q.shape[0],
        "ng": tensor.q.shape[1],
        "gs": tensor.spec.groupsize,
        "neuron_len": tensor.neuron_len,
        "shape": tensor.shape,
        "axis": tensor.axis,
        "device": device,
    }
    if layout == "deploy":
        del q
        return g
    if layout != "experimental":
        raise ValueError(f"unknown NINT GPU layout: {layout!r}")

    eff_pair_h = _eff_pair_h(tensor, device)
    d_eff = neuron_scale[:, None] * sub_scale.to(torch.float32)
    m_eff = neuron_min[:, None] * sub_min.to(torch.float32)
    if bits != 4:
        g.update({
            "q": q,
            "eff_pair_h": eff_pair_h,
            "d_eff": d_eff,
            "m_eff": m_eff,
            "d_eff_h": d_eff.to(torch.float16).contiguous(),
            "m_eff_h": m_eff.to(torch.float16).contiguous(),
        })
        return g

    q_mmq_packed = None
    sub_scale_mmq = None
    sub_min_mmq = None
    d_eff_mmq = None
    m_eff_mmq = None
    gs = int(tensor.spec.groupsize)
    if gs == 16:
        gpk = 16
    elif gs == 24:
        gpk = 10
    elif gs == 32:
        gpk = 8
    elif gs == 48:
        gpk = 5
    else:
        raise ValueError(f"unsupported NINT groupsize {gs}")

    out, ng, qbytes = q_packed.shape
    mmq_y = 64
    ntiles = (out + mmq_y - 1) // mmq_y
    nchunks = (ng + gpk - 1) // gpk
    out_pad = ntiles * mmq_y
    ng_pad = nchunks * gpk

    q_pad = torch.zeros((out_pad, ng_pad, qbytes), device=device, dtype=q_packed.dtype)
    q_pad[:out, :ng, :] = q_packed
    q_mmq_packed = (
        q_pad.reshape(ntiles, mmq_y, nchunks, gpk, qbytes)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(ntiles, nchunks, mmq_y, gpk * qbytes)
    )
    ss_pad = torch.zeros((out_pad, ng_pad), device=device, dtype=sub_scale.dtype)
    sm_pad = torch.zeros((out_pad, ng_pad), device=device, dtype=sub_min.dtype)
    ss_pad[:out, :ng] = sub_scale
    sm_pad[:out, :ng] = sub_min
    sub_scale_mmq = ss_pad.reshape(ntiles, mmq_y, nchunks, gpk).permute(0, 2, 1, 3).contiguous()
    sub_min_mmq = sm_pad.reshape(ntiles, mmq_y, nchunks, gpk).permute(0, 2, 1, 3).contiguous()
    de_pad = torch.zeros((out_pad, ng_pad), device=device, dtype=torch.float32)
    me_pad = torch.zeros((out_pad, ng_pad), device=device, dtype=torch.float32)
    de_pad[:out, :ng] = d_eff
    me_pad[:out, :ng] = m_eff
    d_eff_mmq = de_pad.reshape(ntiles, mmq_y, nchunks, gpk).permute(0, 2, 1, 3).contiguous()
    m_eff_mmq = me_pad.reshape(ntiles, mmq_y, nchunks, gpk).permute(0, 2, 1, 3).contiguous()

    g.update({
        "q": q,                                                              # [out,ng,gs] uint8
        "eff_pair_h": eff_pair_h,                                             # [out,ng,2] f16, d/m packed
        "q_mmq_packed": q_mmq_packed,                                        # [ntile,nchunk,64,gpk*gs/2] uint8
        "d_eff": d_eff,                                                       # [out,ng] f32, execution metadata
        "m_eff": m_eff,                                                       # [out,ng] f32, execution metadata
        "d_eff_h": d_eff.to(torch.float16).contiguous(),                      # [out,ng] f16, compact execution metadata
        "m_eff_h": m_eff.to(torch.float16).contiguous(),                      # [out,ng] f16, compact execution metadata
        "sub_scale_mmq": sub_scale_mmq,                                      # [ntile,nchunk,64,gpk] uint8
        "sub_min_mmq": sub_min_mmq,                                          # [ntile,nchunk,64,gpk] uint8
        "d_eff_mmq": d_eff_mmq,                                              # [ntile,nchunk,64,gpk] f32
        "m_eff_mmq": m_eff_mmq,                                              # [ntile,nchunk,64,gpk] f32
    })
    return g


def dequantize(g: dict, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Fully dequantize to ``dtype`` (fp16 by default) and restore the original shape."""

    if g.get("q") is not None:
        q = g["q"].to(torch.float32)
    else:
        bits = int(g.get("bits", 4))
        if bits != 4 and g["q_packed"].device.type == "cuda":
            from mfq.kernels.cuda._ext import ext

            return ext().nint_dequant_full_packed_compact_bits_cuda(
                g["q_packed"],
                g["sub_scale"],
                g["sub_min"],
                g["neuron_scale"],
                g["neuron_min"],
                int(g["neuron_len"]),
                int(g["gs"]),
                bits,
            ).to(dtype)
        q = _unpack_qbits(g["q_packed"], int(g["gs"]), bits).to(torch.float32)
    recon = (_d_eff(g)[:, :, None] * q - _m_eff(g)[:, :, None])   # [out,ng,gs]
    recon = recon.reshape(g["out"], -1)[:, : g["neuron_len"]]     # [out, neuron_len]
    S, a = g["shape"], g["axis"]
    wt_shape = (S[a],) + S[:a] + S[a + 1:]
    return torch.moveaxis(recon.reshape(wt_shape), 0, a).reshape(S).to(dtype)


def matmul(g: dict, x: torch.Tensor) -> torch.Tensor:
    """llama.cpp-style decomposed matmul: ``y = x * Wq^T - xs * m_eff^T``.

    ``x``: ``[..., in]`` in fp16/f32 and already on the GPU. Returns ``[..., out]`` in fp16.
    """

    out, ng, gs = g["out"], g["ng"], g["gs"]
    if g.get("q") is None and int(g.get("bits", 4)) != 4:
        return x.to(torch.float16) @ dequantize(g).T
    q = g["q"].to(torch.float32) if g.get("q") is not None else _unpack_q4(g["q_packed"], gs).to(torch.float32)
    Wq = (_d_eff(g)[:, :, None] * q).reshape(out, ng * gs)[:, : g["neuron_len"]]
    Wq = Wq.to(torch.float16)

    xb = x.to(torch.float16)
    xin = xb.shape[-1]
    if xin != ng * gs:
        xb = torch.nn.functional.pad(xb, (0, ng * gs - xin))   # Zero-pad the trailing group
    xs = xb.reshape(*xb.shape[:-1], ng, gs).sum(-1).to(torch.float16)   # [..., ng]
    y = xb[..., :xin].to(torch.float16) @ Wq.T                          # [..., out]
    y = y - xs @ _m_eff(g).to(torch.float16).T                          # Zero-point correction
    return y
