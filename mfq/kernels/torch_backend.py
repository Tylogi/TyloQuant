"""torch GPU backend：NINT 反量化 + matmul。

仿 llama.cpp 的内存策略：权重以**压缩形式**（q + scales，~4.5 bpw）常驻 GPU，每次前向
把当前层权重瞬时 dequant 成 fp16 做 cuBLAS GEMM——不常驻全模型 fp16（27B 模型 fp16
需 54GB，24GB 卡装不下；压缩态 ~15GB 装得下）。

matmul 走 **llama.cpp 式分解**（仿射零点的分组求和校正）：

    y[b,o] = Σ_g d_eff[o,g]·(Σ_{i∈g} q[o,i]·x[b,i]) − Σ_g m_eff[o,g]·(Σ_{i∈g} x[b,i])
           = (Wq · xᵀ)[o,b] − (m_eff · xsᵀ)[o,b]

其中 ``Wq = d_eff·q``（不带 zp 的反量化权重），``xs`` 是 x 的 per-group 求和。
当前为「Wq materialize + cuBLAS」的基本实现；**真正的 INT-fused-GEMM（dequant-during-GEMM，
不 materialize Wq）**留作后续 CUDA 扩展（torch.utils.cpp_extension），届时上层零改动。

注：本模块 import torch（重依赖），故不经 ``mfq.kernels.__init__`` 自动导入，需显式引入。
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from mfq.quantize.nint_quant import NintTensor


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
    """全量反量化为 ``dtype``（默认 fp16），还原原始 shape。"""

    if g.get("q") is not None:
        q = g["q"].to(torch.float32)
    else:
        bits = int(g.get("bits", 4))
        if bits != 4:
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
        q = _unpack_q4(g["q_packed"], int(g["gs"])).to(torch.float32)
    recon = (_d_eff(g)[:, :, None] * q - _m_eff(g)[:, :, None])   # [out,ng,gs]
    recon = recon.reshape(g["out"], -1)[:, : g["neuron_len"]]     # [out, neuron_len]
    S, a = g["shape"], g["axis"]
    wt_shape = (S[a],) + S[:a] + S[a + 1:]
    return torch.moveaxis(recon.reshape(wt_shape), 0, a).reshape(S).to(dtype)


def matmul(g: dict, x: torch.Tensor) -> torch.Tensor:
    """llama.cpp 式分解 matmul：``y = x · Wqᵀ − xs · m_effᵀ``。

    ``x``: ``[..., in]`` (fp16/f32, 已在 GPU)。返回 ``[..., out]`` (fp16)。
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
        xb = torch.nn.functional.pad(xb, (0, ng * gs - xin))   # 尾组补零
    xs = xb.reshape(*xb.shape[:-1], ng, gs).sum(-1).to(torch.float16)   # [..., ng]
    y = xb[..., :xin].to(torch.float16) @ Wq.T                          # [..., out]
    y = y - xs @ _m_eff(g).to(torch.float16).T                          # zp 校正
    return y
