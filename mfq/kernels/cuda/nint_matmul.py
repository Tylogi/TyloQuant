"""NINT fused GEMM glue (kernel in nint_matmul.cu, built into the mfq_cuda extension).

``nint_matmul(g, x)`` is signature/semantics-compatible with
:func:`mfq.kernels.torch_backend.matmul` (``y = x . W^T``), but dequantization happens per
group inside the GEMM with no Wq materialization. ``g`` is the GPU dict from
:func:`mfq.kernels.torch_backend.to_gpu`.
"""

from __future__ import annotations

import os

import torch

from mfq.kernels.cuda._ext import ext


def _workspace(g: dict, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reusable qx/xscale/xsum buffers for packed MMVQ/MMQ paths."""

    M = int(x.shape[0])
    K_pad = int(g["ng"]) * int(g["gs"])
    ng = int(g["ng"])
    key = (str(x.device), M, K_pad)
    ws = g.setdefault("_workspace", {})
    cached = ws.get(key)
    if cached is not None:
        qx, xscale, xsum = cached
        if qx.device == x.device and xscale.device == x.device and xsum.device == x.device:
            return qx, xscale, xsum
    qx = torch.empty((M, K_pad), device=x.device, dtype=torch.int8)
    xscale = torch.empty((M, ng), device=x.device, dtype=torch.float32)
    xsum = torch.empty((M, ng), device=x.device, dtype=torch.int32)
    ws[key] = (qx, xscale, xsum)
    return qx, xscale, xsum


def _group32_workspace(
    g: dict, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reusable gs16/gs24 K32-layout and split-K buffers."""

    M = int(x.shape[0])
    ng = int(g["ng"])
    out = int(g["out"])
    m_pad = ((M + 15) // 16) * 16
    nchunks = (ng + 7) // 8
    split_k = 2 if out <= 8192 else 1
    key = (str(x.device), M, ng, out)
    ws = g.setdefault("_group32_workspace", {})
    cached = ws.get(key)
    if cached is not None and all(t.device == x.device for t in cached):
        return cached
    kstride = 36 if int(g.get("bits", 4)) == 2 and int(g["gs"]) == 16 else 68
    qx_mmq = torch.empty(nchunks * m_pad * kstride, device=x.device, dtype=torch.int32)
    xscale = torch.empty((M, ng), device=x.device, dtype=torch.float32)
    xsum = torch.empty((M, ng), device=x.device, dtype=torch.int32)
    partial = torch.empty(split_k * M * out, device=x.device, dtype=torch.float32)
    cached = (qx_mmq, xscale, xsum, partial)
    ws[key] = cached
    return cached


def _nint2_group32_use_m(out: int, m: int) -> bool:
    """Measured crossover against compact dequant + cuBLAS on RTX 3090 Ti."""

    if out < 2048 or m < 9:
        return False
    return m <= (64 if out < 8192 else 128)


def _nint6_int8_mmq_enabled() -> bool:
    """Return whether the lossy INT8-activation NINT6 MMQ was requested."""

    mode = os.environ.get("MFQ_NINT6_MMQ", "fp16").strip().lower()
    if mode not in {"fp16", "int8"}:
        raise ValueError("MFQ_NINT6_MMQ must be fp16 or int8")
    return mode == "int8"


def nint_argmax(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Compute an M=1 NINT projection and return its greedy argmax on CUDA."""

    bits = int(g.get("bits", 4))
    gs = int(g["gs"])
    if not ((bits == 5 and gs == 28) or (bits == 6 and gs in (24, 26))):
        raise ValueError("NINT argmax supports NINT5 gs28 and NINT6 gs24/gs26")
    x = x.reshape(-1, x.shape[-1]).contiguous().to(torch.float16)
    if x.shape[0] != 1:
        raise ValueError("NINT argmax expects one input row")
    neuron_len = int(g["neuron_len"])
    if x.shape[1] > neuron_len:
        raise ValueError(f"x last dim {x.shape[1]} > neuron_len {neuron_len}")
    if x.shape[1] < neuron_len:
        x = torch.nn.functional.pad(x, (0, neuron_len - x.shape[1]))
    qx, xscale, xsum = _workspace(g, x)
    blocks = (int(g["out"]) + 3) // 4
    ws = g.setdefault("_argmax_workspace", {})
    key = str(x.device)
    cached = ws.get(key)
    if cached is None or cached[0].numel() < blocks:
        cached = (
            torch.empty(blocks, device=x.device, dtype=torch.float32),
            torch.empty(blocks, device=x.device, dtype=torch.int32),
        )
        ws[key] = cached
    return ext().nint_gemv_packed_bits_argmax_ws_cuda(
        g["q_packed"],
        g["sub_scale"],
        g["sub_min"],
        g["neuron_scale"],
        g["neuron_min"],
        x,
        gs,
        bits,
        qx,
        xscale,
        xsum,
        cached[0],
        cached[1],
    )


def nint5_q5_exec_repack(g: dict) -> dict:
    """Return a NINT5 gs28 GPU view using the low4/high1 execution layout."""

    if int(g.get("bits", 4)) != 5 or int(g["gs"]) != 28:
        raise ValueError("Q5 execution layout requires NINT5 gs28")
    result = dict(g)
    result["q_packed"] = ext().nint5_gs28_q5_repack_cuda(
        g["q_packed"], g["sub_scale"], g["sub_min"]
    )
    result["_q5_exec_layout"] = True
    result.pop("_workspace", None)
    result.pop("_argmax_workspace", None)
    return result


def nint5_q5_exec_dequant(g: dict) -> torch.Tensor:
    """Dequantize a low4/high1 NINT5 gs28 execution tensor."""

    if not g.get("_q5_exec_layout", False):
        raise ValueError("weight is not in the NINT5 Q5 execution layout")
    return ext().nint5_gs28_q5_dequant_cuda(
        g["q_packed"],
        g["neuron_scale"],
        g["neuron_min"],
        int(g["neuron_len"]),
    )


def nint5_q5_exec_matmul(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Run M<=8 GEMV from the low4/high1 NINT5 gs28 execution layout."""

    if not g.get("_q5_exec_layout", False):
        raise ValueError("weight is not in the NINT5 Q5 execution layout")
    x = x.reshape(-1, x.shape[-1]).contiguous().to(torch.float16)
    neuron_len = int(g["neuron_len"])
    if x.shape[1] > neuron_len:
        raise ValueError(f"x last dim {x.shape[1]} > neuron_len {neuron_len}")
    if x.shape[1] < neuron_len:
        x = torch.nn.functional.pad(x, (0, neuron_len - x.shape[1]))
    qx, xscale, xsum = _workspace(g, x)
    return ext().nint5_gs28_q5_gemv_ws_cuda(
        g["q_packed"],
        g["neuron_scale"],
        g["neuron_min"],
        x,
        qx,
        xscale,
        xsum,
    )


def nint5_q5_exec_argmax(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Run M=1 GEMV and greedy argmax from the NINT5 Q5 execution layout."""

    if not g.get("_q5_exec_layout", False):
        raise ValueError("weight is not in the NINT5 Q5 execution layout")
    x = x.reshape(-1, x.shape[-1]).contiguous().to(torch.float16)
    if x.shape[0] != 1:
        raise ValueError("NINT5 Q5 argmax expects one input row")
    neuron_len = int(g["neuron_len"])
    if x.shape[1] > neuron_len:
        raise ValueError(f"x last dim {x.shape[1]} > neuron_len {neuron_len}")
    if x.shape[1] < neuron_len:
        x = torch.nn.functional.pad(x, (0, neuron_len - x.shape[1]))
    qx, xscale, xsum = _workspace(g, x)
    blocks = (int(g["out"]) + 3) // 4
    ws = g.setdefault("_argmax_workspace", {})
    key = str(x.device)
    cached = ws.get(key)
    if cached is None or cached[0].numel() < blocks:
        cached = (
            torch.empty(blocks, device=x.device, dtype=torch.float32),
            torch.empty(blocks, device=x.device, dtype=torch.int32),
        )
        ws[key] = cached
    return ext().nint5_gs28_q5_argmax_ws_cuda(
        g["q_packed"],
        g["neuron_scale"],
        g["neuron_min"],
        x,
        qx,
        xscale,
        xsum,
        cached[0],
        cached[1],
    )


def nint_matmul(g: dict, x: torch.Tensor) -> torch.Tensor:
    """NINT INT-fused-GEMM: ``y = x . W^T`` (fp16, cuda).

    x: ``[M, K]`` fp16/fp32 (cuda), K <= neuron_len (zero-padded if shorter). Returns ``[M, out]`` fp16.
    """

    x = x.contiguous().to(torch.float16)
    M = x.shape[0]
    nl = g["neuron_len"]
    if x.shape[1] != nl:
        if x.shape[1] > nl:
            raise ValueError(f"x last dim {x.shape[1]} > neuron_len {nl}")
        x = torch.nn.functional.pad(x, (0, nl - x.shape[1]))
    # llama.cpp-style dispatch:
    #   M == 1   -> GEMV: warp-per-output-row dp4a.
    #   M <= 6   -> batched GEMV/MMVQ: one weight sweep for multiple rows.
    #   M <= 64  -> MMQ: tiled dp4a with temporary Q8 activations.
    #   M > 64   -> dequant+cuBLAS until the int8-MMA prefill kernel is faster.
    q_packed = g.get("q_packed")
    bits = int(g.get("bits", 4))
    packed = q_packed is not None
    if packed and bits != 4 and g.get("sub_scale") is not None:
        if bits == 8 and M <= 8:
            qx, xscale, xsum = _workspace(g, x)
            return ext().nint_gemv_packed_u8_ws_cuda(
                q_packed,
                g["sub_scale"],
                g["sub_min"],
                g["neuron_scale"],
                g["neuron_min"],
                x,
                int(g["gs"]),
                qx,
                xscale,
                xsum,
            )
        if M <= 8:
            qx, xscale, xsum = _workspace(g, x)
            if bits == 6 and (int(g["gs"]) % 4) == 0:
                return ext().nint_gemv_packed_int6_ws_cuda(
                    q_packed,
                    g["sub_scale"],
                    g["sub_min"],
                    g["neuron_scale"],
                    g["neuron_min"],
                    x,
                    int(g["gs"]),
                    qx,
                    xscale,
                    xsum,
                )
            return ext().nint_gemv_packed_bits_ws_cuda(
                q_packed,
                g["sub_scale"],
                g["sub_min"],
                g["neuron_scale"],
                g["neuron_min"],
                x,
                int(g["gs"]),
                bits,
                qx,
                xscale,
                xsum,
            )
        if (
            (
                bits == 2
                and int(g["gs"]) == 16
                and _nint2_group32_use_m(int(g["out"]), M)
            )
            or (
                bits == 3
                and int(g["gs"]) == 24
                and int(g["out"]) >= 1024
            )
            or (
                bits == 6
                and int(g["gs"]) == 24
                and int(g["out"]) >= 1024
                and M >= 9
                and _nint6_int8_mmq_enabled()
            )
        ):
            qx_mmq, mmq_xscale, mmq_xsum, partial = _group32_workspace(g, x)
            split_k = (
                1
                if bits == 6 and M >= 128
                else (2 if int(g["out"]) <= 8192 else 1)
            )
            return ext().nint_mmq_gs24_group32_ws_cuda(
                q_packed,
                g["sub_scale"],
                g["sub_min"],
                g["neuron_scale"],
                g["neuron_min"],
                x,
                qx_mmq,
                mmq_xscale,
                mmq_xsum,
                split_k,
                partial,
            )
        if bits == 3 and int(g["gs"]) == 24:
            return ext().nint_mmq_gs24_f16_nint3_cuda(
                q_packed,
                g["sub_scale"],
                g["sub_min"],
                g["neuron_scale"],
                g["neuron_min"],
                x,
            )
        if bits == 6 and int(g["gs"]) == 24 and M >= 16:
            return ext().nint_mmq_f16_packed_cuda(
                q_packed,
                g["sub_scale"],
                g["sub_min"],
                g["neuron_scale"],
                g["neuron_min"],
                x,
                24,
                6,
            )
        if bits == 8 and M <= 64 and os.environ.get("MFQ_NINT8_MMQ") == "1":
            qx, xscale, xsum = _workspace(g, x)
            return ext().nint_mmq_packed_u8_ws_cuda(
                q_packed,
                g["sub_scale"],
                g["sub_min"],
                g["neuron_scale"],
                g["neuron_min"],
                x,
                int(g["gs"]),
                qx,
                xscale,
                xsum,
            )
        w = ext().nint_dequant_full_packed_compact_bits_cuda(
            q_packed,
            g["sub_scale"],
            g["sub_min"],
            g["neuron_scale"],
            g["neuron_min"],
            int(g["neuron_len"]),
            int(g["gs"]),
            bits,
        )
        return x @ w.T

    deploy_eff = packed and bits == 4 and g.get("eff_pair_h") is not None and g.get("sub_scale") is None
    if deploy_eff:
        qx, xscale, xsum = _workspace(g, x)
        if M <= 8:
            return ext().nint_gemv_packed_batch_eff2_ws_cuda(
                q_packed, g["eff_pair_h"], x, int(g["gs"]), qx, xscale, xsum
            )
        if int(g["gs"]) == 24:
            w = ext().nint_dequant_full_packed_gs24_x2h2_cuda(q_packed, g["eff_pair_h"], int(g["neuron_len"]))
        else:
            w = ext().nint_dequant_full_packed_h2_cuda(
                q_packed, g["eff_pair_h"], int(g["neuron_len"]), int(g["gs"])
            )
        return x @ w.T

    q = q_packed if packed else g["q"]
    args = (q, g["sub_scale"], g["sub_min"], g["neuron_scale"], g["neuron_min"], x, int(g["gs"]))
    if packed:
        qx, xscale, xsum = _workspace(g, x)
        if M == 1:
            return ext().nint_gemv_packed_ws_cuda(*args, qx, xscale, xsum)
        if M in (2, 4, 7) and g.get("eff_pair_h") is not None:
            eff_args = (q, g["eff_pair_h"], x, int(g["gs"]))
            return ext().nint_gemv_packed_batch_eff2_ws_cuda(*eff_args, qx, xscale, xsum)
        if M == 3 and g.get("d_eff_h") is not None:
            eff_args = (q, g["d_eff_h"], g["m_eff_h"], x, int(g["gs"]))
            return ext().nint_gemv_packed_batch_eff_ws_cuda(*eff_args, qx, xscale, xsum)
        if M <= 6:
            return ext().nint_gemv_packed_batch_ws_cuda(*args, qx, xscale, xsum)
    elif M <= 3:
        return ext().nint_gemv_cuda(*args)
    if M <= 64:
        if packed:
            q_mmq_packed = g.get("q_mmq_packed")
            if q_mmq_packed is not None:
                exec_args = (
                    q_mmq_packed,
                    g["sub_scale_mmq"],
                    g["sub_min_mmq"],
                    g["neuron_scale"],
                    g["neuron_min"],
                    x,
                    int(g["ng"]),
                    int(g["gs"]),
                )
                return ext().nint_mmq_packed_exec_ws_cuda(*exec_args, qx, xscale, xsum)
            return ext().nint_mmq_packed_ws_cuda(*args, qx, xscale, xsum)
        return ext().nint_mmq_cuda(*args)
    if packed and g.get("sub_scale") is not None:
        w = ext().nint_dequant_full_packed_compact_cuda(
            q,
            g["sub_scale"],
            g["sub_min"],
            g["neuron_scale"],
            g["neuron_min"],
            int(g["neuron_len"]),
            int(g["gs"]),
        )
        return x @ w.T

    if packed and g.get("d_eff") is not None and g.get("m_eff_h") is not None:
        if g.get("m_eff") is not None:
            if int(g["gs"]) == 24:
                w = ext().nint_dequant_full_packed_gs24_x2_cuda(q, g["d_eff"], g["m_eff"], int(g["neuron_len"]))
                return x @ w.T
            w = ext().nint_dequant_full_packed_cuda(q, g["d_eff"], g["m_eff"], int(g["neuron_len"]), int(g["gs"]))
            return x @ w.T
        wq = ext().nint_dequant_wq_packed_cuda(q, g["d_eff"], int(g["neuron_len"]), int(g["gs"]))
        K_pad = int(g["ng"]) * int(g["gs"])
        xp = torch.nn.functional.pad(x, (0, K_pad - int(g["neuron_len"]))) if K_pad != int(g["neuron_len"]) else x
        xs = xp.reshape(M, int(g["ng"]), int(g["gs"])).sum(-1).to(torch.float16)
        return x @ wq.T - xs @ g["m_eff_h"].T
    from mfq.kernels.torch_backend import matmul as _dq_cublas
    return _dq_cublas(g, x)


def nint_matmul_input_mul(g: dict, x: torch.Tensor, gate: torch.Tensor, activation: str) -> torch.Tensor:
    """NINT matmul with input-side gate applied during activation quantization.

    ``activation`` is ``"sigmoid"`` for ``x * sigmoid(gate)`` or ``"silu"`` for
    ``x * silu(gate)``. The deploy decode path avoids materializing the gated
    activation; other paths keep the reference expression and call
    :func:`nint_matmul`.
    """

    if activation == "sigmoid":
        mode = 1
    elif activation == "silu":
        mode = 2
    else:
        raise ValueError(f"unsupported activation {activation!r}")
    x = x.contiguous().to(torch.float16)
    gate = gate.contiguous().to(torch.float16)
    if x.shape != gate.shape:
        raise ValueError(f"x and gate must have the same shape, got {tuple(x.shape)} and {tuple(gate.shape)}")
    M = x.shape[0]
    nl = int(g["neuron_len"])
    if x.shape[1] != nl:
        if x.shape[1] > nl:
            raise ValueError(f"x last dim {x.shape[1]} > neuron_len {nl}")
        pad = (0, nl - x.shape[1])
        x = torch.nn.functional.pad(x, pad)
        gate = torch.nn.functional.pad(gate, pad)

    q_packed = g.get("q_packed")
    bits = int(g.get("bits", 4))
    if q_packed is not None and bits != 4 and g.get("sub_scale") is not None:
        if activation == "sigmoid":
            return nint_matmul(g, x * torch.sigmoid(gate))
        return nint_matmul(g, x * torch.nn.functional.silu(gate))
    deploy_eff = q_packed is not None and bits == 4 and g.get("eff_pair_h") is not None and g.get("sub_scale") is None
    if deploy_eff and M <= 8:
        qx, xscale, xsum = _workspace(g, x)
        return ext().nint_gemv_packed_batch_eff2_gate_ws_cuda(
            q_packed, g["eff_pair_h"], x, gate, int(g["gs"]), mode, qx, xscale, xsum
        )
    if q_packed is not None and g.get("sub_scale") is not None and M == 1:
        qx, xscale, xsum = _workspace(g, x)
        return ext().nint_gemv_packed_gate_ws_cuda(
            q_packed,
            g["sub_scale"],
            g["sub_min"],
            g["neuron_scale"],
            g["neuron_min"],
            x,
            gate,
            int(g["gs"]),
            mode,
            qx,
            xscale,
            xsum,
        )

    if activation == "sigmoid":
        return nint_matmul(g, x * torch.sigmoid(gate))
    return nint_matmul(g, x * torch.nn.functional.silu(gate))
