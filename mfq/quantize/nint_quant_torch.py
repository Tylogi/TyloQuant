"""Torch/CUDA NINT tensor quantization.

This mirrors :mod:`mfq.quantize.nint_quant` but keeps the per-group search on
GPU. The returned object is still the existing CPU-side ``NintTensor`` so the
MFQ file format and runtime loaders do not change.
"""

from __future__ import annotations

import numpy as np
import torch

from mfq.formats.nint import NintSpec, _uint_dtype
from mfq.quantize.nint_quant import NintTensor


_IMATRIX_SUPERBLOCK = 256


def _qkx2_search_params(nmax: int) -> tuple[float, float, int]:
    if nmax <= 15:
        return -1.0, 0.1, 20
    return -0.5, 0.1, 15


def make_qkx2_torch(
    x: torch.Tensor,
    w: torch.Tensor,
    nmax: int = 15,
    rmin: float | None = None,
    rdelta: float = 0.1,
    nstep: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted least-squares affine quantizer search on torch tensors."""

    if rmin is None or nstep is None:
        default_rmin, default_rdelta, default_nstep = _qkx2_search_params(nmax)
        if rmin is None:
            rmin = default_rmin
        if nstep is None:
            nstep = default_nstep
        rdelta = default_rdelta if rdelta is None else rdelta

    zero = torch.zeros((), device=x.device, dtype=x.dtype)
    one = torch.ones((), device=x.device, dtype=x.dtype)
    mn = torch.minimum(x.amin(dim=-1), zero)
    mx = x.amax(dim=-1)
    sum_w = w.sum(dim=-1)
    sum_x = (w * x).sum(dim=-1)
    degen = (mx == mn) | (sum_w <= 0)
    rng = torch.where(degen, one, mx - mn)

    iscale0 = float(nmax) / rng
    scale0 = 1.0 / iscale0
    L0 = torch.clamp(torch.round(iscale0.unsqueeze(-1) * (x - mn.unsqueeze(-1))), 0, nmax)
    diff = scale0.unsqueeze(-1) * L0 + mn.unsqueeze(-1) - x
    best_err = (w * diff * diff).sum(dim=-1)
    best_scale = scale0.clone()
    best_min = mn.clone()

    for i in range(int(nstep) + 1):
        iscale = (float(rmin) + float(rdelta) * i + float(nmax)) / rng
        Laux = torch.clamp(torch.round(iscale.unsqueeze(-1) * (x - mn.unsqueeze(-1))), 0, nmax)
        sl = (w * Laux).sum(dim=-1)
        sl2 = (w * Laux * Laux).sum(dim=-1)
        sxl = (w * Laux * x).sum(dim=-1)
        D = sum_w * sl2 - sl * sl
        valid = D > 0
        Ds = torch.where(valid, D, one)
        ts = (sum_w * sxl - sum_x * sl) / Ds
        tm = (sl2 * sum_x - sl * sxl) / Ds
        pos = tm > 0
        sl2s = torch.where(sl2 > 0, sl2, one)
        ts = torch.where(pos, sxl / sl2s, ts)
        tm = torch.where(pos, zero, tm)
        cd = ts.unsqueeze(-1) * Laux + tm.unsqueeze(-1) - x
        ce = (w * cd * cd).sum(dim=-1)
        better = valid & (ce < best_err)
        best_err = torch.where(better, ce, best_err)
        best_scale = torch.where(better, ts, best_scale)
        best_min = torch.where(better, tm, best_min)

    best_scale = torch.where(degen, zero, best_scale)
    best_min = torch.where(degen, torch.minimum(mn, zero), best_min)
    return best_scale, best_min


def make_qkx3_torch(
    x: torch.Tensor,
    w: torch.Tensor,
    nmax: int,
    rmin: float = -0.9,
    rdelta: float = 0.05,
    nstep: int = 36,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted affine search matching llama.cpp make_qkx3_quants."""

    zero = torch.zeros((), device=x.device, dtype=x.dtype)
    one = torch.ones((), device=x.device, dtype=x.dtype)
    mn = torch.minimum(x.amin(dim=-1), zero)
    mx = x.amax(dim=-1)
    sum_w = w.sum(dim=-1)
    sum_x = (w * x).sum(dim=-1)
    degen = (mx <= mn) | (sum_w <= 0)
    rng = torch.where(degen, one, mx - mn)

    iscale0 = float(nmax) / rng
    scale0 = 1.0 / iscale0
    levels0 = torch.clamp(
        torch.round(iscale0.unsqueeze(-1) * (x - mn.unsqueeze(-1))),
        0,
        nmax,
    )
    diff0 = scale0.unsqueeze(-1) * levels0 + mn.unsqueeze(-1) - x
    best_error = (w * diff0 * diff0).sum(dim=-1)
    best_scale = scale0.clone()
    best_min = mn.clone()

    for step in range(nstep + 1):
        iscale = (
            float(rmin + rdelta * step + nmax) / rng
        )
        levels = torch.clamp(
            torch.round(iscale.unsqueeze(-1) * (x - mn.unsqueeze(-1))),
            0,
            nmax,
        )
        sum_l = (w * levels).sum(dim=-1)
        sum_l2 = (w * levels * levels).sum(dim=-1)
        sum_xl = (w * levels * x).sum(dim=-1)
        determinant = sum_w * sum_l2 - sum_l * sum_l
        valid = determinant > 0
        divisor = torch.where(valid, determinant, one)
        candidate_scale = (
            sum_w * sum_xl - sum_x * sum_l
        ) / divisor
        candidate_min = (
            sum_l2 * sum_x - sum_l * sum_xl
        ) / divisor
        positive_min = candidate_min > 0
        safe_sum_l2 = torch.where(sum_l2 > 0, sum_l2, one)
        candidate_scale = torch.where(
            positive_min, sum_xl / safe_sum_l2, candidate_scale
        )
        candidate_min = torch.where(
            positive_min, zero, candidate_min
        )
        candidate_diff = (
            candidate_scale.unsqueeze(-1) * levels
            + candidate_min.unsqueeze(-1)
            - x
        )
        candidate_error = (
            w * candidate_diff * candidate_diff
        ).sum(dim=-1)
        better = valid & (candidate_error < best_error)
        best_error = torch.where(
            better, candidate_error, best_error
        )
        best_scale = torch.where(
            better, candidate_scale, best_scale
        )
        best_min = torch.where(better, candidate_min, best_min)

    best_scale = torch.where(degen, zero, best_scale)
    best_min = torch.where(degen, torch.minimum(mn, zero), best_min)
    return best_scale, best_min


def make_qkx3_cuda(
    x: torch.Tensor,
    w: torch.Tensor,
    nmax: int,
    rmin: float = -0.9,
    rdelta: float = 0.05,
    nstep: int = 36,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused CUDA implementation of :func:`make_qkx3_torch`."""

    from mfq.quantize.cuda._ext import ext

    scale, minimum = ext().nint_make_qkx3(
        x.contiguous(),
        w.contiguous(),
        int(nmax),
        float(rmin),
        float(rdelta),
        int(nstep),
    )
    return scale, minimum


def make_qp_torch(
    x: torch.Tensor,
    weights: torch.Tensor,
    nmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted neuron-level scale quantization from llama.cpp make_qp_quants."""

    maximum = x.amax(dim=-1)
    active = maximum >= 1e-15
    safe_maximum = torch.where(
        active, maximum, torch.ones_like(maximum)
    )
    iscale = float(nmax) / safe_maximum
    levels = torch.clamp(
        torch.round(iscale.unsqueeze(-1) * x), 0, nmax
    ).to(torch.int32)
    scale = 1.0 / iscale
    difference = x - scale.unsqueeze(-1) * levels
    best_error = (weights * difference * difference).sum(dim=-1)

    for offset in range(-4, 5):
        if offset == 0:
            continue
        candidate_iscale = (
            float(nmax + 0.1 * offset) / safe_maximum
        )
        candidate_scale = 1.0 / candidate_iscale
        candidate_levels = torch.clamp(
            torch.round(candidate_iscale.unsqueeze(-1) * x),
            0,
            nmax,
        )
        candidate_difference = (
            x - candidate_scale.unsqueeze(-1) * candidate_levels
        )
        candidate_error = (
            weights * candidate_difference * candidate_difference
        ).sum(dim=-1)
        better = active & (candidate_error < best_error)
        best_error = torch.where(
            better, candidate_error, best_error
        )
        iscale = torch.where(better, candidate_iscale, iscale)

    levels = torch.clamp(
        torch.round(iscale.unsqueeze(-1) * x), 0, nmax
    ).to(torch.int32)
    levels_f = levels.to(torch.float32)
    sum_lx = (weights * x * levels_f).sum(dim=-1)
    sum_l2 = (weights * levels_f * levels_f).sum(dim=-1)

    for _ in range(5):
        for index in range(x.shape[-1]):
            old_level = levels[:, index].to(torch.float32)
            weight = weights[:, index]
            value = x[:, index]
            candidate_lx = sum_lx - weight * value * old_level
            candidate_l2 = sum_l2 - weight * old_level * old_level
            valid = (candidate_lx > 0) & (candidate_l2 > 0)
            safe_lx = torch.where(
                valid, candidate_lx, torch.ones_like(candidate_lx)
            )
            new_level = torch.clamp(
                torch.round(value * candidate_l2 / safe_lx),
                0,
                nmax,
            ).to(torch.int32)
            new_level_f = new_level.to(torch.float32)
            updated_lx = (
                candidate_lx + weight * value * new_level_f
            )
            updated_l2 = (
                candidate_l2 + weight * new_level_f * new_level_f
            )
            accept = (
                valid
                & (new_level != levels[:, index])
                & (
                    updated_lx * updated_lx * sum_l2
                    > sum_lx * sum_lx * updated_l2
                )
            )
            levels[:, index] = torch.where(
                accept, new_level, levels[:, index]
            )
            sum_lx = torch.where(accept, updated_lx, sum_lx)
            sum_l2 = torch.where(accept, updated_l2, sum_l2)

    scale = torch.where(
        active & (sum_l2 > 0),
        sum_lx
        / torch.where(sum_l2 > 0, sum_l2, torch.ones_like(sum_l2)),
        torch.zeros_like(sum_lx),
    )
    levels = torch.where(
        active.unsqueeze(-1), levels, torch.zeros_like(levels)
    )
    return scale, levels


def make_qp_cuda(
    x: torch.Tensor,
    weights: torch.Tensor,
    nmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused CUDA implementation of :func:`make_qp_torch`."""

    from mfq.quantize.cuda._ext import ext

    scale, levels = ext().nint_make_qp(
        x.contiguous(),
        weights.contiguous(),
        int(nmax),
    )
    return scale, levels


def _imatrix_element_weights(
    rows: torch.Tensor,
    importance_rows: torch.Tensor,
    neuron_len: int,
) -> torch.Tensor:
    """Build llama.cpp-style element weights using 256-value sigma² blocks."""

    block = _IMATRIX_SUPERBLOCK
    block_pad = (-neuron_len) % block
    real = rows[:, :neuron_len]
    if block_pad:
        real = torch.nn.functional.pad(real, (0, block_pad))
    blocks = real.reshape(rows.shape[0], -1, block)
    counts = torch.full(
        (blocks.shape[1],),
        float(block),
        dtype=torch.float32,
        device=rows.device,
    )
    if block_pad:
        counts[-1] -= float(block_pad)
    sigma2 = 2.0 * (blocks * blocks).sum(dim=-1) / counts.unsqueeze(0)
    sigma2_elements = torch.repeat_interleave(
        sigma2, block, dim=1
    )[:, :neuron_len]
    weights = importance_rows * torch.sqrt(
        sigma2_elements
        + rows[:, :neuron_len] * rows[:, :neuron_len]
    )
    if rows.shape[1] != neuron_len:
        weights = torch.nn.functional.pad(
            weights, (0, rows.shape[1] - neuron_len)
        )
    return weights.contiguous()


def _importance_as_rows(
    importance: np.ndarray | torch.Tensor,
    out: int,
    neuron_len: int,
    device: str | torch.device,
) -> torch.Tensor:
    values = torch.as_tensor(importance, dtype=torch.float32, device=device)
    if values.dim() == 1:
        if int(values.shape[0]) != neuron_len:
            raise ValueError(
                f"NINT importance width {int(values.shape[0])} != neuron length {neuron_len}"
            )
        rows = values.unsqueeze(0).expand(out, neuron_len)
    elif tuple(values.shape) == (out, neuron_len):
        rows = values
    else:
        raise ValueError(
            "NINT importance must be one input-channel vector or "
            f"{(out, neuron_len)} row weights; got {tuple(values.shape)}"
        )
    if not bool(torch.isfinite(rows).all().item()) or bool((rows < 0).any().item()):
        raise ValueError("NINT importance must contain finite non-negative values")
    return rows.contiguous()


def quantize_axis0(
    weight: torch.Tensor,
    spec: NintSpec,
    device: str | torch.device = "cuda",
    importance: np.ndarray | torch.Tensor | None = None,
    use_cuda_imatrix_kernels: bool = True,
) -> NintTensor:
    """Quantize a 2D ``[out, in]`` tensor with axis=0 on GPU."""

    if weight.dim() != 2:
        raise ValueError(f"quantize_axis0 expects a 2D tensor, got {tuple(weight.shape)}")
    W = weight.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    out, neuron_len = (int(W.shape[0]), int(W.shape[1]))
    gs = int(spec.groupsize)
    nmax = int(spec.nmax)
    k = int(spec.sub_bits)
    K = (1 << k) - 1
    pad = (-neuron_len) % gs
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    ng = int(W.shape[1] // gs)
    grps = W.reshape(out, ng, gs)

    sx2 = (grps * grps).sum(dim=-1)
    if importance is None:
        av = torch.sqrt(sx2 / float(gs))
        ww = av.unsqueeze(-1) + grps.abs()
        if pad:
            ww[:, -1, gs - pad :] = 0.0
    else:
        importance_rows = _importance_as_rows(
            importance, out, neuron_len, device
        )
        ww = _imatrix_element_weights(
            W, importance_rows, neuron_len
        ).reshape(out, ng, gs)

    fused_imatrix = (
        importance is not None
        and W.is_cuda
        and use_cuda_imatrix_kernels
    )
    if importance is None:
        scale, zp = make_qkx2_torch(grps, ww, nmax=nmax)
    elif fused_imatrix:
        scale, zp = make_qkx3_cuda(grps, ww, nmax=nmax)
    else:
        scale, zp = make_qkx3_torch(grps, ww, nmax=nmax)
    the_min = -zp
    if importance is None:
        neu_s = scale.amax(dim=-1)
        neu_m = the_min.amax(dim=-1)
        neu_d = torch.where(
            neu_s > 0,
            (neu_s / float(K)).to(torch.float16).to(torch.float32),
            torch.zeros_like(neu_s),
        )
        neu_dm = torch.where(
            neu_m > 0,
            (neu_m / float(K)).to(torch.float16).to(torch.float32),
            torch.zeros_like(neu_m),
        )

        nss = torch.where(neu_s > 0, neu_s, torch.ones_like(neu_s))
        nmm = torch.where(neu_m > 0, neu_m, torch.ones_like(neu_m))
        sub_scale = torch.clamp(torch.round(float(K) * scale / nss.unsqueeze(-1)), 0, K)
        sub_min = torch.clamp(torch.round(float(K) * the_min / nmm.unsqueeze(-1)), 0, K)
    elif fused_imatrix:
        group_weights = ww.sum(dim=-1)
        neu_d, sub_scale = make_qp_cuda(
            scale, group_weights, nmax=K
        )
        neu_dm, sub_min = make_qp_cuda(
            the_min, group_weights, nmax=K
        )
        neu_d = neu_d.to(torch.float16).to(torch.float32)
        neu_dm = neu_dm.to(torch.float16).to(torch.float32)
    else:
        group_weights = ww.sum(dim=-1)
        neu_d, sub_scale = make_qp_torch(
            scale, group_weights, nmax=K
        )
        neu_dm, sub_min = make_qp_torch(
            the_min, group_weights, nmax=K
        )
        neu_d = neu_d.to(torch.float16).to(torch.float32)
        neu_dm = neu_dm.to(torch.float16).to(torch.float32)

    d_eff = neu_d.unsqueeze(-1) * sub_scale
    m_eff = neu_dm.unsqueeze(-1) * sub_min
    de = torch.where(d_eff > 0, d_eff, torch.ones_like(d_eff))
    q = torch.clamp(torch.round((grps + m_eff.unsqueeze(-1)) / de.unsqueeze(-1)), 0, nmax)

    sub_dtype = _uint_dtype(K)
    q_dtype = _uint_dtype(nmax)
    return NintTensor(
        spec=spec,
        shape=(out, neuron_len),
        axis=0,
        q=q.to(torch.uint8).cpu().numpy().astype(q_dtype, copy=False),
        neuron_scale=neu_d.cpu().numpy().astype(np.float32, copy=False),
        neuron_min=neu_dm.cpu().numpy().astype(np.float32, copy=False),
        sub_scale=sub_scale.to(torch.uint8).cpu().numpy().astype(sub_dtype, copy=False),
        sub_min=sub_min.to(torch.uint8).cpu().numpy().astype(sub_dtype, copy=False),
        neuron_len=neuron_len,
    )
