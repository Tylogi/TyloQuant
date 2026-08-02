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
_IMATRIX_PRIORITY_GROUPS = 64
_IMATRIX_PRIORITY_RADIUS = 8
_IMATRIX_PRIORITY_ROW_CHUNK = 128
_IMATRIX_PRIORITY_PAIR_CHUNK = 128


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


def _refine_imatrix_final_encoding_torch(
    groups: torch.Tensor,
    weights: torch.Tensor,
    q: torch.Tensor,
    neuron_scale: torch.Tensor,
    neuron_min: torch.Tensor,
    sub_scale: torch.Tensor,
    sub_min: torch.Tensor,
    nmax: int,
    sub_nmax: int,
    passes: int = 2,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Refine the final stored NINT reconstruction under imatrix weights."""

    values = groups.to(torch.float32)
    objective_weight = weights.to(torch.float32)
    levels = q.to(torch.int32).clone()
    row_scale = neuron_scale.to(torch.float32).clone()
    row_min = neuron_min.to(torch.float32).clone()
    scale_levels = sub_scale.to(torch.int32).clone()
    min_levels = sub_min.to(torch.int32).clone()

    def statistics(current_levels: torch.Tensor):
        levels_f = current_levels.to(torch.float32)
        sum_w = objective_weight.sum(dim=-1)
        sum_x = (objective_weight * values).sum(dim=-1)
        sum_x2 = (objective_weight * values.square()).sum(dim=-1)
        sum_q = (objective_weight * levels_f).sum(dim=-1)
        sum_q2 = (objective_weight * levels_f.square()).sum(dim=-1)
        sum_qx = (objective_weight * levels_f * values).sum(dim=-1)
        return sum_w, sum_x, sum_x2, sum_q, sum_q2, sum_qx

    def group_error(
        stats,
        candidate_scale: torch.Tensor,
        candidate_min: torch.Tensor,
        candidate_scale_levels: torch.Tensor,
        candidate_min_levels: torch.Tensor,
    ) -> torch.Tensor:
        sum_w, sum_x, sum_x2, sum_q, sum_q2, sum_qx = stats
        effective_scale = (
            candidate_scale.unsqueeze(-1)
            * candidate_scale_levels.to(torch.float32)
        )
        effective_min = (
            candidate_min.unsqueeze(-1)
            * candidate_min_levels.to(torch.float32)
        )
        return (
            effective_scale.square() * sum_q2
            + effective_min.square() * sum_w
            + sum_x2
            - 2.0 * effective_scale * effective_min * sum_q
            - 2.0 * effective_scale * sum_qx
            + 2.0 * effective_min * sum_x
        )

    def fp16_round(values_: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            torch.nan_to_num(values_, nan=0.0, posinf=65504.0, neginf=0.0),
            min=0.0,
            max=65504.0,
        ).to(torch.float16).to(torch.float32)

    stats = statistics(levels)
    best_error = group_error(
        stats, row_scale, row_min, scale_levels, min_levels
    ).sum(dim=-1)

    for _ in range(max(0, int(passes))):
        candidate_levels = levels.clone()
        candidate_scale = row_scale.clone()
        candidate_min = row_min.clone()
        candidate_scale_levels = scale_levels.clone()
        candidate_min_levels = min_levels.clone()
        candidate_stats = stats

        for _coordinate_pass in range(2):
            sum_w, sum_x, _, sum_q, sum_q2, sum_qx = candidate_stats
            effective_min = (
                candidate_min.unsqueeze(-1)
                * candidate_min_levels.to(torch.float32)
            )
            scale_denominator = candidate_scale.unsqueeze(-1) * sum_q2
            valid_scale = scale_denominator > 0.0
            proposed_scale_levels = torch.clamp(
                torch.round(
                    (sum_qx + effective_min * sum_q)
                    / torch.where(
                        valid_scale,
                        scale_denominator,
                        torch.ones_like(scale_denominator),
                    )
                ),
                0,
                sub_nmax,
            ).to(torch.int32)
            proposed_scale_levels = torch.where(
                valid_scale, proposed_scale_levels, candidate_scale_levels
            )
            old_group_error = group_error(
                candidate_stats,
                candidate_scale,
                candidate_min,
                candidate_scale_levels,
                candidate_min_levels,
            )
            proposed_group_error = group_error(
                candidate_stats,
                candidate_scale,
                candidate_min,
                proposed_scale_levels,
                candidate_min_levels,
            )
            better = proposed_group_error < old_group_error
            candidate_scale_levels = torch.where(
                better, proposed_scale_levels, candidate_scale_levels
            )

            effective_scale = (
                candidate_scale.unsqueeze(-1)
                * candidate_scale_levels.to(torch.float32)
            )
            min_denominator = candidate_min.unsqueeze(-1) * sum_w
            valid_min = min_denominator > 0.0
            proposed_min_levels = torch.clamp(
                torch.round(
                    (effective_scale * sum_q - sum_x)
                    / torch.where(
                        valid_min,
                        min_denominator,
                        torch.ones_like(min_denominator),
                    )
                ),
                0,
                sub_nmax,
            ).to(torch.int32)
            proposed_min_levels = torch.where(
                valid_min, proposed_min_levels, candidate_min_levels
            )
            old_group_error = group_error(
                candidate_stats,
                candidate_scale,
                candidate_min,
                candidate_scale_levels,
                candidate_min_levels,
            )
            proposed_group_error = group_error(
                candidate_stats,
                candidate_scale,
                candidate_min,
                candidate_scale_levels,
                proposed_min_levels,
            )
            better = proposed_group_error < old_group_error
            candidate_min_levels = torch.where(
                better, proposed_min_levels, candidate_min_levels
            )

        sum_w, sum_x, _, sum_q, sum_q2, sum_qx = candidate_stats
        scale_level_f = candidate_scale_levels.to(torch.float32)
        min_level_f = candidate_min_levels.to(torch.float32)
        sum_aa = (scale_level_f.square() * sum_q2).sum(dim=-1)
        sum_mm = (min_level_f.square() * sum_w).sum(dim=-1)
        sum_am = (scale_level_f * min_level_f * sum_q).sum(dim=-1)
        sum_ax = (scale_level_f * sum_qx).sum(dim=-1)
        sum_mx = (min_level_f * sum_x).sum(dim=-1)
        determinant = sum_aa * sum_mm - sum_am.square()

        row_best_error = group_error(
            candidate_stats,
            candidate_scale,
            candidate_min,
            candidate_scale_levels,
            candidate_min_levels,
        ).sum(dim=-1)
        row_best_scale = candidate_scale.clone()
        row_best_min = candidate_min.clone()

        def consider(
            scale_value: torch.Tensor,
            min_value: torch.Tensor,
            current_stats=candidate_stats,
            current_scale_levels=candidate_scale_levels,
            current_min_levels=candidate_min_levels,
        ) -> None:
            nonlocal row_best_error, row_best_scale, row_best_min
            scale_value = fp16_round(scale_value)
            min_value = fp16_round(min_value)
            error = group_error(
                current_stats,
                scale_value,
                min_value,
                current_scale_levels,
                current_min_levels,
            ).sum(dim=-1)
            improve = error < row_best_error
            row_best_error = torch.where(improve, error, row_best_error)
            row_best_scale = torch.where(
                improve, scale_value, row_best_scale
            )
            row_best_min = torch.where(improve, min_value, row_best_min)

        valid_joint = determinant > 0.0
        safe_determinant = torch.where(
            valid_joint, determinant, torch.ones_like(determinant)
        )
        joint_scale = (
            sum_mm * sum_ax - sum_am * sum_mx
        ) / safe_determinant
        joint_min = (
            sum_am * sum_ax - sum_aa * sum_mx
        ) / safe_determinant
        joint_scale = torch.where(valid_joint, joint_scale, row_best_scale)
        joint_min = torch.where(valid_joint, joint_min, row_best_min)
        consider(joint_scale, joint_min)
        consider(
            sum_ax
            / torch.where(sum_aa > 0.0, sum_aa, torch.ones_like(sum_aa)),
            torch.zeros_like(sum_ax),
        )
        consider(
            torch.zeros_like(sum_mx),
            -sum_mx
            / torch.where(sum_mm > 0.0, sum_mm, torch.ones_like(sum_mm)),
        )
        candidate_scale = row_best_scale
        candidate_min = row_best_min

        effective_scale = (
            candidate_scale.unsqueeze(-1)
            * candidate_scale_levels.to(torch.float32)
        )
        effective_min = (
            candidate_min.unsqueeze(-1)
            * candidate_min_levels.to(torch.float32)
        )
        safe_effective_scale = torch.where(
            effective_scale > 0.0,
            effective_scale,
            torch.ones_like(effective_scale),
        )
        candidate_levels = torch.clamp(
            torch.round(
                (values + effective_min.unsqueeze(-1))
                / safe_effective_scale.unsqueeze(-1)
            ),
            0,
            nmax,
        ).to(torch.int32)
        candidate_levels = torch.where(
            (effective_scale > 0.0).unsqueeze(-1),
            candidate_levels,
            torch.zeros_like(candidate_levels),
        )
        candidate_stats = statistics(candidate_levels)
        candidate_error = group_error(
            candidate_stats,
            candidate_scale,
            candidate_min,
            candidate_scale_levels,
            candidate_min_levels,
        ).sum(dim=-1)
        improve = candidate_error < best_error
        if not bool(improve.any().item()):
            break
        best_error = torch.where(improve, candidate_error, best_error)
        levels = torch.where(
            improve[:, None, None], candidate_levels, levels
        )
        row_scale = torch.where(improve, candidate_scale, row_scale)
        row_min = torch.where(improve, candidate_min, row_min)
        scale_levels = torch.where(
            improve[:, None], candidate_scale_levels, scale_levels
        )
        min_levels = torch.where(
            improve[:, None], candidate_min_levels, min_levels
        )
        stats = statistics(levels)

    return levels, row_scale, row_min, scale_levels, min_levels


@torch.inference_mode()
def _refine_imatrix_priority_groups_torch(
    groups: torch.Tensor,
    weights: torch.Tensor,
    q: torch.Tensor,
    neuron_scale: torch.Tensor,
    neuron_min: torch.Tensor,
    sub_scale: torch.Tensor,
    sub_min: torch.Tensor,
    nmax: int,
    sub_nmax: int,
    *,
    priority_groups: int = _IMATRIX_PRIORITY_GROUPS,
    radius: int = _IMATRIX_PRIORITY_RADIUS,
    row_chunk: int = _IMATRIX_PRIORITY_ROW_CHUNK,
    pair_chunk: int = _IMATRIX_PRIORITY_PAIR_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Jointly refine the final codes of the highest-mass imatrix groups."""

    values = groups.to(torch.float32)
    objective_weight = weights.to(torch.float32)
    levels = q.to(torch.int32).clone()
    scale_levels = sub_scale.to(torch.int32).clone()
    min_levels = sub_min.to(torch.int32).clone()
    row_scale = neuron_scale.to(torch.float32)
    row_min = neuron_min.to(torch.float32)
    rows, group_count, group_size = values.shape
    selected_count = min(max(0, int(priority_groups)), group_count)
    if selected_count == 0 or radius < 0:
        return levels, scale_levels, min_levels

    offsets = torch.arange(
        -radius,
        radius + 1,
        device=values.device,
        dtype=torch.int32,
    )
    scale_offsets = offsets[:, None].expand(-1, offsets.numel()).reshape(-1)
    min_offsets = offsets[None, :].expand(offsets.numel(), -1).reshape(-1)
    row_chunk = max(1, int(row_chunk))
    pair_chunk = max(1, int(pair_chunk))

    for row_begin in range(0, rows, row_chunk):
        row_end = min(row_begin + row_chunk, rows)
        chunk_values = values[row_begin:row_end]
        chunk_weights = objective_weight[row_begin:row_end]
        group_mass = chunk_weights.sum(dim=-1)
        selected = torch.topk(
            group_mass, selected_count, dim=-1
        ).indices
        local_rows = torch.arange(
            row_end - row_begin, device=values.device
        )[:, None].expand_as(selected)
        global_rows = local_rows + row_begin
        flat_values = chunk_values[local_rows, selected].reshape(
            -1, group_size
        )
        flat_weights = chunk_weights[local_rows, selected].reshape(
            -1, group_size
        )
        flat_rows = global_rows.reshape(-1)
        flat_q = levels[global_rows, selected].reshape(
            -1, group_size
        ).clone()
        flat_scale_levels = scale_levels[global_rows, selected].reshape(
            -1
        ).clone()
        flat_min_levels = min_levels[global_rows, selected].reshape(
            -1
        ).clone()
        base_scale_levels = flat_scale_levels.clone()
        base_min_levels = flat_min_levels.clone()
        flat_row_scale = row_scale[flat_rows]
        flat_row_min = row_min[flat_rows]
        effective_scale = (
            flat_row_scale * flat_scale_levels.to(torch.float32)
        )
        effective_min = flat_row_min * flat_min_levels.to(torch.float32)
        reconstruction = (
            effective_scale[:, None] * flat_q.to(torch.float32)
            - effective_min[:, None]
        )
        best_error = (
            flat_weights * (reconstruction - flat_values).square()
        ).sum(dim=-1)

        for pair_begin in range(0, scale_offsets.numel(), pair_chunk):
            pair_end = min(
                pair_begin + pair_chunk, scale_offsets.numel()
            )
            candidate_scale_levels = torch.clamp(
                base_scale_levels[:, None]
                + scale_offsets[None, pair_begin:pair_end],
                0,
                sub_nmax,
            )
            candidate_min_levels = torch.clamp(
                base_min_levels[:, None]
                + min_offsets[None, pair_begin:pair_end],
                0,
                sub_nmax,
            )
            candidate_scale = (
                flat_row_scale[:, None]
                * candidate_scale_levels.to(torch.float32)
            )
            candidate_min = (
                flat_row_min[:, None]
                * candidate_min_levels.to(torch.float32)
            )
            safe_scale = torch.where(
                candidate_scale > 0.0,
                candidate_scale,
                torch.ones_like(candidate_scale),
            )
            candidate_q = torch.clamp(
                torch.round(
                    (flat_values[:, None, :] + candidate_min[:, :, None])
                    / safe_scale[:, :, None]
                ),
                0,
                nmax,
            ).to(torch.int32)
            candidate_q = torch.where(
                (candidate_scale > 0.0)[:, :, None],
                candidate_q,
                torch.zeros_like(candidate_q),
            )
            candidate_reconstruction = (
                candidate_scale[:, :, None]
                * candidate_q.to(torch.float32)
                - candidate_min[:, :, None]
            )
            candidate_error = (
                flat_weights[:, None, :]
                * (candidate_reconstruction - flat_values[:, None, :])
                .square()
            ).sum(dim=-1)
            chunk_error, chunk_index = candidate_error.min(dim=1)
            improve = chunk_error < best_error
            gather_rows = torch.arange(
                candidate_q.shape[0], device=values.device
            )
            chosen_q = candidate_q[gather_rows, chunk_index]
            chosen_scale_levels = candidate_scale_levels[
                gather_rows, chunk_index
            ]
            chosen_min_levels = candidate_min_levels[
                gather_rows, chunk_index
            ]
            best_error = torch.where(improve, chunk_error, best_error)
            flat_q = torch.where(improve[:, None], chosen_q, flat_q)
            flat_scale_levels = torch.where(
                improve, chosen_scale_levels, flat_scale_levels
            )
            flat_min_levels = torch.where(
                improve, chosen_min_levels, flat_min_levels
            )

        levels[global_rows, selected] = flat_q.reshape(
            row_end - row_begin, selected_count, group_size
        )
        scale_levels[global_rows, selected] = flat_scale_levels.reshape(
            row_end - row_begin, selected_count
        )
        min_levels[global_rows, selected] = flat_min_levels.reshape(
            row_end - row_begin, selected_count
        )

    return levels, scale_levels, min_levels


def quantize_axis0(
    weight: torch.Tensor,
    spec: NintSpec,
    device: str | torch.device = "cuda",
    importance: np.ndarray | torch.Tensor | None = None,
    use_cuda_imatrix_kernels: bool = True,
    use_priority_group_refinement: bool = True,
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
    if importance is not None:
        q, neu_d, neu_dm, sub_scale, sub_min = (
            _refine_imatrix_final_encoding_torch(
                grps,
                ww,
                q,
                neu_d,
                neu_dm,
                sub_scale,
                sub_min,
                nmax=nmax,
                sub_nmax=K,
            )
        )
        if use_priority_group_refinement:
            base_q = q.clone()
            base_sub_scale = sub_scale.clone()
            base_sub_min = sub_min.clone()
            candidate_q, candidate_sub_scale, candidate_sub_min = (
                _refine_imatrix_priority_groups_torch(
                    grps,
                    ww,
                    q,
                    neu_d,
                    neu_dm,
                    sub_scale,
                    sub_min,
                    nmax=nmax,
                    sub_nmax=K,
                )
            )
            base_reconstruction = (
                neu_d[:, None, None]
                * base_sub_scale[:, :, None].to(torch.float32)
                * base_q.to(torch.float32)
                - neu_dm[:, None, None]
                * base_sub_min[:, :, None].to(torch.float32)
            )
            candidate_reconstruction = (
                neu_d[:, None, None]
                * candidate_sub_scale[:, :, None].to(torch.float32)
                * candidate_q.to(torch.float32)
                - neu_dm[:, None, None]
                * candidate_sub_min[:, :, None].to(torch.float32)
            )
            base_error = (
                ww * (base_reconstruction - grps).square()
            ).sum(dim=(1, 2))
            candidate_error = (
                ww * (candidate_reconstruction - grps).square()
            ).sum(dim=(1, 2))
            accept = candidate_error <= base_error * (1.0 + 1.0e-7)
            q = torch.where(
                accept[:, None, None],
                candidate_q,
                base_q.to(candidate_q.dtype),
            )
            sub_scale = torch.where(
                accept[:, None],
                candidate_sub_scale,
                base_sub_scale.to(candidate_sub_scale.dtype),
            )
            sub_min = torch.where(
                accept[:, None],
                candidate_sub_min,
                base_sub_min.to(candidate_sub_min.dtype),
            )
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
