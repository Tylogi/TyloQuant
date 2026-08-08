"""CUDA offline quantizers for NVQ1-L, NVQ2, and NVQ3.

The solvers mirror the NumPy reference objectives. They operate on one row
chunk at a time and return the existing CPU-side packed tensor classes, so the
MFQ format is shared by CPU and CUDA conversion paths.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import torch

from mfq.formats.nvq import NvqSpec, NvqTensor, codebook_for, validate_codebook
from mfq.formats.nvq1_l import (
    IQ1S_TERNARY_2048,
    Nvq1LSpec,
    Nvq1LTensor,
    validate_ternary_codebook,
)


def _prepare_weight(
    weight: torch.Tensor,
    device: str | torch.device,
) -> tuple[torch.Tensor, int, int]:
    if weight.dim() != 2:
        raise ValueError(f"CUDA NVQ quantization expects [out, in], got {tuple(weight.shape)}")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0:
        raise ValueError("NVQ input dimensions must be positive")
    value = weight.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("NVQ input must contain only finite values")
    return value, int(value.shape[0]), int(value.shape[1])


def _pad_weight(
    value: torch.Tensor,
    groupsize: int,
    importance: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    out, neuron_len = value.shape
    if importance is None:
        objective_weight = torch.ones_like(value)
    else:
        objective_weight = torch.as_tensor(
            importance, device=value.device, dtype=torch.float32
        )
        if objective_weight.dim() == 1:
            if objective_weight.numel() != neuron_len:
                raise ValueError(
                    f"importance has {objective_weight.numel()} entries, expected {neuron_len}"
                )
            objective_weight = objective_weight.unsqueeze(0).expand(out, neuron_len)
        elif tuple(objective_weight.shape) != (out, neuron_len):
            raise ValueError(
                f"importance must have shape ({neuron_len},) or {(out, neuron_len)}, "
                f"got {tuple(objective_weight.shape)}"
            )
        if not bool(torch.isfinite(objective_weight).all()) or bool(
            (objective_weight < 0).any()
        ):
            raise ValueError("importance weights must be finite and non-negative")
        objective_weight = objective_weight.contiguous()
    pad = (-int(neuron_len)) % groupsize
    if pad:
        value = torch.nn.functional.pad(value, (0, pad))
        objective_weight = torch.nn.functional.pad(objective_weight, (0, pad))
    return value, objective_weight, int(value.shape[1] // groupsize)


def _fp16_round(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float16).to(torch.float32)


def _encode_even_parity_signs(
    value: torch.Tensor,
    objective_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, padded_len = value.shape
    if padded_len % 8:
        raise ValueError("internal NVQ sign padding must be divisible by 8")
    x8 = value.reshape(out, padded_len // 8, 8)
    w8 = objective_weight.reshape_as(x8)
    encoded_negative = x8 < 0
    odd = encoded_negative.sum(dim=-1).bitwise_and(1).bool()
    flip_index = torch.argmin(w8 * x8 * x8, dim=-1)
    toggle = torch.nn.functional.one_hot(flip_index, num_classes=8).bool()
    encoded_negative = torch.logical_xor(encoded_negative, toggle & odd.unsqueeze(-1))
    bit_weights = (1 << torch.arange(7, device=value.device, dtype=torch.int64)).view(1, 1, 7)
    masks = (encoded_negative[..., :7].to(torch.int64) * bit_weights).sum(dim=-1).to(torch.uint8)
    sign = torch.where(encoded_negative, -torch.ones_like(x8), torch.ones_like(x8))
    return (x8 * sign).reshape(out, padded_len), masks


def _indices_at_scale(
    cross: torch.Tensor,
    quad: torch.Tensor,
    scale_per_vector: torch.Tensor,
) -> torch.Tensor:
    scale = scale_per_vector.unsqueeze(-1)
    distance = scale * scale * quad - 2.0 * scale * cross
    return torch.argmin(distance, dim=-1)


def _refit_group_scale(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    indices: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    code = codebook[indices].reshape_as(xgroup)
    numerator = (wgroup * xgroup * code).sum(dim=-1)
    denominator = (wgroup * code * code).sum(dim=-1)
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator)).clamp_min(0)


def _search_nvq_groups(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    spec: NvqSpec,
    codebook: torch.Tensor,
    *,
    search_steps: int,
    group_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if search_steps < 1:
        raise ValueError("search_steps must be positive")
    n_groups, groupsize = xgroup.shape
    vectors_per_group = groupsize // spec.vector_size
    scales = torch.empty(n_groups, device=xgroup.device, dtype=torch.float32)
    indices_out = torch.empty(
        (n_groups, vectors_per_group), device=xgroup.device, dtype=torch.int64
    )
    qmax = float(codebook.max().item())
    offsets = torch.linspace(
        -0.12 * qmax,
        0.12 * qmax,
        search_steps,
        device=xgroup.device,
        dtype=torch.float32,
    )
    codebook_t = codebook.T.contiguous()
    codebook2_t = (codebook * codebook).T.contiguous()

    for start in range(0, n_groups, group_chunk):
        stop = min(start + group_chunk, n_groups)
        xg = xgroup[start:stop]
        wg = wgroup[start:stop]
        groups = stop - start
        xv = xg.reshape(groups * vectors_per_group, spec.vector_size)
        wv = wg.reshape_as(xv)
        cross = (wv * xv) @ codebook_t
        quad = wv @ codebook2_t
        max_abs = xg.abs().amax(dim=-1)
        best_error = torch.full((groups,), torch.inf, device=xgroup.device)
        best_scale = torch.zeros(groups, device=xgroup.device)
        best_indices = torch.zeros(
            (groups, vectors_per_group), device=xgroup.device, dtype=torch.int64
        )

        for offset in offsets:
            denominator = qmax + offset
            initial = torch.where(max_abs > 0, max_abs / denominator, torch.zeros_like(max_abs))
            idx = _indices_at_scale(
                cross,
                quad,
                initial.repeat_interleave(vectors_per_group),
            ).reshape(groups, vectors_per_group)
            scale = _refit_group_scale(xg, wg, idx, codebook)
            idx = _indices_at_scale(
                cross,
                quad,
                scale.repeat_interleave(vectors_per_group),
            ).reshape(groups, vectors_per_group)
            scale = _refit_group_scale(xg, wg, idx, codebook)
            code = codebook[idx].reshape_as(xg)
            error = (wg * (scale.unsqueeze(-1) * code - xg).square()).sum(dim=-1)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_scale = torch.where(better, scale, best_scale)
            best_indices[better] = idx[better]

        scales[start:stop] = best_scale
        indices_out[start:stop] = best_indices
    return scales, indices_out


def _reassign_nvq(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    scale: torch.Tensor,
    spec: NvqSpec,
    codebook: torch.Tensor,
    *,
    group_chunk: int,
) -> torch.Tensor:
    n_groups, groupsize = xgroup.shape
    vectors_per_group = groupsize // spec.vector_size
    result = torch.empty(
        (n_groups, vectors_per_group), device=xgroup.device, dtype=torch.int64
    )
    codebook_t = codebook.T.contiguous()
    codebook2_t = (codebook * codebook).T.contiguous()
    for start in range(0, n_groups, group_chunk):
        stop = min(start + group_chunk, n_groups)
        groups = stop - start
        xv = xgroup[start:stop].reshape(groups * vectors_per_group, spec.vector_size)
        wv = wgroup[start:stop].reshape_as(xv)
        cross = (wv * xv) @ codebook_t
        quad = wv @ codebook2_t
        result[start:stop] = _indices_at_scale(
            cross,
            quad,
            scale[start:stop].repeat_interleave(vectors_per_group),
        ).reshape(groups, vectors_per_group)
    return result


def _refit_nvq_anchor(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    sub_scale: torch.Tensor,
    indices: torch.Tensor,
    spec: NvqSpec,
    codebook: torch.Tensor,
    *,
    out: int,
    ng: int,
) -> torch.Tensor:
    code = codebook[indices].reshape_as(xgroup)
    basis = sub_scale.reshape(-1, 1).to(torch.float32) * code
    numerator = (wgroup * xgroup * basis).reshape(out, ng, spec.groupsize).sum(dim=(1, 2))
    denominator = (wgroup * basis * basis).reshape(out, ng, spec.groupsize).sum(dim=(1, 2))
    anchor = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))
    return _fp16_round(anchor.clamp_min(0))


def _quantize_nvq(
    weight: torch.Tensor,
    spec: NvqSpec,
    *,
    device: str | torch.device,
    importance: np.ndarray | torch.Tensor | None,
    search_steps: int,
    group_chunk: int,
    custom_codebook: np.ndarray | None,
    native_assignment: bool,
) -> NvqTensor:
    if spec.sign_mode != "even":
        raise ValueError(
            "Torch NVQ quantization does not support index-parity signs"
        )
    value, out, neuron_len = _prepare_weight(weight, device)
    value, objective_weight, ng = _pad_weight(value, spec.groupsize, importance)
    target, signs = _encode_even_parity_signs(value, objective_weight)
    xgroup = target.reshape(out * ng, spec.groupsize)
    wgroup = objective_weight.reshape_as(xgroup)
    codebook_cpu = (
        codebook_for(spec)
        if custom_codebook is None
        else validate_codebook(spec, custom_codebook)
    )
    codebook = torch.as_tensor(codebook_cpu, device=value.device, dtype=torch.float32)
    native_codebook = codebook.to(torch.int8).contiguous() if native_assignment else None

    if native_codebook is not None:
        from mfq.quantize.cuda._ext import ext

        raw_scale, indices = ext().nvq_search(
            xgroup,
            wgroup.contiguous(),
            native_codebook,
            ng,
            neuron_len - (ng - 1) * spec.groupsize,
            spec.vector_size,
            search_steps,
            float(codebook_cpu.max()),
        )
    else:
        raw_scale, indices = _search_nvq_groups(
            xgroup,
            wgroup,
            spec,
            codebook,
            search_steps=search_steps,
            group_chunk=group_chunk,
        )
    scale_levels = (1 << spec.sub_bits) - 1
    raw_scale_2d = raw_scale.reshape(out, ng)
    row_max = raw_scale_2d.amax(dim=-1)
    neuron_scale = _fp16_round(
        torch.where(row_max > 0, row_max / float(scale_levels), torch.zeros_like(row_max))
    )
    safe_anchor = torch.where(neuron_scale > 0, neuron_scale, torch.ones_like(neuron_scale))
    sub_scale = torch.clamp(
        torch.round(raw_scale_2d / safe_anchor.unsqueeze(-1)), 0, scale_levels
    ).to(torch.uint8)

    effective_scale = (neuron_scale.unsqueeze(-1) * sub_scale).reshape(-1)
    if native_codebook is not None:
        indices = ext().nvq_reassign(
            xgroup,
            wgroup.contiguous(),
            effective_scale.contiguous(),
            native_codebook,
            ng,
            neuron_len - (ng - 1) * spec.groupsize,
            spec.vector_size,
        )
    else:
        indices = _reassign_nvq(
            xgroup,
            wgroup,
            effective_scale,
            spec,
            codebook,
            group_chunk=group_chunk,
        )
    neuron_scale = _refit_nvq_anchor(
        xgroup,
        wgroup,
        sub_scale,
        indices,
        spec,
        codebook,
        out=out,
        ng=ng,
    )
    effective_scale = (neuron_scale.unsqueeze(-1) * sub_scale).reshape(-1)
    if native_codebook is not None:
        indices = ext().nvq_reassign(
            xgroup,
            wgroup.contiguous(),
            effective_scale.contiguous(),
            native_codebook,
            ng,
            neuron_len - (ng - 1) * spec.groupsize,
            spec.vector_size,
        )
    else:
        indices = _reassign_nvq(
            xgroup,
            wgroup,
            effective_scale,
            spec,
            codebook,
            group_chunk=group_chunk,
        )
    neuron_scale = _refit_nvq_anchor(
        xgroup,
        wgroup,
        sub_scale,
        indices,
        spec,
        codebook,
        out=out,
        ng=ng,
    )

    nvec = math.ceil(neuron_len / spec.vector_size)
    nsign = math.ceil(neuron_len / 8)
    indices = indices.reshape(out, -1)[:, :nvec]
    index_dtype = np.uint8 if spec.index_bits <= 8 else np.uint16
    return NvqTensor(
        spec=spec,
        shape=(out, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=neuron_scale.cpu().numpy().astype(np.float32, copy=False),
        sub_scale=sub_scale.cpu().numpy().astype(np.uint8, copy=False),
        indices=(
            indices.to(torch.int32)
            .cpu()
            .numpy()
            .astype(index_dtype, copy=False)
        ),
        signs=signs[:, :nsign].cpu().numpy().astype(np.uint8, copy=False),
        codebook=(None if custom_codebook is None else codebook_cpu.copy()),
    )


def _assign_nvq1_l_groups(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    group_anchor: torch.Tensor,
    spec: Nvq1LSpec,
    codebook: torch.Tensor,
    *,
    group_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_groups, groupsize = xgroup.shape
    vectors_per_group = groupsize // spec.vector_size
    qmax = (1 << spec.sub_bits) - 1
    best_scale = torch.zeros(n_groups, device=xgroup.device, dtype=torch.uint8)
    best_delta = torch.zeros(n_groups, device=xgroup.device, dtype=torch.uint8)
    best_indices = torch.zeros(
        (n_groups, vectors_per_group), device=xgroup.device, dtype=torch.int64
    )

    for start in range(0, n_groups, group_chunk):
        stop = min(start + group_chunk, n_groups)
        groups = stop - start
        xg = xgroup[start:stop]
        wg = wgroup[start:stop]
        anchors = group_anchor[start:stop]
        xv = xg.reshape(groups * vectors_per_group, spec.vector_size)
        wv = wg.reshape_as(xv)
        weighted_x = wv * xv
        const = (weighted_x * xv).sum(dim=-1)
        chunk_error = torch.full((groups,), torch.inf, device=xgroup.device)
        chunk_scale = torch.zeros(groups, device=xgroup.device, dtype=torch.uint8)
        chunk_delta = torch.zeros(groups, device=xgroup.device, dtype=torch.uint8)
        chunk_indices = torch.zeros(
            (groups, vectors_per_group), device=xgroup.device, dtype=torch.int64
        )

        for delta_bit, delta in ((0, spec.delta), (1, -spec.delta)):
            shifted = codebook + float(delta)
            cross = weighted_x @ shifted.T
            quad = wv @ (shifted * shifted).T
            for q in range(qmax + 1):
                vector_scale = (anchors * float(q)).repeat_interleave(vectors_per_group)
                scale_column = vector_scale.unsqueeze(-1)
                variable = scale_column * scale_column * quad - 2.0 * scale_column * cross
                indices = torch.argmin(variable, dim=-1)
                rows = torch.arange(indices.numel(), device=xgroup.device)
                vector_error = const + variable[rows, indices]
                group_error = vector_error.reshape(groups, vectors_per_group).sum(dim=-1)
                improve = group_error < chunk_error
                chunk_error = torch.where(improve, group_error, chunk_error)
                chunk_scale[improve] = q
                chunk_delta[improve] = delta_bit
                chunk_indices[improve] = indices.reshape(groups, vectors_per_group)[improve]

        best_scale[start:stop] = chunk_scale
        best_delta[start:stop] = chunk_delta
        best_indices[start:stop] = chunk_indices
    return best_scale, best_delta, best_indices


def _build_nvq1_l_assignment_cache(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    spec: Nvq1LSpec,
    codebook: torch.Tensor,
    *,
    group_chunk: int,
) -> list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Cache codebook products shared by every anchor/refit assignment."""

    n_groups, groupsize = xgroup.shape
    vectors_per_group = groupsize // spec.vector_size
    cache = []
    codebook_t = codebook.T.contiguous()
    for start in range(0, n_groups, group_chunk):
        stop = min(start + group_chunk, n_groups)
        groups = stop - start
        xv = xgroup[start:stop].reshape(groups * vectors_per_group, spec.vector_size)
        wv = wgroup[start:stop].reshape_as(xv)
        vector_weight = wv[:, :1]
        if not bool(torch.equal(wv, vector_weight.expand_as(wv))):
            raise ValueError("cached CUDA NVQ1-L currently requires uniform weight inside each 8-vector")
        weighted_x = wv * xv
        cache.append(
            (
                start,
                stop,
                weighted_x @ codebook_t,
                weighted_x.sum(dim=-1),
                vector_weight.reshape(-1),
                (weighted_x * xv).sum(dim=-1),
            )
        )
    return cache


def _assign_nvq1_l_groups_cached(
    group_anchor: torch.Tensor,
    spec: Nvq1LSpec,
    codebook: torch.Tensor,
    cache: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    n_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vectors_per_group = spec.groupsize // spec.vector_size
    qmax = (1 << spec.sub_bits) - 1
    device = group_anchor.device
    best_scale = torch.zeros(n_groups, device=device, dtype=torch.uint8)
    best_delta = torch.zeros(n_groups, device=device, dtype=torch.uint8)
    best_indices = torch.zeros((n_groups, vectors_per_group), device=device, dtype=torch.int64)

    for start, stop, base_cross, weighted_sum, vector_weight, const in cache:
        groups = stop - start
        anchors = group_anchor[start:stop]
        chunk_error = torch.full((groups,), torch.inf, device=device)
        chunk_scale = torch.zeros(groups, device=device, dtype=torch.uint8)
        chunk_delta = torch.zeros(groups, device=device, dtype=torch.uint8)
        chunk_indices = torch.zeros(
            (groups, vectors_per_group), device=device, dtype=torch.int64
        )
        for delta_bit, delta in ((0, spec.delta), (1, -spec.delta)):
            shifted_norm = (codebook + float(delta)).square().sum(dim=-1)
            cross = base_cross + float(delta) * weighted_sum.unsqueeze(-1)
            quad = vector_weight.unsqueeze(-1) * shifted_norm.unsqueeze(0)
            for q in range(qmax + 1):
                vector_scale = (anchors * float(q)).repeat_interleave(vectors_per_group)
                scale_column = vector_scale.unsqueeze(-1)
                variable = scale_column * scale_column * quad - 2.0 * scale_column * cross
                indices = torch.argmin(variable, dim=-1)
                rows = torch.arange(indices.numel(), device=device)
                vector_error = const + variable[rows, indices]
                group_error = vector_error.reshape(groups, vectors_per_group).sum(dim=-1)
                improve = group_error < chunk_error
                chunk_error = torch.where(improve, group_error, chunk_error)
                chunk_scale[improve] = q
                chunk_delta[improve] = delta_bit
                chunk_indices[improve] = indices.reshape(groups, vectors_per_group)[improve]
        best_scale[start:stop] = chunk_scale
        best_delta[start:stop] = chunk_delta
        best_indices[start:stop] = chunk_indices
    return best_scale, best_delta, best_indices


@lru_cache(maxsize=8)
def _nvq1_l_candidate_table_cpu(candidate_count: int) -> np.ndarray:
    if not 1 <= candidate_count <= 2048:
        raise ValueError("NVQ1-L candidate count must be in [1, 2048]")
    ids = np.arange(3**8, dtype=np.int32)
    powers = (3 ** np.arange(8, dtype=np.int32))[None, :]
    grid = ((ids[:, None] // powers) % 3 - 1).astype(np.float32)
    codebook = np.asarray(IQ1S_TERNARY_2048, dtype=np.float32)
    distance = np.square(grid[:, None, :] - codebook[None, :, :]).sum(axis=-1)
    if candidate_count == 2048:
        result = np.argsort(distance, axis=-1, kind="stable")
    else:
        result = np.argpartition(distance, candidate_count - 1, axis=-1)[:, :candidate_count]
        local_distance = np.take_along_axis(distance, result, axis=-1)
        order = np.argsort(local_distance, axis=-1, kind="stable")
        result = np.take_along_axis(result, order, axis=-1)
    return np.ascontiguousarray(result, dtype=np.int16)


def _assign_nvq1_l_groups_candidates(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    group_anchor: torch.Tensor,
    spec: Nvq1LSpec,
    codebook: torch.Tensor,
    *,
    group_chunk: int,
    candidate_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_groups, groupsize = xgroup.shape
    vectors_per_group = groupsize // spec.vector_size
    qmax = (1 << spec.sub_bits) - 1
    best_scale = torch.zeros(n_groups, device=xgroup.device, dtype=torch.uint8)
    best_delta = torch.zeros(n_groups, device=xgroup.device, dtype=torch.uint8)
    best_indices = torch.zeros(
        (n_groups, vectors_per_group), device=xgroup.device, dtype=torch.int64
    )
    candidates = torch.as_tensor(
        _nvq1_l_candidate_table_cpu(candidate_count).astype(np.int32),
        device=xgroup.device,
        dtype=torch.int64,
    )
    ternary_powers = 3 ** torch.arange(8, device=xgroup.device, dtype=torch.int64)

    for start in range(0, n_groups, group_chunk):
        stop = min(start + group_chunk, n_groups)
        groups = stop - start
        xg = xgroup[start:stop]
        wg = wgroup[start:stop]
        anchors = group_anchor[start:stop]
        xv = xg.reshape(groups * vectors_per_group, spec.vector_size)
        wv = wg.reshape_as(xv)
        const = (wv * xv * xv).sum(dim=-1)
        chunk_error = const.reshape(groups, vectors_per_group).sum(dim=-1)
        chunk_scale = torch.zeros(groups, device=xgroup.device, dtype=torch.uint8)
        chunk_delta = torch.zeros(groups, device=xgroup.device, dtype=torch.uint8)
        chunk_indices = torch.zeros(
            (groups, vectors_per_group), device=xgroup.device, dtype=torch.int64
        )

        for delta_bit, delta in ((0, spec.delta), (1, -spec.delta)):
            for q in range(1, qmax + 1):
                vector_scale = (anchors * float(q)).repeat_interleave(vectors_per_group)
                safe_scale = torch.where(
                    vector_scale > 0, vector_scale, torch.ones_like(vector_scale)
                )
                normalized = xv / safe_scale.unsqueeze(-1) - float(delta)
                ternary = torch.round(normalized).clamp(-1, 1).to(torch.int64)
                grid_id = ((ternary + 1) * ternary_powers).sum(dim=-1)
                candidate_index = candidates[grid_id]
                candidate_code = codebook[candidate_index] + float(delta)
                diff = vector_scale[:, None, None] * candidate_code - xv[:, None, :]
                error = (wv[:, None, :] * diff.square()).sum(dim=-1)
                local = torch.argmin(error, dim=-1)
                rows = torch.arange(local.numel(), device=xgroup.device)
                indices = candidate_index[rows, local]
                vector_error = error[rows, local]
                group_error = vector_error.reshape(groups, vectors_per_group).sum(dim=-1)
                improve = group_error < chunk_error
                chunk_error = torch.where(improve, group_error, chunk_error)
                chunk_scale[improve] = q
                chunk_delta[improve] = delta_bit
                chunk_indices[improve] = indices.reshape(groups, vectors_per_group)[improve]

        best_scale[start:stop] = chunk_scale
        best_delta[start:stop] = chunk_delta
        best_indices[start:stop] = chunk_indices
    return best_scale, best_delta, best_indices


def _nvq1_l_basis(
    sub_scale: torch.Tensor,
    delta_sign: torch.Tensor,
    indices: torch.Tensor,
    spec: Nvq1LSpec,
    codebook: torch.Tensor,
) -> torch.Tensor:
    code = codebook[indices].reshape(sub_scale.numel(), spec.groupsize)
    delta = torch.where(
        delta_sign != 0,
        torch.full_like(delta_sign, -spec.delta, dtype=torch.float32),
        torch.full_like(delta_sign, spec.delta, dtype=torch.float32),
    )
    return sub_scale.to(torch.float32).unsqueeze(-1) * (code + delta.unsqueeze(-1))


def _refit_nvq1_l_anchor(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    sub_scale: torch.Tensor,
    delta_sign: torch.Tensor,
    indices: torch.Tensor,
    *,
    out: int,
    ng: int,
    spec: Nvq1LSpec,
    codebook: torch.Tensor,
) -> torch.Tensor:
    basis = _nvq1_l_basis(sub_scale, delta_sign, indices, spec, codebook)
    numerator = (wgroup * xgroup * basis).reshape(out, ng * spec.groupsize).sum(dim=-1)
    denominator = (wgroup * basis * basis).reshape(out, ng * spec.groupsize).sum(dim=-1)
    anchor = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))
    return _fp16_round(anchor.clamp_min(0))


def _nvq1_l_row_error(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    sub_scale: torch.Tensor,
    delta_sign: torch.Tensor,
    indices: torch.Tensor,
    anchor: torch.Tensor,
    *,
    out: int,
    ng: int,
    spec: Nvq1LSpec,
    codebook: torch.Tensor,
) -> torch.Tensor:
    basis = _nvq1_l_basis(sub_scale, delta_sign, indices, spec, codebook)
    reconstruction = basis * anchor.repeat_interleave(ng).unsqueeze(-1)
    return (wgroup * (xgroup - reconstruction).square()).reshape(
        out, ng * spec.groupsize
    ).sum(dim=-1)


def _solve_nvq1_l_from_anchor(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    initial_anchor: torch.Tensor,
    *,
    out: int,
    ng: int,
    spec: Nvq1LSpec,
    codebook: torch.Tensor,
    group_chunk: int,
    refine_steps: int,
    candidate_count: int,
    assignment_cache: list[
        tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]
    | None,
    native_codebook: torch.Tensor | None,
    groups_per_row: int,
    valid_last: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor = _fp16_round(initial_anchor)
    if native_codebook is not None:
        def assign(current_anchor):
            from mfq.quantize.cuda._ext import ext

            return ext().nvq1_l_assign(
                xgroup,
                wgroup.contiguous(),
                current_anchor.repeat_interleave(groups_per_row).contiguous(),
                native_codebook,
                groups_per_row,
                valid_last,
                spec.sub_bits,
                spec.delta,
            )
    elif assignment_cache is not None:
        def assign(current_anchor):
            return _assign_nvq1_l_groups_cached(
                current_anchor.repeat_interleave(ng),
                spec,
                codebook,
                assignment_cache,
                xgroup.shape[0],
            )
    else:
        assign_impl = _assign_nvq1_l_groups_candidates if candidate_count else _assign_nvq1_l_groups
        assign_kwargs = {"candidate_count": candidate_count} if candidate_count else {}

        def assign(current_anchor):
            return assign_impl(
                xgroup,
                wgroup,
                current_anchor.repeat_interleave(ng),
                spec,
                codebook,
                group_chunk=group_chunk,
                **assign_kwargs,
            )

    scale, delta, indices = assign(anchor)
    error = _nvq1_l_row_error(
        xgroup,
        wgroup,
        scale,
        delta,
        indices,
        anchor,
        out=out,
        ng=ng,
        spec=spec,
        codebook=codebook,
    )
    for _ in range(refine_steps):
        candidate_anchor = _refit_nvq1_l_anchor(
            xgroup,
            wgroup,
            scale,
            delta,
            indices,
            out=out,
            ng=ng,
            spec=spec,
            codebook=codebook,
        )
        candidate_scale, candidate_delta, candidate_indices = assign(candidate_anchor)
        candidate_error = _nvq1_l_row_error(
            xgroup,
            wgroup,
            candidate_scale,
            candidate_delta,
            candidate_indices,
            candidate_anchor,
            out=out,
            ng=ng,
            spec=spec,
            codebook=codebook,
        )
        improve = candidate_error < error
        if not bool(improve.any()):
            break
        group_improve = improve.repeat_interleave(ng)
        scale[group_improve] = candidate_scale[group_improve]
        delta[group_improve] = candidate_delta[group_improve]
        indices[group_improve] = candidate_indices[group_improve]
        anchor[improve] = candidate_anchor[improve]
        error[improve] = candidate_error[improve]
    return scale, delta, indices, anchor, error


def _quantize_nvq1_l(
    weight: torch.Tensor,
    spec: Nvq1LSpec,
    *,
    device: str | torch.device,
    importance: np.ndarray | torch.Tensor | None,
    anchor_multipliers: Sequence[float],
    refine_steps: int,
    group_chunk: int,
    candidate_count: int,
    custom_codebook: np.ndarray | None,
    native_assignment: bool,
) -> Nvq1LTensor:
    value, out, neuron_len = _prepare_weight(weight, device)
    value, objective_weight, ng = _pad_weight(value, spec.groupsize, importance)
    xgroup = value.reshape(out * ng, spec.groupsize)
    wgroup = objective_weight.reshape_as(xgroup)
    if custom_codebook is not None and candidate_count:
        raise ValueError("NVQ1-L candidate lookup currently requires the fixed IQ1_S codebook")
    codebook_cpu = (
        np.array(IQ1S_TERNARY_2048, copy=True)
        if custom_codebook is None
        else validate_ternary_codebook(custom_codebook)
    )
    codebook = torch.tensor(
        codebook_cpu,
        device=value.device,
        dtype=torch.float32,
    )
    qmax = (1 << spec.sub_bits) - 1
    row_peak = value.abs().amax(dim=-1)
    base_anchor = torch.where(
        row_peak > 0,
        row_peak / float(qmax * (1.0 + spec.delta)),
        torch.zeros_like(row_peak),
    )
    best_error = torch.full((out,), torch.inf, device=value.device)
    best_scale = torch.zeros((out, ng), device=value.device, dtype=torch.uint8)
    best_delta = torch.zeros((out, ng), device=value.device, dtype=torch.uint8)
    best_indices = torch.zeros((out, ng, 3), device=value.device, dtype=torch.int64)
    best_anchor = torch.zeros(out, device=value.device)
    assignment_cache = None
    native_codebook = codebook.to(torch.int8).contiguous() if native_assignment else None
    if not candidate_count and not native_assignment and neuron_len % spec.vector_size == 0:
        assignment_cache = _build_nvq1_l_assignment_cache(
            xgroup,
            wgroup,
            spec,
            codebook,
            group_chunk=group_chunk,
        )

    for multiplier in anchor_multipliers:
        scale, delta, indices, anchor, error = _solve_nvq1_l_from_anchor(
            xgroup,
            wgroup,
            base_anchor * float(multiplier),
            out=out,
            ng=ng,
            spec=spec,
            codebook=codebook,
            group_chunk=group_chunk,
            refine_steps=refine_steps,
            candidate_count=candidate_count,
            assignment_cache=assignment_cache,
            native_codebook=native_codebook,
            groups_per_row=ng,
            valid_last=neuron_len - (ng - 1) * spec.groupsize,
        )
        improve = error < best_error
        group_improve = improve.repeat_interleave(ng)
        best_scale.reshape(-1)[group_improve] = scale[group_improve]
        best_delta.reshape(-1)[group_improve] = delta[group_improve]
        best_indices.reshape(-1, 3)[group_improve] = indices[group_improve]
        best_anchor[improve] = anchor[improve]
        best_error[improve] = error[improve]

    nvec = math.ceil(neuron_len / spec.vector_size)
    indices = best_indices.reshape(out, -1)[:, :nvec]
    return Nvq1LTensor(
        spec=spec,
        shape=(out, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=best_anchor.cpu().numpy().astype(np.float32, copy=False),
        sub_scale=best_scale.cpu().numpy().astype(np.uint8, copy=False),
        indices=indices.to(torch.int32).cpu().numpy().astype(np.uint16, copy=False),
        delta_sign=best_delta.cpu().numpy().astype(np.uint8, copy=False),
        codebook=(None if custom_codebook is None else codebook_cpu.copy()),
    )


@torch.inference_mode()
def quantize_axis0(
    weight: torch.Tensor,
    spec: NvqSpec | Nvq1LSpec,
    *,
    device: str | torch.device = "cuda",
    importance: np.ndarray | torch.Tensor | None = None,
    search_steps: int = 19,
    anchor_multipliers: Sequence[float] = (0.75, 1.0, 1.25),
    refine_steps: int = 2,
    group_chunk: int = 1024,
    nvq1_l_candidates: int = 0,
    codebook: np.ndarray | None = None,
    nvq_native_assignment: bool = True,
    nvq1_l_native_assignment: bool = True,
) -> NvqTensor | Nvq1LTensor:
    """Quantize one ``[out, in]`` chunk with the reference NVQ objective on CUDA."""

    if group_chunk <= 0:
        raise ValueError("group_chunk must be positive")
    if refine_steps < 0:
        raise ValueError("refine_steps must be non-negative")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if isinstance(spec, Nvq1LSpec):
            return _quantize_nvq1_l(
                weight,
                spec,
                device=device,
                importance=importance,
                anchor_multipliers=anchor_multipliers,
                refine_steps=refine_steps,
                group_chunk=group_chunk,
                candidate_count=nvq1_l_candidates,
                custom_codebook=codebook,
                native_assignment=nvq1_l_native_assignment,
            )
        return _quantize_nvq(
            weight,
            spec,
            device=device,
            importance=importance,
            search_steps=search_steps,
            group_chunk=group_chunk,
            custom_codebook=codebook,
            native_assignment=nvq_native_assignment,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
