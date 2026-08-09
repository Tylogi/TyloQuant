"""Offline exact-search additive codebooks for NVQ1-L and NVQ1-S.

The packed weight streams stay unchanged: two indices replace the original
11-bit or 9-bit vector index, while the group scale, delta bit, and FP16 neuron
anchor retain their existing meanings.  This module is a numeric experiment;
it does not add a serialization format or an inference kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from mfq.quantize.nvq_product import _nearest, _weighted_kmeans
from mfq.quantize.nvq_quant_torch import _fp16_round, _pad_weight, _prepare_weight

_GROUP_SIZE = 24
_VECTOR_SIZE = 8
_VECTORS_PER_GROUP = 3


@dataclass(frozen=True)
class NvqLowBitAdditiveConfig:
    index_bits: int
    first_bits: int
    sub_bits: int
    delta: float
    banks: int
    codebook_step: float = 0.25
    iterations: int = 3
    assignment_refine_steps: int = 2
    fixed_refine_steps: int = 2
    kmeans_iterations: int = 6
    kmeans_initialization_points: int = 8192
    group_chunk: int = 128
    anchor_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    seed: int = 20260721

    def __post_init__(self) -> None:
        if self.index_bits not in {9, 11}:
            raise ValueError("low-bit AQ index_bits must be 9 or 11")
        if not 1 <= self.first_bits < self.index_bits:
            raise ValueError("first_bits must split the vector index")
        if self.banks not in {1, 2}:
            raise ValueError("low-bit AQ banks must be 1 or 2")
        if self.sub_bits <= 0 or not math.isfinite(self.delta) or self.delta < 0:
            raise ValueError("sub_bits and delta must define a valid group scale")
        if not math.isfinite(self.codebook_step) or self.codebook_step <= 0:
            raise ValueError("codebook_step must be finite and positive")
        counts = (
            self.iterations,
            self.assignment_refine_steps,
            self.fixed_refine_steps,
            self.kmeans_iterations,
        )
        if any(value < 0 for value in counts):
            raise ValueError("iteration counts must be non-negative")
        if self.kmeans_initialization_points <= 0 or self.group_chunk <= 0:
            raise ValueError("chunk and initialization sizes must be positive")
        if not self.anchor_multipliers or any(
            not math.isfinite(value) or value <= 0 for value in self.anchor_multipliers
        ):
            raise ValueError("anchor multipliers must be finite and positive")

    @property
    def second_bits(self) -> int:
        return self.index_bits - self.first_bits

    @property
    def first_entries(self) -> int:
        return 1 << self.first_bits

    @property
    def second_entries(self) -> int:
        return 1 << self.second_bits

    @property
    def effective_entries(self) -> int:
        return 1 << self.index_bits


@dataclass(frozen=True)
class NvqLowBitAdditiveTables:
    codebook_step: np.ndarray
    first_codebooks: np.ndarray
    second_codebooks: np.ndarray


@dataclass
class NvqLowBitAssignment:
    shape: tuple[int, int]
    neuron_scale: np.ndarray
    sub_scale: np.ndarray
    delta_sign: np.ndarray
    indices: np.ndarray


@dataclass
class NvqLowBitAdditiveTensor:
    shape: tuple[int, int]
    neuron_scale: np.ndarray
    sub_scale: np.ndarray
    delta_sign: np.ndarray
    first_indices: np.ndarray
    second_indices: np.ndarray
    codebook_step: np.ndarray
    first_codebooks: np.ndarray
    second_codebooks: np.ndarray


@dataclass(frozen=True)
class NvqLowBitAdditiveIteration:
    iteration: int
    weighted_sse: float
    weighted_nmse_percent: float
    used_first_codes: tuple[int, ...]
    used_second_codes: tuple[int, ...]


def _validate_effective_codebooks(
    codebooks: np.ndarray,
    config: NvqLowBitAdditiveConfig,
) -> np.ndarray:
    value = np.asarray(codebooks)
    expected = (config.banks, config.effective_entries, _VECTOR_SIZE)
    if value.shape != expected:
        raise ValueError(f"effective codebooks have shape {value.shape}, expected {expected}")
    if not np.isfinite(value).all():
        raise ValueError("effective codebooks must be finite")
    return np.ascontiguousarray(value, dtype=np.float32)


def _validate_additive_tables(
    tables: NvqLowBitAdditiveTables,
    config: NvqLowBitAdditiveConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = np.asarray(tables.codebook_step, dtype=np.float32).reshape(-1)
    first = np.asarray(tables.first_codebooks)
    second = np.asarray(tables.second_codebooks)
    expected_first = (config.banks, config.first_entries, _VECTOR_SIZE)
    expected_second = (config.banks, config.second_entries, _VECTOR_SIZE)
    if step.shape != (config.banks,) or not np.isfinite(step).all() or np.any(step <= 0):
        raise ValueError(f"codebook_step must contain {config.banks} positive values")
    if first.shape != expected_first or second.shape != expected_second:
        raise ValueError(
            f"additive codebooks have shapes {first.shape}/{second.shape}, "
            f"expected {expected_first}/{expected_second}"
        )
    for name, value in (("first", first), ("second", second)):
        rounded = np.rint(value)
        if not np.array_equal(value, rounded):
            raise ValueError(f"{name} additive codebook must contain integers")
        if np.any(rounded < -127) or np.any(rounded > 127):
            raise ValueError(f"{name} additive codebook exceeds signed int8")
    return (
        np.ascontiguousarray(step, dtype=np.float32),
        np.ascontiguousarray(first, dtype=np.int8),
        np.ascontiguousarray(second, dtype=np.int8),
    )


def combined_codebooks(
    tables: NvqLowBitAdditiveTables,
    config: NvqLowBitAdditiveConfig,
) -> np.ndarray:
    step, first, second = _validate_additive_tables(tables, config)
    combined = first.astype(np.int16)[:, :, None, :] + second.astype(np.int16)[:, None, :, :]
    combined = combined.reshape(config.banks, -1, _VECTOR_SIZE).astype(np.float32)
    return np.ascontiguousarray(combined * step[:, None, None])


def additive_tables_from_tensor(
    tensor: NvqLowBitAdditiveTensor,
) -> NvqLowBitAdditiveTables:
    return NvqLowBitAdditiveTables(
        codebook_step=np.asarray(tensor.codebook_step, dtype=np.float32).copy(),
        first_codebooks=np.asarray(tensor.first_codebooks, dtype=np.int8).copy(),
        second_codebooks=np.asarray(tensor.second_codebooks, dtype=np.int8).copy(),
    )


def _assign_groups(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    neuron_scale: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    ng: int,
    config: NvqLowBitAdditiveConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = xgroup.shape[0]
    group_anchor = neuron_scale.repeat_interleave(ng)
    qmax = (1 << config.sub_bits) - 1
    best_error = torch.full((groups,), torch.inf, device=xgroup.device)
    best_scale = torch.zeros(groups, dtype=torch.int64, device=xgroup.device)
    best_delta = torch.zeros(groups, dtype=torch.int64, device=xgroup.device)
    best_indices = torch.zeros(
        (groups, _VECTORS_PER_GROUP), dtype=torch.int64, device=xgroup.device
    )

    for start in range(0, groups, config.group_chunk):
        stop = min(start + config.group_chunk, groups)
        count = stop - start
        xv = xgroup[start:stop].reshape(-1, _VECTOR_SIZE)
        wv = wgroup[start:stop].reshape_as(xv)
        weighted_x = wv * xv
        constant = (weighted_x * xv).sum(1)
        local_error = best_error[start:stop]
        local_scale = best_scale[start:stop]
        local_delta = best_delta[start:stop]
        local_indices = best_indices[start:stop]

        for delta_bit, delta in ((0, config.delta), (1, -config.delta)):
            bank = delta_bit if config.banks == 2 else 0
            shifted = codebooks[bank] + float(delta)
            cross = weighted_x @ shifted.T
            quadratic = wv @ shifted.square().T
            for q in range(qmax + 1):
                scale = (group_anchor[start:stop] * float(q)).repeat_interleave(_VECTORS_PER_GROUP)
                variable = scale[:, None].square() * quadratic - 2.0 * scale[:, None] * cross
                indices = variable.argmin(1)
                vector_error = constant + variable.gather(1, indices[:, None]).squeeze(1)
                group_error = vector_error.reshape(count, _VECTORS_PER_GROUP).sum(1)
                better = group_error < local_error
                local_error = torch.where(better, group_error, local_error)
                local_scale = torch.where(better, torch.full_like(local_scale, q), local_scale)
                local_delta = torch.where(
                    better, torch.full_like(local_delta, delta_bit), local_delta
                )
                local_indices[better] = indices.reshape(count, _VECTORS_PER_GROUP)[better]

        best_error[start:stop] = local_error
        best_scale[start:stop] = local_scale
        best_delta[start:stop] = local_delta
        best_indices[start:stop] = local_indices
    return best_scale, best_delta, best_indices, best_error


def _group_basis(
    sub_scale: torch.Tensor,
    delta_sign: torch.Tensor,
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    config: NvqLowBitAdditiveConfig,
) -> torch.Tensor:
    bank = delta_sign if config.banks == 2 else torch.zeros_like(delta_sign)
    vector_bank = bank.repeat_interleave(_VECTORS_PER_GROUP)
    code = codebooks[vector_bank, indices.reshape(-1)].reshape(-1, _GROUP_SIZE)
    delta = torch.where(delta_sign != 0, -config.delta, config.delta)
    return sub_scale.to(torch.float32).unsqueeze(1) * (code + delta.unsqueeze(1))


def _refit_anchor(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    sub_scale: torch.Tensor,
    delta_sign: torch.Tensor,
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    out: int,
    ng: int,
    config: NvqLowBitAdditiveConfig,
) -> torch.Tensor:
    basis = _group_basis(sub_scale, delta_sign, indices, codebooks, config)
    numerator = (wgroup * xgroup * basis).reshape(out, -1).sum(1)
    denominator = (wgroup * basis.square()).reshape(out, -1).sum(1)
    anchor = torch.where(denominator > 0, numerator / denominator, 0.0).clamp_min(0.0)
    return _fp16_round(anchor)


def _tensor_from_assignment(
    shape: tuple[int, int],
    neuron_scale: torch.Tensor,
    sub_scale: torch.Tensor,
    delta_sign: torch.Tensor,
    indices: torch.Tensor,
) -> NvqLowBitAssignment:
    out, neuron_len = shape
    ng = math.ceil(neuron_len / _GROUP_SIZE)
    nvec = math.ceil(neuron_len / _VECTOR_SIZE)
    return NvqLowBitAssignment(
        shape=shape,
        neuron_scale=neuron_scale.cpu().numpy().astype(np.float32, copy=False),
        sub_scale=sub_scale.reshape(out, ng).cpu().numpy().astype(np.uint8, copy=False),
        delta_sign=delta_sign.reshape(out, ng).cpu().numpy().astype(np.uint8, copy=False),
        indices=indices.reshape(out, -1)[:, :nvec].cpu().numpy().astype(np.uint16, copy=False),
    )


@torch.inference_mode()
def quantize_effective_fixed(
    weight: torch.Tensor,
    codebooks: np.ndarray,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    config: NvqLowBitAdditiveConfig,
    device: str | torch.device = "cuda",
) -> NvqLowBitAssignment:
    table_np = _validate_effective_codebooks(codebooks, config)
    value, out, neuron_len = _prepare_weight(weight, device)
    value, objective_weight, ng = _pad_weight(value, _GROUP_SIZE, importance)
    xgroup = value.reshape(out * ng, _GROUP_SIZE).contiguous()
    wgroup = objective_weight.reshape_as(xgroup).contiguous()
    table = torch.as_tensor(table_np, device=value.device, dtype=torch.float32)
    qmax = (1 << config.sub_bits) - 1
    positive = table + config.delta
    negative = table - config.delta
    maximum = max(float(positive.abs().max().item()), float(negative.abs().max().item()), 1.0)
    base_anchor = value.abs().amax(1) / (float(qmax) * maximum)

    best_row_error = torch.full((out,), torch.inf, device=value.device)
    best_anchor = torch.zeros(out, device=value.device)
    best_scale = torch.zeros(out * ng, dtype=torch.int64, device=value.device)
    best_delta = torch.zeros_like(best_scale)
    best_indices = torch.zeros(
        (out * ng, _VECTORS_PER_GROUP), dtype=torch.int64, device=value.device
    )
    for multiplier in config.anchor_multipliers:
        anchor = _fp16_round(base_anchor * multiplier)
        scale, delta, indices, error = _assign_groups(
            xgroup, wgroup, anchor, table, ng=ng, config=config
        )
        for _ in range(config.fixed_refine_steps):
            anchor = _refit_anchor(
                xgroup,
                wgroup,
                scale,
                delta,
                indices,
                table,
                out=out,
                ng=ng,
                config=config,
            )
            scale, delta, indices, error = _assign_groups(
                xgroup, wgroup, anchor, table, ng=ng, config=config
            )
        row_error = error.reshape(out, ng).sum(1)
        better = row_error < best_row_error
        best_row_error = torch.where(better, row_error, best_row_error)
        best_anchor = torch.where(better, anchor, best_anchor)
        group_better = better.repeat_interleave(ng)
        best_scale[group_better] = scale[group_better]
        best_delta[group_better] = delta[group_better]
        best_indices[group_better] = indices[group_better]
    return _tensor_from_assignment(
        (out, neuron_len), best_anchor, best_scale, best_delta, best_indices
    )


def dequantize_effective(
    tensor: NvqLowBitAssignment,
    codebooks: np.ndarray,
    config: NvqLowBitAdditiveConfig,
) -> np.ndarray:
    table = _validate_effective_codebooks(codebooks, config)
    out, neuron_len = tensor.shape
    ng = math.ceil(neuron_len / _GROUP_SIZE)
    nvec_padded = ng * _VECTORS_PER_GROUP
    indices = np.zeros((out, nvec_padded), dtype=np.uint16)
    indices[:, : tensor.indices.shape[1]] = tensor.indices
    bank = tensor.delta_sign if config.banks == 2 else np.zeros_like(tensor.delta_sign)
    vector_bank = np.repeat(bank, _VECTORS_PER_GROUP, axis=1)
    code = table[vector_bank, indices].reshape(out, -1)[:, :neuron_len]
    delta = np.where(tensor.delta_sign != 0, -config.delta, config.delta).astype(np.float32)
    group_scale = tensor.neuron_scale[:, None] * tensor.sub_scale.astype(np.float32)
    scale = np.repeat(group_scale, _GROUP_SIZE, axis=1)[:, :neuron_len]
    shift = np.repeat(delta, _GROUP_SIZE, axis=1)[:, :neuron_len]
    return scale * (code + shift)


def _initial_additive_codebooks(
    value: torch.Tensor,
    objective_weight: torch.Tensor,
    initial: NvqLowBitAssignment,
    config: NvqLowBitAdditiveConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, padded_len = value.shape
    ng = padded_len // _GROUP_SIZE
    vectors = value.reshape(-1, _VECTOR_SIZE)
    weights = objective_weight.reshape_as(vectors)
    anchor = torch.as_tensor(initial.neuron_scale, device=value.device, dtype=torch.float32)
    q = torch.as_tensor(initial.sub_scale.reshape(-1), device=value.device, dtype=torch.float32)
    delta_sign = torch.as_tensor(
        initial.delta_sign.reshape(-1), device=value.device, dtype=torch.int64
    )
    scale = (anchor.repeat_interleave(ng) * q).repeat_interleave(_VECTORS_PER_GROUP)
    delta = torch.where(delta_sign != 0, -config.delta, config.delta).repeat_interleave(
        _VECTORS_PER_GROUP
    )
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = (vectors / safe_scale.unsqueeze(1) - delta.unsqueeze(1)) / config.codebook_step
    normalized_weight = weights * scale.square().unsqueeze(1)
    vector_bank = (
        delta_sign.repeat_interleave(_VECTORS_PER_GROUP)
        if config.banks == 2
        else torch.zeros(vectors.shape[0], dtype=torch.int64, device=value.device)
    )
    first = torch.empty(
        (config.banks, config.first_entries, _VECTOR_SIZE), dtype=torch.int8, device=value.device
    )
    second = torch.empty(
        (config.banks, config.second_entries, _VECTOR_SIZE), dtype=torch.int8, device=value.device
    )
    all_vectors = torch.ones_like(vector_bank, dtype=torch.bool)
    for bank_id in range(config.banks):
        selected = vector_bank == bank_id
        if not bool(selected.any()):
            selected = all_vectors
        samples = normalized[selected]
        sample_weight = normalized_weight[selected]
        first[bank_id] = _weighted_kmeans(
            samples,
            sample_weight,
            config.first_entries,
            iterations=config.kmeans_iterations,
            initialization_points=config.kmeans_initialization_points,
            seed=config.seed + 2 * bank_id,
        )
        first_index, _ = _nearest(
            samples,
            sample_weight,
            torch.ones(samples.shape[0], device=value.device),
            first[bank_id],
        )
        residual = samples - first[bank_id][first_index].to(torch.float32)
        second[bank_id] = _weighted_kmeans(
            residual,
            sample_weight,
            config.second_entries,
            iterations=config.kmeans_iterations,
            initialization_points=config.kmeans_initialization_points,
            seed=config.seed + 2 * bank_id + 1,
        )
    return first, second


def _update_one_codebook(
    samples: torch.Tensor,
    objective_weight: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    numerator = torch.zeros_like(previous, dtype=torch.float32)
    denominator = torch.zeros_like(previous, dtype=torch.float32)
    expanded = indices.unsqueeze(1).expand(-1, _VECTOR_SIZE)
    numerator.scatter_add_(0, expanded, objective_weight * scale.unsqueeze(1) * samples)
    denominator.scatter_add_(0, expanded, objective_weight * scale.square().unsqueeze(1))
    centroid = torch.where(denominator > 0, numerator / denominator, previous.to(torch.float32))
    return centroid.round().clamp(-127, 127).to(torch.int8).contiguous()


def _update_codebooks(
    value: torch.Tensor,
    objective_weight: torch.Tensor,
    assignment: NvqLowBitAssignment,
    first: torch.Tensor,
    second: torch.Tensor,
    config: NvqLowBitAdditiveConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, padded_len = value.shape
    ng = padded_len // _GROUP_SIZE
    nvec_padded = padded_len // _VECTOR_SIZE
    full_index = torch.zeros((out, nvec_padded), dtype=torch.int64, device=value.device)
    source_index = torch.as_tensor(assignment.indices, dtype=torch.int64, device=value.device)
    full_index[:, : source_index.shape[1]] = source_index
    first_index = (full_index // config.second_entries).reshape(-1)
    second_index = (full_index % config.second_entries).reshape(-1)
    delta_sign = torch.as_tensor(
        assignment.delta_sign.reshape(-1), dtype=torch.int64, device=value.device
    )
    vector_bank = (
        delta_sign.repeat_interleave(_VECTORS_PER_GROUP)
        if config.banks == 2
        else torch.zeros(first_index.shape[0], dtype=torch.int64, device=value.device)
    )
    delta = torch.where(delta_sign != 0, -config.delta, config.delta).repeat_interleave(
        _VECTORS_PER_GROUP
    )
    anchor = torch.as_tensor(assignment.neuron_scale, dtype=torch.float32, device=value.device)
    q = torch.as_tensor(assignment.sub_scale.reshape(-1), dtype=torch.float32, device=value.device)
    scale = (anchor.repeat_interleave(ng) * q).repeat_interleave(_VECTORS_PER_GROUP)
    samples = value.reshape(-1, _VECTOR_SIZE)
    weights = objective_weight.reshape_as(samples)
    updated_first = first.clone()
    updated_second = second.clone()
    for bank_id in range(config.banks):
        selected = vector_bank == bank_id
        if not bool(selected.any()):
            continue
        step = float(config.codebook_step)
        second_code = second[bank_id][second_index[selected]].to(torch.float32)
        first_residual = samples[selected] - scale[selected, None] * (
            step * second_code + delta[selected, None]
        )
        updated_first[bank_id] = _update_one_codebook(
            first_residual,
            weights[selected],
            scale[selected] * step,
            first_index[selected],
            first[bank_id],
        )
        first_code = updated_first[bank_id][first_index[selected]].to(torch.float32)
        second_residual = samples[selected] - scale[selected, None] * (
            step * first_code + delta[selected, None]
        )
        updated_second[bank_id] = _update_one_codebook(
            second_residual,
            weights[selected],
            scale[selected] * step,
            second_index[selected],
            second[bank_id],
        )
    return updated_first, updated_second


def _candidate_from_assignment(
    assignment: NvqLowBitAssignment,
    first: torch.Tensor,
    second: torch.Tensor,
    config: NvqLowBitAdditiveConfig,
) -> NvqLowBitAdditiveTensor:
    first_dtype = (
        np.uint8 if config.first_entries <= 256 else np.uint16)
    second_dtype = (
        np.uint8 if config.second_entries <= 256 else np.uint16)
    effective = assignment.indices.astype(np.uint16)
    return NvqLowBitAdditiveTensor(
        shape=assignment.shape,
        neuron_scale=assignment.neuron_scale,
        sub_scale=assignment.sub_scale,
        delta_sign=assignment.delta_sign,
        first_indices=(effective // config.second_entries).astype(
            first_dtype),
        second_indices=(effective % config.second_entries).astype(
            second_dtype),
        codebook_step=np.full(config.banks, config.codebook_step, dtype=np.float32),
        first_codebooks=first.cpu().numpy().astype(np.int8, copy=False),
        second_codebooks=second.cpu().numpy().astype(np.int8, copy=False),
    )


@torch.inference_mode()
def train_lowbit_additive(
    weight: torch.Tensor,
    baseline_codebooks: np.ndarray,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    config: NvqLowBitAdditiveConfig,
    device: str | torch.device = "cuda",
) -> tuple[NvqLowBitAdditiveTensor, tuple[NvqLowBitAdditiveIteration, ...]]:
    baseline = _validate_effective_codebooks(baseline_codebooks, config)
    value, out, neuron_len = _prepare_weight(weight, device)
    value, objective_weight, _ng = _pad_weight(value, _GROUP_SIZE, importance)
    initial = quantize_effective_fixed(
        weight, baseline, importance=importance, config=config, device=device
    )
    first, second = _initial_additive_codebooks(value, objective_weight, initial, config)
    signal = float((value.square() * objective_weight).sum().item())
    history: list[NvqLowBitAdditiveIteration] = []
    best: tuple[float, NvqLowBitAssignment, torch.Tensor, torch.Tensor] | None = None

    for iteration in range(config.iterations + 1):
        tables = NvqLowBitAdditiveTables(
            np.full(config.banks, config.codebook_step, dtype=np.float32),
            first.cpu().numpy(),
            second.cpu().numpy(),
        )
        combined = combined_codebooks(tables, config)
        assignment = quantize_effective_fixed(
            weight, combined, importance=importance, config=config, device=device
        )
        reconstruction = dequantize_effective(assignment, combined, config)
        original = weight.detach().cpu().numpy().astype(np.float32, copy=False)
        if importance is None:
            objective = np.ones_like(original)
        else:
            objective = np.asarray(
                (
                    importance.detach().cpu().numpy()
                    if isinstance(importance, torch.Tensor)
                    else importance
                ),
                dtype=np.float32,
            )
            if objective.ndim == 1:
                objective = np.broadcast_to(objective, original.shape)
        sse = float(np.sum(objective * np.square(original - reconstruction), dtype=np.float64))
        first_used: list[int] = []
        second_used: list[int] = []
        group_bank = (
            assignment.delta_sign if config.banks == 2 else np.zeros_like(assignment.delta_sign)
        )
        vector_bank = np.repeat(group_bank, _VECTORS_PER_GROUP, axis=1)[
            :, : assignment.indices.shape[1]
        ]
        first_index = assignment.indices // config.second_entries
        second_index = assignment.indices % config.second_entries
        for bank_id in range(config.banks):
            selected = vector_bank == bank_id
            first_used.append(int(np.unique(first_index[selected]).size) if np.any(selected) else 0)
            second_used.append(
                int(np.unique(second_index[selected]).size) if np.any(selected) else 0
            )
        history.append(
            NvqLowBitAdditiveIteration(
                iteration=iteration,
                weighted_sse=sse,
                weighted_nmse_percent=100.0 * sse / signal if signal else 0.0,
                used_first_codes=tuple(first_used),
                used_second_codes=tuple(second_used),
            )
        )
        if best is None or sse < best[0]:
            best = (sse, assignment, first.clone(), second.clone())
        if iteration == config.iterations:
            break
        first, second = _update_codebooks(
            value, objective_weight, assignment, first, second, config
        )

    assert best is not None
    return _candidate_from_assignment(best[1], best[2], best[3], config), tuple(history)


@torch.inference_mode()
def quantize_lowbit_additive_fixed(
    weight: torch.Tensor,
    tables: NvqLowBitAdditiveTables,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    config: NvqLowBitAdditiveConfig,
    device: str | torch.device = "cuda",
) -> NvqLowBitAdditiveTensor:
    _step_np, first_np, second_np = _validate_additive_tables(tables, config)
    combined = combined_codebooks(tables, config)
    assignment = quantize_effective_fixed(
        weight, combined, importance=importance, config=config, device=device
    )
    first = torch.as_tensor(first_np)
    second = torch.as_tensor(second_np)
    return _candidate_from_assignment(assignment, first, second, config)


def dequantize_lowbit_additive(
    tensor: NvqLowBitAdditiveTensor,
    config: NvqLowBitAdditiveConfig,
) -> np.ndarray:
    tables = additive_tables_from_tensor(tensor)
    combined = combined_codebooks(tables, config)
    effective = tensor.first_indices.astype(np.uint16) * config.second_entries
    effective += tensor.second_indices.astype(np.uint16)
    assignment = NvqLowBitAssignment(
        shape=tensor.shape,
        neuron_scale=tensor.neuron_scale,
        sub_scale=tensor.sub_scale,
        delta_sign=tensor.delta_sign,
        indices=effective,
    )
    return dequantize_effective(assignment, combined, config)


def projected_nbytes(
    out: int,
    neuron_len: int,
    config: NvqLowBitAdditiveConfig,
    *,
    include_codebooks: bool,
) -> int:
    ng = math.ceil(neuron_len / _GROUP_SIZE)
    nvec = math.ceil(neuron_len / _VECTOR_SIZE)
    bits = 16 * out
    bits += (config.sub_bits + 1) * out * ng
    bits += config.index_bits * out * nvec
    payload = (bits + 7) // 8
    if include_codebooks:
        payload += config.banks * (config.first_entries + config.second_entries) * _VECTOR_SIZE
        payload += 2 * config.banks
    return payload


__all__ = [
    "NvqLowBitAdditiveConfig",
    "NvqLowBitAdditiveIteration",
    "NvqLowBitAdditiveTables",
    "NvqLowBitAdditiveTensor",
    "additive_tables_from_tensor",
    "combined_codebooks",
    "dequantize_effective",
    "dequantize_lowbit_additive",
    "projected_nbytes",
    "quantize_effective_fixed",
    "quantize_lowbit_additive_fixed",
    "train_lowbit_additive",
]
