"""Offline signed-product-codebook experiment for neuron-anchored NVQ2.

The candidate keeps the NVQ2-JSC neuron anchor and four-bit group state, but
replaces each 8-D magnitude index plus seven sign bits with two signed 4-D
indices.  The first subspace has 256 entries and the second has 128 entries,
so both layouts spend exactly 15 index bits per eight weights.

This module intentionally does not define a serialized format or runtime
kernel.  It is a numeric experiment used to decide whether that engineering
work is justified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from mfq.formats.nvq import NvqJscTensor
from mfq.quantize.nvq_quant_torch import _fp16_round, _pad_weight, _prepare_weight

_GROUP_SIZE = 24
_VECTOR_SIZE = 8
_SUBVECTOR_SIZE = 4
_VECTORS_PER_GROUP = _GROUP_SIZE // _VECTOR_SIZE
_STATE_COUNT = 16
_FIRST_ENTRIES = 256
_SECOND_ENTRIES = 128


@dataclass(frozen=True)
class NvqProductConfig:
    banks: int = 4
    iterations: int = 3
    assignment_refine_steps: int = 2
    fixed_refine_steps: int = 3
    kmeans_iterations: int = 6
    kmeans_initialization_points: int = 8192
    group_chunk: int = 256
    anchor_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    seed: int = 20260721

    def __post_init__(self) -> None:
        if self.banks not in {1, 2, 4}:
            raise ValueError("NVQ-SPQ banks must be 1, 2, or 4")
        if self.iterations < 0 or self.assignment_refine_steps < 0:
            raise ValueError("NVQ-SPQ iteration counts must be non-negative")
        if self.fixed_refine_steps < 0 or self.kmeans_iterations < 0:
            raise ValueError("NVQ-SPQ refinement counts must be non-negative")
        if self.kmeans_initialization_points <= 0 or self.group_chunk <= 0:
            raise ValueError("NVQ-SPQ chunk sizes must be positive")
        if not self.anchor_multipliers or any(
            not math.isfinite(value) or value <= 0 for value in self.anchor_multipliers
        ):
            raise ValueError("NVQ-SPQ anchor multipliers must be finite and positive")


@dataclass(frozen=True)
class NvqProductTables:
    scale_lut: np.ndarray
    bank_for_state: np.ndarray
    first_codebooks: np.ndarray
    second_codebooks: np.ndarray


@dataclass
class NvqProductTensor:
    shape: tuple[int, int]
    neuron_scale: np.ndarray
    scale_lut: np.ndarray
    bank_for_state: np.ndarray
    state: np.ndarray
    first_indices: np.ndarray
    second_indices: np.ndarray
    first_codebooks: np.ndarray
    second_codebooks: np.ndarray

    @property
    def neuron_len(self) -> int:
        return self.shape[1]

    @property
    def payload_nbytes(self) -> int:
        out, neuron_len = self.shape
        ng = math.ceil(neuron_len / _GROUP_SIZE)
        nvec = math.ceil(neuron_len / _VECTOR_SIZE)
        streams = 2 * out
        streams += (out * ng * 4 + 7) // 8
        streams += out * nvec
        streams += (out * nvec * 7 + 7) // 8
        tables = 16 * 2 + 16
        tables += int(self.first_codebooks.size + self.second_codebooks.size)
        return streams + tables

    @property
    def payload_bpw(self) -> float:
        return 8.0 * self.payload_nbytes / math.prod(self.shape)


@dataclass(frozen=True)
class NvqProductIteration:
    iteration: int
    weighted_sse: float
    weighted_nmse_percent: float
    used_states: int
    used_banks: int
    used_first_codes: tuple[int, ...]
    used_second_codes: tuple[int, ...]


def _validate_tables(
    tables: NvqProductTables,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.asarray(tables.scale_lut, dtype=np.float32).reshape(-1)
    bank = np.asarray(tables.bank_for_state).reshape(-1)
    first = np.asarray(tables.first_codebooks)
    second = np.asarray(tables.second_codebooks)
    if alpha.shape != (_STATE_COUNT,) or not np.isfinite(alpha).all() or np.any(alpha < 0):
        raise ValueError("NVQ-SPQ scale_lut must contain 16 finite non-negative values")
    if float(alpha.max()) <= 0:
        raise ValueError("NVQ-SPQ scale_lut must contain a positive value")
    if bank.shape != (_STATE_COUNT,) or not np.issubdtype(bank.dtype, np.integer):
        raise ValueError("NVQ-SPQ bank_for_state must contain 16 integers")
    if first.ndim != 3 or first.shape[1:] != (_FIRST_ENTRIES, _SUBVECTOR_SIZE):
        raise ValueError("NVQ-SPQ first codebooks must have shape [banks,256,4]")
    if second.shape != (first.shape[0], _SECOND_ENTRIES, _SUBVECTOR_SIZE):
        raise ValueError("NVQ-SPQ second codebooks must have shape [banks,128,4]")
    if first.shape[0] not in {1, 2, 4}:
        raise ValueError("NVQ-SPQ must have 1, 2, or 4 banks")
    if np.any(bank < 0) or np.any(bank >= first.shape[0]):
        raise ValueError("NVQ-SPQ state references a missing bank")
    for name, value in (("first", first), ("second", second)):
        rounded = np.rint(value)
        if (
            not np.isfinite(value).all()
            or not np.array_equal(value, rounded)
            or np.any(rounded < -127)
            or np.any(rounded > 127)
        ):
            raise ValueError(f"NVQ-SPQ {name} codebooks must be int8-valued")
    return (
        np.ascontiguousarray(alpha),
        np.ascontiguousarray(bank, dtype=np.uint8),
        np.ascontiguousarray(first, dtype=np.int8),
        np.ascontiguousarray(second, dtype=np.int8),
    )


def product_tables_from_tensor(tensor: NvqProductTensor) -> NvqProductTables:
    return NvqProductTables(
        scale_lut=np.asarray(tensor.scale_lut, dtype=np.float32).copy(),
        bank_for_state=np.asarray(tensor.bank_for_state, dtype=np.uint8).copy(),
        first_codebooks=np.asarray(tensor.first_codebooks, dtype=np.int8).copy(),
        second_codebooks=np.asarray(tensor.second_codebooks, dtype=np.int8).copy(),
    )


def _nearest(
    samples: torch.Tensor,
    objective_weight: torch.Tensor,
    scale: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    table = codebook.to(torch.float32)
    cross = (objective_weight * samples) @ table.T
    quadratic = objective_weight @ table.square().T
    scaled = scale.unsqueeze(1)
    variable = scaled.square() * quadratic - 2.0 * scaled * cross
    minimum, index = variable.min(dim=1)
    constant = (objective_weight * samples.square()).sum(dim=1)
    return index, constant + minimum


def _assign_states(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    neuron_scale: torch.Tensor,
    alpha: torch.Tensor,
    bank_for_state: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
    *,
    ng: int,
    group_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = xgroup.shape[0]
    group_anchor = neuron_scale.repeat_interleave(ng)
    best_error = torch.full((groups,), torch.inf, device=xgroup.device)
    best_state = torch.zeros(groups, dtype=torch.int64, device=xgroup.device)
    best_first = torch.zeros((groups, _VECTORS_PER_GROUP), dtype=torch.int64, device=xgroup.device)
    best_second = torch.zeros_like(best_first)

    for start in range(0, groups, group_chunk):
        stop = min(start + group_chunk, groups)
        count = stop - start
        vectors = xgroup[start:stop].reshape(count, _VECTORS_PER_GROUP, 2, 4)
        weights = wgroup[start:stop].reshape_as(vectors)
        first_x = vectors[:, :, 0].reshape(-1, 4)
        second_x = vectors[:, :, 1].reshape(-1, 4)
        first_w = weights[:, :, 0].reshape_as(first_x)
        second_w = weights[:, :, 1].reshape_as(second_x)
        local_error = best_error[start:stop]
        local_state = best_state[start:stop]
        local_first = best_first[start:stop]
        local_second = best_second[start:stop]

        for state_id in range(_STATE_COUNT):
            bank_id = int(bank_for_state[state_id].item())
            group_scale = group_anchor[start:stop] * alpha[state_id]
            vector_scale = group_scale.repeat_interleave(_VECTORS_PER_GROUP)
            first_index, first_error = _nearest(
                first_x, first_w, vector_scale, first_codebooks[bank_id]
            )
            second_index, second_error = _nearest(
                second_x, second_w, vector_scale, second_codebooks[bank_id]
            )
            error = (first_error + second_error).reshape(count, -1).sum(1)
            better = error < local_error
            local_error = torch.where(better, error, local_error)
            local_state = torch.where(better, torch.full_like(local_state, state_id), local_state)
            local_first[better] = first_index.reshape(count, -1)[better]
            local_second[better] = second_index.reshape(count, -1)[better]

        best_error[start:stop] = local_error
        best_state[start:stop] = local_state
        best_first[start:stop] = local_first
        best_second[start:stop] = local_second
    return best_state, best_first, best_second, best_error


def _codes_for_assignment(
    state: torch.Tensor,
    bank_for_state: torch.Tensor,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
) -> torch.Tensor:
    groups = state.numel()
    result = torch.empty((groups, _GROUP_SIZE), dtype=torch.float32, device=state.device)
    group_bank = bank_for_state[state]
    for bank_id in range(first_codebooks.shape[0]):
        selected = group_bank == bank_id
        if not bool(selected.any()):
            continue
        first = first_codebooks[bank_id][first_indices[selected]].to(torch.float32)
        second = second_codebooks[bank_id][second_indices[selected]].to(torch.float32)
        result[selected] = torch.cat((first, second), dim=-1).reshape(-1, _GROUP_SIZE)
    return result


def _refit_anchor_and_alpha(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    state: torch.Tensor,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
    neuron_scale: torch.Tensor,
    alpha: torch.Tensor,
    bank_for_state: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
    *,
    out: int,
    ng: int,
    learn_alpha: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    code = _codes_for_assignment(
        state,
        bank_for_state,
        first_indices,
        second_indices,
        first_codebooks,
        second_codebooks,
    )
    basis = alpha[state].unsqueeze(1) * code
    numerator = (wgroup * xgroup * basis).reshape(out, ng, -1).sum((1, 2))
    denominator = (wgroup * basis.square()).reshape(out, ng, -1).sum((1, 2))
    fitted_anchor = torch.where(denominator > 0, numerator / denominator, neuron_scale).clamp_min(0)
    fitted_anchor = _fp16_round(fitted_anchor)
    if not learn_alpha:
        return fitted_anchor, alpha

    group_anchor = fitted_anchor.repeat_interleave(ng)
    alpha_num = torch.zeros_like(alpha)
    alpha_den = torch.zeros_like(alpha)
    alpha_num.scatter_add_(0, state, (wgroup * xgroup * code).sum(1) * group_anchor)
    alpha_den.scatter_add_(0, state, (wgroup * code.square()).sum(1) * group_anchor.square())
    fitted_alpha = torch.where(alpha_den > 0, alpha_num / alpha_den, alpha).clamp_min(0)
    maximum = fitted_alpha.max()
    if float(maximum.item()) > 0:
        factor = maximum / 15.0
        fitted_alpha = fitted_alpha / factor
        fitted_anchor = _fp16_round(fitted_anchor * factor)
    return fitted_anchor, _fp16_round(fitted_alpha)


def _update_one_codebook(
    samples: torch.Tensor,
    objective_weight: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    entries = previous.shape[0]
    numerator = torch.zeros_like(previous, dtype=torch.float32)
    denominator = torch.zeros_like(previous, dtype=torch.float32)
    expanded = indices.unsqueeze(1).expand(-1, _SUBVECTOR_SIZE)
    numerator.scatter_add_(
        0,
        expanded,
        objective_weight * scale.unsqueeze(1) * samples,
    )
    denominator.scatter_add_(
        0,
        expanded,
        objective_weight * scale.square().unsqueeze(1),
    )
    centroid = torch.where(denominator > 0, numerator / denominator, previous.to(torch.float32))
    return centroid.round().clamp(-127, 127).to(torch.int8).reshape(entries, 4)


def _update_codebooks(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    state: torch.Tensor,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
    neuron_scale: torch.Tensor,
    alpha: torch.Tensor,
    bank_for_state: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
    *,
    ng: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = xgroup.reshape(-1, _VECTORS_PER_GROUP, 2, 4)
    weights = wgroup.reshape_as(vectors)
    group_bank = bank_for_state[state]
    group_scale = neuron_scale.repeat_interleave(ng) * alpha[state]
    updated_first = first_codebooks.clone()
    updated_second = second_codebooks.clone()
    for bank_id in range(first_codebooks.shape[0]):
        selected = group_bank == bank_id
        if not bool(selected.any()):
            continue
        scale = group_scale[selected].repeat_interleave(_VECTORS_PER_GROUP)
        updated_first[bank_id] = _update_one_codebook(
            vectors[selected, :, 0].reshape(-1, 4),
            weights[selected, :, 0].reshape(-1, 4),
            scale,
            first_indices[selected].reshape(-1),
            first_codebooks[bank_id],
        )
        updated_second[bank_id] = _update_one_codebook(
            vectors[selected, :, 1].reshape(-1, 4),
            weights[selected, :, 1].reshape(-1, 4),
            scale,
            second_indices[selected].reshape(-1),
            second_codebooks[bank_id],
        )
    return updated_first.contiguous(), updated_second.contiguous()


def _weighted_kmeans(
    samples: torch.Tensor,
    objective_weight: torch.Tensor,
    entries: int,
    *,
    iterations: int,
    initialization_points: int,
    seed: int,
) -> torch.Tensor:
    dimensions = samples.shape[1]
    valid = objective_weight.sum(1) > 0
    samples = samples[valid]
    objective_weight = objective_weight[valid]
    if not samples.shape[0]:
        return torch.zeros((entries, dimensions), dtype=torch.int8, device=samples.device)

    generator = torch.Generator(device=samples.device)
    generator.manual_seed(seed)
    if samples.shape[0] > initialization_points:
        chosen = torch.randperm(samples.shape[0], generator=generator, device=samples.device)[
            :initialization_points
        ]
        init_x = samples[chosen]
        init_w = objective_weight[chosen]
    else:
        init_x = samples
        init_w = objective_weight

    importance = init_w.sum(1)
    first = int(torch.argmax(importance * init_x.square().sum(1)).item())
    centers = [init_x[first]]
    minimum = (init_w * (init_x - centers[0]).square()).sum(1)
    for _ in range(1, min(entries, init_x.shape[0])):
        probabilities = minimum.clamp_min(0)
        total = probabilities.sum()
        if float(total.item()) <= 0:
            next_index = len(centers) % init_x.shape[0]
        else:
            next_index = int(
                torch.multinomial(probabilities / total, 1, generator=generator).item()
            )
        center = init_x[next_index]
        centers.append(center)
        distance = (init_w * (init_x - center).square()).sum(1)
        minimum = torch.minimum(minimum, distance)
    table = torch.stack(centers)
    if table.shape[0] < entries:
        repeat = torch.arange(entries - table.shape[0], device=samples.device)
        table = torch.cat((table, table[repeat % table.shape[0]]), dim=0)
    table = table.round().clamp(-127, 127).to(torch.int8)

    for _ in range(iterations):
        updated_num = torch.zeros((entries, dimensions), device=samples.device)
        updated_den = torch.zeros_like(updated_num)
        for start in range(0, samples.shape[0], 4096):
            stop = min(start + 4096, samples.shape[0])
            index, _ = _nearest(
                samples[start:stop],
                objective_weight[start:stop],
                torch.ones(stop - start, device=samples.device),
                table,
            )
            expanded = index.unsqueeze(1).expand(-1, dimensions)
            updated_num.scatter_add_(
                0, expanded, objective_weight[start:stop] * samples[start:stop]
            )
            updated_den.scatter_add_(0, expanded, objective_weight[start:stop])
        candidate = (
            torch.where(updated_den > 0, updated_num / updated_den, table.to(torch.float32))
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
        )
        if torch.equal(candidate, table):
            break
        table = candidate
    return table.contiguous()


def _initial_product_codebooks(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    initial_jsc: NvqJscTensor,
    config: NvqProductConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.as_tensor(initial_jsc.state.reshape(-1), device=xgroup.device, dtype=torch.int64)
    alpha = torch.as_tensor(initial_jsc.scale_lut, device=xgroup.device, dtype=torch.float32)
    bank_for_state = torch.as_tensor(
        initial_jsc.bank_for_state, device=xgroup.device, dtype=torch.int64
    )
    anchor = torch.as_tensor(initial_jsc.neuron_scale, device=xgroup.device, dtype=torch.float32)
    ng = initial_jsc.state.shape[1]
    scale = anchor.repeat_interleave(ng) * alpha[state]
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = xgroup / safe_scale.unsqueeze(1)
    vectors = normalized.reshape(-1, _VECTORS_PER_GROUP, 2, 4)
    weights = wgroup.reshape_as(vectors)
    group_bank = bank_for_state[state]
    first = torch.empty((config.banks, _FIRST_ENTRIES, 4), dtype=torch.int8, device=xgroup.device)
    second = torch.empty((config.banks, _SECOND_ENTRIES, 4), dtype=torch.int8, device=xgroup.device)
    all_groups = torch.ones_like(group_bank, dtype=torch.bool)
    for bank_id in range(config.banks):
        selected = group_bank == bank_id
        if not bool(selected.any()):
            selected = all_groups
        first[bank_id] = _weighted_kmeans(
            vectors[selected, :, 0].reshape(-1, 4),
            weights[selected, :, 0].reshape(-1, 4),
            _FIRST_ENTRIES,
            iterations=config.kmeans_iterations,
            initialization_points=config.kmeans_initialization_points,
            seed=config.seed + 2 * bank_id,
        )
        second[bank_id] = _weighted_kmeans(
            vectors[selected, :, 1].reshape(-1, 4),
            weights[selected, :, 1].reshape(-1, 4),
            _SECOND_ENTRIES,
            iterations=config.kmeans_iterations,
            initialization_points=config.kmeans_initialization_points,
            seed=config.seed + 2 * bank_id + 1,
        )
    return first, second


def _tensor_from_assignment(
    shape: tuple[int, int],
    neuron_scale: torch.Tensor,
    alpha: torch.Tensor,
    bank_for_state: torch.Tensor,
    state: torch.Tensor,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
) -> NvqProductTensor:
    out, neuron_len = shape
    ng = math.ceil(neuron_len / _GROUP_SIZE)
    nvec = math.ceil(neuron_len / _VECTOR_SIZE)
    return NvqProductTensor(
        shape=shape,
        neuron_scale=neuron_scale.cpu().numpy().astype(np.float32, copy=False),
        scale_lut=alpha.cpu().numpy().astype(np.float32, copy=False),
        bank_for_state=bank_for_state.cpu().numpy().astype(np.uint8, copy=False),
        state=state.reshape(out, ng).cpu().numpy().astype(np.uint8, copy=False),
        first_indices=(
            first_indices.reshape(out, -1)[:, :nvec].cpu().numpy().astype(np.uint8, copy=False)
        ),
        second_indices=(
            second_indices.reshape(out, -1)[:, :nvec].cpu().numpy().astype(np.uint8, copy=False)
        ),
        first_codebooks=first_codebooks.cpu().numpy().astype(np.int8, copy=False),
        second_codebooks=second_codebooks.cpu().numpy().astype(np.int8, copy=False),
    )


@torch.inference_mode()
def train_nvq_product(
    weight: torch.Tensor,
    initial_jsc: NvqJscTensor,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    config: NvqProductConfig | None = None,
    device: str | torch.device = "cuda",
) -> tuple[NvqProductTensor, tuple[NvqProductIteration, ...]]:
    """Train signed product codebooks from a same-data NVQ2-JSC initialization."""

    config = NvqProductConfig() if config is None else config
    value, out, neuron_len = _prepare_weight(weight, device)
    if initial_jsc.shape != (out, neuron_len):
        raise ValueError("NVQ-SPQ initialization shape does not match training weight")
    if initial_jsc.codebooks.shape[0] != config.banks:
        raise ValueError("NVQ-SPQ bank count does not match JSC initialization")
    value, objective_weight, ng = _pad_weight(value, _GROUP_SIZE, importance)
    xgroup = value.reshape(out * ng, _GROUP_SIZE).contiguous()
    wgroup = objective_weight.reshape_as(xgroup).contiguous()
    alpha = torch.as_tensor(
        initial_jsc.scale_lut, device=value.device, dtype=torch.float32
    ).contiguous()
    bank_for_state = torch.as_tensor(
        initial_jsc.bank_for_state, device=value.device, dtype=torch.int64
    ).contiguous()
    neuron_scale = torch.as_tensor(
        initial_jsc.neuron_scale, device=value.device, dtype=torch.float32
    ).contiguous()
    first_codebooks, second_codebooks = _initial_product_codebooks(
        xgroup, wgroup, initial_jsc, config
    )

    signal = float((wgroup * xgroup.square()).sum().item())
    history: list[NvqProductIteration] = []
    best: (
        tuple[
            float,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    ) = None
    for iteration in range(config.iterations + 1):
        state, first_indices, second_indices, error = _assign_states(
            xgroup,
            wgroup,
            neuron_scale,
            alpha,
            bank_for_state,
            first_codebooks,
            second_codebooks,
            ng=ng,
            group_chunk=config.group_chunk,
        )
        for _ in range(config.assignment_refine_steps):
            neuron_scale, alpha = _refit_anchor_and_alpha(
                xgroup,
                wgroup,
                state,
                first_indices,
                second_indices,
                neuron_scale,
                alpha,
                bank_for_state,
                first_codebooks,
                second_codebooks,
                out=out,
                ng=ng,
                learn_alpha=True,
            )
            state, first_indices, second_indices, error = _assign_states(
                xgroup,
                wgroup,
                neuron_scale,
                alpha,
                bank_for_state,
                first_codebooks,
                second_codebooks,
                ng=ng,
                group_chunk=config.group_chunk,
            )
        total_error = float(error.sum().item())
        group_bank = bank_for_state[state]
        history.append(
            NvqProductIteration(
                iteration=iteration,
                weighted_sse=total_error,
                weighted_nmse_percent=100.0 * total_error / signal if signal else 0.0,
                used_states=int(torch.unique(state).numel()),
                used_banks=int(torch.unique(group_bank).numel()),
                used_first_codes=tuple(
                    (
                        int(torch.unique(first_indices[group_bank == bank]).numel())
                        if bool((group_bank == bank).any())
                        else 0
                    )
                    for bank in range(config.banks)
                ),
                used_second_codes=tuple(
                    (
                        int(torch.unique(second_indices[group_bank == bank]).numel())
                        if bool((group_bank == bank).any())
                        else 0
                    )
                    for bank in range(config.banks)
                ),
            )
        )
        if best is None or total_error < best[0]:
            best = (
                total_error,
                neuron_scale.clone(),
                alpha.clone(),
                state.clone(),
                first_indices.clone(),
                second_indices.clone(),
                first_codebooks.clone(),
                second_codebooks.clone(),
            )
        if iteration == config.iterations:
            break
        first_codebooks, second_codebooks = _update_codebooks(
            xgroup,
            wgroup,
            state,
            first_indices,
            second_indices,
            neuron_scale,
            alpha,
            bank_for_state,
            first_codebooks,
            second_codebooks,
            ng=ng,
        )

    assert best is not None
    (
        _best_error,
        neuron_scale,
        alpha,
        state,
        first_indices,
        second_indices,
        first_codebooks,
        second_codebooks,
    ) = best
    return (
        _tensor_from_assignment(
            (out, neuron_len),
            neuron_scale,
            alpha,
            bank_for_state,
            state,
            first_indices,
            second_indices,
            first_codebooks,
            second_codebooks,
        ),
        tuple(history),
    )


@torch.inference_mode()
def quantize_nvq_product_fixed(
    weight: torch.Tensor,
    tables: NvqProductTables,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    config: NvqProductConfig | None = None,
    device: str | torch.device = "cuda",
) -> NvqProductTensor:
    """Assign weights to fixed signed-product tables and fit row anchors."""

    config = NvqProductConfig() if config is None else config
    alpha_np, bank_np, first_np, second_np = _validate_tables(tables)
    if first_np.shape[0] != config.banks:
        raise ValueError("NVQ-SPQ fixed table bank count does not match config")
    value, out, neuron_len = _prepare_weight(weight, device)
    value, objective_weight, ng = _pad_weight(value, _GROUP_SIZE, importance)
    xgroup = value.reshape(out * ng, _GROUP_SIZE).contiguous()
    wgroup = objective_weight.reshape_as(xgroup).contiguous()
    alpha = torch.as_tensor(alpha_np, device=value.device, dtype=torch.float32)
    bank_for_state = torch.as_tensor(bank_np, device=value.device, dtype=torch.int64)
    first_codebooks = torch.as_tensor(first_np, device=value.device, dtype=torch.int8)
    second_codebooks = torch.as_tensor(second_np, device=value.device, dtype=torch.int8)
    maximum = max(int(np.abs(first_np).max()), int(np.abs(second_np).max()), 1)
    base_anchor = value.abs().amax(1) / (float(alpha.max().item()) * maximum)

    best_row_error = torch.full((out,), torch.inf, device=value.device)
    best_anchor = torch.zeros(out, device=value.device)
    best_state = torch.zeros(out * ng, dtype=torch.int64, device=value.device)
    best_first = torch.zeros((out * ng, _VECTORS_PER_GROUP), dtype=torch.int64, device=value.device)
    best_second = torch.zeros_like(best_first)
    for multiplier in config.anchor_multipliers:
        anchor = _fp16_round(base_anchor * multiplier)
        state, first_indices, second_indices, error = _assign_states(
            xgroup,
            wgroup,
            anchor,
            alpha,
            bank_for_state,
            first_codebooks,
            second_codebooks,
            ng=ng,
            group_chunk=config.group_chunk,
        )
        for _ in range(config.fixed_refine_steps):
            anchor, _ = _refit_anchor_and_alpha(
                xgroup,
                wgroup,
                state,
                first_indices,
                second_indices,
                anchor,
                alpha,
                bank_for_state,
                first_codebooks,
                second_codebooks,
                out=out,
                ng=ng,
                learn_alpha=False,
            )
            state, first_indices, second_indices, error = _assign_states(
                xgroup,
                wgroup,
                anchor,
                alpha,
                bank_for_state,
                first_codebooks,
                second_codebooks,
                ng=ng,
                group_chunk=config.group_chunk,
            )
        row_error = error.reshape(out, ng).sum(1)
        better = row_error < best_row_error
        best_row_error = torch.where(better, row_error, best_row_error)
        best_anchor = torch.where(better, anchor, best_anchor)
        group_better = better.repeat_interleave(ng)
        best_state[group_better] = state[group_better]
        best_first[group_better] = first_indices[group_better]
        best_second[group_better] = second_indices[group_better]

    return _tensor_from_assignment(
        (out, neuron_len),
        best_anchor,
        alpha,
        bank_for_state,
        best_state,
        best_first,
        best_second,
        first_codebooks,
        second_codebooks,
    )


def dequantize_nvq_product(tensor: NvqProductTensor) -> np.ndarray:
    out, neuron_len = tensor.shape
    ng = math.ceil(neuron_len / _GROUP_SIZE)
    nvec_padded = ng * _VECTORS_PER_GROUP
    first_indices = np.zeros((out, nvec_padded), dtype=np.uint8)
    second_indices = np.zeros_like(first_indices)
    first_indices[:, : tensor.first_indices.shape[1]] = tensor.first_indices
    second_indices[:, : tensor.second_indices.shape[1]] = tensor.second_indices
    group_bank = tensor.bank_for_state[tensor.state]
    vector_bank = np.repeat(group_bank, _VECTORS_PER_GROUP, axis=1)
    first = tensor.first_codebooks[vector_bank, first_indices]
    second = tensor.second_codebooks[vector_bank, second_indices]
    code = np.concatenate((first, second), axis=-1).reshape(out, -1)[:, :neuron_len]
    group_scale = tensor.neuron_scale[:, None] * tensor.scale_lut[tensor.state]
    scale = np.repeat(group_scale, _GROUP_SIZE, axis=1)[:, :neuron_len]
    return code.astype(np.float32) * scale


__all__ = [
    "NvqProductConfig",
    "NvqProductIteration",
    "NvqProductTables",
    "NvqProductTensor",
    "dequantize_nvq_product",
    "product_tables_from_tensor",
    "quantize_nvq_product_fixed",
    "train_nvq_product",
]
