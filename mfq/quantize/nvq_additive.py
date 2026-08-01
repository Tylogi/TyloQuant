"""Offline additive-codebook experiment for neuron-anchored NVQ2.

Each 8-D vector is reconstructed from the sum of one 256-entry signed int8
codeword and one 128-entry signed int8 codeword.  Their 8+7 bit indices replace
the NVQ2 magnitude index and sign stream without changing the 15-bit vector
payload.  Serialization and runtime kernels are intentionally out of scope
until the numeric experiment justifies them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from mfq.formats.nvq import NvqJscTensor
from mfq.quantize.nvq_product import _nearest, _weighted_kmeans
from mfq.quantize.nvq_quant_torch import _fp16_round, _pad_weight, _prepare_weight

_GROUP_SIZE = 24
_VECTOR_SIZE = 8
_VECTORS_PER_GROUP = 3
_STATE_COUNT = 16
_FIRST_ENTRIES = 256
_SECOND_ENTRIES = 128


@dataclass(frozen=True)
class NvqAdditiveConfig:
    banks: int = 4
    iterations: int = 3
    assignment_refine_steps: int = 2
    fixed_refine_steps: int = 3
    kmeans_iterations: int = 6
    kmeans_initialization_points: int = 8192
    beam_size: int = 8
    pair_refine_steps: int = 2
    group_chunk: int = 128
    anchor_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    seed: int = 20260721

    def __post_init__(self) -> None:
        if self.banks not in {1, 2, 4}:
            raise ValueError("NVQ-AQ banks must be 1, 2, or 4")
        counts = (
            self.iterations,
            self.assignment_refine_steps,
            self.fixed_refine_steps,
            self.kmeans_iterations,
            self.pair_refine_steps,
        )
        if any(value < 0 for value in counts):
            raise ValueError("NVQ-AQ iteration counts must be non-negative")
        if self.kmeans_initialization_points <= 0 or self.group_chunk <= 0:
            raise ValueError("NVQ-AQ chunk sizes must be positive")
        if not 1 <= self.beam_size <= _SECOND_ENTRIES:
            raise ValueError("NVQ-AQ beam_size must be in [1, 128]")
        if not self.anchor_multipliers or any(
            not math.isfinite(value) or value <= 0 for value in self.anchor_multipliers
        ):
            raise ValueError("NVQ-AQ anchor multipliers must be finite and positive")


@dataclass(frozen=True)
class NvqAdditiveTables:
    scale_lut: np.ndarray
    bank_for_state: np.ndarray
    first_codebooks: np.ndarray
    second_codebooks: np.ndarray


@dataclass
class NvqAdditiveTensor:
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
class NvqAdditiveIteration:
    iteration: int
    weighted_sse: float
    weighted_nmse_percent: float
    used_states: int
    used_banks: int
    used_first_codes: tuple[int, ...]
    used_second_codes: tuple[int, ...]


def _validate_tables(
    tables: NvqAdditiveTables,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.asarray(tables.scale_lut, dtype=np.float32).reshape(-1)
    bank = np.asarray(tables.bank_for_state).reshape(-1)
    first = np.asarray(tables.first_codebooks)
    second = np.asarray(tables.second_codebooks)
    if alpha.shape != (_STATE_COUNT,) or not np.isfinite(alpha).all() or np.any(alpha < 0):
        raise ValueError("NVQ-AQ scale_lut must contain 16 finite non-negative values")
    if float(alpha.max()) <= 0:
        raise ValueError("NVQ-AQ scale_lut must contain a positive value")
    if bank.shape != (_STATE_COUNT,) or not np.issubdtype(bank.dtype, np.integer):
        raise ValueError("NVQ-AQ bank_for_state must contain 16 integers")
    if first.ndim != 3 or first.shape[1:] != (_FIRST_ENTRIES, _VECTOR_SIZE):
        raise ValueError("NVQ-AQ first codebooks must have shape [banks,256,8]")
    if second.shape != (first.shape[0], _SECOND_ENTRIES, _VECTOR_SIZE):
        raise ValueError("NVQ-AQ second codebooks must have shape [banks,128,8]")
    if first.shape[0] not in {1, 2, 4}:
        raise ValueError("NVQ-AQ must have 1, 2, or 4 banks")
    if np.any(bank < 0) or np.any(bank >= first.shape[0]):
        raise ValueError("NVQ-AQ state references a missing bank")
    for name, value in (("first", first), ("second", second)):
        rounded = np.rint(value)
        if (
            not np.isfinite(value).all()
            or not np.array_equal(value, rounded)
            or np.any(rounded < -127)
            or np.any(rounded > 127)
        ):
            raise ValueError(f"NVQ-AQ {name} codebooks must be int8-valued")
    return (
        np.ascontiguousarray(alpha),
        np.ascontiguousarray(bank, dtype=np.uint8),
        np.ascontiguousarray(first, dtype=np.int8),
        np.ascontiguousarray(second, dtype=np.int8),
    )


def additive_tables_from_tensor(tensor: NvqAdditiveTensor) -> NvqAdditiveTables:
    return NvqAdditiveTables(
        scale_lut=np.asarray(tensor.scale_lut, dtype=np.float32).copy(),
        bank_for_state=np.asarray(tensor.bank_for_state, dtype=np.uint8).copy(),
        first_codebooks=np.asarray(tensor.first_codebooks, dtype=np.int8).copy(),
        second_codebooks=np.asarray(tensor.second_codebooks, dtype=np.int8).copy(),
    )


def _top_indices(
    samples: torch.Tensor,
    objective_weight: torch.Tensor,
    scale: torch.Tensor,
    codebook: torch.Tensor,
    beam_size: int,
) -> torch.Tensor:
    table = codebook.to(torch.float32)
    cross = (objective_weight * samples) @ table.T
    quadratic = objective_weight @ table.square().T
    scaled = scale.unsqueeze(1)
    variable = scaled.square() * quadratic - 2.0 * scaled * cross
    return torch.topk(variable, k=min(beam_size, table.shape[0]), dim=1, largest=False).indices


def _candidate_from_base(
    samples: torch.Tensor,
    objective_weight: torch.Tensor,
    scale: torch.Tensor,
    first_codebook: torch.Tensor,
    second_codebook: torch.Tensor,
    *,
    first_is_base: bool,
    beam_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = first_codebook if first_is_base else second_codebook
    residual_table = second_codebook if first_is_base else first_codebook
    base_indices = _top_indices(samples, objective_weight, scale, base, beam_size)
    beams = base_indices.shape[1]
    base_codes = base[base_indices].to(torch.float32)
    residual = samples[:, None, :] - scale[:, None, None] * base_codes
    flat_residual = residual.reshape(-1, _VECTOR_SIZE)
    flat_weight = objective_weight[:, None, :].expand(-1, beams, -1).reshape(-1, _VECTOR_SIZE)
    flat_scale = scale.repeat_interleave(beams)
    residual_indices, _ = _nearest(
        flat_residual,
        flat_weight,
        flat_scale,
        residual_table,
    )
    residual_indices = residual_indices.reshape(-1, beams)
    residual_codes = residual_table[residual_indices].to(torch.float32)
    summed = base_codes + residual_codes
    error = (
        objective_weight[:, None, :]
        * (scale[:, None, None] * summed - samples[:, None, :]).square()
    ).sum(2)
    choice = error.argmin(1)
    row = torch.arange(samples.shape[0], device=samples.device)
    selected_base = base_indices[row, choice]
    selected_residual = residual_indices[row, choice]
    selected_error = error[row, choice]
    if first_is_base:
        return selected_base, selected_residual, selected_error
    return selected_residual, selected_base, selected_error


def _additive_nearest(
    samples: torch.Tensor,
    objective_weight: torch.Tensor,
    scale: torch.Tensor,
    first_codebook: torch.Tensor,
    second_codebook: torch.Tensor,
    *,
    beam_size: int,
    pair_refine_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first_a, second_a, error_a = _candidate_from_base(
        samples,
        objective_weight,
        scale,
        first_codebook,
        second_codebook,
        first_is_base=True,
        beam_size=beam_size,
    )
    first_b, second_b, error_b = _candidate_from_base(
        samples,
        objective_weight,
        scale,
        first_codebook,
        second_codebook,
        first_is_base=False,
        beam_size=beam_size,
    )
    use_b = error_b < error_a
    first = torch.where(use_b, first_b, first_a)
    second = torch.where(use_b, second_b, second_a)
    error = torch.where(use_b, error_b, error_a)

    for _ in range(pair_refine_steps):
        residual = samples - scale.unsqueeze(1) * second_codebook[second].to(torch.float32)
        first_new, _ = _nearest(residual, objective_weight, scale, first_codebook)
        residual = samples - scale.unsqueeze(1) * first_codebook[first_new].to(torch.float32)
        second_new, _ = _nearest(residual, objective_weight, scale, second_codebook)
        reconstruction = scale.unsqueeze(1) * (
            first_codebook[first_new].to(torch.float32)
            + second_codebook[second_new].to(torch.float32)
        )
        new_error = (objective_weight * (reconstruction - samples).square()).sum(1)
        better = new_error < error
        first = torch.where(better, first_new, first)
        second = torch.where(better, second_new, second)
        error = torch.where(better, new_error, error)
    return first, second, error


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
    config: NvqAdditiveConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = xgroup.shape[0]
    group_anchor = neuron_scale.repeat_interleave(ng)
    best_error = torch.full((groups,), torch.inf, device=xgroup.device)
    best_state = torch.zeros(groups, dtype=torch.int64, device=xgroup.device)
    best_first = torch.zeros((groups, 3), dtype=torch.int64, device=xgroup.device)
    best_second = torch.zeros_like(best_first)
    for start in range(0, groups, config.group_chunk):
        stop = min(start + config.group_chunk, groups)
        count = stop - start
        samples = xgroup[start:stop].reshape(-1, _VECTOR_SIZE)
        weights = wgroup[start:stop].reshape_as(samples)
        local_error = best_error[start:stop]
        local_state = best_state[start:stop]
        local_first = best_first[start:stop]
        local_second = best_second[start:stop]
        for state_id in range(_STATE_COUNT):
            bank_id = int(bank_for_state[state_id].item())
            group_scale = group_anchor[start:stop] * alpha[state_id]
            first, second, vector_error = _additive_nearest(
                samples,
                weights,
                group_scale.repeat_interleave(_VECTORS_PER_GROUP),
                first_codebooks[bank_id],
                second_codebooks[bank_id],
                beam_size=config.beam_size,
                pair_refine_steps=config.pair_refine_steps,
            )
            error = vector_error.reshape(count, _VECTORS_PER_GROUP).sum(1)
            better = error < local_error
            local_error = torch.where(better, error, local_error)
            local_state = torch.where(better, torch.full_like(local_state, state_id), local_state)
            local_first[better] = first.reshape(count, -1)[better]
            local_second[better] = second.reshape(count, -1)[better]
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
    result = torch.empty((state.numel(), _GROUP_SIZE), dtype=torch.float32, device=state.device)
    group_bank = bank_for_state[state]
    for bank_id in range(first_codebooks.shape[0]):
        selected = group_bank == bank_id
        if not bool(selected.any()):
            continue
        first = first_codebooks[bank_id][first_indices[selected]].to(torch.float32)
        second = second_codebooks[bank_id][second_indices[selected]].to(torch.float32)
        result[selected] = (first + second).reshape(-1, _GROUP_SIZE)
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
    numerator = torch.zeros_like(previous, dtype=torch.float32)
    denominator = torch.zeros_like(previous, dtype=torch.float32)
    expanded = indices.unsqueeze(1).expand(-1, _VECTOR_SIZE)
    numerator.scatter_add_(0, expanded, objective_weight * scale.unsqueeze(1) * samples)
    denominator.scatter_add_(0, expanded, objective_weight * scale.square().unsqueeze(1))
    centroid = torch.where(denominator > 0, numerator / denominator, previous.to(torch.float32))
    return centroid.round().clamp(-127, 127).to(torch.int8).contiguous()


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
    samples = xgroup.reshape(-1, _VECTOR_SIZE)
    weights = wgroup.reshape_as(samples)
    group_bank = bank_for_state[state]
    group_scale = neuron_scale.repeat_interleave(ng) * alpha[state]
    vector_bank = group_bank.repeat_interleave(_VECTORS_PER_GROUP)
    vector_scale = group_scale.repeat_interleave(_VECTORS_PER_GROUP)
    flat_first = first_indices.reshape(-1)
    flat_second = second_indices.reshape(-1)
    updated_first = first_codebooks.clone()
    updated_second = second_codebooks.clone()
    for bank_id in range(first_codebooks.shape[0]):
        selected = vector_bank == bank_id
        if not bool(selected.any()):
            continue
        second_code = second_codebooks[bank_id][flat_second[selected]].to(torch.float32)
        first_residual = samples[selected] - vector_scale[selected].unsqueeze(1) * second_code
        updated_first[bank_id] = _update_one_codebook(
            first_residual,
            weights[selected],
            vector_scale[selected],
            flat_first[selected],
            first_codebooks[bank_id],
        )
        first_code = updated_first[bank_id][flat_first[selected]].to(torch.float32)
        second_residual = samples[selected] - vector_scale[selected].unsqueeze(1) * first_code
        updated_second[bank_id] = _update_one_codebook(
            second_residual,
            weights[selected],
            vector_scale[selected],
            flat_second[selected],
            second_codebooks[bank_id],
        )
    return updated_first.contiguous(), updated_second.contiguous()


def _initial_additive_codebooks(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    initial_jsc: NvqJscTensor,
    config: NvqAdditiveConfig,
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
    samples = (xgroup / safe_scale.unsqueeze(1)).reshape(-1, _VECTOR_SIZE)
    weights = wgroup.reshape_as(samples)
    vector_bank = bank_for_state[state].repeat_interleave(_VECTORS_PER_GROUP)
    first = torch.empty(
        (config.banks, _FIRST_ENTRIES, _VECTOR_SIZE),
        dtype=torch.int8,
        device=xgroup.device,
    )
    second = torch.empty(
        (config.banks, _SECOND_ENTRIES, _VECTOR_SIZE),
        dtype=torch.int8,
        device=xgroup.device,
    )
    all_vectors = torch.ones_like(vector_bank, dtype=torch.bool)
    for bank_id in range(config.banks):
        selected = vector_bank == bank_id
        if not bool(selected.any()):
            selected = all_vectors
        bank_samples = samples[selected]
        bank_weights = weights[selected]
        first[bank_id] = _weighted_kmeans(
            bank_samples,
            bank_weights,
            _FIRST_ENTRIES,
            iterations=config.kmeans_iterations,
            initialization_points=config.kmeans_initialization_points,
            seed=config.seed + 2 * bank_id,
        )
        first_index, _ = _nearest(
            bank_samples,
            bank_weights,
            torch.ones(bank_samples.shape[0], device=xgroup.device),
            first[bank_id],
        )
        residual = bank_samples - first[bank_id][first_index].to(torch.float32)
        second[bank_id] = _weighted_kmeans(
            residual,
            bank_weights,
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
) -> NvqAdditiveTensor:
    out, neuron_len = shape
    ng = math.ceil(neuron_len / _GROUP_SIZE)
    nvec = math.ceil(neuron_len / _VECTOR_SIZE)
    return NvqAdditiveTensor(
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
def train_nvq_additive(
    weight: torch.Tensor,
    initial_jsc: NvqJscTensor,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    config: NvqAdditiveConfig | None = None,
    device: str | torch.device = "cuda",
) -> tuple[NvqAdditiveTensor, tuple[NvqAdditiveIteration, ...]]:
    config = NvqAdditiveConfig() if config is None else config
    value, out, neuron_len = _prepare_weight(weight, device)
    if initial_jsc.shape != (out, neuron_len):
        raise ValueError("NVQ-AQ initialization shape does not match training weight")
    if initial_jsc.codebooks.shape[0] != config.banks:
        raise ValueError("NVQ-AQ bank count does not match JSC initialization")
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
    first_codebooks, second_codebooks = _initial_additive_codebooks(
        xgroup, wgroup, initial_jsc, config
    )

    signal = float((wgroup * xgroup.square()).sum().item())
    history: list[NvqAdditiveIteration] = []
    best = None
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
            config=config,
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
                config=config,
            )
        total_error = float(error.sum().item())
        group_bank = bank_for_state[state]
        history.append(
            NvqAdditiveIteration(
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
def quantize_nvq_additive_fixed(
    weight: torch.Tensor,
    tables: NvqAdditiveTables,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    config: NvqAdditiveConfig | None = None,
    device: str | torch.device = "cuda",
) -> NvqAdditiveTensor:
    config = NvqAdditiveConfig() if config is None else config
    alpha_np, bank_np, first_np, second_np = _validate_tables(tables)
    if first_np.shape[0] != config.banks:
        raise ValueError("NVQ-AQ fixed table bank count does not match config")
    value, out, neuron_len = _prepare_weight(weight, device)
    value, objective_weight, ng = _pad_weight(value, _GROUP_SIZE, importance)
    xgroup = value.reshape(out * ng, _GROUP_SIZE).contiguous()
    wgroup = objective_weight.reshape_as(xgroup).contiguous()
    alpha = torch.as_tensor(alpha_np, device=value.device, dtype=torch.float32)
    bank_for_state = torch.as_tensor(bank_np, device=value.device, dtype=torch.int64)
    first_codebooks = torch.as_tensor(first_np, device=value.device, dtype=torch.int8)
    second_codebooks = torch.as_tensor(second_np, device=value.device, dtype=torch.int8)
    maximum = max(
        int(np.max(np.abs(first_np.astype(np.int16)) + np.abs(second_np).max())),
        1,
    )
    base_anchor = value.abs().amax(1) / (float(alpha.max().item()) * maximum)

    best_row_error = torch.full((out,), torch.inf, device=value.device)
    best_anchor = torch.zeros(out, device=value.device)
    best_state = torch.zeros(out * ng, dtype=torch.int64, device=value.device)
    best_first = torch.zeros((out * ng, 3), dtype=torch.int64, device=value.device)
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
            config=config,
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
                config=config,
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


def dequantize_nvq_additive(tensor: NvqAdditiveTensor) -> np.ndarray:
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
    code = (first.astype(np.int16) + second.astype(np.int16)).reshape(out, -1)
    code = code[:, :neuron_len]
    group_scale = tensor.neuron_scale[:, None] * tensor.scale_lut[tensor.state]
    scale = np.repeat(group_scale, _GROUP_SIZE, axis=1)[:, :neuron_len]
    return code.astype(np.float32) * scale


__all__ = [
    "NvqAdditiveConfig",
    "NvqAdditiveIteration",
    "NvqAdditiveTables",
    "NvqAdditiveTensor",
    "additive_tables_from_tensor",
    "dequantize_nvq_additive",
    "quantize_nvq_additive_fixed",
    "train_nvq_additive",
]
