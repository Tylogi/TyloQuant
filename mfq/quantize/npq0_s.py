"""Weighted-SSE PQ3+3 trainer and fixed-table quantizer for NPQ0-S."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from mfq.formats.npq0_s import Npq0STensor, validate_npq0_s_tables
from mfq.quantize.nvq_quant_torch import _fp16_round, _pad_weight, _prepare_weight

_GROUP_SIZE = 24
_VECTOR_SIZE = 8
_SUBVECTOR_SIZE = 4
_VECTORS_PER_GROUP = 3
_STATE_COUNT = 4
_FIRST_ENTRIES = 8
_SECOND_ENTRIES = 8


@dataclass(frozen=True)
class Npq0SConfig:
    iterations: int = 4
    assignment_refine_steps: int = 2
    fixed_refine_steps: int = 3
    kmeans_iterations: int = 8
    kmeans_initialization_points: int = 16384
    group_chunk: int = 256
    anchor_multipliers: tuple[float, ...] = (0.625, 0.8, 1.0, 1.25, 1.6)
    seed: int = 20260722

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.assignment_refine_steps < 0:
            raise ValueError("NPQ0-S iteration counts must be non-negative")
        if self.fixed_refine_steps < 0 or self.kmeans_iterations < 0:
            raise ValueError("NPQ0-S refinement counts must be non-negative")
        if self.kmeans_initialization_points <= 0 or self.group_chunk <= 0:
            raise ValueError("NPQ0-S chunk sizes must be positive")
        if not self.anchor_multipliers or any(
            not math.isfinite(value) or value <= 0 for value in self.anchor_multipliers
        ):
            raise ValueError("NPQ0-S anchor multipliers must be finite and positive")


@dataclass(frozen=True)
class Npq0STables:
    scale_lut: np.ndarray
    first_codebooks: np.ndarray
    second_codebooks: np.ndarray


@dataclass(frozen=True)
class Npq0SIteration:
    iteration: int
    weighted_sse: float
    weighted_nmse_percent: float
    used_states: int
    used_first_codes: tuple[int, ...]
    used_second_codes: tuple[int, ...]


def npq0_s_tables_from_tensor(tensor: Npq0STensor) -> Npq0STables:
    return Npq0STables(
        scale_lut=np.asarray(tensor.scale_lut, dtype=np.float32).copy(),
        first_codebooks=np.asarray(tensor.first_codebooks, dtype=np.int8).copy(),
        second_codebooks=np.asarray(tensor.second_codebooks, dtype=np.int8).copy(),
    )


def _validate_tables(
    tables: Npq0STables,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return validate_npq0_s_tables(
        tables.scale_lut,
        tables.first_codebooks,
        tables.second_codebooks,
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


def _assign_groups(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    neuron_scale: torch.Tensor,
    scale_lut: torch.Tensor,
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
    best_first = torch.zeros(
        (groups, _VECTORS_PER_GROUP),
        dtype=torch.int64,
        device=xgroup.device,
    )
    best_second = torch.zeros_like(best_first)

    for start in range(0, groups, group_chunk):
        stop = min(start + group_chunk, groups)
        count = stop - start
        vectors = xgroup[start:stop].reshape(
            count, _VECTORS_PER_GROUP, 2, _SUBVECTOR_SIZE
        )
        weights = wgroup[start:stop].reshape_as(vectors)
        first_x = vectors[:, :, 0].reshape(-1, _SUBVECTOR_SIZE)
        second_x = vectors[:, :, 1].reshape(-1, _SUBVECTOR_SIZE)
        first_w = weights[:, :, 0].reshape_as(first_x)
        second_w = weights[:, :, 1].reshape_as(second_x)
        local_error = best_error[start:stop]
        local_state = best_state[start:stop]
        local_first = best_first[start:stop]
        local_second = best_second[start:stop]
        for state_id in range(_STATE_COUNT):
            group_scale = group_anchor[start:stop] * scale_lut[state_id]
            vector_scale = group_scale.repeat_interleave(_VECTORS_PER_GROUP)
            first_index, first_error = _nearest(
                first_x,
                first_w,
                vector_scale,
                first_codebooks[state_id],
            )
            second_index, second_error = _nearest(
                second_x,
                second_w,
                vector_scale,
                second_codebooks[state_id],
            )
            error = (first_error + second_error).reshape(
                count, _VECTORS_PER_GROUP
            ).sum(1)
            better = error < local_error
            local_error = torch.where(better, error, local_error)
            local_state = torch.where(
                better,
                torch.full_like(local_state, state_id),
                local_state,
            )
            local_first[better] = first_index.reshape(
                count, _VECTORS_PER_GROUP
            )[better]
            local_second[better] = second_index.reshape(
                count, _VECTORS_PER_GROUP
            )[better]
        best_error[start:stop] = local_error
        best_state[start:stop] = local_state
        best_first[start:stop] = local_first
        best_second[start:stop] = local_second
    return best_state, best_first, best_second, best_error


def _codes_for_assignment(
    state: torch.Tensor,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
) -> torch.Tensor:
    result = torch.empty(
        (state.numel(), _GROUP_SIZE),
        device=state.device,
        dtype=torch.float32,
    )
    for state_id in range(_STATE_COUNT):
        selected = state == state_id
        if bool(selected.any()):
            first = first_codebooks[state_id][first_indices[selected]].to(torch.float32)
            second = second_codebooks[state_id][second_indices[selected]].to(torch.float32)
            result[selected] = torch.cat((first, second), dim=-1).reshape(
                -1, _GROUP_SIZE
            )
    return result


def _refit_anchor_and_lut(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    state: torch.Tensor,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
    neuron_scale: torch.Tensor,
    scale_lut: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
    *,
    out: int,
    ng: int,
    learn_lut: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    code = _codes_for_assignment(
        state,
        first_indices,
        second_indices,
        first_codebooks,
        second_codebooks,
    )
    basis = scale_lut[state].unsqueeze(1) * code
    numerator = (wgroup * xgroup * basis).reshape(out, ng, _GROUP_SIZE).sum((1, 2))
    denominator = (wgroup * basis.square()).reshape(out, ng, _GROUP_SIZE).sum((1, 2))
    fitted_anchor = torch.where(
        denominator > 0,
        numerator / denominator,
        neuron_scale,
    ).clamp_min(0)
    fitted_anchor = _fp16_round(fitted_anchor)
    if not learn_lut:
        return fitted_anchor, scale_lut

    group_anchor = fitted_anchor.repeat_interleave(ng)
    lut_num = torch.zeros_like(scale_lut)
    lut_den = torch.zeros_like(scale_lut)
    lut_num.scatter_add_(
        0,
        state,
        (wgroup * xgroup * code).sum(1) * group_anchor,
    )
    lut_den.scatter_add_(
        0,
        state,
        (wgroup * code.square()).sum(1) * group_anchor.square(),
    )
    fitted_lut = torch.where(lut_den > 0, lut_num / lut_den, scale_lut).clamp_min(0)
    maximum = fitted_lut.max()
    if float(maximum.item()) > 0:
        fitted_lut = fitted_lut / maximum
        fitted_anchor = _fp16_round(fitted_anchor * maximum)
    return fitted_anchor, _fp16_round(fitted_lut)


def _update_one_codebook(
    samples: torch.Tensor,
    objective_weight: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
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
    return (
        torch.where(
            denominator > 0,
            numerator / denominator,
            previous.to(torch.float32),
        )
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
    )


def _update_codebooks(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    state: torch.Tensor,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
    neuron_scale: torch.Tensor,
    scale_lut: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
    *,
    ng: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = xgroup.reshape(
        -1, _VECTORS_PER_GROUP, 2, _SUBVECTOR_SIZE
    )
    weights = wgroup.reshape_as(vectors)
    group_scale = neuron_scale.repeat_interleave(ng) * scale_lut[state]
    updated_first = first_codebooks.clone()
    updated_second = second_codebooks.clone()
    for state_id in range(_STATE_COUNT):
        selected = state == state_id
        if not bool(selected.any()):
            continue
        scale = group_scale[selected].repeat_interleave(_VECTORS_PER_GROUP)
        updated_first[state_id] = _update_one_codebook(
            vectors[selected, :, 0].reshape(-1, _SUBVECTOR_SIZE),
            weights[selected, :, 0].reshape(-1, _SUBVECTOR_SIZE),
            scale,
            first_indices[selected].reshape(-1),
            first_codebooks[state_id],
        )
        updated_second[state_id] = _update_one_codebook(
            vectors[selected, :, 1].reshape(-1, _SUBVECTOR_SIZE),
            weights[selected, :, 1].reshape(-1, _SUBVECTOR_SIZE),
            scale,
            second_indices[selected].reshape(-1),
            second_codebooks[state_id],
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
    valid = objective_weight.sum(1) > 0
    samples = samples[valid]
    objective_weight = objective_weight[valid]
    if not samples.shape[0]:
        return torch.zeros(
            (entries, _SUBVECTOR_SIZE), dtype=torch.int8, device=samples.device
        )

    generator = torch.Generator(device=samples.device)
    generator.manual_seed(seed)
    if samples.shape[0] > initialization_points:
        chosen = torch.randperm(
            samples.shape[0],
            generator=generator,
            device=samples.device,
        )[:initialization_points]
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
                torch.multinomial(
                    probabilities / total,
                    1,
                    generator=generator,
                ).item()
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
        numerator = torch.zeros((entries, _SUBVECTOR_SIZE), device=samples.device)
        denominator = torch.zeros_like(numerator)
        for start in range(0, samples.shape[0], 8192):
            stop = min(start + 8192, samples.shape[0])
            index, _ = _nearest(
                samples[start:stop],
                objective_weight[start:stop],
                torch.ones(stop - start, device=samples.device),
                table,
            )
            expanded = index.unsqueeze(1).expand(-1, _SUBVECTOR_SIZE)
            numerator.scatter_add_(
                0,
                expanded,
                objective_weight[start:stop] * samples[start:stop],
            )
            denominator.scatter_add_(0, expanded, objective_weight[start:stop])
        candidate = (
            torch.where(
                denominator > 0,
                numerator / denominator,
                table.to(torch.float32),
            )
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
        )
        if torch.equal(candidate, table):
            break
        table = candidate
    return table.contiguous()


def _initialize(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    *,
    out: int,
    ng: int,
    config: Npq0SConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group_peak = xgroup.abs().amax(1).reshape(out, ng)
    row_peak = group_peak.amax(1)
    neuron_scale = _fp16_round(row_peak / 127.0)
    safe_anchor = torch.where(neuron_scale > 0, neuron_scale, torch.ones_like(neuron_scale))
    ratio = group_peak.reshape(-1) / (safe_anchor.repeat_interleave(ng) * 127.0)
    scale_lut = torch.linspace(
        1.0 / _STATE_COUNT,
        1.0,
        _STATE_COUNT,
        device=xgroup.device,
        dtype=torch.float32,
    )
    state = (ratio.unsqueeze(1) - scale_lut.unsqueeze(0)).abs().argmin(1)
    vectors = xgroup.reshape(
        -1, _VECTORS_PER_GROUP, 2, _SUBVECTOR_SIZE
    )
    weights = wgroup.reshape_as(vectors)
    first = torch.empty(
        (_STATE_COUNT, _FIRST_ENTRIES, _SUBVECTOR_SIZE),
        dtype=torch.int8,
        device=xgroup.device,
    )
    second = torch.empty(
        (_STATE_COUNT, _SECOND_ENTRIES, _SUBVECTOR_SIZE),
        dtype=torch.int8,
        device=xgroup.device,
    )
    all_groups = torch.ones_like(state, dtype=torch.bool)
    group_anchor = safe_anchor.repeat_interleave(ng)
    for state_id in range(_STATE_COUNT):
        selected = state == state_id
        if not bool(selected.any()):
            selected = all_groups
        denominator = group_anchor[selected] * scale_lut[state_id]
        normalized = vectors[selected] / denominator[:, None, None, None]
        first[state_id] = _weighted_kmeans(
            normalized[:, :, 0].reshape(-1, _SUBVECTOR_SIZE),
            weights[selected, :, 0].reshape(-1, _SUBVECTOR_SIZE),
            _FIRST_ENTRIES,
            iterations=config.kmeans_iterations,
            initialization_points=config.kmeans_initialization_points,
            seed=config.seed + 2 * state_id,
        )
        second[state_id] = _weighted_kmeans(
            normalized[:, :, 1].reshape(-1, _SUBVECTOR_SIZE),
            weights[selected, :, 1].reshape(-1, _SUBVECTOR_SIZE),
            _SECOND_ENTRIES,
            iterations=config.kmeans_iterations,
            initialization_points=config.kmeans_initialization_points,
            seed=config.seed + 2 * state_id + 1,
        )
    return neuron_scale, _fp16_round(scale_lut), first, second


def _tensor_from_assignment(
    shape: tuple[int, int],
    neuron_scale: torch.Tensor,
    scale_lut: torch.Tensor,
    state: torch.Tensor,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
) -> Npq0STensor:
    out, neuron_len = shape
    ng = math.ceil(neuron_len / _GROUP_SIZE)
    nvec = neuron_len // _VECTOR_SIZE
    composite = first_indices | (second_indices << 3)
    return Npq0STensor(
        shape=shape,
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=neuron_scale.cpu().numpy().astype(np.float32, copy=False),
        scale_lut=scale_lut.cpu().numpy().astype(np.float32, copy=False),
        state=state.reshape(out, ng).cpu().numpy().astype(np.uint8, copy=False),
        indices=(
            composite.reshape(out, ng * _VECTORS_PER_GROUP)[:, :nvec]
            .cpu()
            .numpy()
            .astype(np.uint8, copy=False)
        ),
        first_codebooks=first_codebooks.cpu().numpy().astype(np.int8, copy=False),
        second_codebooks=second_codebooks.cpu().numpy().astype(np.int8, copy=False),
    )


@torch.inference_mode()
def train_npq0_s(
    weight: torch.Tensor,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    config: Npq0SConfig | None = None,
    device: str | torch.device = "cuda",
) -> tuple[Npq0STensor, tuple[Npq0SIteration, ...]]:
    """Train tensor-wise state-conditioned PQ3+3 tables by coordinate descent."""

    config = Npq0SConfig() if config is None else config
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("NPQ0-S CUDA training requested without a CUDA device")
    value, out, neuron_len = _prepare_weight(weight, device)
    if neuron_len % _VECTOR_SIZE:
        raise ValueError("NPQ0-S requires K divisible by 8")
    value, objective_weight, ng = _pad_weight(value, _GROUP_SIZE, importance)
    xgroup = value.reshape(out * ng, _GROUP_SIZE).contiguous()
    wgroup = objective_weight.reshape_as(xgroup).contiguous()
    neuron_scale, scale_lut, first_codebooks, second_codebooks = _initialize(
        xgroup,
        wgroup,
        out=out,
        ng=ng,
        config=config,
    )

    signal = float((wgroup * xgroup.square()).sum().item())
    history: list[Npq0SIteration] = []
    best: tuple[
        float,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None = None
    for iteration in range(config.iterations + 1):
        state, first_indices, second_indices, error = _assign_groups(
            xgroup,
            wgroup,
            neuron_scale,
            scale_lut,
            first_codebooks,
            second_codebooks,
            ng=ng,
            group_chunk=config.group_chunk,
        )
        for _ in range(config.assignment_refine_steps):
            neuron_scale, scale_lut = _refit_anchor_and_lut(
                xgroup,
                wgroup,
                state,
                first_indices,
                second_indices,
                neuron_scale,
                scale_lut,
                first_codebooks,
                second_codebooks,
                out=out,
                ng=ng,
                learn_lut=True,
            )
            state, first_indices, second_indices, error = _assign_groups(
                xgroup,
                wgroup,
                neuron_scale,
                scale_lut,
                first_codebooks,
                second_codebooks,
                ng=ng,
                group_chunk=config.group_chunk,
            )
        total_error = float(error.sum().item())
        history.append(
            Npq0SIteration(
                iteration=iteration,
                weighted_sse=total_error,
                weighted_nmse_percent=100.0 * total_error / signal if signal else 0.0,
                used_states=int(torch.unique(state).numel()),
                used_first_codes=tuple(
                    (
                        int(torch.unique(first_indices[state == item]).numel())
                        if bool((state == item).any())
                        else 0
                    )
                    for item in range(_STATE_COUNT)
                ),
                used_second_codes=tuple(
                    (
                        int(torch.unique(second_indices[state == item]).numel())
                        if bool((state == item).any())
                        else 0
                    )
                    for item in range(_STATE_COUNT)
                ),
            )
        )
        if best is None or total_error < best[0]:
            best = (
                total_error,
                neuron_scale.clone(),
                scale_lut.clone(),
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
            scale_lut,
            first_codebooks,
            second_codebooks,
            ng=ng,
        )

    assert best is not None
    (
        _,
        neuron_scale,
        scale_lut,
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
            scale_lut,
            state,
            first_indices,
            second_indices,
            first_codebooks,
            second_codebooks,
        ),
        tuple(history),
    )


@torch.inference_mode()
def quantize_npq0_s_fixed(
    weight: torch.Tensor,
    tables: Npq0STables,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    config: Npq0SConfig | None = None,
    device: str | torch.device = "cuda",
) -> Npq0STensor:
    """Assign a matrix to fixed product tables and fit FP16 row anchors."""

    config = Npq0SConfig() if config is None else config
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("NPQ0-S CUDA quantization requested without a CUDA device")
    scale_np, first_np, second_np = _validate_tables(tables)
    value, out, neuron_len = _prepare_weight(weight, device)
    if neuron_len % _VECTOR_SIZE:
        raise ValueError("NPQ0-S requires K divisible by 8")
    value, objective_weight, ng = _pad_weight(value, _GROUP_SIZE, importance)
    xgroup = value.reshape(out * ng, _GROUP_SIZE).contiguous()
    wgroup = objective_weight.reshape_as(xgroup).contiguous()
    scale_lut = torch.as_tensor(scale_np, device=value.device, dtype=torch.float32)
    first_codebooks = torch.as_tensor(first_np, device=value.device, dtype=torch.int8)
    second_codebooks = torch.as_tensor(second_np, device=value.device, dtype=torch.int8)
    maximum_code = max(int(np.abs(first_np).max()), int(np.abs(second_np).max()), 1)
    maximum_scale = max(float(scale_np.max()), 1e-12)
    base_anchor = value.abs().amax(1) / (maximum_code * maximum_scale)

    if value.is_cuda:
        from mfq.quantize.cuda._ext import ext

        native_scale = torch.as_tensor(
            scale_np, device=value.device, dtype=torch.float32
        ).contiguous()
        native_first = torch.as_tensor(
            first_np, device=value.device, dtype=torch.int8
        ).contiguous()
        native_second = torch.as_tensor(
            second_np, device=value.device, dtype=torch.int8
        ).contiguous()
        best_row_error = torch.full((out,), torch.inf, device=value.device)
        best_anchor = torch.zeros(out, device=value.device)
        best_state = torch.zeros((out, ng), dtype=torch.uint8, device=value.device)
        best_first = torch.zeros(
            (out, ng, _VECTORS_PER_GROUP),
            dtype=torch.uint8,
            device=value.device,
        )
        best_second = torch.zeros_like(best_first)
        for multiplier in config.anchor_multipliers:
            initial_anchor = _fp16_round(base_anchor * multiplier).contiguous()
            anchor, state, first_indices, second_indices, row_error = (
                ext().npq0_s_assign(
                    value.contiguous(),
                    objective_weight.contiguous(),
                    initial_anchor,
                    native_scale,
                    native_first,
                    native_second,
                    neuron_len,
                    config.fixed_refine_steps,
                )
            )
            better_rows = row_error < best_row_error
            best_row_error = torch.where(
                better_rows, row_error, best_row_error
            )
            best_anchor = torch.where(better_rows, anchor, best_anchor)
            best_state[better_rows] = state[better_rows]
            best_first[better_rows] = first_indices[better_rows]
            best_second[better_rows] = second_indices[better_rows]
        return _tensor_from_assignment(
            (out, neuron_len),
            best_anchor,
            native_scale,
            best_state.reshape(-1),
            best_first.reshape(-1, _VECTORS_PER_GROUP),
            best_second.reshape(-1, _VECTORS_PER_GROUP),
            native_first,
            native_second,
        )

    best_row_error = torch.full((out,), torch.inf, device=value.device)
    best_anchor = torch.zeros(out, device=value.device)
    best_state = torch.zeros(out * ng, dtype=torch.int64, device=value.device)
    best_first = torch.zeros(
        (out * ng, _VECTORS_PER_GROUP),
        dtype=torch.int64,
        device=value.device,
    )
    best_second = torch.zeros_like(best_first)
    for multiplier in config.anchor_multipliers:
        anchor = _fp16_round(base_anchor * multiplier)
        state, first_indices, second_indices, error = _assign_groups(
            xgroup,
            wgroup,
            anchor,
            scale_lut,
            first_codebooks,
            second_codebooks,
            ng=ng,
            group_chunk=config.group_chunk,
        )
        for _ in range(config.fixed_refine_steps):
            anchor, _ = _refit_anchor_and_lut(
                xgroup,
                wgroup,
                state,
                first_indices,
                second_indices,
                anchor,
                scale_lut,
                first_codebooks,
                second_codebooks,
                out=out,
                ng=ng,
                learn_lut=False,
            )
            state, first_indices, second_indices, error = _assign_groups(
                xgroup,
                wgroup,
                anchor,
                scale_lut,
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
        scale_lut,
        best_state,
        best_first,
        best_second,
        first_codebooks,
        second_codebooks,
    )


def dequantize_npq0_s(tensor: Npq0STensor) -> np.ndarray:
    out = int(np.asarray(tensor.neuron_scale).size)
    neuron_len = int(tensor.neuron_len)
    nvec = neuron_len // _VECTOR_SIZE
    vector_state = np.repeat(np.asarray(tensor.state, dtype=np.uint8), 3, axis=1)[:, :nvec]
    composite = np.asarray(tensor.indices, dtype=np.uint8)
    first_index = composite & 7
    second_index = composite >> 3
    first = np.asarray(tensor.first_codebooks, dtype=np.int8)[
        vector_state, first_index
    ]
    second = np.asarray(tensor.second_codebooks, dtype=np.int8)[
        vector_state, second_index
    ]
    code = np.concatenate((first, second), axis=-1)
    group_scale = (
        np.asarray(tensor.neuron_scale, dtype=np.float32)[:, None]
        * np.asarray(tensor.scale_lut, dtype=np.float32)[np.asarray(tensor.state)]
    )
    scale = np.repeat(group_scale, _GROUP_SIZE, axis=1)[:, :neuron_len]
    matrix = code.reshape(out, neuron_len).astype(np.float32) * scale
    moved_shape = (tensor.shape[tensor.axis],) + tuple(
        tensor.shape[index] for index in range(len(tensor.shape)) if index != tensor.axis
    )
    return np.moveaxis(matrix.reshape(moved_shape), 0, tensor.axis)


__all__ = [
    "Npq0SConfig",
    "Npq0SIteration",
    "Npq0STables",
    "dequantize_npq0_s",
    "npq0_s_tables_from_tensor",
    "quantize_npq0_s_fixed",
    "train_npq0_s",
]
