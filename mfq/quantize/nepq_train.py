"""Batched GPU trainer for the tensor-wise banks used by NEPQ0-S."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from mfq.formats.nepq import rotation_signs
from mfq.formats.npq0_s import NPQ0_S_TABLE_BYTES, pack_npq0_s_tables


_GROUP_SIZE = 24
_VECTORS_PER_GROUP = 3
_SUBVECTOR_SIZE = 4
_STATE_COUNT = 4
_ENTRIES = 8


@dataclass(frozen=True)
class NepqBankTrainConfig:
    iterations: int = 3
    assignment_refine_steps: int = 1
    kmeans_iterations: int = 6
    expert_batch: int = 32
    seed: int = 20260723

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.assignment_refine_steps < 0:
            raise ValueError("NEPQ bank iteration counts must be non-negative")
        if self.kmeans_iterations < 0 or self.expert_batch <= 0:
            raise ValueError("invalid NEPQ bank trainer configuration")


@dataclass(frozen=True)
class NepqBankTraining:
    table_payloads: np.ndarray
    weighted_nmse_percent: np.ndarray


def _fp16_round(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float16).to(torch.float32)


def signed_hadamard_rotate(
    value: torch.Tensor,
    block: int,
    seed: int,
) -> torch.Tensor:
    """Apply the exact signed, orthonormal block Hadamard used by NEPQ."""

    if block <= 0 or block & (block - 1) or value.shape[-1] % block:
        raise ValueError("rotation block must be a power of two dividing K")
    signs = torch.as_tensor(
        rotation_signs(int(value.shape[-1]), block, seed),
        device=value.device,
        dtype=torch.float32,
    )
    result = (value.to(torch.float32) * signs).reshape(-1, block).contiguous()
    stride = 1
    while stride < block:
        paired = result.reshape(-1, 2, stride)
        first = paired[:, 0].clone()
        second = paired[:, 1].clone()
        paired[:, 0] = first + second
        paired[:, 1] = first - second
        stride *= 2
    result.mul_(1.0 / math.sqrt(block))
    return result.reshape_as(value)


def hadamard_diagonal_importance(
    importance: torch.Tensor,
    block: int,
) -> torch.Tensor:
    """Return diag(H diag(importance) H) for each Hadamard block."""

    if block <= 0 or block & (block - 1) or importance.shape[-1] % block:
        raise ValueError("importance width must be divisible by a power-of-two block")
    value = importance.to(torch.float32)
    block_mean = value.reshape(*value.shape[:-1], -1, block).mean(-1, keepdim=True)
    return block_mean.expand(*block_mean.shape[:-1], block).reshape_as(value)


def _gather_centers(
    samples: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    batch = torch.arange(samples.shape[0], device=samples.device)
    return samples[batch, indices]


def _batched_kmeans(
    samples: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    *,
    iterations: int,
) -> torch.Tensor:
    """Train one 8x4 int8 table for every leading batch item."""

    batches, points, width = samples.shape
    if (
        width != _SUBVECTOR_SIZE
        or weights.shape != samples.shape
        or valid.shape != (batches, points)
    ):
        raise ValueError("invalid batched k-means shapes")
    positive = weights.sum(2) > 0
    valid = valid & positive
    if not bool(valid.any(dim=1).all()):
        raise ValueError("every batched k-means item needs at least one sample")

    norm = (weights * samples.square()).sum(2)
    first = norm.masked_fill(~valid, -torch.inf).argmax(1)
    centers = torch.empty(
        (batches, _ENTRIES, width),
        device=samples.device,
        dtype=torch.float32,
    )
    centers[:, 0] = _gather_centers(samples, first)
    minimum = (weights * (samples - centers[:, :1]).square()).sum(2)
    minimum.masked_fill_(~valid, -torch.inf)
    for center_id in range(1, _ENTRIES):
        selected = minimum.argmax(1)
        centers[:, center_id] = _gather_centers(samples, selected)
        distance = (
            weights
            * (samples - centers[:, center_id : center_id + 1]).square()
        ).sum(2)
        minimum = torch.minimum(minimum, distance)
        minimum.masked_fill_(~valid, -torch.inf)

    centers = centers.round().clamp(-127, 127)
    batch_offset = (
        torch.arange(batches, device=samples.device, dtype=torch.int64)
        * _ENTRIES
    )[:, None]
    coordinate_weight = weights * valid[:, :, None]
    for _ in range(iterations):
        distance = (
            weights[:, :, None, :]
            * (samples[:, :, None, :] - centers[:, None, :, :]).square()
        ).sum(3)
        index = distance.argmin(2)
        key = (batch_offset + index).reshape(-1)
        numerator = torch.zeros(
            (batches * _ENTRIES, width),
            device=samples.device,
            dtype=torch.float32,
        )
        denominator = torch.zeros(
            (batches * _ENTRIES, width),
            device=samples.device,
            dtype=torch.float32,
        )
        numerator.scatter_add_(
            0,
            key[:, None].expand(-1, width),
            (samples * coordinate_weight).reshape(-1, width),
        )
        denominator.scatter_add_(
            0,
            key[:, None].expand(-1, width),
            coordinate_weight.reshape(-1, width),
        )
        candidate = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1e-20),
            centers.reshape(-1, width),
        ).reshape_as(centers)
        candidate = candidate.round().clamp(-127, 127)
        if torch.equal(candidate, centers):
            break
        centers = candidate
    return centers.to(torch.int8).contiguous()


def _initialize(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    *,
    rows: int,
    ng: int,
    kmeans_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batches = xgroup.shape[0]
    group_peak = xgroup.abs().amax(2).reshape(batches, rows, ng)
    neuron_scale = _fp16_round(group_peak.amax(2) / 127.0)
    safe_anchor = torch.where(
        neuron_scale > 0, neuron_scale, torch.ones_like(neuron_scale)
    )
    scale_lut = torch.linspace(
        0.25,
        1.0,
        _STATE_COUNT,
        device=xgroup.device,
        dtype=torch.float32,
    )[None].repeat(batches, 1)
    group_anchor = safe_anchor[:, :, None].expand(-1, -1, ng).reshape(batches, -1)
    ratio = group_peak.reshape(batches, -1) / (group_anchor * 127.0)
    initial_state = (
        ratio[:, :, None] - scale_lut[:, None, :]
    ).abs().argmin(2)
    vectors = xgroup.reshape(
        batches, -1, _VECTORS_PER_GROUP, 2, _SUBVECTOR_SIZE
    )
    vector_weights = wgroup.reshape_as(vectors)
    first_codebooks = torch.empty(
        (batches, _STATE_COUNT, _ENTRIES, _SUBVECTOR_SIZE),
        device=xgroup.device,
        dtype=torch.int8,
    )
    second_codebooks = torch.empty_like(first_codebooks)
    for state in range(_STATE_COUNT):
        valid_group = initial_state == state
        have_state = valid_group.any(1)
        valid_group = torch.where(
            have_state[:, None],
            valid_group,
            torch.ones_like(valid_group),
        )
        denominator = (
            group_anchor * scale_lut[:, state : state + 1]
        ).clamp_min(1e-20)
        normalized = vectors / denominator[:, :, None, None, None]
        valid_vector = valid_group[:, :, None].expand(
            -1, -1, _VECTORS_PER_GROUP
        ).reshape(batches, -1)
        first_codebooks[:, state] = _batched_kmeans(
            normalized[:, :, :, 0].reshape(batches, -1, _SUBVECTOR_SIZE),
            vector_weights[:, :, :, 0].reshape(
                batches, -1, _SUBVECTOR_SIZE
            ),
            valid_vector,
            iterations=kmeans_iterations,
        )
        second_codebooks[:, state] = _batched_kmeans(
            normalized[:, :, :, 1].reshape(batches, -1, _SUBVECTOR_SIZE),
            vector_weights[:, :, :, 1].reshape(
                batches, -1, _SUBVECTOR_SIZE
            ),
            valid_vector,
            iterations=kmeans_iterations,
        )
    return neuron_scale, _fp16_round(scale_lut), first_codebooks, second_codebooks


def _nearest_subvectors(
    samples: torch.Tensor,
    weights: torch.Tensor,
    codebook: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    table = codebook.to(torch.float32)
    cross = torch.einsum("bgvc,bec->bgve", weights * samples, table)
    sample_norm = (weights * samples.square()).sum(3, keepdim=True)
    table_norm = torch.einsum(
        "bgvc,bec->bgve", weights, table.square()
    )
    scaled = scale[:, :, None, None]
    error = sample_norm + scaled.square() * table_norm - 2.0 * scaled * cross
    minimum, index = error.min(3)
    return index, minimum


def _assign(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    neuron_scale: torch.Tensor,
    scale_lut: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
    *,
    rows: int,
    ng: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batches, groups, _ = xgroup.shape
    vectors = xgroup.reshape(
        batches, groups, _VECTORS_PER_GROUP, 2, _SUBVECTOR_SIZE
    )
    weights = wgroup.reshape_as(vectors)
    group_anchor = (
        neuron_scale[:, :, None].expand(-1, -1, ng).reshape(batches, groups)
    )
    best_error = torch.full(
        (batches, groups), torch.inf, device=xgroup.device, dtype=torch.float32
    )
    best_state = torch.zeros(
        (batches, groups), device=xgroup.device, dtype=torch.int64
    )
    best_first = torch.zeros(
        (batches, groups, _VECTORS_PER_GROUP),
        device=xgroup.device,
        dtype=torch.int64,
    )
    best_second = torch.zeros_like(best_first)
    for state in range(_STATE_COUNT):
        scale = group_anchor * scale_lut[:, state : state + 1]
        first_index, first_error = _nearest_subvectors(
            vectors[:, :, :, 0],
            weights[:, :, :, 0],
            first_codebooks[:, state],
            scale,
        )
        second_index, second_error = _nearest_subvectors(
            vectors[:, :, :, 1],
            weights[:, :, :, 1],
            second_codebooks[:, state],
            scale,
        )
        error = (first_error + second_error).sum(2)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_state = torch.where(
            better, torch.full_like(best_state, state), best_state
        )
        best_first = torch.where(better[:, :, None], first_index, best_first)
        best_second = torch.where(better[:, :, None], second_index, best_second)
    return best_state, best_first, best_second, best_error


def _assigned_code(
    state: torch.Tensor,
    first_index: torch.Tensor,
    second_index: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
) -> torch.Tensor:
    batches = state.shape[0]
    batch = torch.arange(
        batches, device=state.device, dtype=torch.int64
    )[:, None, None]
    state_vector = state[:, :, None]
    first = first_codebooks[batch, state_vector, first_index].to(torch.float32)
    second = second_codebooks[batch, state_vector, second_index].to(torch.float32)
    return torch.cat((first, second), dim=3).reshape(
        batches, state.shape[1], _GROUP_SIZE
    )


def _scatter_table_sum(
    key: torch.Tensor,
    value: torch.Tensor,
    entries: int,
) -> torch.Tensor:
    result = torch.zeros(
        (entries, value.shape[-1]),
        device=value.device,
        dtype=torch.float32,
    )
    result.scatter_add_(
        0,
        key.reshape(-1, 1).expand(-1, value.shape[-1]),
        value.reshape(-1, value.shape[-1]),
    )
    return result


def _refit_anchor_and_lut(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    state: torch.Tensor,
    first_index: torch.Tensor,
    second_index: torch.Tensor,
    neuron_scale: torch.Tensor,
    scale_lut: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
    *,
    rows: int,
    ng: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batches, groups, _ = xgroup.shape
    code = _assigned_code(
        state,
        first_index,
        second_index,
        first_codebooks,
        second_codebooks,
    )
    alpha = torch.gather(scale_lut, 1, state)
    basis = alpha[:, :, None] * code
    numerator = (wgroup * xgroup * basis).reshape(
        batches, rows, ng, _GROUP_SIZE
    ).sum((2, 3))
    denominator = (wgroup * basis.square()).reshape(
        batches, rows, ng, _GROUP_SIZE
    ).sum((2, 3))
    fitted_anchor = _fp16_round(
        torch.where(
            denominator > 0,
            numerator / denominator,
            neuron_scale,
        ).clamp_min(0)
    )

    group_anchor = (
        fitted_anchor[:, :, None].expand(-1, -1, ng).reshape(batches, groups)
    )
    batch_offset = (
        torch.arange(batches, device=xgroup.device, dtype=torch.int64)
        * _STATE_COUNT
    )[:, None]
    key = batch_offset + state
    raw_num = (wgroup * xgroup * code).sum(2) * group_anchor
    raw_den = (wgroup * code.square()).sum(2) * group_anchor.square()
    lut_num = torch.zeros(
        batches * _STATE_COUNT, device=xgroup.device, dtype=torch.float32
    )
    lut_den = torch.zeros_like(lut_num)
    lut_num.scatter_add_(0, key.reshape(-1), raw_num.reshape(-1))
    lut_den.scatter_add_(0, key.reshape(-1), raw_den.reshape(-1))
    previous = scale_lut.reshape(-1)
    fitted_lut = torch.where(
        lut_den > 0, lut_num / lut_den, previous
    ).reshape_as(scale_lut).clamp_min(0)
    maximum = fitted_lut.amax(1).clamp_min(1e-20)
    fitted_lut = fitted_lut / maximum[:, None]
    fitted_anchor = _fp16_round(fitted_anchor * maximum[:, None])
    return fitted_anchor, _fp16_round(fitted_lut)


def _update_codebooks(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    state: torch.Tensor,
    first_index: torch.Tensor,
    second_index: torch.Tensor,
    neuron_scale: torch.Tensor,
    scale_lut: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
    *,
    rows: int,
    ng: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batches, groups, _ = xgroup.shape
    vectors = xgroup.reshape(
        batches, groups, _VECTORS_PER_GROUP, 2, _SUBVECTOR_SIZE
    )
    weights = wgroup.reshape_as(vectors)
    group_anchor = (
        neuron_scale[:, :, None].expand(-1, -1, ng).reshape(batches, groups)
    )
    group_scale = group_anchor * torch.gather(scale_lut, 1, state)
    batch_offset = (
        torch.arange(batches, device=xgroup.device, dtype=torch.int64)
        * (_STATE_COUNT * _ENTRIES)
    )[:, None, None]
    state_offset = state[:, :, None] * _ENTRIES
    entries = batches * _STATE_COUNT * _ENTRIES

    def update(
        samples: torch.Tensor,
        objective: torch.Tensor,
        indices: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        key = batch_offset + state_offset + indices
        scale = group_scale[:, :, None, None]
        numerator = _scatter_table_sum(
            key,
            samples * objective * scale,
            entries,
        )
        denominator = _scatter_table_sum(
            key,
            objective * scale.square(),
            entries,
        )
        old = previous.reshape(entries, _SUBVECTOR_SIZE).to(torch.float32)
        result = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1e-20),
            old,
        )
        return result.round().clamp(-127, 127).to(torch.int8).reshape_as(previous)

    return (
        update(
            vectors[:, :, :, 0],
            weights[:, :, :, 0],
            first_index,
            first_codebooks,
        ).contiguous(),
        update(
            vectors[:, :, :, 1],
            weights[:, :, :, 1],
            second_index,
            second_codebooks,
        ).contiguous(),
    )


def _train_batch(
    value: torch.Tensor,
    config: NepqBankTrainConfig,
    importance: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if value.ndim != 3:
        raise ValueError("NEPQ bank training expects [experts,rows,K]")
    batches, rows, width = (int(item) for item in value.shape)
    if width % 8:
        raise ValueError("NEPQ bank training requires K divisible by 8")
    if importance is None:
        objective = torch.ones_like(value, dtype=torch.float32)
    else:
        objective = importance.to(device=value.device, dtype=torch.float32)
        if objective.shape == (batches, width):
            objective = objective[:, None, :].expand(-1, rows, -1)
        elif objective.shape != value.shape:
            raise ValueError(
                "NEPQ bank importance must have shape [experts,K] or "
                "[experts,rows,K]"
            )
        if not torch.isfinite(objective).all() or torch.any(objective < 0):
            raise ValueError("NEPQ bank importance must be finite and non-negative")
        mean = objective.mean((1, 2), keepdim=True)
        if torch.any(mean <= 0):
            raise ValueError("every NEPQ bank importance item needs positive mass")
        objective = objective / mean
    pad = (-width) % _GROUP_SIZE
    if pad:
        value = torch.nn.functional.pad(value, (0, pad))
        objective = torch.nn.functional.pad(objective, (0, pad))
    ng = value.shape[2] // _GROUP_SIZE
    xgroup = value.to(torch.float32).reshape(
        batches, rows * ng, _GROUP_SIZE
    ).contiguous()
    wgroup = objective.reshape_as(xgroup).contiguous()
    neuron_scale, scale_lut, first_codebooks, second_codebooks = _initialize(
        xgroup,
        wgroup,
        rows=rows,
        ng=ng,
        kmeans_iterations=config.kmeans_iterations,
    )
    signal = (wgroup * xgroup.square()).sum((1, 2))
    best_error = torch.full(
        (batches,), torch.inf, device=value.device, dtype=torch.float32
    )
    best_lut = scale_lut.clone()
    best_first = first_codebooks.clone()
    best_second = second_codebooks.clone()
    for iteration in range(config.iterations + 1):
        state, first_index, second_index, group_error = _assign(
            xgroup,
            wgroup,
            neuron_scale,
            scale_lut,
            first_codebooks,
            second_codebooks,
            rows=rows,
            ng=ng,
        )
        for _ in range(config.assignment_refine_steps):
            neuron_scale, scale_lut = _refit_anchor_and_lut(
                xgroup,
                wgroup,
                state,
                first_index,
                second_index,
                neuron_scale,
                scale_lut,
                first_codebooks,
                second_codebooks,
                rows=rows,
                ng=ng,
            )
            state, first_index, second_index, group_error = _assign(
                xgroup,
                wgroup,
                neuron_scale,
                scale_lut,
                first_codebooks,
                second_codebooks,
                rows=rows,
                ng=ng,
            )
        error = group_error.sum(1)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_lut = torch.where(better[:, None], scale_lut, best_lut)
        best_first = torch.where(
            better[:, None, None, None], first_codebooks, best_first
        )
        best_second = torch.where(
            better[:, None, None, None], second_codebooks, best_second
        )
        if iteration == config.iterations:
            break
        first_codebooks, second_codebooks = _update_codebooks(
            xgroup,
            wgroup,
            state,
            first_index,
            second_index,
            neuron_scale,
            scale_lut,
            first_codebooks,
            second_codebooks,
            rows=rows,
            ng=ng,
        )
    nmse = (
        100.0
        * best_error
        / torch.where(signal > 0, signal, torch.ones_like(signal))
    )
    return (
        np.stack(
            [
                np.frombuffer(
                    pack_npq0_s_tables(
                        best_lut[index].cpu().numpy(),
                        best_first[index].cpu().numpy(),
                        best_second[index].cpu().numpy(),
                    ),
                    dtype=np.uint8,
                )
                for index in range(batches)
            ]
        ),
        nmse.cpu().numpy().astype(np.float64, copy=False),
    )


@torch.inference_mode()
def train_nepq0_s_banks(
    samples: torch.Tensor,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    config: NepqBankTrainConfig | None = None,
    rotation_block: int = 0,
    rotation_seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> NepqBankTraining:
    """Train one NPQ0-S bank per expert while batching experts on one GPU."""

    config = NepqBankTrainConfig() if config is None else config
    if samples.ndim != 3:
        raise ValueError("samples must have shape [experts,rows,K]")
    if not samples.is_cuda:
        raise ValueError("production NEPQ bank training requires CUDA samples")
    experts = int(samples.shape[0])
    objective = None
    if importance is not None:
        objective = torch.as_tensor(
            importance, device=samples.device, dtype=torch.float32
        )
        if objective.shape not in {
            torch.Size((experts, int(samples.shape[2]))),
            samples.shape,
        }:
            raise ValueError(
                "NEPQ bank importance must have shape [experts,K] or "
                "[experts,rows,K]"
            )
        if not torch.isfinite(objective).all() or torch.any(objective < 0):
            raise ValueError("NEPQ bank importance must be finite and non-negative")
    payloads = np.empty((experts, NPQ0_S_TABLE_BYTES), dtype=np.uint8)
    nmse = np.empty(experts, dtype=np.float64)
    for start in range(0, experts, config.expert_batch):
        stop = min(start + config.expert_batch, experts)
        value = samples[start:stop]
        local_objective = None if objective is None else objective[start:stop]
        if rotation_block:
            value = signed_hadamard_rotate(value, rotation_block, rotation_seed)
            if local_objective is not None:
                local_objective = hadamard_diagonal_importance(
                    local_objective, rotation_block
                )
        elif rotation_seed:
            raise ValueError("rotation_seed requires rotation_block")
        table, local_nmse = _train_batch(
            value, config, importance=local_objective
        )
        payloads[start:stop] = table
        nmse[start:stop] = local_nmse
        if progress is not None:
            progress(stop, experts)
    return NepqBankTraining(payloads, nmse)


__all__ = [
    "NepqBankTrainConfig",
    "NepqBankTraining",
    "hadamard_diagonal_importance",
    "signed_hadamard_rotate",
    "train_nepq0_s_banks",
]
