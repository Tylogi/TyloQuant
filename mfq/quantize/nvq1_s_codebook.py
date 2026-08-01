"""Offline 512-entry ternary codebook training for NVQ1-S.

The trainer alternates the real NVQ1-S assignment path with a weighted centroid
update. Centroids are globally projected onto unique points from the complete
3^8 ternary grid.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from mfq.formats.nvq1_l import IQ1S_TERNARY_2048
from mfq.formats.nvq1_s import NVQ1_S_BOOTSTRAP_512, validate_nvq1_s_codebook
from mfq.quantize.nvq1_l_quant import _importance_matrix
from mfq.quantize.nvq1_l_quant import quantize as quantize_full
from mfq.quantize.nvq1_s_quant import quantize

ProgressCallback = Callable[[dict[str, float | int]], None]


@dataclass(frozen=True)
class Nvq1STrainingMatrix:
    name: str
    weight: np.ndarray
    importance: np.ndarray | None = None

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32)
        if weight.ndim != 2 or weight.shape[1] % 8:
            raise ValueError(
                f"NVQ1-S training expects [out, in] with in divisible by 8, got {weight.shape}"
            )
        if not np.isfinite(weight).all():
            raise ValueError(f"training matrix {self.name} contains non-finite values")
        object.__setattr__(self, "weight", np.ascontiguousarray(weight))
        if self.importance is not None:
            importance = np.asarray(self.importance, dtype=np.float32)
            if importance.ndim == 1 and importance.size != weight.shape[1]:
                raise ValueError(f"importance length mismatch for {self.name}")
            if importance.ndim != 1 and importance.shape != weight.shape:
                raise ValueError(f"importance shape mismatch for {self.name}")
            if not np.isfinite(importance).all() or np.any(importance < 0):
                raise ValueError("importance must be finite and non-negative")
            object.__setattr__(self, "importance", np.ascontiguousarray(importance))


@dataclass(frozen=True)
class Nvq1SCodebookTrainingConfig:
    iterations: int = 6
    anchor_multipliers: tuple[float, ...] = (0.75, 1.0, 1.25)
    refine_steps: int = 2
    group_chunk: int = 64
    projection_candidates: int = 24
    reseed_pool_size: int = 2048
    min_relative_improvement: float = 1e-5

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.refine_steps < 0:
            raise ValueError("iterations and refine_steps must be non-negative")
        if self.group_chunk <= 0 or self.projection_candidates <= 0:
            raise ValueError("group_chunk and projection_candidates must be positive")
        if self.reseed_pool_size < 0:
            raise ValueError("reseed_pool_size must be non-negative")
        if not self.anchor_multipliers or any(
            not np.isfinite(value) or value <= 0 for value in self.anchor_multipliers
        ):
            raise ValueError("anchor_multipliers must be finite and positive")
        if not 0 <= self.min_relative_improvement < 1:
            raise ValueError("min_relative_improvement must be in [0, 1)")


@dataclass
class _CodebookStats:
    numerator: np.ndarray
    denominator: np.ndarray
    counts: np.ndarray
    signal: float
    sse: float
    elements: int
    reseed_vectors: np.ndarray
    reseed_errors: np.ndarray


@dataclass(frozen=True)
class _FullSearchSpec:
    delta: float
    groupsize: int = 24
    sub_bits: int = 4
    vector_size: int = 8
    index_bits: int = 11


@lru_cache(maxsize=1)
def _ternary_grid() -> np.ndarray:
    encoded = np.arange(3**8, dtype=np.int32)
    powers = (3 ** np.arange(8, dtype=np.int32))[None, :]
    digits = (encoded[:, None] // powers) % 3
    return np.ascontiguousarray(digits - 1, dtype=np.int8)


def _encode_ternary(table: np.ndarray) -> np.ndarray:
    digits = table.astype(np.int32) + 1
    powers = 3 ** np.arange(8, dtype=np.int32)
    return np.sum(digits * powers[None, :], axis=1)


def initialize_nvq1_s_banks_from_full(
    matrices: Sequence[Nvq1STrainingMatrix],
    *,
    delta: float = 0.15625,
    anchor_multipliers: tuple[float, ...] = (0.75, 1.0, 1.25),
    refine_steps: int = 2,
    group_chunk: int = 64,
) -> np.ndarray:
    """Build two 512-entry banks from full 2048-table assignments."""

    if not matrices:
        raise ValueError("at least one NVQ1-S training matrix is required")
    counts = np.zeros((2, 3**8), dtype=np.float64)
    powers = 3 ** np.arange(8, dtype=np.int64)
    spec = _FullSearchSpec(delta=delta)

    for matrix in matrices:
        encoded = quantize_full(
            matrix.weight,
            spec,
            importance=matrix.importance,
            codebook=IQ1S_TERNARY_2048,
            anchor_multipliers=anchor_multipliers,
            refine_steps=refine_steps,
            group_chunk=group_chunk,
        )
        out, neuron_len = matrix.weight.shape
        nvec = neuron_len // 8
        scale = np.repeat(
            encoded.neuron_scale[:, None] * encoded.sub_scale.astype(np.float32),
            3,
            axis=1,
        )[:, :nvec]
        bank = np.repeat(encoded.delta_sign, 3, axis=1)[:, :nvec]
        shift = np.where(bank != 0, -delta, delta).astype(np.float32)
        vectors = matrix.weight.reshape(out, nvec, 8)
        normalized = np.zeros_like(vectors)
        nonzero = scale > 0
        normalized[nonzero] = (
            vectors[nonzero] / scale[nonzero, None] - shift[nonzero, None]
        )
        pattern = np.clip(np.rint(normalized), -1, 1).astype(np.int8)
        pattern_id = np.sum(
            (pattern.astype(np.int64) + 1) * powers[None, None, :],
            axis=-1,
        )
        objective_weight = _importance_matrix(
            matrix.importance,
            matrix.weight.shape,
            0,
            out,
            neuron_len,
        ).reshape(out, nvec, 8)
        vector_weight = (objective_weight * scale[..., None] ** 2).sum(axis=-1)
        for bank_id in (0, 1):
            selected = bank == bank_id
            counts[bank_id] += np.bincount(
                pattern_id[selected],
                weights=vector_weight[selected],
                minlength=3**8,
            )

    tables = []
    for bank_id in (0, 1):
        selected = np.argsort(counts[bank_id])[-512:]
        digits = (selected[:, None] // powers[None, :]) % 3
        tables.append(validate_nvq1_s_codebook(digits.astype(np.int8) - 1))
    return np.stack(tables, axis=0)


def _collect_stats(
    matrices: Sequence[Nvq1STrainingMatrix],
    table: np.ndarray,
    config: Nvq1SCodebookTrainingConfig,
) -> _CodebookStats:
    codebook = validate_nvq1_s_codebook(table).astype(np.float32)
    numerator = np.zeros((512, 8), dtype=np.float64)
    denominator = np.zeros((512, 8), dtype=np.float64)
    counts = np.zeros(512, dtype=np.int64)
    signal = 0.0
    sse = 0.0
    elements = 0
    pool_vectors: list[np.ndarray] = []
    pool_errors: list[np.ndarray] = []

    for matrix in matrices:
        encoded = quantize(
            matrix.weight,
            importance=matrix.importance,
            codebook=codebook,
            anchor_multipliers=config.anchor_multipliers,
            refine_steps=config.refine_steps,
            group_chunk=config.group_chunk,
        )
        weight = matrix.weight
        out, neuron_len = weight.shape
        objective_weight = _importance_matrix(
            matrix.importance,
            weight.shape,
            0,
            out,
            neuron_len,
        )
        ng = math.ceil(neuron_len / 24)
        padded_len = ng * 24
        if padded_len != neuron_len:
            tail = padded_len - neuron_len
            weight = np.pad(weight, ((0, 0), (0, tail)))
            objective_weight = np.pad(objective_weight, ((0, 0), (0, tail)))

        vectors_per_row = padded_len // 8
        valid_vectors = neuron_len // 8
        full_indices = np.zeros((out, vectors_per_row), dtype=np.uint16)
        full_indices[:, :valid_vectors] = encoded.indices
        vectors = weight.reshape(out, vectors_per_row, 8)
        vector_weight = objective_weight.reshape(out, vectors_per_row, 8)
        group_scale = encoded.neuron_scale[:, None] * encoded.sub_scale.astype(np.float32)
        vector_scale = np.repeat(group_scale, 3, axis=1)
        vector_delta = np.repeat(
            np.where(encoded.delta_sign != 0, -encoded.spec.delta, encoded.spec.delta),
            3,
            axis=1,
        ).astype(np.float32)
        reconstruction = vector_scale[..., None] * (
            codebook[full_indices] + vector_delta[..., None]
        )
        error = vector_weight * (vectors - reconstruction) ** 2
        vector_error = error.sum(axis=-1)

        flat_indices = full_indices.reshape(-1)
        flat_scale = vector_scale.reshape(-1, 1).astype(np.float64)
        flat_delta = vector_delta.reshape(-1, 1).astype(np.float64)
        flat_vectors = vectors.reshape(-1, 8).astype(np.float64)
        flat_weight = vector_weight.reshape(-1, 8).astype(np.float64)
        adjusted = flat_vectors - flat_scale * flat_delta
        np.add.at(numerator, flat_indices, flat_weight * flat_scale * adjusted)
        np.add.at(denominator, flat_indices, flat_weight * flat_scale * flat_scale)
        np.add.at(counts, flat_indices, np.any(flat_weight > 0, axis=1).astype(np.int64))
        signal += float(np.sum(flat_weight * flat_vectors * flat_vectors))
        sse += float(np.sum(error, dtype=np.float64))
        elements += int(np.count_nonzero(flat_weight))

        if config.reseed_pool_size:
            valid_scale = vector_scale.reshape(-1)
            normalized = np.zeros_like(flat_vectors, dtype=np.float32)
            nonzero = valid_scale > 0
            normalized[nonzero] = (
                flat_vectors[nonzero] / valid_scale[nonzero, None]
                - flat_delta[nonzero]
            ).astype(np.float32)
            take = min(config.reseed_pool_size, vector_error.size)
            top = np.argpartition(vector_error.reshape(-1), -take)[-take:]
            pool_vectors.append(np.clip(normalized[top], -1.0, 1.0))
            pool_errors.append(vector_error.reshape(-1)[top].astype(np.float64))

    if pool_errors:
        errors = np.concatenate(pool_errors)
        vectors = np.concatenate(pool_vectors)
        keep = min(config.reseed_pool_size, errors.size)
        selected = np.argpartition(errors, -keep)[-keep:]
        order = selected[np.argsort(errors[selected])[::-1]]
        reseed_vectors = vectors[order]
        reseed_errors = errors[order]
    else:
        reseed_vectors = np.empty((0, 8), dtype=np.float32)
        reseed_errors = np.empty(0, dtype=np.float64)

    return _CodebookStats(
        numerator=numerator,
        denominator=denominator,
        counts=counts,
        signal=signal,
        sse=sse,
        elements=elements,
        reseed_vectors=reseed_vectors,
        reseed_errors=reseed_errors,
    )


def _project_unique(
    centroids: np.ndarray,
    weights: np.ndarray,
    previous: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("NVQ1-S codebook training requires scipy") from exc

    grid = _ternary_grid().astype(np.float64)
    candidate_ids: list[np.ndarray] = []
    take = min(candidate_count, grid.shape[0])
    for centroid, weight in zip(centroids, weights):
        cost = np.sum(weight[None, :] * (grid - centroid[None, :]) ** 2, axis=1)
        candidate_ids.append(np.argpartition(cost, take - 1)[:take])
    candidate_ids.append(_encode_ternary(previous))
    union = np.unique(np.concatenate(candidate_ids))
    if union.size < 512:
        missing = 512 - union.size
        extra = np.setdiff1d(np.arange(grid.shape[0]), union, assume_unique=True)[:missing]
        union = np.concatenate([union, extra])

    candidate = grid[union]
    weighted_centroid = weights * centroids
    constant = np.sum(weights * centroids * centroids, axis=1)
    cost = (
        (candidate * candidate) @ weights.T
        - 2.0 * candidate @ weighted_centroid.T
        + constant[None, :]
    ).T
    rows, columns = linear_sum_assignment(cost)
    if rows.size != 512:
        raise RuntimeError("failed to assign 512 unique ternary codewords")
    result = np.empty((512, 8), dtype=np.int8)
    result[rows] = candidate[columns].astype(np.int8)
    return validate_nvq1_s_codebook(result)


def _updated_codebook(
    previous: np.ndarray,
    stats: _CodebookStats,
    config: Nvq1SCodebookTrainingConfig,
) -> np.ndarray:
    denominator = stats.denominator.copy()
    centroids = np.divide(
        stats.numerator,
        denominator,
        out=previous.astype(np.float64),
        where=denominator > 0,
    )
    empty = np.flatnonzero(stats.counts == 0)
    for position, code in enumerate(empty):
        if position >= stats.reseed_vectors.shape[0]:
            break
        centroids[code] = stats.reseed_vectors[position]
        denominator[code] = 1.0
    return _project_unique(
        centroids,
        denominator,
        previous,
        config.projection_candidates,
    )


def train_nvq1_s_codebook(
    matrices: Sequence[Nvq1STrainingMatrix],
    config: Nvq1SCodebookTrainingConfig = Nvq1SCodebookTrainingConfig(),
    *,
    initial: np.ndarray = NVQ1_S_BOOTSTRAP_512,
    progress: ProgressCallback | None = None,
) -> tuple[np.ndarray, tuple[dict[str, float | int], ...]]:
    if not matrices:
        raise ValueError("at least one NVQ1-S training matrix is required")
    table = validate_nvq1_s_codebook(initial)
    best_table = table.copy()
    best_nmse = math.inf
    previous_nmse = math.inf
    history: list[dict[str, float | int]] = []

    for iteration in range(config.iterations + 1):
        stats = _collect_stats(matrices, table, config)
        nmse = 100.0 * stats.sse / stats.signal if stats.signal else 0.0
        snr = 10.0 * math.log10(stats.signal / stats.sse) if stats.sse else math.inf
        row: dict[str, float | int] = {
            "iteration": iteration,
            "sse": stats.sse,
            "nmse_percent": nmse,
            "snr_db": snr,
            "used_codes": int(np.count_nonzero(stats.counts)),
            "elements": stats.elements,
        }
        history.append(row)
        if progress is not None:
            progress(dict(row))
        if nmse < best_nmse:
            best_nmse = nmse
            best_table = table.copy()
        if iteration == config.iterations:
            break
        if np.isfinite(previous_nmse):
            improvement = (previous_nmse - nmse) / max(previous_nmse, 1e-30)
            if 0 <= improvement < config.min_relative_improvement:
                break
        updated = _updated_codebook(table, stats, config)
        if np.array_equal(updated, table):
            break
        previous_nmse = nmse
        table = updated

    return best_table, tuple(history)
