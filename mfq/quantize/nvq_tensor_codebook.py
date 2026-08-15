"""Tensor-wise NVQ codebook training on the real packed quantization path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

from mfq.formats.nvq import (
    NvqSpec,
    NvqTensor,
    codebook_for,
    validate_codebook,
)
from mfq.formats.nvq1_l import (
    IQ1S_TERNARY_2048,
    Nvq1LSpec,
    Nvq1LTensor,
    validate_ternary_codebook,
)
from mfq.quantize.nvq1_l_quant import dequantize as dequantize_nvq1_l
from mfq.quantize.nvq1_l_quant import quantize as quantize_nvq1_l_cpu
from mfq.quantize.nvq_quant import _encode_even_parity_signs
from mfq.quantize.nvq_quant import dequantize as dequantize_nvq
from mfq.quantize.nvq_quant import quantize as quantize_nvq_cpu

NvqAnySpec = NvqSpec | Nvq1LSpec
NvqAnyTensor = NvqTensor | Nvq1LTensor


@dataclass(frozen=True)
class TensorCodebookTrainingConfig:
    iterations: int = 4
    projection_candidates: int = 48
    quant_backend: Literal["cuda", "metal", "cpu"] = "cuda"
    device: str = "cuda"
    group_chunk: int = 32768
    row_chunk: int = 512
    search_steps: int = 19
    nvq1_l_anchor_multipliers: tuple[float, ...] = (0.75,)
    nvq1_l_refine_steps: int = 2
    nvq_native_assignment: bool = True
    nvq1_l_native_assignment: bool = True
    min_validation_improvement: float = 0.0
    initializations: tuple[Literal["builtin", "frequency"], ...] = (
        "builtin",
        "frequency",
    )

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if self.projection_candidates <= 0:
            raise ValueError("projection_candidates must be positive")
        if self.group_chunk <= 0 or self.row_chunk <= 0 or self.search_steps <= 0:
            raise ValueError("group_chunk, row_chunk, and search_steps must be positive")
        if self.quant_backend not in {"cuda", "metal", "cpu"}:
            raise ValueError(f"unsupported quant backend: {self.quant_backend}")
        if self.nvq1_l_refine_steps < 0:
            raise ValueError("nvq1_l_refine_steps must be non-negative")
        if not self.nvq1_l_anchor_multipliers or any(
            not np.isfinite(value) or value <= 0
            for value in self.nvq1_l_anchor_multipliers
        ):
            raise ValueError("NVQ1-L anchor multipliers must be finite and positive")
        if not 0.0 <= self.min_validation_improvement < 1.0:
            raise ValueError("min_validation_improvement must be in [0, 1)")
        if not self.initializations or any(
            value not in {"builtin", "frequency"} for value in self.initializations
        ):
            raise ValueError("initializations must contain builtin and/or frequency")


@dataclass(frozen=True)
class TensorCodebookTrainingResult:
    tensor_name: str
    format_label: str
    trained_codebook: np.ndarray
    selected_codebook: np.ndarray | None
    history: tuple[dict[str, float | int | str], ...]
    train_rows: int
    validation_rows: int
    fixed_validation_sse: float
    trained_validation_sse: float
    fixed_validation_snr_db: float
    trained_validation_snr_db: float

    @property
    def selected_custom(self) -> bool:
        return self.selected_codebook is not None

    @property
    def validation_sse_improvement_percent(self) -> float:
        if self.fixed_validation_sse == 0:
            return 0.0
        return 100.0 * (
            self.fixed_validation_sse - self.trained_validation_sse
        ) / self.fixed_validation_sse


@dataclass(frozen=True)
class _CodebookStats:
    numerator: np.ndarray
    denominator: np.ndarray
    counts: np.ndarray
    signal: float
    sse: float
    elements: int


def _builtin_codebook(spec: NvqAnySpec) -> np.ndarray:
    if isinstance(spec, Nvq1LSpec):
        return np.array(IQ1S_TERNARY_2048, copy=True)
    return np.array(codebook_for(spec), copy=True)


def _validate_training_matrix(weight: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(weight, dtype=np.float32)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError(f"{name} must be a non-empty [rows, in] matrix, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(value)


def _validate_importance(
    importance: np.ndarray | None,
    weight: np.ndarray,
    name: str,
) -> np.ndarray | None:
    if importance is None:
        return None
    value = np.asarray(importance, dtype=np.float32)
    if value.ndim == 1:
        if value.size != weight.shape[1]:
            raise ValueError(
                f"{name} has {value.size} entries, expected {weight.shape[1]}"
            )
    elif value.shape != weight.shape:
        raise ValueError(
            f"{name} must have shape ({weight.shape[1]},) or {weight.shape}, got {value.shape}"
        )
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return np.ascontiguousarray(value)


def _slice_importance(
    importance: np.ndarray | None,
    start: int,
    stop: int,
) -> np.ndarray | None:
    if importance is None or importance.ndim == 1:
        return importance
    return importance[start:stop]


def _quantize(
    weight: np.ndarray,
    spec: NvqAnySpec,
    codebook: np.ndarray | None,
    config: TensorCodebookTrainingConfig,
    importance: np.ndarray | None = None,
) -> NvqAnyTensor:
    if config.quant_backend in {"cuda", "metal"}:
        import torch

        if config.quant_backend == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("tensor-wise NVQ training requested CUDA, but CUDA is unavailable")
        if config.quant_backend == "metal" and not torch.backends.mps.is_available():
            raise RuntimeError("tensor-wise NVQ training requested Metal, but MPS is unavailable")
        from mfq.quantize.nvq_quant_torch import quantize_axis0

        return quantize_axis0(
            torch.from_numpy(weight),
            spec,
            device=config.device,
            importance=importance,
            search_steps=config.search_steps,
            anchor_multipliers=config.nvq1_l_anchor_multipliers,
            refine_steps=config.nvq1_l_refine_steps,
            group_chunk=config.group_chunk,
            nvq1_l_candidates=0,
            codebook=codebook,
            nvq_native_assignment=(
                config.quant_backend in {"cuda", "metal"}
                and config.nvq_native_assignment
            ),
            nvq1_l_native_assignment=(
                config.quant_backend in {"cuda", "metal"}
                and config.nvq1_l_native_assignment
            ),
        )
    if isinstance(spec, Nvq1LSpec):
        return quantize_nvq1_l_cpu(
            weight,
            spec,
            axis=0,
            importance=importance,
            anchor_multipliers=config.nvq1_l_anchor_multipliers,
            refine_steps=config.nvq1_l_refine_steps,
            group_chunk=min(config.group_chunk, 256),
            codebook=codebook,
        )
    return quantize_nvq_cpu(
        weight,
        spec,
        axis=0,
        importance=importance,
        search_steps=config.search_steps,
        group_chunk=min(config.group_chunk, 1024),
        codebook=codebook,
        scale_refine_steps=2,
    )


def _dequantize(tensor: NvqAnyTensor) -> np.ndarray:
    if isinstance(tensor, Nvq1LTensor):
        return dequantize_nvq1_l(tensor)
    return dequantize_nvq(tensor)


def _padded_weight(
    weight: np.ndarray,
    groupsize: int,
    importance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    rows, neuron_len = weight.shape
    ng = math.ceil(neuron_len / groupsize)
    padded_len = ng * groupsize
    value = np.zeros((rows, padded_len), dtype=np.float32)
    objective_weight = np.zeros_like(value)
    value[:, :neuron_len] = weight
    if importance is None:
        objective_weight[:, :neuron_len] = 1.0
    elif importance.ndim == 1:
        objective_weight[:, :neuron_len] = importance
    else:
        objective_weight[:, :neuron_len] = importance
    return value, objective_weight, ng


def _full_indices(
    indices: np.ndarray,
    rows: int,
    vectors_per_row: int,
    dtype: np.dtype,
) -> np.ndarray:
    result = np.zeros((rows, vectors_per_row), dtype=dtype)
    result[:, : indices.shape[1]] = indices
    return result


def _stats_nvq(
    weight: np.ndarray,
    encoded: NvqTensor,
    codebook: np.ndarray,
    importance: np.ndarray | None,
) -> _CodebookStats:
    spec = encoded.spec
    value, objective_weight, ng = _padded_weight(weight, spec.groupsize, importance)
    target, _ = _encode_even_parity_signs(value, objective_weight)
    rows, padded_len = target.shape
    vectors_per_row = padded_len // spec.vector_size
    indices = _full_indices(encoded.indices, rows, vectors_per_row, np.dtype(np.uint8))
    vectors = target.reshape(rows, vectors_per_row, spec.vector_size)
    vector_weight = objective_weight.reshape(vectors.shape)
    group_scale = encoded.neuron_scale[:, None] * encoded.sub_scale.astype(np.float32)
    vector_scale = np.repeat(group_scale, spec.groupsize // spec.vector_size, axis=1)

    flat_index = indices.reshape(-1)
    flat_scale = vector_scale.reshape(-1, 1).astype(np.float64)
    flat_vector = vectors.reshape(-1, spec.vector_size).astype(np.float64)
    flat_weight = vector_weight.reshape(-1, spec.vector_size).astype(np.float64)
    numerator = np.zeros((256, spec.vector_size), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    counts = np.zeros(256, dtype=np.int64)
    np.add.at(numerator, flat_index, flat_weight * flat_scale * flat_vector)
    np.add.at(denominator, flat_index, flat_weight * flat_scale * flat_scale)
    np.add.at(counts, flat_index, np.any(flat_weight > 0, axis=1).astype(np.int64))

    reconstruction = _dequantize(encoded)
    metric_weight = objective_weight[:, : weight.shape[1]]
    signal = float((metric_weight * np.square(weight, dtype=np.float32)).sum(dtype=np.float64))
    sse = float(
        (metric_weight * np.square(weight - reconstruction, dtype=np.float32)).sum(
            dtype=np.float64
        )
    )
    return _CodebookStats(numerator, denominator, counts, signal, sse, weight.size)


def _stats_nvq1_l(
    weight: np.ndarray,
    encoded: Nvq1LTensor,
    codebook: np.ndarray,
    importance: np.ndarray | None,
) -> _CodebookStats:
    spec = encoded.spec
    value, objective_weight, ng = _padded_weight(weight, spec.groupsize, importance)
    rows, padded_len = value.shape
    vectors_per_row = padded_len // spec.vector_size
    indices = _full_indices(encoded.indices, rows, vectors_per_row, np.dtype(np.uint16))
    vectors = value.reshape(rows, vectors_per_row, spec.vector_size)
    vector_weight = objective_weight.reshape(vectors.shape)
    group_scale = encoded.neuron_scale[:, None] * encoded.sub_scale.astype(np.float32)
    vector_scale = np.repeat(group_scale, spec.groupsize // spec.vector_size, axis=1)
    group_delta = np.where(encoded.delta_sign != 0, -spec.delta, spec.delta).astype(np.float32)
    vector_delta = np.repeat(group_delta, spec.groupsize // spec.vector_size, axis=1)

    flat_index = indices.reshape(-1)
    flat_scale = vector_scale.reshape(-1, 1).astype(np.float64)
    flat_delta = vector_delta.reshape(-1, 1).astype(np.float64)
    flat_vector = vectors.reshape(-1, spec.vector_size).astype(np.float64)
    flat_weight = vector_weight.reshape(-1, spec.vector_size).astype(np.float64)
    numerator = np.zeros((2048, spec.vector_size), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    counts = np.zeros(2048, dtype=np.int64)
    centered = flat_vector - flat_scale * flat_delta
    np.add.at(numerator, flat_index, flat_weight * flat_scale * centered)
    np.add.at(denominator, flat_index, flat_weight * flat_scale * flat_scale)
    np.add.at(counts, flat_index, np.any(flat_weight > 0, axis=1).astype(np.int64))

    reconstruction = _dequantize(encoded)
    metric_weight = objective_weight[:, : weight.shape[1]]
    signal = float((metric_weight * np.square(weight, dtype=np.float32)).sum(dtype=np.float64))
    sse = float(
        (metric_weight * np.square(weight - reconstruction, dtype=np.float32)).sum(
            dtype=np.float64
        )
    )
    return _CodebookStats(numerator, denominator, counts, signal, sse, weight.size)


def _collect_stats(
    weight: np.ndarray,
    encoded: NvqAnyTensor,
    codebook: np.ndarray,
    importance: np.ndarray | None,
) -> _CodebookStats:
    if isinstance(encoded, Nvq1LTensor):
        return _stats_nvq1_l(weight, encoded, codebook, importance)
    return _stats_nvq(weight, encoded, codebook, importance)


def _training_stats(
    weight: np.ndarray,
    importance: np.ndarray | None,
    spec: NvqAnySpec,
    codebook: np.ndarray,
    config: TensorCodebookTrainingConfig,
) -> _CodebookStats:
    entries, dims = codebook.shape
    numerator = np.zeros((entries, dims), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    counts = np.zeros(entries, dtype=np.int64)
    signal = 0.0
    sse = 0.0
    elements = 0
    for start in range(0, weight.shape[0], config.row_chunk):
        chunk = weight[start : start + config.row_chunk]
        chunk_importance = _slice_importance(importance, start, start + chunk.shape[0])
        encoded = _quantize(chunk, spec, codebook, config, chunk_importance)
        stats = _collect_stats(chunk, encoded, codebook, chunk_importance)
        numerator += stats.numerator
        denominator += stats.denominator
        counts += stats.counts
        signal += stats.signal
        sse += stats.sse
        elements += stats.elements
    return _CodebookStats(numerator, denominator, counts, signal, sse, elements)


@lru_cache(maxsize=1)
def _ternary_grid() -> np.ndarray:
    ids = np.arange(3**8, dtype=np.int32)
    powers = (3 ** np.arange(8, dtype=np.int32))[None, :]
    return np.ascontiguousarray((ids[:, None] // powers) % 3 - 1, dtype=np.int8)


@lru_cache(maxsize=1)
def _e8_grid() -> np.ndarray:
    ids = np.arange(4**8, dtype=np.uint32)
    shifts = (2 * np.arange(8, dtype=np.uint32))[None, :]
    return np.ascontiguousarray(2 * ((ids[:, None] >> shifts) & 3) + 1, dtype=np.int8)


@lru_cache(maxsize=1)
def _d4_grid() -> np.ndarray:
    ids = np.arange(8**4, dtype=np.int32)
    powers = (8 ** np.arange(4, dtype=np.int32))[None, :]
    return np.ascontiguousarray(2 * ((ids[:, None] // powers) % 8) + 1, dtype=np.int8)


def _legal_grid(spec: NvqAnySpec) -> np.ndarray:
    if isinstance(spec, Nvq1LSpec):
        return _ternary_grid()
    return _e8_grid() if spec.vector_size == 8 else _d4_grid()


def _grid_ids(spec: NvqAnySpec, codebook: np.ndarray) -> np.ndarray:
    value = np.asarray(codebook, dtype=np.int32)
    if isinstance(spec, Nvq1LSpec):
        digits = value + 1
        powers = 3 ** np.arange(8, dtype=np.int32)
    elif spec.vector_size == 8:
        digits = (value - 1) // 2
        powers = 4 ** np.arange(8, dtype=np.int32)
    else:
        digits = (value - 1) // 2
        powers = 8 ** np.arange(4, dtype=np.int32)
    return (digits * powers[None, :]).sum(axis=1, dtype=np.int64)


def _normalized_code_targets(
    weight: np.ndarray,
    encoded: NvqAnyTensor,
    importance: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    spec = encoded.spec
    value, objective_weight, _ng = _padded_weight(weight, spec.groupsize, importance)
    rows, padded_len = value.shape
    vectors_per_row = padded_len // spec.vector_size
    if isinstance(encoded, Nvq1LTensor):
        vectors = value.reshape(rows, vectors_per_row, spec.vector_size)
        group_delta = np.where(
            encoded.delta_sign != 0,
            -spec.delta,
            spec.delta,
        ).astype(np.float32)
        vector_delta = np.repeat(
            group_delta,
            spec.groupsize // spec.vector_size,
            axis=1,
        )
    else:
        target, _ = _encode_even_parity_signs(value, objective_weight)
        vectors = target.reshape(rows, vectors_per_row, spec.vector_size)
        vector_delta = np.zeros((rows, vectors_per_row), dtype=np.float32)
    vector_weight = objective_weight.reshape(vectors.shape)
    group_scale = encoded.neuron_scale[:, None] * encoded.sub_scale.astype(np.float32)
    vector_scale = np.repeat(
        group_scale,
        spec.groupsize // spec.vector_size,
        axis=1,
    )
    flat_scale = vector_scale.reshape(-1)
    flat_weight = vector_weight.reshape(-1, spec.vector_size)
    valid = (flat_scale > 0) & np.any(flat_weight > 0, axis=1)
    normalized = (
        vectors.reshape(-1, spec.vector_size)[valid]
        / flat_scale[valid, None]
        - vector_delta.reshape(-1)[valid, None]
    )
    sensitivity = (
        np.square(flat_scale[valid], dtype=np.float32)
        * flat_weight[valid].sum(axis=1)
    )
    return normalized, sensitivity


def _frequency_initial_codebook(
    weight: np.ndarray,
    importance: np.ndarray | None,
    spec: NvqAnySpec,
    previous: np.ndarray,
    config: TensorCodebookTrainingConfig,
) -> np.ndarray:
    grid = _legal_grid(spec)
    score = np.zeros(grid.shape[0], dtype=np.float64)
    for start in range(0, weight.shape[0], config.row_chunk):
        chunk = weight[start : start + config.row_chunk]
        chunk_importance = _slice_importance(importance, start, start + chunk.shape[0])
        encoded = _quantize(chunk, spec, None, config, chunk_importance)
        normalized, sensitivity = _normalized_code_targets(
            chunk, encoded, chunk_importance
        )
        if isinstance(spec, Nvq1LSpec):
            digits = np.clip(np.rint(normalized), -1, 1).astype(np.int32) + 1
            powers = 3 ** np.arange(8, dtype=np.int32)
        elif spec.vector_size == 8:
            digits = np.clip(np.rint((normalized - 1.0) / 2.0), 0, 3).astype(np.int32)
            powers = 4 ** np.arange(8, dtype=np.int32)
        else:
            digits = np.clip(np.rint((normalized - 1.0) / 2.0), 0, 7).astype(np.int32)
            powers = 8 ** np.arange(4, dtype=np.int32)
        ids = (digits * powers[None, :]).sum(axis=1, dtype=np.int64)
        score += np.bincount(ids, weights=sensitivity, minlength=grid.shape[0])
    ranked = np.argsort(score, kind="stable")[::-1]
    entries = previous.shape[0]
    selected = [int(value) for value in ranked[:entries] if score[value] > 0]
    selected_set = set(selected)
    for value in _grid_ids(spec, previous):
        index = int(value)
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
            if len(selected) == entries:
                break
    if len(selected) < entries:
        for index in range(grid.shape[0]):
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
                if len(selected) == entries:
                    break
    result = np.ascontiguousarray(grid[np.asarray(selected[:entries])], dtype=np.int8)
    if isinstance(spec, Nvq1LSpec):
        return validate_ternary_codebook(result)
    return validate_codebook(spec, result)


def _project_unique(
    spec: NvqAnySpec,
    previous: np.ndarray,
    stats: _CodebookStats,
    candidate_count: int,
) -> np.ndarray:
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    except ImportError as exc:
        raise RuntimeError("tensor-wise NVQ codebook training requires scipy") from exc

    grid = _legal_grid(spec).astype(np.float64)
    entries, dims = previous.shape
    if stats.numerator.shape != (entries, dims):
        raise ValueError("codebook statistics do not match the current table")
    previous_ids = _grid_ids(spec, previous)
    rows_parts: list[np.ndarray] = []
    columns_parts: list[np.ndarray] = []
    costs_parts: list[np.ndarray] = []
    take = min(candidate_count, grid.shape[0])

    for index in range(entries):
        denominator = stats.denominator[index]
        if not np.any(denominator > 0):
            centroid = previous[index].astype(np.float64)
            weight = np.ones(dims, dtype=np.float64)
        else:
            weight = np.where(denominator > 0, denominator, 0.0)
            centroid = np.divide(
                stats.numerator[index],
                denominator,
                out=previous[index].astype(np.float64),
                where=denominator > 0,
            )
        cost = np.sum(weight[None, :] * np.square(grid - centroid[None, :]), axis=1)
        nearest = np.argpartition(cost, take - 1)[:take]
        candidates = np.unique(np.append(nearest, previous_ids[index])).astype(np.int32)
        rows_parts.append(np.full(candidates.size, index, dtype=np.int32))
        columns_parts.append(candidates)
        costs_parts.append(cost[candidates] + np.finfo(np.float64).tiny)

    graph = csr_matrix(
        (
            np.concatenate(costs_parts),
            (np.concatenate(rows_parts), np.concatenate(columns_parts)),
        ),
        shape=(entries, grid.shape[0]),
    )
    row_ind, column_ind = min_weight_full_bipartite_matching(graph)
    if row_ind.size != entries:
        raise RuntimeError("failed to assign one unique legal point to every NVQ codeword")
    result = np.empty_like(previous)
    result[row_ind] = grid[column_ind].astype(np.int8)
    if isinstance(spec, Nvq1LSpec):
        return validate_ternary_codebook(result)
    return validate_codebook(spec, result)


def _metric(
    weight: np.ndarray,
    encoded: NvqAnyTensor,
    importance: np.ndarray | None,
) -> tuple[float, float]:
    reconstruction = _dequantize(encoded)
    if importance is None:
        objective_weight = np.ones_like(weight)
    elif importance.ndim == 1:
        objective_weight = np.broadcast_to(importance, weight.shape)
    else:
        objective_weight = importance
    signal = float(
        (objective_weight * np.square(weight, dtype=np.float32)).sum(dtype=np.float64)
    )
    sse = float(
        (objective_weight * np.square(weight - reconstruction, dtype=np.float32)).sum(
            dtype=np.float64
        )
    )
    snr = math.inf if sse == 0 else 10.0 * math.log10(signal / sse)
    return sse, snr


def _evaluate(
    weight: np.ndarray,
    importance: np.ndarray | None,
    spec: NvqAnySpec,
    codebook: np.ndarray | None,
    config: TensorCodebookTrainingConfig,
) -> tuple[float, float]:
    signal = 0.0
    sse = 0.0
    for start in range(0, weight.shape[0], config.row_chunk):
        chunk = weight[start : start + config.row_chunk]
        chunk_importance = _slice_importance(importance, start, start + chunk.shape[0])
        encoded = _quantize(chunk, spec, codebook, config, chunk_importance)
        chunk_sse, _chunk_snr = _metric(chunk, encoded, chunk_importance)
        if chunk_importance is None:
            objective_weight = np.ones_like(chunk)
        elif chunk_importance.ndim == 1:
            objective_weight = np.broadcast_to(chunk_importance, chunk.shape)
        else:
            objective_weight = chunk_importance
        signal += float(
            (objective_weight * np.square(chunk, dtype=np.float32)).sum(dtype=np.float64)
        )
        sse += chunk_sse
    snr = math.inf if sse == 0 else 10.0 * math.log10(signal / sse)
    return sse, snr


def train_tensor_codebook(
    tensor_name: str,
    train_weight: np.ndarray,
    validation_weight: np.ndarray,
    spec: NvqAnySpec,
    config: TensorCodebookTrainingConfig = TensorCodebookTrainingConfig(),
    *,
    train_importance: np.ndarray | None = None,
    validation_importance: np.ndarray | None = None,
) -> TensorCodebookTrainingResult:
    """Train one codebook and select it only when held-out rows beat the fixed table."""

    train = _validate_training_matrix(train_weight, "train_weight")
    validation = _validate_training_matrix(validation_weight, "validation_weight")
    if train.shape[1] != validation.shape[1]:
        raise ValueError("training and validation matrices must have the same input width")
    train_objective = _validate_importance(
        train_importance, train, "train_importance"
    )
    validation_objective = _validate_importance(
        validation_importance, validation, "validation_importance"
    )

    fixed_sse, fixed_snr = _evaluate(
        validation, validation_objective, spec, None, config
    )
    builtin = _builtin_codebook(spec)
    starts: list[tuple[str, np.ndarray]] = []
    if "builtin" in config.initializations:
        starts.append(("builtin", builtin.copy()))
    if "frequency" in config.initializations:
        frequency = _frequency_initial_codebook(
            train,
            train_objective,
            spec,
            builtin,
            config,
        )
        if not any(np.array_equal(frequency, table) for _name, table in starts):
            starts.append(("frequency", frequency))

    history: list[dict[str, float | int | str]] = []
    candidate_tables: list[np.ndarray] = []
    for start_name, initial in starts:
        table = initial
        best_table = table.copy()
        best_sse = math.inf
        for iteration in range(config.iterations + 1):
            stats = _training_stats(train, train_objective, spec, table, config)
            snr = (
                math.inf
                if stats.sse == 0
                else 10.0 * math.log10(stats.signal / stats.sse)
            )
            history.append(
                {
                    "start": start_name,
                    "iteration": iteration,
                    "sse": stats.sse,
                    "snr_db": snr,
                    "used_codes": int(np.count_nonzero(stats.counts)),
                }
            )
            if stats.sse < best_sse:
                best_sse = stats.sse
                best_table = table.copy()
            if iteration == config.iterations:
                break
            updated = _project_unique(
                spec,
                table,
                stats,
                config.projection_candidates,
            )
            if np.array_equal(updated, table):
                break
            table = updated
        candidate_tables.append(best_table)

    best_table = builtin.copy()
    trained_sse = fixed_sse
    trained_snr = fixed_snr
    for table in candidate_tables:
        candidate_sse, candidate_snr = _evaluate(
            validation, validation_objective, spec, table, config
        )
        if candidate_sse < trained_sse:
            best_table = table
            trained_sse = candidate_sse
            trained_snr = candidate_snr
    required = fixed_sse * (1.0 - config.min_validation_improvement)
    selected = best_table.copy() if trained_sse < required else None
    return TensorCodebookTrainingResult(
        tensor_name=tensor_name,
        format_label=spec.label,
        trained_codebook=best_table,
        selected_codebook=selected,
        history=tuple(history),
        train_rows=train.shape[0],
        validation_rows=validation.shape[0],
        fixed_validation_sse=fixed_sse,
        trained_validation_sse=trained_sse,
        fixed_validation_snr_db=fixed_snr,
        trained_validation_snr_db=trained_snr,
    )
