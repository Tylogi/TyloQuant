"""Offline model-specific codebook training for NVQ2.

The trainer keeps the runtime-friendly E8 alphabet: every codeword has eight
odd int8 coordinates in {1, 3, 5, 7}. It alternates the real NVQ2 assignment
path with a weighted centroid update, then globally projects centroids onto
unique legal grid points. This module is an offline training path and makes no
runtime throughput claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np

from mfq.formats.nvq import E8_256, NVQ2_E8, NvqSpec
from mfq.quantize.nvq_quant import (
    _encode_even_parity_signs,
    _encode_index_parity_signs,
    _importance_matrix,
    quantize,
)


CodebookScope = Literal["model", "family", "tensor"]
ProgressCallback = Callable[[dict[str, object]], None]

_ARTIFACT_FORMAT = "mfq.nvq2.codebooks"
_LEGACY_ARTIFACT_FORMAT = "mfq.niq2.codebooks"
_ARTIFACT_VERSION = 1
_E8_LEVELS = np.asarray([1, 3, 5, 7], dtype=np.int8)


@dataclass(frozen=True)
class NvqTrainingMatrix:
    name: str
    weight: np.ndarray
    importance: np.ndarray | None = None

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError(f"NVQ codebook training expects [out, in], got {weight.shape}")
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
class NvqCodebookTrainingConfig:
    scope: CodebookScope = "model"
    sign_mode: Literal["even", "index_parity"] = "even"
    iterations: int = 8
    search_steps: int = 19
    scale_refine_steps: int = 3
    group_chunk: int = 512
    projection_candidates: int = 48
    reseed_pool_size: int = 2048
    seed: int = 20260716
    min_relative_improvement: float = 1e-5

    def __post_init__(self) -> None:
        if self.scope not in {"model", "family", "tensor"}:
            raise ValueError(f"unsupported codebook scope: {self.scope}")
        if self.sign_mode not in {"even", "index_parity"}:
            raise ValueError(f"unsupported sign mode: {self.sign_mode}")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        for name in ("search_steps", "group_chunk", "projection_candidates"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.scale_refine_steps < 0 or self.reseed_pool_size < 0:
            raise ValueError("scale_refine_steps and reseed_pool_size must be non-negative")
        if not 0 <= self.min_relative_improvement < 1:
            raise ValueError("min_relative_improvement must be in [0, 1)")


@dataclass(frozen=True)
class TrainedNvqCodebook:
    key: str
    tensor_names: tuple[str, ...]
    table: np.ndarray
    history: tuple[dict[str, float | int], ...]
    train_elements: int
    train_signal: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "table", validate_e8_codebook(self.table))


@dataclass(frozen=True)
class NvqCodebookArtifact:
    source_model: str
    config: NvqCodebookTrainingConfig
    codebooks: tuple[TrainedNvqCodebook, ...]
    source_rows: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class _CodebookStats:
    numerator: np.ndarray
    denominator: np.ndarray
    counts: np.ndarray
    signal: float
    sse: float
    elements: int
    reseed_vectors: np.ndarray
    reseed_banks: np.ndarray
    reseed_errors: np.ndarray


def validate_e8_codebook(table: np.ndarray) -> np.ndarray:
    value = np.asarray(table)
    if value.shape != (256, 8):
        raise ValueError(f"E8 codebook must have shape (256, 8), got {value.shape}")
    rounded = np.rint(value)
    if not np.array_equal(value, rounded):
        raise ValueError("E8 codebook must contain integer values")
    result = np.ascontiguousarray(rounded, dtype=np.int8)
    if not np.isin(result, _E8_LEVELS).all():
        raise ValueError("E8 codebook values must be in {1, 3, 5, 7}")
    return result


def pack_e8_codebook(table: np.ndarray) -> bytes:
    value = validate_e8_codebook(table)
    digits = ((value.astype(np.uint16) - 1) // 2).astype(np.uint16)
    shifts = (2 * np.arange(8, dtype=np.uint16))[None, :]
    packed = np.bitwise_or.reduce(digits << shifts, axis=1).astype("<u2")
    return packed.tobytes()


def unpack_e8_codebook(payload: bytes) -> np.ndarray:
    if len(payload) != 512:
        raise ValueError(f"packed E8 codebook must be 512 bytes, got {len(payload)}")
    packed = np.frombuffer(payload, dtype="<u2")
    shifts = (2 * np.arange(8, dtype=np.uint16))[None, :]
    digits = (packed[:, None] >> shifts) & 3
    return np.ascontiguousarray(2 * digits + 1, dtype=np.int8)


def infer_tensor_family(name: str) -> str:
    lower = name.lower()
    if ".mlp.gate_proj." in lower or ".mlp.up_proj." in lower:
        return "ffn_gate_up"
    if ".mlp.down_proj." in lower:
        return "ffn_down"
    if ".linear_attn.in_proj_qkv." in lower:
        return "linear_attn_qkv"
    if ".linear_attn.out_proj." in lower:
        return "linear_attn_out"
    if ".self_attn.q_proj." in lower or ".self_attn.k_proj." in lower:
        return "full_attn_qk"
    if ".self_attn.v_proj." in lower:
        return "full_attn_v"
    if ".self_attn.o_proj." in lower:
        return "full_attn_o"
    return "other_linear"


def codebook_scope_key(tensor_name: str, scope: CodebookScope) -> str:
    if scope == "model":
        return "model"
    if scope == "family":
        return infer_tensor_family(tensor_name)
    if scope == "tensor":
        return tensor_name
    raise ValueError(f"unsupported codebook scope: {scope}")


def _spec_for(sign_mode: str) -> NvqSpec:
    return NvqSpec(
        NVQ2_E8.codebook,
        groupsize=NVQ2_E8.groupsize,
        sub_bits=NVQ2_E8.sub_bits,
        sign_mode=sign_mode,
    )


def _farthest_128(table: np.ndarray) -> np.ndarray:
    value = table.astype(np.float32)
    unit = value / np.linalg.norm(value, axis=1, keepdims=True)
    selected = [int(np.argmin(np.linalg.norm(unit - unit.mean(axis=0), axis=1)))]
    min_distance = np.full(value.shape[0], np.inf, dtype=np.float32)
    for _ in range(1, 128):
        delta = unit - unit[selected[-1]]
        min_distance = np.minimum(min_distance, np.sum(delta * delta, axis=1))
        min_distance[selected] = -1.0
        selected.append(int(np.argmax(min_distance)))
    return np.asarray(selected, dtype=np.int64)


def initial_e8_codebook(sign_mode: str) -> np.ndarray:
    if sign_mode == "even":
        return E8_256.copy()
    if sign_mode == "index_parity":
        half = E8_256[_farthest_128(E8_256)]
        return np.concatenate([half, half], axis=0).astype(np.int8)
    raise ValueError(f"unsupported sign mode: {sign_mode}")


def _padded_training_view(
    matrix: NvqTrainingMatrix,
    spec: NvqSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, int]:
    weight = matrix.weight
    out, neuron_len = weight.shape
    objective_weight = _importance_matrix(
        matrix.importance,
        weight.shape,
        0,
        out,
        neuron_len,
    )
    ng = math.ceil(neuron_len / spec.groupsize)
    padded_len = ng * spec.groupsize
    if padded_len != neuron_len:
        pad = padded_len - neuron_len
        weight = np.pad(weight, ((0, 0), (0, pad)))
        objective_weight = np.pad(objective_weight, ((0, 0), (0, pad)))
    if spec.sign_mode == "index_parity":
        target, _, banks = _encode_index_parity_signs(weight)
    else:
        target, _ = _encode_even_parity_signs(weight, objective_weight)
        banks = None
    return target, objective_weight, banks, neuron_len, ng


def _collect_stats(
    matrices: Sequence[NvqTrainingMatrix],
    table: np.ndarray,
    config: NvqCodebookTrainingConfig,
) -> _CodebookStats:
    spec = _spec_for(config.sign_mode)
    codebook = validate_e8_codebook(table).astype(np.float32)
    numerator = np.zeros((256, 8), dtype=np.float64)
    denominator = np.zeros((256, 8), dtype=np.float64)
    counts = np.zeros(256, dtype=np.int64)
    signal = 0.0
    sse = 0.0
    elements = 0
    pool_vectors: list[np.ndarray] = []
    pool_banks: list[np.ndarray] = []
    pool_errors: list[np.ndarray] = []

    for matrix in matrices:
        encoded = quantize(
            matrix.weight,
            spec,
            importance=matrix.importance,
            search_steps=config.search_steps,
            group_chunk=config.group_chunk,
            codebook=codebook,
            scale_refine_steps=config.scale_refine_steps,
        )
        target, objective_weight, banks, neuron_len, ng = _padded_training_view(matrix, spec)
        out, padded_len = target.shape
        vectors_per_row = padded_len // 8
        valid_vectors = math.ceil(neuron_len / 8)
        full_indices = np.zeros((out, vectors_per_row), dtype=np.uint8)
        full_indices[:, :valid_vectors] = encoded.indices
        if banks is None:
            vector_banks = np.zeros((out, vectors_per_row), dtype=np.uint8)
        else:
            vector_banks = banks

        vectors = target.reshape(out, vectors_per_row, 8)
        vector_weight = objective_weight.reshape(out, vectors_per_row, 8)
        group_scale = encoded.neuron_scale[:, None] * encoded.sub_scale.astype(np.float32)
        vector_scale = np.repeat(group_scale, spec.groupsize // 8, axis=1)
        reconstruction = codebook[full_indices] * vector_scale[..., None]
        error = vector_weight * (vectors - reconstruction) ** 2
        vector_error = error.sum(axis=-1)

        flat_indices = full_indices.reshape(-1)
        flat_scale = vector_scale.reshape(-1, 1).astype(np.float64)
        flat_vectors = vectors.reshape(-1, 8).astype(np.float64)
        flat_weight = vector_weight.reshape(-1, 8).astype(np.float64)
        np.add.at(numerator, flat_indices, flat_weight * flat_scale * flat_vectors)
        np.add.at(denominator, flat_indices, flat_weight * flat_scale * flat_scale)
        np.add.at(counts, flat_indices, np.any(flat_weight > 0, axis=1).astype(np.int64))
        signal += float(np.sum(flat_weight * flat_vectors * flat_vectors))
        sse += float(np.sum(error, dtype=np.float64))
        elements += int(np.count_nonzero(flat_weight))

        if config.reseed_pool_size:
            take = min(config.reseed_pool_size, vector_error.size)
            top = np.argpartition(vector_error.reshape(-1), -take)[-take:]
            selected_scale = flat_scale[top, 0]
            selected_vectors = flat_vectors[top]
            fallback = np.max(selected_vectors, axis=1) / 7.0
            normalizer = np.where(selected_scale > 0, selected_scale, np.maximum(fallback, 1e-12))
            pool_vectors.append((selected_vectors / normalizer[:, None]).astype(np.float32))
            pool_banks.append(vector_banks.reshape(-1)[top])
            pool_errors.append(vector_error.reshape(-1)[top].astype(np.float64))

    if pool_errors:
        errors = np.concatenate(pool_errors)
        keep = min(config.reseed_pool_size, errors.size)
        selected = np.argpartition(errors, -keep)[-keep:]
        reseed_vectors = np.concatenate(pool_vectors)[selected]
        reseed_banks = np.concatenate(pool_banks)[selected]
        reseed_errors = errors[selected]
        order = np.argsort(reseed_errors)[::-1]
        reseed_vectors = reseed_vectors[order]
        reseed_banks = reseed_banks[order]
        reseed_errors = reseed_errors[order]
    else:
        reseed_vectors = np.empty((0, 8), dtype=np.float32)
        reseed_banks = np.empty(0, dtype=np.uint8)
        reseed_errors = np.empty(0, dtype=np.float64)

    return _CodebookStats(
        numerator,
        denominator,
        counts,
        signal,
        sse,
        elements,
        reseed_vectors,
        reseed_banks,
        reseed_errors,
    )


@lru_cache(maxsize=1)
def _candidate_grid() -> np.ndarray:
    encoded = np.arange(1 << 16, dtype=np.uint32)
    shifts = (2 * np.arange(8, dtype=np.uint32))[None, :]
    digits = (encoded[:, None] >> shifts) & 3
    return np.ascontiguousarray(2 * digits + 1, dtype=np.int8)


def _encoded_grid_index(table: np.ndarray) -> np.ndarray:
    digits = ((table.astype(np.uint32) - 1) // 2).astype(np.uint32)
    shifts = (2 * np.arange(8, dtype=np.uint32))[None, :]
    return np.bitwise_or.reduce(digits << shifts, axis=1)


def _project_bank_unique(
    centroids: np.ndarray,
    weights: np.ndarray,
    previous: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("NVQ codebook training requires scipy; install mfq[train]") from exc

    grid = _candidate_grid().astype(np.float32)
    grid_sq = grid * grid
    candidate_ids: list[np.ndarray] = []
    take = min(candidate_count, grid.shape[0])
    for centroid, weight in zip(centroids, weights):
        cost = np.sum(weight[None, :] * (grid - centroid[None, :]) ** 2, axis=1)
        candidate_ids.append(np.argpartition(cost, take - 1)[:take])
    candidate_ids.append(_encoded_grid_index(previous))
    union = np.unique(np.concatenate(candidate_ids))
    if union.size < centroids.shape[0]:
        missing = centroids.shape[0] - union.size
        extra = np.setdiff1d(np.arange(grid.shape[0]), union, assume_unique=True)[:missing]
        union = np.concatenate([union, extra])

    candidate = grid[union]
    candidate_sq = grid_sq[union]
    weighted_centroid = weights * centroids
    constant = np.sum(weights * centroids * centroids, axis=1)
    cost = (
        candidate_sq @ weights.T
        - 2.0 * candidate @ weighted_centroid.T
        + constant[None, :]
    ).T
    rows, columns = linear_sum_assignment(cost)
    if rows.size != centroids.shape[0]:
        raise RuntimeError("failed to assign a unique legal E8 point to every centroid")
    result = np.empty_like(previous)
    result[rows] = candidate[columns].astype(np.int8)
    return result


def _updated_codebook(
    previous: np.ndarray,
    stats: _CodebookStats,
    config: NvqCodebookTrainingConfig,
) -> np.ndarray:
    denominator = stats.denominator.copy()
    centroids = np.divide(
        stats.numerator,
        denominator,
        out=previous.astype(np.float64),
        where=denominator > 0,
    )
    empty = stats.counts == 0
    if np.any(empty) and stats.reseed_vectors.size:
        for code in np.flatnonzero(empty):
            bank = code // 128 if config.sign_mode == "index_parity" else 0
            eligible = np.flatnonzero(stats.reseed_banks == bank)
            if eligible.size:
                vector = stats.reseed_vectors[eligible[code % eligible.size]]
                centroids[code] = np.clip(vector, 1.0, 7.0)
                denominator[code] = 1.0

    if config.sign_mode == "index_parity":
        result = np.empty_like(previous)
        for bank in (0, 1):
            part = slice(bank * 128, (bank + 1) * 128)
            result[part] = _project_bank_unique(
                centroids[part],
                denominator[part],
                previous[part],
                config.projection_candidates,
            )
        return result
    return _project_bank_unique(
        centroids,
        denominator,
        previous,
        config.projection_candidates,
    )


def train_nvq2_codebook(
    matrices: Sequence[NvqTrainingMatrix],
    config: NvqCodebookTrainingConfig,
    *,
    initial: np.ndarray | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[np.ndarray, tuple[dict[str, float | int], ...], int, float]:
    if not matrices:
        raise ValueError("at least one training matrix is required")
    table = validate_e8_codebook(
        initial_e8_codebook(config.sign_mode) if initial is None else initial
    )
    best_table = table.copy()
    best_nmse = math.inf
    history: list[dict[str, float | int]] = []
    previous_nmse = math.inf
    best_elements = 0
    best_signal = 0.0

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
        }
        history.append(row)
        if progress is not None:
            progress(dict(row))
        if nmse < best_nmse:
            best_nmse = nmse
            best_table = table.copy()
            best_elements = stats.elements
            best_signal = stats.signal
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

    return best_table, tuple(history), best_elements, best_signal


def train_nvq2_codebook_set(
    matrices: Sequence[NvqTrainingMatrix],
    config: NvqCodebookTrainingConfig,
    *,
    source_model: str,
    source_rows: dict[str, list[int]] | None = None,
    progress: ProgressCallback | None = None,
) -> NvqCodebookArtifact:
    grouped: dict[str, list[NvqTrainingMatrix]] = {}
    for matrix in matrices:
        grouped.setdefault(codebook_scope_key(matrix.name, config.scope), []).append(matrix)
    trained: list[TrainedNvqCodebook] = []
    for key in sorted(grouped):
        subset = grouped[key]

        def scoped_progress(event: dict[str, object]) -> None:
            if progress is not None:
                progress({"scope_key": key, **event})

        table, history, elements, signal = train_nvq2_codebook(
            subset,
            config,
            progress=scoped_progress,
        )
        trained.append(
            TrainedNvqCodebook(
                key=key,
                tensor_names=tuple(matrix.name for matrix in subset),
                table=table,
                history=history,
                train_elements=elements,
                train_signal=signal,
            )
        )
    return NvqCodebookArtifact(
        source_model=source_model,
        config=config,
        codebooks=tuple(trained),
        source_rows=source_rows or {},
    )


def save_nvq_codebook_artifact(
    artifact: NvqCodebookArtifact,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite codebook artifact: {output}")
    entries = []
    for trained in artifact.codebooks:
        payload = pack_e8_codebook(trained.table)
        entries.append(
            {
                "key": trained.key,
                "tensor_names": list(trained.tensor_names),
                "packed_u16_le_hex": payload.hex(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "history": list(trained.history),
                "train_elements": trained.train_elements,
                "train_signal": trained.train_signal,
            }
        )
    document = {
        "format": _ARTIFACT_FORMAT,
        "version": _ARTIFACT_VERSION,
        "source_model": artifact.source_model,
        "config": asdict(artifact.config),
        "source_rows": artifact.source_rows,
        "codebooks": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(output)
    return output


def load_nvq_codebook_artifact(path: str | Path) -> NvqCodebookArtifact:
    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    if (
        document.get("format") not in {_ARTIFACT_FORMAT, _LEGACY_ARTIFACT_FORMAT}
        or document.get("version") != _ARTIFACT_VERSION
    ):
        raise ValueError(f"unsupported NVQ codebook artifact: {source}")
    config = NvqCodebookTrainingConfig(**document["config"])
    entries = []
    for raw in document["codebooks"]:
        payload = bytes.fromhex(raw["packed_u16_le_hex"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != raw["sha256"]:
            raise ValueError(f"codebook checksum mismatch for {raw['key']}")
        entries.append(
            TrainedNvqCodebook(
                key=raw["key"],
                tensor_names=tuple(raw["tensor_names"]),
                table=unpack_e8_codebook(payload),
                history=tuple(raw["history"]),
                train_elements=int(raw["train_elements"]),
                train_signal=float(raw["train_signal"]),
            )
        )
    return NvqCodebookArtifact(
        source_model=str(document["source_model"]),
        config=config,
        codebooks=tuple(entries),
        source_rows={key: list(value) for key, value in document.get("source_rows", {}).items()},
    )
