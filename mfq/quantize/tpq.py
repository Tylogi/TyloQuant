"""Native TPQ product-VQ training and Euclidean assignment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mfq.formats.tpq import (
    CccpInt4Tensor,
    CccpPqSpec,
    CccpPqTensor,
)


@dataclass(frozen=True)
class CccpKmeansConfig:
    """Training controls for one learned CCCP codebook."""

    iterations: int = 12
    restarts: int = 2
    sample_points: int = 100_000
    seed: int = 0
    distance_bytes: int = 1 << 30

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.restarts <= 0:
            raise ValueError("CCCP k-means iterations and restarts must be positive")
        if self.sample_points <= 0 or self.distance_bytes <= 0:
            raise ValueError("CCCP k-means sample and distance budget must be positive")


@dataclass(frozen=True)
class CccpKmeansResult:
    codebook: np.ndarray
    sse: float
    history: tuple[float, ...]


def _device(value: str | torch.device | None) -> torch.device:
    if value is not None:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_points(
    points: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    values = torch.as_tensor(points, dtype=torch.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("CCCP points must be a non-empty [N,D] matrix")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("CCCP points must be finite")
    return values


def _distance_chunk_rows(
    codebook_entries: int,
    distance_bytes: int,
) -> int:
    return max(1024, int(distance_bytes) // (int(codebook_entries) * 4))


def _distance(
    points: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    result = (
        (points * points).sum(1, keepdim=True)
        + (codebook * codebook).sum(1)
        - 2.0 * points @ codebook.t()
    )
    return result.clamp_min_(0)


def _assign_device(
    points: torch.Tensor,
    codebook: torch.Tensor,
    *,
    distance_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = points.shape[0]
    chunk = _distance_chunk_rows(codebook.shape[0], distance_bytes)
    labels = torch.empty(count, dtype=torch.int64, device=points.device)
    errors = torch.empty(count, dtype=torch.float32, device=points.device)
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        distance = _distance(points[start:end], codebook)
        error, label = distance.min(dim=1)
        labels[start:end] = label
        errors[start:end] = error
    return labels, errors


@torch.no_grad()
def assign_cccp_codebook(
    points: np.ndarray | torch.Tensor,
    codebook: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device | None = None,
    distance_bytes: int = 1 << 30,
) -> np.ndarray:
    """Assign each point to its nearest codeword by squared L2 distance."""

    values = _validate_points(points)
    table = torch.as_tensor(codebook, dtype=torch.float32)
    if table.ndim != 2 or table.shape[1] != values.shape[1]:
        raise ValueError("CCCP codebook must have shape [K,D]")
    if not bool(torch.isfinite(table).all()):
        raise ValueError("CCCP codebook must be finite")
    target = _device(device)
    values = values.to(target)
    table = table.to(target)
    labels, _ = _assign_device(
        values,
        table,
        distance_bytes=distance_bytes,
    )
    dtype = np.uint8 if table.shape[0] <= 256 else np.uint16
    return labels.cpu().numpy().astype(dtype, copy=False)


def _kmeans_plus_plus(
    points: torch.Tensor,
    entries: int,
    *,
    seed: int,
) -> torch.Tensor:
    count, width = points.shape
    generator = torch.Generator(device=points.device)
    generator.manual_seed(int(seed))
    codebook = torch.empty(
        (entries, width),
        dtype=torch.float32,
        device=points.device,
    )
    first = torch.randint(
        count,
        (1,),
        generator=generator,
        device=points.device,
    )
    codebook[0] = points[first]
    delta = points - codebook[0]
    distance = (delta * delta).sum(1)
    for index in range(1, entries):
        total = distance.sum()
        if float(total) <= 0:
            selected = torch.randint(
                count,
                (1,),
                generator=generator,
                device=points.device,
            )
        else:
            selected = torch.multinomial(
                distance,
                1,
                replacement=True,
                generator=generator,
            )
        codebook[index] = points[selected]
        delta = points - codebook[index]
        candidate = (delta * delta).sum(1)
        distance = torch.minimum(distance, candidate)
    return codebook


@torch.no_grad()
def train_cccp_codebook(
    points: np.ndarray | torch.Tensor,
    spec: CccpPqSpec,
    *,
    config: CccpKmeansConfig = CccpKmeansConfig(),
    device: str | torch.device | None = None,
) -> CccpKmeansResult:
    """Train one CCCP codebook with unweighted Euclidean Lloyd updates."""

    values = _validate_points(points)
    if values.shape[1] != spec.vector_size:
        raise ValueError(
            f"CCCP-{spec.tier} expects {spec.vector_size}-D points"
        )
    if values.shape[0] < spec.codebook_entries:
        raise ValueError(
            f"CCCP-{spec.tier} needs at least {spec.codebook_entries} points"
        )
    take = min(values.shape[0], config.sample_points)
    best_codebook: torch.Tensor | None = None
    best_sse = float("inf")
    best_history: tuple[float, ...] = ()
    target = _device(device)
    for restart in range(config.restarts):
        rng = np.random.default_rng(config.seed + restart * 7919)
        sample_ids = rng.choice(values.shape[0], take, replace=False)
        sample_index = torch.as_tensor(
            sample_ids,
            dtype=torch.int64,
            device=values.device,
        )
        sample = values.index_select(0, sample_index).to(target)
        codebook = _kmeans_plus_plus(
            sample,
            spec.codebook_entries,
            seed=config.seed + restart * 7919,
        )
        history: list[float] = []
        for _ in range(config.iterations):
            labels, errors = _assign_device(
                sample,
                codebook,
                distance_bytes=config.distance_bytes,
            )
            numerator = torch.zeros_like(codebook)
            numerator.index_add_(0, labels, sample)
            denominator = torch.bincount(
                labels, minlength=spec.codebook_entries
            ).to(torch.float32)
            live = denominator > 0
            codebook[live] = numerator[live] / denominator[live, None]
            empty = ~live
            empty_count = int(empty.sum())
            if empty_count:
                replacement = errors.topk(empty_count).indices
                codebook[empty] = sample[replacement]
            history.append(float(errors.sum()))
        _, final_errors = _assign_device(
            sample,
            codebook,
            distance_bytes=config.distance_bytes,
        )
        final_sse = float(final_errors.sum())
        if final_sse < best_sse:
            best_sse = final_sse
            best_codebook = codebook.clone()
            best_history = tuple(history[:-1] + [final_sse])
        del sample, codebook
    assert best_codebook is not None
    return CccpKmeansResult(
        codebook=best_codebook.cpu().numpy().astype(np.float32),
        sse=best_sse,
        history=best_history,
    )


@torch.no_grad()
def quantize_cccp_pq_fixed(
    weight: np.ndarray | torch.Tensor,
    spec: CccpPqSpec,
    codebook: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device | None = None,
    distance_bytes: int = 1 << 30,
) -> CccpPqTensor:
    """Quantize one matrix with a frozen CCCP codebook."""

    matrix = torch.as_tensor(weight, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[1] % spec.vector_size:
        raise ValueError(
            f"CCCP-{spec.tier} expects [rows, columns] with columns divisible "
            f"by {spec.vector_size}"
        )
    shape = (int(matrix.shape[0]), int(matrix.shape[1]))
    indices = assign_cccp_codebook(
        matrix.reshape(-1, spec.vector_size),
        codebook,
        device=device,
        distance_bytes=distance_bytes,
    )
    return CccpPqTensor(
        spec=spec,
        shape=shape,
        axis=0,
        neuron_len=shape[1],
        indices=indices.reshape(shape[0], shape[1] // spec.vector_size),
        codebook=(
            codebook.detach().cpu().numpy().astype(np.float32, copy=False)
            if isinstance(codebook, torch.Tensor)
            else np.asarray(codebook, dtype=np.float32)
        ),
    )


@torch.no_grad()
def cccp_reconstruction_sums(
    weight: np.ndarray | torch.Tensor,
    spec: CccpPqSpec,
    codebook: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device | None = None,
    distance_bytes: int = 1 << 30,
) -> tuple[float, float]:
    """Return Euclidean reconstruction SSE and source squared norm."""

    matrix = torch.as_tensor(weight, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[1] % spec.vector_size:
        raise ValueError(
            f"CCCP-{spec.tier} expects a compatible 2-D matrix"
        )
    target = _device(device)
    points = matrix.reshape(-1, spec.vector_size).to(target)
    table = torch.as_tensor(codebook, dtype=torch.float32, device=target)
    if tuple(table.shape) != (
        spec.codebook_entries,
        spec.vector_size,
    ):
        raise ValueError(
            f"CCCP-{spec.tier} codebook shape is {tuple(table.shape)}"
        )
    chunk = _distance_chunk_rows(spec.codebook_entries, distance_bytes)
    sse = 0.0
    signal = 0.0
    for start in range(0, points.shape[0], chunk):
        part = points[start : start + chunk]
        labels = _distance(part, table).argmin(1)
        delta = part - table.index_select(0, labels)
        sse += float((delta * delta).sum())
        signal += float((part * part).sum())
    return sse, signal


@torch.no_grad()
def train_cccp_pq(
    weight: np.ndarray | torch.Tensor,
    spec: CccpPqSpec,
    *,
    config: CccpKmeansConfig = CccpKmeansConfig(),
    device: str | torch.device | None = None,
) -> tuple[CccpPqTensor, CccpKmeansResult]:
    """Train and apply one matrix-level CCCP codebook."""

    matrix = torch.as_tensor(weight, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[1] % spec.vector_size:
        raise ValueError("CCCP training expects a compatible 2-D matrix")
    result = train_cccp_codebook(
        matrix.reshape(-1, spec.vector_size),
        spec,
        config=config,
        device=device,
    )
    tensor = quantize_cccp_pq_fixed(
        matrix,
        spec,
        result.codebook,
        device=device,
        distance_bytes=config.distance_bytes,
    )
    return tensor, result


@torch.no_grad()
def train_cccp_expert_codebook(
    samples: np.ndarray | torch.Tensor,
    expert_ids: np.ndarray | torch.Tensor,
    spec: CccpPqSpec,
    *,
    config: CccpKmeansConfig = CccpKmeansConfig(),
    device: str | torch.device | None = None,
) -> CccpKmeansResult:
    """Train one layer/tier codebook from sampled ``[E,R,K]`` experts."""

    values = torch.as_tensor(samples, dtype=torch.float32)
    if values.ndim != 3 or values.shape[2] % spec.vector_size:
        raise ValueError("CCCP expert samples must have shape [E,R,K]")
    ids = torch.as_tensor(
        expert_ids,
        dtype=torch.int64,
        device=values.device,
    ).reshape(-1)
    if ids.numel() == 0 or bool((ids < 0).any()) or bool(
        (ids >= values.shape[0]).any()
    ):
        raise ValueError("CCCP expert cohort IDs are invalid")
    selected = values.index_select(0, ids)
    return train_cccp_codebook(
        selected.reshape(-1, spec.vector_size),
        spec,
        config=config,
        device=device,
    )


def dequantize_cccp_pq(tensor: CccpPqTensor) -> np.ndarray:
    """Restore a CCCP product-VQ matrix to float32."""

    indices = tensor.indices.reshape(-1)
    if tensor.spec.index_bits in {12, 14}:
        from mfq.formats.tpq import unpack_cccp_indices

        count = int(tensor.shape[0]) * (
            int(tensor.shape[1]) // tensor.spec.vector_size
        )
        indices, offset = unpack_cccp_indices(
            memoryview(indices),
            0,
            count,
            tensor.spec.index_bits,
        )
        if offset != (
            indices.size * tensor.spec.index_bits + 7
        ) // 8:
            raise ValueError("CCCP packed index payload has an invalid tail")
    return tensor.codebook[indices].reshape(tensor.shape)


def quantize_cccp_int4(
    weight: np.ndarray | torch.Tensor,
    *,
    group_size: int = 64,
) -> CccpInt4Tensor:
    """Apply CCCP's symmetric int4-g64 dense quantizer."""

    matrix = np.asarray(torch.as_tensor(weight, dtype=torch.float32).cpu())
    if matrix.ndim != 2 or matrix.shape[1] % group_size or matrix.shape[1] % 2:
        raise ValueError("CCCP-I4 expects a 2-D matrix aligned to its group size")
    rows, columns = matrix.shape
    groups = matrix.reshape(rows, columns // group_size, group_size)
    scales = np.maximum(np.abs(groups).max(axis=2) / 7.0, 1e-12)
    values = np.clip(
        np.rint(groups / scales[:, :, None]),
        -7,
        7,
    ).astype(np.int8)
    values = values.reshape(rows, columns) + 8
    packed = (
        values[:, 0::2].astype(np.uint8)
        | (values[:, 1::2].astype(np.uint8) << 4)
    )
    return CccpInt4Tensor(
        shape=(rows, columns),
        axis=0,
        neuron_len=columns,
        group_size=group_size,
        packed=packed,
        scales=scales.astype(np.float16),
    )


def dequantize_cccp_int4(tensor: CccpInt4Tensor) -> np.ndarray:
    """Restore a CCCP symmetric int4 matrix to float32."""

    low = (tensor.packed & 0x0F).astype(np.int8) - 8
    high = (tensor.packed >> 4).astype(np.int8) - 8
    values = np.stack((low, high), axis=2).reshape(tensor.shape)
    rows, columns = tensor.shape
    groups = values.astype(np.float32).reshape(
        rows,
        columns // tensor.group_size,
        tensor.group_size,
    )
    return (
        groups * tensor.scales.astype(np.float32)[:, :, None]
    ).reshape(tensor.shape)


def save_cccp_codebook_artifact(
    path: str | Path,
    spec: CccpPqSpec,
    result: CccpKmeansResult,
) -> None:
    """Atomically save one frozen codebook for streaming MFQ conversion."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            family=np.asarray(spec.label),
            vector_size=np.asarray(spec.vector_size, dtype=np.int32),
            codebook_entries=np.asarray(spec.codebook_entries, dtype=np.int32),
            codebook=np.asarray(result.codebook, dtype=np.float32),
            objective=np.asarray("euclidean_sse"),
            sse=np.asarray(result.sse, dtype=np.float64),
            history=np.asarray(result.history, dtype=np.float64),
        )
    os.replace(temporary, output)


__all__ = [
    "CccpKmeansConfig",
    "CccpKmeansResult",
    "assign_cccp_codebook",
    "cccp_reconstruction_sums",
    "dequantize_cccp_int4",
    "dequantize_cccp_pq",
    "quantize_cccp_int4",
    "quantize_cccp_pq_fixed",
    "save_cccp_codebook_artifact",
    "train_cccp_codebook",
    "train_cccp_expert_codebook",
    "train_cccp_pq",
]


# Canonical TPQ API aliases.
TpqKmeansConfig = CccpKmeansConfig
TpqKmeansResult = CccpKmeansResult
assign_tpq_codebook = assign_cccp_codebook
tpq_reconstruction_sums = cccp_reconstruction_sums
dequantize_tpq_int4 = dequantize_cccp_int4
dequantize_tpq_pq = dequantize_cccp_pq
quantize_tpq_int4 = quantize_cccp_int4
quantize_tpq_pq_fixed = quantize_cccp_pq_fixed
save_tpq_codebook_artifact = save_cccp_codebook_artifact
train_tpq_codebook = train_cccp_codebook
train_tpq_expert_codebook = train_cccp_expert_codebook
train_tpq_pq = train_cccp_pq

__all__ += [
    "TpqKmeansConfig",
    "TpqKmeansResult",
    "assign_tpq_codebook",
    "dequantize_tpq_int4",
    "dequantize_tpq_pq",
    "quantize_tpq_int4",
    "quantize_tpq_pq_fixed",
    "save_tpq_codebook_artifact",
    "tpq_reconstruction_sums",
    "train_tpq_codebook",
    "train_tpq_expert_codebook",
    "train_tpq_pq",
]
