#!/usr/bin/env python3
"""Compare full-row MXFP4 centroid VQ with production NVQ on DSV4F.

One routed Gate/Up expert is a 4096-by-4096 matrix.  This experiment treats
each complete row as one vector, learns ordinary Lloyd k-means centroids, and
projects every centroid back to a valid OCP MXFP4 row after each M step.  The
MXFP4 codebook and one packed index per source row are both charged against
the comparison format's real payload budget.

This is deliberately an offline rate/distortion experiment.  It does not add
an MFQ container dtype or a runtime kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform

from mfq.formats.nvq import (
    NVQ2_E8,
    NVQ2_E8_1024,
    NVQ2_E8_4096,
    NVQ3_D4,
    NVQ3_D4_512,
    NVQ3_D4_1024,
    NvqSpec,
)
from mfq.formats.nvq1_s import NVQ1_S
from mfq.quantize.mxfp import decode_mxfp4
from mfq.quantize.nvq1_s_quant import dequantize as dequantize_nvq1_s
from mfq.quantize.nvq1_s_quant_torch import quantize_axis0 as quantize_nvq1_s
from mfq.quantize.nvq_quant import dequantize as dequantize_nvq
from mfq.quantize.nvq_quant_torch import quantize_axis0 as quantize_nvq
from mfq.quantize.v4f_source import V4FCheckpoint

_MXFP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MXFP4_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_NVQ_SPECS: dict[str, NvqSpec | str] = {
    "NVQ1-S": "NVQ1-S",
    NVQ2_E8.label: NVQ2_E8,
    NVQ2_E8_1024.label: NVQ2_E8_1024,
    NVQ2_E8_4096.label: NVQ2_E8_4096,
    NVQ3_D4.label: NVQ3_D4,
    NVQ3_D4_512.label: NVQ3_D4_512,
    NVQ3_D4_1024.label: NVQ3_D4_1024,
}


@dataclass(frozen=True)
class RateMatch:
    entries: int
    index_bits: int
    codebook_nbytes: int
    index_nbytes: int
    payload_nbytes: int
    payload_bpw: float
    budget_nbytes: int


def mxfp4_matrix_nbytes(rows: int, columns: int) -> int:
    """Return packed E2M1 plus one E8M0 byte per 32 values."""

    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("MXFP4 matrices require positive rows and K divisible by 32")
    return rows * (columns // 2 + columns // 32)


def row_vq_payload_nbytes(rows: int, columns: int, entries: int) -> int:
    if not 1 <= entries <= rows:
        raise ValueError("row-VQ entries must be within [1, rows]")
    index_bits = max(1, (entries - 1).bit_length())
    return (
        mxfp4_matrix_nbytes(entries, columns)
        + (rows * index_bits + 7) // 8
    )


def match_row_vq_rate(rows: int, columns: int, budget_nbytes: int) -> RateMatch:
    """Use the largest full-row MXFP4 codebook that fits ``budget_nbytes``."""

    if budget_nbytes <= 0:
        raise ValueError("rate-match budget must be positive")
    best = 0
    for entries in range(1, rows + 1):
        if row_vq_payload_nbytes(rows, columns, entries) <= budget_nbytes:
            best = entries
    if not best:
        raise ValueError("budget cannot hold one MXFP4 centroid and row indices")
    index_bits = max(1, (best - 1).bit_length())
    codebook_nbytes = mxfp4_matrix_nbytes(best, columns)
    index_nbytes = (rows * index_bits + 7) // 8
    payload_nbytes = codebook_nbytes + index_nbytes
    return RateMatch(
        entries=best,
        index_bits=index_bits,
        codebook_nbytes=codebook_nbytes,
        index_nbytes=index_nbytes,
        payload_nbytes=payload_nbytes,
        payload_bpw=8.0 * payload_nbytes / (rows * columns),
        budget_nbytes=budget_nbytes,
    )


def _nearest_mxfp4(
    normalized_abs: torch.Tensor,
    levels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    level_ids = torch.zeros_like(normalized_abs, dtype=torch.int64)
    for threshold in _MXFP4_THRESHOLDS:
        level_ids.add_(normalized_abs > threshold)
    return levels[level_ids], level_ids


@torch.inference_mode()
def project_mxfp4(
    value: torch.Tensor,
    *,
    row_chunk: int = 128,
    exponent_min_offset: int = -8,
    exponent_max_offset: int = 6,
    return_storage: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Project rows to valid block-32 E2M1/E8M0 MXFP4 values.

    The shared exponent is selected by direct block SSE.  Candidate exponents
    span well beyond the no-saturation scale; scales that would quantize every
    value to zero are included at the upper end.  The returned optional value
    stream is low-nibble-first and the scale stream stores E8M0 bytes.
    """

    if value.ndim != 2 or value.shape[1] % 32:
        raise ValueError("MXFP4 projection requires [rows,K] with K divisible by 32")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    if exponent_min_offset > exponent_max_offset:
        raise ValueError("invalid exponent offset range")
    if not value.is_floating_point():
        raise TypeError("MXFP4 projection requires floating-point input")

    rows, columns = (int(item) for item in value.shape)
    device = value.device
    levels = torch.tensor(_MXFP4_LEVELS, device=device, dtype=torch.float32)
    offsets = torch.arange(
        exponent_min_offset,
        exponent_max_offset + 1,
        device=device,
        dtype=torch.int32,
    )
    output = torch.empty((rows, columns), device=device, dtype=torch.float32)
    packed_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []

    for start in range(0, rows, row_chunk):
        stop = min(rows, start + row_chunk)
        source = value[start:stop].to(torch.float32).reshape(-1, 32)
        absolute = source.abs()
        peak = absolute.amax(dim=1)
        safe_peak = torch.where(peak > 0, peak, torch.ones_like(peak))
        base_exponent = torch.ceil(torch.log2(safe_peak / 6.0)).to(torch.int32)
        candidate_exponent = torch.clamp(
            base_exponent[:, None] + offsets[None, :], -127, 127
        )
        candidate_scale = torch.pow(
            torch.tensor(2.0, device=device), candidate_exponent.to(torch.float32)
        )
        normalized = absolute[:, None, :] / candidate_scale[:, :, None]
        magnitude, level_ids = _nearest_mxfp4(normalized, levels)
        quantized = torch.copysign(magnitude, source[:, None, :])
        quantized.mul_(candidate_scale[:, :, None])
        error = (quantized - source[:, None, :]).square().sum(dim=2)
        best = error.argmin(dim=1)
        block = torch.arange(source.shape[0], device=device)
        chosen = quantized[block, best]
        chosen = torch.where(peak[:, None] > 0, chosen, torch.zeros_like(chosen))
        output[start:stop] = chosen.reshape(stop - start, columns)

        if return_storage:
            chosen_levels = level_ids[block, best].to(torch.uint8)
            sign = torch.where(source < 0, 8, 0).to(torch.uint8)
            codes = torch.where(
                chosen_levels != 0,
                chosen_levels | sign,
                chosen_levels,
            )
            codes = torch.where(peak[:, None] > 0, codes, torch.zeros_like(codes))
            packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
            raw_scale = (candidate_exponent[block, best] + 127).to(torch.uint8)
            raw_scale = torch.where(peak > 0, raw_scale, torch.zeros_like(raw_scale))
            packed_parts.append(packed.reshape(stop - start, columns // 2).cpu())
            scale_parts.append(raw_scale.reshape(stop - start, columns // 32).cpu())

    packed_result = torch.cat(packed_parts) if return_storage else None
    scale_result = torch.cat(scale_parts) if return_storage else None
    return output, packed_result, scale_result


@torch.inference_mode()
def _assign_rows(
    value: torch.Tensor,
    centroids: torch.Tensor,
    *,
    row_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    centroid_norm = centroids.square().sum(dim=1)
    transpose = centroids.transpose(0, 1).contiguous()
    assignments: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []
    for start in range(0, value.shape[0], row_chunk):
        source = value[start : start + row_chunk]
        distance = (
            source.square().sum(dim=1, keepdim=True)
            + centroid_norm[None, :]
            - 2.0 * (source @ transpose)
        )
        error, assignment = distance.min(dim=1)
        assignments.append(assignment)
        errors.append(error.clamp_min_(0))
    return torch.cat(assignments), torch.cat(errors)


@torch.inference_mode()
def _pairwise_distance_cpu(value: torch.Tensor) -> np.ndarray:
    """Compute a dense squared-Euclidean matrix on the active device."""

    norm = value.square().sum(dim=1)
    distance = (
        norm[:, None]
        + norm[None, :]
        - 2.0 * (value @ value.transpose(0, 1))
    ).clamp_min_(0)
    result = distance.cpu().numpy()
    np.fill_diagonal(result, np.inf)
    return result


def _ordered_nearest_pairing(
    distance: np.ndarray,
    order: np.ndarray,
) -> tuple[np.ndarray, float]:
    rows = int(distance.shape[0])
    active = np.ones(rows, dtype=np.bool_)
    pairs: list[tuple[int, int]] = []
    score = 0.0
    for raw_index in order:
        first = int(raw_index)
        if not active[first]:
            continue
        active[first] = False
        candidate_distance = np.where(active, distance[first], np.inf)
        second = int(candidate_distance.argmin())
        if not np.isfinite(candidate_distance[second]):
            raise RuntimeError("nearest-pair initialization left an unmatched row")
        active[second] = False
        pairs.append((first, second))
        score += 0.5 * float(distance[first, second])
    if active.any() or len(pairs) * 2 != rows:
        raise RuntimeError("nearest-pair initialization is incomplete")
    return np.asarray(pairs, dtype=np.int64), score


def _globally_greedy_pairing(distance: np.ndarray) -> tuple[np.ndarray, float]:
    """Greedily accept globally shortest disjoint edges."""

    rows = int(distance.shape[0])
    left, right = np.triu_indices(rows, k=1)
    edge_order = np.argsort(distance[left, right], kind="stable")
    active = np.ones(rows, dtype=np.bool_)
    pairs = np.empty((rows // 2, 2), dtype=np.int64)
    count = 0
    score = 0.0
    for raw_edge in edge_order:
        first = int(left[raw_edge])
        second = int(right[raw_edge])
        if not (active[first] and active[second]):
            continue
        active[first] = False
        active[second] = False
        pairs[count] = (first, second)
        score += 0.5 * float(distance[first, second])
        count += 1
        if count == rows // 2:
            break
    if active.any() or count != rows // 2:
        raise RuntimeError("global nearest-pair initialization is incomplete")
    return pairs, score


@torch.inference_mode()
def _nearest_pair_initial_centroids(
    source: torch.Tensor,
    entries: int,
    *,
    seed: int,
    pairing_restarts: int = 8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Seed K means from a low-cost partition of every row into pairs/quads.

    For the experiment's N=4096 and K=1969, all 4096 rows first enter a
    nearest-neighbour pair.  The 79 closest disjoint pairs-of-pairs are then
    merged, yielding exactly 1,890 two-row groups and 79 four-row groups.
    Centroids start at group means, so no codeword is pinned to an exact data
    row before Lloyd optimization begins.
    """

    rows, columns = (int(item) for item in source.shape)
    pair_count = rows // 2
    if rows % 2 or not rows // 4 <= entries <= pair_count:
        raise ValueError(
            "nearest-pair initialization requires even N and N/4 <= K <= N/2"
        )
    if pairing_restarts <= 0:
        raise ValueError("pairing_restarts must be positive")

    distance = _pairwise_distance_cpu(source)
    candidates: list[tuple[str, np.ndarray, float]] = []
    global_pairs, global_score = _globally_greedy_pairing(distance)
    candidates.append(("global-edge", global_pairs, global_score))
    generator = np.random.default_rng(seed)
    for restart in range(pairing_restarts):
        pairs, score = _ordered_nearest_pairing(
            distance, generator.permutation(rows)
        )
        candidates.append((f"ordered-{restart}", pairs, score))
    pairing_name, pairs, base_pair_sse = min(candidates, key=lambda item: item[2])
    del distance

    pair_ids = torch.as_tensor(pairs, device=source.device)
    pair_centroids = source[pair_ids].mean(dim=1)
    merges_needed = pair_count - entries
    merge_edges: list[tuple[int, int]] = []
    merge_penalty = 0.0
    if merges_needed:
        pair_distance = _pairwise_distance_cpu(pair_centroids)
        left, right = np.triu_indices(pair_count, k=1)
        edge_order = np.argsort(pair_distance[left, right], kind="stable")
        active_pairs = np.ones(pair_count, dtype=np.bool_)
        for raw_edge in edge_order:
            first = int(left[raw_edge])
            second = int(right[raw_edge])
            if not (active_pairs[first] and active_pairs[second]):
                continue
            active_pairs[first] = False
            active_pairs[second] = False
            merge_edges.append((first, second))
            # Ward increase for merging two equally sized two-row clusters.
            merge_penalty += float(pair_distance[first, second])
            if len(merge_edges) == merges_needed:
                break
        if len(merge_edges) != merges_needed:
            raise RuntimeError("could not construct the requested pair merges")

    consumed = np.zeros(pair_count, dtype=np.bool_)
    groups: list[np.ndarray] = []
    for first, second in merge_edges:
        consumed[first] = True
        consumed[second] = True
        groups.append(np.concatenate((pairs[first], pairs[second])))
    groups.extend(pairs[index] for index in np.flatnonzero(~consumed))
    if len(groups) != entries:
        raise RuntimeError("nearest-pair initialization produced the wrong K")

    labels = np.empty(rows, dtype=np.int64)
    group_sizes = np.empty(entries, dtype=np.int64)
    for cluster, members in enumerate(groups):
        labels[members] = cluster
        group_sizes[cluster] = len(members)
    label_tensor = torch.as_tensor(labels, device=source.device)
    sums = torch.zeros((entries, columns), device=source.device, dtype=torch.float32)
    sums.index_add_(0, label_tensor, source)
    counts = torch.as_tensor(group_sizes, device=source.device, dtype=torch.float32)
    means = sums / counts[:, None]
    return means, {
        "pairing_candidate": pairing_name,
        "pairing_restarts": pairing_restarts,
        "partition_sse_before_mxfp4": base_pair_sse + merge_penalty,
        "partition_cluster_min": int(group_sizes.min()),
        "partition_cluster_max": int(group_sizes.max()),
        "partition_two_row_clusters": int((group_sizes == 2).sum()),
        "partition_four_row_clusters": int((group_sizes == 4).sum()),
    }


@torch.inference_mode()
def _ward_initial_centroids(
    source: torch.Tensor,
    entries: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run full Ward agglomeration and cut the tree at exactly K clusters."""

    rows, columns = (int(item) for item in source.shape)
    if not 1 <= entries <= rows:
        raise ValueError("Ward initialization requires 1 <= K <= N")

    distance_start = time.perf_counter()
    print(
        "Ward initialization: computing the full row-distance matrix...",
        file=sys.stderr,
        flush=True,
    )
    distance = _pairwise_distance_cpu(source)
    np.fill_diagonal(distance, 0.0)
    np.sqrt(distance, out=distance)
    condensed = squareform(distance, checks=False)
    del distance
    distance_seconds = time.perf_counter() - distance_start

    linkage_start = time.perf_counter()
    print(
        f"Ward initialization: merging {rows} rows to {entries} clusters...",
        file=sys.stderr,
        flush=True,
    )
    tree = linkage(condensed, method="ward", optimal_ordering=False)
    del condensed
    raw_labels = cut_tree(tree, n_clusters=[entries]).reshape(-1)
    _, labels = np.unique(raw_labels, return_inverse=True)
    labels = labels.astype(np.int64, copy=False)
    if int(labels.max()) + 1 != entries:
        raise RuntimeError("Ward cut did not produce exactly K clusters")
    linkage_seconds = time.perf_counter() - linkage_start

    label_tensor = torch.as_tensor(labels, device=source.device)
    group_sizes = np.bincount(labels, minlength=entries)
    sums = torch.zeros((entries, columns), device=source.device, dtype=torch.float32)
    sums.index_add_(0, label_tensor, source)
    counts = torch.as_tensor(group_sizes, device=source.device, dtype=torch.float32)
    means = sums / counts[:, None]
    partition_sse = float(
        (source - means[label_tensor]).square().sum().cpu()
    )
    return means, {
        "distance_seconds": distance_seconds,
        "linkage_seconds": linkage_seconds,
        "partition_sse_before_mxfp4": partition_sse,
        "partition_cluster_min": int(group_sizes.min()),
        "partition_cluster_max": int(group_sizes.max()),
        "partition_singleton_clusters": int((group_sizes == 1).sum()),
        "partition_empty_clusters": int((group_sizes == 0).sum()),
    }


@torch.inference_mode()
def quantize_row_vq(
    value: torch.Tensor,
    entries: int,
    *,
    device: str | torch.device,
    seed: int,
    iterations: int,
    initialization: str = "random",
    assignment_row_chunk: int = 512,
    projection_row_chunk: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run deterministic Lloyd k-means with deployed MXFP4 centroids."""

    if value.ndim != 2:
        raise ValueError("row VQ requires a matrix")
    rows, columns = (int(item) for item in value.shape)
    if not 1 <= entries <= rows:
        raise ValueError("entries must be within [1, rows]")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if initialization not in {
        "random",
        "energy-tail",
        "nearest-pair",
        "ward",
    }:
        raise ValueError(f"unsupported initialization: {initialization}")

    target = torch.device(device)
    source = value.to(device=target, dtype=torch.float32).contiguous()
    initialization_meta: dict[str, Any] = {}
    if initialization == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        initial = torch.randperm(rows, generator=generator)[:entries].to(target)
        initial_centroids = source[initial]
    elif initialization == "nearest-pair":
        initial_centroids, initialization_meta = _nearest_pair_initial_centroids(
            source, entries, seed=seed
        )
    elif initialization == "ward":
        initial_centroids, initialization_meta = _ward_initial_centroids(
            source, entries
        )
    elif entries == 1:
        initial_centroids = source.mean(dim=0, keepdim=True)
    else:
        # In this high-K regime a random data-row seed locks many centroids into
        # arbitrary singletons.  Preserve the highest-energy K-1 rows and seed
        # the final centroid at the mean of the remaining low-energy tail.
        energy = source.square().sum(dim=1)
        head = torch.topk(energy, k=entries - 1, largest=True).indices
        tail_mask = torch.ones(rows, device=target, dtype=torch.bool)
        tail_mask[head] = False
        tail_mean = source[tail_mask].mean(dim=0, keepdim=True)
        initial_centroids = torch.cat((source[head], tail_mean), dim=0)
    centroids, _, _ = project_mxfp4(
        initial_centroids, row_chunk=projection_row_chunk
    )
    assignment, error = _assign_rows(
        source, centroids, row_chunk=assignment_row_chunk
    )
    initial_counts = torch.bincount(assignment, minlength=entries)
    best_sse = float(error.sum().cpu())
    best_centroids = centroids.clone()
    best_assignment = assignment.clone()
    trace: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "sse": best_sse,
            "changed": rows,
            "empty_clusters": int((initial_counts == 0).sum().cpu()),
        }
    ]
    print(
        f"row-VQ iteration 0: SSE={best_sse:.9g}",
        file=sys.stderr,
        flush=True,
    )

    for iteration in range(1, iterations + 1):
        previous_assignment = assignment
        counts = torch.bincount(assignment, minlength=entries)
        sums = torch.zeros((entries, columns), device=target, dtype=torch.float32)
        sums.index_add_(0, assignment, source)
        means = sums / counts.clamp_min(1).to(torch.float32)[:, None]
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
        if empty.numel():
            worst = torch.topk(error, k=int(empty.numel()), largest=True).indices
            means[empty] = source[worst]
        centroids, _, _ = project_mxfp4(
            means, row_chunk=projection_row_chunk
        )
        assignment, error = _assign_rows(
            source, centroids, row_chunk=assignment_row_chunk
        )
        sse = float(error.sum().cpu())
        changed = int((assignment != previous_assignment).sum().cpu())
        trace.append(
            {
                "iteration": iteration,
                "sse": sse,
                "changed": changed,
                "empty_clusters": int(empty.numel()),
            }
        )
        print(
            f"row-VQ iteration {iteration}: SSE={sse:.9g}, "
            f"changed={changed}, empty={int(empty.numel())}",
            file=sys.stderr,
            flush=True,
        )
        if sse < best_sse:
            best_sse = sse
            best_centroids = centroids.clone()
            best_assignment = assignment.clone()
        if changed == 0:
            break

    final_counts = torch.bincount(best_assignment, minlength=entries).cpu()
    reconstruction = best_centroids[best_assignment]
    metadata = {
        "initialization_metadata": initialization_meta,
        "trace": trace,
        "iterations_completed": len(trace) - 1,
        "cluster_min": int(final_counts.min()),
        "cluster_max": int(final_counts.max()),
        "cluster_mean": float(final_counts.to(torch.float32).mean()),
        "singleton_clusters": int((final_counts == 1).sum()),
        "empty_clusters": int((final_counts == 0).sum()),
    }
    return reconstruction.cpu(), best_centroids.cpu(), metadata


def _metrics(reference: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    source = reference.to(torch.float64)
    residual = reconstructed.to(torch.float64) - source
    signal = float(source.square().sum())
    sse = float(residual.square().sum())
    return {
        "signal_sse": signal,
        "error_sse": sse,
        "mse": sse / source.numel(),
        "snr_db": math.inf if sse == 0 else 10.0 * math.log10(signal / sse),
        "max_abs_error": float(residual.abs().max()),
    }


def _nvq_budget(label: str, rows: int, columns: int) -> int:
    spec = _NVQ_SPECS[label]
    if spec == "NVQ1-S":
        return NVQ1_S.payload_nbytes(rows, columns)
    assert isinstance(spec, NvqSpec)
    return spec.payload_nbytes(rows, columns)


def _quantize_baseline(
    value: torch.Tensor,
    label: str,
    *,
    device: str | torch.device,
) -> tuple[torch.Tensor, int]:
    spec = _NVQ_SPECS[label]
    if spec == "NVQ1-S":
        encoded = quantize_nvq1_s(value, device=device)
        reconstructed = torch.from_numpy(dequantize_nvq1_s(encoded))
    else:
        assert isinstance(spec, NvqSpec)
        encoded = quantize_nvq(value, spec, device=device)
        reconstructed = torch.from_numpy(dequantize_nvq(encoded))
    return reconstructed, encoded.payload_nbytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity(root: Path) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=root, text=True
    )
    return {
        "head": head,
        "dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--baseline", choices=tuple(_NVQ_SPECS), default=NVQ2_E8.label)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument(
        "--initialization",
        choices=("random", "energy-tail", "nearest-pair", "ward"),
        default="ward",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.time()
    script_root = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    checkpoint = V4FCheckpoint(model)
    source_reader = checkpoint.expert_source(args.layer, "gate_up")
    print(
        f"loading layer {args.layer} expert {args.expert} gate_up...",
        file=sys.stderr,
        flush=True,
    )
    source = source_reader.read_expert_rows(
        args.expert, 0, source_reader.rows_per_expert, device="cpu"
    ).contiguous()
    rows, columns = (int(item) for item in source.shape)

    budget = _nvq_budget(args.baseline, rows, columns)
    rate = match_row_vq_rate(rows, columns, budget)
    print(
        f"rate match: {args.baseline}={budget} bytes; "
        f"row-VQ={rate.entries} centroids, {rate.payload_nbytes} bytes",
        file=sys.stderr,
        flush=True,
    )

    baseline_start = time.perf_counter()
    print(f"quantizing {args.baseline} baseline...", file=sys.stderr, flush=True)
    baseline_reconstruction, baseline_bytes = _quantize_baseline(
        source, args.baseline, device=args.device
    )
    baseline_seconds = time.perf_counter() - baseline_start
    if baseline_bytes != budget:
        raise RuntimeError(f"baseline payload changed: {baseline_bytes} != {budget}")

    if str(torch.device(args.device)) == "mps":
        torch.mps.empty_cache()
    row_vq_start = time.perf_counter()
    print("running full-row MXFP4 k-means...", file=sys.stderr, flush=True)
    row_vq_reconstruction, centroids, row_vq_meta = quantize_row_vq(
        source,
        rate.entries,
        device=args.device,
        seed=args.seed,
        iterations=args.iterations,
        initialization=args.initialization,
    )
    row_vq_seconds = time.perf_counter() - row_vq_start

    # Materialize the final physical streams once and verify byte accounting.
    centroid_device = centroids.to(args.device)
    centroid_roundtrip, packed, scales = project_mxfp4(
        centroid_device, return_storage=True
    )
    assert packed is not None and scales is not None
    if packed.numel() + scales.numel() != rate.codebook_nbytes:
        raise RuntimeError("MXFP4 centroid byte accounting is inconsistent")
    if not torch.equal(centroid_roundtrip.cpu(), centroids):
        raise RuntimeError("stored MXFP4 centroids do not round-trip exactly")
    decoded_centroids = decode_mxfp4(
        packed.numpy(), scales.numpy(), device="cpu"
    )
    if not torch.equal(decoded_centroids, centroids):
        raise RuntimeError("physical MXFP4 streams do not decode exactly")

    result = {
        "schema": 1,
        "experiment": "dsv4f-full-row-mxfp4-centroid-vq",
        "created_unix": started,
        "workspace": _git_identity(script_root),
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(torch.device(args.device)),
        },
        "source": {
            "model": str(model),
            "config_sha256": _sha256(model / "config.json"),
            "index_sha256": _sha256(model / "model.safetensors.index.json"),
            "layer": args.layer,
            "expert": args.expert,
            "projection": "gate_up",
            "shape": [rows, columns],
            "logical_dtype": "MXFP4/E2M1+E8M0-g32",
            "signal_sse": float(source.to(torch.float64).square().sum()),
        },
        "baseline": {
            "format": args.baseline,
            "payload_nbytes": baseline_bytes,
            "payload_bpw": 8.0 * baseline_bytes / (rows * columns),
            "seconds": baseline_seconds,
            **_metrics(source, baseline_reconstruction),
        },
        "row_vq": {
            "format": "MXFP4-full-row-centroid-VQ",
            "vector_size": columns,
            "entries": rate.entries,
            "initialization": args.initialization,
            "index_bits": rate.index_bits,
            "codebook_nbytes": rate.codebook_nbytes,
            "index_nbytes": rate.index_nbytes,
            "payload_nbytes": rate.payload_nbytes,
            "payload_bpw": rate.payload_bpw,
            "budget_slack_nbytes": rate.budget_nbytes - rate.payload_nbytes,
            "physical_storage_roundtrip_verified": True,
            "seconds": row_vq_seconds,
            **_metrics(source, row_vq_reconstruction),
            **row_vq_meta,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
