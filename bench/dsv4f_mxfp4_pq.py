#!/usr/bin/env python3
"""Compare learned 8-D MXFP4 product VQ with production NVQ2 on DSV4F.

Unlike the rejected one-index-per-row experiment, this codec preserves NVQ's
compositional topology: every 8-value subvector independently selects a
learned signed centroid.  The centroids are projected to E2M1 with one E8M0
scale per 8-D codeword; no E8 or D4 lattice is used.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bench.dsv4f_mxfp4_row_vq import (
    _git_identity,
    _metrics,
    _quantize_baseline,
    _sha256,
    project_mxfp4,
)
from mfq.formats.nvq import NVQ2_E8
from mfq.quantize.mxfp import decode_mxfp4
from mfq.quantize.v4f_source import V4FCheckpoint


@dataclass(frozen=True)
class PqRate:
    entries: int
    vector_size: int
    index_bits: int
    scale_group_size: int
    scale_bits: int
    codebook_nbytes: int
    index_nbytes: int
    anchor_nbytes: int
    scale_nbytes: int
    payload_nbytes: int
    payload_bpw: float


def mxfp4_pq_rate(
    rows: int,
    columns: int,
    *,
    entries: int = 16384,
    vector_size: int = 8,
    index_bits: int = 14,
    scale_group_size: int = 24,
    scale_bits: int = 6,
) -> PqRate:
    if columns % vector_size:
        raise ValueError("columns must be divisible by the PQ vector size")
    if entries > 1 << index_bits:
        raise ValueError("codebook entries exceed the packed index range")
    if vector_size != 8:
        raise ValueError("the prototype currently stores 8-D MXFP4 centroids")
    if not 1 <= scale_bits <= 8:
        raise ValueError("scale_bits must be in [1,8]")
    vectors = rows * (columns // vector_size)
    groups = rows * math.ceil(columns / scale_group_size)
    # Eight E2M1 values occupy four bytes; every independently scaled learned
    # centroid carries one E8M0 exponent byte.
    codebook_nbytes = entries * 5
    index_nbytes = (vectors * index_bits + 7) // 8
    anchor_nbytes = rows * 2
    scale_nbytes = (groups * scale_bits + 7) // 8
    payload_nbytes = (
        codebook_nbytes + index_nbytes + anchor_nbytes + scale_nbytes
    )
    return PqRate(
        entries=entries,
        vector_size=vector_size,
        index_bits=index_bits,
        scale_group_size=scale_group_size,
        scale_bits=scale_bits,
        codebook_nbytes=codebook_nbytes,
        index_nbytes=index_nbytes,
        anchor_nbytes=anchor_nbytes,
        scale_nbytes=scale_nbytes,
        payload_nbytes=payload_nbytes,
        payload_bpw=8.0 * payload_nbytes / (rows * columns),
    )


@torch.inference_mode()
def project_mxfp4_codebook(
    value: torch.Tensor,
    *,
    return_storage: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Project 8-D rows to E2M1 with one independent E8M0 scale each."""

    if value.ndim != 2 or value.shape[1] != 8:
        raise ValueError("learned MXFP4-PQ codewords must have shape [K,8]")
    padded = torch.nn.functional.pad(value.to(torch.float32), (0, 24))
    projected, packed, scales = project_mxfp4(
        padded,
        row_chunk=256,
        return_storage=return_storage,
    )
    packed8 = packed[:, :4].contiguous() if packed is not None else None
    scale8 = scales[:, :1].contiguous() if scales is not None else None
    return projected[:, :8].contiguous(), packed8, scale8


@torch.inference_mode()
def _assign_vectors(
    vectors: torch.Tensor,
    vector_group: torch.Tensor,
    effective_scale: torch.Tensor,
    codebook: torch.Tensor,
    *,
    vector_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vector_chunk <= 0:
        raise ValueError("vector_chunk must be positive")
    codebook_t = codebook.transpose(0, 1).contiguous()
    codebook_norm = codebook.square().sum(dim=1)
    assignments = torch.empty(
        vectors.shape[0], device=vectors.device, dtype=torch.int64
    )
    errors = torch.empty(vectors.shape[0], device=vectors.device)
    for start in range(0, vectors.shape[0], vector_chunk):
        stop = min(int(vectors.shape[0]), start + vector_chunk)
        source = vectors[start:stop]
        scale = effective_scale[vector_group[start:stop]]
        distance = source @ codebook_t
        distance.mul_(-2.0 * scale[:, None])
        distance.add_(scale.square()[:, None] * codebook_norm[None, :])
        variable, assignment = distance.min(dim=1)
        assignments[start:stop] = assignment
        errors[start:stop] = (
            variable + source.square().sum(dim=1)
        ).clamp_min_(0)
    return assignments, errors


@torch.inference_mode()
def _initial_scales(
    source: torch.Tensor,
    *,
    group_size: int,
    scale_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, columns = (int(item) for item in source.shape)
    groups_per_row = math.ceil(columns / group_size)
    padded = torch.nn.functional.pad(
        source, (0, groups_per_row * group_size - columns)
    )
    continuous = (
        padded.reshape(rows, groups_per_row, group_size)
        .abs()
        .amax(dim=2)
        / 6.0
    )
    qmax = (1 << scale_bits) - 1
    anchor = (continuous.amax(dim=1) / qmax).to(torch.float16).to(torch.float32)
    safe_anchor = torch.where(anchor > 0, anchor, torch.ones_like(anchor))
    sub_scale = torch.clamp(
        torch.round(continuous / safe_anchor[:, None]), 0, qmax
    ).to(torch.uint8)
    effective = anchor[:, None] * sub_scale.to(torch.float32)
    return anchor, sub_scale, effective.reshape(-1)


@torch.inference_mode()
def _refit_quantized_scales(
    vectors: torch.Tensor,
    vector_group: torch.Tensor,
    assignments: torch.Tensor,
    codebook: torch.Tensor,
    anchor: torch.Tensor,
    *,
    groups_per_row: int,
    scale_bits: int,
    refinement_steps: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = codebook[assignments]
    vector_cross = (vectors * selected).sum(dim=1)
    vector_quad = selected.square().sum(dim=1)
    group_count = int(anchor.numel()) * groups_per_row
    group_cross = torch.zeros(group_count, device=vectors.device)
    group_quad = torch.zeros(group_count, device=vectors.device)
    group_cross.index_add_(0, vector_group, vector_cross)
    group_quad.index_add_(0, vector_group, vector_quad)
    group_row = torch.arange(
        anchor.numel(), device=vectors.device, dtype=torch.int64
    ).repeat_interleave(groups_per_row)
    qmax = (1 << scale_bits) - 1
    best_anchor = anchor
    best_q = torch.zeros(group_count, device=vectors.device)

    for _ in range(refinement_steps):
        denominator = best_anchor[group_row] * group_quad
        best_q = torch.where(
            denominator > 0,
            torch.round(group_cross / denominator),
            torch.zeros_like(group_cross),
        ).clamp_(0, qmax)
        row_numerator = torch.zeros_like(best_anchor)
        row_denominator = torch.zeros_like(best_anchor)
        row_numerator.index_add_(0, group_row, best_q * group_cross)
        row_denominator.index_add_(
            0, group_row, best_q.square() * group_quad
        )
        best_anchor = torch.where(
            row_denominator > 0,
            row_numerator / row_denominator,
            torch.zeros_like(row_numerator),
        ).clamp_min_(0)
        best_anchor = best_anchor.to(torch.float16).to(torch.float32)

    sub_scale = best_q.to(torch.uint8).reshape(-1, groups_per_row)
    effective = best_anchor[:, None] * sub_scale.to(torch.float32)
    return best_anchor, sub_scale, effective.reshape(-1)


@torch.inference_mode()
def quantize_mxfp4_pq(
    value: torch.Tensor,
    rate: PqRate,
    *,
    device: str | torch.device,
    seed: int,
    iterations: int,
    vector_chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if value.ndim != 2 or value.shape[1] % rate.vector_size:
        raise ValueError("MXFP4-PQ requires [rows,K] with K divisible by 8")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    target = torch.device(device)
    source = value.to(device=target, dtype=torch.float32).contiguous()
    rows, columns = (int(item) for item in source.shape)
    vectors_per_row = columns // rate.vector_size
    groups_per_row = math.ceil(columns / rate.scale_group_size)
    vectors = source.reshape(rows * vectors_per_row, rate.vector_size)
    local_vector = torch.arange(
        vectors_per_row, device=target, dtype=torch.int64
    )
    vector_group = (
        torch.arange(rows, device=target, dtype=torch.int64)[:, None]
        * groups_per_row
        + (local_vector // (rate.scale_group_size // rate.vector_size))[None, :]
    ).reshape(-1)

    anchor, sub_scale, effective_scale = _initial_scales(
        source,
        group_size=rate.scale_group_size,
        scale_bits=rate.scale_bits,
    )
    tiny = torch.finfo(torch.float32).tiny
    vector_scale = effective_scale[vector_group].clamp_min(tiny)
    normalized = vectors / vector_scale[:, None]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    initial = torch.randperm(vectors.shape[0], generator=generator)[
        : rate.entries
    ].to(target)
    codebook, _, _ = project_mxfp4_codebook(normalized[initial])
    del normalized

    assignments, errors = _assign_vectors(
        vectors,
        vector_group,
        effective_scale,
        codebook,
        vector_chunk=vector_chunk,
    )
    best_sse = float(errors.sum().cpu())
    best_codebook = codebook.clone()
    best_assignments = assignments.clone()
    best_anchor = anchor.clone()
    best_sub_scale = sub_scale.clone()
    trace: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "sse": best_sse,
            "changed": int(vectors.shape[0]),
            "empty_codewords": int(
                (torch.bincount(assignments, minlength=rate.entries) == 0)
                .sum()
                .cpu()
            ),
        }
    ]
    print(
        f"MXFP4-PQ iteration 0: SSE={best_sse:.9g}",
        file=sys.stderr,
        flush=True,
    )

    for iteration in range(1, iterations + 1):
        previous = assignments
        vector_scale = effective_scale[vector_group]
        denominator = torch.zeros(rate.entries, device=target)
        numerator = torch.zeros(
            (rate.entries, rate.vector_size), device=target
        )
        denominator.index_add_(0, assignments, vector_scale.square())
        numerator.index_add_(
            0, assignments, vectors * vector_scale[:, None]
        )
        means = numerator / denominator.clamp_min(tiny)[:, None]
        empty = torch.nonzero(denominator == 0, as_tuple=False).flatten()
        if empty.numel():
            worst = torch.topk(errors, k=int(empty.numel())).indices
            worst_scale = vector_scale[worst].clamp_min(tiny)
            means[empty] = vectors[worst] / worst_scale[:, None]
        codebook, _, _ = project_mxfp4_codebook(means)
        anchor, sub_scale, effective_scale = _refit_quantized_scales(
            vectors,
            vector_group,
            assignments,
            codebook,
            anchor,
            groups_per_row=groups_per_row,
            scale_bits=rate.scale_bits,
        )
        assignments, errors = _assign_vectors(
            vectors,
            vector_group,
            effective_scale,
            codebook,
            vector_chunk=vector_chunk,
        )
        sse = float(errors.sum().cpu())
        changed = int((assignments != previous).sum().cpu())
        counts = torch.bincount(assignments, minlength=rate.entries)
        empty_count = int((counts == 0).sum().cpu())
        trace.append(
            {
                "iteration": iteration,
                "sse": sse,
                "changed": changed,
                "empty_codewords": empty_count,
            }
        )
        print(
            f"MXFP4-PQ iteration {iteration}: SSE={sse:.9g}, "
            f"changed={changed}, empty={empty_count}",
            file=sys.stderr,
            flush=True,
        )
        if sse < best_sse:
            best_sse = sse
            best_codebook = codebook.clone()
            best_assignments = assignments.clone()
            best_anchor = anchor.clone()
            best_sub_scale = sub_scale.clone()
        if changed == 0:
            break

    best_effective = (
        best_anchor[:, None] * best_sub_scale.to(torch.float32)
    ).reshape(-1)
    reconstruction = (
        best_codebook[best_assignments]
        * best_effective[vector_group, None]
    ).reshape(rows, columns)
    final_counts = torch.bincount(best_assignments, minlength=rate.entries)
    physical, packed, scales = project_mxfp4_codebook(
        best_codebook, return_storage=True
    )
    assert packed is not None and scales is not None
    if not torch.equal(physical, best_codebook):
        raise RuntimeError("final learned codebook does not re-project exactly")
    packed32 = torch.zeros((rate.entries, 16), dtype=torch.uint8)
    packed32[:, :4] = packed
    decoded = decode_mxfp4(packed32.numpy(), scales.numpy(), device="cpu")[:, :8]
    if not torch.equal(decoded, best_codebook.cpu()):
        raise RuntimeError("physical learned MXFP4 codebook does not decode exactly")
    physical_keys = torch.cat((packed, scales), dim=1).numpy()
    metadata = {
        "trace": trace,
        "iterations_completed": len(trace) - 1,
        "used_codewords": int((final_counts > 0).sum().cpu()),
        "empty_codewords": int((final_counts == 0).sum().cpu()),
        "cluster_min_nonzero": int(final_counts[final_counts > 0].min().cpu()),
        "cluster_max": int(final_counts.max().cpu()),
        "unique_physical_codewords": int(
            np.unique(physical_keys, axis=0).shape[0]
        ),
        "physical_storage_roundtrip_verified": True,
    }
    return (
        reconstruction.cpu(),
        best_codebook.cpu(),
        best_assignments.cpu(),
        best_sub_scale.cpu(),
        metadata,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--vector-chunk", type=int, default=4096)
    parser.add_argument("--entries", type=int, default=16384)
    parser.add_argument("--index-bits", type=int, default=14)
    parser.add_argument("--scale-bits", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.time()
    script_root = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    checkpoint = V4FCheckpoint(model)
    reader = checkpoint.expert_source(args.layer, "gate_up")
    print(
        f"loading layer {args.layer} expert {args.expert} gate_up...",
        file=sys.stderr,
        flush=True,
    )
    source = reader.read_expert_rows(
        args.expert, 0, reader.rows_per_expert, device="cpu"
    ).contiguous()
    rows, columns = (int(item) for item in source.shape)
    rate = mxfp4_pq_rate(
        rows,
        columns,
        entries=args.entries,
        index_bits=args.index_bits,
        scale_bits=args.scale_bits,
    )
    baseline_budget = NVQ2_E8.payload_nbytes(rows, columns)
    if rate.payload_nbytes > baseline_budget:
        raise RuntimeError("learned MXFP4-PQ exceeds the NVQ2 payload budget")
    print(
        f"rate match: NVQ2={baseline_budget} bytes; "
        f"learned MXFP4-PQ={rate.payload_nbytes} bytes",
        file=sys.stderr,
        flush=True,
    )

    baseline_start = time.perf_counter()
    baseline_reconstruction, baseline_bytes = _quantize_baseline(
        source, NVQ2_E8.label, device=args.device
    )
    baseline_seconds = time.perf_counter() - baseline_start
    if baseline_bytes != baseline_budget:
        raise RuntimeError("NVQ2 baseline payload accounting changed")
    if str(torch.device(args.device)) == "mps":
        torch.mps.empty_cache()

    pq_start = time.perf_counter()
    reconstruction, _, _, _, metadata = quantize_mxfp4_pq(
        source,
        rate,
        device=args.device,
        seed=args.seed,
        iterations=args.iterations,
        vector_chunk=args.vector_chunk,
    )
    pq_seconds = time.perf_counter() - pq_start
    result = {
        "schema": 1,
        "experiment": "dsv4f-learned-8d-mxfp4-pq",
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
            "logical_dtype": "native MXFP4/E2M1+E8M0-g32",
        },
        "baseline": {
            "format": NVQ2_E8.label,
            "payload_nbytes": baseline_bytes,
            "payload_bpw": 8.0 * baseline_bytes / (rows * columns),
            "seconds": baseline_seconds,
            **_metrics(source, baseline_reconstruction),
        },
        "learned_mxfp4_pq": {
            "format": "learned-signed-8D-MXFP4-PQ",
            **rate.__dict__,
            "budget_slack_nbytes": baseline_budget - rate.payload_nbytes,
            "seconds": pq_seconds,
            **_metrics(source, reconstruction),
            **metadata,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
