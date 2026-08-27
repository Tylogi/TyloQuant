#!/usr/bin/env python3
"""Freeze a 32-entry SQ2 palette catalog from eight-state expert-0 partitions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from bench.dsv4f_mxfp4_adaptive_sq import (
    FIXED16_PALETTE_IDS,
    PALETTE_VALUES,
    _raw_gate_up_native,
    _unpack_source,
)
from bench.dsv4f_mxfp4_row_vq import _git_identity, _sha256
from bench.dsv4f_mxfp4_sq2 import (
    SCREENED32_PALETTE_IDS,
    _solve_sq2_row,
)
from bench.dsv4f_mxfp4_xor_sq import _candidate_xor_errors
from mfq.quantize.v4f_source import V4FCheckpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--palette-chunk", type=int, default=64)
    parser.add_argument("--maximum-refinement-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _greedy_catalog(
    partition_errors: np.ndarray,
    *,
    seed: np.ndarray = FIXED16_PALETTE_IDS,
    size: int = 32,
) -> list[int]:
    selected = [int(item) for item in seed]
    if not 0 < len(selected) <= size <= partition_errors.shape[1]:
        raise ValueError("invalid palette catalog selection geometry")
    covered = partition_errors[:, selected].min(axis=1)
    available = np.ones(partition_errors.shape[1], dtype=np.bool_)
    available[selected] = False
    while len(selected) < size:
        improvement = np.maximum(covered[:, None] - partition_errors, 0.0).sum(axis=0)
        improvement[~available] = -np.inf
        chosen = int(improvement.argmax())
        selected.append(chosen)
        available[chosen] = False
        covered = np.minimum(covered, partition_errors[:, chosen])
    return selected


def main() -> None:
    args = _parse_args()
    if args.sample_rows <= 0 or args.palette_chunk <= 0 or args.maximum_refinement_steps <= 0:
        raise ValueError("screen controls must be positive")
    started = time.time()
    script_root = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    checkpoint = V4FCheckpoint(model)
    packed, source_scales = _raw_gate_up_native(checkpoint, args.layer, args.expert)
    source_nibbles, _ = _unpack_source(packed, source_scales)
    matrix_scale_base = int(source_scales.min())
    if int(source_scales.max()) > matrix_scale_base + 3:
        raise ValueError("source E8M0 range exceeds the four-value matrix scale window")
    row_indices = np.unique(
        np.linspace(
            0,
            source_nibbles.shape[0] - 1,
            num=args.sample_rows,
            dtype=np.int64,
        )
    )
    partition_errors: list[np.ndarray] = []
    winner_counts = np.zeros(len(PALETTE_VALUES), dtype=np.int64)
    scale_values = np.arange(matrix_scale_base, matrix_scale_base + 4, dtype=np.uint8)
    for sample_index, row in enumerate(row_indices):
        solution = _solve_sq2_row(
            source_nibbles[row],
            source_scales[row],
            matrix_scale_base=matrix_scale_base,
            maximum_steps=args.maximum_refinement_steps,
            palette_ids=SCREENED32_PALETTE_IDS,
        )
        active_tags = [tag for tag in range(8) if bool((solution.block_tags == tag).any())]
        local_errors = {tag: np.empty(len(PALETTE_VALUES), dtype=np.float64) for tag in active_tags}
        for start in range(0, len(PALETTE_VALUES), args.palette_chunk):
            palette_ids = np.arange(
                start,
                min(start + args.palette_chunk, len(PALETTE_VALUES)),
                dtype=np.int16,
            )
            state_scales = np.repeat(scale_values, len(palette_ids))
            state_palette_ids = np.tile(palette_ids, len(scale_values))
            errors = _candidate_xor_errors(
                source_nibbles[row],
                source_scales[row],
                state_scales,
                state_palette_ids,
            )
            for tag in active_tags:
                selected_blocks = solution.block_tags == tag
                state_error = errors[tag & 3][:, selected_blocks].sum(axis=1)
                palette_error = state_error.reshape(len(scale_values), len(palette_ids)).min(axis=0)
                local_errors[tag][start : start + len(palette_ids)] = palette_error
        for tag in active_tags:
            winner_counts[int(local_errors[tag].argmin())] += 1
            partition_errors.append(local_errors[tag])
        if (sample_index + 1) % 8 == 0 or sample_index + 1 == len(row_indices):
            print(
                f"screened {sample_index + 1}/{len(row_indices)} rows",
                file=sys.stderr,
                flush=True,
            )
    errors_array = np.stack(partition_errors)
    selected = _greedy_catalog(errors_array)
    fixed16_error = float(errors_array[:, FIXED16_PALETTE_IDS].min(axis=1).sum())
    prior32_error = float(errors_array[:, SCREENED32_PALETTE_IDS].min(axis=1).sum())
    selected_error = float(errors_array[:, selected].min(axis=1).sum())
    result = {
        "schema": 1,
        "experiment": "dsv4f-mxfp4-sq2-palette-catalog-screen",
        "created_unix": started,
        "workspace": _git_identity(script_root),
        "source": {
            "model": str(model),
            "config_sha256": _sha256(model / "config.json"),
            "index_sha256": _sha256(model / "model.safetensors.index.json"),
            "layer": args.layer,
            "expert": args.expert,
            "projection": "gate_up",
            "sample_rows_requested": args.sample_rows,
            "sample_rows_actual": len(row_indices),
            "row_indices": row_indices.tolist(),
        },
        "screen": {
            "source_partitions": "screened32-MXFP4-SQ2",
            "full_palette_catalog_size": len(PALETTE_VALUES),
            "partition_count": len(partition_errors),
            "matrix_scale_base": matrix_scale_base,
            "selection": "greedy-partition-error-coverage-seeded-by-fixed16",
            "fixed16_partition_sse": fixed16_error,
            "prior_screened32_partition_sse": prior32_error,
            "selected32_partition_sse": selected_error,
            "selected32_delta_vs_fixed16_percent": 100.0 * (selected_error / fixed16_error - 1.0),
            "selected32_delta_vs_prior32_percent": 100.0 * (selected_error / prior32_error - 1.0),
        },
        "prior_screened32_palette_ids": SCREENED32_PALETTE_IDS.tolist(),
        "selected_palette_ids": selected,
        "selected_palette_values": PALETTE_VALUES[selected].tolist(),
        "partition_winner_counts": [
            {"palette_id": int(index), "count": int(winner_counts[index])}
            for index in np.flatnonzero(winner_counts)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
