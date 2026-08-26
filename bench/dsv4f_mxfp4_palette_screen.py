#!/usr/bin/env python3
"""Freeze a larger scalar-palette catalog from sampled hybrid-SQ partitions."""

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
    _row_candidate_exponents,
    _unpack_source,
)
from bench.dsv4f_mxfp4_hybrid_sq import (
    _hybrid_labeled_errors,
    _solve_hybrid_row,
)
from bench.dsv4f_mxfp4_row_vq import _git_identity, _sha256
from bench.dsv4f_mxfp4_xor_sq import _candidate_xor_errors
from mfq.quantize.v4f_source import V4FCheckpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=256)
    parser.add_argument("--catalog-size", type=int, default=128)
    parser.add_argument("--palette-chunk", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _greedy_catalog(partition_errors: np.ndarray, size: int) -> list[int]:
    selected = [int(item) for item in FIXED16_PALETTE_IDS]
    if size < len(selected) or size > len(PALETTE_VALUES):
        raise ValueError("invalid target catalog size")
    covered = partition_errors[:, selected].min(axis=1)
    available = np.ones(len(PALETTE_VALUES), dtype=np.bool_)
    available[selected] = False
    while len(selected) < size:
        improvement = np.maximum(
            covered[:, None] - partition_errors,
            0.0,
        ).sum(axis=0)
        improvement[~available] = -np.inf
        chosen = int(improvement.argmax())
        selected.append(chosen)
        available[chosen] = False
        covered = np.minimum(covered, partition_errors[:, chosen])
    return selected


def main() -> None:
    args = _parse_args()
    started = time.time()
    script_root = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    checkpoint = V4FCheckpoint(model)
    packed, source_scales = _raw_gate_up_native(checkpoint, args.layer, args.expert)
    source_nibbles, _ = _unpack_source(packed, source_scales)
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
    for sample_index, row in enumerate(row_indices):
        solution = _solve_hybrid_row(
            source_nibbles[row],
            source_scales[row],
            exponent_radius=0,
            maximum_steps=10,
        )
        candidate_exponents = _row_candidate_exponents(source_scales[row], 0)
        active_tags = [tag for tag in range(4) if bool((solution.block_tags == tag).any())]
        local_errors = {tag: np.empty(len(PALETTE_VALUES), dtype=np.float64) for tag in active_tags}
        for start in range(0, len(PALETTE_VALUES), args.palette_chunk):
            palette_ids = np.arange(
                start,
                min(start + args.palette_chunk, len(PALETTE_VALUES)),
                dtype=np.int16,
            )
            state_scales = np.repeat(
                np.asarray(candidate_exponents, dtype=np.uint8),
                len(palette_ids),
            )
            state_palette_ids = np.tile(palette_ids, len(candidate_exponents))
            errors = _hybrid_labeled_errors(
                _candidate_xor_errors(
                    source_nibbles[row],
                    source_scales[row],
                    state_scales,
                    state_palette_ids,
                )
            )
            for tag in active_tags:
                selected_blocks = solution.block_tags == tag
                state_error = errors[tag, :, selected_blocks].sum(axis=0)
                palette_error = state_error.reshape(len(candidate_exponents), len(palette_ids)).min(
                    axis=0
                )
                local_errors[tag][start : start + len(palette_ids)] = palette_error
        for tag in active_tags:
            winner_counts[int(local_errors[tag].argmin())] += 1
            partition_errors.append(local_errors[tag])
        if (sample_index + 1) % 16 == 0 or sample_index + 1 == len(row_indices):
            print(
                f"screened {sample_index + 1}/{len(row_indices)} rows",
                file=sys.stderr,
                flush=True,
            )
    errors_array = np.stack(partition_errors)
    selected = _greedy_catalog(errors_array, args.catalog_size)
    fixed16_error = float(errors_array[:, FIXED16_PALETTE_IDS].min(axis=1).sum())
    selected_error = float(errors_array[:, selected].min(axis=1).sum())
    result = {
        "schema": 1,
        "experiment": "dsv4f-hybrid-mxfp4-sq-palette-catalog-screen",
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
            "source_partitions": "fixed16-hybrid-four-state",
            "full_palette_catalog_size": len(PALETTE_VALUES),
            "partition_count": len(partition_errors),
            "candidate_scale_range": "source-row-range-only",
            "selection": "greedy-partition-error-coverage-seeded-by-fixed16",
            "fixed16_partition_sse": fixed16_error,
            "selected_partition_sse": selected_error,
            "relative_partition_sse_delta_percent": 100.0 * (selected_error / fixed16_error - 1.0),
        },
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
