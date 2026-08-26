#!/usr/bin/env python3
"""Search a native-MXFP4-aware two-bit scalar transcoder.

The source is already block-32 MXFP4.  Every output row stores two scalar
states; each state is one E8M0 scale plus one four-level E2M1 palette.  A
single bit per native block selects the row state, and every weight stores a
two-bit scalar symbol inside the selected palette.  The output therefore
remains exactly representable as native MXFP4 and contains no FP16 scale,
centroid, or vector codebook.

Three palette modes are useful:

* ``fixed4`` selects from four format-level palettes frozen after the expert-0
  design screen.  Its two-state solve is exact within that fixed catalog.
* ``fixed16`` expands that frozen catalog to the 16 palettes exercised most by
  the expert-0 full search.  It still has an exact two-state solve and needs
  only four stored palette-ID bits per row state.
* ``full`` selects from the implicit lexicographic catalog of all 1,365
  four-value subsets of the 15 distinct E2M1 values.  Deterministic hard EM
  refines two row states and serves as the higher scalar-quality bound.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import sys
import time
from collections import Counter
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
)
from mfq.formats.nvq import NVQ2_E8
from mfq.quantize.mxfp import decode_mxfp4
from mfq.quantize.v4f_source import V4FCheckpoint

E2M1_VALUES = np.asarray(
    (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
    dtype=np.float64,
)
E2M1_NIBBLES = np.asarray(
    (0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7),
    dtype=np.uint8,
)
NIBBLE_VALUES = np.asarray(
    (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
    dtype=np.float64,
)
NIBBLE_TO_LEVEL = np.asarray(
    [int(np.flatnonzero(value == E2M1_VALUES)[0]) for value in NIBBLE_VALUES],
    dtype=np.uint8,
)
PALETTE_LEVEL_INDICES = np.asarray(
    list(itertools.combinations(range(len(E2M1_VALUES)), 4)),
    dtype=np.int16,
)
PALETTE_VALUES = E2M1_VALUES[PALETTE_LEVEL_INDICES]
PALETTE_NIBBLES = E2M1_NIBBLES[PALETTE_LEVEL_INDICES]

# Four fixed scalar modes selected once on the expert-0 design matrix.  They
# are format constants, not per-matrix stored values.  Their full-catalog IDs
# make the mapping auditable and deterministic.
FIXED4_PALETTE_IDS = np.asarray((518, 558, 767, 966), dtype=np.int16)
FIXED16_PALETTE_IDS = np.asarray(
    (
        120,
        127,
        187,
        504,
        512,
        518,
        547,
        548,
        558,
        562,
        592,
        767,
        806,
        833,
        966,
        967,
    ),
    dtype=np.int16,
)


def _palette_ids_for_mode(mode: str) -> np.ndarray:
    if mode == "fixed4":
        return FIXED4_PALETTE_IDS
    if mode == "fixed16":
        return FIXED16_PALETTE_IDS
    if mode == "full":
        return np.arange(len(PALETTE_VALUES), dtype=np.int16)
    raise ValueError(f"unknown adaptive MXFP4-SQ mode: {mode}")


def _palette_id_bits_for_mode(mode: str) -> int:
    return {"fixed4": 2, "fixed16": 4, "full": 11}[mode]


@dataclass(frozen=True)
class AdaptiveSqRate:
    bits_per_symbol: int
    native_block_size: int
    block_selector_bits: int
    palette_id_bits: int
    states_per_row: int
    symbol_nbytes: int
    block_selector_nbytes: int
    state_scale_nbytes: int
    state_palette_nbytes: int
    payload_nbytes: int
    payload_bpw: float


@dataclass(frozen=True)
class RowSolution:
    error_sse: float
    state_scale_raw: tuple[int, int]
    state_palette_ids: tuple[int, int]
    selectors: np.ndarray
    refinement_steps: int


@dataclass(frozen=True)
class AdaptiveSqEncoding:
    state_scale_raw: np.ndarray
    state_palette_ids: np.ndarray
    block_selectors: np.ndarray
    symbols: np.ndarray
    packed_mxfp4: np.ndarray
    native_scale_raw: np.ndarray
    searched_sse: float


def adaptive_mxfp4_sq_rate(
    rows: int,
    columns: int,
    *,
    palette_id_bits: int,
) -> AdaptiveSqRate:
    """Return exact packed payload for two states per row."""

    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("adaptive MXFP4-SQ requires positive block-32 matrices")
    if palette_id_bits <= 0:
        raise ValueError("palette_id_bits must be positive")
    bits_per_symbol = 2
    native_block_size = 32
    block_selector_bits = 1
    states_per_row = 2
    weights = rows * columns
    blocks = weights // native_block_size
    symbol_nbytes = (weights * bits_per_symbol + 7) // 8
    block_selector_nbytes = (blocks * block_selector_bits + 7) // 8
    state_scale_nbytes = rows * states_per_row
    state_palette_nbytes = (rows * states_per_row * palette_id_bits + 7) // 8
    payload_nbytes = (
        symbol_nbytes + block_selector_nbytes + state_scale_nbytes + state_palette_nbytes
    )
    return AdaptiveSqRate(
        bits_per_symbol=bits_per_symbol,
        native_block_size=native_block_size,
        block_selector_bits=block_selector_bits,
        palette_id_bits=palette_id_bits,
        states_per_row=states_per_row,
        symbol_nbytes=symbol_nbytes,
        block_selector_nbytes=block_selector_nbytes,
        state_scale_nbytes=state_scale_nbytes,
        state_palette_nbytes=state_palette_nbytes,
        payload_nbytes=payload_nbytes,
        payload_bpw=8.0 * payload_nbytes / weights,
    )


def _raw_gate_up_native(
    checkpoint: V4FCheckpoint,
    layer: int,
    expert: int,
) -> tuple[np.ndarray, np.ndarray]:
    source = checkpoint.expert_source(layer, "gate_up")
    packed_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    for part in ("w1", "w3"):
        packed, raw_scale = source._raw_part(expert, part)
        packed_parts.append(np.asarray(packed, dtype=np.uint8).reshape(2048, 2048))
        scale_parts.append(np.asarray(raw_scale, dtype=np.uint8).reshape(2048, 128))
    return np.concatenate(packed_parts), np.concatenate(scale_parts)


def _unpack_source(
    packed: np.ndarray,
    scale_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scale_raw, dtype=np.uint8)
    if raw.ndim != 2:
        raise ValueError("packed MXFP4 source must be rank 2")
    rows, packed_columns = raw.shape
    columns = packed_columns * 2
    blocks = columns // 32
    if tuple(scales.shape) != (rows, blocks):
        raise ValueError("source scale geometry does not match packed MXFP4")
    block_bytes = raw.reshape(rows, blocks, 16)
    nibbles = np.empty((rows, blocks, 32), dtype=np.uint8)
    nibbles[:, :, 0::2] = block_bytes & 0x0F
    nibbles[:, :, 1::2] = block_bytes >> 4
    levels = NIBBLE_TO_LEVEL[nibbles]
    histograms = np.empty((rows, blocks, len(E2M1_VALUES)), dtype=np.uint8)
    for level in range(len(E2M1_VALUES)):
        histograms[:, :, level] = np.count_nonzero(levels == level, axis=2)
    return nibbles, histograms


def _distortion_tables(
    source_scale_raw: np.ndarray,
    *,
    exponent_radius: int,
) -> dict[tuple[int, int], np.ndarray]:
    source_scales = np.asarray(source_scale_raw, dtype=np.uint8)
    source_min = int(source_scales.min())
    source_max = int(source_scales.max())
    reconstruction_min = max(0, source_min - exponent_radius)
    reconstruction_max = min(254, source_max + exponent_radius)
    result: dict[tuple[int, int], np.ndarray] = {}
    for source_raw in range(source_min, source_max + 1):
        source_values = E2M1_VALUES * math.ldexp(1.0, source_raw - 127)
        for reconstruction_raw in range(reconstruction_min, reconstruction_max + 1):
            reconstruction_values = PALETTE_VALUES * math.ldexp(1.0, reconstruction_raw - 127)
            result[(source_raw, reconstruction_raw)] = np.square(
                source_values[:, None, None] - reconstruction_values[None, :, :]
            ).min(axis=2)
    return result


def _row_candidate_exponents(
    source_scale_row: np.ndarray,
    exponent_radius: int,
) -> list[int]:
    return list(
        range(
            max(0, int(source_scale_row.min()) - exponent_radius),
            min(254, int(source_scale_row.max()) + exponent_radius) + 1,
        )
    )


def _state_block_errors(
    histogram_row: np.ndarray,
    source_scale_row: np.ndarray,
    reconstruction_scale_raw: int,
    palette_id: int,
    distortions: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    result = np.empty(histogram_row.shape[0], dtype=np.float64)
    for source_raw in np.unique(source_scale_row):
        mask = source_scale_row == source_raw
        result[mask] = (
            histogram_row[mask]
            @ distortions[(int(source_raw), reconstruction_scale_raw)][:, palette_id]
        )
    return result


def _best_state_for_partition(
    histogram_row: np.ndarray,
    source_scale_row: np.ndarray,
    selected_blocks: np.ndarray,
    candidate_exponents: list[int],
    palette_ids: np.ndarray,
    distortions: dict[tuple[int, int], np.ndarray],
) -> tuple[int, int] | None:
    if not bool(selected_blocks.any()):
        return None
    best_error = float("inf")
    best_state: tuple[int, int] | None = None
    for reconstruction_raw in candidate_exponents:
        palette_error = np.zeros(len(palette_ids), dtype=np.float64)
        for source_raw in np.unique(source_scale_row):
            mask = selected_blocks & (source_scale_row == source_raw)
            if bool(mask.any()):
                counts = histogram_row[mask].sum(axis=0, dtype=np.float64)
                palette_error += (
                    counts @ distortions[(int(source_raw), reconstruction_raw)][:, palette_ids]
                )
        local = int(palette_error.argmin())
        error = float(palette_error[local])
        if error < best_error:
            best_error = error
            best_state = (reconstruction_raw, int(palette_ids[local]))
    return best_state


def _exact_two_state_row(
    histogram_row: np.ndarray,
    source_scale_row: np.ndarray,
    candidate_exponents: list[int],
    palette_ids: np.ndarray,
    distortions: dict[tuple[int, int], np.ndarray],
) -> RowSolution:
    states = [
        (scale_raw, int(palette_id))
        for scale_raw in candidate_exponents
        for palette_id in palette_ids
    ]
    errors = np.stack(
        [
            _state_block_errors(
                histogram_row,
                source_scale_row,
                scale_raw,
                palette_id,
                distortions,
            )
            for scale_raw, palette_id in states
        ],
        axis=1,
    )
    best_error = float("inf")
    best_pair = (0, 1)
    for left in range(len(states) - 1):
        pair_error = np.minimum(errors[:, left, None], errors[:, left + 1 :]).sum(axis=0)
        right_offset = int(pair_error.argmin())
        error = float(pair_error[right_offset])
        if error < best_error:
            best_error = error
            best_pair = (left, left + 1 + right_offset)
    left, right = best_pair
    selectors = errors[:, right] < errors[:, left]
    return RowSolution(
        error_sse=best_error,
        state_scale_raw=(states[left][0], states[right][0]),
        state_palette_ids=(states[left][1], states[right][1]),
        selectors=selectors,
        refinement_steps=0,
    )


def _balanced_partition(partition: np.ndarray) -> np.ndarray:
    result = np.asarray(partition, dtype=np.bool_).copy()
    if not bool(result.any()) or bool(result.all()):
        result = np.arange(result.size) % 2 == 0
    return result


def _refine_two_state_row(
    histogram_row: np.ndarray,
    source_scale_row: np.ndarray,
    initial_selectors: np.ndarray,
    candidate_exponents: list[int],
    distortions: dict[tuple[int, int], np.ndarray],
    *,
    maximum_steps: int,
) -> RowSolution | None:
    palette_ids = np.arange(len(PALETTE_VALUES), dtype=np.int16)
    selectors = _balanced_partition(initial_selectors)
    best: RowSolution | None = None
    for step in range(1, maximum_steps + 1):
        state0 = _best_state_for_partition(
            histogram_row,
            source_scale_row,
            ~selectors,
            candidate_exponents,
            palette_ids,
            distortions,
        )
        state1 = _best_state_for_partition(
            histogram_row,
            source_scale_row,
            selectors,
            candidate_exponents,
            palette_ids,
            distortions,
        )
        if state0 is None or state1 is None:
            return None
        error0 = _state_block_errors(
            histogram_row,
            source_scale_row,
            state0[0],
            state0[1],
            distortions,
        )
        error1 = _state_block_errors(
            histogram_row,
            source_scale_row,
            state1[0],
            state1[1],
            distortions,
        )
        updated = error1 < error0
        error = float(np.minimum(error0, error1).sum())
        candidate = RowSolution(
            error_sse=error,
            state_scale_raw=(state0[0], state1[0]),
            state_palette_ids=(state0[1], state1[1]),
            selectors=updated.copy(),
            refinement_steps=step,
        )
        if best is None or candidate.error_sse < best.error_sse:
            best = candidate
        if np.array_equal(updated, selectors):
            break
        selectors = _balanced_partition(updated)
    return best


def _full_catalog_row(
    histogram_row: np.ndarray,
    source_scale_row: np.ndarray,
    candidate_exponents: list[int],
    distortions: dict[tuple[int, int], np.ndarray],
    *,
    maximum_steps: int,
) -> RowSolution:
    fixed = _exact_two_state_row(
        histogram_row,
        source_scale_row,
        candidate_exponents,
        FIXED16_PALETTE_IDS,
        distortions,
    )
    signed_sum = histogram_row @ E2M1_VALUES
    normalized_energy = histogram_row @ np.square(E2M1_VALUES)
    starts = [
        fixed.selectors,
        source_scale_row > np.median(source_scale_row),
        signed_sum > np.median(signed_sum),
        normalized_energy > np.median(normalized_energy),
        np.arange(histogram_row.shape[0]) % 2 == 0,
        np.arange(histogram_row.shape[0]) % 4 < 2,
    ]
    unique_starts: list[np.ndarray] = []
    seen: set[bytes] = set()
    for start in starts:
        balanced = _balanced_partition(start)
        key = np.packbits(balanced).tobytes()
        inverse_key = np.packbits(~balanced).tobytes()
        if key in seen or inverse_key in seen:
            continue
        seen.add(key)
        unique_starts.append(balanced)
    candidates = [fixed]
    for start in unique_starts:
        candidate = _refine_two_state_row(
            histogram_row,
            source_scale_row,
            start,
            candidate_exponents,
            distortions,
            maximum_steps=maximum_steps,
        )
        if candidate is not None:
            candidates.append(candidate)
    return min(candidates, key=lambda item: item.error_sse)


def _materialize_encoding(
    source_nibbles: np.ndarray,
    source_scale_raw: np.ndarray,
    solutions: list[RowSolution],
    *,
    row_chunk: int,
) -> AdaptiveSqEncoding:
    rows, blocks, block_size = source_nibbles.shape
    state_scale_raw = np.asarray(
        [solution.state_scale_raw for solution in solutions], dtype=np.uint8
    )
    state_palette_ids = np.asarray(
        [solution.state_palette_ids for solution in solutions], dtype=np.int16
    )
    block_selectors = np.stack([solution.selectors for solution in solutions]).astype(
        np.bool_, copy=False
    )
    symbols = np.empty((rows, blocks, block_size), dtype=np.uint8)
    output_nibbles = np.empty_like(symbols)
    native_scale_raw = np.empty((rows, blocks), dtype=np.uint8)

    for start in range(0, rows, row_chunk):
        stop = min(rows, start + row_chunk)
        local_rows = np.arange(stop - start)[:, None]
        selected_state = block_selectors[start:stop].astype(np.int64)
        selected_scale = state_scale_raw[start:stop][local_rows, selected_state]
        selected_palette = state_palette_ids[start:stop][local_rows, selected_state]
        palette_values = PALETTE_VALUES[selected_palette]
        palette_nibbles = PALETTE_NIBBLES[selected_palette]
        target = NIBBLE_VALUES[source_nibbles[start:stop]] * np.exp2(
            source_scale_raw[start:stop, :, None].astype(np.int16) - 127
        )
        reconstruction_levels = palette_values * np.exp2(
            selected_scale[:, :, None].astype(np.int16) - 127
        )
        distance = np.square(target[:, :, :, None] - reconstruction_levels[:, :, None, :])
        local_symbols = distance.argmin(axis=3).astype(np.uint8)
        local_nibbles = np.take_along_axis(
            palette_nibbles[:, :, None, :],
            local_symbols[:, :, :, None],
            axis=3,
        )[:, :, :, 0]
        symbols[start:stop] = local_symbols
        output_nibbles[start:stop] = local_nibbles
        native_scale_raw[start:stop] = selected_scale

    packed_blocks = output_nibbles[:, :, 0::2] | (output_nibbles[:, :, 1::2] << 4)
    packed_mxfp4 = packed_blocks.reshape(rows, blocks * 16)
    return AdaptiveSqEncoding(
        state_scale_raw=state_scale_raw,
        state_palette_ids=state_palette_ids,
        block_selectors=block_selectors,
        symbols=symbols.reshape(rows, blocks * block_size),
        packed_mxfp4=packed_mxfp4,
        native_scale_raw=native_scale_raw,
        searched_sse=float(sum(solution.error_sse for solution in solutions)),
    )


def _encoding_metadata(
    encoding: AdaptiveSqEncoding,
    source_scale_raw: np.ndarray,
    *,
    mode: str,
    exponent_radius: int,
    refinement_steps: list[int],
) -> dict[str, Any]:
    palette_counts = Counter(int(item) for item in encoding.state_palette_ids.flat)
    source_unique_counts = Counter(int(np.unique(row).size) for row in source_scale_raw)
    return {
        "palette_mode": mode,
        "palette_catalog_size": len(_palette_ids_for_mode(mode)),
        "implicit_full_e2m1_palette_catalog": mode == "full",
        "fixed4_full_catalog_ids": FIXED4_PALETTE_IDS.tolist(),
        "fixed4_palette_values": PALETTE_VALUES[FIXED4_PALETTE_IDS].tolist(),
        "fixed16_full_catalog_ids": FIXED16_PALETTE_IDS.tolist(),
        "fixed16_palette_values": PALETTE_VALUES[FIXED16_PALETTE_IDS].tolist(),
        "state_layout": "two-row-states:(E8M0-scale,palette-id)",
        "block_selector_bits": 1,
        "scale_dtype": "E8M0",
        "scale_exponent_radius": exponent_radius,
        "optimizer": (
            "exact-two-state-enumeration"
            if mode in {"fixed4", "fixed16"}
            else "deterministic-multistart-hard-em"
        ),
        "optimizer_is_global_within_catalog": mode in {"fixed4", "fixed16"},
        "maximum_refinement_steps_used": max(refinement_steps, default=0),
        "mean_refinement_steps": float(np.mean(refinement_steps)),
        "state_scale_byte_min": int(encoding.state_scale_raw.min()),
        "state_scale_byte_max": int(encoding.state_scale_raw.max()),
        "same_scale_two_state_row_fraction": float(
            np.mean(encoding.state_scale_raw[:, 0] == encoding.state_scale_raw[:, 1])
        ),
        "high_state_selector_fraction": float(encoding.block_selectors.mean()),
        "effective_scale_matches_source_fraction": float(
            np.mean(encoding.native_scale_raw == source_scale_raw)
        ),
        "source_unique_scale_count_rows": {
            str(key): value for key, value in sorted(source_unique_counts.items())
        },
        "unique_palette_ids": len(palette_counts),
        "top_state_palette_ids": [
            {
                "palette_id": palette_id,
                "count": count,
                "values": PALETTE_VALUES[palette_id].tolist(),
                "nibbles": PALETTE_NIBBLES[palette_id].tolist(),
            }
            for palette_id, count in palette_counts.most_common(32)
        ],
        "learned_vector_codebook": False,
        "stored_fp16_scale_or_centroid": False,
        "physical_storage_roundtrip_verified": True,
        "final_values_are_native_block32_mxfp4": True,
    }


@torch.inference_mode()
def quantize_adaptive_mxfp4_sq(
    packed: np.ndarray,
    source_scale_raw: np.ndarray,
    *,
    mode: str,
    exponent_radius: int = 2,
    maximum_refinement_steps: int = 20,
    row_chunk: int = 64,
    progress: bool = False,
) -> tuple[torch.Tensor, AdaptiveSqEncoding, dict[str, Any]]:
    """Quantize native MXFP4 bytes to two-state, two-bit scalar storage."""

    if mode not in {"fixed4", "fixed16", "full"}:
        raise ValueError("mode must be fixed4, fixed16, or full")
    if exponent_radius < 0 or maximum_refinement_steps <= 0 or row_chunk <= 0:
        raise ValueError("invalid search control")
    source_nibbles, histograms = _unpack_source(packed, source_scale_raw)
    source_scales = np.asarray(source_scale_raw, dtype=np.uint8)
    distortions = _distortion_tables(source_scales, exponent_radius=exponent_radius)
    solutions: list[RowSolution] = []
    for row in range(histograms.shape[0]):
        candidate_exponents = _row_candidate_exponents(source_scales[row], exponent_radius)
        if mode in {"fixed4", "fixed16"}:
            solution = _exact_two_state_row(
                histograms[row],
                source_scales[row],
                candidate_exponents,
                _palette_ids_for_mode(mode),
                distortions,
            )
        else:
            solution = _full_catalog_row(
                histograms[row],
                source_scales[row],
                candidate_exponents,
                distortions,
                maximum_steps=maximum_refinement_steps,
            )
        solutions.append(solution)
        if progress and ((row + 1) % 512 == 0 or row + 1 == histograms.shape[0]):
            print(
                f"{mode}: solved {row + 1}/{histograms.shape[0]} rows",
                file=sys.stderr,
                flush=True,
            )
    encoding = _materialize_encoding(
        source_nibbles,
        source_scales,
        solutions,
        row_chunk=row_chunk,
    )
    reconstruction = decode_mxfp4(
        encoding.packed_mxfp4,
        encoding.native_scale_raw,
        device="cpu",
    ).contiguous()
    source = decode_mxfp4(packed, source_scales, device="cpu")
    measured_sse = float(
        (source.to(torch.float64) - reconstruction.to(torch.float64)).square().sum()
    )
    if not math.isclose(
        measured_sse,
        encoding.searched_sse,
        rel_tol=2e-9,
        abs_tol=1e-10,
    ):
        raise RuntimeError(
            "adaptive MXFP4-SQ search/materialization SSE mismatch: "
            f"{encoding.searched_sse} != {measured_sse}"
        )
    metadata = _encoding_metadata(
        encoding,
        source_scales,
        mode=mode,
        exponent_radius=exponent_radius,
        refinement_steps=[solution.refinement_steps for solution in solutions],
    )
    return reconstruction, encoding, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--mode",
        choices=("fixed4", "fixed16", "full", "all"),
        default="all",
    )
    parser.add_argument("--exponent-radius", type=int, default=2)
    parser.add_argument("--maximum-refinement-steps", type=int, default=20)
    parser.add_argument("--row-chunk", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.time()
    script_root = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    checkpoint = V4FCheckpoint(model)
    print(
        f"loading native layer {args.layer} expert {args.expert} gate_up...",
        file=sys.stderr,
        flush=True,
    )
    packed, source_scale_raw = _raw_gate_up_native(checkpoint, args.layer, args.expert)
    source = decode_mxfp4(packed, source_scale_raw, device="cpu").contiguous()
    rows, columns = (int(item) for item in source.shape)
    baseline_budget = NVQ2_E8.payload_nbytes(rows, columns)
    baseline_start = time.perf_counter()
    baseline_reconstruction, baseline_bytes = _quantize_baseline(
        source, NVQ2_E8.label, device=args.device
    )
    baseline_seconds = time.perf_counter() - baseline_start
    if baseline_bytes != baseline_budget:
        raise RuntimeError("NVQ2 baseline payload accounting changed")
    if str(torch.device(args.device)) == "mps":
        torch.mps.empty_cache()

    modes = ("fixed4", "fixed16", "full") if args.mode == "all" else (args.mode,)
    candidates: list[dict[str, Any]] = []
    for mode in modes:
        palette_bits = _palette_id_bits_for_mode(mode)
        rate = adaptive_mxfp4_sq_rate(rows, columns, palette_id_bits=palette_bits)
        if rate.payload_nbytes > baseline_budget:
            raise RuntimeError(f"{mode} adaptive SQ exceeds NVQ2 payload")
        print(
            f"searching {mode}: {rate.payload_nbytes} bytes / {rate.payload_bpw:.9f} BPW...",
            file=sys.stderr,
            flush=True,
        )
        candidate_start = time.perf_counter()
        reconstruction, _, metadata = quantize_adaptive_mxfp4_sq(
            packed,
            source_scale_raw,
            mode=mode,
            exponent_radius=args.exponent_radius,
            maximum_refinement_steps=args.maximum_refinement_steps,
            row_chunk=args.row_chunk,
            progress=True,
        )
        seconds = time.perf_counter() - candidate_start
        metrics = _metrics(source, reconstruction)
        baseline_sse = _metrics(source, baseline_reconstruction)["error_sse"]
        candidates.append(
            {
                "format": f"adaptive-native-MXFP4-SQ-{mode}",
                **rate.__dict__,
                "budget_slack_nbytes": baseline_budget - rate.payload_nbytes,
                "seconds": seconds,
                **metrics,
                "sse_delta_vs_nvq2_percent": 100.0
                * (float(metrics["error_sse"]) / float(baseline_sse) - 1.0),
                **metadata,
            }
        )
        print(
            f"{mode}: SSE={metrics['error_sse']:.9f}, SNR={metrics['snr_db']:.6f} dB",
            file=sys.stderr,
            flush=True,
        )

    baseline_metrics = _metrics(source, baseline_reconstruction)
    result = {
        "schema": 1,
        "experiment": "dsv4f-adaptive-native-mxfp4-scalar-transcoding",
        "created_unix": started,
        "workspace": _git_identity(script_root),
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "baseline_device": str(torch.device(args.device)),
            "adaptive_search_device": "cpu/numpy",
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
            **baseline_metrics,
        },
        "adaptive_mxfp4_sq_candidates": candidates,
        "best_adaptive_mxfp4_sq": min(candidates, key=lambda item: float(item["error_sse"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
