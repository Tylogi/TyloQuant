#!/usr/bin/env python3
"""Search an eight-state native-output MXFP4 two-bit scalar format.

The XOR of the two-bit symbols in each native 32-weight block supplies two
in-band state bits.  One explicit bit per block supplies the high state bit.
Each row has eight states, where a state contains a five-bit fixed-catalog
palette ID and a two-bit E8M0 scale offset from one matrix-level base byte.

The frozen 32-entry catalog costs 4,288,513 bytes, or 2.044922351837158 BPW,
for a 4096 by 4096 matrix.  It remains 2,047 bytes below production NVQ2.
The original fixed16 catalog is retained as a 2.042969226837158 BPW ablation.
Decoding expands to ordinary E2M1 nibbles and E8M0 block scales before calling
the production MXFP4 decoder.
"""

from __future__ import annotations

import argparse
import hashlib
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

from bench.dsv4f_mxfp4_adaptive_sq import (
    FIXED16_PALETTE_IDS,
    NIBBLE_VALUES,
    PALETTE_NIBBLES,
    PALETTE_VALUES,
    _raw_gate_up_native,
    _unpack_source,
)
from bench.dsv4f_mxfp4_hybrid_sq import (
    _pack_block_selectors,
    _unpack_block_selectors,
    hybrid_mxfp4_sq_rate,
    quantize_hybrid_mxfp4_sq,
)
from bench.dsv4f_mxfp4_row_vq import (
    _git_identity,
    _metrics,
    _quantize_baseline,
    _sha256,
)
from bench.dsv4f_mxfp4_sq3 import _pack_fixed_width, _unpack_fixed_width
from bench.dsv4f_mxfp4_xor_sq import (
    _candidate_xor_errors,
    _pack_two_bit_symbols,
    _quantize_blocks_for_tag,
    _unpack_two_bit_symbols,
)
from mfq.formats.nvq import NVQ2_E8
from mfq.quantize.mxfp import decode_mxfp4
from mfq.quantize.v4f_source import V4FCheckpoint

# The first 16 entries are the accepted SQ2 catalog.  The five following
# entries were the only additional useful choices selected by the existing
# expert-0 full-catalog partition screen; the remaining slots are fixed filler
# so every state retains a uniform five-bit code.
SCREENED32_PALETTE_IDS = np.asarray(
    [
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
        287,
        1112,
        971,
        802,
        192,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ],
    dtype=np.int16,
)

# Frozen candidate from a 128-row expert-0 screen of the actual eight-state
# partitions.  It is evaluated as a separate design candidate before any
# holdout use; it must not be refitted outside expert 0.
FIXED32_PALETTE_IDS = np.asarray(
    [
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
        192,
        1112,
        971,
        802,
        559,
        240,
        112,
        226,
        232,
        182,
        188,
        121,
        505,
        848,
        0,
        1,
    ],
    dtype=np.int16,
)

PALETTE_CATALOGS = {
    "fixed16": FIXED16_PALETTE_IDS,
    "screened32": SCREENED32_PALETTE_IDS,
    "fixed32": FIXED32_PALETTE_IDS,
}


@dataclass(frozen=True)
class EightSq2Rate:
    bits_per_symbol: int
    native_block_size: int
    explicit_block_selector_bits: int
    implicit_symbol_tag_bits: int
    palette_id_bits: int
    scale_offset_bits: int
    states_per_row: int
    matrix_scale_base_nbytes: int
    symbol_nbytes: int
    block_selector_nbytes: int
    state_scale_nbytes: int
    state_palette_nbytes: int
    payload_nbytes: int
    payload_bpw: float


@dataclass(frozen=True)
class EightSq2RowSolution:
    error_sse: float
    state_scale_offsets: tuple[int, int, int, int, int, int, int, int]
    state_palette_ids: tuple[int, int, int, int, int, int, int, int]
    block_tags: np.ndarray
    refinement_steps: int


@dataclass(frozen=True)
class EightSq2Encoding:
    matrix_scale_base: int
    state_scale_offsets: np.ndarray
    packed_state_scales: np.ndarray
    state_palette_codes: np.ndarray
    packed_state_palettes: np.ndarray
    packed_symbols: np.ndarray
    packed_block_selectors: np.ndarray
    block_tags: np.ndarray
    packed_mxfp4: np.ndarray
    native_scale_raw: np.ndarray
    searched_sse: float


def eight_mxfp4_sq2_rate(
    rows: int,
    columns: int,
    *,
    palette_id_bits: int = 5,
) -> EightSq2Rate:
    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("eight-state MXFP4-SQ2 requires positive block-32 matrices")
    if palette_id_bits not in {4, 5}:
        raise ValueError("eight-state MXFP4-SQ2 requires four- or five-bit palette IDs")
    weights = rows * columns
    blocks = weights // 32
    symbol_nbytes = (weights * 2 + 7) // 8
    block_selector_nbytes = (blocks + 7) // 8
    state_scale_nbytes = (rows * 8 * 2 + 7) // 8
    state_palette_nbytes = (rows * 8 * palette_id_bits + 7) // 8
    payload_nbytes = (
        1 + symbol_nbytes + block_selector_nbytes + state_scale_nbytes + state_palette_nbytes
    )
    return EightSq2Rate(
        bits_per_symbol=2,
        native_block_size=32,
        explicit_block_selector_bits=1,
        implicit_symbol_tag_bits=2,
        palette_id_bits=palette_id_bits,
        scale_offset_bits=2,
        states_per_row=8,
        matrix_scale_base_nbytes=1,
        symbol_nbytes=symbol_nbytes,
        block_selector_nbytes=block_selector_nbytes,
        state_scale_nbytes=state_scale_nbytes,
        state_palette_nbytes=state_palette_nbytes,
        payload_nbytes=payload_nbytes,
        payload_bpw=8.0 * payload_nbytes / weights,
    )


def _refine_from_assignments(
    low2_errors: np.ndarray,
    initial_assignments: np.ndarray,
    *,
    maximum_steps: int,
) -> tuple[float, tuple[int, ...], np.ndarray, int]:
    assignments = np.asarray(initial_assignments, dtype=np.uint8).copy()
    states = np.zeros(8, dtype=np.int64)
    best: tuple[float, np.ndarray, np.ndarray, int] | None = None
    for step in range(1, maximum_steps + 1):
        updated_states = states.copy()
        for tag in range(8):
            selected = assignments == tag
            if bool(selected.any()):
                updated_states[tag] = int(low2_errors[tag & 3, :, selected].sum(axis=0).argmin())
        state_error = np.stack([low2_errors[tag & 3, updated_states[tag]] for tag in range(8)])
        updated_assignments = state_error.argmin(axis=0).astype(np.uint8)
        error = float(state_error.min(axis=0).sum())
        if best is None or error < best[0]:
            best = (
                error,
                updated_states.copy(),
                updated_assignments.copy(),
                step,
            )
        if np.array_equal(updated_states, states) and np.array_equal(
            updated_assignments,
            assignments,
        ):
            break
        states = updated_states
        assignments = updated_assignments
    if best is None:
        raise RuntimeError("eight-state SQ2 refinement produced no candidate")
    return best[0], tuple(int(item) for item in best[1]), best[2], best[3]


def _greedy_assignments(
    low2_errors: np.ndarray,
    tag_order: tuple[int, ...],
) -> np.ndarray:
    covered = np.full(low2_errors.shape[2], np.inf, dtype=np.float64)
    states = np.zeros(8, dtype=np.int64)
    chosen_by_low_tag: list[list[int]] = [[], [], [], []]
    for tag in tag_order:
        totals = np.minimum(low2_errors[tag & 3], covered[None, :]).sum(axis=1)
        if chosen_by_low_tag[tag & 3]:
            totals[np.asarray(chosen_by_low_tag[tag & 3])] = np.inf
        states[tag] = int(totals.argmin())
        chosen_by_low_tag[tag & 3].append(states[tag])
        covered = np.minimum(covered, low2_errors[tag & 3, states[tag]])
    state_error = np.stack([low2_errors[tag & 3, states[tag]] for tag in range(8)])
    return state_error.argmin(axis=0).astype(np.uint8)


def _tag_orders() -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    for low_order in itertools.permutations(range(4)):
        result.append(tuple(low_order) + tuple(item + 4 for item in low_order))
        result.append(tuple(tag for item in low_order for tag in (item, item + 4)))
    return tuple(result)


EIGHT_STATE_TAG_ORDERS = _tag_orders()


def _solve_eight_state_row(
    source_nibbles_row: np.ndarray,
    source_scale_row: np.ndarray,
    *,
    matrix_scale_base: int,
    maximum_steps: int,
    palette_ids: np.ndarray,
) -> EightSq2RowSolution:
    scale_values = np.arange(
        matrix_scale_base,
        matrix_scale_base + 4,
        dtype=np.uint8,
    )
    state_scales = np.repeat(scale_values, len(palette_ids))
    state_palette_ids = np.tile(palette_ids, len(scale_values))
    errors = _candidate_xor_errors(
        source_nibbles_row,
        source_scale_row,
        state_scales,
        state_palette_ids,
    )
    target = NIBBLE_VALUES[source_nibbles_row] * np.exp2(
        source_scale_row[:, None].astype(np.int16) - 127
    )
    starts = [_greedy_assignments(errors, order) for order in EIGHT_STATE_TAG_ORDERS]
    for values in (
        source_scale_row,
        target.mean(axis=1),
        np.square(target).sum(axis=1),
        np.abs(target).max(axis=1),
        np.count_nonzero(target, axis=1),
    ):
        clusters = np.empty(target.shape[0], dtype=np.uint8)
        order = np.argsort(values, kind="stable")
        clusters[order] = np.minimum(
            7,
            np.arange(target.shape[0]) * 8 // target.shape[0],
        )
        starts.extend(clusters ^ np.uint8(mask) for mask in range(8))
    candidates = [
        _refine_from_assignments(
            errors,
            start,
            maximum_steps=maximum_steps,
        )
        for start in starts
    ]
    error, state_indices, block_tags, refinement_steps = min(
        candidates,
        key=lambda item: item[0],
    )
    return EightSq2RowSolution(
        error_sse=error,
        state_scale_offsets=tuple(
            int(state_scales[index]) - matrix_scale_base for index in state_indices
        ),
        state_palette_ids=tuple(int(state_palette_ids[index]) for index in state_indices),
        block_tags=block_tags,
        refinement_steps=refinement_steps,
    )


def decode_eight_mxfp4_sq2(
    packed_symbols: np.ndarray,
    packed_block_selectors: np.ndarray,
    matrix_scale_base: int,
    packed_state_scales: np.ndarray,
    packed_state_palettes: np.ndarray,
    *,
    palette_ids: np.ndarray = FIXED32_PALETTE_IDS,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    symbols = _unpack_two_bit_symbols(packed_symbols)
    rows, columns = symbols.shape
    if columns % 32 or not 0 <= matrix_scale_base <= 251:
        raise ValueError("eight-state MXFP4-SQ2 payload geometry mismatch")
    blocks = columns // 32
    block_symbols = symbols.reshape(rows, blocks, 32)
    explicit_high_bit = _unpack_block_selectors(
        packed_block_selectors,
        rows=rows,
        blocks=blocks,
    )
    implicit_low_bits = np.bitwise_xor.reduce(block_symbols, axis=2)
    block_tags = (explicit_high_bit << 2) | implicit_low_bits
    state_scale_offsets = _unpack_fixed_width(
        packed_state_scales,
        2,
        count=rows * 8,
    ).reshape(rows, 8)
    state_scales = matrix_scale_base + state_scale_offsets
    catalog = np.asarray(palette_ids, dtype=np.int16)
    palette_id_bits = int(math.log2(len(catalog)))
    if len(catalog) not in {16, 32} or len(np.unique(catalog)) != len(catalog):
        raise ValueError("eight-state SQ2 palette catalog must contain 16 or 32 unique entries")
    palette_codes = _unpack_fixed_width(
        packed_state_palettes,
        palette_id_bits,
        count=rows * 8,
    ).reshape(rows, 8)
    state_palette_ids = catalog[palette_codes]
    local_rows = np.arange(rows)[:, None]
    selected_scale = state_scales[local_rows, block_tags].astype(np.uint8)
    selected_palette = state_palette_ids[local_rows, block_tags]
    output_nibbles = np.take_along_axis(
        PALETTE_NIBBLES[selected_palette][:, :, None, :],
        block_symbols[:, :, :, None],
        axis=3,
    )[:, :, :, 0]
    packed_blocks = output_nibbles[:, :, 0::2] | (output_nibbles[:, :, 1::2] << 4)
    packed_mxfp4 = packed_blocks.reshape(rows, columns // 2)
    reconstruction = decode_mxfp4(
        packed_mxfp4,
        selected_scale,
        device=device,
    ).contiguous()
    return reconstruction, packed_mxfp4, selected_scale, block_tags


def _materialize_encoding(
    source_nibbles: np.ndarray,
    source_scale_raw: np.ndarray,
    solutions: list[EightSq2RowSolution],
    *,
    matrix_scale_base: int,
    palette_ids: np.ndarray,
) -> EightSq2Encoding:
    rows, blocks, block_size = source_nibbles.shape
    state_scale_offsets = np.asarray(
        [solution.state_scale_offsets for solution in solutions],
        dtype=np.uint8,
    )
    state_palette_ids = np.asarray(
        [solution.state_palette_ids for solution in solutions],
        dtype=np.int16,
    )
    palette_to_local = {int(palette_id): index for index, palette_id in enumerate(palette_ids)}
    state_palette_codes = np.vectorize(
        palette_to_local.__getitem__,
        otypes=[np.uint8],
    )(state_palette_ids)
    block_tags = np.stack([solution.block_tags for solution in solutions]).astype(
        np.uint8,
        copy=False,
    )
    symbols = np.empty((rows, blocks, block_size), dtype=np.uint8)
    output_nibbles = np.empty_like(symbols)
    native_scale_raw = np.empty((rows, blocks), dtype=np.uint8)
    measured_search_sse = 0.0
    for row in range(rows):
        target = NIBBLE_VALUES[source_nibbles[row]] * np.exp2(
            source_scale_raw[row, :, None].astype(np.int16) - 127
        )
        for tag in range(8):
            selected = block_tags[row] == tag
            if not bool(selected.any()):
                continue
            palette_id = int(state_palette_ids[row, tag])
            scale_raw = matrix_scale_base + int(state_scale_offsets[row, tag])
            levels = PALETTE_VALUES[palette_id] * math.ldexp(
                1.0,
                scale_raw - 127,
            )
            local_symbols, local_error = _quantize_blocks_for_tag(
                target[selected],
                levels,
                tag & 3,
            )
            symbols[row, selected] = local_symbols
            output_nibbles[row, selected] = PALETTE_NIBBLES[palette_id][local_symbols]
            native_scale_raw[row, selected] = scale_raw
            measured_search_sse += float(local_error.sum())
    packed_symbols = _pack_two_bit_symbols(symbols.reshape(rows, blocks * block_size))
    packed_block_selectors = _pack_block_selectors(block_tags >> 2)
    packed_state_scales = _pack_fixed_width(state_scale_offsets, 2)
    palette_id_bits = int(math.log2(len(palette_ids)))
    packed_state_palettes = _pack_fixed_width(state_palette_codes, palette_id_bits)
    packed_blocks = output_nibbles[:, :, 0::2] | (output_nibbles[:, :, 1::2] << 4)
    packed_mxfp4 = packed_blocks.reshape(rows, blocks * 16)
    searched_sse = float(sum(solution.error_sse for solution in solutions))
    if not math.isclose(
        searched_sse,
        measured_search_sse,
        rel_tol=2e-9,
        abs_tol=1e-10,
    ):
        raise RuntimeError(
            "eight-state SQ2 search/materialization SSE mismatch: "
            f"{searched_sse} != {measured_search_sse}"
        )
    encoding = EightSq2Encoding(
        matrix_scale_base=matrix_scale_base,
        state_scale_offsets=state_scale_offsets,
        packed_state_scales=packed_state_scales,
        state_palette_codes=state_palette_codes,
        packed_state_palettes=packed_state_palettes,
        packed_symbols=packed_symbols,
        packed_block_selectors=packed_block_selectors,
        block_tags=block_tags,
        packed_mxfp4=packed_mxfp4,
        native_scale_raw=native_scale_raw,
        searched_sse=searched_sse,
    )
    _, decoded_mxfp4, decoded_scale, decoded_tags = decode_eight_mxfp4_sq2(
        encoding.packed_symbols,
        encoding.packed_block_selectors,
        encoding.matrix_scale_base,
        encoding.packed_state_scales,
        encoding.packed_state_palettes,
        palette_ids=palette_ids,
        device="cpu",
    )
    if not np.array_equal(decoded_mxfp4, packed_mxfp4):
        raise RuntimeError("eight-state SQ2 physical payload changed nibbles")
    if not np.array_equal(decoded_scale, native_scale_raw):
        raise RuntimeError("eight-state SQ2 physical payload changed scales")
    if not np.array_equal(decoded_tags, block_tags):
        raise RuntimeError("eight-state SQ2 physical payload changed block tags")
    return encoding


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _metadata(
    encoding: EightSq2Encoding,
    source_scale_raw: np.ndarray,
    solutions: list[EightSq2RowSolution],
    *,
    palette_ids: np.ndarray,
) -> dict[str, Any]:
    state_palette_ids = palette_ids[encoding.state_palette_codes]
    palette_counts = Counter(int(item) for item in state_palette_ids.flat)
    tag_counts = Counter(int(item) for item in encoding.block_tags.flat)
    return {
        "state_layout": "eight-row-states:(2-bit-relative-E8M0-scale,fixed-catalog-four-level-E2M1-palette)",
        "state_tag": "explicit-high-bit-plus-two-bit-symbol-xor-tag",
        "explicit_block_selector_bits": 1,
        "implicit_symbol_tag_bits": 2,
        "scale_dtype": "matrix-E8M0-base-plus-2-bit-state-offset",
        "matrix_scale_base": encoding.matrix_scale_base,
        "optimizer": "deterministic-multistart-hard-em-with-exact-one-or-two-symbol-tag-repair",
        "optimizer_is_global_within_catalog": False,
        "maximum_refinement_steps_used": max(
            (solution.refinement_steps for solution in solutions), default=0
        ),
        "state_scale_raw_min": int(encoding.matrix_scale_base + encoding.state_scale_offsets.min()),
        "state_scale_raw_max": int(encoding.matrix_scale_base + encoding.state_scale_offsets.max()),
        "effective_scale_matches_source_fraction": float(
            np.mean(encoding.native_scale_raw == source_scale_raw)
        ),
        "block_tag_fractions": {
            str(tag): count / encoding.block_tags.size for tag, count in sorted(tag_counts.items())
        },
        "palette_catalog_size": len(palette_ids),
        "palette_id_bits": int(math.log2(len(palette_ids))),
        "palette_catalog_ids": palette_ids.tolist(),
        "palette_catalog_values": PALETTE_VALUES[palette_ids].tolist(),
        "palette_catalog_sha256": _sha256_array(palette_ids),
        "unique_palette_ids": len(palette_counts),
        "top_state_palette_ids": [
            {
                "palette_id": palette_id,
                "count": count,
                "values": PALETTE_VALUES[palette_id].tolist(),
                "nibbles": PALETTE_NIBBLES[palette_id].tolist(),
            }
            for palette_id, count in palette_counts.most_common()
        ],
        "packed_symbols_sha256": _sha256_array(encoding.packed_symbols),
        "packed_block_selectors_sha256": _sha256_array(encoding.packed_block_selectors),
        "packed_state_scales_sha256": _sha256_array(encoding.packed_state_scales),
        "packed_state_palettes_sha256": _sha256_array(encoding.packed_state_palettes),
        "packed_native_mxfp4_sha256": _sha256_array(encoding.packed_mxfp4),
        "native_scale_sha256": _sha256_array(encoding.native_scale_raw),
        "learned_vector_codebook": False,
        "stored_fp16_scale_or_centroid": False,
        "physical_storage_roundtrip_verified": True,
        "final_values_are_native_block32_mxfp4": True,
    }


@torch.inference_mode()
def quantize_eight_mxfp4_sq2(
    packed: np.ndarray,
    source_scale_raw: np.ndarray,
    *,
    matrix_scale_base: int | None = None,
    palette_ids: np.ndarray = FIXED32_PALETTE_IDS,
    maximum_refinement_steps: int = 10,
    progress: bool = False,
) -> tuple[torch.Tensor, EightSq2Encoding, dict[str, Any]]:
    if maximum_refinement_steps <= 0:
        raise ValueError("maximum_refinement_steps must be positive")
    source_nibbles, _ = _unpack_source(packed, source_scale_raw)
    source_scales = np.asarray(source_scale_raw, dtype=np.uint8)
    catalog = np.asarray(palette_ids, dtype=np.int16)
    if len(catalog) not in {16, 32} or len(np.unique(catalog)) != len(catalog):
        raise ValueError("eight-state SQ2 palette catalog must contain 16 or 32 unique entries")
    if int(catalog.min()) < 0 or int(catalog.max()) >= len(PALETTE_VALUES):
        raise ValueError("eight-state SQ2 palette catalog ID is out of range")
    scale_base = int(source_scales.min()) if matrix_scale_base is None else matrix_scale_base
    if not 0 <= scale_base <= 251:
        raise ValueError("matrix_scale_base must leave room for four E8M0 values")
    if int(source_scales.max()) > scale_base + 3:
        raise ValueError("source E8M0 range exceeds the four-value matrix scale window")
    solutions: list[EightSq2RowSolution] = []
    for row in range(source_nibbles.shape[0]):
        solutions.append(
            _solve_eight_state_row(
                source_nibbles[row],
                source_scales[row],
                matrix_scale_base=scale_base,
                maximum_steps=maximum_refinement_steps,
                palette_ids=catalog,
            )
        )
        if progress and ((row + 1) % 128 == 0 or row + 1 == source_nibbles.shape[0]):
            print(
                f"SQ2 eight: solved {row + 1}/{source_nibbles.shape[0]} rows",
                file=sys.stderr,
                flush=True,
            )
    encoding = _materialize_encoding(
        source_nibbles,
        source_scales,
        solutions,
        matrix_scale_base=scale_base,
        palette_ids=catalog,
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
            f"eight-state SQ2 physical decode SSE mismatch: {encoding.searched_sse} != {measured_sse}"
        )
    return (
        reconstruction,
        encoding,
        _metadata(
            encoding,
            source_scales,
            solutions,
            palette_ids=catalog,
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--matrix-scale-base", type=int)
    parser.add_argument(
        "--palette-catalog",
        choices=tuple(PALETTE_CATALOGS),
        default="fixed32",
    )
    parser.add_argument("--maximum-refinement-steps", type=int, default=10)
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
    packed, source_scale_raw = _raw_gate_up_native(
        checkpoint,
        args.layer,
        args.expert,
    )
    source = decode_mxfp4(packed, source_scale_raw, device="cpu").contiguous()
    rows, columns = (int(item) for item in source.shape)
    nvq2_budget = NVQ2_E8.payload_nbytes(rows, columns)

    baseline_started = time.perf_counter()
    nvq2_reconstruction, nvq2_bytes = _quantize_baseline(
        source,
        NVQ2_E8.label,
        device=args.device,
    )
    nvq2_seconds = time.perf_counter() - baseline_started
    if nvq2_bytes != nvq2_budget:
        raise RuntimeError("NVQ2 baseline payload accounting changed")
    nvq2_metrics = _metrics(source, nvq2_reconstruction)
    del nvq2_reconstruction
    if str(torch.device(args.device)) == "mps":
        torch.mps.empty_cache()

    print("searching current four-state SQ2 control...", file=sys.stderr, flush=True)
    baseline_started = time.perf_counter()
    old_reconstruction, _, old_metadata = quantize_hybrid_mxfp4_sq(
        packed,
        source_scale_raw,
        exponent_radius=0,
        maximum_refinement_steps=args.maximum_refinement_steps,
        progress=True,
    )
    old_seconds = time.perf_counter() - baseline_started
    old_rate = hybrid_mxfp4_sq_rate(rows, columns)
    old_metrics = _metrics(source, old_reconstruction)
    del old_reconstruction

    palette_ids = PALETTE_CATALOGS[args.palette_catalog]
    rate = eight_mxfp4_sq2_rate(
        rows,
        columns,
        palette_id_bits=int(math.log2(len(palette_ids))),
    )
    if rate.payload_nbytes > nvq2_budget:
        raise RuntimeError("eight-state MXFP4-SQ2 exceeds NVQ2 payload")
    print(
        f"searching SQ2 eight-state: {rate.payload_nbytes} bytes / {rate.payload_bpw:.9f} BPW...",
        file=sys.stderr,
        flush=True,
    )
    search_started = time.perf_counter()
    reconstruction, _, metadata = quantize_eight_mxfp4_sq2(
        packed,
        source_scale_raw,
        matrix_scale_base=args.matrix_scale_base,
        palette_ids=palette_ids,
        maximum_refinement_steps=args.maximum_refinement_steps,
        progress=True,
    )
    search_seconds = time.perf_counter() - search_started
    metrics = _metrics(source, reconstruction)
    candidate = {
        "format": f"eight-state-native-MXFP4-SQ2-{args.palette_catalog}",
        **rate.__dict__,
        "budget_slack_vs_nvq2_nbytes": nvq2_budget - rate.payload_nbytes,
        "seconds": search_seconds,
        **metrics,
        "sse_delta_vs_nvq2_percent": 100.0
        * (float(metrics["error_sse"]) / float(nvq2_metrics["error_sse"]) - 1.0),
        "sse_delta_vs_four_state_sq2_percent": 100.0
        * (float(metrics["error_sse"]) / float(old_metrics["error_sse"]) - 1.0),
        **metadata,
    }
    print(
        f"SQ2 eight: SSE={metrics['error_sse']:.9f}, SNR={metrics['snr_db']:.6f} dB",
        file=sys.stderr,
        flush=True,
    )
    result: dict[str, Any] = {
        "schema": 1,
        "experiment": "dsv4f-eight-state-native-mxfp4-two-bit-scalar-transcoding",
        "created_unix": started,
        "workspace": _git_identity(script_root),
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "nvq2_device": str(torch.device(args.device)),
            "sq2_search_device": "cpu/numpy",
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
        "baselines": [
            {
                "format": NVQ2_E8.label,
                "payload_nbytes": nvq2_bytes,
                "payload_bpw": 8.0 * nvq2_bytes / (rows * columns),
                "seconds": nvq2_seconds,
                **nvq2_metrics,
            },
            {
                "format": "hybrid-tagged-four-state-native-MXFP4-SQ-fixed16",
                **old_rate.__dict__,
                "budget_slack_vs_nvq2_nbytes": nvq2_budget - old_rate.payload_nbytes,
                "seconds": old_seconds,
                **old_metrics,
                **old_metadata,
            },
        ],
        "eight_state_mxfp4_sq2": candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
