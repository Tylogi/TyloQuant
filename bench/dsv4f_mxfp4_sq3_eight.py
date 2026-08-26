#!/usr/bin/env python3
"""Search an eight-state native-output MXFP4 three-bit scalar format.

The low two bits of the XOR of 32 scalar symbols form an in-band state tag.
One explicit bit per native block supplies the high tag bit.  Each row has
eight states, where a state contains a five-bit fixed-catalog palette ID and
a two-bit E8M0 scale offset from one matrix-level base byte.

For a 4096 by 4096 matrix this payload is 6,385,665 bytes, or
3.044922351837158 BPW.  It remains 2,047 bytes below NVQ3-D4-256.  Decoding
expands to ordinary E2M1 nibbles and E8M0 block scales before calling the
production MXFP4 decoder.
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

from bench.dsv4f_mxfp4_adaptive_sq import (
    NIBBLE_VALUES,
    _raw_gate_up_native,
    _unpack_source,
)
from bench.dsv4f_mxfp4_row_vq import (
    _git_identity,
    _metrics,
    _quantize_baseline,
    _sha256,
)
from bench.dsv4f_mxfp4_sq3 import (
    LLAMA_CPP_FORMATS,
    SQ3_PALETTE_NIBBLES,
    SQ3_PALETTE_VALUES,
    _llama_cpp_quantize,
    _pack_block_selectors,
    _pack_fixed_width,
    _pack_three_bit_symbols,
    _unpack_block_selectors,
    _unpack_fixed_width,
    _unpack_three_bit_symbols,
)
from bench.dsv4f_mxfp4_xor_sq import (
    _best_distinct_pair_indices,
    _distinct_pair_cost,
)
from mfq.formats.nvq import NVQ3_D4, NVQ3_D4_512
from mfq.quantize.mxfp import decode_mxfp4
from mfq.quantize.v4f_source import V4FCheckpoint

# Frozen from the expert-0 eight-state design screen.  This changes only the
# contents of the format-level lookup table; every state still stores one
# uniform five-bit palette code.
EIGHT_FIXED32_SQ3_PALETTE_IDS = np.asarray(
    (
        499,
        2407,
        523,
        5738,
        1196,
        2001,
        5497,
        4705,
        1087,
        2400,
        2402,
        1077,
        1078,
        4116,
        5503,
        2002,
        3726,
        2010,
        5498,
        4117,
        5552,
        4706,
        500,
        1483,
        3717,
        3727,
        1478,
        4818,
        5811,
        2003,
        2989,
        2035,
    ),
    dtype=np.int16,
)


@dataclass(frozen=True)
class EightSq3Rate:
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
class EightSq3RowSolution:
    error_sse: float
    state_scale_offsets: tuple[int, int, int, int, int, int, int, int]
    state_palette_ids: tuple[int, int, int, int, int, int, int, int]
    block_tags: np.ndarray
    refinement_steps: int


@dataclass(frozen=True)
class EightSq3Encoding:
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


def eight_mxfp4_sq3_rate(rows: int, columns: int) -> EightSq3Rate:
    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("eight-state MXFP4-SQ3 requires positive block-32 matrices")
    weights = rows * columns
    blocks = weights // 32
    symbol_nbytes = (weights * 3 + 7) // 8
    block_selector_nbytes = (blocks + 7) // 8
    state_scale_nbytes = (rows * 8 * 2 + 7) // 8
    state_palette_nbytes = (rows * 8 * 5 + 7) // 8
    payload_nbytes = (
        1 + symbol_nbytes + block_selector_nbytes + state_scale_nbytes + state_palette_nbytes
    )
    return EightSq3Rate(
        bits_per_symbol=3,
        native_block_size=32,
        explicit_block_selector_bits=1,
        implicit_symbol_tag_bits=2,
        palette_id_bits=5,
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


def _candidate_low2_errors(
    source_nibbles_row: np.ndarray,
    source_scale_row: np.ndarray,
    state_scales: np.ndarray,
    state_palette_ids: np.ndarray,
) -> np.ndarray:
    """Return one-or-two-symbol constrained errors as (low tag, state, block)."""

    target = NIBBLE_VALUES[source_nibbles_row] * np.exp2(
        source_scale_row[:, None].astype(np.int16) - 127
    )
    levels = SQ3_PALETTE_VALUES[state_palette_ids] * np.exp2(
        state_scales[:, None].astype(np.int16) - 127
    )
    costs = np.square(target[None, :, :, None] - levels[:, None, None, :])
    base_symbols = costs.argmin(axis=3).astype(np.uint8)
    base_costs = np.take_along_axis(
        costs,
        base_symbols[:, :, :, None],
        axis=3,
    )[:, :, :, 0]
    base_error = base_costs.sum(axis=2)
    base_tag = np.bitwise_xor.reduce(base_symbols & 3, axis=2)
    symbol_groups = np.arange(8, dtype=np.uint8) & 3

    penalties = [np.zeros_like(base_costs)]
    for delta in range(1, 4):
        desired_group = (base_symbols & 3) ^ np.uint8(delta)
        alternative_cost = np.empty_like(base_costs)
        for group in range(4):
            selected = desired_group == group
            group_cost = costs[:, :, :, symbol_groups == group].min(axis=3)
            alternative_cost[selected] = group_cost[selected]
        penalties.append(alternative_cost - base_costs)

    pair_changes = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    adjustment = np.zeros((4, len(state_scales), target.shape[0]), dtype=np.float64)
    for delta in range(1, 4):
        left, right = pair_changes[delta]
        adjustment[delta] = np.minimum(
            penalties[delta].min(axis=2),
            _distinct_pair_cost(penalties[left], penalties[right]),
        )
    moved_adjustment = np.moveaxis(adjustment, 0, 2)
    result = np.empty_like(adjustment)
    for required_tag in range(4):
        delta = base_tag ^ np.uint8(required_tag)
        result[required_tag] = (
            base_error
            + np.take_along_axis(
                moved_adjustment,
                delta[:, :, None],
                axis=2,
            )[:, :, 0]
        )
    if not np.allclose(result.min(axis=0), base_error, rtol=0.0, atol=0.0):
        raise RuntimeError("eight-state SQ3 constraints lost the unconstrained solution")
    return result


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
        raise RuntimeError("eight-state SQ3 refinement produced no candidate")
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
) -> EightSq3RowSolution:
    scale_values = np.arange(
        matrix_scale_base,
        matrix_scale_base + 4,
        dtype=np.uint8,
    )
    state_scales = np.repeat(scale_values, len(EIGHT_FIXED32_SQ3_PALETTE_IDS))
    state_palette_ids = np.tile(EIGHT_FIXED32_SQ3_PALETTE_IDS, len(scale_values))
    errors = _candidate_low2_errors(
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
    return EightSq3RowSolution(
        error_sse=error,
        state_scale_offsets=tuple(
            int(state_scales[index]) - matrix_scale_base for index in state_indices
        ),
        state_palette_ids=tuple(int(state_palette_ids[index]) for index in state_indices),
        block_tags=block_tags,
        refinement_steps=refinement_steps,
    )


def _quantize_blocks_for_low2_tag(
    target: np.ndarray,
    levels: np.ndarray,
    required_tag: int,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.square(target[:, :, None] - levels[None, None, :])
    symbols = costs.argmin(axis=2).astype(np.uint8)
    rows = np.arange(target.shape[0])
    positions = np.arange(target.shape[1])
    base_cost = costs[rows[:, None], positions[None, :], symbols]
    base_tag = np.bitwise_xor.reduce(symbols & 3, axis=1)
    symbol_groups = np.arange(8, dtype=np.uint8) & 3

    penalties = [np.zeros_like(base_cost)]
    alternative_symbols = [symbols.copy()]
    for delta in range(1, 4):
        desired_group = (symbols & 3) ^ np.uint8(delta)
        alternative_cost = np.empty_like(base_cost)
        alternative_symbol = np.empty_like(symbols)
        for group in range(4):
            selected = desired_group == group
            candidates = np.flatnonzero(symbol_groups == group)
            group_cost = costs[:, :, candidates]
            local_index = group_cost.argmin(axis=2)
            local_min = np.take_along_axis(
                group_cost,
                local_index[:, :, None],
                axis=2,
            )[:, :, 0]
            local_symbol = candidates[local_index]
            alternative_cost[selected] = local_min[selected]
            alternative_symbol[selected] = local_symbol[selected]
        penalties.append(alternative_cost - base_cost)
        alternative_symbols.append(alternative_symbol)

    pair_changes = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    required_delta = base_tag ^ np.uint8(required_tag)
    for delta in range(1, 4):
        selected_blocks = np.flatnonzero(required_delta == delta)
        if not selected_blocks.size:
            continue
        single_position = penalties[delta][selected_blocks].argmin(axis=1)
        single_cost = penalties[delta][selected_blocks, single_position]
        left_delta, right_delta = pair_changes[delta]
        pair_cost, left_position, right_position = _best_distinct_pair_indices(
            penalties[left_delta][selected_blocks],
            penalties[right_delta][selected_blocks],
        )
        use_pair = pair_cost < single_cost
        single_blocks = selected_blocks[~use_pair]
        symbols[single_blocks, single_position[~use_pair]] = alternative_symbols[delta][
            single_blocks, single_position[~use_pair]
        ]
        pair_blocks = selected_blocks[use_pair]
        symbols[pair_blocks, left_position[use_pair]] = alternative_symbols[left_delta][
            pair_blocks, left_position[use_pair]
        ]
        symbols[pair_blocks, right_position[use_pair]] = alternative_symbols[right_delta][
            pair_blocks, right_position[use_pair]
        ]
    if not np.all(np.bitwise_xor.reduce(symbols & 3, axis=1) == required_tag):
        raise RuntimeError("failed to embed eight-state SQ3 low tag")
    error = costs[rows[:, None], positions[None, :], symbols].sum(axis=1)
    return symbols, error


def decode_eight_mxfp4_sq3(
    packed_symbols: np.ndarray,
    packed_block_selectors: np.ndarray,
    matrix_scale_base: int,
    packed_state_scales: np.ndarray,
    packed_state_palettes: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    symbols = _unpack_three_bit_symbols(packed_symbols)
    rows, columns = symbols.shape
    if columns % 32 or not 0 <= matrix_scale_base <= 251:
        raise ValueError("eight-state MXFP4-SQ3 payload geometry mismatch")
    blocks = columns // 32
    block_symbols = symbols.reshape(rows, blocks, 32)
    explicit_high_bit = _unpack_block_selectors(
        packed_block_selectors,
        rows=rows,
        blocks=blocks,
    )
    implicit_low_bits = np.bitwise_xor.reduce(block_symbols & 3, axis=2)
    block_tags = (explicit_high_bit << 2) | implicit_low_bits
    state_scale_offsets = _unpack_fixed_width(
        packed_state_scales,
        2,
        count=rows * 8,
    ).reshape(rows, 8)
    state_scales = matrix_scale_base + state_scale_offsets
    palette_codes = _unpack_fixed_width(
        packed_state_palettes,
        5,
        count=rows * 8,
    ).reshape(rows, 8)
    state_palette_ids = EIGHT_FIXED32_SQ3_PALETTE_IDS[palette_codes]
    local_rows = np.arange(rows)[:, None]
    selected_scale = state_scales[local_rows, block_tags].astype(np.uint8)
    selected_palette = state_palette_ids[local_rows, block_tags]
    output_nibbles = np.take_along_axis(
        SQ3_PALETTE_NIBBLES[selected_palette][:, :, None, :],
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
    solutions: list[EightSq3RowSolution],
    *,
    matrix_scale_base: int,
) -> EightSq3Encoding:
    rows, blocks, block_size = source_nibbles.shape
    state_scale_offsets = np.asarray(
        [solution.state_scale_offsets for solution in solutions],
        dtype=np.uint8,
    )
    state_palette_ids = np.asarray(
        [solution.state_palette_ids for solution in solutions],
        dtype=np.int16,
    )
    palette_to_local = {
        int(palette_id): index for index, palette_id in enumerate(EIGHT_FIXED32_SQ3_PALETTE_IDS)
    }
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
            levels = SQ3_PALETTE_VALUES[palette_id] * math.ldexp(
                1.0,
                scale_raw - 127,
            )
            local_symbols, local_error = _quantize_blocks_for_low2_tag(
                target[selected],
                levels,
                tag & 3,
            )
            symbols[row, selected] = local_symbols
            output_nibbles[row, selected] = SQ3_PALETTE_NIBBLES[palette_id][local_symbols]
            native_scale_raw[row, selected] = scale_raw
            measured_search_sse += float(local_error.sum())
    packed_symbols = _pack_three_bit_symbols(symbols.reshape(rows, blocks * block_size))
    packed_block_selectors = _pack_block_selectors(block_tags >> 2)
    packed_state_scales = _pack_fixed_width(state_scale_offsets, 2)
    packed_state_palettes = _pack_fixed_width(state_palette_codes, 5)
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
            f"eight-state SQ3 search/materialization SSE mismatch: {searched_sse} != {measured_search_sse}"
        )
    encoding = EightSq3Encoding(
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
    _, decoded_mxfp4, decoded_scale, decoded_tags = decode_eight_mxfp4_sq3(
        encoding.packed_symbols,
        encoding.packed_block_selectors,
        encoding.matrix_scale_base,
        encoding.packed_state_scales,
        encoding.packed_state_palettes,
        device="cpu",
    )
    if not np.array_equal(decoded_mxfp4, packed_mxfp4):
        raise RuntimeError("eight-state SQ3 physical payload changed nibbles")
    if not np.array_equal(decoded_scale, native_scale_raw):
        raise RuntimeError("eight-state SQ3 physical payload changed scales")
    if not np.array_equal(decoded_tags, block_tags):
        raise RuntimeError("eight-state SQ3 physical payload changed block tags")
    return encoding


def _metadata(
    encoding: EightSq3Encoding,
    source_scale_raw: np.ndarray,
    solutions: list[EightSq3RowSolution],
) -> dict[str, Any]:
    palette_ids = EIGHT_FIXED32_SQ3_PALETTE_IDS[encoding.state_palette_codes]
    palette_counts = Counter(int(item) for item in palette_ids.flat)
    tag_counts = Counter(int(item) for item in encoding.block_tags.flat)
    return {
        "state_layout": "eight-row-states:(2-bit-relative-E8M0-scale,fixed32-eight-level-E2M1-palette)",
        "state_tag": "explicit-high-bit-plus-symbol-low2-xor-tag",
        "explicit_block_selector_bits": 1,
        "implicit_symbol_tag_bits": 2,
        "scale_dtype": "matrix-E8M0-base-plus-2-bit-state-offset",
        "matrix_scale_base": encoding.matrix_scale_base,
        "optimizer": "deterministic-multistart-hard-em-with-one-or-two-symbol-tag-repair",
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
            str(tag): tag_counts[tag] / encoding.block_tags.size for tag in range(8)
        },
        "fixed32_full_catalog_ids": EIGHT_FIXED32_SQ3_PALETTE_IDS.tolist(),
        "fixed32_palette_values": SQ3_PALETTE_VALUES[EIGHT_FIXED32_SQ3_PALETTE_IDS].tolist(),
        "unique_palette_ids": len(palette_counts),
        "top_state_palette_ids": [
            {
                "palette_id": palette_id,
                "count": count,
                "values": SQ3_PALETTE_VALUES[palette_id].tolist(),
                "nibbles": SQ3_PALETTE_NIBBLES[palette_id].tolist(),
            }
            for palette_id, count in palette_counts.most_common()
        ],
        "learned_vector_codebook": False,
        "stored_fp16_scale_or_centroid": False,
        "physical_storage_roundtrip_verified": True,
        "final_values_are_native_block32_mxfp4": True,
    }


@torch.inference_mode()
def quantize_eight_mxfp4_sq3(
    packed: np.ndarray,
    source_scale_raw: np.ndarray,
    *,
    matrix_scale_base: int | None = None,
    maximum_refinement_steps: int = 10,
    progress: bool = False,
) -> tuple[torch.Tensor, EightSq3Encoding, dict[str, Any]]:
    if maximum_refinement_steps <= 0:
        raise ValueError("maximum_refinement_steps must be positive")
    source_nibbles, _ = _unpack_source(packed, source_scale_raw)
    source_scales = np.asarray(source_scale_raw, dtype=np.uint8)
    scale_base = int(source_scales.min()) if matrix_scale_base is None else matrix_scale_base
    if not 0 <= scale_base <= 251:
        raise ValueError("matrix_scale_base must leave room for four E8M0 values")
    solutions: list[EightSq3RowSolution] = []
    for row in range(source_nibbles.shape[0]):
        solutions.append(
            _solve_eight_state_row(
                source_nibbles[row],
                source_scales[row],
                matrix_scale_base=scale_base,
                maximum_steps=maximum_refinement_steps,
            )
        )
        if progress and ((row + 1) % 128 == 0 or row + 1 == source_nibbles.shape[0]):
            print(
                f"SQ3 eight: solved {row + 1}/{source_nibbles.shape[0]} rows",
                file=sys.stderr,
                flush=True,
            )
    encoding = _materialize_encoding(
        source_nibbles,
        source_scales,
        solutions,
        matrix_scale_base=scale_base,
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
            f"eight-state SQ3 physical decode SSE mismatch: {encoding.searched_sse} != {measured_sse}"
        )
    return reconstruction, encoding, _metadata(encoding, source_scales, solutions)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--matrix-scale-base", type=int)
    parser.add_argument("--maximum-refinement-steps", type=int, default=10)
    parser.add_argument("--llama-lib", type=Path)
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

    baselines: list[dict[str, Any]] = []
    for spec in (NVQ3_D4, NVQ3_D4_512):
        baseline_started = time.perf_counter()
        reconstruction, payload_nbytes = _quantize_baseline(
            source,
            spec.label,
            device=args.device,
        )
        seconds = time.perf_counter() - baseline_started
        baselines.append(
            {
                "format": spec.label,
                "implementation": "MFQ",
                "payload_nbytes": payload_nbytes,
                "payload_bpw": 8.0 * payload_nbytes / source.numel(),
                "quantize_and_dequantize_seconds": seconds,
                **_metrics(source, reconstruction),
            }
        )
        del reconstruction
        if str(torch.device(args.device)) == "mps":
            torch.mps.empty_cache()

    if args.llama_lib is not None:
        llama_library = args.llama_lib.resolve()
        for spec in LLAMA_CPP_FORMATS:
            reconstruction, payload_nbytes, quantize_seconds, dequantize_seconds = (
                _llama_cpp_quantize(source, llama_library, spec)
            )
            baselines.append(
                {
                    "format": spec.label,
                    "implementation": "llama.cpp",
                    "library": str(llama_library),
                    "payload_nbytes": payload_nbytes,
                    "payload_bpw": 8.0 * payload_nbytes / source.numel(),
                    "quantize_seconds": quantize_seconds,
                    "dequantize_seconds": dequantize_seconds,
                    **_metrics(source, reconstruction),
                }
            )

    rate = eight_mxfp4_sq3_rate(rows, columns)
    nvq3_budget = NVQ3_D4.payload_nbytes(rows, columns)
    if rate.payload_nbytes > nvq3_budget:
        raise RuntimeError("eight-state MXFP4-SQ3 exceeds NVQ3 payload")
    print(
        f"searching SQ3 eight-state: {rate.payload_nbytes} bytes / {rate.payload_bpw:.9f} BPW...",
        file=sys.stderr,
        flush=True,
    )
    search_started = time.perf_counter()
    reconstruction, _, metadata = quantize_eight_mxfp4_sq3(
        packed,
        source_scale_raw,
        matrix_scale_base=args.matrix_scale_base,
        maximum_refinement_steps=args.maximum_refinement_steps,
        progress=True,
    )
    search_seconds = time.perf_counter() - search_started
    candidate_metrics = _metrics(source, reconstruction)
    comparison = {
        baseline["format"]: 100.0
        * (float(candidate_metrics["error_sse"]) / float(baseline["error_sse"]) - 1.0)
        for baseline in baselines
    }
    candidate = {
        "format": "eight-state-native-MXFP4-SQ3-fixed32",
        **rate.__dict__,
        "budget_slack_vs_nvq3_nbytes": nvq3_budget - rate.payload_nbytes,
        "seconds": search_seconds,
        **candidate_metrics,
        "sse_delta_percent": comparison,
        **metadata,
    }
    result: dict[str, Any] = {
        "schema": 1,
        "experiment": "dsv4f-eight-state-native-mxfp4-three-bit-scalar-transcoding",
        "created_unix": started,
        "workspace": _git_identity(script_root),
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "baseline_device": str(torch.device(args.device)),
            "sq3_search_device": "cpu/numpy",
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
        "baselines": baselines,
        "eight_state_mxfp4_sq3": candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
