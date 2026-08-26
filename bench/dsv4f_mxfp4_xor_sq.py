#!/usr/bin/env python3
"""Search a four-state native-MXFP4 scalar format with in-band tags.

Each row stores four ``(E8M0 scale, E2M1 four-level palette)`` states.  Every
weight still stores one two-bit scalar symbol.  The XOR of the 32 symbols in a
native MXFP4 block is the two-bit row-state tag, so no explicit block selector
is stored.  Encoding enforces the tag with the minimum-cost one- or two-symbol
change.  Decoding derives the tag first and then expands the selected scalar
palette back to ordinary block-32 MXFP4.
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
    FIXED16_PALETTE_IDS,
    NIBBLE_VALUES,
    PALETTE_NIBBLES,
    PALETTE_VALUES,
    _raw_gate_up_native,
    _row_candidate_exponents,
    _unpack_source,
)
from bench.dsv4f_mxfp4_row_vq import (
    _git_identity,
    _metrics,
    _quantize_baseline,
    _sha256,
)
from mfq.formats.nvq import NVQ2_E8
from mfq.quantize.mxfp import decode_mxfp4
from mfq.quantize.v4f_source import V4FCheckpoint

XOR_STATES_PER_ROW = 4
XOR_TAG_BITS = 2


@dataclass(frozen=True)
class XorSqRate:
    bits_per_symbol: int
    native_block_size: int
    implicit_tag_bits: int
    explicit_block_selector_nbytes: int
    palette_id_bits: int
    states_per_row: int
    symbol_nbytes: int
    state_scale_nbytes: int
    state_palette_nbytes: int
    payload_nbytes: int
    payload_bpw: float


@dataclass(frozen=True)
class XorRowSolution:
    error_sse: float
    state_scale_raw: tuple[int, int, int, int]
    state_palette_ids: tuple[int, int, int, int]
    block_tags: np.ndarray
    refinement_steps: int


@dataclass(frozen=True)
class XorSqEncoding:
    state_scale_raw: np.ndarray
    state_palette_codes: np.ndarray
    packed_state_palettes: np.ndarray
    packed_symbols: np.ndarray
    block_tags: np.ndarray
    packed_mxfp4: np.ndarray
    native_scale_raw: np.ndarray
    searched_sse: float


def xor_mxfp4_sq_rate(rows: int, columns: int) -> XorSqRate:
    """Return the exact four-state XOR-tag payload."""

    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("XOR MXFP4-SQ requires positive block-32 matrices")
    bits_per_symbol = 2
    native_block_size = 32
    palette_id_bits = 4
    weights = rows * columns
    symbol_nbytes = (weights * bits_per_symbol + 7) // 8
    state_scale_nbytes = rows * XOR_STATES_PER_ROW
    state_palette_nbytes = (rows * XOR_STATES_PER_ROW * palette_id_bits + 7) // 8
    payload_nbytes = symbol_nbytes + state_scale_nbytes + state_palette_nbytes
    return XorSqRate(
        bits_per_symbol=bits_per_symbol,
        native_block_size=native_block_size,
        implicit_tag_bits=XOR_TAG_BITS,
        explicit_block_selector_nbytes=0,
        palette_id_bits=palette_id_bits,
        states_per_row=XOR_STATES_PER_ROW,
        symbol_nbytes=symbol_nbytes,
        state_scale_nbytes=state_scale_nbytes,
        state_palette_nbytes=state_palette_nbytes,
        payload_nbytes=payload_nbytes,
        payload_bpw=8.0 * payload_nbytes / weights,
    )


def _second_smallest(values: np.ndarray) -> np.ndarray:
    return np.partition(values, kth=1, axis=2)[:, :, 1]


def _distinct_pair_cost(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Minimum left[i] + right[j] subject to i != j."""

    left_index = left.argmin(axis=2)
    right_index = right.argmin(axis=2)
    left_min = np.take_along_axis(left, left_index[:, :, None], axis=2)[:, :, 0]
    right_min = np.take_along_axis(right, right_index[:, :, None], axis=2)[:, :, 0]
    collision = left_index == right_index
    direct = left_min + right_min
    repaired = np.minimum(
        left_min + _second_smallest(right),
        _second_smallest(left) + right_min,
    )
    return np.where(collision, repaired, direct)


def _candidate_xor_errors(
    source_nibbles_row: np.ndarray,
    source_scale_row: np.ndarray,
    state_scales: np.ndarray,
    state_palette_ids: np.ndarray,
) -> np.ndarray:
    """Return exact constrained errors with shape (tag, state, block)."""

    target = NIBBLE_VALUES[source_nibbles_row] * np.exp2(
        source_scale_row[:, None].astype(np.int16) - 127
    )
    levels = PALETTE_VALUES[state_palette_ids] * np.exp2(
        state_scales[:, None].astype(np.int16) - 127
    )
    costs = np.square(target[None, :, :, None] - levels[:, None, None, :])
    base_symbols = costs.argmin(axis=3).astype(np.uint8)
    base_costs = np.take_along_axis(costs, base_symbols[:, :, :, None], axis=3)[:, :, :, 0]
    base_error = base_costs.sum(axis=2)
    base_xor = np.bitwise_xor.reduce(base_symbols, axis=2)

    penalties = [np.zeros_like(base_costs)]
    for delta in range(1, 4):
        alternative = base_symbols ^ np.uint8(delta)
        penalties.append(
            np.take_along_axis(costs, alternative[:, :, :, None], axis=3)[:, :, :, 0] - base_costs
        )
    single = [item.min(axis=2) for item in penalties]
    adjustment = np.stack(
        (
            np.zeros_like(base_error),
            np.minimum(single[1], _distinct_pair_cost(penalties[2], penalties[3])),
            np.minimum(single[2], _distinct_pair_cost(penalties[1], penalties[3])),
            np.minimum(single[3], _distinct_pair_cost(penalties[1], penalties[2])),
        ),
        axis=2,
    )
    result = np.empty((4, len(state_scales), target.shape[0]), dtype=np.float64)
    for tag in range(4):
        delta = base_xor ^ np.uint8(tag)
        result[tag] = (
            base_error + np.take_along_axis(adjustment, delta[:, :, None], axis=2)[:, :, 0]
        )
    if not np.allclose(result.min(axis=0), base_error, rtol=0.0, atol=0.0):
        raise RuntimeError("XOR constraints lost the unconstrained state")
    return result


def _rank_partition(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="stable")
    result = np.empty(order.size, dtype=np.uint8)
    result[order] = np.minimum(
        XOR_STATES_PER_ROW - 1,
        np.arange(order.size) * XOR_STATES_PER_ROW // order.size,
    )
    return result


def _refine_unconstrained_states(
    errors: np.ndarray,
    initial_assignments: np.ndarray,
    *,
    maximum_steps: int,
) -> tuple[int, int, int, int]:
    assignments = np.asarray(initial_assignments, dtype=np.uint8).copy()
    states = np.zeros(XOR_STATES_PER_ROW, dtype=np.int64)
    for _ in range(maximum_steps):
        for state_tag in range(XOR_STATES_PER_ROW):
            selected = assignments == state_tag
            if bool(selected.any()):
                states[state_tag] = int(errors[:, selected].sum(axis=1).argmin())
        updated = np.stack([errors[state] for state in states]).argmin(axis=0)
        if np.array_equal(updated, assignments):
            break
        assignments = updated.astype(np.uint8)
    return tuple(int(item) for item in states)


def _unconstrained_state_sets(
    source_nibbles_row: np.ndarray,
    source_scale_row: np.ndarray,
    constrained_errors: np.ndarray,
    *,
    maximum_steps: int,
) -> list[tuple[int, int, int, int]]:
    errors = constrained_errors.min(axis=0)
    blocks = errors.shape[1]
    selected_states: list[int] = []
    combined = np.full(blocks, np.inf, dtype=np.float64)
    for _ in range(XOR_STATES_PER_ROW):
        totals = np.minimum(errors, combined[None, :]).sum(axis=1)
        if selected_states:
            totals[np.asarray(selected_states)] = np.inf
        selected = int(totals.argmin())
        selected_states.append(selected)
        combined = np.minimum(combined, errors[selected])
    greedy_assignments = np.stack([errors[state] for state in selected_states]).argmin(axis=0)

    target = NIBBLE_VALUES[source_nibbles_row] * np.exp2(
        source_scale_row[:, None].astype(np.int16) - 127
    )
    starts = [
        greedy_assignments,
        _rank_partition(source_scale_row),
        _rank_partition(target.mean(axis=1)),
        _rank_partition(np.square(target).sum(axis=1)),
        _rank_partition(np.abs(target).max(axis=1)),
        np.arange(blocks, dtype=np.uint8) % XOR_STATES_PER_ROW,
        (np.arange(blocks, dtype=np.uint8) // 2) % XOR_STATES_PER_ROW,
    ]
    result: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, ...]] = set()
    for start in starts:
        states = _refine_unconstrained_states(
            errors,
            start,
            maximum_steps=maximum_steps,
        )
        key = tuple(sorted(states))
        if key not in seen:
            seen.add(key)
            result.append(states)
    return result


def _refine_constrained_states(
    constrained_errors: np.ndarray,
    initial_states: tuple[int, int, int, int],
    *,
    maximum_steps: int,
) -> tuple[float, tuple[int, int, int, int], np.ndarray, int]:
    states = np.asarray(initial_states, dtype=np.int64)
    state_error = np.stack([constrained_errors[tag, states[tag]] for tag in range(4)])
    assignments = state_error.argmin(axis=0).astype(np.uint8)
    best = (float(state_error.min(axis=0).sum()), states.copy(), assignments.copy(), 0)
    for step in range(1, maximum_steps + 1):
        updated_states = states.copy()
        for tag in range(4):
            selected = assignments == tag
            if bool(selected.any()):
                updated_states[tag] = int(constrained_errors[tag, :, selected].sum(axis=0).argmin())
        state_error = np.stack([constrained_errors[tag, updated_states[tag]] for tag in range(4)])
        updated_assignments = state_error.argmin(axis=0).astype(np.uint8)
        error = float(state_error.min(axis=0).sum())
        if error < best[0]:
            best = (
                error,
                updated_states.copy(),
                updated_assignments.copy(),
                step,
            )
        if np.array_equal(updated_states, states) and np.array_equal(
            updated_assignments, assignments
        ):
            break
        states = updated_states
        assignments = updated_assignments
    return best[0], tuple(int(item) for item in best[1]), best[2], best[3]


def _solve_xor_row(
    source_nibbles_row: np.ndarray,
    source_scale_row: np.ndarray,
    *,
    exponent_radius: int,
    maximum_steps: int,
) -> XorRowSolution:
    candidate_exponents = _row_candidate_exponents(source_scale_row, exponent_radius)
    state_scales = np.repeat(
        np.asarray(candidate_exponents, dtype=np.uint8),
        len(FIXED16_PALETTE_IDS),
    )
    state_palette_ids = np.tile(FIXED16_PALETTE_IDS, len(candidate_exponents))
    errors = _candidate_xor_errors(
        source_nibbles_row,
        source_scale_row,
        state_scales,
        state_palette_ids,
    )
    state_sets = _unconstrained_state_sets(
        source_nibbles_row,
        source_scale_row,
        errors,
        maximum_steps=maximum_steps,
    )
    candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray, int]] = []
    for state_set in state_sets:
        ordered_candidates = []
        for permutation in set(itertools.permutations(state_set)):
            error = float(
                np.stack([errors[tag, permutation[tag]] for tag in range(4)]).min(axis=0).sum()
            )
            ordered_candidates.append((error, permutation))
        for _, ordered in sorted(ordered_candidates)[:4]:
            candidates.append(
                _refine_constrained_states(
                    errors,
                    ordered,
                    maximum_steps=maximum_steps,
                )
            )
    error, state_indices, block_tags, refinement_steps = min(candidates, key=lambda item: item[0])
    return XorRowSolution(
        error_sse=error,
        state_scale_raw=tuple(int(state_scales[index]) for index in state_indices),
        state_palette_ids=tuple(int(state_palette_ids[index]) for index in state_indices),
        block_tags=block_tags,
        refinement_steps=refinement_steps,
    )


def _best_distinct_pair_indices(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_index = left.argmin(axis=1)
    right_index = right.argmin(axis=1)
    rows = np.arange(left.shape[0])
    collision = left_index == right_index
    left_alternative = left.copy()
    right_alternative = right.copy()
    left_alternative[rows, left_index] = np.inf
    right_alternative[rows, right_index] = np.inf
    second_left_index = left_alternative.argmin(axis=1)
    second_right_index = right_alternative.argmin(axis=1)
    use_second_right = (
        left[rows, left_index] + right[rows, second_right_index]
        <= left[rows, second_left_index] + right[rows, right_index]
    )
    repaired_left = np.where(use_second_right, left_index, second_left_index)
    repaired_right = np.where(use_second_right, second_right_index, right_index)
    left_index = np.where(collision, repaired_left, left_index)
    right_index = np.where(collision, repaired_right, right_index)
    cost = left[rows, left_index] + right[rows, right_index]
    return cost, left_index, right_index


def _quantize_blocks_for_tag(
    target: np.ndarray,
    levels: np.ndarray,
    tag: int,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.square(target[:, :, None] - levels[None, None, :])
    symbols = costs.argmin(axis=2).astype(np.uint8)
    rows = np.arange(target.shape[0])
    positions = np.arange(target.shape[1])
    base_cost = costs[rows[:, None], positions[None, :], symbols]
    base_xor = np.bitwise_xor.reduce(symbols, axis=1)
    delta = base_xor ^ np.uint8(tag)
    penalties = [np.zeros_like(base_cost)]
    for change in range(1, 4):
        penalties.append(
            costs[
                rows[:, None],
                positions[None, :],
                symbols ^ np.uint8(change),
            ]
            - base_cost
        )
    pair_changes = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    for change in range(1, 4):
        selected = np.flatnonzero(delta == change)
        if not selected.size:
            continue
        single_position = penalties[change][selected].argmin(axis=1)
        single_cost = penalties[change][selected, single_position]
        left_change, right_change = pair_changes[change]
        pair_cost, left_position, right_position = _best_distinct_pair_indices(
            penalties[left_change][selected], penalties[right_change][selected]
        )
        use_pair = pair_cost < single_cost
        single_blocks = selected[~use_pair]
        symbols[single_blocks, single_position[~use_pair]] ^= np.uint8(change)
        pair_blocks = selected[use_pair]
        symbols[pair_blocks, left_position[use_pair]] ^= np.uint8(left_change)
        symbols[pair_blocks, right_position[use_pair]] ^= np.uint8(right_change)
    if not np.all(np.bitwise_xor.reduce(symbols, axis=1) == tag):
        raise RuntimeError("failed to embed XOR block tag")
    error = costs[rows[:, None], positions[None, :], symbols].sum(axis=1)
    return symbols, error


def _pack_two_bit_symbols(symbols: np.ndarray) -> np.ndarray:
    values = np.asarray(symbols, dtype=np.uint8)
    if values.shape[-1] % 4:
        raise ValueError("two-bit symbol count must be divisible by four")
    return (
        values[..., 0::4]
        | (values[..., 1::4] << 2)
        | (values[..., 2::4] << 4)
        | (values[..., 3::4] << 6)
    )


def _unpack_two_bit_symbols(packed: np.ndarray) -> np.ndarray:
    values = np.asarray(packed, dtype=np.uint8)
    result = np.empty((*values.shape[:-1], values.shape[-1] * 4), dtype=np.uint8)
    result[..., 0::4] = values & 0x03
    result[..., 1::4] = (values >> 2) & 0x03
    result[..., 2::4] = (values >> 4) & 0x03
    result[..., 3::4] = values >> 6
    return result


def _pack_palette_codes(codes: np.ndarray) -> np.ndarray:
    values = np.asarray(codes, dtype=np.uint8)
    if values.shape[-1] != 4 or int(values.max()) >= 16:
        raise ValueError("four local four-bit palette codes are required")
    return values[:, 0::2] | (values[:, 1::2] << 4)


def _unpack_palette_codes(packed: np.ndarray) -> np.ndarray:
    values = np.asarray(packed, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("packed XOR palette metadata must be rows by two")
    result = np.empty((values.shape[0], 4), dtype=np.uint8)
    result[:, 0::2] = values & 0x0F
    result[:, 1::2] = values >> 4
    return result


def decode_xor_mxfp4_sq(
    packed_symbols: np.ndarray,
    state_scale_raw: np.ndarray,
    packed_state_palettes: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Decode the physical XOR-SQ payload through native MXFP4."""

    symbols = _unpack_two_bit_symbols(packed_symbols)
    rows, columns = symbols.shape
    if columns % 32 or tuple(state_scale_raw.shape) != (rows, 4):
        raise ValueError("XOR-SQ payload geometry mismatch")
    blocks = columns // 32
    block_symbols = symbols.reshape(rows, blocks, 32)
    block_tags = np.bitwise_xor.reduce(block_symbols, axis=2)
    local_palette_codes = _unpack_palette_codes(packed_state_palettes)
    state_palette_ids = FIXED16_PALETTE_IDS[local_palette_codes]
    local_rows = np.arange(rows)[:, None]
    selected_scale = np.asarray(state_scale_raw, dtype=np.uint8)[local_rows, block_tags]
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


def _materialize_xor_encoding(
    source_nibbles: np.ndarray,
    source_scale_raw: np.ndarray,
    solutions: list[XorRowSolution],
) -> XorSqEncoding:
    rows, blocks, block_size = source_nibbles.shape
    state_scale_raw = np.asarray(
        [solution.state_scale_raw for solution in solutions], dtype=np.uint8
    )
    state_palette_ids = np.asarray(
        [solution.state_palette_ids for solution in solutions], dtype=np.int16
    )
    palette_to_local = {
        int(palette_id): index for index, palette_id in enumerate(FIXED16_PALETTE_IDS)
    }
    state_palette_codes = np.vectorize(palette_to_local.__getitem__, otypes=[np.uint8])(
        state_palette_ids
    )
    block_tags = np.stack([solution.block_tags for solution in solutions]).astype(
        np.uint8, copy=False
    )
    symbols = np.empty((rows, blocks, block_size), dtype=np.uint8)
    measured_search_sse = 0.0
    for row in range(rows):
        target = NIBBLE_VALUES[source_nibbles[row]] * np.exp2(
            source_scale_raw[row, :, None].astype(np.int16) - 127
        )
        for tag in range(4):
            selected = block_tags[row] == tag
            if not bool(selected.any()):
                continue
            levels = PALETTE_VALUES[state_palette_ids[row, tag]] * math.ldexp(
                1.0, int(state_scale_raw[row, tag]) - 127
            )
            local_symbols, local_error = _quantize_blocks_for_tag(target[selected], levels, tag)
            symbols[row, selected] = local_symbols
            measured_search_sse += float(local_error.sum())
    packed_symbols = _pack_two_bit_symbols(symbols.reshape(rows, blocks * block_size))
    packed_state_palettes = _pack_palette_codes(state_palette_codes)
    reconstruction, packed_mxfp4, native_scale_raw, decoded_tags = decode_xor_mxfp4_sq(
        packed_symbols,
        state_scale_raw,
        packed_state_palettes,
        device="cpu",
    )
    if not np.array_equal(decoded_tags, block_tags):
        raise RuntimeError("stored symbols did not recover searched XOR tags")
    searched_sse = float(sum(solution.error_sse for solution in solutions))
    if not math.isclose(
        searched_sse,
        measured_search_sse,
        rel_tol=2e-9,
        abs_tol=1e-10,
    ):
        raise RuntimeError(
            f"XOR-SQ search/materialization SSE mismatch: {searched_sse} != {measured_search_sse}"
        )
    return XorSqEncoding(
        state_scale_raw=state_scale_raw,
        state_palette_codes=state_palette_codes,
        packed_state_palettes=packed_state_palettes,
        packed_symbols=packed_symbols,
        block_tags=block_tags,
        packed_mxfp4=packed_mxfp4,
        native_scale_raw=native_scale_raw,
        searched_sse=searched_sse,
    )


def _xor_metadata(
    encoding: XorSqEncoding,
    source_scale_raw: np.ndarray,
    solutions: list[XorRowSolution],
    *,
    exponent_radius: int,
) -> dict[str, Any]:
    palette_ids = FIXED16_PALETTE_IDS[encoding.state_palette_codes]
    palette_counts = Counter(int(item) for item in palette_ids.flat)
    tag_counts = Counter(int(item) for item in encoding.block_tags.flat)
    return {
        "state_layout": "four-row-states:(E8M0-scale,fixed16-palette-id)",
        "state_tag": "xor-reduction-of-32-two-bit-symbols",
        "explicit_block_selector_bits": 0,
        "scale_dtype": "E8M0",
        "scale_exponent_radius": exponent_radius,
        "optimizer": "deterministic-multistart-hard-em-with-exact-xor-costs",
        "optimizer_is_global_within_catalog": False,
        "maximum_refinement_steps_used": max(
            (solution.refinement_steps for solution in solutions), default=0
        ),
        "state_scale_byte_min": int(encoding.state_scale_raw.min()),
        "state_scale_byte_max": int(encoding.state_scale_raw.max()),
        "effective_scale_matches_source_fraction": float(
            np.mean(encoding.native_scale_raw == source_scale_raw)
        ),
        "block_tag_fractions": {
            str(tag): tag_counts[tag] / encoding.block_tags.size for tag in range(4)
        },
        "fixed16_full_catalog_ids": FIXED16_PALETTE_IDS.tolist(),
        "fixed16_palette_values": PALETTE_VALUES[FIXED16_PALETTE_IDS].tolist(),
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
        "learned_vector_codebook": False,
        "stored_fp16_scale_or_centroid": False,
        "physical_storage_roundtrip_verified": True,
        "final_values_are_native_block32_mxfp4": True,
    }


@torch.inference_mode()
def quantize_xor_mxfp4_sq(
    packed: np.ndarray,
    source_scale_raw: np.ndarray,
    *,
    exponent_radius: int = 0,
    maximum_refinement_steps: int = 10,
    progress: bool = False,
) -> tuple[torch.Tensor, XorSqEncoding, dict[str, Any]]:
    """Encode native MXFP4 with four row states and in-band XOR tags."""

    if exponent_radius < 0 or maximum_refinement_steps <= 0:
        raise ValueError("invalid XOR-SQ search control")
    source_nibbles, _ = _unpack_source(packed, source_scale_raw)
    source_scales = np.asarray(source_scale_raw, dtype=np.uint8)
    solutions: list[XorRowSolution] = []
    for row in range(source_nibbles.shape[0]):
        solutions.append(
            _solve_xor_row(
                source_nibbles[row],
                source_scales[row],
                exponent_radius=exponent_radius,
                maximum_steps=maximum_refinement_steps,
            )
        )
        if progress and ((row + 1) % 256 == 0 or row + 1 == source_nibbles.shape[0]):
            print(
                f"xor4: solved {row + 1}/{source_nibbles.shape[0]} rows",
                file=sys.stderr,
                flush=True,
            )
    encoding = _materialize_xor_encoding(
        source_nibbles,
        source_scales,
        solutions,
    )
    reconstruction, _, _, _ = decode_xor_mxfp4_sq(
        encoding.packed_symbols,
        encoding.state_scale_raw,
        encoding.packed_state_palettes,
        device="cpu",
    )
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
            f"XOR-SQ physical decode SSE mismatch: {encoding.searched_sse} != {measured_sse}"
        )
    return (
        reconstruction,
        encoding,
        _xor_metadata(
            encoding,
            source_scales,
            solutions,
            exponent_radius=exponent_radius,
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--exponent-radius", type=int, default=0)
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

    rate = xor_mxfp4_sq_rate(rows, columns)
    if rate.payload_nbytes > baseline_budget:
        raise RuntimeError("XOR-SQ exceeds NVQ2 payload")
    print(
        f"searching xor4: {rate.payload_nbytes} bytes / {rate.payload_bpw:.9f} BPW...",
        file=sys.stderr,
        flush=True,
    )
    search_start = time.perf_counter()
    reconstruction, _, metadata = quantize_xor_mxfp4_sq(
        packed,
        source_scale_raw,
        exponent_radius=args.exponent_radius,
        maximum_refinement_steps=args.maximum_refinement_steps,
        progress=True,
    )
    search_seconds = time.perf_counter() - search_start
    baseline_metrics = _metrics(source, baseline_reconstruction)
    metrics = _metrics(source, reconstruction)
    candidate = {
        "format": "xor-tagged-four-state-native-MXFP4-SQ-fixed16",
        **rate.__dict__,
        "budget_slack_nbytes": baseline_budget - rate.payload_nbytes,
        "seconds": search_seconds,
        **metrics,
        "sse_delta_vs_nvq2_percent": 100.0
        * (float(metrics["error_sse"]) / float(baseline_metrics["error_sse"]) - 1.0),
        **metadata,
    }
    print(
        f"xor4: SSE={metrics['error_sse']:.9f}, SNR={metrics['snr_db']:.6f} dB",
        file=sys.stderr,
        flush=True,
    )
    result: dict[str, Any] = {
        "schema": 1,
        "experiment": "dsv4f-xor-tagged-native-mxfp4-scalar-transcoding",
        "created_unix": started,
        "workspace": _git_identity(script_root),
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "baseline_device": str(torch.device(args.device)),
            "xor_search_device": "cpu/numpy",
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
        "xor_mxfp4_sq": candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
