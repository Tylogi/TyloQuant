#!/usr/bin/env python3
"""Search a four-state native-output MXFP4 three-bit scalar format.

Each block stores one explicit selector bit.  The XOR parity of the low bit
of its 32 three-bit symbols stores a second selector bit.  Four row-local
states therefore fit below the equal-shape NVQ3 payload.  Every state remains
one E8M0 scale and one eight-value E2M1 palette, so physical decoding expands
directly to standard block-32 MXFP4.

The 32-palette catalog is frozen from the expert-0 full-catalog design run.
It is a format constant rather than per-model learned storage.
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
    _row_candidate_exponents,
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
from mfq.formats.nvq import NVQ3_D4, NVQ3_D4_512
from mfq.quantize.mxfp import decode_mxfp4
from mfq.quantize.v4f_source import V4FCheckpoint

FIXED32_SQ3_PALETTE_IDS = np.asarray(
    (
        2407,
        499,
        1483,
        500,
        2401,
        2011,
        522,
        2002,
        254,
        2423,
        2120,
        4705,
        255,
        1196,
        2989,
        1499,
        2003,
        523,
        2001,
        5497,
        5496,
        1077,
        3726,
        4116,
        3717,
        2010,
        1478,
        3727,
        1087,
        4117,
        2012,
        4704,
    ),
    dtype=np.int16,
)


@dataclass(frozen=True)
class HybridSq3Rate:
    bits_per_symbol: int
    native_block_size: int
    explicit_block_selector_bits: int
    implicit_symbol_tag_bits: int
    palette_id_bits: int
    states_per_row: int
    symbol_nbytes: int
    block_selector_nbytes: int
    state_scale_nbytes: int
    state_palette_nbytes: int
    payload_nbytes: int
    payload_bpw: float


@dataclass(frozen=True)
class HybridSq3RowSolution:
    error_sse: float
    state_scale_raw: tuple[int, int, int, int]
    state_palette_ids: tuple[int, int, int, int]
    block_tags: np.ndarray
    refinement_steps: int


@dataclass(frozen=True)
class HybridSq3Encoding:
    state_scale_raw: np.ndarray
    state_palette_codes: np.ndarray
    packed_state_palettes: np.ndarray
    packed_symbols: np.ndarray
    packed_block_selectors: np.ndarray
    block_tags: np.ndarray
    packed_mxfp4: np.ndarray
    native_scale_raw: np.ndarray
    searched_sse: float


def hybrid_mxfp4_sq3_rate(rows: int, columns: int) -> HybridSq3Rate:
    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("hybrid MXFP4-SQ3 requires positive block-32 matrices")
    weights = rows * columns
    blocks = weights // 32
    symbol_nbytes = (weights * 3 + 7) // 8
    block_selector_nbytes = (blocks + 7) // 8
    state_scale_nbytes = rows * 4
    state_palette_nbytes = (rows * 4 * 5 + 7) // 8
    payload_nbytes = (
        symbol_nbytes + block_selector_nbytes + state_scale_nbytes + state_palette_nbytes
    )
    return HybridSq3Rate(
        bits_per_symbol=3,
        native_block_size=32,
        explicit_block_selector_bits=1,
        implicit_symbol_tag_bits=1,
        palette_id_bits=5,
        states_per_row=4,
        symbol_nbytes=symbol_nbytes,
        block_selector_nbytes=block_selector_nbytes,
        state_scale_nbytes=state_scale_nbytes,
        state_palette_nbytes=state_palette_nbytes,
        payload_nbytes=payload_nbytes,
        payload_bpw=8.0 * payload_nbytes / weights,
    )


def _candidate_parity_errors(
    source_nibbles_row: np.ndarray,
    source_scale_row: np.ndarray,
    state_scales: np.ndarray,
    state_palette_ids: np.ndarray,
) -> np.ndarray:
    """Return exact errors with shape (required parity, state, block)."""

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
    base_parity = np.bitwise_xor.reduce(base_symbols & 1, axis=2)

    symbol_parity = np.arange(8, dtype=np.uint8) & 1
    opposite_cost = np.empty_like(base_costs)
    for parity in (0, 1):
        selected = base_symbols % 2 == parity
        alternative = costs[:, :, :, symbol_parity != parity].min(axis=3)
        opposite_cost[selected] = alternative[selected]
    parity_flip_penalty = (opposite_cost - base_costs).min(axis=2)
    errors = np.empty((2, len(state_scales), target.shape[0]), dtype=np.float64)
    for required_parity in (0, 1):
        errors[required_parity] = base_error + np.where(
            base_parity == required_parity,
            0.0,
            parity_flip_penalty,
        )
    if not np.allclose(errors.min(axis=0), base_error, rtol=0.0, atol=0.0):
        raise RuntimeError("SQ3 parity constraints lost the unconstrained solution")
    return errors


def _rank_partition(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="stable")
    result = np.empty(order.size, dtype=np.uint8)
    result[order] = np.minimum(3, np.arange(order.size) * 4 // order.size)
    return result


def _refine_from_assignments(
    parity_errors: np.ndarray,
    initial_assignments: np.ndarray,
    *,
    maximum_steps: int,
) -> tuple[float, tuple[int, int, int, int], np.ndarray, int]:
    assignments = np.asarray(initial_assignments, dtype=np.uint8).copy()
    states = np.zeros(4, dtype=np.int64)
    best: tuple[float, np.ndarray, np.ndarray, int] | None = None
    for step in range(1, maximum_steps + 1):
        updated_states = states.copy()
        for tag in range(4):
            selected = assignments == tag
            if bool(selected.any()):
                updated_states[tag] = int(parity_errors[tag & 1, :, selected].sum(axis=0).argmin())
        state_error = np.stack([parity_errors[tag & 1, updated_states[tag]] for tag in range(4)])
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
        raise RuntimeError("hybrid SQ3 refinement produced no candidate")
    return best[0], tuple(int(item) for item in best[1]), best[2], best[3]


def _greedy_assignments(parity_errors: np.ndarray, parity_order: tuple[int, ...]) -> np.ndarray:
    blocks = parity_errors.shape[2]
    covered = np.full(blocks, np.inf, dtype=np.float64)
    chosen: list[int] = []
    chosen_parities: list[int] = []
    for parity in parity_order:
        totals = np.minimum(parity_errors[parity], covered[None, :]).sum(axis=1)
        for previous, previous_parity in zip(chosen, chosen_parities, strict=True):
            if previous_parity == parity:
                totals[previous] = np.inf
        state = int(totals.argmin())
        chosen.append(state)
        chosen_parities.append(parity)
        covered = np.minimum(covered, parity_errors[parity, state])
    state_error = np.stack(
        [parity_errors[parity, state] for parity, state in zip(parity_order, chosen, strict=True)]
    )
    local_assignment = state_error.argmin(axis=0)
    parity_tags = {0: (0, 2), 1: (1, 3)}
    occurrences = [0, 0]
    tags = np.empty(4, dtype=np.uint8)
    for local, parity in enumerate(parity_order):
        tags[local] = parity_tags[parity][occurrences[parity]]
        occurrences[parity] += 1
    return tags[local_assignment]


def _solve_hybrid_row(
    source_nibbles_row: np.ndarray,
    source_scale_row: np.ndarray,
    *,
    exponent_radius: int,
    maximum_steps: int,
) -> HybridSq3RowSolution:
    candidate_exponents = _row_candidate_exponents(
        source_scale_row,
        exponent_radius,
    )
    state_scales = np.repeat(
        np.asarray(candidate_exponents, dtype=np.uint8),
        len(FIXED32_SQ3_PALETTE_IDS),
    )
    state_palette_ids = np.tile(
        FIXED32_SQ3_PALETTE_IDS,
        len(candidate_exponents),
    )
    errors = _candidate_parity_errors(
        source_nibbles_row,
        source_scale_row,
        state_scales,
        state_palette_ids,
    )
    target = NIBBLE_VALUES[source_nibbles_row] * np.exp2(
        source_scale_row[:, None].astype(np.int16) - 127
    )
    cluster_starts = [
        _rank_partition(source_scale_row),
        _rank_partition(target.mean(axis=1)),
        _rank_partition(np.square(target).sum(axis=1)),
        _rank_partition(np.abs(target).max(axis=1)),
        _rank_partition(np.count_nonzero(target, axis=1)),
        np.arange(target.shape[0], dtype=np.uint8) % 4,
        (np.arange(target.shape[0], dtype=np.uint8) // 2) % 4,
    ]
    candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray, int]] = []
    for clusters in cluster_starts:
        for parity_zero_clusters in itertools.combinations(range(4), 2):
            parity_zero = set(parity_zero_clusters)
            tags = np.empty(4, dtype=np.uint8)
            occurrence = [0, 0]
            parity_tags = {0: (0, 2), 1: (1, 3)}
            for cluster in range(4):
                parity = 0 if cluster in parity_zero else 1
                tags[cluster] = parity_tags[parity][occurrence[parity]]
                occurrence[parity] += 1
            candidates.append(
                _refine_from_assignments(
                    errors,
                    tags[clusters],
                    maximum_steps=maximum_steps,
                )
            )
    for parity_order in set(itertools.permutations((0, 0, 1, 1))):
        candidates.append(
            _refine_from_assignments(
                errors,
                _greedy_assignments(errors, parity_order),
                maximum_steps=maximum_steps,
            )
        )
    error, state_indices, block_tags, refinement_steps = min(
        candidates,
        key=lambda item: item[0],
    )
    return HybridSq3RowSolution(
        error_sse=error,
        state_scale_raw=tuple(int(state_scales[index]) for index in state_indices),
        state_palette_ids=tuple(int(state_palette_ids[index]) for index in state_indices),
        block_tags=block_tags,
        refinement_steps=refinement_steps,
    )


def _quantize_blocks_for_parity(
    target: np.ndarray,
    levels: np.ndarray,
    required_parity: int,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.square(target[:, :, None] - levels[None, None, :])
    symbols = costs.argmin(axis=2).astype(np.uint8)
    rows = np.arange(target.shape[0])
    positions = np.arange(target.shape[1])
    base_cost = costs[rows[:, None], positions[None, :], symbols]
    base_parity = np.bitwise_xor.reduce(symbols & 1, axis=1)
    repair_blocks = np.flatnonzero(base_parity != required_parity)
    if repair_blocks.size:
        source_symbols = symbols[repair_blocks]
        symbol_parity = np.arange(8, dtype=np.uint8) & 1
        alternative_cost = np.empty_like(base_cost[repair_blocks])
        alternative_symbol = np.empty_like(source_symbols)
        for parity in (0, 1):
            selected = source_symbols % 2 == parity
            alternatives = np.flatnonzero(symbol_parity != parity)
            local_cost = costs[repair_blocks][:, :, alternatives]
            local_index = local_cost.argmin(axis=2)
            local_symbol = alternatives[local_index]
            local_min = np.take_along_axis(
                local_cost,
                local_index[:, :, None],
                axis=2,
            )[:, :, 0]
            alternative_cost[selected] = local_min[selected]
            alternative_symbol[selected] = local_symbol[selected]
        penalty = alternative_cost - base_cost[repair_blocks]
        repair_position = penalty.argmin(axis=1)
        symbols[repair_blocks, repair_position] = alternative_symbol[
            np.arange(repair_blocks.size), repair_position
        ]
    if not np.all(np.bitwise_xor.reduce(symbols & 1, axis=1) == required_parity):
        raise RuntimeError("failed to embed SQ3 parity tag")
    error = costs[rows[:, None], positions[None, :], symbols].sum(axis=1)
    return symbols, error


def decode_hybrid_mxfp4_sq3(
    packed_symbols: np.ndarray,
    packed_block_selectors: np.ndarray,
    state_scale_raw: np.ndarray,
    packed_state_palettes: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    symbols = _unpack_three_bit_symbols(packed_symbols)
    rows, columns = symbols.shape
    if columns % 32 or tuple(state_scale_raw.shape) != (rows, 4):
        raise ValueError("hybrid MXFP4-SQ3 payload geometry mismatch")
    blocks = columns // 32
    block_symbols = symbols.reshape(rows, blocks, 32)
    explicit_high_bit = _unpack_block_selectors(
        packed_block_selectors,
        rows=rows,
        blocks=blocks,
    )
    implicit_low_bit = np.bitwise_xor.reduce(block_symbols & 1, axis=2)
    block_tags = (explicit_high_bit << 1) | implicit_low_bit
    palette_codes = _unpack_fixed_width(
        packed_state_palettes,
        5,
        count=rows * 4,
    ).reshape(rows, 4)
    state_palette_ids = FIXED32_SQ3_PALETTE_IDS[palette_codes]
    local_rows = np.arange(rows)[:, None]
    selected_scale = np.asarray(state_scale_raw, dtype=np.uint8)[local_rows, block_tags]
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
    solutions: list[HybridSq3RowSolution],
) -> HybridSq3Encoding:
    rows, blocks, block_size = source_nibbles.shape
    state_scale_raw = np.asarray(
        [solution.state_scale_raw for solution in solutions], dtype=np.uint8
    )
    state_palette_ids = np.asarray(
        [solution.state_palette_ids for solution in solutions], dtype=np.int16
    )
    palette_to_local = {
        int(palette_id): index for index, palette_id in enumerate(FIXED32_SQ3_PALETTE_IDS)
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
        for tag in range(4):
            selected = block_tags[row] == tag
            if not bool(selected.any()):
                continue
            palette_id = int(state_palette_ids[row, tag])
            levels = SQ3_PALETTE_VALUES[palette_id] * math.ldexp(
                1.0,
                int(state_scale_raw[row, tag]) - 127,
            )
            local_symbols, local_error = _quantize_blocks_for_parity(
                target[selected],
                levels,
                tag & 1,
            )
            symbols[row, selected] = local_symbols
            output_nibbles[row, selected] = SQ3_PALETTE_NIBBLES[palette_id][local_symbols]
            native_scale_raw[row, selected] = state_scale_raw[row, tag]
            measured_search_sse += float(local_error.sum())
    packed_symbols = _pack_three_bit_symbols(symbols.reshape(rows, blocks * block_size))
    packed_block_selectors = _pack_block_selectors(block_tags >> 1)
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
            f"hybrid SQ3 search/materialization SSE mismatch: {searched_sse} != {measured_search_sse}"
        )
    encoding = HybridSq3Encoding(
        state_scale_raw=state_scale_raw,
        state_palette_codes=state_palette_codes,
        packed_state_palettes=packed_state_palettes,
        packed_symbols=packed_symbols,
        packed_block_selectors=packed_block_selectors,
        block_tags=block_tags,
        packed_mxfp4=packed_mxfp4,
        native_scale_raw=native_scale_raw,
        searched_sse=searched_sse,
    )
    _, decoded_mxfp4, decoded_scale, decoded_tags = decode_hybrid_mxfp4_sq3(
        encoding.packed_symbols,
        encoding.packed_block_selectors,
        encoding.state_scale_raw,
        encoding.packed_state_palettes,
        device="cpu",
    )
    if not np.array_equal(decoded_mxfp4, packed_mxfp4):
        raise RuntimeError("hybrid SQ3 physical payload changed nibbles")
    if not np.array_equal(decoded_scale, native_scale_raw):
        raise RuntimeError("hybrid SQ3 physical payload changed scales")
    if not np.array_equal(decoded_tags, block_tags):
        raise RuntimeError("hybrid SQ3 physical payload changed block tags")
    return encoding


def _metadata(
    encoding: HybridSq3Encoding,
    source_scale_raw: np.ndarray,
    solutions: list[HybridSq3RowSolution],
    *,
    exponent_radius: int,
) -> dict[str, Any]:
    palette_ids = FIXED32_SQ3_PALETTE_IDS[encoding.state_palette_codes]
    palette_counts = Counter(int(item) for item in palette_ids.flat)
    tag_counts = Counter(int(item) for item in encoding.block_tags.flat)
    return {
        "state_layout": "four-row-states:(E8M0-scale,fixed32-eight-level-E2M1-palette)",
        "state_tag": "explicit-high-bit-plus-symbol-low-bit-xor-parity",
        "explicit_block_selector_bits": 1,
        "implicit_symbol_tag_bits": 1,
        "scale_dtype": "E8M0",
        "scale_exponent_radius": exponent_radius,
        "optimizer": "deterministic-multistart-hard-em-with-exact-parity-cost",
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
        "fixed32_full_catalog_ids": FIXED32_SQ3_PALETTE_IDS.tolist(),
        "fixed32_palette_values": SQ3_PALETTE_VALUES[FIXED32_SQ3_PALETTE_IDS].tolist(),
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
def quantize_hybrid_mxfp4_sq3(
    packed: np.ndarray,
    source_scale_raw: np.ndarray,
    *,
    exponent_radius: int = 0,
    maximum_refinement_steps: int = 10,
    progress: bool = False,
) -> tuple[torch.Tensor, HybridSq3Encoding, dict[str, Any]]:
    if exponent_radius < 0 or maximum_refinement_steps <= 0:
        raise ValueError("invalid hybrid SQ3 search control")
    source_nibbles, _ = _unpack_source(packed, source_scale_raw)
    source_scales = np.asarray(source_scale_raw, dtype=np.uint8)
    solutions: list[HybridSq3RowSolution] = []
    for row in range(source_nibbles.shape[0]):
        solutions.append(
            _solve_hybrid_row(
                source_nibbles[row],
                source_scales[row],
                exponent_radius=exponent_radius,
                maximum_steps=maximum_refinement_steps,
            )
        )
        if progress and ((row + 1) % 256 == 0 or row + 1 == source_nibbles.shape[0]):
            print(
                f"SQ3 hybrid4: solved {row + 1}/{source_nibbles.shape[0]} rows",
                file=sys.stderr,
                flush=True,
            )
    encoding = _materialize_encoding(
        source_nibbles,
        source_scales,
        solutions,
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
            f"hybrid SQ3 physical decode SSE mismatch: {encoding.searched_sse} != {measured_sse}"
        )
    return (
        reconstruction,
        encoding,
        _metadata(
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

    rate = hybrid_mxfp4_sq3_rate(rows, columns)
    nvq3_budget = NVQ3_D4.payload_nbytes(rows, columns)
    if rate.payload_nbytes > nvq3_budget:
        raise RuntimeError("hybrid MXFP4-SQ3 exceeds NVQ3 payload")
    print(
        f"searching SQ3 hybrid4: {rate.payload_nbytes} bytes / {rate.payload_bpw:.9f} BPW...",
        file=sys.stderr,
        flush=True,
    )
    search_started = time.perf_counter()
    reconstruction, _, metadata = quantize_hybrid_mxfp4_sq3(
        packed,
        source_scale_raw,
        exponent_radius=args.exponent_radius,
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
        "format": "hybrid-four-state-native-MXFP4-SQ3-fixed32",
        **rate.__dict__,
        "budget_slack_vs_nvq3_nbytes": nvq3_budget - rate.payload_nbytes,
        "seconds": search_seconds,
        **candidate_metrics,
        "sse_delta_percent": comparison,
        **metadata,
    }
    result: dict[str, Any] = {
        "schema": 1,
        "experiment": "dsv4f-hybrid-native-mxfp4-three-bit-scalar-transcoding",
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
        "hybrid_mxfp4_sq3": candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
