#!/usr/bin/env python3
"""Search a four-state native-MXFP4 SQ format with one in-band tag bit.

Each native block stores one explicit selector bit.  The second row-state bit
is the low bit of the XOR reduction of its 32 two-bit scalar symbols.  This
uses four native ``(E8M0 scale, E2M1 palette)`` states per row while remaining
below the equal-shape NVQ2 payload.
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
from bench.dsv4f_mxfp4_xor_sq import (
    XorRowSolution,
    _candidate_xor_errors,
    _pack_palette_codes,
    _pack_two_bit_symbols,
    _quantize_blocks_for_tag,
    _refine_constrained_states,
    _unconstrained_state_sets,
    _unpack_palette_codes,
    _unpack_two_bit_symbols,
)
from mfq.formats.nvq import NVQ2_E8
from mfq.quantize.mxfp import decode_mxfp4
from mfq.quantize.v4f_source import V4FCheckpoint


@dataclass(frozen=True)
class HybridSqRate:
    bits_per_symbol: int
    native_block_size: int
    explicit_block_selector_bits: int
    implicit_tag_bits: int
    palette_id_bits: int
    states_per_row: int
    symbol_nbytes: int
    block_selector_nbytes: int
    state_scale_nbytes: int
    state_palette_nbytes: int
    payload_nbytes: int
    payload_bpw: float


@dataclass(frozen=True)
class HybridSqEncoding:
    state_scale_raw: np.ndarray
    state_palette_codes: np.ndarray
    packed_state_palettes: np.ndarray
    packed_symbols: np.ndarray
    packed_block_selectors: np.ndarray
    block_tags: np.ndarray
    packed_mxfp4: np.ndarray
    native_scale_raw: np.ndarray
    searched_sse: float


def hybrid_mxfp4_sq_rate(rows: int, columns: int) -> HybridSqRate:
    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("hybrid MXFP4-SQ requires positive block-32 matrices")
    weights = rows * columns
    blocks = weights // 32
    symbol_nbytes = (weights * 2 + 7) // 8
    block_selector_nbytes = (blocks + 7) // 8
    state_scale_nbytes = rows * 4
    state_palette_nbytes = (rows * 4 * 4 + 7) // 8
    payload_nbytes = (
        symbol_nbytes + block_selector_nbytes + state_scale_nbytes + state_palette_nbytes
    )
    return HybridSqRate(
        bits_per_symbol=2,
        native_block_size=32,
        explicit_block_selector_bits=1,
        implicit_tag_bits=1,
        palette_id_bits=4,
        states_per_row=4,
        symbol_nbytes=symbol_nbytes,
        block_selector_nbytes=block_selector_nbytes,
        state_scale_nbytes=state_scale_nbytes,
        state_palette_nbytes=state_palette_nbytes,
        payload_nbytes=payload_nbytes,
        payload_bpw=8.0 * payload_nbytes / weights,
    )


def _hybrid_labeled_errors(full_xor_errors: np.ndarray) -> np.ndarray:
    parity_zero = np.minimum(full_xor_errors[0], full_xor_errors[2])
    parity_one = np.minimum(full_xor_errors[1], full_xor_errors[3])
    return np.stack((parity_zero, parity_one, parity_zero, parity_one))


def _solve_hybrid_row(
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
    full_xor_errors = _candidate_xor_errors(
        source_nibbles_row,
        source_scale_row,
        state_scales,
        state_palette_ids,
    )
    errors = _hybrid_labeled_errors(full_xor_errors)
    state_sets = _unconstrained_state_sets(
        source_nibbles_row,
        source_scale_row,
        errors,
        maximum_steps=maximum_steps,
    )
    candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray, int]] = []
    for state_set in state_sets:
        ordered_candidates: dict[
            tuple[tuple[int, int], tuple[int, int]],
            tuple[float, tuple[int, int, int, int]],
        ] = {}
        for permutation in set(itertools.permutations(state_set)):
            error = float(
                np.stack([errors[tag, permutation[tag]] for tag in range(4)]).min(axis=0).sum()
            )
            parity_key = (
                tuple(sorted((permutation[0], permutation[2]))),
                tuple(sorted((permutation[1], permutation[3]))),
            )
            previous = ordered_candidates.get(parity_key)
            if previous is None or error < previous[0]:
                ordered_candidates[parity_key] = (error, permutation)
        for _, ordered in sorted(ordered_candidates.values()):
            candidates.append(
                _refine_constrained_states(
                    errors,
                    ordered,
                    maximum_steps=maximum_steps,
                )
            )
    for parity_order in set(itertools.permutations((0, 0, 1, 1))):
        tag_for_occurrence = {0: (0, 2), 1: (1, 3)}
        occurrence = [0, 0]
        initial_states = np.zeros(4, dtype=np.int64)
        covered = np.full(errors.shape[2], np.inf, dtype=np.float64)
        selected_by_parity: list[list[int]] = [[], []]
        for parity in parity_order:
            tag = tag_for_occurrence[parity][occurrence[parity]]
            occurrence[parity] += 1
            totals = np.minimum(errors[tag], covered[None, :]).sum(axis=1)
            if selected_by_parity[parity]:
                totals[np.asarray(selected_by_parity[parity])] = np.inf
            selected_state = int(totals.argmin())
            initial_states[tag] = selected_state
            selected_by_parity[parity].append(selected_state)
            covered = np.minimum(covered, errors[tag, selected_state])
        candidates.append(
            _refine_constrained_states(
                errors,
                tuple(int(item) for item in initial_states),
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


def _pack_block_selectors(selectors: np.ndarray) -> np.ndarray:
    values = np.asarray(selectors, dtype=np.uint8)
    return np.packbits(values.reshape(-1), bitorder="little")


def _unpack_block_selectors(
    packed: np.ndarray,
    *,
    rows: int,
    blocks: int,
) -> np.ndarray:
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), bitorder="little", count=rows * blocks
    ).reshape(rows, blocks)


def decode_hybrid_mxfp4_sq(
    packed_symbols: np.ndarray,
    packed_block_selectors: np.ndarray,
    state_scale_raw: np.ndarray,
    packed_state_palettes: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    symbols = _unpack_two_bit_symbols(packed_symbols)
    rows, columns = symbols.shape
    if columns % 32 or tuple(state_scale_raw.shape) != (rows, 4):
        raise ValueError("hybrid MXFP4-SQ payload geometry mismatch")
    blocks = columns // 32
    block_symbols = symbols.reshape(rows, blocks, 32)
    explicit_high_bit = _unpack_block_selectors(
        packed_block_selectors,
        rows=rows,
        blocks=blocks,
    )
    implicit_low_bit = np.bitwise_xor.reduce(block_symbols, axis=2) & 1
    block_tags = (explicit_high_bit << 1) | implicit_low_bit
    state_palette_ids = FIXED16_PALETTE_IDS[_unpack_palette_codes(packed_state_palettes)]
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


def _materialize_hybrid_encoding(
    source_nibbles: np.ndarray,
    source_scale_raw: np.ndarray,
    solutions: list[XorRowSolution],
) -> HybridSqEncoding:
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
        for state_tag in range(4):
            selected = block_tags[row] == state_tag
            if not bool(selected.any()):
                continue
            levels = PALETTE_VALUES[state_palette_ids[row, state_tag]] * math.ldexp(
                1.0, int(state_scale_raw[row, state_tag]) - 127
            )
            low_tag = state_tag & 1
            symbols_low, error_low = _quantize_blocks_for_tag(target[selected], levels, low_tag)
            symbols_high, error_high = _quantize_blocks_for_tag(
                target[selected], levels, low_tag + 2
            )
            choose_high = error_high < error_low
            symbols_low[choose_high] = symbols_high[choose_high]
            selected_error = np.where(choose_high, error_high, error_low)
            symbols[row, selected] = symbols_low
            measured_search_sse += float(selected_error.sum())
    packed_symbols = _pack_two_bit_symbols(symbols.reshape(rows, blocks * block_size))
    packed_block_selectors = _pack_block_selectors(block_tags >> 1)
    packed_state_palettes = _pack_palette_codes(state_palette_codes)
    reconstruction, packed_mxfp4, native_scale_raw, decoded_tags = decode_hybrid_mxfp4_sq(
        packed_symbols,
        packed_block_selectors,
        state_scale_raw,
        packed_state_palettes,
        device="cpu",
    )
    if not np.array_equal(decoded_tags, block_tags):
        raise RuntimeError("hybrid payload did not recover searched block tags")
    searched_sse = float(sum(solution.error_sse for solution in solutions))
    if not math.isclose(
        searched_sse,
        measured_search_sse,
        rel_tol=2e-9,
        abs_tol=1e-10,
    ):
        raise RuntimeError(
            "hybrid SQ search/materialization SSE mismatch: "
            f"{searched_sse} != {measured_search_sse}"
        )
    return HybridSqEncoding(
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


def _hybrid_metadata(
    encoding: HybridSqEncoding,
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
        "state_tag": "explicit-high-bit-plus-symbol-xor-low-bit",
        "explicit_block_selector_bits": 1,
        "implicit_symbol_tag_bits": 1,
        "scale_dtype": "E8M0",
        "scale_exponent_radius": exponent_radius,
        "optimizer": "deterministic-multistart-hard-em-with-exact-parity-costs",
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
def quantize_hybrid_mxfp4_sq(
    packed: np.ndarray,
    source_scale_raw: np.ndarray,
    *,
    exponent_radius: int = 0,
    maximum_refinement_steps: int = 10,
    progress: bool = False,
) -> tuple[torch.Tensor, HybridSqEncoding, dict[str, Any]]:
    if exponent_radius < 0 or maximum_refinement_steps <= 0:
        raise ValueError("invalid hybrid SQ search control")
    source_nibbles, _ = _unpack_source(packed, source_scale_raw)
    source_scales = np.asarray(source_scale_raw, dtype=np.uint8)
    solutions: list[XorRowSolution] = []
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
                f"hybrid4: solved {row + 1}/{source_nibbles.shape[0]} rows",
                file=sys.stderr,
                flush=True,
            )
    encoding = _materialize_hybrid_encoding(
        source_nibbles,
        source_scales,
        solutions,
    )
    reconstruction, _, _, _ = decode_hybrid_mxfp4_sq(
        encoding.packed_symbols,
        encoding.packed_block_selectors,
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
            f"hybrid SQ physical decode SSE mismatch: {encoding.searched_sse} != {measured_sse}"
        )
    return (
        reconstruction,
        encoding,
        _hybrid_metadata(
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

    rate = hybrid_mxfp4_sq_rate(rows, columns)
    if rate.payload_nbytes > baseline_budget:
        raise RuntimeError("hybrid SQ exceeds NVQ2 payload")
    print(
        f"searching hybrid4: {rate.payload_nbytes} bytes / {rate.payload_bpw:.9f} BPW...",
        file=sys.stderr,
        flush=True,
    )
    search_start = time.perf_counter()
    reconstruction, _, metadata = quantize_hybrid_mxfp4_sq(
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
        "format": "hybrid-tagged-four-state-native-MXFP4-SQ-fixed16",
        **rate.__dict__,
        "budget_slack_nbytes": baseline_budget - rate.payload_nbytes,
        "seconds": search_seconds,
        **metrics,
        "sse_delta_vs_nvq2_percent": 100.0
        * (float(metrics["error_sse"]) / float(baseline_metrics["error_sse"]) - 1.0),
        **metadata,
    }
    print(
        f"hybrid4: SSE={metrics['error_sse']:.9f}, SNR={metrics['snr_db']:.6f} dB",
        file=sys.stderr,
        flush=True,
    )
    result: dict[str, Any] = {
        "schema": 1,
        "experiment": "dsv4f-hybrid-tagged-native-mxfp4-scalar-transcoding",
        "created_unix": started,
        "workspace": _git_identity(script_root),
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "baseline_device": str(torch.device(args.device)),
            "hybrid_search_device": "cpu/numpy",
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
        "hybrid_mxfp4_sq": candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
