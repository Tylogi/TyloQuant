#!/usr/bin/env python3
"""Benchmark a native-output three-bit MXFP4 scalar transcoder.

The source tensor is already block-32 MXFP4.  Each output row stores two
states.  A state is one E8M0 scale and one eight-level subset of the 15
distinct E2M1 values.  One bit per native block selects the state, and each
weight stores a three-bit scalar symbol.  Decoding first expands the payload
to ordinary E2M1 nibbles plus E8M0 block scales and then uses the production
MXFP4 decoder.

The full palette catalog has C(15, 8) = 6,435 entries and is implicit in the
format.  Two 13-bit palette IDs and two E8M0 bytes are stored per row.  For a
4096 by 4096 matrix the exact payload is 6,378,496 bytes (3.04150390625 BPW),
which is 9,216 bytes smaller than MFQ NVQ3-D4-256.
"""

from __future__ import annotations

import argparse
import ctypes
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
    E2M1_NIBBLES,
    E2M1_VALUES,
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
from mfq.formats.nvq import NVQ3_D4, NVQ3_D4_512
from mfq.quantize.mxfp import decode_mxfp4
from mfq.quantize.v4f_source import V4FCheckpoint

SQ3_PALETTE_LEVEL_INDICES = np.asarray(
    list(itertools.combinations(range(len(E2M1_VALUES)), 8)),
    dtype=np.int16,
)
SQ3_PALETTE_VALUES = E2M1_VALUES[SQ3_PALETTE_LEVEL_INDICES]
SQ3_PALETTE_NIBBLES = E2M1_NIBBLES[SQ3_PALETTE_LEVEL_INDICES]
SQ3_PALETTE_ID_BITS = 13


@dataclass(frozen=True)
class Sq3Rate:
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
class Sq3RowSolution:
    error_sse: float
    state_scale_raw: tuple[int, int]
    state_palette_ids: tuple[int, int]
    block_selectors: np.ndarray
    refinement_steps: int


@dataclass(frozen=True)
class Sq3Encoding:
    state_scale_raw: np.ndarray
    state_palette_ids: np.ndarray
    packed_state_palettes: np.ndarray
    packed_symbols: np.ndarray
    packed_block_selectors: np.ndarray
    block_selectors: np.ndarray
    packed_mxfp4: np.ndarray
    native_scale_raw: np.ndarray
    searched_sse: float


@dataclass(frozen=True)
class LlamaCppFormat:
    ggml_type: int
    label: str
    dequantize_symbol: str


LLAMA_CPP_FORMATS = (
    LlamaCppFormat(18, "IQ3_XXS", "dequantize_row_iq3_xxs"),
    LlamaCppFormat(11, "Q3_K", "dequantize_row_q3_K"),
    LlamaCppFormat(21, "IQ3_S", "dequantize_row_iq3_s"),
)


def mxfp4_sq3_rate(rows: int, columns: int) -> Sq3Rate:
    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("MXFP4-SQ3 requires positive block-32 matrices")
    weights = rows * columns
    blocks = weights // 32
    symbol_nbytes = (weights * 3 + 7) // 8
    block_selector_nbytes = (blocks + 7) // 8
    state_scale_nbytes = rows * 2
    state_palette_nbytes = (rows * 2 * SQ3_PALETTE_ID_BITS + 7) // 8
    payload_nbytes = (
        symbol_nbytes + block_selector_nbytes + state_scale_nbytes + state_palette_nbytes
    )
    return Sq3Rate(
        bits_per_symbol=3,
        native_block_size=32,
        block_selector_bits=1,
        palette_id_bits=SQ3_PALETTE_ID_BITS,
        states_per_row=2,
        symbol_nbytes=symbol_nbytes,
        block_selector_nbytes=block_selector_nbytes,
        state_scale_nbytes=state_scale_nbytes,
        state_palette_nbytes=state_palette_nbytes,
        payload_nbytes=payload_nbytes,
        payload_bpw=8.0 * payload_nbytes / weights,
    )


def _pack_fixed_width(values: np.ndarray, bits: int) -> np.ndarray:
    items = np.asarray(values)
    if bits <= 0 or bits > 16 or items.size == 0:
        raise ValueError("invalid fixed-width payload")
    if int(items.min()) < 0 or int(items.max()) >= 1 << bits:
        raise ValueError("value does not fit fixed-width payload")
    shifts = np.arange(bits, dtype=np.uint16)
    bit_values = ((items.reshape(-1, 1).astype(np.uint16) >> shifts) & 1).astype(np.uint8)
    return np.packbits(bit_values.reshape(-1), bitorder="little")


def _unpack_fixed_width(
    packed: np.ndarray,
    bits: int,
    *,
    count: int,
) -> np.ndarray:
    if bits <= 0 or bits > 16 or count <= 0:
        raise ValueError("invalid fixed-width payload geometry")
    raw_bits = np.unpackbits(
        np.asarray(packed, dtype=np.uint8),
        bitorder="little",
        count=count * bits,
    ).reshape(count, bits)
    shifts = (np.uint16(1) << np.arange(bits, dtype=np.uint16))[None, :]
    return (raw_bits.astype(np.uint16) * shifts).sum(axis=1, dtype=np.uint16)


def _pack_three_bit_symbols(symbols: np.ndarray) -> np.ndarray:
    values = np.asarray(symbols, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] % 32 or int(values.max()) >= 8:
        raise ValueError("three-bit symbols must be a rows-by-block32 matrix")
    packed = _pack_fixed_width(values, 3)
    return packed.reshape(values.shape[0], values.shape[1] * 3 // 8)


def _unpack_three_bit_symbols(packed: np.ndarray) -> np.ndarray:
    values = np.asarray(packed, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] % 12:
        raise ValueError("packed three-bit symbols have invalid row geometry")
    rows = values.shape[0]
    columns = values.shape[1] * 8 // 3
    return (
        _unpack_fixed_width(values, 3, count=rows * columns).astype(np.uint8).reshape(rows, columns)
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
        np.asarray(packed, dtype=np.uint8),
        bitorder="little",
        count=rows * blocks,
    ).reshape(rows, blocks)


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
            reconstruction_values = SQ3_PALETTE_VALUES * math.ldexp(1.0, reconstruction_raw - 127)
            result[(source_raw, reconstruction_raw)] = np.square(
                source_values[:, None, None] - reconstruction_values[None, :, :]
            ).min(axis=2)
    return result


def _state_block_errors(
    histogram_row: np.ndarray,
    source_scale_row: np.ndarray,
    reconstruction_scale_raw: int,
    palette_id: int,
    distortions: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    result = np.empty(histogram_row.shape[0], dtype=np.float64)
    for source_raw in np.unique(source_scale_row):
        selected = source_scale_row == source_raw
        result[selected] = (
            histogram_row[selected]
            @ distortions[(int(source_raw), reconstruction_scale_raw)][:, palette_id]
        )
    return result


def _best_state_for_partition(
    histogram_row: np.ndarray,
    source_scale_row: np.ndarray,
    selected_blocks: np.ndarray,
    candidate_exponents: list[int],
    distortions: dict[tuple[int, int], np.ndarray],
) -> tuple[int, int] | None:
    if not bool(selected_blocks.any()):
        return None
    best_error = float("inf")
    best_state: tuple[int, int] | None = None
    for reconstruction_raw in candidate_exponents:
        palette_error = np.zeros(len(SQ3_PALETTE_VALUES), dtype=np.float64)
        for source_raw in np.unique(source_scale_row):
            selected = selected_blocks & (source_scale_row == source_raw)
            if bool(selected.any()):
                counts = histogram_row[selected].sum(axis=0, dtype=np.float64)
                palette_error += counts @ distortions[(int(source_raw), reconstruction_raw)]
        palette_id = int(palette_error.argmin())
        error = float(palette_error[palette_id])
        if error < best_error:
            best_error = error
            best_state = (reconstruction_raw, palette_id)
    return best_state


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
) -> Sq3RowSolution | None:
    selectors = _balanced_partition(initial_selectors)
    best: Sq3RowSolution | None = None
    for step in range(1, maximum_steps + 1):
        state0 = _best_state_for_partition(
            histogram_row,
            source_scale_row,
            ~selectors,
            candidate_exponents,
            distortions,
        )
        state1 = _best_state_for_partition(
            histogram_row,
            source_scale_row,
            selectors,
            candidate_exponents,
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
        candidate = Sq3RowSolution(
            error_sse=error,
            state_scale_raw=(state0[0], state1[0]),
            state_palette_ids=(state0[1], state1[1]),
            block_selectors=updated.copy(),
            refinement_steps=step,
        )
        if best is None or candidate.error_sse < best.error_sse:
            best = candidate
        if np.array_equal(updated, selectors):
            break
        selectors = _balanced_partition(updated)
    return best


def _solve_full_catalog_row(
    histogram_row: np.ndarray,
    source_scale_row: np.ndarray,
    *,
    exponent_radius: int,
    maximum_steps: int,
    distortions: dict[tuple[int, int], np.ndarray],
) -> Sq3RowSolution:
    candidate_exponents = _row_candidate_exponents(
        source_scale_row,
        exponent_radius,
    )
    signed_sum = histogram_row @ E2M1_VALUES
    normalized_energy = histogram_row @ np.square(E2M1_VALUES)
    starts = [
        source_scale_row > np.median(source_scale_row),
        signed_sum > np.median(signed_sum),
        normalized_energy > np.median(normalized_energy),
        histogram_row[:, 7] > np.median(histogram_row[:, 7]),
        np.arange(histogram_row.shape[0]) % 2 == 0,
        np.arange(histogram_row.shape[0]) % 4 < 2,
    ]
    candidates: list[Sq3RowSolution] = []
    seen: set[bytes] = set()
    for start in starts:
        balanced = _balanced_partition(start)
        key = np.packbits(balanced).tobytes()
        inverse_key = np.packbits(~balanced).tobytes()
        if key in seen or inverse_key in seen:
            continue
        seen.add(key)
        candidate = _refine_two_state_row(
            histogram_row,
            source_scale_row,
            balanced,
            candidate_exponents,
            distortions,
            maximum_steps=maximum_steps,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError("SQ3 row search produced no candidate")
    return min(candidates, key=lambda item: item.error_sse)


def decode_mxfp4_sq3(
    packed_symbols: np.ndarray,
    packed_block_selectors: np.ndarray,
    state_scale_raw: np.ndarray,
    packed_state_palettes: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    symbols = _unpack_three_bit_symbols(packed_symbols)
    rows, columns = symbols.shape
    if columns % 32 or tuple(state_scale_raw.shape) != (rows, 2):
        raise ValueError("MXFP4-SQ3 payload geometry mismatch")
    blocks = columns // 32
    block_selectors = _unpack_block_selectors(
        packed_block_selectors,
        rows=rows,
        blocks=blocks,
    )
    state_palette_ids = _unpack_fixed_width(
        packed_state_palettes,
        SQ3_PALETTE_ID_BITS,
        count=rows * 2,
    ).reshape(rows, 2)
    if int(state_palette_ids.max()) >= len(SQ3_PALETTE_VALUES):
        raise ValueError("MXFP4-SQ3 palette ID is outside the implicit catalog")
    local_rows = np.arange(rows)[:, None]
    selected_scale = np.asarray(state_scale_raw, dtype=np.uint8)[local_rows, block_selectors]
    selected_palette = state_palette_ids[local_rows, block_selectors]
    block_symbols = symbols.reshape(rows, blocks, 32)
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
    return reconstruction, packed_mxfp4, selected_scale, block_selectors


def _materialize_encoding(
    source_nibbles: np.ndarray,
    source_scale_raw: np.ndarray,
    solutions: list[Sq3RowSolution],
    *,
    row_chunk: int,
) -> Sq3Encoding:
    rows, blocks, block_size = source_nibbles.shape
    state_scale_raw = np.asarray(
        [solution.state_scale_raw for solution in solutions], dtype=np.uint8
    )
    state_palette_ids = np.asarray(
        [solution.state_palette_ids for solution in solutions], dtype=np.uint16
    )
    block_selectors = np.stack([solution.block_selectors for solution in solutions]).astype(
        np.uint8, copy=False
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
        palette_values = SQ3_PALETTE_VALUES[selected_palette]
        palette_nibbles = SQ3_PALETTE_NIBBLES[selected_palette]
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

    packed_symbols = _pack_three_bit_symbols(symbols.reshape(rows, blocks * block_size))
    packed_block_selectors = _pack_block_selectors(block_selectors)
    packed_state_palettes = _pack_fixed_width(
        state_palette_ids,
        SQ3_PALETTE_ID_BITS,
    )
    packed_blocks = output_nibbles[:, :, 0::2] | (output_nibbles[:, :, 1::2] << 4)
    packed_mxfp4 = packed_blocks.reshape(rows, blocks * 16)
    encoding = Sq3Encoding(
        state_scale_raw=state_scale_raw,
        state_palette_ids=state_palette_ids,
        packed_state_palettes=packed_state_palettes,
        packed_symbols=packed_symbols,
        packed_block_selectors=packed_block_selectors,
        block_selectors=block_selectors,
        packed_mxfp4=packed_mxfp4,
        native_scale_raw=native_scale_raw,
        searched_sse=float(sum(solution.error_sse for solution in solutions)),
    )
    decoded, decoded_mxfp4, decoded_scale, decoded_selectors = decode_mxfp4_sq3(
        encoding.packed_symbols,
        encoding.packed_block_selectors,
        encoding.state_scale_raw,
        encoding.packed_state_palettes,
        device="cpu",
    )
    if not np.array_equal(decoded_mxfp4, packed_mxfp4):
        raise RuntimeError("SQ3 physical payload changed reconstructed nibbles")
    if not np.array_equal(decoded_scale, native_scale_raw):
        raise RuntimeError("SQ3 physical payload changed reconstructed scales")
    if not np.array_equal(decoded_selectors, block_selectors):
        raise RuntimeError("SQ3 physical payload changed block selectors")
    del decoded
    return encoding


def _encoding_metadata(
    encoding: Sq3Encoding,
    source_scale_raw: np.ndarray,
    solutions: list[Sq3RowSolution],
    *,
    exponent_radius: int,
) -> dict[str, Any]:
    palette_counts = Counter(int(item) for item in encoding.state_palette_ids.flat)
    return {
        "palette_mode": "implicit-full-catalog",
        "palette_catalog_size": len(SQ3_PALETTE_VALUES),
        "palette_id_bits": SQ3_PALETTE_ID_BITS,
        "state_layout": "two-row-states:(E8M0-scale,eight-level-E2M1-palette)",
        "block_selector_bits": 1,
        "scale_dtype": "E8M0",
        "scale_exponent_radius": exponent_radius,
        "optimizer": "deterministic-multistart-hard-em",
        "optimizer_is_global_within_catalog": False,
        "maximum_refinement_steps_used": max(
            (solution.refinement_steps for solution in solutions), default=0
        ),
        "mean_refinement_steps": float(
            np.mean([solution.refinement_steps for solution in solutions])
        ),
        "state_scale_byte_min": int(encoding.state_scale_raw.min()),
        "state_scale_byte_max": int(encoding.state_scale_raw.max()),
        "same_scale_two_state_row_fraction": float(
            np.mean(encoding.state_scale_raw[:, 0] == encoding.state_scale_raw[:, 1])
        ),
        "high_state_selector_fraction": float(encoding.block_selectors.mean()),
        "effective_scale_matches_source_fraction": float(
            np.mean(encoding.native_scale_raw == source_scale_raw)
        ),
        "unique_palette_ids": len(palette_counts),
        "top_state_palette_ids": [
            {
                "palette_id": palette_id,
                "count": count,
                "values": SQ3_PALETTE_VALUES[palette_id].tolist(),
                "nibbles": SQ3_PALETTE_NIBBLES[palette_id].tolist(),
            }
            for palette_id, count in palette_counts.most_common(64)
        ],
        "learned_vector_codebook": False,
        "stored_fp16_scale_or_centroid": False,
        "physical_storage_roundtrip_verified": True,
        "final_values_are_native_block32_mxfp4": True,
    }


@torch.inference_mode()
def quantize_mxfp4_sq3(
    packed: np.ndarray,
    source_scale_raw: np.ndarray,
    *,
    exponent_radius: int = 2,
    maximum_refinement_steps: int = 12,
    row_chunk: int = 64,
    progress: bool = False,
) -> tuple[torch.Tensor, Sq3Encoding, dict[str, Any]]:
    if exponent_radius < 0 or maximum_refinement_steps <= 0 or row_chunk <= 0:
        raise ValueError("invalid SQ3 search control")
    source_nibbles, histograms = _unpack_source(packed, source_scale_raw)
    source_scales = np.asarray(source_scale_raw, dtype=np.uint8)
    distortions = _distortion_tables(
        source_scales,
        exponent_radius=exponent_radius,
    )
    solutions: list[Sq3RowSolution] = []
    for row in range(histograms.shape[0]):
        solutions.append(
            _solve_full_catalog_row(
                histograms[row],
                source_scales[row],
                exponent_radius=exponent_radius,
                maximum_steps=maximum_refinement_steps,
                distortions=distortions,
            )
        )
        if progress and ((row + 1) % 256 == 0 or row + 1 == histograms.shape[0]):
            print(
                f"SQ3 full: solved {row + 1}/{histograms.shape[0]} rows",
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
            f"SQ3 search/materialization SSE mismatch: {encoding.searched_sse} != {measured_sse}"
        )
    return (
        reconstruction,
        encoding,
        _encoding_metadata(
            encoding,
            source_scales,
            solutions,
            exponent_radius=exponent_radius,
        ),
    )


def _llama_cpp_quantize(
    source: torch.Tensor,
    library: Path,
    spec: LlamaCppFormat,
) -> tuple[torch.Tensor, int, float, float]:
    value = np.ascontiguousarray(source.cpu().numpy(), dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("llama.cpp comparison requires a matrix")
    lib = ctypes.CDLL(str(library))
    lib.ggml_type_size.argtypes = [ctypes.c_int]
    lib.ggml_type_size.restype = ctypes.c_size_t
    lib.ggml_blck_size.argtypes = [ctypes.c_int]
    lib.ggml_blck_size.restype = ctypes.c_int64
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    block_size = int(lib.ggml_blck_size(spec.ggml_type))
    type_size = int(lib.ggml_type_size(spec.ggml_type))
    if value.shape[1] % block_size:
        raise ValueError(f"{spec.label} requires rows divisible by {block_size}")
    payload_nbytes = value.size // block_size * type_size
    payload = np.empty(payload_nbytes, dtype=np.uint8)
    started = time.perf_counter()
    written = int(
        lib.ggml_quantize_chunk(
            spec.ggml_type,
            value.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            payload.ctypes.data,
            0,
            value.shape[0],
            value.shape[1],
            None,
        )
    )
    quantize_seconds = time.perf_counter() - started
    if written != payload_nbytes:
        raise RuntimeError(
            f"llama.cpp {spec.label} wrote {written} bytes, expected {payload_nbytes}"
        )
    reconstructed = np.empty_like(value)
    dequantize = getattr(lib, spec.dequantize_symbol)
    dequantize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
    ]
    dequantize.restype = None
    started = time.perf_counter()
    dequantize(
        payload.ctypes.data,
        reconstructed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        value.size,
    )
    dequantize_seconds = time.perf_counter() - started
    lib.ggml_quantize_free()
    return (
        torch.from_numpy(reconstructed),
        payload_nbytes,
        quantize_seconds,
        dequantize_seconds,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--exponent-radius", type=int, default=2)
    parser.add_argument("--maximum-refinement-steps", type=int, default=12)
    parser.add_argument("--row-chunk", type=int, default=64)
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

    rate = mxfp4_sq3_rate(rows, columns)
    nvq3_budget = NVQ3_D4.payload_nbytes(rows, columns)
    if rate.payload_nbytes > nvq3_budget:
        raise RuntimeError("MXFP4-SQ3 exceeds NVQ3 payload")
    print(
        f"searching SQ3 full catalog: {rate.payload_nbytes} bytes / {rate.payload_bpw:.9f} BPW...",
        file=sys.stderr,
        flush=True,
    )
    search_started = time.perf_counter()
    reconstruction, _, metadata = quantize_mxfp4_sq3(
        packed,
        source_scale_raw,
        exponent_radius=args.exponent_radius,
        maximum_refinement_steps=args.maximum_refinement_steps,
        row_chunk=args.row_chunk,
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
        "format": "two-state-native-MXFP4-SQ3-full6435",
        **rate.__dict__,
        "budget_slack_vs_nvq3_nbytes": nvq3_budget - rate.payload_nbytes,
        "seconds": search_seconds,
        **candidate_metrics,
        "sse_delta_percent": comparison,
        **metadata,
    }
    result: dict[str, Any] = {
        "schema": 1,
        "experiment": "dsv4f-native-mxfp4-three-bit-scalar-transcoding",
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
        "mxfp4_sq3": candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
