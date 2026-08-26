#!/usr/bin/env python3
"""Compare strict MXFP4 scalar quantization with production NVQ2.

This is ordinary scalar quantization: every weight carries a fixed two-bit
symbol and every aligned group carries one E8M0 scale.  The symbol alphabet is
an immutable subset of E2M1, so there is no learned table and the reconstructed
matrix can be materialized byte-for-byte as native block-32 MXFP4.
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
)
from mfq.formats.nvq import NVQ2_E8
from mfq.quantize.mxfp import decode_mxfp4
from mfq.quantize.v4f_source import V4FCheckpoint


@dataclass(frozen=True)
class SqAlphabet:
    label: str
    low_magnitude: float
    high_magnitude: float
    # Two-bit symbol -> native E2M1 nibble.  Duplicate zero is allowed for the
    # ternary signed-zero layout.
    e2m1_nibbles: tuple[int, int, int, int]


@dataclass(frozen=True)
class SqRate:
    bits_per_symbol: int
    scale_group_size: int
    symbol_nbytes: int
    scale_nbytes: int
    payload_nbytes: int
    payload_bpw: float


@dataclass(frozen=True)
class SqRowScaleRate:
    bits_per_symbol: int
    native_block_size: int
    scale_selector_bits: int
    row_delta_bits: int
    symbol_nbytes: int
    row_base_nbytes: int
    row_delta_nbytes: int
    scale_selector_nbytes: int
    payload_nbytes: int
    payload_bpw: float


ALPHABETS: dict[str, SqAlphabet] = {
    "ternary": SqAlphabet("ternary", 0.0, 1.0, (0x0, 0x2, 0xA, 0x8)),
    "sym-0p5-1": SqAlphabet("sym-0p5-1", 0.5, 1.0, (0xA, 0x9, 0x1, 0x2)),
    "sym-0p5-1p5": SqAlphabet(
        "sym-0p5-1p5", 0.5, 1.5, (0xB, 0x9, 0x1, 0x3)
    ),
    "sym-0p5-2": SqAlphabet("sym-0p5-2", 0.5, 2.0, (0xC, 0x9, 0x1, 0x4)),
    "sym-0p5-3": SqAlphabet("sym-0p5-3", 0.5, 3.0, (0xD, 0x9, 0x1, 0x5)),
    "sym-0p5-4": SqAlphabet("sym-0p5-4", 0.5, 4.0, (0xE, 0x9, 0x1, 0x6)),
    "sym-0p5-6": SqAlphabet("sym-0p5-6", 0.5, 6.0, (0xF, 0x9, 0x1, 0x7)),
    "sym-1-1p5": SqAlphabet("sym-1-1p5", 1.0, 1.5, (0xB, 0xA, 0x2, 0x3)),
    "sym-1-2": SqAlphabet("sym-1-2", 1.0, 2.0, (0xC, 0xA, 0x2, 0x4)),
    "sym-1-3": SqAlphabet("sym-1-3", 1.0, 3.0, (0xD, 0xA, 0x2, 0x5)),
    "sym-1-4": SqAlphabet("sym-1-4", 1.0, 4.0, (0xE, 0xA, 0x2, 0x6)),
    "sym-1-6": SqAlphabet("sym-1-6", 1.0, 6.0, (0xF, 0xA, 0x2, 0x7)),
    "sym-1p5-2": SqAlphabet("sym-1p5-2", 1.5, 2.0, (0xC, 0xB, 0x3, 0x4)),
    "sym-1p5-3": SqAlphabet("sym-1p5-3", 1.5, 3.0, (0xD, 0xB, 0x3, 0x5)),
    "sym-1p5-4": SqAlphabet("sym-1p5-4", 1.5, 4.0, (0xE, 0xB, 0x3, 0x6)),
    "sym-1p5-6": SqAlphabet("sym-1p5-6", 1.5, 6.0, (0xF, 0xB, 0x3, 0x7)),
    "sym-2-3": SqAlphabet("sym-2-3", 2.0, 3.0, (0xD, 0xC, 0x4, 0x5)),
    "sym-2-4": SqAlphabet("sym-2-4", 2.0, 4.0, (0xE, 0xC, 0x4, 0x6)),
    "sym-2-6": SqAlphabet("sym-2-6", 2.0, 6.0, (0xF, 0xC, 0x4, 0x7)),
    "sym-3-4": SqAlphabet("sym-3-4", 3.0, 4.0, (0xE, 0xD, 0x5, 0x6)),
    "sym-3-6": SqAlphabet("sym-3-6", 3.0, 6.0, (0xF, 0xD, 0x5, 0x7)),
    "sym-4-6": SqAlphabet("sym-4-6", 4.0, 6.0, (0xF, 0xE, 0x6, 0x7)),
}


def strict_mxfp4_sq_rate(
    rows: int,
    columns: int,
    *,
    bits_per_symbol: int = 2,
    scale_group_size: int = 192,
) -> SqRate:
    """Account for fixed scalar symbols plus one E8M0 byte per scale group."""

    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("strict MXFP4-SQ requires positive block-32 matrices")
    if bits_per_symbol != 2:
        raise ValueError("this strict MXFP4-SQ screen uses two-bit symbols")
    if scale_group_size <= 0 or scale_group_size % 32:
        raise ValueError("the E8M0 scale group must align to native block-32")
    symbol_nbytes = (rows * columns * bits_per_symbol + 7) // 8
    scale_nbytes = rows * math.ceil(columns / scale_group_size)
    payload_nbytes = symbol_nbytes + scale_nbytes
    return SqRate(
        bits_per_symbol=bits_per_symbol,
        scale_group_size=scale_group_size,
        symbol_nbytes=symbol_nbytes,
        scale_nbytes=scale_nbytes,
        payload_nbytes=payload_nbytes,
        payload_bpw=8.0 * payload_nbytes / (rows * columns),
    )


def strict_mxfp4_sq_row_scale_rate(
    rows: int,
    columns: int,
    *,
    bits_per_symbol: int = 2,
    native_block_size: int = 32,
    scale_selector_bits: int = 1,
    row_delta_bits: int = 2,
) -> SqRowScaleRate:
    """Account for row E8M0 bases and one scale-selector bit per MX block."""

    if rows <= 0 or columns <= 0 or columns % native_block_size:
        raise ValueError("row-scale MXFP4-SQ requires positive block-32 matrices")
    if bits_per_symbol != 2 or scale_selector_bits != 1 or row_delta_bits != 2:
        raise ValueError("the prototype uses 2-bit symbols, 1-bit scales, 2-bit deltas")
    weights = rows * columns
    blocks = rows * (columns // native_block_size)
    symbol_nbytes = (weights * bits_per_symbol + 7) // 8
    row_base_nbytes = rows
    row_delta_nbytes = (rows * row_delta_bits + 7) // 8
    scale_selector_nbytes = (blocks * scale_selector_bits + 7) // 8
    payload_nbytes = (
        symbol_nbytes
        + row_base_nbytes
        + row_delta_nbytes
        + scale_selector_nbytes
    )
    return SqRowScaleRate(
        bits_per_symbol=bits_per_symbol,
        native_block_size=native_block_size,
        scale_selector_bits=scale_selector_bits,
        row_delta_bits=row_delta_bits,
        symbol_nbytes=symbol_nbytes,
        row_base_nbytes=row_base_nbytes,
        row_delta_nbytes=row_delta_nbytes,
        scale_selector_nbytes=scale_selector_nbytes,
        payload_nbytes=payload_nbytes,
        payload_bpw=8.0 * payload_nbytes / weights,
    )


def _scale_value(exponent: torch.Tensor) -> torch.Tensor:
    return torch.ldexp(
        torch.ones_like(exponent, dtype=torch.float32), exponent
    )


@torch.inference_mode()
def quantize_strict_mxfp4_sq(
    value: torch.Tensor,
    alphabet: SqAlphabet,
    rate: SqRate,
    *,
    device: str | torch.device,
    group_chunk: int = 4096,
    exponent_radius: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Quantize with a fixed E2M1 alphabet and searched E8M0 group scales."""

    if value.ndim != 2 or value.shape[1] % 32:
        raise ValueError("strict MXFP4-SQ requires [rows,K] with K divisible by 32")
    if group_chunk <= 0 or exponent_radius <= 0:
        raise ValueError("group_chunk and exponent_radius must be positive")
    rows, columns = (int(item) for item in value.shape)
    group_size = rate.scale_group_size
    groups_per_row = math.ceil(columns / group_size)
    padded_columns = groups_per_row * group_size
    target = torch.device(device)
    source = value.to(device=target, dtype=torch.float32).contiguous()
    padded = torch.nn.functional.pad(source, (0, padded_columns - columns))
    groups = padded.reshape(rows * groups_per_row, group_size)
    valid = torch.ones(
        (rows, groups_per_row, group_size), device=target, dtype=torch.bool
    )
    if padded_columns != columns:
        valid[:, -1, columns % group_size :] = False
    valid = valid.reshape(rows * groups_per_row, group_size)
    group_count = int(groups.shape[0])
    best_exponent = torch.empty(group_count, device=target, dtype=torch.int32)
    best_error = torch.empty(group_count, device=target)
    low = float(alphabet.low_magnitude)
    high = float(alphabet.high_magnitude)
    threshold_factor = 0.5 * (low + high)
    offsets = torch.arange(
        -exponent_radius,
        exponent_radius + 1,
        device=target,
        dtype=torch.int32,
    )

    for start in range(0, group_count, group_chunk):
        stop = min(group_count, start + group_chunk)
        absolute = groups[start:stop].abs()
        mask = valid[start:stop]
        peak = torch.where(mask, absolute, torch.zeros_like(absolute)).amax(dim=1)
        safe_peak = torch.where(peak > 0, peak, torch.ones_like(peak))
        base = torch.ceil(torch.log2(safe_peak / high)).to(torch.int32)
        candidate_exponent = (base[:, None] + offsets[None, :]).clamp(
            -127, 127
        )
        chunk_best_error = torch.full(
            (stop - start,), float("inf"), device=target
        )
        chunk_best_exponent = candidate_exponent[:, 0].clone()
        for candidate in range(candidate_exponent.shape[1]):
            exponent = candidate_exponent[:, candidate]
            scale = _scale_value(exponent)
            magnitude = torch.where(
                absolute <= threshold_factor * scale[:, None], low, high
            )
            error = (
                (absolute - magnitude * scale[:, None]).square()
                * mask
            ).sum(dim=1)
            better = error < chunk_best_error
            chunk_best_error = torch.where(better, error, chunk_best_error)
            chunk_best_exponent = torch.where(
                better, exponent, chunk_best_exponent
            )
        best_error[start:stop] = chunk_best_error
        best_exponent[start:stop] = chunk_best_exponent

    scale = _scale_value(best_exponent)
    absolute = groups.abs()
    high_symbol = absolute > threshold_factor * scale[:, None]
    if low == 0.0:
        magnitude = high_symbol.to(torch.float32) * high
        reconstruction_groups = torch.copysign(
            magnitude * scale[:, None], groups
        )
        symbols = torch.zeros_like(groups, dtype=torch.uint8)
        symbols = torch.where(
            high_symbol & (groups >= 0),
            torch.ones_like(symbols),
            symbols,
        )
        symbols = torch.where(
            high_symbol & (groups < 0),
            torch.full_like(symbols, 2),
            symbols,
        )
    else:
        magnitude = torch.where(high_symbol, high, low)
        reconstruction_groups = torch.copysign(
            magnitude * scale[:, None], groups
        )
        # 0=-high, 1=-low, 2=+low, 3=+high.
        symbols = torch.where(
            groups < 0,
            torch.where(
                high_symbol,
                torch.zeros_like(groups, dtype=torch.uint8),
                torch.ones_like(groups, dtype=torch.uint8),
            ),
            torch.where(
                high_symbol,
                torch.full_like(groups, 3, dtype=torch.uint8),
                torch.full_like(groups, 2, dtype=torch.uint8),
            ),
        )
    reconstruction = reconstruction_groups.reshape(rows, padded_columns)[
        :, :columns
    ].contiguous()
    symbols = symbols.reshape(rows, padded_columns)[:, :columns].contiguous()
    scale_raw = (best_exponent + 127).to(torch.uint8).reshape(
        rows, groups_per_row
    )

    nibble_map = torch.tensor(
        alphabet.e2m1_nibbles, device=target, dtype=torch.uint8
    )
    nibbles = nibble_map[symbols.to(torch.int64)]
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    native_blocks = columns // 32
    block_position = torch.arange(native_blocks, device=target) * 32
    block_group = torch.div(
        block_position, group_size, rounding_mode="floor"
    )
    native_scale_raw = scale_raw[:, block_group]
    decoded = decode_mxfp4(
        packed.cpu(), native_scale_raw.cpu(), device="cpu"
    )
    if not torch.equal(decoded, reconstruction.cpu()):
        raise RuntimeError("strict scalar result failed native MXFP4 round-trip")
    measured_sse = float((source - reconstruction).square().sum().cpu())
    searched_sse = float(best_error.sum().cpu())
    if not math.isclose(measured_sse, searched_sse, rel_tol=5e-6, abs_tol=1e-8):
        raise RuntimeError("E8M0 scale search and reconstructed SSE disagree")
    metadata = {
        "alphabet": alphabet.label,
        "alphabet_levels": (
            [0.0, high, -high, -0.0]
            if low == 0.0
            else [-high, -low, low, high]
        ),
        "alphabet_e2m1_nibbles": list(alphabet.e2m1_nibbles),
        "learned_codebook": False,
        "scale_dtype": "E8M0",
        "scale_exponent_radius": exponent_radius,
        "unique_scale_bytes": int(scale_raw.unique().numel()),
        "scale_byte_min": int(scale_raw.min().cpu()),
        "scale_byte_max": int(scale_raw.max().cpu()),
        "physical_storage_roundtrip_verified": True,
        "final_values_are_native_block32_mxfp4": True,
    }
    return reconstruction.cpu(), symbols.cpu(), scale_raw.cpu(), metadata


@torch.inference_mode()
def quantize_row_scale_strict_mxfp4_sq(
    value: torch.Tensor,
    alphabet: SqAlphabet,
    rate: SqRowScaleRate,
    *,
    device: str | torch.device,
    exponent_radius: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Use a row E8M0 base plus a 1-bit native-block exponent selector."""

    if value.ndim != 2 or value.shape[1] % rate.native_block_size:
        raise ValueError("row-scale MXFP4-SQ requires [rows,K], K divisible by 32")
    if exponent_radius <= 0:
        raise ValueError("exponent_radius must be positive")
    target = torch.device(device)
    source = value.to(device=target, dtype=torch.float32).contiguous()
    rows, columns = (int(item) for item in source.shape)
    blocks_per_row = columns // rate.native_block_size
    blocks = source.reshape(rows, blocks_per_row, rate.native_block_size)
    absolute = blocks.abs()
    low = float(alphabet.low_magnitude)
    high = float(alphabet.high_magnitude)
    threshold_factor = 0.5 * (low + high)
    peak = absolute.amax(dim=(1, 2))
    safe_peak = torch.where(peak > 0, peak, torch.ones_like(peak))
    base_guess = torch.ceil(torch.log2(safe_peak / high)).to(torch.int32)
    offsets = torch.arange(
        -exponent_radius,
        exponent_radius + 1,
        device=target,
        dtype=torch.int32,
    )
    best_error = torch.full((rows,), float("inf"), device=target)
    best_base = base_guess.clone()
    best_delta = torch.ones(rows, device=target, dtype=torch.int32)

    def block_error(exponent: torch.Tensor) -> torch.Tensor:
        scale = _scale_value(exponent)
        magnitude = torch.where(
            absolute <= threshold_factor * scale[:, None, None], low, high
        )
        return (
            absolute - magnitude * scale[:, None, None]
        ).square().sum(dim=2)

    for delta_value in range(1, 5):
        for offset in offsets:
            base = (base_guess + offset).clamp(-127, 127 - delta_value)
            error_low = block_error(base)
            error_high = block_error(base + delta_value)
            row_error = torch.minimum(error_low, error_high).sum(dim=1)
            better = row_error < best_error
            best_error = torch.where(better, row_error, best_error)
            best_base = torch.where(better, base, best_base)
            best_delta = torch.where(
                better,
                torch.full_like(best_delta, delta_value),
                best_delta,
            )

    error_low = block_error(best_base)
    error_high = block_error(best_base + best_delta)
    selector = error_high < error_low
    block_exponent = best_base[:, None] + selector.to(torch.int32) * best_delta[:, None]
    block_scale = _scale_value(block_exponent)
    high_symbol = absolute > threshold_factor * block_scale[:, :, None]
    if low == 0.0:
        magnitude = high_symbol.to(torch.float32) * high
        reconstruction_blocks = torch.copysign(
            magnitude * block_scale[:, :, None], blocks
        )
        symbols = torch.zeros_like(blocks, dtype=torch.uint8)
        symbols = torch.where(
            high_symbol & (blocks >= 0), torch.ones_like(symbols), symbols
        )
        symbols = torch.where(
            high_symbol & (blocks < 0),
            torch.full_like(symbols, 2),
            symbols,
        )
    else:
        magnitude = torch.where(high_symbol, high, low)
        reconstruction_blocks = torch.copysign(
            magnitude * block_scale[:, :, None], blocks
        )
        symbols = torch.where(
            blocks < 0,
            torch.where(
                high_symbol,
                torch.zeros_like(blocks, dtype=torch.uint8),
                torch.ones_like(blocks, dtype=torch.uint8),
            ),
            torch.where(
                high_symbol,
                torch.full_like(blocks, 3, dtype=torch.uint8),
                torch.full_like(blocks, 2, dtype=torch.uint8),
            ),
        )
    reconstruction = reconstruction_blocks.reshape(rows, columns).contiguous()
    symbols = symbols.reshape(rows, columns).contiguous()
    native_scale_raw = (block_exponent + 127).to(torch.uint8)
    nibble_map = torch.tensor(
        alphabet.e2m1_nibbles, device=target, dtype=torch.uint8
    )
    nibbles = nibble_map[symbols.to(torch.int64)]
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    decoded = decode_mxfp4(
        packed.cpu(), native_scale_raw.cpu(), device="cpu"
    )
    if not torch.equal(decoded, reconstruction.cpu()):
        raise RuntimeError("row-scale scalar result failed native MXFP4 round-trip")
    measured_sse = float((source - reconstruction).square().sum().cpu())
    searched_sse = float(best_error.sum().cpu())
    if not math.isclose(measured_sse, searched_sse, rel_tol=5e-6, abs_tol=1e-8):
        raise RuntimeError("row-scale search and reconstructed SSE disagree")
    delta_counts = torch.bincount(best_delta, minlength=5)
    metadata = {
        "alphabet": alphabet.label,
        "alphabet_levels": (
            [0.0, high, -high, -0.0]
            if low == 0.0
            else [-high, -low, low, high]
        ),
        "alphabet_e2m1_nibbles": list(alphabet.e2m1_nibbles),
        "learned_codebook": False,
        "scale_layout": "row-E8M0-base+2bit-delta+1bit-per-block-selector",
        "scale_dtype": "E8M0",
        "scale_exponent_radius": exponent_radius,
        "row_base_byte_min": int((best_base + 127).min().cpu()),
        "row_base_byte_max": int((best_base + 127).max().cpu()),
        "row_delta_counts": {
            str(delta): int(delta_counts[delta].cpu()) for delta in range(1, 5)
        },
        "high_scale_selector_fraction": float(selector.float().mean().cpu()),
        "unique_effective_scale_bytes": int(native_scale_raw.unique().numel()),
        "physical_storage_roundtrip_verified": True,
        "final_values_are_native_block32_mxfp4": True,
    }
    return reconstruction.cpu(), symbols.cpu(), native_scale_raw.cpu(), metadata


def _raw_gate_up_source(
    checkpoint: V4FCheckpoint, layer: int, expert: int
) -> torch.Tensor:
    source = checkpoint.expert_source(layer, "gate_up")
    values: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    for part in ("w1", "w3"):
        packed, raw_scale = source._raw_part(expert, part)
        values.append(np.asarray(packed, dtype=np.uint8).reshape(2048, 2048))
        scales.append(np.asarray(raw_scale, dtype=np.uint8).reshape(2048, 128))
    packed = np.concatenate(values, axis=0)
    raw_scale = np.concatenate(scales, axis=0)
    return decode_mxfp4(packed, raw_scale, device="cpu").contiguous()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--scale-group-size", type=int, default=192)
    parser.add_argument(
        "--scale-layout",
        choices=("group-byte", "row-base-1bit"),
        default="group-byte",
    )
    parser.add_argument("--group-chunk", type=int, default=4096)
    parser.add_argument("--exponent-radius", type=int, default=8)
    parser.add_argument(
        "--alphabet",
        choices=(*ALPHABETS, "all"),
        default="all",
    )
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
    source = _raw_gate_up_source(checkpoint, args.layer, args.expert)
    rows, columns = (int(item) for item in source.shape)
    if args.scale_layout == "group-byte":
        rate: SqRate | SqRowScaleRate = strict_mxfp4_sq_rate(
            rows, columns, scale_group_size=args.scale_group_size
        )
    else:
        rate = strict_mxfp4_sq_row_scale_rate(rows, columns)
    baseline_budget = NVQ2_E8.payload_nbytes(rows, columns)
    if rate.payload_nbytes > baseline_budget:
        raise RuntimeError("strict MXFP4-SQ exceeds the NVQ2 payload budget")
    print(
        f"rate match: NVQ2={baseline_budget} bytes; "
        f"strict MXFP4-SQ={rate.payload_nbytes} bytes",
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

    labels = list(ALPHABETS) if args.alphabet == "all" else [args.alphabet]
    candidates: list[dict[str, Any]] = []
    for label in labels:
        print(f"quantizing fixed alphabet {label}...", file=sys.stderr, flush=True)
        candidate_start = time.perf_counter()
        if isinstance(rate, SqRate):
            reconstruction, _, _, metadata = quantize_strict_mxfp4_sq(
                source,
                ALPHABETS[label],
                rate,
                device=args.device,
                group_chunk=args.group_chunk,
                exponent_radius=args.exponent_radius,
            )
        else:
            reconstruction, _, _, metadata = quantize_row_scale_strict_mxfp4_sq(
                source,
                ALPHABETS[label],
                rate,
                device=args.device,
                exponent_radius=args.exponent_radius,
            )
        seconds = time.perf_counter() - candidate_start
        metrics = _metrics(source, reconstruction)
        candidates.append(
            {
                "format": f"strict-MXFP4-SQ-{label}",
                **rate.__dict__,
                "budget_slack_nbytes": baseline_budget - rate.payload_nbytes,
                "seconds": seconds,
                **metrics,
                **metadata,
            }
        )
        print(
            f"{label}: SSE={metrics['error_sse']:.9g}, "
            f"SNR={metrics['snr_db']:.6f} dB",
            file=sys.stderr,
            flush=True,
        )
        if str(torch.device(args.device)) == "mps":
            torch.mps.empty_cache()
    best = min(candidates, key=lambda item: float(item["error_sse"]))
    result = {
        "schema": 1,
        "experiment": "dsv4f-strict-mxfp4-scalar-quantization",
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
        "strict_mxfp4_sq_candidates": candidates,
        "best_strict_mxfp4_sq": best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
