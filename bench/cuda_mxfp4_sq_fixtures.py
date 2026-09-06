"""Deterministic, independently decoded frozen Metal SQ2/SQ3 wire fixtures.

Standard-library only. Produces small test artifacts, never model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct

ROOT = Path(__file__).resolve().parents[1]


def palette_nibbles(bits: int) -> list[int]:
    source = (ROOT / f"cpp_runtime/backends/metal/ops/mlx_mxfp4_sq{bits}.h").read_text()
    match = re.search(rf"kMxfp4Sq{bits}PaletteNibbles\s*\{{(.*?)\}};", source, re.S)
    if match is None:
        raise ValueError("frozen Metal palette definition missing")
    result = [int(x) for x in re.findall(r"\d+", match.group(1))]
    if len(result) != 32 * (1 << bits):
        raise ValueError("unexpected Metal palette size")
    return result


def pack(values: list[int], bits: int) -> bytes:
    out = bytearray((len(values) * bits + 7) // 8)
    for i, value in enumerate(values):
        if not 0 <= value < 1 << bits:
            raise ValueError("symbol out of range")
        for bit in range(bits):
            offset = i * bits + bit
            out[offset // 8] |= ((value >> bit) & 1) << (offset % 8)
    return bytes(out)


def fixture(bits: int, rows: int, width: int, base: int = 120) -> tuple[bytes, bytes]:
    if bits not in (2, 3) or rows <= 0 or width <= 0 or width % 32 or not 0 <= base <= 251:
        raise ValueError("invalid fixture geometry")
    rng = random.Random(20260907 + bits * 100000 + rows * 100 + width)
    symbols = [rng.randrange(1 << bits) for _ in range(rows * width)]
    selectors = [(i // 4) & 1 for i in range(rows * width // 32)]
    scales = [i % 4 for i in range(rows * 8)]
    palettes = [i % 32 for i in range(rows * 8)]
    table = palette_nibbles(bits)
    magnitudes = (0., .5, 1., 1.5, 2., 3., 4., 6.)
    dense = bytearray()
    for block in range(rows * width // 32):
        first = block * 32
        # Force all implicit tag values, retaining the independent high SQ3
        # symbol bit. The reference computes XOR over symbols, not word masks.
        low = 0
        for symbol in symbols[first:first + 32]:
            low ^= symbol & 3
        symbols[first] ^= low ^ (block % 4)
        tag = 0
        for symbol in symbols[first:first + 32]:
            tag ^= symbol & 3
        tag |= selectors[block] << 2
        state = (first // width) * 8 + tag
        scale = math.ldexp(1., base + scales[state] - 127)
        for symbol in symbols[first:first + 32]:
            nibble = table[palettes[state] * (1 << bits) + symbol]
            value = math.copysign(magnitudes[nibble & 7] * scale, -1. if nibble & 8 else 1.)
            try:
                dense.extend(struct.pack("<f", value))
            except OverflowError:
                dense.extend(struct.pack("<f", math.copysign(math.inf, value)))
    header = struct.pack("<4sBBHQQ", f"SQ{bits}\0".encode(), 1, base, 0, rows, width)
    blob = header + pack(symbols, bits) + pack(selectors, 1) + pack(scales, 2) + pack(palettes, 5)
    return blob, bytes(dense)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    shapes = [(n, k, 120) for n in (1, 7, 33, 128) for k in (32, 96, 256)]
    shapes += [(128, 256, b) for b in (0, 1, 251)]
    for bits in (2, 3):
        for n, k, base in shapes:
            name = f"sq{bits}-n{n}-k{k}-b{base}"
            blob, dense = fixture(bits, n, k, base)
            (args.output / f"{name}.sq").write_bytes(blob)
            (args.output / f"{name}.f32").write_bytes(dense)
            records.append({"name": name, "blob_sha256": hashlib.sha256(blob).hexdigest(),
                            "dense_sha256": hashlib.sha256(dense).hexdigest()})
    (args.output / "manifest.json").write_text(json.dumps({"seed": 20260907, "fixtures": records}, indent=2))
    print(json.dumps({"fixtures": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
