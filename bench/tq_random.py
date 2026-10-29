"""Deterministic random TQ rows with an explicit serialized byte budget."""

from __future__ import annotations

import numpy as np

from mfq.formats.tq import TQTensor


def row_bits(rows: int, bit_sum: int, seed: int) -> np.ndarray:
    """Sample unsorted 1..8-bit rows while preserving the exact total bits."""
    if rows < 1 or not rows <= bit_sum <= 8 * rows:
        raise ValueError("Row bit sum must be between rows and 8*rows")
    if bit_sum in (rows, 8 * rows):
        return np.full(rows, bit_sum // rows, dtype=np.uint8)
    mean = bit_sum / rows
    levels = np.arange(1, 9, dtype=np.float64)
    lo, hi = -40.0, 40.0
    for _ in range(64):
        slope = (lo + hi) / 2
        logits = slope * levels
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        if float(probabilities @ levels) < mean:
            lo = slope
        else:
            hi = slope
    rng = np.random.default_rng(seed)
    bits = rng.choice(np.arange(1, 9, dtype=np.uint8), rows, p=probabilities)
    delta = bit_sum - int(bits.sum(dtype=np.int64))
    while delta:
        available = np.flatnonzero(bits < 8 if delta > 0 else bits > 1)
        chosen = rng.choice(available, min(abs(delta), len(available)), replace=False)
        if delta > 0:
            bits[chosen] += 1
            delta -= len(chosen)
        else:
            bits[chosen] -= 1
            delta += len(chosen)
    return bits


def layout(rows: int, columns: int, target_bytes: int, *, groupsize: int = 512) -> dict:
    if rows < 1 or columns < 1:
        raise ValueError("Positive matrix dimensions required")
    if groupsize < 16 or groupsize > 2**20 or groupsize & (groupsize - 1):
        raise ValueError("Invalid TQ subgroup size")
    stored = (columns + 255) // 256 * 256
    groups = (stored + groupsize - 1) // groupsize
    metadata_bytes = 64 + rows * 8 + (rows * groups + 1) // 2
    bit_sum = int(np.clip(round((target_bytes - metadata_bytes) * 8 / stored), rows, 8 * rows))
    packed_bytes = stored * bit_sum // 8
    if packed_bytes > 0xFFFFFFFF:
        raise ValueError("RTRELL05 payload exceeds 32-bit byte count; split the row range")
    return dict(rows=rows, columns=columns, stored_columns=stored, groupsize=groupsize,
                bit_sum=bit_sum, metadata_bytes=metadata_bytes,
                packed_bytes=packed_bytes, nbytes=metadata_bytes + packed_bytes,
                target_bytes=int(target_bytes),
                effective_bpw=(metadata_bytes + packed_bytes) * 8 / (rows * columns))


def random_tensor(rows: int, columns: int, target_bytes: int, *, seed: int = 1,
                  groupsize: int = 512) -> TQTensor:
    info = layout(rows, columns, target_bytes, groupsize=groupsize)
    bits = row_bits(rows, info['bit_sum'], seed).astype(np.uint64)
    words = info['stored_columns'] * bits // 32
    offsets = np.r_[np.uint64(0), np.cumsum(words[:-1], dtype=np.uint64)]
    rng = np.random.default_rng(seed ^ 0x5451524F)
    # Random nonzero anchors keep fake activation magnitudes bounded. The
    # procedural book has approximately unit variance; no encoder is needed.
    anchors = (rng.uniform(.6, 1.0, rows) / np.sqrt(columns)).astype(np.float16)
    descriptors = offsets | ((bits - 1) << np.uint64(32))
    descriptors |= anchors.view(np.uint16).astype(np.uint64) << np.uint64(40)
    groups = (info['stored_columns'] + groupsize - 1) // groupsize
    scales = rng.integers(0, 256, (rows * groups + 1) // 2, dtype=np.uint8)
    packed = rng.integers(0, 256, info['packed_bytes'], dtype=np.uint8)
    return TQTensor(rows, info['stored_columns'], groupsize, False, descriptors, scales, packed)
