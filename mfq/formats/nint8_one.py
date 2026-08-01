"""llama.cpp-compatible Q8_1 activation blocks for controlled KLD tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GROUP_SIZE = 32
QMAX = 127


@dataclass(frozen=True)
class NINT8OneBlocks:
    q: np.ndarray
    d: np.ndarray
    s: np.ndarray
    reconstructed: np.ndarray


def _roundf(values: np.ndarray) -> np.ndarray:
    """Match C ``roundf``: halfway cases round away from zero."""

    return np.where(
        values >= 0.0,
        np.floor(values + np.float32(0.5)),
        np.ceil(values - np.float32(0.5)),
    )


def quantize_nint8_one(values: np.ndarray) -> NINT8OneBlocks:
    """Quantize the last dimension into Q8_1-compatible 32-value groups."""

    source = np.asarray(values, dtype=np.float32)
    if source.ndim < 1 or source.size == 0 or source.shape[-1] == 0:
        raise ValueError("NINT8-1 input must be non-empty")
    if not np.isfinite(source).all():
        raise ValueError("NINT8-1 input must contain only finite values")

    width = source.shape[-1]
    groups = (width + GROUP_SIZE - 1) // GROUP_SIZE
    padded_width = groups * GROUP_SIZE
    padded = np.zeros((*source.shape[:-1], padded_width), dtype=np.float32)
    padded[..., :width] = source
    grouped = padded.reshape(*source.shape[:-1], groups, GROUP_SIZE)

    amax = np.max(np.abs(grouped), axis=-1)
    scale = amax / np.float32(QMAX)
    inverse = np.zeros_like(scale)
    np.divide(np.float32(1.0), scale, out=inverse, where=scale != 0.0)
    codes_f32 = _roundf(grouped * inverse[..., None])
    codes_f32 = np.clip(codes_f32, -QMAX, QMAX)
    q = codes_f32.astype(np.int8)

    d = scale.astype(np.float16)
    q_sum = q.astype(np.int32).sum(axis=-1)
    s = (q_sum.astype(np.float32) * scale).astype(np.float16)
    reconstructed = (
        q.astype(np.float32) * d.astype(np.float32)[..., None]
    ).astype(np.float16)
    reconstructed = reconstructed.reshape(*source.shape[:-1], padded_width)
    reconstructed = reconstructed[..., :width].copy()
    return NINT8OneBlocks(q=q, d=d, s=s, reconstructed=reconstructed)


__all__ = [
    "GROUP_SIZE",
    "NINT8OneBlocks",
    "QMAX",
    "quantize_nint8_one",
]
