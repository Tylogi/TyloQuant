"""Hadamard rotation。

Randomized orthogonal Hadamard transforms make tensor distributions more uniform and reduce outlier channels,
substantially improving quantizability at ultra-low precision.

Use cases (see the development documentation):
- INT layers below 4 bits (trading a small amount of speed for accuracy);
- additional rotation for accuracy-critical experts or tensors;
- rotation for KV quantization is "free" (it does not use the weight path), so KV INT is always rotated.
"""

from __future__ import annotations

import numpy as np


def hadamard_matrix(n: int) -> np.ndarray:
    """Return an n-by-n Hadamard matrix, where n must be a power of two."""

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n 必须是 2 的幂，得到 {n}")
    h = np.array([[1]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.vstack([np.hstack([h, h]), np.hstack([h, -h])])
    return h / np.sqrt(n)


def rotate(weight: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply a Hadamard rotation to weights along ``axis``."""

    raise NotImplementedError
