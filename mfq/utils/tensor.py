"""Tensor helpers: metrics and reshape utilities."""

from __future__ import annotations

import numpy as np


def snr(original: np.ndarray, approx: np.ndarray) -> float:
    """Signal-to-noise ratio in dB: ``10*log10(sum(x^2) / sum((x-x_hat)^2))``.

    Compare element-wise, using the shorter length when lengths differ. Return ``inf`` for zero error.
    """

    o = np.asarray(original, dtype=np.float64).reshape(-1)
    a = np.asarray(approx, dtype=np.float64).reshape(-1)
    n = min(o.size, a.size)
    o, a = o[:n], a[:n]
    err = o - a
    denom = float((err * err).sum())
    if denom <= 0.0:
        return float("inf")
    return 10.0 * np.log10(float((o * o).sum()) / denom)


def mse(original: np.ndarray, approx: np.ndarray) -> float:
    """Mean squared error, using the shorter length when lengths differ."""

    o = np.asarray(original, dtype=np.float64).reshape(-1)
    a = np.asarray(approx, dtype=np.float64).reshape(-1)
    n = min(o.size, a.size)
    return float(((o[:n] - a[:n]) ** 2).mean())
