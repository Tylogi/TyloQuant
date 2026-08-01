"""张量辅助函数：度量与 reshape 工具。"""

from __future__ import annotations

import numpy as np


def snr(original: np.ndarray, approx: np.ndarray) -> float:
    """信噪比 (dB)：``10·log10(Σx² / Σ(x−x̃)²)``。

    逐元素比较，长度不等时取较短者。误差为 0 返回 ``inf``。
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
    """均方误差。长度不等取较短者。"""

    o = np.asarray(original, dtype=np.float64).reshape(-1)
    a = np.asarray(approx, dtype=np.float64).reshape(-1)
    n = min(o.size, a.size)
    return float(((o[:n] - a[:n]) ** 2).mean())
