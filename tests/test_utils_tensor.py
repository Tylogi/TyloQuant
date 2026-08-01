"""utils.tensor 度量测试。"""

from __future__ import annotations

import numpy as np

from mfq.utils.tensor import mse, snr


def test_snr_zero_db():
    # x=[1,0], approx=[1,1]: Σx²=1, Σe²=1 -> 0 dB
    assert abs(snr(np.array([1.0, 0.0]), np.array([1.0, 1.0]))) < 1e-9


def test_snr_perfect_is_inf():
    x = np.arange(100.0)
    assert snr(x, x) == float("inf")


def test_mse_basic():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 0.0])
    assert abs(mse(x, y) - 3.0) < 1e-9  # (0+0+9)/3
