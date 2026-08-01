"""nint_quant 张量级量化测试。"""

from __future__ import annotations

import numpy as np
import pytest

from mfq.formats.nint import NintSpec
from mfq.quantize import nint_quant


def test_quantize_preserves_shape():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.05, size=(64, 480)).astype(np.float32)
    t = nint_quant.quantize(W, NintSpec(4, 24, 6), axis=0)
    assert t.shape == W.shape
    r = nint_quant.dequantize(t)
    assert r.shape == W.shape


def test_axis_1():
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.05, size=(480, 64)).astype(np.float32)
    t = nint_quant.quantize(W, NintSpec(4, 16, 6), axis=1)
    assert t.shape == W.shape and t.axis == 1
    r = nint_quant.dequantize(t)
    assert r.shape == W.shape
    assert np.isfinite(r).all()


def test_gaussian_snr():
    rng = np.random.default_rng(2)
    W = rng.normal(0, 0.05, size=(128, 528)).astype(np.float32)
    t = nint_quant.quantize(W, NintSpec(4, 24, 6), axis=0)
    r = nint_quant.dequantize(t)
    err = W - r
    snr = 10 * np.log10((W ** 2).sum() / (err ** 2).sum())
    assert snr > 20.0


def test_rejects_1d():
    with pytest.raises(ValueError):
        nint_quant.quantize(np.zeros(64, dtype=np.float32), NintSpec())


def test_allzero_row_stays_finite():
    W = np.zeros((4, 48), dtype=np.float32)
    W[0] = np.random.default_rng(3).normal(0, 0.05, 48)
    t = nint_quant.quantize(W, NintSpec(4, 24, 6), axis=0)
    r = nint_quant.dequantize(t)
    assert np.isfinite(r).all()
