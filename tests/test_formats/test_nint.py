"""Basic correctness tests for the nint codec (Neuron-anchored INT)."""

from __future__ import annotations

import numpy as np
import pytest

from mfq.formats import nint
from mfq.formats.nint import NINT2_SPEC, NintSpec


def test_spec_defaults():
    s = NintSpec()
    assert s.bits == 4 and s.groupsize == 24 and s.sub_bits == 6
    assert s.nmax == 15


def test_spec_bpw():
    s = NintSpec(4, 24, 6)
    assert abs(s.bpw(5120) - (4 + 32 / 5120 + 12 / 24)) < 1e-9


def test_make_qkx2_returns_affine_model():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.03, size=(8, 24)).astype(np.float32)
    w = np.abs(x) + 0.01
    scale, zp = nint.make_qkx2(x, w, nmax=15)
    assert scale.shape == (8,) and zp.shape == (8,)
    assert (zp <= 1e-6).all()  # zp ≤ 0


def test_quantize_dequantize_shape_and_range():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.05, size=5120).astype(np.float32)
    spec = NintSpec(4, 24, 6)
    code = nint.quantize(x, spec)
    assert code.n == 5120
    assert code.q.size >= 5120            # Includes trailing-group padding
    assert int(code.q.max()) <= spec.nmax
    K = (1 << spec.sub_bits) - 1
    assert int(code.sub_scale.max()) <= K
    assert int(code.sub_min.max()) <= K
    xr = nint.dequantize(code)
    assert xr.shape == (5120,)
    assert np.isfinite(xr).all()


def test_quantize_gaussian_snr():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.05, size=5120).astype(np.float32)
    code = nint.quantize(x, NintSpec(4, 24, 6))
    xr = nint.dequantize(code)
    err = x[: xr.size] - xr
    snr = 10 * np.log10((x[: xr.size] ** 2).sum() / (err ** 2).sum())
    assert snr > 18.0


def test_pad_when_groupsize_not_divide():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.05, size=5100).astype(np.float32)  # 5100 % 24 != 0
    code = nint.quantize(x, NintSpec(4, 24, 6))
    assert code.n == 5100
    xr = nint.dequantize(code)
    assert xr.shape == (5100,)


def test_bits5_nmax():
    s = NintSpec(bits=5, groupsize=32, sub_bits=6)
    assert s.nmax == 31
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.05, size=1024).astype(np.float32)
    code = nint.quantize(x, s)
    assert int(code.q.max()) <= 31
    assert nint.dequantize(code).shape == (1024,)


def test_bits2_profile():
    spec = NINT2_SPEC
    assert spec.nmax == 3
    assert abs(spec.bpw(5120) - 2.63125) < 1e-9
    rng = np.random.default_rng(2165)
    x = rng.normal(0, 0.05, size=1537).astype(np.float32)
    code = nint.quantize(x, spec)
    assert int(code.q.max()) <= 3
    assert nint.dequantize(code).shape == x.shape


@pytest.mark.parametrize("bits,gs,k", [(6, 24, 6), (8, 16, 8)])
def test_bits6_bits8_codec(bits, gs, k):
    s = NintSpec(bits=bits, groupsize=gs, sub_bits=k)
    rng = np.random.default_rng(bits * 100 + gs)
    x = rng.normal(0, 0.05, size=1537).astype(np.float32)
    code = nint.quantize(x, s)
    assert int(code.q.max()) <= s.nmax
    K = (1 << k) - 1
    assert int(code.sub_scale.max()) <= K
    assert int(code.sub_min.max()) <= K
    xr = nint.dequantize(code)
    assert xr.shape == x.shape
    assert np.isfinite(xr).all()
