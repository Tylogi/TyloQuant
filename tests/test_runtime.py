"""runtime 推理参考实现测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.header import FileHeader
from mfq.formats.nint import NintSpec
from mfq.quantize import nint_quant
from mfq.quantize.nint_quant import NintTensor
from mfq.runtime import (
    NintLinear,
    NintModel,
    SwiGLUFFN,
    clear_backends,
    register_backend,
)
from mfq.runtime.dequantize import dequantize


def _qt(W, spec=NintSpec(4, 24, 6), axis=0) -> NintTensor:
    return nint_quant.quantize(W.astype(np.float32), spec, axis=axis)


def test_nintlinear_matches_dequant_matmul():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.05, size=(32, 128)).astype(np.float32)
    x = rng.normal(0, 1, size=(4, 128)).astype(np.float32)
    lin = NintLinear(_qt(W))
    y = lin(x)
    ref = x @ nint_quant.dequantize(_qt(W)).T
    np.testing.assert_allclose(y, ref, atol=1e-4)


def test_nintlinear_bias_and_caching():
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.05, size=(16, 64)).astype(np.float32)
    b = rng.normal(0, 0.01, size=16).astype(np.float32)
    lin = NintLinear(_qt(W), bias=b)
    assert lin._w is None
    x = rng.normal(0, 1, size=(2, 64)).astype(np.float32)
    y = lin(x)
    assert lin._w is not None            # 惰性反量化已触发
    np.testing.assert_allclose(y, x @ lin.weight.T + b, atol=1e-5)


def test_silu_ffn_matches_float_reference():
    rng = np.random.default_rng(2)
    din, dinter = 64, 96
    Wg = rng.normal(0, 0.05, (dinter, din)).astype(np.float32)
    Wu = rng.normal(0, 0.05, (dinter, din)).astype(np.float32)
    Wd = rng.normal(0, 0.05, (din, dinter)).astype(np.float32)
    ffn = SwiGLUFFN(NintLinear(_qt(Wg)), NintLinear(_qt(Wu)), NintLinear(_qt(Wd)))
    x = rng.normal(0, 1, size=(3, din)).astype(np.float32)
    y = ffn(x)
    # float 参考（用反量化权重）
    g = x @ nint_quant.dequantize(_qt(Wg)).T
    u = x @ nint_quant.dequantize(_qt(Wu)).T
    a = g / (1 + np.exp(-g))
    ref = (a * u) @ nint_quant.dequantize(_qt(Wd)).T
    np.testing.assert_allclose(y, ref, atol=1e-4)
    assert y.shape == (3, din)


def test_model_roundtrip_and_linear(tmp_path: Path):
    rng = np.random.default_rng(3)
    Wg = rng.normal(0, 0.05, (32, 64)).astype(np.float32)
    Wd = rng.normal(0, 0.05, (64, 32)).astype(np.float32)
    tensors = {"blk.0.gate": _qt(Wg), "blk.0.down": _qt(Wd)}
    path = tmp_path / "m.mfq"
    io.save(path, FileHeader(model_arch="t", num_tensors=2), tensors)

    model = NintModel.from_mfq(path)
    assert set(model.tensors) == {"blk.0.gate", "blk.0.down"}
    lin = model.linear("blk.0.gate")
    x = rng.normal(0, 1, size=(2, 64)).astype(np.float32)
    np.testing.assert_allclose(lin(x), x @ nint_quant.dequantize(_qt(Wg)).T, atol=1e-4)
    with pytest.raises(KeyError):
        model.linear("missing")


def test_model_mmap_roundtrip_and_linear(tmp_path: Path):
    rng = np.random.default_rng(33)
    W = rng.normal(0, 0.05, (16, 64)).astype(np.float32)
    tensors = {"blk.0.gate": _qt(W)}
    path = tmp_path / "mmap.mfq"
    io.save(path, FileHeader(model_arch="t", num_tensors=1), tensors)

    model = NintModel.from_mfq(path, mmap=True)
    lin = model.linear("blk.0.gate")
    x = rng.normal(0, 1, size=(2, 64)).astype(np.float32)
    np.testing.assert_allclose(lin(x), x @ nint_quant.dequantize(_qt(W)).T, atol=1e-4)
    model.tensors.close()


def test_profile_dispatch_uses_registered_backend():
    rng = np.random.default_rng(4)
    W = rng.normal(0, 0.05, size=(8, 96)).astype(np.float32)
    t = _qt(W, NintSpec(4, 24, 6))
    calls = {"n": 0}

    def fake_kernel(tensor):
        calls["n"] += 1
        return np.ones_like(nint_quant.dequantize(tensor))  # 哨兵：全 1

    try:
        register_backend("NINT4-24", fake_kernel)
        out = dequantize(t)
        assert calls["n"] == 1
        assert (out == 1.0).all()
    finally:
        clear_backends()
    # 清除后回退到默认
    out2 = dequantize(t)
    assert not (out2 == 1.0).all()


def test_profile_label():
    assert NintSpec(4, 24, 6).profile_label == "NINT4-24"
    assert NintSpec(4, 32, 8).profile_label == "NINT4-32"
