"""Apple-silicon native MXFP4/MXFP8 Metal kernel tests."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats.header import FileHeader  # noqa: E402
from mfq.formats.io import save  # noqa: E402
from mfq.formats.mx import MxTensor  # noqa: E402
from mfq.kernels.metal.mx import (  # noqa: E402
    MetalMxWeight,
    mx_dequantize,
    mx_embedding,
    mx_matmul,
)
from mfq.runtime.mlx_linear import MlxMxEmbedding, MlxMxLinear, MlxNintModel  # noqa: E402


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _e8m0(raw: np.ndarray) -> np.ndarray:
    return np.exp2(raw.astype(np.int16) - 127).astype(np.float32)


def _fp4(raw: np.ndarray) -> np.ndarray:
    table = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
    )
    values = table[raw & 7]
    return np.where((raw & 8) == 0, values, -values)


def _fp8(raw: np.ndarray) -> np.ndarray:
    unsigned = raw.astype(np.uint16)
    exponent = (unsigned >> 3) & 15
    mantissa = unsigned & 7
    subnormal = np.ldexp(mantissa.astype(np.float32) / 8.0, -6)
    normal = np.ldexp(
        1.0 + mantissa.astype(np.float32) / 8.0,
        exponent.astype(np.int16) - 7,
    )
    values = np.where(exponent == 0, subnormal, normal)
    return np.where((unsigned & 128) == 0, values, -values).astype(np.float32)


def _fixture(dtype: str, *, out: int = 5) -> tuple[MxTensor, np.ndarray]:
    rng = np.random.default_rng(904 if dtype == "MXFP4" else 908)
    if dtype == "MXFP4":
        width = 96
        values = rng.integers(0, 256, size=(out, width // 2), dtype=np.uint8)
        scales = rng.integers(124, 130, size=(out, width // 32), dtype=np.uint8)
        low = values & 15
        high = values >> 4
        codes = np.stack((low, high), axis=-1).reshape(out, width)
        dense = _fp4(codes) * np.repeat(_e8m0(scales), 32, axis=1)
    else:
        width = 128
        values = rng.integers(0, 255, size=(out, width), dtype=np.uint8)
        values[(values & 127) == 127] = 126
        scales = rng.integers(124, 130, size=(1, 1), dtype=np.uint8)
        dense = _fp8(values) * _e8m0(scales)[0, 0]
    return MxTensor(dtype, (out, width), values, scales), dense


@pytest.mark.parametrize("dtype", ["MXFP4", "MXFP8"])
@pytest.mark.parametrize("rows", [1, 7, 64])
def test_mx_packed_gemv_mmq_and_large_gemm(dtype: str, rows: int):
    tensor, dense = _fixture(dtype)
    source = np.random.default_rng(1000 + rows).normal(
        0.0, 0.03, size=(rows, tensor.shape[1])
    ).astype(np.float32)
    weight = MetalMxWeight.from_tensor(tensor)

    actual = _array(mx_matmul(weight, source))
    expected = (
        _array(mx.array(source) @ mx.array(dense).T)
        if rows >= 64
        else source @ dense.T
    )

    tolerance = 2e-5
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", ["MXFP4", "MXFP8"])
def test_mx_dequantize_and_embedding(dtype: str):
    tensor, dense = _fixture(dtype)
    weight = MetalMxWeight.from_tensor(tensor)

    np.testing.assert_allclose(
        _array(mx_dequantize(weight, dtype=mx.float32)), dense, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        _array(mx_embedding(weight, np.array([4, 1], dtype=np.int32), dtype=mx.float32)),
        dense[[4, 1]],
        rtol=0,
        atol=0,
    )


def test_mmap_model_constructs_native_mx_layers(tmp_path):
    tensor, dense = _fixture("MXFP4")
    path = tmp_path / "native-mx.mfq"
    save(path, FileHeader(version=2, model_arch="native-mx"), {"weight": tensor})

    with MlxNintModel.from_mfq(path) as model:
        linear = model.linear("weight")
        embedding = model.embedding("weight")
        assert isinstance(linear, MlxMxLinear)
        assert isinstance(embedding, MlxMxEmbedding)
        np.testing.assert_allclose(
            _array(embedding(np.array([3, 0], dtype=np.int32))),
            dense[[3, 0]].astype(np.float16),
            rtol=0,
            atol=0,
        )
