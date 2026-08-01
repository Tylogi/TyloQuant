"""Apple-silicon NINT8-0 kernel and runtime tests."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats import io  # noqa: E402
from mfq.formats.header import FileHeader  # noqa: E402
from mfq.formats.nint8_zero import (  # noqa: E402
    dequantize_nint8_zero,
    quantize_nint8_zero,
)
from mfq.kernels.metal.nint8_zero import (  # noqa: E402
    MetalNint8ZeroWeight,
    nint8_zero_dequantize,
    nint8_zero_embedding,
    nint8_zero_gemm,
    nint8_zero_gemv,
    nint8_zero_matmul,
    nint8_zero_mmq,
)
from mfq.runtime import (  # noqa: E402
    MlxNint8ZeroEmbedding,
    MlxNint8ZeroLinear,
    MlxNintModel,
)


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _tensor():
    dense = (
        np.random.default_rng(20260728)
        .normal(
            0,
            0.1,
            size=(13, 64),
        )
        .astype(np.float32)
    )
    tensor = quantize_nint8_zero(dense)
    return tensor, dequantize_nint8_zero(tensor)


def test_nint8_zero_dequant_and_embedding():
    tensor, decoded = _tensor()
    weight = MetalNint8ZeroWeight.from_tensor(tensor)
    actual = _array(nint8_zero_dequantize(weight, dtype=mx.float32))
    np.testing.assert_array_equal(actual, decoded)
    ids = np.asarray([[0, 5], [12, 2]], dtype=np.int32)
    selected = _array(nint8_zero_embedding(weight, ids, dtype=mx.float32))
    np.testing.assert_array_equal(selected, decoded[ids])


@pytest.mark.parametrize("rows", [1, 4, 64])
def test_nint8_zero_matmul_paths(rows: int):
    tensor, decoded = _tensor()
    weight = MetalNint8ZeroWeight.from_tensor(tensor)
    source = (
        np.random.default_rng(100 + rows)
        .normal(
            0,
            0.1,
            size=(rows, tensor.neuron_len),
        )
        .astype(np.float16 if rows >= 64 else np.float32)
    )
    actual = _array(nint8_zero_matmul(weight, source))
    expected = source.astype(np.float32) @ decoded.T
    tolerance = 4e-3 if rows >= 64 else 3e-5
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "rows,operation",
    [
        (1, nint8_zero_gemv),
        (4, nint8_zero_mmq),
        (33, nint8_zero_gemm),
    ],
)
def test_explicit_nint8_zero_gemv_mmq_gemm(rows: int, operation):
    tensor, decoded = _tensor()
    weight = MetalNint8ZeroWeight.from_tensor(tensor)
    source = (
        np.random.default_rng(200 + rows)
        .normal(
            0,
            0.1,
            size=(rows, tensor.neuron_len),
        )
        .astype(np.float16 if rows >= 17 else np.float32)
    )
    actual = _array(operation(weight, source))
    expected = source.astype(np.float32) @ decoded.T
    tolerance = 4e-3 if rows >= 17 else 3e-5
    if rows <= 16:
        assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_nint8_zero_fp16_gemm_partial_tiles():
    dense = (
        np.random.default_rng(401)
        .normal(
            0,
            0.1,
            size=(67, 96),
        )
        .astype(np.float32)
    )
    tensor = quantize_nint8_zero(dense)
    decoded = dequantize_nint8_zero(tensor)
    source = (
        np.random.default_rng(402)
        .normal(
            0,
            0.1,
            size=(65, tensor.neuron_len),
        )
        .astype(np.float16)
    )
    actual = _array(nint8_zero_gemm(MetalNint8ZeroWeight.from_tensor(tensor), source))
    expected = source.astype(np.float32) @ decoded.T
    np.testing.assert_allclose(actual, expected, rtol=4e-3, atol=4e-3)


def test_nint8_zero_mmap_linear_and_embedding(tmp_path):
    tensor, decoded = _tensor()
    path = tmp_path / "nint8-zero.mfq"
    io.save(
        path,
        FileHeader(model_arch="metal-nint8-zero", num_tensors=1),
        {"weight": tensor},
    )
    source = (
        np.random.default_rng(300)
        .normal(
            0,
            0.1,
            size=(3, tensor.neuron_len),
        )
        .astype(np.float32)
    )
    ids = np.asarray([0, 5, 12], dtype=np.int32)
    with MlxNintModel.from_mfq(path) as model:
        linear = model.linear("weight")
        embedding = model.embedding("weight")
        assert isinstance(linear, MlxNint8ZeroLinear)
        assert isinstance(embedding, MlxNint8ZeroEmbedding)
        np.testing.assert_allclose(
            _array(linear(source)),
            source @ decoded.T,
            rtol=3e-5,
            atol=3e-5,
        )
        np.testing.assert_array_equal(
            _array(embedding.forward(ids, dtype=mx.float32)),
            decoded[ids],
        )
        assert isinstance(model.tensors, io.MMapTensorStore)
        assert not model.tensors._cache
