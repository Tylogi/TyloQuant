from __future__ import annotations

import numpy as np
import pytest

from mfq.formats.io import _pack_tensor, _unpack_tensor
from mfq.formats.nint8_zero import (
    Nint8ZeroTensor,
    dequantize_nint8_zero,
    pack_nint8_zero,
    payload_nbytes,
    quantize_nint8_zero,
    unpack_nint8_zero,
)


def _tensor() -> Nint8ZeroTensor:
    scale = np.asarray([[0.5, 0.25], [1.0, 2.0]], dtype=np.float16)
    q = np.arange(-64, 64, dtype=np.int8).reshape(2, 2, 32)
    return Nint8ZeroTensor(
        shape=(2, 64),
        axis=0,
        scale=scale,
        q=q,
        neuron_len=64,
    )


def test_nint8_zero_roundtrip_preserves_q8_blocks() -> None:
    tensor = _tensor()
    blob = pack_nint8_zero(tensor)
    restored = unpack_nint8_zero(blob)
    assert len(blob) == payload_nbytes(tensor.shape, tensor.axis, tensor.neuron_len)
    assert restored.shape == tensor.shape
    assert restored.axis == tensor.axis
    np.testing.assert_array_equal(restored.scale.view(np.uint16), tensor.scale.view(np.uint16))
    np.testing.assert_array_equal(restored.q, tensor.q)
    assert pack_nint8_zero(restored) == blob


def test_nint8_zero_dequantization() -> None:
    tensor = _tensor()
    expected = (
        tensor.scale.astype(np.float32)[..., None]
        * tensor.q.astype(np.float32)
    ).reshape(2, 64)
    np.testing.assert_array_equal(dequantize_nint8_zero(tensor), expected)


def test_nint8_zero_dispatches_through_public_dtype() -> None:
    tensor = _tensor()
    dtype, blob = _pack_tensor(tensor)
    assert dtype == "NINT8-0"
    restored = _unpack_tensor(dtype, blob)
    assert isinstance(restored, Nint8ZeroTensor)
    np.testing.assert_array_equal(restored.q, tensor.q)


def test_nint8_zero_direct_quantization_matches_gguf_q8_0() -> None:
    from gguf import GGMLQuantizationType
    from gguf.quants import quantize

    rng = np.random.default_rng(20260726)
    weight = rng.normal(0.0, 0.2, (7, 96)).astype(np.float32)
    actual = pack_nint8_zero(quantize_nint8_zero(weight))
    header_nbytes = len(actual) - 7 * 3 * 34
    expected_blocks = quantize(weight, GGMLQuantizationType.Q8_0)

    np.testing.assert_array_equal(
        np.frombuffer(actual[header_nbytes:], dtype=np.uint8).reshape(7, -1),
        expected_blocks,
    )


@pytest.mark.parametrize("scale", [np.inf, np.nan, -1.0, 1e10])
def test_nint8_zero_rejects_invalid_fp16_scales(scale: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        Nint8ZeroTensor(
            shape=(1, 32),
            axis=0,
            scale=np.asarray([[scale]], dtype=np.float32),
            q=np.zeros((1, 1, 32), dtype=np.int8),
            neuron_len=32,
        )


def test_nint8_zero_quantizer_rejects_nonfinite_weights() -> None:
    weight = np.zeros((1, 32), dtype=np.float32)
    weight[0, 0] = np.nan
    with pytest.raises(ValueError, match="weights must be finite"):
        quantize_nint8_zero(weight)
