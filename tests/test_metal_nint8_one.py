"""Apple-silicon Q8_1 activation quantization tests."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats.nint8_one import quantize_nint8_one  # noqa: E402
from mfq.kernels.metal.nint8_one import (  # noqa: E402
    nint8_one_quantize_reconstruct,
)


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def test_nint8_one_metal_matches_cpu_oracle_with_tail():
    values = np.zeros((2, 35), dtype=np.float16)
    values[0, :6] = [127.0, 63.5, -63.5, 0.5, -0.5, 1.25]
    values[1] = np.linspace(-2.0, 2.0, 35, dtype=np.float16)
    oracle = quantize_nint8_one(values.astype(np.float32))

    actual = nint8_one_quantize_reconstruct(mx.array(values))

    np.testing.assert_array_equal(_array(actual.q), oracle.q)
    np.testing.assert_array_equal(_array(actual.d), oracle.d)
    np.testing.assert_array_equal(_array(actual.s), oracle.s)
    np.testing.assert_array_equal(
        _array(actual.reconstructed),
        oracle.reconstructed,
    )


def test_nint8_one_metal_zero_group_and_prefix_shape():
    values = np.zeros((2, 3, 32), dtype=np.float16)
    actual = nint8_one_quantize_reconstruct(values)

    assert actual.q.shape == (2, 3, 1, 32)
    assert actual.d.shape == (2, 3, 1)
    assert actual.s.shape == (2, 3, 1)
    assert actual.reconstructed.shape == values.shape
    np.testing.assert_array_equal(_array(actual.q), 0)
    np.testing.assert_array_equal(_array(actual.d), 0)
    np.testing.assert_array_equal(_array(actual.s), 0)
    np.testing.assert_array_equal(_array(actual.reconstructed), 0)


def test_nint8_one_metal_requires_nonempty_fp16_input():
    with pytest.raises(ValueError, match="FP16"):
        nint8_one_quantize_reconstruct(np.ones((1, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty"):
        nint8_one_quantize_reconstruct(np.empty((0, 32), dtype=np.float16))
