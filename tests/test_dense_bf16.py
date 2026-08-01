from __future__ import annotations

import numpy as np
import pytest
import torch

from mfq.formats.io import BFloat16Array
from mfq.runtime.torch_linear import TorchDenseLinear


def _bf16_array(values: torch.Tensor) -> BFloat16Array:
    return (
        values.to(torch.bfloat16)
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .view(BFloat16Array)
    )


def test_torch_dense_linear_reinterprets_bf16_bits() -> None:
    weight = _bf16_array(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    linear = TorchDenseLinear(weight, device="cpu")

    assert linear.weight.dtype == torch.bfloat16
    actual = linear(torch.tensor([[2.0, -1.0]], dtype=torch.bfloat16))
    torch.testing.assert_close(
        actual.float(),
        torch.tensor([[0.0, 2.0]]),
        rtol=0,
        atol=0,
    )


def test_mlx_dense_linear_reinterprets_bf16_bits() -> None:
    mx = pytest.importorskip("mlx.core")
    from mfq.runtime.mlx_linear import MlxDenseLinear, mlx_dense_array

    weight = _bf16_array(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    linear = MlxDenseLinear(weight)

    assert linear.weight.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.asarray(mlx_dense_array(weight, dtype=mx.float32)),
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    actual = np.asarray(linear(np.asarray([[2.0, -1.0]], dtype=np.float32)).astype(mx.float32))
    np.testing.assert_array_equal(actual, np.asarray([[0.0, 2.0]], dtype=np.float32))
