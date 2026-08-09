"""NintLinear: a linear layer whose weights are :class:`~mfq.quantize.nint_quant.NintTensor`.

Lazy dequantization (computed and cached on first access) plus matmul. This is the NumPy reference implementation;
real kernels will fuse dequantization and matmul in backends registered with :mod:`mfq.runtime.dequantize`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mfq.quantize.nint_quant import NintTensor
from mfq.runtime.dequantize import dequantize


class NintLinear:
    """``y = x * W^T (+ b)``, where W is an NintTensor (axis=0 for quantization along the neuron axis).

    Attributes:
        weight_tensor: Quantized weights with restored shape ``[out, in]``.
        bias: Optional float32 bias of shape ``[out]``.
    """

    def __init__(self, weight_tensor: NintTensor, bias: Optional[np.ndarray] = None) -> None:
        self.weight_tensor = weight_tensor
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        self._w: Optional[np.ndarray] = None

    @property
    def weight(self) -> np.ndarray:
        """Full-precision weights that are lazily dequantized and cached."""
        if self._w is None:
            self._w = dequantize(self.weight_tensor)
        return self._w

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
