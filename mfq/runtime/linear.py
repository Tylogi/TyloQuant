"""NintLinear：权重为 :class:`~mfq.quantize.nint_quant.NintTensor` 的线性层。

惰性反量化（首次访问时算并缓存）+ matmul。numpy 参考实现；真实 kernel 时代
dequant+matmul 会融合（在 :mod:`mfq.runtime.dequantize` 注册的 backend 里）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mfq.quantize.nint_quant import NintTensor
from mfq.runtime.dequantize import dequantize


class NintLinear:
    """``y = x · Wᵀ (+ b)``，W 为 NintTensor（按 neuron 轴 quantize 时的 axis=0）。

    Attributes:
        weight_tensor: 量化权重（shape 还原后为 ``[out, in]``）。
        bias: 可选偏置（float32，``[out]``）。
    """

    def __init__(self, weight_tensor: NintTensor, bias: Optional[np.ndarray] = None) -> None:
        self.weight_tensor = weight_tensor
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        self._w: Optional[np.ndarray] = None

    @property
    def weight(self) -> np.ndarray:
        """惰性反量化并缓存的全精度权重。"""
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
