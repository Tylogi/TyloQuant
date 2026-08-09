"""SwiGLU FFN：``down(silu(gate(x)) * up(x))``。

Gate, up, and down are all :class:`~mfq.runtime.linear.NintLinear`. Their Hadamard-product coupling is exactly
the product-error objective discussed in development documentation v2 sections 1.10 and 2.5; the calibrator should weight this structure.
"""

from __future__ import annotations

import numpy as np

from mfq.runtime.linear import NintLinear


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class SwiGLUFFN:
    """SwiGLU FFN composed of three NintLinear layers."""

    def __init__(self, gate: NintLinear, up: NintLinear, down: NintLinear) -> None:
        self.gate = gate
        self.up = up
        self.down = down

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = self.gate(x)            # [..., inter]
        u = self.up(x)              # [..., inter]
        return self.down(silu(a) * u)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
