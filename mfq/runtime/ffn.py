"""SwiGLU FFN：``down(silu(gate(x)) * up(x))``。

gate / up / down 都是 :class:`~mfq.runtime.linear.NintLinear`。Hadamard 积耦合正是
开发文档 v2 §1.10、§2.5 讨论的「乘积误差」目标所在——校准器应按此结构加权。
"""

from __future__ import annotations

import numpy as np

from mfq.runtime.linear import NintLinear


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class SwiGLUFFN:
    """三层 NintLinear 组成的 SwiGLU FFN。"""

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
