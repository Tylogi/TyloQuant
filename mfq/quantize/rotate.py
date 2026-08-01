"""Hadamard rotation。

随机化的正交 Hadamard 变换可以让张量分布更接近均匀、降低离群通道，
从而在超低精度下显著提升可量化性。

使用场景（见开发文档）：
- 4bit 以下的 INT 层（用少量速度换精度）；
- 精度关键的 expert / tensor 额外叠加旋转；
- KV 量化的旋转是「免费的」（不占用权重路径），故 KV INT 始终旋转。
"""

from __future__ import annotations

import numpy as np


def hadamard_matrix(n: int) -> np.ndarray:
    """返回 n×n 的 Hadamard 矩阵（n 需为 2 的幂）。"""

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n 必须是 2 的幂，得到 {n}")
    h = np.array([[1]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.vstack([np.hstack([h, h]), np.hstack([h, -h])])
    return h / np.sqrt(n)


def rotate(weight: np.ndarray, axis: int = -1) -> np.ndarray:
    """沿 ``axis`` 对权重施加 Hadamard rotation。"""

    raise NotImplementedError
