"""按 profile 分派 NINT 反量化。

numpy 参考实现里所有 profile 走统一的 :func:`mfq.quantize.nint_quant.dequantize`；
真实硬件 kernel（CUDA / Metal）实现后，用 :func:`register_backend` 按
``profile_label``（如 ``NINT4-24``）注册专用反量化，本模块自动分派。

dequant 算术对所有 profile 相同（``d_eff·q − m_eff``），kernel 只随 ``gs`` 变化 tiling；
``k`` 与 ``bits`` 已烘焙进 scale / q 宽度，对 kernel 不可见（开发文档 v2 §2.1）。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mfq.quantize import nint_quant
from mfq.quantize.nint_quant import NintTensor

_BackendFn = Callable[[NintTensor], np.ndarray]
_BACKENDS: dict[str, _BackendFn] = {}
_DEFAULT: _BackendFn = nint_quant.dequantize


def register_backend(profile_label: str, fn: _BackendFn) -> None:
    """为某 profile（如 ``NINT4-24``）注册专用反量化 kernel。"""
    _BACKENDS[profile_label] = fn


def clear_backends() -> None:
    """清除所有已注册的硬件 backend（回到 numpy 默认）。"""
    _BACKENDS.clear()


def dequantize(tensor: NintTensor) -> np.ndarray:
    """按 ``tensor.spec.profile_label`` 分派反量化；未注册则走 numpy 默认。"""
    fn = _BACKENDS.get(tensor.spec.profile_label, _DEFAULT)
    return fn(tensor)
