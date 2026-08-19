"""Dispatch NINT dequantization by profile.

In the NumPy reference implementation, all profiles use :func:`mfq.quantize.nint_quant.dequantize`.
After real hardware kernels (CUDA / Metal) are implemented, use :func:`register_backend` to register
specialized dequantization by ``profile_label`` (for example ``NINT4-24``); this module dispatches automatically.

Dequantization arithmetic is identical for all profiles (``d_eff*q - m_eff``); kernel tiling varies only with ``gs``.
``k`` and ``bits`` are baked into scale and q width and are invisible to the kernel (development documentation v2 section 2.1).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mfq.formats.nint import NintTensor
from mfq.quantize import nint_quant

_BackendFn = Callable[[NintTensor], np.ndarray]
_BACKENDS: dict[str, _BackendFn] = {}
_DEFAULT: _BackendFn = nint_quant.dequantize


def register_backend(profile_label: str, fn: _BackendFn) -> None:
    """Register a specialized dequantization kernel for a profile such as ``NINT4-24``."""
    _BACKENDS[profile_label] = fn


def clear_backends() -> None:
    """Clear all registered hardware backends and return to the NumPy default."""
    _BACKENDS.clear()


def dequantize(tensor: NintTensor) -> np.ndarray:
    """Dispatch dequantization by ``tensor.spec.profile_label``; use the NumPy default when unregistered."""
    fn = _BACKENDS.get(tensor.spec.profile_label, _DEFAULT)
    return fn(tensor)
