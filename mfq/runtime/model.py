"""NintModel：从 ``.mfq`` 加载量化权重，按需构建 NintLinear。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import numpy as np

from mfq.formats import io
from mfq.quantize.nint_quant import NintTensor
from mfq.runtime.linear import NintLinear


class NintModel:
    """持有一组命名 NintTensor，按名字构建 :class:`NintLinear`。"""

    def __init__(self, tensors: Mapping[str, NintTensor]) -> None:
        self.tensors = tensors

    @classmethod
    def from_mfq(cls, path: str | Path, mmap: bool = False) -> "NintModel":
        """从 ``.mfq`` 文件加载（见 :mod:`mfq.formats.io`）。"""
        _header, tensors = io.load_mmap(path) if mmap else io.load(path)
        return cls(tensors)

    def linear(self, name: str, bias: Optional[np.ndarray] = None) -> NintLinear:
        if name not in self.tensors:
            raise KeyError(f"tensor {name!r} 不在模型中；已有: {list(self.tensors)}")
        return NintLinear(self.tensors[name], bias=bias)
