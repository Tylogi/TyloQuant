"""MLX modules for TPQ packed matrices."""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's MLX runtime requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats.tpq import TpqInt4Tensor, TpqPqTensor
from mfq.kernels.metal.tpq import (
    MetalTpqInt4Weight,
    MetalTpqPqWeight,
    tpq_int4_dequantize,
    tpq_int4_embedding,
    tpq_int4_matmul,
    tpq_pq_dequantize,
    tpq_pq_matmul,
)


class MlxTpqInt4Linear:
    """TPQ2 symmetric int4-g64 dense projection."""

    def __init__(self, tensor: TpqInt4Tensor) -> None:
        self.packed_weight = MetalTpqInt4Weight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(
        cls,
        weight: MetalTpqInt4Weight,
    ) -> MlxTpqInt4Linear:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    @property
    def packed_nbytes(self) -> int:
        return self.packed_weight.packed_nbytes

    @property
    def weight(self) -> mx.array:
        return tpq_int4_dequantize(self.packed_weight)

    def forward(self, x: mx.array | np.ndarray) -> mx.array:
        return tpq_int4_matmul(self.packed_weight, x)

    def __call__(self, x: mx.array | np.ndarray) -> mx.array:
        return self.forward(x)


class MlxTpqInt4Embedding:
    """TPQ2 int4 embedding that decodes only requested rows."""

    def __init__(self, tensor: TpqInt4Tensor) -> None:
        self.packed_weight = MetalTpqInt4Weight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(
        cls,
        weight: MetalTpqInt4Weight,
    ) -> MlxTpqInt4Embedding:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    def forward(
        self,
        token_ids: mx.array | np.ndarray,
        *,
        dtype: mx.Dtype = mx.float16,
    ) -> mx.array:
        return tpq_int4_embedding(
            self.packed_weight,
            token_ids,
            dtype=dtype,
        )

    def __call__(self, token_ids: mx.array | np.ndarray) -> mx.array:
        return self.forward(token_ids)


class MlxTpqPqLinear:
    """TPQ2 learned product-VQ projection."""

    def __init__(self, tensor: TpqPqTensor) -> None:
        self.packed_weight = MetalTpqPqWeight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(
        cls,
        weight: MetalTpqPqWeight,
    ) -> MlxTpqPqLinear:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    @property
    def packed_nbytes(self) -> int:
        return self.packed_weight.packed_nbytes

    @property
    def weight(self) -> mx.array:
        return tpq_pq_dequantize(self.packed_weight)

    def forward(self, x: mx.array | np.ndarray) -> mx.array:
        return tpq_pq_matmul(self.packed_weight, x)

    def __call__(self, x: mx.array | np.ndarray) -> mx.array:
        return self.forward(x)


__all__ = [
    "MlxTpqInt4Embedding",
    "MlxTpqInt4Linear",
    "MlxTpqPqLinear",
]
