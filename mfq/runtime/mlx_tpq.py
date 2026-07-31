"""MLX modules for TPQ packed matrices."""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's MLX runtime requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats.tpq import CccpInt4Tensor, CccpPqTensor
from mfq.kernels.metal.tpq import (
    MetalCccpInt4Weight,
    MetalCccpPqWeight,
    cccp_int4_dequantize,
    cccp_int4_embedding,
    cccp_int4_matmul,
    cccp_pq_dequantize,
    cccp_pq_matmul,
)


class MlxCccpInt4Linear:
    """TPQ2 symmetric int4-g64 dense projection."""

    def __init__(self, tensor: CccpInt4Tensor) -> None:
        self.packed_weight = MetalCccpInt4Weight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(
        cls,
        weight: MetalCccpInt4Weight,
    ) -> MlxCccpInt4Linear:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    @property
    def packed_nbytes(self) -> int:
        return self.packed_weight.packed_nbytes

    @property
    def weight(self) -> mx.array:
        return cccp_int4_dequantize(self.packed_weight)

    def forward(self, x: mx.array | np.ndarray) -> mx.array:
        return cccp_int4_matmul(self.packed_weight, x)

    def __call__(self, x: mx.array | np.ndarray) -> mx.array:
        return self.forward(x)


class MlxCccpInt4Embedding:
    """TPQ2 int4 embedding that decodes only requested rows."""

    def __init__(self, tensor: CccpInt4Tensor) -> None:
        self.packed_weight = MetalCccpInt4Weight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(
        cls,
        weight: MetalCccpInt4Weight,
    ) -> MlxCccpInt4Embedding:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    def forward(
        self,
        token_ids: mx.array | np.ndarray,
        *,
        dtype: mx.Dtype = mx.float16,
    ) -> mx.array:
        return cccp_int4_embedding(
            self.packed_weight,
            token_ids,
            dtype=dtype,
        )

    def __call__(self, token_ids: mx.array | np.ndarray) -> mx.array:
        return self.forward(token_ids)


class MlxCccpPqLinear:
    """TPQ2 learned product-VQ projection."""

    def __init__(self, tensor: CccpPqTensor) -> None:
        self.packed_weight = MetalCccpPqWeight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(
        cls,
        weight: MetalCccpPqWeight,
    ) -> MlxCccpPqLinear:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    @property
    def packed_nbytes(self) -> int:
        return self.packed_weight.packed_nbytes

    @property
    def weight(self) -> mx.array:
        return cccp_pq_dequantize(self.packed_weight)

    def forward(self, x: mx.array | np.ndarray) -> mx.array:
        return cccp_pq_matmul(self.packed_weight, x)

    def __call__(self, x: mx.array | np.ndarray) -> mx.array:
        return self.forward(x)


MlxTpqInt4Embedding = MlxCccpInt4Embedding
MlxTpqInt4Linear = MlxCccpInt4Linear
MlxTpqPqLinear = MlxCccpPqLinear


__all__ = [
    "MlxTpqInt4Embedding",
    "MlxTpqInt4Linear",
    "MlxTpqPqLinear",
    "MlxCccpInt4Embedding",
    "MlxCccpInt4Linear",
    "MlxCccpPqLinear",
]
