"""MLX packed NVQ, NPQ, and NEPQ linear runtime."""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's MLX runtime requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.kernels.metal.vq import (
    MetalVqWeight,
    VqTensor,
    vq_dequantize,
    vq_embedding,
    vq_matmul,
)


class MlxVqLinear:
    """Packed NVQ/NPQ/NEPQ projection with automatic GEMV/MMQ/GEMM dispatch."""

    def __init__(self, tensor: VqTensor) -> None:
        self.packed_weight = MetalVqWeight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(cls, weight: MetalVqWeight) -> MlxVqLinear:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    @classmethod
    def from_blob(cls, dtype: str, blob: bytes | memoryview) -> MlxVqLinear:
        return cls.from_packed_weight(MetalVqWeight.from_blob(dtype, blob))

    @property
    def packed_nbytes(self) -> int:
        return self.packed_weight.packed_nbytes

    @property
    def weight(self) -> mx.array:
        return vq_dequantize(self.packed_weight)

    def forward(self, x: mx.array | np.ndarray) -> mx.array:
        return vq_matmul(self.packed_weight, x)

    def __call__(self, x: mx.array | np.ndarray) -> mx.array:
        return self.forward(x)


class MlxVqEmbedding:
    """Embedding lookup that decodes only selected packed NVQ/NPQ rows."""

    def __init__(self, tensor: VqTensor) -> None:
        self.packed_weight = MetalVqWeight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(cls, weight: MetalVqWeight) -> MlxVqEmbedding:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    @classmethod
    def from_blob(
        cls,
        dtype: str,
        blob: bytes | memoryview,
    ) -> MlxVqEmbedding:
        return cls.from_packed_weight(MetalVqWeight.from_blob(dtype, blob))

    def forward(
        self,
        token_ids: mx.array | np.ndarray,
        *,
        dtype: mx.Dtype = mx.float16,
    ) -> mx.array:
        return vq_embedding(self.packed_weight, token_ids, dtype=dtype)

    def __call__(self, token_ids: mx.array | np.ndarray) -> mx.array:
        return self.forward(token_ids)


__all__ = ["MlxVqEmbedding", "MlxVqLinear"]
