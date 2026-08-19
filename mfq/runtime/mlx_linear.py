"""MLX runtime primitives for MFQ on Apple silicon.

This module is deliberately independent from the existing Torch/CUDA runtime.
It keeps MFQ NINT, NVQ, NPQ, and NEPQ weights packed in Metal memory and
executes them with the custom kernels from :mod:`mfq.kernels.metal`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's MLX runtime requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats import io
from mfq.formats.io import MfqTensor
from mfq.formats.mx import MxTensor
from mfq.formats.nepq import NepqTensor
from mfq.formats.nint import NintTensor
from mfq.formats.nint8_zero import Nint8ZeroTensor
from mfq.formats.npq0_l import Npq0LTensor
from mfq.formats.npq0_s import Npq0STensor
from mfq.formats.nvq import NvqJscTensor, NvqTensor
from mfq.formats.nvq1_l import Nvq1LTensor
from mfq.formats.nvq1_s import Nvq1STensor
from mfq.formats.tpq import TpqInt4Tensor, TpqPqTensor
from mfq.kernels.metal.grouped_linear import (
    MetalLinearGroupWeight,
    PackedLinearWeight,
    grouped_linear_matmul,
)
from mfq.kernels.metal.mx import (
    MetalMxWeight,
    mx_dequantize,
    mx_embedding,
    mx_matmul,
)
from mfq.kernels.metal.nint import (
    MetalNintWeight,
    nint_embedding,
    nint_matmul,
    nint_swiglu,
)
from mfq.kernels.metal.nint8_zero import (
    MetalNint8ZeroWeight,
    nint8_zero_dequantize,
    nint8_zero_embedding,
    nint8_zero_matmul,
)
from mfq.kernels.metal.ops import silu_mul
from mfq.kernels.metal.tpq import (
    MetalTpqInt4Weight,
    MetalTpqPqWeight,
)
from mfq.kernels.metal.vq import (
    MetalVqWeight,
    vq_swiglu,
    vq_swiglu_compatible,
)
from mfq.runtime.mlx_tpq import (
    MlxTpqInt4Embedding,
    MlxTpqInt4Linear,
    MlxTpqPqLinear,
)
from mfq.runtime.mlx_vq import MlxVqEmbedding, MlxVqLinear

_VQ_TENSOR_TYPES = (
    NvqTensor,
    NvqJscTensor,
    Nvq1LTensor,
    Nvq1STensor,
    Npq0LTensor,
    Npq0STensor,
    NepqTensor,
)
_VQ_DTYPES = {
    "NVQ2",
    "NVQ2J",
    "NVQ2J-L",
    "NVQ2J-XL",
    "NVQ3",
    "NVQ3J",
    "NVQ3J-512",
    "NVQ3J-L",
    "NVQ1-L",
    "NVQ1-S",
    "NPQ0-L",
    "NPQ0-S",
    "NEPQ0-L",
    "NEPQ0-S",
    "NEPQ1-L",
    "NEPQ1-S",
    "NEPQ0-A",
    "NEPQ1-A",
}


def mlx_dense_array(
    tensor: np.ndarray,
    *,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Create an MLX dense array while preserving tagged MFQ BF16 bits."""

    if io.is_bfloat16_array(tensor):
        bits = mx.array(np.ascontiguousarray(tensor, dtype=np.uint16))
        value = bits.view(mx.bfloat16)
    else:
        source_dtype = mx.float32 if tensor.dtype == np.float32 else mx.float16
        value = mx.array(np.ascontiguousarray(tensor)).astype(source_dtype)
    return value if dtype is None or value.dtype == dtype else value.astype(dtype)


class MlxNintLinear:
    """A packed NINT linear layer executed by a custom Metal kernel."""

    def __init__(self, tensor: NintTensor) -> None:
        self.packed_weight = MetalNintWeight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(cls, weight: MetalNintWeight) -> MlxNintLinear:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    @classmethod
    def from_blob(cls, blob: bytes | memoryview) -> MlxNintLinear:
        return cls.from_packed_weight(MetalNintWeight.from_blob(blob))

    @property
    def packed_nbytes(self) -> int:
        return self.packed_weight.packed_nbytes

    @property
    def weight(self) -> mx.array:
        """Materialize the dequantized weight for debugging, not deployment."""

        ids = mx.arange(self.packed_weight.out, dtype=mx.int32)
        return nint_embedding(self.packed_weight, ids, dtype=mx.float32)

    def forward(self, x: mx.array | np.ndarray) -> mx.array:
        return nint_matmul(self.packed_weight, x)

    def __call__(self, x: mx.array | np.ndarray) -> mx.array:
        return self.forward(x)


class MlxDenseLinear:
    """Dense BF16/F16/F32 projection used for unquantized MFQ tensors."""

    def __init__(self, tensor: np.ndarray) -> None:
        if tensor.ndim != 2:
            raise ValueError(f"MlxDenseLinear expects a 2D tensor, got {tensor.shape}")
        self.weight = mlx_dense_array(tensor)

    def forward(self, x: mx.array | np.ndarray) -> mx.array:
        source = x if isinstance(x, mx.array) else mx.array(x)
        return source.astype(self.weight.dtype) @ self.weight.T

    def __call__(self, x: mx.array | np.ndarray) -> mx.array:
        return self.forward(x)


class MlxDenseEmbedding:
    """Dense BF16/F16/F32 embedding for unquantized MFQ model tables."""

    def __init__(self, tensor: np.ndarray) -> None:
        if tensor.ndim != 2:
            raise ValueError(f"MlxDenseEmbedding expects a 2D tensor, got {tensor.shape}")
        self.weight = mlx_dense_array(tensor)

    def forward(self, token_ids: mx.array | np.ndarray) -> mx.array:
        ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
        return self.weight[ids.astype(mx.int32)]

    def __call__(self, token_ids: mx.array | np.ndarray) -> mx.array:
        return self.forward(token_ids)


class MlxMxLinear:
    """Native OCP MXFP4/MXFP8 projection backed by packed Metal kernels."""

    def __init__(self, tensor: MxTensor) -> None:
        self.packed_weight = MetalMxWeight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(cls, weight: MetalMxWeight) -> MlxMxLinear:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    @property
    def packed_nbytes(self) -> int:
        return self.packed_weight.packed_nbytes

    @property
    def weight(self) -> mx.array:
        return mx_dequantize(self.packed_weight)

    def forward(self, x: mx.array | np.ndarray) -> mx.array:
        return mx_matmul(self.packed_weight, x)

    def __call__(self, x: mx.array | np.ndarray) -> mx.array:
        return self.forward(x)


class MlxMxEmbedding:
    """Embedding lookup for native OCP MXFP4/MXFP8 tensors."""

    def __init__(self, tensor: MxTensor) -> None:
        self.packed_weight = MetalMxWeight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(cls, weight: MetalMxWeight) -> MlxMxEmbedding:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    def forward(
        self,
        token_ids: mx.array | np.ndarray,
        *,
        dtype: mx.Dtype = mx.float16,
    ) -> mx.array:
        return mx_embedding(self.packed_weight, token_ids, dtype=dtype)

    def __call__(self, token_ids: mx.array | np.ndarray) -> mx.array:
        return self.forward(token_ids)


class MlxNint8ZeroLinear:
    """Packed GGML-compatible Q8_0 projection."""

    def __init__(self, tensor: Nint8ZeroTensor) -> None:
        self.packed_weight = MetalNint8ZeroWeight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(
        cls,
        weight: MetalNint8ZeroWeight,
    ) -> MlxNint8ZeroLinear:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    @property
    def weight(self) -> mx.array:
        return nint8_zero_dequantize(self.packed_weight)

    def forward(self, x: mx.array | np.ndarray) -> mx.array:
        return nint8_zero_matmul(self.packed_weight, x)

    def __call__(self, x: mx.array | np.ndarray) -> mx.array:
        return self.forward(x)


class MlxNintEmbedding:
    """Embedding lookup that decodes only selected packed NINT rows."""

    def __init__(self, tensor: NintTensor) -> None:
        self.packed_weight = MetalNintWeight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(cls, weight: MetalNintWeight) -> MlxNintEmbedding:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    @classmethod
    def from_blob(cls, blob: bytes | memoryview) -> MlxNintEmbedding:
        return cls.from_packed_weight(MetalNintWeight.from_blob(blob))

    def forward(
        self,
        token_ids: mx.array | np.ndarray,
        *,
        dtype: mx.Dtype = mx.float16,
    ) -> mx.array:
        return nint_embedding(self.packed_weight, token_ids, dtype=dtype)

    def __call__(self, token_ids: mx.array | np.ndarray) -> mx.array:
        return self.forward(token_ids)


class MlxNint8ZeroEmbedding:
    """Embedding lookup that decodes only selected Q8_0 rows."""

    def __init__(self, tensor: Nint8ZeroTensor) -> None:
        self.packed_weight = MetalNint8ZeroWeight.from_tensor(tensor)

    @classmethod
    def from_packed_weight(
        cls,
        weight: MetalNint8ZeroWeight,
    ) -> MlxNint8ZeroEmbedding:
        result = object.__new__(cls)
        result.packed_weight = weight
        return result

    def forward(
        self,
        token_ids: mx.array | np.ndarray,
        *,
        dtype: mx.Dtype = mx.float16,
    ) -> mx.array:
        return nint8_zero_embedding(
            self.packed_weight,
            token_ids,
            dtype=dtype,
        )

    def __call__(self, token_ids: mx.array | np.ndarray) -> mx.array:
        return self.forward(token_ids)


class MlxLinearGroup:
    """Execute multiple packed or dense projections with a shared input."""

    def __init__(
        self,
        tensors: Sequence[
            MfqTensor
            | MlxNintLinear
            | MlxNint8ZeroLinear
            | MlxVqLinear
            | MlxTpqInt4Linear
            | MlxTpqPqLinear
            | MlxMxLinear
            | MlxDenseLinear
        ],
        *,
        grouped_max_rows: int | None = 16,
    ) -> None:
        if len(tensors) < 2:
            raise ValueError("MlxLinearGroup requires at least two tensors")
        self.layers = [_linear(tensor) for tensor in tensors]
        packed: list[PackedLinearWeight] = []
        for layer in self.layers:
            weight = getattr(layer, "packed_weight", None)
            if not isinstance(
                weight,
                (
                    MetalNintWeight,
                    MetalNint8ZeroWeight,
                    MetalVqWeight,
                    MetalTpqInt4Weight,
                    MetalTpqPqWeight,
                ),
            ):
                break
            packed.append(weight)
        residual_vq = any(
            isinstance(weight, MetalVqWeight)
            and weight.residual_position_bits > 0
            for weight in packed
        )
        self.grouped_weight = (
            MetalLinearGroupWeight.from_weights(tuple(packed))
            if len(packed) == len(self.layers) and not residual_vq
            else None
        )
        if grouped_max_rows is not None and int(grouped_max_rows) <= 0:
            raise ValueError("grouped_max_rows must be positive or None")
        self.grouped_max_rows = None if grouped_max_rows is None else int(grouped_max_rows)

    @property
    def uses_grouped_kernel(self) -> bool:
        return self.grouped_weight is not None

    def forward(self, x: mx.array | np.ndarray) -> tuple[mx.array, ...]:
        if self.grouped_weight is not None:
            source = x if isinstance(x, mx.array) else mx.array(x)
            rows = (
                int(
                    np.prod(
                        tuple(int(value) for value in source.shape[:-1]),
                        dtype=np.int64,
                    )
                )
                if source.ndim > 1
                else 1
            )
            if self.grouped_max_rows is None or rows <= self.grouped_max_rows:
                return grouped_linear_matmul(self.grouped_weight, source)
        return tuple(layer(x) for layer in self.layers)

    def forward_swiglu(self, x: mx.array | np.ndarray) -> mx.array:
        if all(isinstance(layer, MlxNintLinear) for layer in self.layers):
            gate, up = self.layers
            return nint_swiglu(gate.packed_weight, up.packed_weight, x)
        if all(isinstance(layer, MlxVqLinear) for layer in self.layers):
            gate, up = self.layers
            if vq_swiglu_compatible(gate.packed_weight, up.packed_weight):
                return vq_swiglu(gate.packed_weight, up.packed_weight, x)
        gate, up = self.forward(x)
        return silu_mul(gate, up)

    def __call__(self, x: mx.array | np.ndarray) -> tuple[mx.array, ...]:
        return self.forward(x)


class MlxSwiGLUFFN:
    """SwiGLU FFN with an optional independent important-neuron branch."""

    def __init__(
        self,
        gate: MfqTensor | MlxNintLinear | MlxVqLinear | MlxDenseLinear,
        up: MfqTensor | MlxNintLinear | MlxVqLinear | MlxDenseLinear,
        down: MfqTensor | MlxNintLinear | MlxVqLinear | MlxDenseLinear,
        *,
        important_neurons: MlxSwiGLUFFN | None = None,
    ) -> None:
        self.gate_up = MlxLinearGroup((gate, up))
        self.down = _linear(down)
        self.important_neurons = important_neurons

    def forward(self, x: mx.array | np.ndarray) -> mx.array:
        low = self.down(self.gate_up.forward_swiglu(x))
        if self.important_neurons is None:
            return low
        return low + self.important_neurons.forward(x)

    def __call__(self, x: mx.array | np.ndarray) -> mx.array:
        return self.forward(x)


class MlxNintModel:
    """Load an MFQ file and construct Apple-silicon execution primitives."""

    def __init__(self, tensors: Mapping[str, MfqTensor]) -> None:
        self.tensors = tensors

    @classmethod
    def from_mfq(cls, path: str | Path, *, mmap: bool = True) -> MlxNintModel:
        _header, tensors = io.load_mmap(path) if mmap else io.load(path)
        return cls(tensors)

    def linear(
        self,
        name: str,
    ) -> (
        MlxNintLinear
        | MlxNint8ZeroLinear
        | MlxVqLinear
        | MlxTpqInt4Linear
        | MlxTpqPqLinear
        | MlxMxLinear
        | MlxDenseLinear
    ):
        packed_mx = self._packed_mx(name)
        if packed_mx is not None:
            return MlxMxLinear.from_packed_weight(packed_mx)
        packed = self._packed_nint(name)
        if packed is not None:
            return MlxNintLinear.from_packed_weight(packed)
        packed_q8 = self._packed_nint8_zero(name)
        if packed_q8 is not None:
            return MlxNint8ZeroLinear.from_packed_weight(packed_q8)
        packed_vq = self._packed_vq(name)
        if packed_vq is not None:
            return MlxVqLinear.from_packed_weight(packed_vq)
        return _linear(self._require(name))

    def embedding(
        self,
        name: str,
    ) -> (
        MlxNintEmbedding
        | MlxNint8ZeroEmbedding
        | MlxVqEmbedding
        | MlxTpqInt4Embedding
        | MlxMxEmbedding
        | MlxDenseEmbedding
    ):
        packed_mx = self._packed_mx(name)
        if packed_mx is not None:
            return MlxMxEmbedding.from_packed_weight(packed_mx)
        packed = self._packed_nint(name)
        if packed is not None:
            return MlxNintEmbedding.from_packed_weight(packed)
        packed_q8 = self._packed_nint8_zero(name)
        if packed_q8 is not None:
            return MlxNint8ZeroEmbedding.from_packed_weight(packed_q8)
        packed_vq = self._packed_vq(name)
        if packed_vq is not None:
            return MlxVqEmbedding.from_packed_weight(packed_vq)
        tensor = self._require(name)
        if isinstance(tensor, NintTensor):
            return MlxNintEmbedding(tensor)
        if isinstance(tensor, Nint8ZeroTensor):
            return MlxNint8ZeroEmbedding(tensor)
        if isinstance(tensor, _VQ_TENSOR_TYPES):
            return MlxVqEmbedding(tensor)
        if isinstance(tensor, TpqInt4Tensor):
            return MlxTpqInt4Embedding(tensor)
        if isinstance(tensor, MxTensor):
            return MlxMxEmbedding(tensor)
        if isinstance(tensor, np.ndarray):
            return MlxDenseEmbedding(tensor)
        raise TypeError(f"tensor {name!r} is not a packed embedding weight")

    def ffn(self, gate_name: str, up_name: str, down_name: str) -> MlxSwiGLUFFN:
        high_names = tuple(
            name + ".in_high"
            for name in (gate_name, up_name, down_name)
        )
        present = tuple(name in self.tensors for name in high_names)
        if any(present) and not all(present):
            raise ValueError(
                "important-neuron FFN requires matching "
                "gate/up/down .in_high records"
            )
        high = (
            MlxSwiGLUFFN(*(self.linear(name) for name in high_names))
            if all(present)
            else None
        )
        return MlxSwiGLUFFN(
            self.linear(gate_name),
            self.linear(up_name),
            self.linear(down_name),
            important_neurons=high,
        )

    def close(self) -> None:
        close = getattr(self.tensors, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> MlxNintModel:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _packed_nint(self, name: str) -> MetalNintWeight | None:
        if not isinstance(self.tensors, io.MMapTensorStore):
            return None
        if name not in self.tensors.records:
            raise KeyError(f"tensor {name!r} is not present in the MFQ model")
        record = self.tensors.records[name]
        if not (record.dtype.startswith("NINT") and record.dtype[4:].isdigit()):
            return None
        return MetalNintWeight.from_blob(self.tensors.read_blob(name))

    def _packed_mx(self, name: str) -> MetalMxWeight | None:
        if not isinstance(self.tensors, io.MMapTensorStore):
            return None
        if name not in self.tensors.records:
            raise KeyError(f"tensor {name!r} is not present in the MFQ model")
        record = self.tensors.records[name]
        if record.dtype not in {"MXFP4", "MXFP8"}:
            return None
        return MetalMxWeight.from_blob(record.dtype, self.tensors.read_blob(name))

    def _packed_nint8_zero(
        self,
        name: str,
    ) -> MetalNint8ZeroWeight | None:
        if not isinstance(self.tensors, io.MMapTensorStore):
            return None
        if name not in self.tensors.records:
            raise KeyError(f"tensor {name!r} is not present in the MFQ model")
        record = self.tensors.records[name]
        if record.dtype != "NINT8-0":
            return None
        return MetalNint8ZeroWeight.from_blob(self.tensors.read_blob(name))

    def _packed_vq(self, name: str) -> MetalVqWeight | None:
        if not isinstance(self.tensors, io.MMapTensorStore):
            return None
        if name not in self.tensors.records:
            raise KeyError(f"tensor {name!r} is not present in the MFQ model")
        record = self.tensors.records[name]
        if record.dtype not in _VQ_DTYPES:
            return None
        return MetalVqWeight.from_blob(record.dtype, self.tensors.read_blob(name))

    def _require(self, name: str) -> MfqTensor:
        if name not in self.tensors:
            raise KeyError(f"tensor {name!r} is not present in the MFQ model")
        return self.tensors[name]


def _unsupported_tensor(tensor: object):
    raise TypeError(
        "the MLX runtime supports MXFP4, MXFP8, NINT, NINT8-0, NVQ, NPQ, "
        "NEPQ, TPQ, and dense tensors; "
        f"received {type(tensor).__name__}"
    )


def _linear(
    tensor: (
        MfqTensor
        | MlxNintLinear
        | MlxNint8ZeroLinear
        | MlxVqLinear
        | MlxTpqInt4Linear
        | MlxTpqPqLinear
        | MlxMxLinear
        | MlxDenseLinear
    ),
) -> (
    MlxNintLinear
    | MlxNint8ZeroLinear
    | MlxVqLinear
    | MlxTpqInt4Linear
    | MlxTpqPqLinear
    | MlxMxLinear
    | MlxDenseLinear
):
    if isinstance(
        tensor,
        (
            MlxNintLinear,
            MlxNint8ZeroLinear,
            MlxVqLinear,
            MlxTpqInt4Linear,
            MlxTpqPqLinear,
            MlxMxLinear,
            MlxDenseLinear,
        ),
    ):
        return tensor
    if isinstance(tensor, NintTensor):
        return MlxNintLinear(tensor)
    if isinstance(tensor, Nint8ZeroTensor):
        return MlxNint8ZeroLinear(tensor)
    if isinstance(tensor, _VQ_TENSOR_TYPES):
        return MlxVqLinear(tensor)
    if isinstance(tensor, TpqInt4Tensor):
        return MlxTpqInt4Linear(tensor)
    if isinstance(tensor, TpqPqTensor):
        return MlxTpqPqLinear(tensor)
    if isinstance(tensor, MxTensor):
        return MlxMxLinear(tensor)
    if isinstance(tensor, np.ndarray):
        return MlxDenseLinear(tensor)
    return _unsupported_tensor(tensor)


__all__ = [
    "MlxTpqInt4Embedding",
    "MlxTpqInt4Linear",
    "MlxTpqPqLinear",
    "MlxDenseEmbedding",
    "MlxDenseLinear",
    "MlxLinearGroup",
    "MlxMxEmbedding",
    "MlxMxLinear",
    "MlxNintEmbedding",
    "MlxNint8ZeroEmbedding",
    "MlxNint8ZeroLinear",
    "MlxNintLinear",
    "MlxNintModel",
    "MlxSwiGLUFFN",
    "MlxVqLinear",
    "MlxVqEmbedding",
    "mlx_dense_array",
]
