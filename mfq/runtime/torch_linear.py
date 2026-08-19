"""Torch GPU runtime: keep weights compressed on the GPU and use llama.cpp-style decomposed matmul in forward passes.

This complements the NumPy reference in :mod:`mfq.runtime.linear`; this module uses the GPU. Usage::

    model = TorchNintModel.from_mfq("m.mfq", device="cuda")
    gate = model.linear("blk.0.gate")
    y = gate(x_cuda)        # x_cuda: torch.Tensor on cuda

Import explicitly because this module imports torch, a heavy dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from mfq.formats import io
from mfq.formats.io import MfqTensor
from mfq.formats.mx import MxTensor
from mfq.formats.nint import NintSpec, NintTensor
from mfq.formats.nint8_zero import Nint8ZeroTensor
from mfq.formats.npq0_l import Npq0LTensor
from mfq.formats.npq0_s import Npq0STensor
from mfq.formats.nvq import NvqJscTensor, NvqTensor
from mfq.formats.nvq1_l import Nvq1LTensor
from mfq.formats.nvq1_s import Nvq1STensor
from mfq.formats.tpq import TpqInt4Tensor, TpqPqTensor
from mfq.kernels import torch_backend
from mfq.kernels.cuda.embedding import nint_embedding
from mfq.kernels.cuda.mx_matmul import mx_dequantize, mx_embedding, mx_matmul, to_gpu_mx
from mfq.kernels.cuda.nint8_zero_matmul import (
    nint8_zero_dequantize,
    nint8_zero_embedding,
    nint8_zero_matmul,
    to_gpu_nint8_zero,
)
from mfq.kernels.cuda.nint_matmul import nint_matmul, nint_matmul_input_mul
from mfq.kernels.cuda.nvq_matmul import (
    nvq_dequantize,
    nvq_embedding,
    nvq_ffn_swiglu_down,
    nvq_matmul,
    nvq_matmul_input_mul,
    nvq_matmul_multi2,
    nvq_matmul_swiglu,
    to_gpu_nvq,
)
from mfq.kernels.cuda.tpq_matmul import (
    to_gpu_tpq,
    tpq_dequantize,
    tpq_embedding,
    tpq_matmul,
)

TensorMapping = Mapping[str, MfqTensor]
NvqAnyTensor = NvqTensor | NvqJscTensor | Npq0LTensor | Npq0STensor | Nvq1LTensor | Nvq1STensor
TpqTensor = TpqInt4Tensor | TpqPqTensor
QuantizedTensor = NintTensor | Nint8ZeroTensor | NvqAnyTensor | MxTensor | TpqTensor


def _device_guard(device: str | torch.device):
    value = torch.device(device)
    return torch.cuda.device(value) if value.type == "cuda" else nullcontext()


def is_nvq_tensor(tensor: object) -> bool:
    return isinstance(
        tensor,
        (NvqTensor, NvqJscTensor, Npq0LTensor, Npq0STensor, Nvq1LTensor, Nvq1STensor),
    )


def is_quantized_tensor(tensor: object) -> bool:
    return isinstance(
        tensor,
        (NintTensor, Nint8ZeroTensor, MxTensor, TpqInt4Tensor, TpqPqTensor),
    ) or is_nvq_tensor(tensor)


class TorchNintLinear:
    """GPU linear layer with NintTensor weights; forward uses decomposed matmul without resident fp16 weights."""

    def __init__(self, tensor: NintTensor, device: str | torch.device = "cuda") -> None:
        self.g = torch_backend.to_gpu(tensor, device)
        self.device = device

    @staticmethod
    def deploy_arrays(tensor: NintTensor) -> dict[str, np.ndarray]:
        return torch_backend.nint_deploy_arrays(tensor)

    @classmethod
    def from_deploy_arrays(
        cls,
        arrays: Mapping[str, np.ndarray],
        *,
        spec: NintSpec,
        shape: Sequence[int],
        axis: int,
        neuron_len: int,
        device: str | torch.device = "cuda",
    ) -> TorchNintLinear:
        result = object.__new__(cls)
        result.g = torch_backend.nint_deploy_to_gpu(
            arrays,
            bits=spec.bits,
            groupsize=spec.groupsize,
            neuron_len=neuron_len,
            shape=shape,
            axis=axis,
            device=device,
        )
        result.device = device
        return result

    def shared_weights_clone(self) -> TorchNintLinear:
        """Share immutable packed weights while keeping execution workspaces private."""

        result = object.__new__(TorchNintLinear)
        result.g = dict(self.g)
        result.g.pop("_workspace", None)
        result.g.pop("_argmax_workspace", None)
        result.device = self.device
        return result

    def row_slice(self, ranges: Sequence[tuple[int, int]]) -> TorchNintLinear:
        """Return a packed row shard without materializing the dense weight."""

        if int(self.g["axis"]) != 0 or len(self.g["shape"]) != 2:
            raise ValueError("packed NINT row slicing requires an axis-0 matrix")
        normalized = tuple((int(start), int(end)) for start, end in ranges)
        if not normalized or any(
            start < 0 or end <= start or end > int(self.g["out"])
            for start, end in normalized
        ):
            raise ValueError("invalid packed NINT row ranges")
        result = object.__new__(TorchNintLinear)
        result.g = dict(self.g)
        for key in ("q_packed", "sub_scale", "sub_min", "neuron_scale", "neuron_min"):
            result.g[key] = torch.cat(
                [self.g[key][start:end] for start, end in normalized], dim=0
            ).contiguous()
        for key in tuple(result.g):
            if key.startswith("_") or key in {
                "q",
                "eff_pair_h",
                "d_eff",
                "m_eff",
                "d_eff_h",
                "m_eff_h",
                "q_mmq_packed",
                "sub_scale_mmq",
                "sub_min_mmq",
                "d_eff_mmq",
                "m_eff_mmq",
            }:
                result.g.pop(key, None)
        rows = sum(end - start for start, end in normalized)
        result.g["out"] = rows
        result.g["shape"] = (rows, int(self.g["shape"][1]))
        result.device = self.device
        return result

    def row_range(self, start: int, end: int) -> TorchNintLinear:
        """Return one contiguous packed row range as zero-copy tensor views."""

        if int(self.g["axis"]) != 0 or len(self.g["shape"]) != 2:
            raise ValueError("packed NINT row ranges require an axis-0 matrix")
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > int(self.g["out"]):
            raise ValueError("invalid packed NINT row range")
        result = object.__new__(TorchNintLinear)
        result.g = dict(self.g)
        for key in ("q_packed", "sub_scale", "sub_min", "neuron_scale", "neuron_min"):
            result.g[key] = self.g[key][start:end]
        for key in tuple(result.g):
            if key.startswith("_") or key in {
                "q",
                "eff_pair_h",
                "d_eff",
                "m_eff",
                "d_eff_h",
                "m_eff_h",
                "q_mmq_packed",
                "sub_scale_mmq",
                "sub_min_mmq",
                "d_eff_mmq",
                "m_eff_mmq",
            }:
                result.g.pop(key, None)
        result.g["out"] = end - start
        result.g["shape"] = (end - start, int(self.g["shape"][1]))
        result.device = self.device
        return result

    @property
    def weight(self) -> torch.Tensor:
        """Fully dequantize fp16 weights; callers needing a resident cache should retrieve and retain them once."""
        with _device_guard(self.device):
            return torch_backend.dequantize(self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device)
        x2 = x.reshape(-1, x.shape[-1])
        device = torch.device(self.device)
        with _device_guard(device):
            y = (
                nint_matmul(self.g, x2)
                if device.type == "cuda"
                else torch_backend.matmul(self.g, x2)
            )
        return y.reshape(*x.shape[:-1], y.shape[-1])

    def forward_input_mul(
        self, x: torch.Tensor, gate: torch.Tensor, activation: str
    ) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device)
        gate = torch.as_tensor(gate, device=self.device)
        if x.shape != gate.shape:
            raise ValueError(
                f"x and gate must have the same shape, got {tuple(x.shape)} and {tuple(gate.shape)}"
            )
        x2 = x.reshape(-1, x.shape[-1])
        gate2 = gate.reshape(-1, gate.shape[-1])
        if torch.device(self.device).type == "cuda":
            with _device_guard(self.device):
                y = nint_matmul_input_mul(self.g, x2, gate2, activation)
        else:
            if activation == "silu":
                value = x2 * torch.nn.functional.silu(gate2)
            elif activation == "sigmoid":
                value = x2 * torch.sigmoid(gate2)
            else:
                raise ValueError(f"unsupported activation: {activation}")
            y = self.forward(value).reshape(-1, int(self.g["out"]))
        return y.reshape(*x.shape[:-1], y.shape[-1])

    def forward_swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return self.forward_input_mul(up, gate, "silu")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TorchNvqLinear:
    """Compact NPQ/NVQ GPU linear layer."""

    def __init__(self, tensor: NvqAnyTensor, device: str | torch.device = "cuda") -> None:
        self.g = to_gpu_nvq(tensor, device)
        self.device = device

    @property
    def weight(self) -> torch.Tensor:
        with _device_guard(self.device):
            return nvq_dequantize(self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _device_guard(self.device):
            return nvq_matmul(self.g, torch.as_tensor(x, device=self.device))

    def forward_input_mul(
        self, x: torch.Tensor, gate: torch.Tensor, activation: str
    ) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device)
        gate = torch.as_tensor(gate, device=self.device)
        if x.shape != gate.shape:
            raise ValueError(
                f"x and gate must have the same shape, got {tuple(x.shape)} and {tuple(gate.shape)}"
            )
        with _device_guard(self.device):
            return nvq_matmul_input_mul(self.g, x, gate, activation)

    def forward_swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return self.forward_input_mul(up, gate, "silu")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TorchMxLinear:
    """Packed MXFP4/MXFP8 GPU linear layer."""

    def __init__(self, tensor: MxTensor, device: str | torch.device = "cuda") -> None:
        self.g = to_gpu_mx(tensor, device)
        self.device = device

    @property
    def weight(self) -> torch.Tensor:
        return mx_dequantize(self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mx_matmul(self.g, x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TorchTpqLinear:
    """Packed TPQ-I4/TPQ-PQ GPU linear layer."""

    def __init__(
        self,
        tensor: TpqTensor,
        device: str | torch.device = "cuda",
    ) -> None:
        self.g = to_gpu_tpq(tensor, device)
        self.device = device

    @property
    def weight(self) -> torch.Tensor:
        return tpq_dequantize(self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tpq_matmul(self.g, x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TorchNint8ZeroLinear:
    """Packed symmetric NINT8-0 GPU linear layer."""

    def __init__(
        self,
        tensor: Nint8ZeroTensor,
        device: str | torch.device = "cuda",
    ) -> None:
        self.g = to_gpu_nint8_zero(tensor, device)
        self.device = device

    @property
    def weight(self) -> torch.Tensor:
        with _device_guard(self.device):
            return nint8_zero_dequantize(self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _device_guard(self.device):
            return nint8_zero_matmul(self.g, x)

    def forward_input_mul(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        activation: str,
    ) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device, dtype=torch.float16)
        gate = torch.as_tensor(gate, device=self.device, dtype=torch.float16)
        if x.shape != gate.shape:
            raise ValueError("NINT8-0 x and gate must have matching shapes")
        if activation == "silu":
            value = x * torch.nn.functional.silu(gate)
        elif activation == "sigmoid":
            value = x * torch.sigmoid(gate)
        else:
            raise ValueError(f"unsupported activation: {activation}")
        return self.forward(value)

    def forward_swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return self.forward_input_mul(up, gate, "silu")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def _torch_dense_tensor(
    tensor: np.ndarray,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if io.is_bfloat16_array(tensor):
        value = torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.uint16)).view(torch.bfloat16)
    else:
        value = torch.from_numpy(np.ascontiguousarray(tensor))
    if dtype is None:
        dtype = (
            torch.bfloat16
            if io.is_bfloat16_array(tensor)
            else torch.float32
            if tensor.dtype == np.float32
            else torch.float16
        )
    return value.to(device=device, dtype=dtype).contiguous()


class TorchDenseLinear:
    """Dense GPU linear layer used for recipe tensors kept as BF16/F16/F32."""

    def __init__(self, tensor: np.ndarray, device: str | torch.device = "cuda") -> None:
        if tensor.ndim != 2:
            raise ValueError(f"TorchDenseLinear expects a 2D tensor, got {tensor.shape}")
        self.weight = _torch_dense_tensor(tensor, device)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device, dtype=self.weight.dtype)
        x2 = x.reshape(-1, x.shape[-1])
        y = x2 @ self.weight.T
        return y.reshape(*x.shape[:-1], y.shape[-1])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def _cat_nint_gpu_dicts(gs: Sequence[dict]) -> dict:
    if not gs:
        raise ValueError("at least one NINT tensor is required")
    first = gs[0]
    for g in gs[1:]:
        for key in ("ng", "gs", "neuron_len", "axis", "bits"):
            if int(g[key]) != int(first[key]):
                raise ValueError(f"cannot fuse NINT tensors with different {key}")
        if tuple(g["shape"][1:]) != tuple(first["shape"][1:]):
            raise ValueError("cannot fuse NINT tensors with different input shape")

    out_sizes = [int(g["out"]) for g in gs]
    fused = dict(first)
    fused.pop("_workspace", None)
    for key in (
        "q",
        "q_packed",
        "sub_scale",
        "sub_min",
        "neuron_scale",
        "neuron_min",
        "eff_pair_h",
        "d_eff",
        "m_eff",
        "d_eff_h",
        "m_eff_h",
    ):
        values = [g.get(key) for g in gs]
        if all(v is not None for v in values):
            fused[key] = torch.cat(values, dim=0).contiguous()
        elif key in fused:
            fused.pop(key, None)
    for key in (
        "q_mmq_packed",
        "q_prefill_u8_mmq",
        "sub_scale_mmq",
        "sub_min_mmq",
        "d_eff_mmq",
        "m_eff_mmq",
    ):
        fused.pop(key, None)
    fused["out"] = sum(out_sizes)
    fused["shape"] = (sum(out_sizes),) + tuple(first["shape"][1:])
    return fused


class TorchNintLinearGroup:
    """Combine multiple NINT linear layers with the same input into one NINT matmul."""

    def __init__(self, tensors: Sequence[NintTensor], device: str | torch.device = "cuda") -> None:
        if len(tensors) < 2:
            raise ValueError("TorchNintLinearGroup requires at least two tensors")
        self.device = device
        self.out_sizes = [int(t.shape[t.axis]) for t in tensors]
        self.g = _cat_nint_gpu_dicts([torch_backend.to_gpu(t, device) for t in tensors])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = torch.as_tensor(x, device=self.device)
        x2 = x.reshape(-1, x.shape[-1])
        y = nint_matmul(self.g, x2)
        ys = y.split(self.out_sizes, dim=-1)
        return tuple(part.reshape(*x.shape[:-1], part.shape[-1]) for part in ys)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.forward(x)


class TorchLinearGroup:
    """Group of linear layers that may mix NINT and dense tensors."""

    def __init__(self, tensors: Sequence[MfqTensor], device: str | torch.device = "cuda") -> None:
        if len(tensors) < 2:
            raise ValueError("TorchLinearGroup requires at least two tensors")
        self.device = device
        self.out_sizes: list[int] | None = None
        self.dense_weight: torch.Tensor | None = None
        self.nvq_pair = False
        if all(isinstance(t, np.ndarray) and t.ndim == 2 for t in tensors):
            arrays = [t for t in tensors if isinstance(t, np.ndarray)]
            dtypes = [
                torch.bfloat16
                if io.is_bfloat16_array(t)
                else torch.float32
                if t.dtype == np.float32
                else torch.float16
                for t in arrays
            ]
            dtype = dtypes[0]
            for value in dtypes[1:]:
                dtype = torch.promote_types(dtype, value)
            self.out_sizes = [int(t.shape[0]) for t in arrays]
            self.dense_weight = torch.cat(
                [_torch_dense_tensor(t, device, dtype=dtype) for t in arrays],
                dim=0,
            ).contiguous()
            self.layers = []
            return
        if all(isinstance(t, NintTensor) for t in tensors):
            try:
                self.layers = [TorchNintLinearGroup(tensors, device)]  # type: ignore[arg-type]
                return
            except ValueError:
                pass
        self.layers = [
            TorchNintLinear(t, device)
            if isinstance(t, NintTensor)
            else TorchNint8ZeroLinear(t, device)
            if isinstance(t, Nint8ZeroTensor)
            else TorchNvqLinear(t, device)
            if is_nvq_tensor(t)
            else TorchMxLinear(t, device)
            if isinstance(t, MxTensor)
            else TorchTpqLinear(t, device)
            if isinstance(t, (TpqInt4Tensor, TpqPqTensor))
            else TorchDenseLinear(t, device)
            for t in tensors
        ]
        if (
            len(self.layers) >= 2
            and isinstance(self.layers[0], TorchNvqLinear)
            and isinstance(self.layers[1], TorchNvqLinear)
        ):
            first, second = self.layers[0].g, self.layers[1].g
            self.nvq_pair = all(
                int(first[key]) == int(second[key]) for key in ("format", "gs", "neuron_len")
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.dense_weight is not None and self.out_sizes is not None:
            x = torch.as_tensor(x, device=self.device, dtype=self.dense_weight.dtype)
            x2 = x.reshape(-1, x.shape[-1])
            y = x2 @ self.dense_weight.T
            ys = y.split(self.out_sizes, dim=-1)
            return tuple(part.reshape(*x.shape[:-1], part.shape[-1]) for part in ys)
        if len(self.layers) == 1 and isinstance(self.layers[0], TorchNintLinearGroup):
            return self.layers[0](x)
        if self.nvq_pair:
            first, second = nvq_matmul_multi2(self.layers[0].g, self.layers[1].g, x)
            return (first, second, *(layer(x) for layer in self.layers[2:]))
        return tuple(layer(x) for layer in self.layers)

    def forward_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        if self.nvq_pair and len(self.layers) == 2:
            return nvq_matmul_swiglu(self.layers[0].g, self.layers[1].g, x)
        gate, up = self.forward(x)
        return torch.nn.functional.silu(gate) * up

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.forward(x)


class TorchNintEmbedding:
    """GPU embedding with NintTensor weights; dequantize only accessed token rows."""

    def __init__(self, tensor: NintTensor, device: str | torch.device = "cuda") -> None:
        self.g = torch_backend.to_gpu(tensor, device)
        self.device = device

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(token_ids, device=self.device, dtype=torch.int64)
        return nint_embedding(self.g, ids)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(token_ids)


class TorchNint8ZeroEmbedding:
    """NINT8-0 embedding that decodes only selected token rows."""

    def __init__(
        self,
        tensor: Nint8ZeroTensor,
        device: str | torch.device = "cuda",
    ) -> None:
        self.g = to_gpu_nint8_zero(tensor, device)
        self.device = device

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return nint8_zero_embedding(self.g, token_ids)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(token_ids)


class TorchNvqEmbedding:
    """NVQ embedding that decodes only selected rows."""

    def __init__(self, tensor: NvqAnyTensor, device: str | torch.device = "cuda") -> None:
        self.g = to_gpu_nvq(tensor, device)
        self.device = device

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return nvq_embedding(
            self.g, torch.as_tensor(token_ids, device=self.device, dtype=torch.int64)
        )

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(token_ids)


class TorchMxEmbedding:
    """Packed MXFP4/MXFP8 embedding that decodes selected rows."""

    def __init__(self, tensor: MxTensor, device: str | torch.device = "cuda") -> None:
        self.g = to_gpu_mx(tensor, device)
        self.device = device

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return mx_embedding(self.g, token_ids)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(token_ids)


class TorchTpqEmbedding:
    """Packed TPQ-I4/TPQ-PQ embedding that decodes selected rows."""

    def __init__(self, tensor: TpqTensor, device: str | torch.device = "cuda") -> None:
        self.g = to_gpu_tpq(tensor, device)
        self.device = device

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return tpq_embedding(self.g, token_ids)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(token_ids)


class TorchSwiGLUFFN:
    """SwiGLU FFN whose gate, up, and down projections are all :class:`TorchNintLinear`."""

    def __init__(
        self,
        gate: TorchNintLinear
        | TorchNint8ZeroLinear
        | TorchNvqLinear
        | TorchMxLinear
        | TorchTpqLinear
        | TorchNintLinearGroup
        | TorchLinearGroup,
        up: TorchNintLinear
        | TorchNint8ZeroLinear
        | TorchNvqLinear
        | TorchMxLinear
        | TorchTpqLinear
        | None,
        down: TorchNintLinear
        | TorchNint8ZeroLinear
        | TorchNvqLinear
        | TorchMxLinear
        | TorchTpqLinear,
    ) -> None:
        self.gate = gate
        self.up = up
        self.down = down

    @classmethod
    def from_tensors(
        cls,
        gate: QuantizedTensor,
        up: QuantizedTensor,
        down: QuantizedTensor,
        device: str | torch.device = "cuda",
    ) -> TorchSwiGLUFFN:
        if isinstance(gate, NintTensor) and isinstance(up, NintTensor):
            gate_up: TorchNintLinearGroup | TorchLinearGroup = TorchNintLinearGroup(
                (gate, up), device
            )
        else:
            gate_up = TorchLinearGroup((gate, up), device)
        down_layer = (
            TorchNintLinear(down, device)
            if isinstance(down, NintTensor)
            else TorchNint8ZeroLinear(down, device)
            if isinstance(down, Nint8ZeroTensor)
            else TorchMxLinear(down, device)
            if isinstance(down, MxTensor)
            else TorchTpqLinear(down, device)
            if isinstance(down, (TpqInt4Tensor, TpqPqTensor))
            else TorchNvqLinear(down, device)
        )
        return cls(gate_up, None, down_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.gate.device, dtype=torch.float16)
        if (
            isinstance(self.gate, TorchLinearGroup)
            and self.gate.nvq_pair
            and len(self.gate.layers) == 2
            and isinstance(self.gate.layers[0], TorchNvqLinear)
            and isinstance(self.gate.layers[1], TorchNvqLinear)
            and isinstance(self.down, TorchNvqLinear)
        ):
            return nvq_ffn_swiglu_down(
                self.gate.layers[0].g,
                self.gate.layers[1].g,
                self.down.g,
                x,
            )
        if isinstance(self.gate, (TorchNintLinearGroup, TorchLinearGroup)):
            gate, up = self.gate(x)
        else:
            if self.up is None:
                raise RuntimeError("SwiGLU up projection is missing")
            gate, up = self.gate(x), self.up(x)
        if isinstance(self.down, (TorchMxLinear, TorchTpqLinear)):
            return self.down(torch.nn.functional.silu(gate) * up)
        return self.down.forward_swiglu(gate, up)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TorchNintModel:
    """Load from ``.mfq`` and construct :class:`TorchNintLinear` layers by name, moving weights to the GPU."""

    def __init__(self, tensors: TensorMapping, device: str | torch.device = "cuda") -> None:
        self.tensors = tensors
        self.device = device

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        device: str | torch.device = "cuda",
        mmap: bool = False,
    ) -> TorchNintModel:
        _header, tensors = io.load_mmap(path) if mmap else io.load(path)
        return cls(tensors, device)

    def linear(
        self, name: str
    ) -> TorchNintLinear | TorchNint8ZeroLinear | TorchNvqLinear | TorchMxLinear | TorchTpqLinear:
        if name not in self.tensors:
            raise KeyError(f"tensor {name!r} 不在模型中；已有: {list(self.tensors)}")
        tensor = self.tensors[name]
        if isinstance(tensor, NintTensor):
            return TorchNintLinear(tensor, self.device)
        if isinstance(tensor, Nint8ZeroTensor):
            return TorchNint8ZeroLinear(tensor, self.device)
        if is_nvq_tensor(tensor):
            return TorchNvqLinear(tensor, self.device)
        if isinstance(tensor, MxTensor):
            return TorchMxLinear(tensor, self.device)
        if isinstance(tensor, (TpqInt4Tensor, TpqPqTensor)):
            return TorchTpqLinear(tensor, self.device)
        raise TypeError(f"tensor {name!r} 不是 NINT/NVQ/MX/TPQ 权重")

    def embedding(
        self, name: str
    ) -> (
        TorchNintEmbedding
        | TorchNint8ZeroEmbedding
        | TorchNvqEmbedding
        | TorchMxEmbedding
        | TorchTpqEmbedding
    ):
        if name not in self.tensors:
            raise KeyError(f"tensor {name!r} 不在模型中；已有: {list(self.tensors)}")
        tensor = self.tensors[name]
        if isinstance(tensor, NintTensor):
            return TorchNintEmbedding(tensor, self.device)
        if isinstance(tensor, Nint8ZeroTensor):
            return TorchNint8ZeroEmbedding(tensor, self.device)
        if is_nvq_tensor(tensor):
            return TorchNvqEmbedding(tensor, self.device)
        if isinstance(tensor, MxTensor):
            return TorchMxEmbedding(tensor, self.device)
        if isinstance(tensor, (TpqInt4Tensor, TpqPqTensor)):
            return TorchTpqEmbedding(tensor, self.device)
        raise TypeError(f"tensor {name!r} 不是 NINT/NVQ/MX/TPQ 权重")

    def dense(self, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if name not in self.tensors:
            raise KeyError(f"tensor {name!r} 不在模型中；已有: {list(self.tensors)}")
        tensor = self.tensors[name]
        if is_quantized_tensor(tensor):
            raise TypeError(f"tensor {name!r} 是量化权重，不是 dense tensor")
        return torch.as_tensor(tensor, device=self.device, dtype=dtype).contiguous()

    def ffn(self, gate_name: str, up_name: str, down_name: str) -> TorchSwiGLUFFN:
        return TorchSwiGLUFFN.from_tensors(
            _require_quantized_mapping(self.tensors, gate_name),
            _require_quantized_mapping(self.tensors, up_name),
            _require_quantized_mapping(self.tensors, down_name),
            self.device,
        )


def _require_quantized_mapping(tensors: TensorMapping, name: str) -> QuantizedTensor:
    if name not in tensors:
        raise KeyError(f"tensor {name!r} 不在模型中；已有: {list(tensors)}")
    tensor = tensors[name]
    if not is_quantized_tensor(tensor):
        raise TypeError(f"tensor {name!r} 不是 NINT/NVQ 权重")
    return tensor
