"""Native CUDA operators for symmetric NINT8-0 tensors."""

from __future__ import annotations

import numpy as np
import torch

from mfq.formats.nint8_zero import Nint8ZeroTensor
from mfq.kernels.cuda._ext import ext


def to_gpu_nint8_zero(
    tensor: Nint8ZeroTensor,
    device: str | torch.device = "cuda",
) -> dict[str, object]:
    q = torch.from_numpy(np.ascontiguousarray(tensor.q, dtype=np.int8))
    scale = torch.from_numpy(np.ascontiguousarray(tensor.scale, dtype=np.float16))
    return {
        "q": q.view(torch.uint8).to(device=device).contiguous(),
        "scale": scale.to(device=device).contiguous(),
        "out": int(tensor.shape[tensor.axis]),
        "ng": int(tensor.q.shape[1]),
        "neuron_len": int(tensor.neuron_len),
        "shape": tuple(int(value) for value in tensor.shape),
    }


def nint8_zero_dequantize(weight: dict[str, object]) -> torch.Tensor:
    return ext().nint8_zero_dequant_cuda(weight["q"], weight["scale"], int(weight["neuron_len"]))


def nint8_zero_matmul(
    weight: dict[str, object],
    x: torch.Tensor,
) -> torch.Tensor:
    q = weight["q"]
    if not isinstance(q, torch.Tensor):
        raise TypeError("NINT8-0 q must be a torch tensor")
    x = torch.as_tensor(x, device=q.device, dtype=torch.float16)
    width = int(weight["neuron_len"])
    if x.shape[-1] != width:
        raise ValueError(f"NINT8-0 input width {x.shape[-1]} does not match {width}")
    original = x.shape
    flat = x.reshape(-1, width).contiguous()
    rows = int(flat.shape[0])
    ng = int(weight["ng"])
    if rows <= 64:
        qx = torch.empty((rows, ng * 32), device=q.device, dtype=torch.int8)
        xscale = torch.empty((rows, ng), device=q.device, dtype=torch.float32)
        function = ext().nint8_zero_gemv_ws_cuda if rows <= 8 else ext().nint8_zero_mmq_ws_cuda
        output = function(weight["q"], weight["scale"], flat, qx, xscale)
    else:
        output = ext().nint8_zero_mmq_f16_packed_cuda(
            weight["q"], weight["scale"], flat, width
        )
    return output.reshape(*original[:-1], int(weight["out"]))


def nint8_zero_embedding(
    weight: dict[str, object],
    token_ids: torch.Tensor,
) -> torch.Tensor:
    q = weight["q"]
    if not isinstance(q, torch.Tensor):
        raise TypeError("NINT8-0 q must be a torch tensor")
    ids = torch.as_tensor(token_ids, device=q.device, dtype=torch.int64).contiguous()
    return ext().nint8_zero_embedding_lookup_cuda(
        weight["q"], weight["scale"], ids, int(weight["neuron_len"])
    )


__all__ = [
    "nint8_zero_dequantize",
    "nint8_zero_embedding",
    "nint8_zero_matmul",
    "to_gpu_nint8_zero",
]
