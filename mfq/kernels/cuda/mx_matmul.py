"""Packed OCP MXFP4/MXFP8 CUDA matrix operators."""

from __future__ import annotations

import numpy as np
import torch

from mfq.formats.mx import MXFP4_DTYPE, MXFP8_DTYPE, MxTensor
from mfq.kernels.cuda._ext import ext


def to_gpu_mx(
    tensor: MxTensor,
    device: str | torch.device = "cuda",
) -> dict[str, object]:
    """Copy one validated packed MX tensor to a CUDA device."""

    values = np.ascontiguousarray(tensor.values, dtype=np.uint8)
    scales = np.ascontiguousarray(tensor.scales, dtype=np.uint8)
    if np.any(scales == 255):
        raise ValueError(f"{tensor.dtype} contains an E8M0 NaN scale")
    if tensor.dtype == MXFP8_DTYPE and np.any((values & 0x7F) == 0x7F):
        raise ValueError("MXFP8 contains an E4M3 NaN code")
    return {
        "dtype": tensor.dtype,
        "shape": tuple(int(value) for value in tensor.shape),
        "out": int(tensor.shape[0]),
        "neuron_len": int(tensor.shape[1]),
        "values": torch.from_numpy(values).to(device=device).contiguous(),
        "scales": torch.from_numpy(scales).to(device=device).contiguous(),
    }


def mx_dequantize(weight: dict[str, object]) -> torch.Tensor:
    """Materialize an FP16 matrix for validation only."""

    dtype = str(weight["dtype"])
    values = weight["values"]
    scales = weight["scales"]
    if not isinstance(values, torch.Tensor) or not isinstance(scales, torch.Tensor):
        raise TypeError("MX packed values and scales must be torch tensors")
    if dtype == MXFP4_DTYPE:
        return ext().mxfp4_dequant_cuda(values, scales)
    if dtype == MXFP8_DTYPE:
        return ext().mxfp8_dequant_cuda(values, scales)
    raise ValueError(f"unsupported MX dtype: {dtype}")


def mx_matmul(weight: dict[str, object], x: torch.Tensor) -> torch.Tensor:
    """Apply a packed MX projection without keeping a dense weight copy."""

    shape = tuple(int(value) for value in weight["shape"])
    if len(shape) != 2:
        raise ValueError("MX weight must be rank-2")
    x = torch.as_tensor(x, device=weight["values"].device, dtype=torch.float16)
    if x.shape[-1] != shape[1]:
        raise ValueError(f"MX input width {x.shape[-1]} does not match weight width {shape[1]}")
    original = x.shape
    flat = x.reshape(-1, original[-1]).contiguous()
    dtype = str(weight["dtype"])
    if dtype == MXFP4_DTYPE:
        output = ext().mxfp4_matmul_f16_cuda(weight["values"], weight["scales"], flat)
    elif dtype == MXFP8_DTYPE:
        output = (
            ext().mxfp8_small_m_cuda(weight["values"], weight["scales"], flat)
            if flat.shape[0] <= 8
            else ext().mxfp8_matmul_f16_cuda(weight["values"], weight["scales"], flat)
        )
    else:
        raise ValueError(f"unsupported MX dtype: {dtype}")
    return output.reshape(*original[:-1], shape[0])


def mx_embedding(
    weight: dict[str, object],
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Decode only the selected rows of a packed MX matrix."""

    values = weight["values"]
    if not isinstance(values, torch.Tensor):
        raise TypeError("MX packed values must be a torch tensor")
    ids = torch.as_tensor(token_ids, device=values.device, dtype=torch.int64).contiguous()
    dtype = str(weight["dtype"])
    if dtype == MXFP4_DTYPE:
        return ext().mxfp4_embedding_lookup_cuda(values, weight["scales"], ids)
    if dtype == MXFP8_DTYPE:
        return ext().mxfp8_embedding_lookup_cuda(values, weight["scales"], ids)
    raise ValueError(f"unsupported MX dtype: {dtype}")


__all__ = ["mx_dequantize", "mx_embedding", "mx_matmul", "to_gpu_mx"]
