"""Packed TPQ-I4 and TPQ-PQ CUDA matrix operators."""

from __future__ import annotations

import numpy as np
import torch

from mfq.formats.tpq import TpqInt4Tensor, TpqPqTensor, pack_tpq_indices
from mfq.kernels.cuda._ext import ext

TpqTensor = TpqInt4Tensor | TpqPqTensor


def to_gpu_tpq(
    tensor: TpqTensor,
    device: str | torch.device = "cuda",
) -> dict[str, object]:
    """Copy one validated packed TPQ tensor to a CUDA device."""

    if isinstance(tensor, TpqInt4Tensor):
        return {
            "int4": True,
            "packed": torch.from_numpy(np.ascontiguousarray(tensor.packed, dtype=np.uint8)).to(
                device=device
            ),
            "scales": torch.from_numpy(np.ascontiguousarray(tensor.scales, dtype=np.float16)).to(
                device=device
            ),
            "shape": tensor.shape,
            "out": int(tensor.shape[0]),
            "neuron_len": int(tensor.shape[1]),
            "group_size": tensor.group_size,
        }
    packed = (
        np.frombuffer(
            pack_tpq_indices(tensor.indices, tensor.spec.index_bits),
            dtype=np.uint8,
        ).copy()
        if tensor.spec.index_bits in {8, 16}
        else np.ascontiguousarray(tensor.indices, dtype=np.uint8).reshape(-1)
    )
    return {
        "int4": False,
        "packed": torch.from_numpy(packed).to(device=device),
        "codebook": torch.from_numpy(np.ascontiguousarray(tensor.codebook, dtype=np.float32)).to(
            device=device
        ),
        "shape": tensor.shape,
        "out": int(tensor.shape[0]),
        "neuron_len": int(tensor.shape[1]),
        "vector_size": tensor.spec.vector_size,
        "index_bits": tensor.spec.index_bits,
    }


def tpq_dequantize(weight: dict[str, object]) -> torch.Tensor:
    """Decode a packed TPQ matrix to FP16 on its CUDA device."""

    if bool(weight["int4"]):
        return ext().tpq_int4_dequant_cuda(
            weight["packed"], weight["scales"], int(weight["group_size"])
        )
    rows, columns = weight["shape"]
    return ext().tpq_pq_dequant_cuda(
        weight["packed"],
        weight["codebook"],
        int(rows),
        int(columns),
        int(weight["vector_size"]),
        int(weight["index_bits"]),
    )


def tpq_matmul(
    weight: dict[str, object],
    x: torch.Tensor,
) -> torch.Tensor:
    """Multiply FP16 activations by a packed TPQ matrix."""

    rows, columns = weight["shape"]
    x = torch.as_tensor(x, device=weight["packed"].device, dtype=torch.float16)
    if x.shape[-1] != columns:
        raise ValueError(f"activation width {x.shape[-1]} does not match TPQ width {columns}")
    original = x.shape
    flat = x.reshape(-1, columns).contiguous()
    if bool(weight["int4"]):
        output = ext().tpq_int4_matmul_f16_cuda(
            weight["packed"],
            weight["scales"],
            flat,
            int(weight["group_size"]),
        )
    else:
        output = ext().tpq_pq_matmul_f16_cuda(
            weight["packed"],
            weight["codebook"],
            flat,
            int(rows),
            int(columns),
            int(weight["vector_size"]),
            int(weight["index_bits"]),
        )
    return output.reshape(*original[:-1], rows)


def tpq_embedding(
    weight: dict[str, object],
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Decode only the selected rows of a packed TPQ matrix."""

    packed = weight["packed"]
    if not isinstance(packed, torch.Tensor):
        raise TypeError("TPQ packed values must be a torch tensor")
    ids = torch.as_tensor(token_ids, device=packed.device, dtype=torch.int64).contiguous()
    rows, columns = weight["shape"]
    if bool(weight["int4"]):
        return ext().tpq_int4_embedding_lookup_cuda(
            packed,
            weight["scales"],
            ids,
            int(weight["group_size"]),
        )
    return ext().tpq_pq_embedding_lookup_cuda(
        packed,
        weight["codebook"],
        ids,
        int(rows),
        int(columns),
        int(weight["vector_size"]),
        int(weight["index_bits"]),
    )


__all__ = [
    "TpqTensor",
    "to_gpu_tpq",
    "tpq_dequantize",
    "tpq_embedding",
    "tpq_matmul",
]
