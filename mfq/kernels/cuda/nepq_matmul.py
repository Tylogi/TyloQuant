"""CUDA runtime glue for cross-expert NEPQ table pools."""

from __future__ import annotations

import numpy as np
import torch

from mfq.formats.nepq import (
    NEPQ0_L,
    NEPQ0_S,
    NEPQ1_L,
    NEPQ1_S,
    NepqTensor,
    _pack_bits,
    rotation_signs as build_rotation_signs,
    validate_nepq,
)
from mfq.formats.npq0_s import unpack_npq0_s_tables
from mfq.formats.nvq1_l import unpack_ternary_codebook
from mfq.formats.nvq1_s import unpack_nvq1_s_banked_codebook
from mfq.kernels.cuda._ext import ext


def _cuda_u8(payload: bytes, device: str | torch.device) -> torch.Tensor:
    values = np.frombuffer(payload, dtype=np.uint8).copy()
    return torch.as_tensor(values, device=device, dtype=torch.uint8).contiguous()


def _runtime_table(payload: np.ndarray, tensor: NepqTensor) -> np.ndarray:
    raw = np.ascontiguousarray(payload, dtype=np.uint8).tobytes()
    if tensor.spec is NEPQ0_S:
        return np.frombuffer(raw, dtype=np.int8).copy()
    if tensor.spec is NEPQ0_L:
        return np.frombuffer(raw, dtype=np.int8).copy()
    if tensor.spec is NEPQ1_S:
        return unpack_nvq1_s_banked_codebook(raw).reshape(-1).astype(
            np.int8, copy=False
        )
    if tensor.spec is NEPQ1_L:
        return unpack_ternary_codebook(raw).reshape(-1).astype(np.int8, copy=False)
    raise ValueError(f"unsupported NEPQ profile: {tensor.spec.label}")


def _npq0_s_expanded_table(payload: np.ndarray) -> np.ndarray:
    raw = np.ascontiguousarray(payload, dtype=np.uint8).tobytes()
    _, first, second, _ = unpack_npq0_s_tables(raw)
    first_product = np.broadcast_to(first[:, None, :, :], (4, 8, 8, 4))
    second_product = np.broadcast_to(second[:, :, None, :], (4, 8, 8, 4))
    product = np.concatenate((first_product, second_product), axis=-1)
    metadata = np.frombuffer(raw[:64], dtype=np.int8)
    return np.concatenate((metadata, product.reshape(-1))).astype(
        np.int8, copy=False
    )


def to_gpu_nepq(
    tensor: NepqTensor,
    device: str | torch.device = "cuda",
) -> dict:
    """Move one complete cross-expert projection to its production layout."""

    ng, _, nsuper, _, rows = validate_nepq(tensor)
    tables = np.stack(
        [_runtime_table(payload, tensor) for payload in tensor.table_payloads]
    )
    if tables.shape[1] != tensor.spec.runtime_table_bytes:
        raise ValueError(
            f"{tensor.spec.label} runtime table width mismatch: {tables.shape[1]}"
        )
    table_pool = torch.as_tensor(
        np.ascontiguousarray(tables, dtype=np.int8),
        device=device,
        dtype=torch.int8,
    ).contiguous()
    if tensor.spec is NEPQ0_S:
        grouped_tables = np.stack(
            [
                _npq0_s_expanded_table(payload)
                for payload in tensor.table_payloads
            ]
        )
        grouped_table_pool = torch.as_tensor(
            np.ascontiguousarray(grouped_tables, dtype=np.int8),
            device=device,
            dtype=torch.int8,
        ).contiguous()
    else:
        grouped_table_pool = table_pool
    aux = (
        np.empty(0, dtype=np.uint8)
        if tensor.aux is None
        else np.asarray(tensor.aux)
    )
    rotation_signs = (
        build_rotation_signs(
            tensor.neuron_len,
            int(tensor.rotation_block),
            int(tensor.rotation_seed),
        )
        if tensor.rotation_block
        else np.empty(0, dtype=np.int8)
    )
    return {
        "indices_packed": _cuda_u8(
            _pack_bits(np.asarray(tensor.indices), tensor.spec.index_bits), device
        ),
        "aux_packed": _cuda_u8(_pack_bits(aux, tensor.spec.aux_bits), device),
        "state_packed": _cuda_u8(
            _pack_bits(np.asarray(tensor.state), tensor.spec.state_bits), device
        ),
        "neuron_scale": torch.as_tensor(
            np.ascontiguousarray(tensor.neuron_scale, dtype=np.float32).reshape(-1),
            device=device,
            dtype=torch.float32,
        ).contiguous(),
        "table_pool": table_pool,
        "grouped_table_pool": grouped_table_pool,
        "bank_ids": torch.as_tensor(
            np.ascontiguousarray(tensor.bank_ids, dtype=np.uint8).reshape(rows, nsuper),
            device=device,
            dtype=torch.uint8,
        ).contiguous(),
        "rotation_signs": torch.as_tensor(
            rotation_signs, device=device, dtype=torch.int8
        ).contiguous(),
        "rotation_block": int(tensor.rotation_block),
        "rotation_seed": int(tensor.rotation_seed),
        "format": int(tensor.spec.base_format_id),
        "sub_bits": int(tensor.spec.state_bits),
        "gs": int(tensor.spec.groupsize),
        "ng": int(ng),
        "rows": int(rows),
        "n_experts": int(tensor.n_experts),
        "out_per_expert": int(tensor.out_per_expert),
        "neuron_len": int(tensor.neuron_len),
        "shape": tuple(int(value) for value in tensor.shape),
        "device": torch.device(device),
    }


def _kernel_args(g: dict) -> tuple:
    return (
        g["indices_packed"],
        g["aux_packed"],
        g["state_packed"],
        g["neuron_scale"],
        g["table_pool"],
        g["bank_ids"],
    )


def _workspace(g: dict, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = int(x.shape[0])
    k_pad = int(g["ng"]) * int(g["gs"])
    key = (str(x.device), m, k_pad)
    cache = g.setdefault("_workspace", {})
    value = cache.get(key)
    if value is not None and value[0].device == x.device:
        return value
    qx = torch.empty((m, k_pad), device=x.device, dtype=torch.int8)
    xscale = torch.empty((m, int(g["ng"])), device=x.device, dtype=torch.float32)
    cache[key] = (qx, xscale)
    return qx, xscale


def _prepare_input(g: dict, x: torch.Tensor) -> torch.Tensor:
    value = x.contiguous().to(device=g["device"], dtype=torch.float16)
    if value.dim() != 2 or value.shape[1] != int(g["neuron_len"]):
        raise ValueError("NEPQ input must be rank-2 [M,K]")
    block = int(g["rotation_block"])
    if block:
        value = ext().nepq_hadamard_input_cuda(
            value, g["rotation_signs"], block
        )
    return value


def _prepare_routed_input(g: dict, x: torch.Tensor) -> torch.Tensor:
    if x.dim() not in (2, 3):
        raise ValueError("NEPQ routed input must have [T,K] or [T,R,K] shape")
    original = tuple(x.shape)
    value = _prepare_input(g, x.reshape(-1, original[-1]))
    return value.reshape(original)


def _moe_workspace(
    g: dict, x: torch.Tensor, input_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    k_pad = int(g["ng"]) * int(g["gs"])
    key = (str(x.device), input_rows, k_pad, "moe")
    cache = g.setdefault("_workspace", {})
    value = cache.get(key)
    if value is not None and value[0].device == x.device:
        return value
    value = (
        torch.empty((input_rows, k_pad), device=x.device, dtype=torch.int8),
        torch.empty((input_rows, int(g["ng"])), device=x.device, dtype=torch.float32),
    )
    cache[key] = value
    return value


def nepq_dequantize(g: dict) -> torch.Tensor:
    """Decode stored-coordinate expert matrices as fp16 ``[E,O,K]``."""

    value = ext().nepq_dequant_cuda(
        *_kernel_args(g),
        int(g["neuron_len"]),
        int(g["sub_bits"]),
        int(g["format"]),
    )
    return value.reshape(g["shape"])


def nepq_gemv(g: dict, x: torch.Tensor) -> torch.Tensor:
    value = _prepare_input(g, x)
    if not 1 <= value.shape[0] <= 16:
        raise ValueError("NEPQ GEMV requires M in [1,16]")
    qx, xscale = _workspace(g, value)
    return ext().nepq_gemv_ws_cuda(
        *_kernel_args(g),
        value,
        int(g["neuron_len"]),
        int(g["sub_bits"]),
        int(g["format"]),
        qx,
        xscale,
    )


def nepq_mmq(g: dict, x: torch.Tensor) -> torch.Tensor:
    value = _prepare_input(g, x)
    if not 4 <= value.shape[0] <= 64:
        raise ValueError("NEPQ MMQ requires M in [4,64]")
    qx, xscale = _workspace(g, value)
    return ext().nepq_mmq_ws_cuda(
        *_kernel_args(g),
        value,
        int(g["neuron_len"]),
        int(g["sub_bits"]),
        int(g["format"]),
        qx,
        xscale,
    )


def nepq_gemm_f16(g: dict, x: torch.Tensor) -> torch.Tensor:
    value = _prepare_input(g, x)
    if not 16 <= value.shape[0] <= 256:
        raise ValueError("NEPQ online FP16 GEMM requires M in [16,256]")
    return ext().nepq_gemm_f16_cuda(
        *_kernel_args(g),
        value,
        int(g["neuron_len"]),
        int(g["sub_bits"]),
        int(g["format"]),
    )


def nepq_matmul(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Compute all expert projections and return ``[...,E,O]``."""

    original = tuple(x.shape)
    if not original:
        raise ValueError("NEPQ matmul input must have at least one dimension")
    x2 = x.reshape(-1, original[-1])
    m = int(x2.shape[0])
    if m <= 16:
        value = nepq_gemv(g, x2)
    elif m <= 64:
        value = nepq_mmq(g, x2)
    elif m <= 256:
        value = nepq_gemm_f16(g, x2)
    else:
        transformed = _prepare_input(g, x2)
        weight = nepq_dequantize(g).reshape(int(g["rows"]), -1)
        value = ext().nint_cublas_gemm_nt_f16acc_cuda(transformed, weight)
    return value.reshape(
        *original[:-1], int(g["n_experts"]), int(g["out_per_expert"])
    )


def nepq_grouped_matmul(
    g: dict,
    x: torch.Tensor,
    route,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute only selected experts and return ``[T,R,O]``."""

    if int(route.n_experts) != int(g["n_experts"]):
        raise ValueError("NEPQ route and weight expert counts differ")
    value = _prepare_routed_input(g, x)
    tokens = int(route.ids.shape[0])
    routes = int(route.ids.shape[1])
    if value.shape[0] != tokens:
        raise ValueError("NEPQ input and route token counts differ")
    if value.dim() == 3 and value.shape[1] != routes:
        raise ValueError("NEPQ routed input and route counts differ")
    expected = (tokens, routes, int(g["out_per_expert"]))
    if out is None:
        out = torch.empty(expected, device=value.device, dtype=torch.float16)
    elif (
        tuple(out.shape) != expected
        or out.device != value.device
        or out.dtype != torch.float16
        or not out.is_contiguous()
    ):
        raise ValueError(f"NEPQ output must be contiguous fp16 {expected}")
    input_rows = tokens * routes if value.dim() == 3 else tokens
    qx, xscale = _moe_workspace(g, value, input_rows)
    return ext().nepq_moe_grouped_matmul_ws_cuda(
        *_kernel_args(g),
        g["grouped_table_pool"],
        value,
        route.ids,
        int(g["n_experts"]),
        int(g["out_per_expert"]),
        int(g["neuron_len"]),
        int(g["sub_bits"]),
        int(g["format"]),
        out,
        qx,
        xscale,
        route.ids_dst,
        route.expert_bounds,
        route.tile_bounds,
        route.tile_experts,
    )


def nepq_grouped_matmul_pool(
    g: dict,
    x: torch.Tensor,
    route,
    expert_local: torch.Tensor,
    pool_experts: int,
    *,
    out: torch.Tensor,
    qx: torch.Tensor | None = None,
    xscale: torch.Tensor | None = None,
    input_quantized: bool = False,
    input_prepared: bool = False,
) -> torch.Tensor:
    """Run one NEPQ cohort inside a global expert routing plan."""

    value = (
        x.contiguous().to(device=g["device"], dtype=torch.float16)
        if input_prepared
        else _prepare_routed_input(g, x)
    )
    tokens, routes = (int(v) for v in route.ids.shape)
    if value.dim() not in (2, 3) or value.shape[-1] != int(g["neuron_len"]):
        raise ValueError("NEPQ routed input must have [T,K] or [T,R,K] shape")
    if value.shape[0] != tokens or (value.dim() == 3 and value.shape[1] != routes):
        raise ValueError("NEPQ input leading dimensions do not match routes")
    if int(g["n_experts"]) != int(pool_experts):
        raise ValueError("NEPQ cohort expert count differs from its packed tensor")
    expected = (tokens, routes, int(g["out_per_expert"]))
    if (
        tuple(out.shape) != expected
        or out.device != value.device
        or out.dtype != torch.float16
        or not out.is_contiguous()
    ):
        raise ValueError(f"NEPQ output must be contiguous fp16 {expected}")
    input_rows = tokens * routes if value.dim() == 3 else tokens
    if qx is None or xscale is None:
        if input_quantized:
            raise ValueError("prequantized NEPQ input requires qx and xscale")
        qx, xscale = _moe_workspace(g, value, input_rows)
    return ext().nepq_moe_grouped_matmul_pool_ws_cuda(
        *_kernel_args(g),
        g["grouped_table_pool"],
        value,
        route.ids,
        expert_local,
        int(route.n_experts),
        int(pool_experts),
        int(g["out_per_expert"]),
        int(g["neuron_len"]),
        int(g["sub_bits"]),
        int(g["format"]),
        bool(input_quantized),
        out,
        qx,
        xscale,
        route.ids_dst,
        route.expert_bounds,
        route.tile_bounds,
        route.tile_experts,
    )


__all__ = [
    "nepq_dequantize",
    "nepq_gemm_f16",
    "nepq_grouped_matmul",
    "nepq_grouped_matmul_pool",
    "nepq_gemv",
    "nepq_matmul",
    "nepq_mmq",
    "to_gpu_nepq",
]
