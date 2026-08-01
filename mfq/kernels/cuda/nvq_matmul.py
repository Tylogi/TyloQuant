"""CUDA runtime glue for compact NPQ and NVQ weights."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch

from mfq.formats.npq0_l import Npq0LTensor, pack_npq0_l_tables
from mfq.formats.npq0_s import Npq0STensor, pack_npq0_s_tables
from mfq.formats.nvq import (
    D4_256,
    E8_256,
    NvqJscTensor,
    NvqTensor,
    pack_jsc_metadata,
)
from mfq.formats.nvq import (
    _pack_bits as _pack_nvq_bits,
)
from mfq.formats.nvq1_l import (
    IQ1S_TERNARY_2048,
    Nvq1LTensor,
)
from mfq.formats.nvq1_l import (
    _pack_bits as _pack_nvq1_l_bits,
)
from mfq.formats.nvq1_s import NVQ1_S_SYNTHETIC_BANKS, Nvq1STensor
from mfq.kernels.cuda._ext import ext

NvqAnyTensor: TypeAlias = (
    NvqTensor | NvqJscTensor | Npq0LTensor | Npq0STensor | Nvq1LTensor | Nvq1STensor
)


def _cuda_u8(payload: bytes, device: str | torch.device) -> torch.Tensor:
    values = np.frombuffer(payload, dtype=np.uint8).copy()
    return torch.as_tensor(values, device=device, dtype=torch.uint8).contiguous()


def _npq0_s_runtime_lut(tensor: Npq0STensor) -> np.ndarray:
    """Expand the stored PQ factors into a read-optimized Cartesian LUT."""

    packed = np.frombuffer(
        pack_npq0_s_tables(
            tensor.scale_lut,
            tensor.first_codebooks,
            tensor.second_codebooks,
        ),
        dtype=np.int8,
    )
    first = np.asarray(tensor.first_codebooks, dtype=np.int8)
    second = np.asarray(tensor.second_codebooks, dtype=np.int8)
    first_product = np.broadcast_to(first[:, None, :, :], (4, 8, 8, 4))
    second_product = np.broadcast_to(second[:, :, None, :], (4, 8, 8, 4))
    product = np.concatenate((first_product, second_product), axis=-1).reshape(4, 64, 8)
    return np.concatenate((packed[:64], product.reshape(-1))).astype(np.int8, copy=False)


def to_gpu_nvq(
    tensor: NvqAnyTensor,
    device: str | torch.device = "cuda",
) -> dict:
    """Move one NVQ matrix to its compact production GPU representation."""

    if tensor.axis != 0 or len(tensor.shape) != 2:
        raise ValueError(
            f"NVQ CUDA linear weights must be rank-2 with axis=0, got "
            f"shape={tensor.shape}, axis={tensor.axis}"
        )
    out = int(tensor.neuron_scale.size)
    if tensor.shape != (out, tensor.neuron_len):
        raise ValueError(
            f"NVQ CUDA matrix shape mismatch: {tensor.shape} != "
            f"{(out, tensor.neuron_len)}"
        )

    if isinstance(tensor, Npq0LTensor):
        format_id = 7
        sign_mode = 0
        codebook = np.frombuffer(
            pack_npq0_l_tables(
                tensor.scale_lut,
                tensor.first_codebooks,
                tensor.second_codebooks,
            ),
            dtype=np.int8,
        ).copy()
        indices = _pack_nvq_bits(tensor.indices, 7)
        aux = b""
        sub_scale = _pack_nvq_bits(tensor.state, 3)
    elif isinstance(tensor, Npq0STensor):
        format_id = 9
        sign_mode = 0
        codebook = _npq0_s_runtime_lut(tensor)
        indices = _pack_nvq_bits(tensor.indices, 6)
        aux = b""
        sub_scale = _pack_nvq_bits(tensor.state, 2)
    elif isinstance(tensor, Nvq1LTensor):
        format_id = 1
        sign_mode = 0
        codebook = tensor.codebook if tensor.codebook is not None else IQ1S_TERNARY_2048
        indices = _pack_nvq1_l_bits(tensor.indices, tensor.spec.index_bits)
        aux = _pack_nvq1_l_bits(tensor.delta_sign, 1)
        sub_scale = _pack_nvq1_l_bits(tensor.sub_scale, tensor.spec.sub_bits)
    elif isinstance(tensor, Nvq1STensor):
        format_id = 8
        sign_mode = 0
        codebook = (
            NVQ1_S_SYNTHETIC_BANKS if tensor.codebook is None else tensor.codebook
        ).reshape(1024, 8)
        indices = _pack_nvq1_l_bits(tensor.indices, 9)
        aux = _pack_nvq1_l_bits(tensor.delta_sign, 1)
        sub_scale = _pack_nvq1_l_bits(tensor.sub_scale, 4)
    elif isinstance(tensor, NvqJscTensor):
        sign_mode = 0
        metadata = pack_jsc_metadata(tensor)
        if tensor.spec.codebook == "e8_256":
            format_id = 5
        elif tensor.spec.codebook == "e8_1024":
            format_id = 13
        elif tensor.spec.codebook == "e8_4096":
            format_id = 14
        elif tensor.spec.codebook == "d4_512":
            format_id = 12
        elif tensor.spec.codebook == "d4_1024":
            format_id = 15
        else:
            format_id = 11 if metadata[3] == 1 and metadata[1] == 2 else 10
        codebook = np.frombuffer(metadata, dtype=np.int8).copy()
        indices = _pack_nvq_bits(tensor.indices, tensor.spec.index_bits)
        aux = _pack_nvq_bits(tensor.signs, 7)
        sub_scale = _pack_nvq_bits(tensor.state, 4)
    else:
        if tensor.spec.codebook not in {"e8_256", "d4_256"}:
            raise ValueError(
                f"{tensor.spec.codebook} requires an NvqJscTensor CUDA profile"
            )
        format_id = 2 if tensor.spec.codebook == "e8_256" else 3
        sign_mode = 1 if tensor.spec.sign_mode == "index_parity" else 0
        builtin = E8_256 if format_id == 2 else D4_256
        codebook = tensor.codebook if tensor.codebook is not None else builtin
        indices = _pack_nvq_bits(tensor.indices, tensor.spec.index_bits)
        aux = _pack_nvq_bits(tensor.signs, 7)
        sub_scale = _pack_nvq_bits(tensor.sub_scale, tensor.spec.sub_bits)

    ng = (tensor.neuron_len + tensor.spec.groupsize - 1) // tensor.spec.groupsize
    return {
        "indices_packed": _cuda_u8(indices, device),
        "aux_packed": _cuda_u8(aux, device),
        "sub_scale_packed": _cuda_u8(sub_scale, device),
        "neuron_scale": torch.as_tensor(
            np.ascontiguousarray(tensor.neuron_scale, dtype=np.float32),
            device=device,
        ).contiguous(),
        "codebook": torch.as_tensor(
            np.array(codebook, dtype=np.int8, copy=True, order="C"),
            device=device,
            dtype=torch.int8,
        ).contiguous(),
        "format": format_id,
        "sign_mode": sign_mode,
        "sub_bits": int(tensor.spec.sub_bits),
        "gs": int(tensor.spec.groupsize),
        "ng": int(ng),
        "out": out,
        "neuron_len": int(tensor.neuron_len),
        "shape": tuple(int(value) for value in tensor.shape),
        "axis": int(tensor.axis),
        "device": torch.device(device),
    }


def to_gpu_nvq_exec(
    tensor: NvqAnyTensor,
    device: str | torch.device = "cuda",
) -> dict:
    """Move NVQ tensors to the compact layout used by the C++ runtime."""

    result = to_gpu_nvq(tensor, device)
    if result["format"] not in {2, 5}:
        return result
    indices = np.asarray(tensor.indices, dtype=np.uint8)
    mask7 = np.asarray(tensor.signs, dtype=np.uint8)
    if indices.shape != mask7.shape:
        raise ValueError(
            f"NVQ execution metadata shape mismatch: {indices.shape} != {mask7.shape}"
        )
    parity_lut = np.fromiter(
        (value.bit_count() & 1 for value in range(128)),
        dtype=np.uint8,
        count=128,
    )
    last = parity_lut[mask7]
    if result["sign_mode"]:
        last ^= indices >> 7
    metadata = np.empty((*indices.shape, 2), dtype=np.uint8)
    metadata[..., 0] = indices
    metadata[..., 1] = mask7 | (last << 7)
    result["storage_format"] = result["format"]
    result["format"] = 4 if result["format"] == 2 else 6
    result["indices_packed"] = torch.as_tensor(
        metadata.reshape(-1), device=device, dtype=torch.uint8
    ).contiguous()
    result["aux_packed"] = torch.empty(0, device=device, dtype=torch.uint8)
    return result


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


def _moe_workspace(
    g: dict, x: torch.Tensor, input_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    k_pad = int(g["ng"]) * int(g["gs"])
    key = (str(x.device), int(input_rows), k_pad, "moe")
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


def _kernel_args(g: dict) -> tuple:
    return (
        g["indices_packed"],
        g["aux_packed"],
        g["sub_scale_packed"],
        g["neuron_scale"],
        g["codebook"],
    )


def _check_pair(first: dict, second: dict, operation: str) -> None:
    for key in ("format", "gs", "neuron_len"):
        if int(first[key]) != int(second[key]):
            raise ValueError(
                f"NVQ {operation} requires matching {key}, got "
                f"{first[key]} and {second[key]}"
            )


def nvq_dequantize(g: dict) -> torch.Tensor:
    """Decode one compact NVQ matrix to fp16 on the current CUDA stream."""

    return ext().nvq_dequant_cuda(
        *_kernel_args(g),
        int(g["neuron_len"]),
        int(g["gs"]),
        int(g["sub_bits"]),
        int(g["format"]),
        int(g["sign_mode"]),
    )


def nvq_mmq(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Run the gs24 int8 Tensor Core candidate for M=4..64."""

    x = x.contiguous().to(torch.float16)
    if x.dim() != 2 or not 4 <= x.shape[0] <= 64 or x.shape[1] != int(g["neuron_len"]):
        raise ValueError("NVQ MMQ input must be [M, neuron_len] with M in [4,64]")
    qx, xscale = _workspace(g, x)
    return ext().nvq_mmq_ws_cuda(
        *_kernel_args(g),
        x,
        int(g["neuron_len"]),
        int(g["gs"]),
        int(g["sub_bits"]),
        int(g["format"]),
        int(g["sign_mode"]),
        qx,
        xscale,
    )


def nvq_gemm_f16(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Run online NVQ decode with FP16 Tensor Core GEMM for M=16..256."""

    x = x.contiguous().to(torch.float16)
    if (
        x.dim() != 2
        or not 16 <= x.shape[0] <= 256
        or x.shape[1] != int(g["neuron_len"])
    ):
        raise ValueError(
            "NVQ FP16 GEMM input must be [M, neuron_len] with M in [16,256]"
        )
    return ext().nvq_gemm_f16_cuda(
        *_kernel_args(g),
        x,
        int(g["neuron_len"]),
        int(g["gs"]),
        int(g["sub_bits"]),
        int(g["format"]),
        int(g["sign_mode"]),
    )


def nvq_gemv_m1_vec8(g: dict, x: torch.Tensor, nwarps: int = 4) -> torch.Tensor:
    """Run the llama.cpp-style one-row-per-block M=1 candidate."""

    x = x.contiguous().to(torch.float16)
    if x.dim() != 2 or x.shape != (1, int(g["neuron_len"])):
        raise ValueError("NVQ vec8 GEMV input must be [1, neuron_len]")
    qx, xscale = _workspace(g, x)
    return ext().nvq_gemv_m1_vec8_ws_cuda(
        *_kernel_args(g),
        x,
        int(g["neuron_len"]),
        int(g["gs"]),
        int(g["sub_bits"]),
        int(g["format"]),
        int(g["sign_mode"]),
        int(nwarps),
        qx,
        xscale,
    )


def nvq_gemv_batch_vec8(g: dict, x: torch.Tensor, nwarps: int = 4) -> torch.Tensor:
    """Run the llama-style weight-reuse candidate for M=2..16."""

    x = x.contiguous().to(torch.float16)
    if x.dim() != 2 or not 2 <= x.shape[0] <= 16 or x.shape[1] != int(g["neuron_len"]):
        raise ValueError("NVQ batch vec8 GEMV input must be [M, neuron_len] with M in [2,16]")
    qx, xscale = _workspace(g, x)
    return ext().nvq_gemv_batch_vec8_ws_cuda(
        *_kernel_args(g),
        x,
        int(g["neuron_len"]),
        int(g["gs"]),
        int(g["sub_bits"]),
        int(g["format"]),
        int(g["sign_mode"]),
        int(nwarps),
        qx,
        xscale,
    )


def nvq_matmul_multi2(
    first: dict,
    second: dict,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute two compatible NVQ projections with one activation quantization."""

    _check_pair(first, second, "multi-projection")
    original = tuple(x.shape)
    if not original:
        raise ValueError("NVQ multi-projection input must have at least one dimension")
    neuron_len = int(first["neuron_len"])
    x2 = x.reshape(-1, original[-1]).contiguous().to(torch.float16)
    if x2.shape[1] > neuron_len:
        raise ValueError(f"x last dim {x2.shape[1]} exceeds NVQ neuron_len {neuron_len}")
    if x2.shape[1] < neuron_len:
        x2 = torch.nn.functional.pad(x2, (0, neuron_len - x2.shape[1]))
    if x2.shape[0] > 8:
        return nvq_matmul(first, x), nvq_matmul(second, x)
    qx, xscale = _workspace(first, x2)
    combined = ext().nvq_gemv_multi2_ws_cuda(
        *_kernel_args(first),
        *_kernel_args(second),
        x2,
        neuron_len,
        int(first["gs"]),
        int(first["sub_bits"]),
        int(first["format"]),
        int(first["sign_mode"]),
        int(second["sub_bits"]),
        int(second["format"]),
        int(second["sign_mode"]),
        qx,
        xscale,
    )
    first_out = int(first["out"])
    second_out = int(second["out"])
    first_y, second_y = combined.split((first_out, second_out), dim=-1)
    return (
        first_y.reshape(*original[:-1], first_out),
        second_y.reshape(*original[:-1], second_out),
    )


def nvq_matmul_swiglu(gate: dict, up: dict, x: torch.Tensor) -> torch.Tensor:
    """Compute ``silu(x @ Wgate.T) * (x @ Wup.T)`` in one NVQ kernel."""

    _check_pair(gate, up, "SwiGLU")
    if int(gate["out"]) != int(up["out"]):
        raise ValueError("NVQ SwiGLU gate/up output widths must match")
    original = tuple(x.shape)
    neuron_len = int(gate["neuron_len"])
    x2 = x.reshape(-1, original[-1]).contiguous().to(torch.float16)
    if x2.shape[1] < neuron_len:
        x2 = torch.nn.functional.pad(x2, (0, neuron_len - x2.shape[1]))
    if x2.shape[1] != neuron_len:
        raise ValueError(f"NVQ SwiGLU input width must be <= {neuron_len}")
    if x2.shape[0] != 1:
        gate_y, up_y = nvq_matmul_multi2(gate, up, x)
        return torch.nn.functional.silu(gate_y) * up_y
    qx, xscale = _workspace(gate, x2)
    output = ext().nvq_gemv_swiglu_ws_cuda(
        *_kernel_args(gate),
        *_kernel_args(up),
        x2,
        neuron_len,
        int(gate["gs"]),
        int(gate["sub_bits"]),
        int(gate["format"]),
        int(gate["sign_mode"]),
        int(up["sub_bits"]),
        int(up["format"]),
        int(up["sign_mode"]),
        qx,
        xscale,
    )
    return output.reshape(*original[:-1], int(gate["out"]))


def nvq2_matmul_swiglu_vec4_ordered(gate: dict, up: dict, x: torch.Tensor) -> torch.Tensor:
    """Benchmark NVQ2 vec4 decode while preserving the standard FMA order."""

    _check_pair(gate, up, "ordered vec4 SwiGLU")
    if int(gate["format"]) != 2 or int(up["format"]) != 2:
        raise ValueError("NVQ2 ordered vec4 SwiGLU requires NVQ2 weights")
    if int(gate["sub_bits"]) != 4 or int(up["sub_bits"]) != 4:
        raise ValueError("NVQ2 ordered vec4 SwiGLU requires sub_bits=4")
    if int(gate["out"]) != int(up["out"]):
        raise ValueError("NVQ2 ordered vec4 SwiGLU gate/up output widths must match")
    original = tuple(x.shape)
    neuron_len = int(gate["neuron_len"])
    x2 = x.reshape(-1, original[-1]).contiguous().to(torch.float16)
    if x2.shape != (1, neuron_len):
        raise ValueError(f"NVQ2 ordered vec4 SwiGLU input must be [1,{neuron_len}]")
    qx, xscale = _workspace(gate, x2)
    output = ext().nvq2_gemv_swiglu_vec4_ordered_ws_cuda(
        *_kernel_args(gate),
        *_kernel_args(up),
        x2,
        neuron_len,
        int(gate["gs"]),
        int(gate["sub_bits"]),
        int(gate["format"]),
        int(gate["sign_mode"]),
        int(up["sub_bits"]),
        int(up["format"]),
        int(up["sign_mode"]),
        qx,
        xscale,
    )
    return output.reshape(*original[:-1], int(gate["out"]))


def nvq_ffn_swiglu_down(
    gate: dict,
    up: dict,
    down: dict,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run NVQ gate/up SwiGLU directly into the q8 workspace of NVQ Wdown."""

    _check_pair(gate, up, "fused FFN")
    if int(gate["out"]) != int(up["out"]) or int(gate["out"]) != int(down["neuron_len"]):
        raise ValueError("NVQ fused FFN gate/up width must equal Wdown input width")
    original = tuple(x.shape)
    neuron_len = int(gate["neuron_len"])
    x2 = x.reshape(-1, original[-1]).contiguous().to(torch.float16)
    if x2.shape[1] < neuron_len:
        x2 = torch.nn.functional.pad(x2, (0, neuron_len - x2.shape[1]))
    if x2.shape[1] != neuron_len:
        raise ValueError(f"NVQ fused FFN input width must be <= {neuron_len}")
    if x2.shape[0] != 1 or int(down["gs"]) not in (24, 28, 32):
        return nvq_matmul(down, nvq_matmul_swiglu(gate, up, x))
    input_qx, input_xscale = _workspace(gate, x2)
    output_qx, output_xscale = _workspace(down, x2)
    scratch_key = (str(x2.device), int(gate["out"]))
    scratch_cache = gate.setdefault("_swiglu_scratch", {})
    swiglu_scratch = scratch_cache.get(scratch_key)
    if swiglu_scratch is None or swiglu_scratch.device != x2.device:
        swiglu_scratch = torch.empty(
            (int(gate["out"]),), device=x2.device, dtype=torch.float32
        )
        scratch_cache[scratch_key] = swiglu_scratch
    ext().nvq_ffn_swiglu_quant_ws_cuda(
        *_kernel_args(gate),
        *_kernel_args(up),
        x2,
        neuron_len,
        int(gate["gs"]),
        int(gate["sub_bits"]),
        int(gate["format"]),
        int(gate["sign_mode"]),
        int(up["sub_bits"]),
        int(up["format"]),
        int(up["sign_mode"]),
        int(down["gs"]),
        input_qx,
        input_xscale,
        output_qx,
        output_xscale,
        swiglu_scratch,
    )
    output = ext().nvq_gemv_qx_ws_cuda(
        *_kernel_args(down),
        int(down["neuron_len"]),
        int(down["gs"]),
        int(down["sub_bits"]),
        int(down["format"]),
        int(down["sign_mode"]),
        output_qx,
        output_xscale,
    )
    return output.reshape(*original[:-1], int(down["out"]))


def _gate_mode(activation: str) -> int:
    if activation == "sigmoid":
        return 1
    if activation == "silu":
        return 2
    raise ValueError(f"unsupported activation {activation!r}")


def nvq_mmq_input_mul(
    g: dict,
    x: torch.Tensor,
    gate: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """Run gs24 int8 MMA while applying the input gate during q8 quantization."""

    mode = _gate_mode(activation)
    x = x.contiguous().to(torch.float16)
    gate = gate.contiguous().to(torch.float16)
    if x.dim() != 2 or not 4 <= x.shape[0] <= 64 or x.shape[1] != int(g["neuron_len"]):
        raise ValueError("NVQ gated MMQ input must be [M, neuron_len] with M in [4,64]")
    if gate.shape != x.shape:
        raise ValueError(f"x and gate must have the same shape, got {tuple(x.shape)} and {tuple(gate.shape)}")
    qx, xscale = _workspace(g, x)
    return ext().nvq_mmq_gate_ws_cuda(
        *_kernel_args(g),
        x,
        gate,
        int(g["neuron_len"]),
        int(g["gs"]),
        int(g["sub_bits"]),
        int(g["format"]),
        int(g["sign_mode"]),
        mode,
        qx,
        xscale,
    )


def _select_matmul_path(g: dict, m: int) -> str:
    storage_format = int(g.get("storage_format", g["format"]))
    out = int(g["out"])
    neuron_len = int(g["neuron_len"])
    e8_family = storage_format in {2, 5, 7, 8, 9, 13, 14}

    if storage_format == 8 and 14 <= m <= 16 and neuron_len >= 2 * out:
        return "dequant_gemm"

    if storage_format == 9 and m >= 13:
        wide_output = out * 8 >= neuron_len * 21
        wide_mmq = out >= 4096 and (out >= 2 * neuron_len or neuron_len >= 2 * out)
        if m <= 15:
            if wide_mmq:
                return "mmq"
            return "gemv" if out >= 1024 else "dequant_gemm"
        if m == 16:
            if out >= 4096:
                return "mmq"
            return "gemv" if out >= 1024 else "dequant_gemm"
        if m <= 31:
            if wide_output:
                return "online_f16"
            if out >= 4096 and out >= neuron_len:
                return "mmq"
            return "dequant_gemm"
        if m == 32:
            return "mmq" if out >= 4096 else "dequant_gemm"
        if m <= 47:
            return "online_f16" if wide_output else "dequant_gemm"
        if m == 48:
            return "mmq" if wide_output else "dequant_gemm"
        if m <= 63:
            return "online_f16" if wide_output else "dequant_gemm"
        if m == 64:
            if wide_output:
                return "online_f16"
            return "mmq" if out >= 8192 else "dequant_gemm"
        return "dequant_gemm"

    if m <= 13:
        return "gemv"
    if m == 14:
        return "gemv" if out >= 2048 else "dequant_gemm"
    if m == 15:
        if e8_family and out >= 8192:
            return "mmq"
        if e8_family and out >= 2048:
            return "gemv"
        return "dequant_gemm"
    if m == 16:
        if e8_family and out >= 6144:
            return "mmq"
        if e8_family and out >= 2048:
            return "gemv"
        if (
            storage_format in {3, 10, 11, 12, 15}
            and out >= 4096
            and neuron_len >= 8192
        ):
            return "mmq"
        return "dequant_gemm"

    wide_expansion = e8_family and out >= 3 * neuron_len
    if wide_expansion and (17 <= m <= 31 or 33 <= m <= 47):
        return "online_f16"
    if m == 32 and e8_family and out >= 8192:
        return "mmq"
    if m == 48 and e8_family and out >= 12288:
        return "mmq"
    if m == 64 and storage_format in {7, 9} and out >= 8192:
        return "mmq"
    return "dequant_gemm"


def nvq_matmul(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ W.T`` from a compact NPQ/NVQ matrix."""

    original = tuple(x.shape)
    if not original:
        raise ValueError("NVQ matmul input must have at least one dimension")
    x2 = x.reshape(-1, original[-1]).contiguous().to(torch.float16)
    neuron_len = int(g["neuron_len"])
    if x2.shape[1] > neuron_len:
        raise ValueError(f"x last dim {x2.shape[1]} exceeds NVQ neuron_len {neuron_len}")
    if x2.shape[1] < neuron_len:
        x2 = torch.nn.functional.pad(x2, (0, neuron_len - x2.shape[1]))

    m = int(x2.shape[0])
    path = _select_matmul_path(g, m)
    if path == "gemv":
        qx, xscale = _workspace(g, x2)
        y = ext().nvq_gemv_ws_cuda(
            *_kernel_args(g),
            x2,
            neuron_len,
            int(g["gs"]),
            int(g["sub_bits"]),
            int(g["format"]),
            int(g["sign_mode"]),
            qx,
            xscale,
        )
    elif path == "mmq":
        y = nvq_mmq(g, x2)
    elif path == "online_f16":
        y = nvq_gemm_f16(g, x2)
    else:
        weight = nvq_dequantize(g)
        y = ext().nint_cublas_gemm_nt_f16acc_cuda(x2, weight)
    return y.reshape(*original[:-1], int(g["out"]))


def nvq_grouped_matmul_pool(
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
) -> torch.Tensor:
    """Run one NVQ/NPQ cohort inside a global expert routing plan."""

    value = x.contiguous().to(device=g["device"], dtype=torch.float16)
    if value.dim() not in (2, 3) or value.shape[-1] != int(g["neuron_len"]):
        raise ValueError("NVQ routed input must have [T,K] or [T,R,K] shape")
    tokens, routes = (int(v) for v in route.ids.shape)
    if value.shape[0] != tokens or (value.dim() == 3 and value.shape[1] != routes):
        raise ValueError("NVQ input leading dimensions do not match routes")
    if int(g["out"]) % int(pool_experts):
        raise ValueError("NVQ cohort rows are not divisible by its expert count")
    out_per_expert = int(g["out"]) // int(pool_experts)
    expected = (tokens, routes, out_per_expert)
    if (
        tuple(out.shape) != expected
        or out.device != value.device
        or out.dtype != torch.float16
        or not out.is_contiguous()
    ):
        raise ValueError(f"NVQ output must be contiguous fp16 {expected}")
    input_rows = tokens * routes if value.dim() == 3 else tokens
    if qx is None or xscale is None:
        if input_quantized:
            raise ValueError("prequantized NVQ input requires qx and xscale")
        qx, xscale = _moe_workspace(g, value, input_rows)
    return ext().nvq_moe_grouped_matmul_pool_ws_cuda(
        *_kernel_args(g),
        value,
        route.ids,
        expert_local,
        int(route.n_experts),
        int(pool_experts),
        out_per_expert,
        int(g["neuron_len"]),
        int(g["gs"]),
        int(g["sub_bits"]),
        int(g["format"]),
        int(g["sign_mode"]),
        bool(input_quantized),
        out,
        qx,
        xscale,
        route.ids_dst,
        route.expert_bounds,
        route.tile_bounds,
        route.tile_experts,
    )


def nvq_matmul_input_mul(
    g: dict,
    x: torch.Tensor,
    gate: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """NVQ matmul with sigmoid/SiLU fused into direct activation quantization."""

    mode = _gate_mode(activation)
    if x.shape != gate.shape:
        raise ValueError(f"x and gate must have the same shape, got {tuple(x.shape)} and {tuple(gate.shape)}")
    original = tuple(x.shape)
    if not original:
        raise ValueError("NVQ gated matmul input must have at least one dimension")
    x2 = x.reshape(-1, original[-1]).contiguous().to(torch.float16)
    gate2 = gate.reshape(-1, original[-1]).contiguous().to(torch.float16)
    neuron_len = int(g["neuron_len"])
    if x2.shape[1] > neuron_len:
        raise ValueError(f"x last dim {x2.shape[1]} exceeds NVQ neuron_len {neuron_len}")
    if x2.shape[1] < neuron_len:
        pad = (0, neuron_len - x2.shape[1])
        x2 = torch.nn.functional.pad(x2, pad)
        gate2 = torch.nn.functional.pad(gate2, pad)

    m = int(x2.shape[0])
    path = _select_matmul_path(g, m)
    if path == "gemv":
        qx, xscale = _workspace(g, x2)
        y = ext().nvq_gemv_gate_ws_cuda(
            *_kernel_args(g),
            x2,
            gate2,
            neuron_len,
            int(g["gs"]),
            int(g["sub_bits"]),
            int(g["format"]),
            int(g["sign_mode"]),
            mode,
            qx,
            xscale,
        )
    elif path == "mmq":
        y = nvq_mmq_input_mul(g, x2, gate2, activation)
    else:
        value = x2 * (torch.sigmoid(gate2) if mode == 1 else torch.nn.functional.silu(gate2))
        y = nvq_matmul(g, value)
    return y.reshape(*original[:-1], int(g["out"]))


def nvq_embedding(g: dict, token_ids: torch.Tensor) -> torch.Tensor:
    """Decode only the selected NVQ embedding rows."""

    ids = token_ids.contiguous().to(device=g["neuron_scale"].device, dtype=torch.int64)
    return ext().nvq_embedding_lookup_cuda(
        *_kernel_args(g),
        ids,
        int(g["neuron_len"]),
        int(g["gs"]),
        int(g["sub_bits"]),
        int(g["format"]),
        int(g["sign_mode"]),
    )
