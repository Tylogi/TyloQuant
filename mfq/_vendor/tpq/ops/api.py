"""模型运行时使用的公共算子入口。"""

from __future__ import annotations

import torch

from .registry import REGISTRY
from .spec import OperatorRequest


_BUILTINS_READY = False
_ATTENTION_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_NORMALIZATION_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_RESIDUAL_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_RESIDUAL_ADD_IMPLEMENTATIONS: dict[str, object] = {}
_HYPER_CONNECTION_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_ACTIVATION_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_ROUTE_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_LINEAR_ROUTE_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_BLOCK_SCALED_GEMV_IMPLEMENTATIONS: dict[
    tuple[str, str, int], object
] = {}
_BLOCK_SCALED_GROUPED_GEMV_IMPLEMENTATIONS: dict[
    tuple[str, str, int], object
] = {}
_RESIDENT_MOE_IMPLEMENTATIONS: dict[
    tuple[
        str,
        tuple[str, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
    object,
] = {}
_PACKED_ROUTE_SLOT_IMPLEMENTATIONS: dict[str, object] = {}


def linear(
    value: torch.Tensor,
    weight,
    *,
    output_dtype: torch.dtype | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """通用单 token Linear，直接读取 BF16 或紧凑 block-FP8 权重。

    ``ProjectionGroup`` 表示逻辑行拼接；每个成员保持自己的 128 行 scale
    原点，只拼接 token 大小的输出。这里不按模型名分派，也不生成完整反量化
    矩阵。调用方需要保持旧 ``F.linear`` dtype 时显式传 ``output_dtype``。
    """
    from ..kernels import BlockFP8Weight, ProjectionGroup

    if isinstance(weight, BlockFP8Weight):
        result = weight.matmul_T_decode_fused(value, output=output)
    elif isinstance(weight, ProjectionGroup):
        result = weight.matmul_T_decode_fused(value, output=output)
    else:
        result = torch.nn.functional.linear(value.to(weight.dtype), weight)
        if output is not None:
            output.copy_(result)
            result = output
    return result if output_dtype is None else result.to(output_dtype)


def _projection_layout_tag(
    packed_formats: tuple[str, ...],
    code_dims: tuple[int, ...],
    codebook_sizes: tuple[int, ...],
) -> int:
    """Select a CUDA fast path from the exact public capability tuple."""
    if not (
        len(packed_formats) == 3
        and len(code_dims) == 3
        and len(codebook_sizes) == 3
    ):
        return 0
    # Tag 2 enables the paired p10 shared-codebook specialization.  All other
    # three-projection layouts use tag 1 and therefore avoid reserving its
    # 40 KiB scratch area when either Gate or Up is not p10.
    return 2 if packed_formats[:2] == ("p10", "p10") else 1


def _ensure_builtins() -> None:
    global _BUILTINS_READY
    if _BUILTINS_READY:
        return
    from . import cpu_backend, cuda_backend

    cpu_backend.register(REGISTRY)
    cuda_backend.register(REGISTRY)
    _BUILTINS_READY = True


def vq_gemv(
    x_rows: torch.Tensor,
    indices: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor | None:
    """按能力分派 VQ GEMV；不在此接口中展开或反量化权重。"""
    _ensure_builtins()
    if indices.dtype == torch.uint8:
        packed_format = "u8"
    elif indices.dtype == torch.uint16:
        packed_format = "u16"
    else:
        return None
    request = OperatorRequest(
        operation="vq_gemv",
        device_type=x_rows.device.type,
        packed_formats=(packed_format,),
        code_dims=(int(codebook.shape[-1]),),
        codebook_sizes=(int(codebook.shape[-2]),),
        activation="none",
        top_k=1,
        batch_size=max(
            int(x_rows.shape[0]),
            int(indices.shape[0]),
        ),
    )
    try:
        return REGISTRY.call(
            request,
            x_rows=x_rows,
            indices=indices,
            codebook=codebook,
        )
    except LookupError:
        return None


def vq_gemv_packed_list(
    x_rows: torch.Tensor,
    payloads: list[torch.Tensor],
    codebook: torch.Tensor,
    rows: int,
    blocks: int,
    bits: int,
    *,
    allow_direct: bool = False,
) -> torch.Tensor | None:
    """Dispatch compact list-backed VQ without expanding packed indices."""
    _ensure_builtins()
    if not 8 <= bits <= 16 or not payloads:
        return None
    request = OperatorRequest(
        operation="vq_gemv:list",
        device_type=x_rows.device.type,
        packed_formats=(f"p{bits}",),
        code_dims=(int(codebook.shape[-1]),),
        codebook_sizes=(int(codebook.shape[-2]),),
        activation="none",
        top_k=len(payloads),
        batch_size=len(payloads),
    )
    try:
        return REGISTRY.call(
            request,
            x_rows=x_rows,
            payloads=payloads,
            codebook=codebook,
            rows=int(rows),
            blocks=int(blocks),
            bits=int(bits),
            allow_direct=bool(allow_direct),
        )
    except LookupError:
        return None


def block_scaled_gemv(
    value: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """直接读取块缩放紧凑权重执行单 token GEMV。"""
    _ensure_builtins()
    if weights.dtype != torch.uint8 or scales.dtype != torch.float32:
        return None
    format_name = "e4m3fn"
    key = (value.device.type, format_name, int(block_size))
    try:
        implementation = _BLOCK_SCALED_GEMV_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="block_scaled_gemv",
                device_type=value.device.type,
                packed_formats=(format_name,),
                code_dims=(int(block_size),),
                activation="none",
                top_k=1,
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _BLOCK_SCALED_GEMV_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            weights=weights,
            scales=scales,
            cols=int(weights.shape[1]),
            block_size=int(block_size),
            output=output,
        )
    except LookupError:
        return None


def block_scaled_grouped_gemv(
    value: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    row_offsets: torch.Tensor,
    *,
    total_rows: int,
    cols: int,
    block_size: int = 128,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """One-token logical row concatenation over compact block-FP8 weights.

    Pointer metadata is a fixed-address device plan.  The underlying compact
    payload remains owned by its original weights, so this public operation
    neither concatenates nor dequantizes a complete matrix.
    """
    _ensure_builtins()
    key = (value.device.type, "e4m3fn", int(block_size))
    try:
        implementation = _BLOCK_SCALED_GROUPED_GEMV_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="block_scaled_grouped_gemv",
                device_type=value.device.type,
                packed_formats=("e4m3fn",),
                code_dims=(int(block_size),),
                activation="none",
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _BLOCK_SCALED_GROUPED_GEMV_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            weight_ptrs=weight_ptrs,
            scale_ptrs=scale_ptrs,
            row_offsets=row_offsets,
            total_rows=int(total_rows),
            cols=int(cols),
            block_size=int(block_size),
            output=output,
        )
    except LookupError:
        return None


def route_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    *,
    scoring_func: str,
    top_k: int,
    normalize: bool,
    scaling: float,
    n_group: int = 1,
    topk_group: int = 1,
    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """配置驱动的 Top-K 路由；注册键只描述数学与设备能力。"""
    _ensure_builtins()
    if int(n_group) != 1 or int(topk_group) != 1:
        return None
    normalized_scoring = scoring_func.strip().lower()
    key = (logits.device.type, normalized_scoring)
    try:
        implementation = _ROUTE_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="route_topk",
                device_type=logits.device.type,
                activation=normalized_scoring,
                top_k=int(top_k),
                batch_size=int(logits.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _ROUTE_IMPLEMENTATIONS[key] = implementation
        return implementation(
            logits=logits,
            bias=bias,
            mask=mask,
            top_k=int(top_k),
            normalize=bool(normalize),
            scaling=float(scaling),
            output_buffers=output_buffers,
        )
    except LookupError:
        return None


def linear_route_topk(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    *,
    scoring_func: str,
    top_k: int,
    normalize: bool,
    scaling: float,
    n_group: int = 1,
    topk_group: int = 1,
    output_buffers: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """通用线性投影与 Top-K 路由接口，保持路由权重的源生精度。"""
    _ensure_builtins()
    if (
        int(n_group) != 1
        or int(topk_group) != 1
        or not normalize
    ):
        return None
    normalized_scoring = scoring_func.strip().lower()
    key = (value.device.type, normalized_scoring)
    try:
        implementation = _LINEAR_ROUTE_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="linear_route_topk",
                device_type=value.device.type,
                activation=normalized_scoring,
                top_k=int(top_k),
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _LINEAR_ROUTE_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            weight=weight,
            bias=bias,
            mask=mask,
            top_k=int(top_k),
            scaling=float(scaling),
            output_buffers=output_buffers,
        )
    except LookupError:
        return None


def attention_step(kind: str, device_type: str, **kwargs):
    """配置驱动的 Attention 注册入口。

    各注意力数学实现按 ``kind`` 注册；公共运行时不按模型名称分派。
    """
    _ensure_builtins()
    normalized_kind = kind.strip().lower()
    normalized_device = device_type.strip().lower()
    key = (normalized_kind, normalized_device)
    implementation = _ATTENTION_IMPLEMENTATIONS.get(key)
    if implementation is None:
        request = OperatorRequest(
            operation=f"attention_step:{normalized_kind}",
            device_type=normalized_device,
            activation="none",
        )
        implementation = REGISTRY.resolve(request).implementation
        _ATTENTION_IMPLEMENTATIONS[key] = implementation
    return implementation(**kwargs)


def rmsnorm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """设备无关的 RMSNorm 注册入口。"""
    _ensure_builtins()
    try:
        batch_size = max(1, value.numel() // value.shape[-1])
        key = (value.device.type, str(batch_size))
        implementation = _NORMALIZATION_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="normalization:rmsnorm",
                device_type=value.device.type,
                activation="none",
                batch_size=batch_size,
            )
            implementation = REGISTRY.resolve(request).implementation
            _NORMALIZATION_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            weight=weight,
            eps=float(eps),
            output=output,
        )
    except LookupError:
        return None


def residual_mix(
    kind: str,
    prefix: torch.Tensor,
    residual: torch.Tensor,
    projection: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    output: torch.Tensor | None = None,
    post_norm_weight: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    residual_inverse: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """配置驱动的残差合并入口。"""
    _ensure_builtins()
    normalized_kind = kind.strip().lower()
    key = (normalized_kind, prefix.device.type)
    try:
        implementation = _RESIDUAL_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation=f"residual_mix:{normalized_kind}",
                device_type=prefix.device.type,
                activation="none",
                batch_size=int(residual.shape[-2]) + 1,
            )
            implementation = REGISTRY.resolve(request).implementation
            _RESIDUAL_IMPLEMENTATIONS[key] = implementation
        return implementation(
            prefix=prefix,
            residual=residual,
            projection=projection,
            norm_weight=norm_weight,
            eps=float(eps),
            output=output,
            post_norm_weight=post_norm_weight,
            score_workspace=workspace,
            residual_inverse=residual_inverse,
        )
    except LookupError:
        return None


def residual_add3(
    residual: torch.Tensor,
    routed: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor | None:
    """按源 dtype 顺序计算 ``residual + (routed + shared)``。"""
    _ensure_builtins()
    device_type = residual.device.type
    try:
        implementation = _RESIDUAL_ADD_IMPLEMENTATIONS.get(device_type)
        if implementation is None:
            request = OperatorRequest(
                operation="residual_add:three_way",
                device_type=device_type,
                activation="none",
                batch_size=max(
                    1,
                    residual.numel() // residual.shape[-1],
                ),
            )
            implementation = REGISTRY.resolve(request).implementation
            _RESIDUAL_ADD_IMPLEMENTATIONS[device_type] = implementation
        return implementation(
            residual=residual,
            routed=routed,
            shared=shared,
        )
    except LookupError:
        return None


def _hyper_connection_implementation(
    operation: str,
    value: torch.Tensor,
    *,
    activation: str,
    batch_size: int,
):
    """Resolve one model-agnostic Hyper-Connection decode capability."""
    _ensure_builtins()
    key = (operation, value.device.type)
    implementation = _HYPER_CONNECTION_IMPLEMENTATIONS.get(key)
    if implementation is None:
        request = OperatorRequest(
            operation=f"hyper_connection:{operation}",
            device_type=value.device.type,
            activation=activation,
            batch_size=max(1, int(batch_size)),
        )
        implementation = REGISTRY.resolve(request).implementation
        _HYPER_CONNECTION_IMPLEMENTATIONS[key] = implementation
    return implementation


def hyper_connection_pre_norm(
    value: torch.Tensor,
    projection: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    sinkhorn_iters: int,
    eps: float,
    *,
    output_buffers: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None = None,
):
    """Fuse H/C input projection, Sinkhorn reduction and RMSNorm.

    The key describes mathematical capability rather than a model family.
    Caller-owned buffers keep decode addresses stable for every compatible
    configuration without forcing a model-specific implementation.
    """
    try:
        implementation = _hyper_connection_implementation(
            "pre_norm",
            value,
            activation="rmsnorm",
            batch_size=value.numel() // (4 * value.shape[-1]),
        )
        return implementation(
            value=value,
            projection=projection,
            scale=scale,
            base=base,
            norm_weight=norm_weight,
            sinkhorn_iters=int(sinkhorn_iters),
            eps=float(eps),
            output_buffers=output_buffers,
        )
    except LookupError:
        return None


def hyper_connection_post(
    value: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    combine: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
):
    """Apply one H/C post mix, optionally into a stable output buffer."""
    try:
        implementation = _hyper_connection_implementation(
            "post",
            residual,
            activation="none",
            batch_size=residual.numel() // (4 * residual.shape[-1]),
        )
        return implementation(
            value=value,
            residual=residual,
            post=post,
            combine=combine,
            output=output,
        )
    except LookupError:
        return None


def hyper_connection_post_moe(
    routed: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    combine: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
):
    """Fuse routed/shared BF16 merge and H/C post without temporaries."""
    try:
        implementation = _hyper_connection_implementation(
            "post_moe",
            residual,
            activation="none",
            batch_size=residual.numel() // (4 * residual.shape[-1]),
        )
        return implementation(
            routed=routed,
            shared=shared,
            residual=residual,
            post=post,
            combine=combine,
            output=output,
        )
    except LookupError:
        return None


def gated_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    activation: str,
    beta: float,
    linear_beta: float | None,
    limit: float = 0.0,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """按激活能力选择融合 Gate×Up 算子。"""
    _ensure_builtins()
    normalized = activation.strip().lower()
    key = (gate.device.type, normalized)
    try:
        implementation = _ACTIVATION_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="gated_activation",
                device_type=gate.device.type,
                activation=normalized,
                batch_size=max(1, gate.numel() // gate.shape[-1]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _ACTIVATION_IMPLEMENTATIONS[key] = implementation
        return implementation(
            gate=gate,
            up=up,
            activation=normalized,
            beta=float(beta),
            linear_beta=linear_beta,
            limit=float(limit),
            output=output,
        )
    except LookupError:
        return None


def packed_moe_topk(
    value: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    metadata: torch.Tensor,
    *,
    activation: str,
    activation_beta: float,
    activation_linear_beta: float,
    hidden_workspace: torch.Tensor,
    output_workspace: torch.Tensor,
    result: torch.Tensor,
    grouped_prefix: int,
    packed_formats: tuple[str, ...] | None = None,
    code_dims: tuple[int, ...] | None = None,
    codebook_sizes: tuple[int, ...] | None = None,
    limit: float = 0.0,
) -> torch.Tensor:
    """直接读取紧凑索引执行 Top-K MoE，不生成完整反量化矩阵。"""
    _ensure_builtins()
    projection_vq = (
        metadata.ndim == 2 and metadata.shape[0] == 15
    )
    if packed_formats is None:
        packed_formats = (
            tuple(f"p{bits}" for bits in range(8, 17))
            if projection_vq
            else ("p8", "p12", "p14")
        )
    if code_dims is None:
        code_dims = (4, 8, 16) if projection_vq else (4, 8)
    if codebook_sizes is None:
        codebook_sizes = (
            (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
            if projection_vq
            else (256, 4096, 16384)
        )
    projection_layout_tag = (
        _projection_layout_tag(
            packed_formats,
            code_dims,
            codebook_sizes,
        )
        if projection_vq
        else 0
    )
    request = OperatorRequest(
        operation="moe_topk",
        device_type=value.device.type,
        packed_formats=packed_formats,
        code_dims=code_dims,
        codebook_sizes=codebook_sizes,
        activation=activation,
        top_k=route_ids.numel(),
        batch_size=value.shape[0],
    )
    output = REGISTRY.call(
        request,
        value=value,
        route_ids=route_ids,
        weights=route_weights,
        metadata=metadata,
        activation=activation,
        beta=activation_beta,
        linear_beta=activation_linear_beta,
        limit=float(limit),
        hidden_workspace=hidden_workspace,
        out_workspace=output_workspace,
        result=result,
        p12_count=grouped_prefix,
        projection_layout_tag=projection_layout_tag,
    )
    if output is None:
        raise RuntimeError(
            f"算子 {REGISTRY.resolve(request).name} 拒绝了兼容输入"
        )
    return output


def packed_route_slots(
    route_ids: torch.Tensor,
    directory: torch.Tensor,
    *,
    output: torch.Tensor,
    hit_mask: torch.Tensor,
) -> bool:
    """Map Top-K expert IDs to stable packed-slot metadata on the device.

    ``directory`` is ``[expert_count, metadata_rows]`` and remains compact:
    it contains only fixed slot pointers and VQ shape tags, never expanded
    expert indices or dequantized matrices.  The operation is model-agnostic
    and graph-safe because all output buffers are supplied by the caller.
    """
    _ensure_builtins()
    device_type = route_ids.device.type
    try:
        implementation = _PACKED_ROUTE_SLOT_IMPLEMENTATIONS.get(device_type)
        if implementation is None:
            request = OperatorRequest(
                operation="packed_route_slots",
                device_type=device_type,
                activation="none",
                top_k=int(route_ids.numel()),
                batch_size=1,
            )
            implementation = REGISTRY.resolve(request).implementation
            _PACKED_ROUTE_SLOT_IMPLEMENTATIONS[device_type] = implementation
        return bool(
            implementation(
                route_ids=route_ids,
                directory=directory,
                output=output,
                hit_mask=hit_mask,
            )
        )
    except LookupError:
        return False


def packed_moe_operator_name(
    *,
    device_type: str,
    activation: str,
    top_k: int,
    packed_formats: tuple[str, ...],
    code_dims: tuple[int, ...],
    codebook_sizes: tuple[int, ...],
    batch_size: int = 1,
) -> str:
    """Resolve the public packed MoE backend for CLI diagnostics."""
    _ensure_builtins()
    return REGISTRY.resolve(
        OperatorRequest(
            operation="moe_topk",
            device_type=str(device_type),
            packed_formats=packed_formats,
            code_dims=code_dims,
            codebook_sizes=codebook_sizes,
            activation=str(activation),
            top_k=int(top_k),
            batch_size=int(batch_size),
        )
    ).name


def packed_moe_selected_topk(
    value: torch.Tensor,
    experts,
    route_weights: torch.Tensor,
    *,
    activation: str,
    activation_beta: float,
    activation_linear_beta: float | None,
    limit: float = 0.0,
) -> torch.Tensor | None:
    """Execute selected packed experts through the common ``moe_topk`` op.

    This is the RAM/LRU form of the resident-metadata interface above.  Model
    code supplies logical packed weights; the selected backend owns decoding,
    fusion and workspace policy.
    """
    if not experts:
        return None
    _ensure_builtins()
    bit_to_format = {
        8: "p8",
        9: "p9",
        10: "p10",
        12: "p12",
        14: "p14",
        16: "p16",
    }
    packed_formats = tuple(
        sorted(
            {
                bit_to_format[int(weight.bits)]
                for pair in experts
                for weight in pair
            }
        )
    )
    code_dims = tuple(
        sorted(
            {
                int(weight.dim)
                for pair in experts
                for weight in pair
            }
        )
    )
    codebook_sizes = tuple(
        sorted(
            {
                int(weight.cb.shape[0])
                for pair in experts
                for weight in pair
            }
        )
    )
    request = OperatorRequest(
        operation="moe_topk",
        device_type=value.device.type,
        packed_formats=packed_formats,
        code_dims=code_dims,
        codebook_sizes=codebook_sizes,
        activation=activation,
        top_k=len(experts),
        batch_size=value.shape[0],
    )
    try:
        return REGISTRY.call(
            request,
            value=value,
            experts=experts,
            weights=route_weights,
            limit=float(limit),
            activation=activation,
            beta=float(activation_beta),
            linear_beta=activation_linear_beta,
        )
    except LookupError:
        return None


def resident_moe_topk(
    value: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    metadata: torch.Tensor,
    *,
    activation: str,
    limit: float,
    codegemm_gu_workspace: torch.Tensor | None,
    codegemm_activation_workspace: torch.Tensor | None,
    codegemm_down_workspace: torch.Tensor | None,
    hidden_workspace: torch.Tensor,
    output_workspace: torch.Tensor,
    result: torch.Tensor,
    packed_formats: tuple[str, ...],
    code_dims: tuple[int, ...],
    codebook_sizes: tuple[int, ...],
) -> torch.Tensor | None:
    """Run mixed resident codebooks without a host-side route split.

    This interface describes storage and math capabilities only.  It is shared
    by every model configuration using a SwiGLU Top-K resident expert layout.
    """
    _ensure_builtins()
    normalized_formats = tuple(sorted(set(packed_formats)))
    normalized_dims = tuple(sorted(set(int(v) for v in code_dims)))
    normalized_sizes = tuple(sorted(set(int(v) for v in codebook_sizes)))
    key = (
        value.device.type,
        normalized_formats,
        normalized_dims,
        normalized_sizes,
    )
    try:
        implementation = _RESIDENT_MOE_IMPLEMENTATIONS.get(key)
        if implementation is None:
            request = OperatorRequest(
                operation="resident_moe_topk",
                device_type=value.device.type,
                packed_formats=normalized_formats,
                code_dims=normalized_dims,
                codebook_sizes=normalized_sizes,
                activation=activation,
                top_k=int(route_ids.numel()),
                batch_size=int(value.shape[0]),
            )
            implementation = REGISTRY.resolve(request).implementation
            _RESIDENT_MOE_IMPLEMENTATIONS[key] = implementation
        return implementation(
            value=value,
            route_ids=route_ids,
            weights=route_weights,
            metadata=metadata,
            limit=float(limit),
            codegemm_gu_workspace=codegemm_gu_workspace,
            codegemm_activation_workspace=(
                codegemm_activation_workspace
            ),
            codegemm_down_workspace=codegemm_down_workspace,
            hidden_workspace=hidden_workspace,
            output_workspace=output_workspace,
            result=result,
            include_k4096=4096 in normalized_sizes,
        )
    except LookupError:
        return None


def create_tensor_parallel(
    kind: str,
    devices: tuple[torch.device, ...],
    spec,
):
    """通过公共能力注册创建有状态 TP executor。"""
    _ensure_builtins()
    normalized = kind.strip().lower()
    activation = str(getattr(spec, "activation", "none")).lower()
    request = OperatorRequest(
        operation=f"tensor_parallel:{normalized}",
        device_type="cuda",
        activation=activation,
        top_k=1,
        batch_size=1,
    )
    return REGISTRY.call(
        request,
        kind=normalized,
        devices=tuple(devices),
        spec=spec,
    )
