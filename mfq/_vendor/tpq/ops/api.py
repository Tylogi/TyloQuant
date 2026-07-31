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
_ACTIVATION_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_ROUTE_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}
_LINEAR_ROUTE_IMPLEMENTATIONS: dict[tuple[str, str], object] = {}


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
    if bits not in (8, 12, 14, 16) or not payloads:
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


def gated_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    activation: str,
    beta: float,
    linear_beta: float | None,
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
    packed_formats: tuple[str, ...] = ("p8", "p12", "p14"),
    code_dims: tuple[int, ...] = (4, 8),
    codebook_sizes: tuple[int, ...] = (256, 4096, 16384),
) -> torch.Tensor:
    """直接读取紧凑索引执行 Top-K MoE，不生成完整反量化矩阵。"""
    _ensure_builtins()
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
        beta=activation_beta,
        linear_beta=activation_linear_beta,
        hidden_workspace=hidden_workspace,
        out_workspace=output_workspace,
        result=result,
        p12_count=grouped_prefix,
    )
    if output is None:
        raise RuntimeError(
            f"算子 {REGISTRY.resolve(request).name} 拒绝了兼容输入"
        )
    return output


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
    bit_to_format = {8: "p8", 12: "p12", 14: "p14", 16: "p16"}
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
