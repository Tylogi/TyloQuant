"""CUDA 通用 packed 算子注册。"""

from __future__ import annotations

import torch

from .registry import OperatorRegistry
from .spec import OperatorCapability


def _vq_gemv(**kwargs):
    from ..fusedext import vq_gemv_fused

    return vq_gemv_fused(
        kwargs["x_rows"],
        kwargs["indices"],
        kwargs["codebook"],
    )


def _packed_moe_topk(**kwargs):
    from ..fusedext import packed_moe_topk_fused

    return packed_moe_topk_fused(**kwargs)


def _create_tensor_parallel(*, kind, devices, spec):
    from .tensor_parallel import (
        TensorParallelGatedMLP,
        TensorParallelKDA,
        TensorParallelMLA,
        TensorParallelMoEPrelude,
        TensorParallelRouteDown,
        TensorParallelRowLinear,
    )

    executors = {
        "gated_mlp": TensorParallelGatedMLP,
        "kda": TensorParallelKDA,
        "mla": TensorParallelMLA,
        "moe_prelude": TensorParallelMoEPrelude,
        "route_down": TensorParallelRouteDown,
        "row_linear": TensorParallelRowLinear,
    }
    try:
        executor_type = executors[kind]
    except KeyError as error:
        raise ValueError(
            f"unsupported tensor-parallel executor {kind!r}"
        ) from error
    return executor_type(tuple(devices), spec)


def _route_topk(**kwargs):
    from ..fusedext import route_topk_sigmoid_fused

    if not kwargs["normalize"]:
        return None
    return route_topk_sigmoid_fused(
        kwargs["logits"],
        kwargs["bias"],
        kwargs["mask"],
        kwargs["top_k"],
        kwargs["scaling"],
        kwargs.get("output_buffers"),
    )


def _linear_route_topk(**kwargs):
    from ..fusedext import linear_route_topk_sigmoid_fused

    value = kwargs["value"]
    with torch.cuda.device(value.device):
        return linear_route_topk_sigmoid_fused(
            value,
            kwargs["weight"],
            kwargs["bias"],
            kwargs["mask"],
            kwargs["top_k"],
            kwargs["scaling"],
            kwargs["output_buffers"],
        )


def _short_conv3(**kwargs):
    from ..fusedext import short_conv3_fused

    return short_conv3_fused(
        kwargs["query"],
        kwargs["key"],
        kwargs["value"],
        kwargs["states"],
        kwargs["weights"],
    )


def _kda_recurrent(**kwargs):
    from ..fusedext import kda_recurrent_fused

    return kda_recurrent_fused(**kwargs)


def _gated_rmsnorm(**kwargs):
    from ..fusedext import gated_rmsnorm_fused

    return gated_rmsnorm_fused(**kwargs)


def _paged_latent_create(**kwargs):
    from ..flashinfer_mla import create_runner

    return create_runner(**kwargs)


def _paged_latent_prepare(**kwargs):
    from ..flashinfer_mla import prepare_runner

    return prepare_runner(
        kwargs["runner"],
        int(kwargs["length"]),
    )


def _paged_latent_decode(**kwargs):
    from ..flashinfer_mla import decode

    return decode(
        kwargs["runner"],
        kwargs["query_nope"],
        kwargs["query_rope"],
        kwargs["latent_cache"],
        kwargs["rope_cache"],
    )


def _latent_mla_decode(**kwargs):
    from ..fusedext import latent_mla_attention_decode_fused

    return latent_mla_attention_decode_fused(
        kwargs["query_nope"],
        kwargs["query_rope"],
        kwargs["latent_cache"],
        kwargs["rope_cache"],
        kwargs["position"],
        float(kwargs["scale_denominator"]),
        kwargs["score_workspace"],
        kwargs.get("output"),
    )


def _rmsnorm(**kwargs):
    from ..fusedext import rmsnorm_bf16_fused, rmsnorm_fused

    value = kwargs["value"]
    implementation = (
        rmsnorm_bf16_fused
        if value.dtype == torch.bfloat16
        else rmsnorm_fused
    )
    with torch.cuda.device(value.device):
        return implementation(
            value,
            kwargs["weight"],
            kwargs["eps"],
            kwargs.get("output"),
        )


def _attention_residual(**kwargs):
    from ..fusedext import attention_residual_bf16_fused

    return attention_residual_bf16_fused(**kwargs)


def _residual_add3(**kwargs):
    from ..fusedext import residual_add3_fused

    return residual_add3_fused(**kwargs)


def _gated_activation(**kwargs):
    from ..fusedext import gated_activation_bf16_fused

    return gated_activation_bf16_fused(**kwargs)


def register(registry: OperatorRegistry) -> None:
    registry.register(
        "cuda.residual_add.three_way.decode",
        OperatorCapability(
            operation="residual_add:three_way",
            device_types=("cuda",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(1, 257)),
        ),
        _residual_add3,
        priority=100,
    )
    registry.register(
        "cuda.linear_route_topk.sigmoid.decode",
        OperatorCapability(
            operation="linear_route_topk",
            device_types=("cuda",),
            activations=("sigmoid",),
            max_top_k=16,
            batch_sizes=(1,),
        ),
        _linear_route_topk,
        priority=100,
    )
    registry.register(
        "cuda.route_topk.sigmoid.decode",
        OperatorCapability(
            operation="route_topk",
            device_types=("cuda",),
            activations=("sigmoid",),
            max_top_k=16,
            batch_sizes=(1,),
        ),
        _route_topk,
        priority=100,
    )
    for name, kind, implementation in (
        (
            "cuda.attention.short_conv3.bf16",
            "short_conv3",
            _short_conv3,
        ),
        (
            "cuda.attention.kda_recurrent.decode",
            "kda_recurrent",
            _kda_recurrent,
        ),
        (
            "cuda.attention.gated_rmsnorm.decode",
            "gated_rmsnorm",
            _gated_rmsnorm,
        ),
        (
            "cuda.attention.paged_latent.create",
            "paged_latent_create",
            _paged_latent_create,
        ),
        (
            "cuda.attention.paged_latent.prepare",
            "paged_latent_prepare",
            _paged_latent_prepare,
        ),
        (
            "cuda.attention.paged_latent.decode",
            "paged_latent_decode",
            _paged_latent_decode,
        ),
        (
            "cuda.attention.compressed_kv.decode",
            "compressed_kv_decode",
            _latent_mla_decode,
        ),
    ):
        registry.register(
            name,
            OperatorCapability(
                operation=f"attention_step:{kind}",
                device_types=("cuda",),
                activations=("none",),
                max_top_k=1,
                batch_sizes=(1,),
            ),
            implementation,
            priority=100,
        )
    registry.register(
        "cuda.normalization.rmsnorm.decode",
        OperatorCapability(
            operation="normalization:rmsnorm",
            device_types=("cuda",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(1, 257)),
        ),
        _rmsnorm,
        priority=100,
    )
    registry.register(
        "cuda.residual_mix.attention.decode",
        OperatorCapability(
            operation="residual_mix:attention",
            device_types=("cuda",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(2, 33)),
        ),
        _attention_residual,
        priority=100,
    )
    registry.register(
        "cuda.gated_activation.bf16.decode",
        OperatorCapability(
            operation="gated_activation",
            device_types=("cuda",),
            activations=("silu", "swiglu", "situ"),
            max_top_k=1,
            batch_sizes=tuple(range(1, 257)),
        ),
        _gated_activation,
        priority=100,
    )
    registry.register(
        "cuda.vq_gemv.index_tensor.batch",
        OperatorCapability(
            operation="vq_gemv",
            device_types=("cuda",),
            packed_formats=("u8", "u16"),
            code_dims=(4, 8),
            codebook_sizes=(256, 4096, 16384),
            activations=("none",),
            max_top_k=1,
            batch_sizes=tuple(range(1, 17)),
        ),
        _vq_gemv,
        priority=50,
    )
    registry.register(
        "cuda.packed_moe_topk.situ.batch1",
        OperatorCapability(
            operation="moe_topk",
            device_types=("cuda",),
            packed_formats=("p8", "p12", "p14"),
            code_dims=(4, 8),
            codebook_sizes=(256, 4096, 16384),
            activations=("situ",),
            max_top_k=16,
            batch_sizes=(1,),
        ),
        _packed_moe_topk,
        priority=100,
    )
    for kind, activations in (
        ("gated_mlp", ("silu", "swiglu", "situ")),
        ("kda", ("none",)),
        ("mla", ("none",)),
        ("moe_prelude", ("silu", "swiglu", "situ")),
        ("route_down", ("none",)),
        ("row_linear", ("none",)),
    ):
        registry.register(
            f"cuda.tensor_parallel.{kind}.decode",
            OperatorCapability(
                operation=f"tensor_parallel:{kind}",
                device_types=("cuda",),
                activations=activations,
                max_top_k=1,
                batch_sizes=(1,),
            ),
            _create_tensor_parallel,
            priority=100,
        )
