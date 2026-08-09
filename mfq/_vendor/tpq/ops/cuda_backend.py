"""Register generic CUDA packed operators."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import OperatorRegistry
from .spec import OperatorCapability


def _vq_gemv(**kwargs):
    from ..fusedext import vq_gemv_fused

    x_rows = kwargs["x_rows"]
    # The extension obtains its stream from the current CUDA device. When the pipeline switches to cuda:1+,
    # tensor movement does not automatically change Python's current device. Without an explicit guard,
    # a kernel holding pointers to another device would launch on the cuda:0 stream and cause illegal memory access.
    with torch.cuda.device(x_rows.device):
        return vq_gemv_fused(
            x_rows,
            kwargs["indices"],
            kwargs["codebook"],
        )


def _block_scaled_gemv(**kwargs):
    from ..fusedext import block_fp8_gemv_fused

    value = kwargs["value"]
    with torch.cuda.device(value.device):
        return block_fp8_gemv_fused(
            value,
            kwargs["weights"],
            kwargs["scales"],
            int(kwargs["cols"]),
            int(kwargs["block_size"]),
            output=kwargs.get("output"),
        )


def _dense_gemv(**kwargs):
    from ..fusedext import bf16_gemv_fused

    value = kwargs["value"]
    with torch.cuda.device(value.device):
        return bf16_gemv_fused(
            value,
            kwargs["weight"],
            kwargs["output"],
        )


def _block_scaled_grouped_gemv(**kwargs):
    from ..fusedext import block_fp8_grouped_gemv_fused

    value = kwargs["value"]
    with torch.cuda.device(value.device):
        return block_fp8_grouped_gemv_fused(
            value,
            kwargs["weight_ptrs"],
            kwargs["scale_ptrs"],
            kwargs["row_offsets"],
            int(kwargs["total_rows"]),
            int(kwargs["cols"]),
            int(kwargs["block_size"]),
            output=kwargs.get("output"),
        )


def _compressed_state_update(**kwargs):
    from ..fusedext import compressed_state_update_fused

    projected = kwargs["projected"]
    with torch.cuda.device(projected.device):
        return compressed_state_update_fused(
            projected,
            kwargs["ape"],
            kwargs["ckv"],
            kwargs["cscore"],
            int(kwargs["ratio"]),
            int(kwargs["position"]),
            int(kwargs["kv_rows"]),
        )


def _head_rmsnorm_rope(**kwargs):
    from ..fusedext import head_rmsnorm_rope_fused

    rows = kwargs["rows"]
    with torch.cuda.device(rows.device):
        return head_rmsnorm_rope_fused(
            rows,
            kwargs["weight"],
            kwargs["cos"],
            kwargs["sin"],
            int(kwargs["rope_width"]),
            float(kwargs["eps"]),
        )


def _packed_moe_topk(**kwargs):
    from ..fusedext import packed_moe_topk_fused

    return packed_moe_topk_fused(**kwargs)


def _packed_route_slots(**kwargs):
    from ..fusedext import packed_route_slots_fused

    route_ids = kwargs["route_ids"]
    with torch.cuda.device(route_ids.device):
        return packed_route_slots_fused(
            route_ids,
            kwargs["directory"],
            kwargs["output"],
            kwargs["hit_mask"],
        )


def _resident_moe_topk(**kwargs):
    """Compose resident u8 Psumbook and u16/K4096 experts in one call.

    Both kernels consume the same route IDs and metadata.  The CUDA backend
    filters by metadata dtype tag, so mixed layers need no host-side route
    split or synchronization.
    """
    from ..fusedext import (
        moe_mlp_routed_codegemm_fused,
        moe_mlp_routed_vv_fused,
    )

    value = kwargs["value"]
    with torch.cuda.device(value.device):
        output = moe_mlp_routed_codegemm_fused(
            value,
            kwargs["route_ids"],
            kwargs["weights"],
            kwargs["metadata"],
            kwargs["codegemm_gu_workspace"],
            kwargs["codegemm_activation_workspace"],
            kwargs["codegemm_down_workspace"],
            kwargs["result"],
        )
        if output is None:
            return None
        if not kwargs.get("include_k4096", False):
            return output
        return moe_mlp_routed_vv_fused(
            value,
            kwargs["route_ids"],
            kwargs["weights"],
            kwargs["metadata"],
            float(kwargs["limit"]),
            kwargs["hidden_workspace"],
            kwargs["output_workspace"],
            kwargs["result"],
            accumulate=True,
        )


def _resident_moe_topk_generic(**kwargs):
    """Run row-major u8/u16 resident experts through the proven VQ kernel."""
    from ..fusedext import moe_mlp_routed_slots_fused

    value = kwargs["value"]
    with torch.cuda.device(value.device):
        return moe_mlp_routed_slots_fused(
            value,
            kwargs["route_ids"],
            kwargs["weights"],
            kwargs["metadata"],
            float(kwargs["limit"]),
            kwargs["hidden_workspace"],
            kwargs["output_workspace"],
            kwargs["result"],
        )


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


def _linear_route_topk_sqrtsoftplus(**kwargs):
    """Source-native sqrt(softplus) router with fixed CLI/Graph buffers."""
    value = kwargs["value"]
    weight = kwargs["weight"]
    bias = kwargs["bias"]
    mask = kwargs["mask"]
    logits, output_weights, output_indices = kwargs["output_buffers"]
    if (
        not value.is_cuda
        or value.shape != (1, weight.shape[1])
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or mask.dtype != torch.bool
    ):
        return None
    with torch.cuda.device(value.device):
        scores = F.softplus(F.linear(value.float(), weight)).sqrt()
        logits.copy_(scores)
        choice = scores.add(bias).masked_fill(~mask[None], -1e30)
        indices = choice.topk(int(kwargs["top_k"]), dim=-1).indices
        weights = scores.gather(1, indices)
        weights.div_(weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights.mul_(float(kwargs["scaling"]))
        output_indices.copy_(indices)
        output_weights.copy_(weights)
    return output_weights, output_indices


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


def _sliding_compressed_mqa_decode(**kwargs):
    from ..fusedext import dsv4_attn_decode_fused

    return dsv4_attn_decode_fused(
        kwargs["query"],
        kwargs["window_kv"],
        kwargs["window_positions"],
        kwargs["compressed_kv"],
        kwargs["sink"],
        kwargs["cos"],
        kwargs["sin"],
        float(kwargs["scale"]),
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


def _hyper_connection_pre_norm(**kwargs):
    from ..fusedext import dsv4_hc_pre_norm_fused

    value = kwargs["value"]
    with torch.cuda.device(value.device):
        return dsv4_hc_pre_norm_fused(
            value,
            kwargs["projection"],
            kwargs["scale"],
            kwargs["base"],
            kwargs["norm_weight"],
            kwargs["sinkhorn_iters"],
            kwargs["eps"],
            output_buffers=kwargs.get("output_buffers"),
        )


def _hyper_connection_post(**kwargs):
    from ..fusedext import dsv4_hc_post_fused

    residual = kwargs["residual"]
    with torch.cuda.device(residual.device):
        return dsv4_hc_post_fused(
            kwargs["value"],
            residual,
            kwargs["post"],
            kwargs["combine"],
            output=kwargs.get("output"),
        )


def _hyper_connection_post_moe(**kwargs):
    from ..fusedext import dsv4_hc_post_moe_fused

    residual = kwargs["residual"]
    with torch.cuda.device(residual.device):
        return dsv4_hc_post_moe_fused(
            kwargs["routed"],
            kwargs["shared"],
            residual,
            kwargs["post"],
            kwargs["combine"],
            output=kwargs.get("output"),
        )


def _gated_activation(**kwargs):
    from ..fusedext import gated_activation_bf16_fused

    return gated_activation_bf16_fused(**kwargs)


def _route_topk_sqrtsoftplus(**kwargs):
    logits = kwargs["logits"]
    bias = kwargs["bias"]
    mask = kwargs["mask"]
    output_buffers = kwargs.get("output_buffers")
    if (
        not logits.is_cuda
        or logits.dtype != torch.float32
        or bias.dtype != torch.float32
        or mask.dtype != torch.bool
        or output_buffers is None
    ):
        return None
    output_weights, output_indices = output_buffers
    with torch.cuda.device(logits.device):
        scores = F.softplus(logits).sqrt()
        choice = scores.add(bias).masked_fill(~mask[None], -1e30)
        indices = choice.topk(int(kwargs["top_k"]), dim=-1).indices
        weights = scores.gather(1, indices)
        if kwargs["normalize"]:
            weights.div_(weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights.mul_(float(kwargs["scaling"]))
        output_indices.copy_(indices)
        output_weights.copy_(weights)
    return output_weights, output_indices


def register(registry: OperatorRegistry) -> None:
    for name, operation, activation, implementation in (
        (
            "cuda.hyper_connection.pre_norm.decode",
            "pre_norm",
            "rmsnorm",
            _hyper_connection_pre_norm,
        ),
        (
            "cuda.hyper_connection.post.decode",
            "post",
            "none",
            _hyper_connection_post,
        ),
        (
            "cuda.hyper_connection.post_moe.decode",
            "post_moe",
            "none",
            _hyper_connection_post_moe,
        ),
    ):
        registry.register(
            name,
            OperatorCapability(
                operation=f"hyper_connection:{operation}",
                device_types=("cuda",),
                activations=(activation,),
                max_top_k=1,
                batch_sizes=tuple(range(1, 257)),
            ),
            implementation,
            priority=100,
        )
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
        "cuda.linear_route_topk.sqrtsoftplus.decode",
        OperatorCapability(
            operation="linear_route_topk",
            device_types=("cuda",),
            activations=("sqrtsoftplus",),
            max_top_k=16,
            batch_sizes=(1,),
        ),
        _linear_route_topk_sqrtsoftplus,
        priority=90,
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
    registry.register(
        "cuda.route_topk.sqrtsoftplus.decode",
        OperatorCapability(
            operation="route_topk",
            device_types=("cuda",),
            activations=("sqrtsoftplus",),
            max_top_k=16,
            batch_sizes=(1,),
        ),
        _route_topk_sqrtsoftplus,
        priority=90,
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
        (
            "cuda.attention.sliding_compressed_mqa.decode",
            "sliding_compressed_mqa_decode",
            _sliding_compressed_mqa_decode,
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
        "cuda.dense_gemv.bf16.decode",
        OperatorCapability(
            operation="dense_gemv",
            device_types=("cuda",),
            packed_formats=("bf16",),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _dense_gemv,
        priority=100,
    )
    registry.register(
        "cuda.block_scaled_gemv.e4m3fn.b128.decode",
        OperatorCapability(
            operation="block_scaled_gemv",
            device_types=("cuda",),
            packed_formats=("e4m3fn",),
            code_dims=(128,),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _block_scaled_gemv,
        priority=100,
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
    registry.register(
        "cuda.packed_route_slots.fixed_metadata.decode",
        OperatorCapability(
            operation="packed_route_slots",
            device_types=("cuda",),
            activations=("none",),
            max_top_k=16,
            batch_sizes=(1,),
        ),
        _packed_route_slots,
        priority=100,
    )
    registry.register(
        "cuda.block_scaled_grouped_gemv.e4m3fn.b128.decode",
        OperatorCapability(
            operation="block_scaled_grouped_gemv",
            device_types=("cuda",),
            packed_formats=("e4m3fn",),
            code_dims=(128,),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _block_scaled_grouped_gemv,
        priority=110,
    )
    registry.register(
        "cuda.compressed_state_update.ring.decode",
        OperatorCapability(
            operation="compressed_state_update",
            device_types=("cuda",),
            code_dims=(4, 128),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _compressed_state_update,
        priority=110,
    )
    registry.register(
        "cuda.head_rmsnorm_rope.f32_bf16.decode",
        OperatorCapability(
            operation="head_rmsnorm_rope",
            device_types=("cuda",),
            code_dims=(64, 128, 256, 512, 1024),
            activations=("none",),
            max_top_k=1,
            batch_sizes=(1,),
        ),
        _head_rmsnorm_rope,
        priority=110,
    )
    registry.register(
        "cuda.packed_moe_topk.three_projection.mixed.gated",
        OperatorCapability(
            operation="moe_topk",
            device_types=("cuda",),
            packed_formats=tuple(f"p{bits}" for bits in range(8, 17)),
            code_dims=(4, 8, 16),
            codebook_sizes=(
                256, 512, 1024, 2048, 4096,
                8192, 16384, 32768, 65536,
            ),
            activations=("silu", "swiglu", "situ"),
            max_top_k=16,
            batch_sizes=(1,),
        ),
        _packed_moe_topk,
        priority=110,
    )
    registry.register(
        "cuda.resident_moe_topk.row_major_mixed.decode",
        OperatorCapability(
            operation="resident_moe_topk",
            device_types=("cuda",),
            packed_formats=("u8", "u16"),
            code_dims=(4, 8, 16),
            codebook_sizes=(256, 4096),
            activations=("silu", "swiglu"),
            max_top_k=8,
            batch_sizes=(1,),
        ),
        _resident_moe_topk_generic,
        priority=200,
    )
    registry.register(
        "cuda.resident_moe_topk.codegemm_mixed.decode",
        OperatorCapability(
            operation="resident_moe_topk",
            device_types=("cuda",),
            packed_formats=("psumbook_u8", "u16"),
            code_dims=(4,),
            codebook_sizes=(256, 4096),
            activations=("silu", "swiglu"),
            max_top_k=8,
            batch_sizes=(1,),
        ),
        _resident_moe_topk,
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
