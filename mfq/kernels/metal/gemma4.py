"""Gemma4-specific fused Transformer kernels for Apple silicon.

The CUDA runtime deliberately rounds the intermediate residual and FFN
branches to the activation dtype between RMSNorm stages.  These kernels keep
the same ordering so router logits and layer outputs stay aligned across the
CUDA and Metal backends.
"""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc


_ATTN_RESIDUAL_PRE_NORMS_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x >> 5;
    if (row >= uint(ROWS)) {
        return;
    }

    threadgroup T shared_x[DIM];
    uint offset = row * uint(DIM);
    float attn_square_sum = 0.0f;
    for (uint column = lane; column < uint(DIM); column += 32u) {
        float value = float(attn[offset + column]);
        attn_square_sum += value * value;
    }
    attn_square_sum = simd_sum(attn_square_sum);
    float attn_inverse = rsqrt(attn_square_sum / float(DIM) + params[0]);

    float x_square_sum = 0.0f;
    for (uint column = lane; column < uint(DIM); column += 32u) {
        T attn_norm = T(
            float(attn[offset + column])
            * attn_inverse
            * attn_post_weight[column]
        );
        T x_value = T(float(residual[offset + column]) + float(attn_norm));
        residual_out[offset + column] = x_value;
        shared_x[column] = x_value;
        float value = float(x_value);
        x_square_sum += value * value;
    }
    x_square_sum = simd_sum(x_square_sum);
    float x_inverse = rsqrt(x_square_sum / float(DIM) + params[0]);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint column = lane; column < uint(DIM); column += 32u) {
        float normalized = float(shared_x[column]) * x_inverse;
        dense_out[offset + column] = T(normalized * dense_pre_weight[column]);
        router_out[offset + column] = normalized * router_weight[column];
        moe_out[offset + column] = T(normalized * moe_pre_weight[column]);
    }
"""


_FFN_MERGE_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x >> 5;
    if (row >= uint(ROWS)) {
        return;
    }

    threadgroup T combined[DIM];
    uint offset = row * uint(DIM);
    float dense_square_sum = 0.0f;
    float moe_square_sum = 0.0f;
    for (uint column = lane; column < uint(DIM); column += 32u) {
        float dense_value = float(dense[offset + column]);
        float moe_value = float(moe[offset + column]);
        dense_square_sum += dense_value * dense_value;
        moe_square_sum += moe_value * moe_value;
    }
    dense_square_sum = simd_sum(dense_square_sum);
    moe_square_sum = simd_sum(moe_square_sum);
    float dense_inverse = rsqrt(
        dense_square_sum / float(DIM) + params[0]
    );
    float moe_inverse = rsqrt(
        moe_square_sum / float(DIM) + params[0]
    );

    float combined_square_sum = 0.0f;
    for (uint column = lane; column < uint(DIM); column += 32u) {
        T dense_norm = T(
            float(dense[offset + column])
            * dense_inverse
            * dense_post_weight[column]
        );
        T moe_norm = T(
            float(moe[offset + column])
            * moe_inverse
            * moe_post_weight[column]
        );
        T value = T(float(dense_norm) + float(moe_norm));
        combined[column] = value;
        float stored = float(value);
        combined_square_sum += stored * stored;
    }
    combined_square_sum = simd_sum(combined_square_sum);
    float combined_inverse = rsqrt(
        combined_square_sum / float(DIM) + params[0]
    );
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float scale = params[1];
    for (uint column = lane; column < uint(DIM); column += 32u) {
        T post = T(
            float(combined[column])
            * combined_inverse
            * final_post_weight[column]
        );
        T residual_sum = T(float(residual[offset + column]) + float(post));
        output[offset + column] = T(float(residual_sum) * scale);
    }
"""


_ATTN_RESIDUAL_PRE_NORMS_KERNEL = mx.fast.metal_kernel(
    name="mfq_gemma4_attn_residual_pre_norms",
    input_names=[
        "residual",
        "attn",
        "attn_post_weight",
        "dense_pre_weight",
        "router_weight",
        "moe_pre_weight",
        "params",
    ],
    output_names=["residual_out", "dense_out", "router_out", "moe_out"],
    source=_ATTN_RESIDUAL_PRE_NORMS_SOURCE,
    compile_options={"math_mode": "fast"},
)


_FFN_MERGE_KERNEL = mx.fast.metal_kernel(
    name="mfq_gemma4_ffn_merge",
    input_names=[
        "dense",
        "moe",
        "residual",
        "dense_post_weight",
        "moe_post_weight",
        "final_post_weight",
        "params",
    ],
    output_names=["output"],
    source=_FFN_MERGE_SOURCE,
    compile_options={"math_mode": "fast"},
)


def _activation(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if result.ndim < 1:
        raise ValueError("Gemma4 activation must have at least one dimension")
    if result.dtype not in (mx.float16, mx.float32):
        result = result.astype(mx.float16)
    return mx.contiguous(result)


def _weight(
    value: mx.array | np.ndarray,
    dimension: int,
    name: str,
) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if result.ndim != 1 or int(result.size) != int(dimension):
        raise ValueError(f"Gemma4 {name} must have shape ({dimension},), got {result.shape}")
    return mx.contiguous(result.astype(mx.float32))


def gemma4_attn_residual_pre_norms(
    residual: mx.array | np.ndarray,
    attention_output: mx.array | np.ndarray,
    attention_post_weight: mx.array | np.ndarray,
    dense_pre_weight: mx.array | np.ndarray,
    router_weight: mx.array | np.ndarray,
    moe_pre_weight: mx.array | np.ndarray,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Fuse the Gemma4 post-attention residual and three pre-FFN norms."""

    residual_value = _activation(residual)
    attention_value = _activation(attention_output)
    if tuple(residual_value.shape) != tuple(attention_value.shape):
        raise ValueError("Gemma4 residual and attention shapes must match")
    dtype = (
        mx.float32 if mx.float32 in (residual_value.dtype, attention_value.dtype) else mx.float16
    )
    residual_value = mx.contiguous(residual_value.astype(dtype))
    attention_value = mx.contiguous(attention_value.astype(dtype))
    shape = tuple(int(item) for item in residual_value.shape)
    dimension = shape[-1]
    if dimension <= 0:
        raise ValueError("Gemma4 hidden dimension must be positive")
    rows = int(residual_value.size) // dimension
    weights = (
        _weight(attention_post_weight, dimension, "attention post weight"),
        _weight(dense_pre_weight, dimension, "dense pre weight"),
        _weight(router_weight, dimension, "router weight"),
        _weight(moe_pre_weight, dimension, "MoE pre weight"),
    )
    if rows == 0:
        empty = mx.zeros(shape, dtype=dtype)
        return empty, empty, mx.zeros(shape, dtype=mx.float32), empty
    params = mx.array([float(eps)], dtype=mx.float32)
    outputs = _ATTN_RESIDUAL_PRE_NORMS_KERNEL(
        inputs=[residual_value, attention_value, *weights, params],
        template=[("T", dtype), ("ROWS", rows), ("DIM", dimension)],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[shape, shape, shape, shape],
        output_dtypes=[dtype, dtype, mx.float32, dtype],
    )
    return outputs[0], outputs[1], outputs[2], outputs[3]


def gemma4_ffn_merge(
    dense_output: mx.array | np.ndarray,
    moe_output: mx.array | np.ndarray,
    residual: mx.array | np.ndarray,
    dense_post_weight: mx.array | np.ndarray,
    moe_post_weight: mx.array | np.ndarray,
    final_post_weight: mx.array | np.ndarray,
    layer_scale: mx.array | np.ndarray | float,
    eps: float = 1e-6,
) -> mx.array:
    """Fuse Gemma4's two branch norms, merge norm, residual, and layer scale."""

    dense_value = _activation(dense_output)
    moe_value = _activation(moe_output)
    residual_value = _activation(residual)
    if not (tuple(dense_value.shape) == tuple(moe_value.shape) == tuple(residual_value.shape)):
        raise ValueError("Gemma4 dense, MoE, and residual shapes must match")
    dtype = (
        mx.float32
        if mx.float32 in (dense_value.dtype, moe_value.dtype, residual_value.dtype)
        else mx.float16
    )
    dense_value = mx.contiguous(dense_value.astype(dtype))
    moe_value = mx.contiguous(moe_value.astype(dtype))
    residual_value = mx.contiguous(residual_value.astype(dtype))
    shape = tuple(int(item) for item in dense_value.shape)
    dimension = shape[-1]
    if dimension <= 0:
        raise ValueError("Gemma4 hidden dimension must be positive")
    rows = int(dense_value.size) // dimension
    weights = (
        _weight(dense_post_weight, dimension, "dense post weight"),
        _weight(moe_post_weight, dimension, "MoE post weight"),
        _weight(final_post_weight, dimension, "final post weight"),
    )
    scale_value = (
        layer_scale
        if isinstance(layer_scale, mx.array)
        else mx.array(layer_scale, dtype=mx.float32)
    )
    if int(scale_value.size) != 1:
        raise ValueError("Gemma4 layer scale must be scalar")
    mx.eval(scale_value)
    scale = float(scale_value.reshape((-1,))[0].item())
    if rows == 0:
        return mx.zeros(shape, dtype=dtype)
    params = mx.array([float(eps), scale], dtype=mx.float32)
    return _FFN_MERGE_KERNEL(
        inputs=[dense_value, moe_value, residual_value, *weights, params],
        template=[("T", dtype), ("ROWS", rows), ("DIM", dimension)],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[shape],
        output_dtypes=[dtype],
    )[0]


__all__ = [
    "gemma4_attn_residual_pre_norms",
    "gemma4_ffn_merge",
]
