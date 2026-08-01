"""DeepSeek-V4 Hyper-Connection pre/post Metal kernels."""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc


_HC_PRE_SOURCE = r"""
    constexpr uint CONNECTIONS = 4u;
    constexpr uint MIX_WIDTH = 24u;
    uint row = threadgroup_position_in_grid.x;
    uint local_thread = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float pre[CONNECTIONS];

    if (simd_group == 0u) {
        uint mix_base = row * MIX_WIDTH;
        float active = lane < CONNECTIONS ? 1.0f : 0.0f;
        uint connection = min(lane, CONNECTIONS - 1u);
        float pre_affine =
            mixes[mix_base + connection] * scale[0]
            + base[connection];
        float post_affine =
            mixes[mix_base + CONNECTIONS + connection] * scale[1]
            + base[CONNECTIONS + connection];
        float pre_value =
            1.0f / (1.0f + metal::fast::exp(-pre_affine)) + params[0];
        float post_value =
            2.0f / (1.0f + metal::fast::exp(-post_affine));
        if (lane < CONNECTIONS) {
            pre[lane] = pre_value;
            post[row * CONNECTIONS + lane] = post_value;
        }

        float4 values =
            (*(const device float4*)(
                mixes + mix_base + 2u * CONNECTIONS
                    + connection * CONNECTIONS
            ) * scale[2]
            + *(const device float4*)(
                base + 2u * CONNECTIONS + connection * CONNECTIONS
            )) * active;
        float maximum = max(
            max(values.x, values.y),
            max(values.z, values.w)
        );
        float4 probabilities =
            metal::fast::exp(values - maximum) * active;
        probabilities =
            probabilities / (
                probabilities.x + probabilities.y
                    + probabilities.z + probabilities.w
                    + params[0]
            )
            + params[0] * active;
        probabilities /= float4(
            simd_sum(probabilities.x),
            simd_sum(probabilities.y),
            simd_sum(probabilities.z),
            simd_sum(probabilities.w)
        ) + params[0];

        for (uint iteration = 1u; iteration < 20u; ++iteration) {
            probabilities *= (
                active / (
                    probabilities.x + probabilities.y
                        + probabilities.z + probabilities.w
                        + params[0]
                )
            );
            probabilities /= float4(
                simd_sum(probabilities.x),
                simd_sum(probabilities.y),
                simd_sum(probabilities.z),
                simd_sum(probabilities.w)
            ) + params[0];
        }
        if (lane < CONNECTIONS) {
            *(device float4*)(
                combination + row * 16u + lane * CONNECTIONS
            ) = probabilities;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint HIDDEN4 = uint(HIDDEN) / 4u;
    const device half4* x0 = (const device half4*)(
        x + (row * CONNECTIONS + 0u) * uint(HIDDEN)
    );
    const device half4* x1 = (const device half4*)(
        x + (row * CONNECTIONS + 1u) * uint(HIDDEN)
    );
    const device half4* x2 = (const device half4*)(
        x + (row * CONNECTIONS + 2u) * uint(HIDDEN)
    );
    const device half4* x3 = (const device half4*)(
        x + (row * CONNECTIONS + 3u) * uint(HIDDEN)
    );
    device half4* reduced4 =
        (device half4*)(reduced + row * uint(HIDDEN));
    for (uint feature4 = local_thread; feature4 < HIDDEN4; feature4 += 256u) {
        float4 value = fma(
            float4(pre[0]), float4(x0[feature4]),
            fma(
                float4(pre[1]), float4(x1[feature4]),
                fma(
                    float4(pre[2]), float4(x2[feature4]),
                    float4(pre[3]) * float4(x3[feature4])
                )
            )
        );
        reduced4[feature4] = half4(value);
    }
"""


_HC_POST_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }
    uint feature = index % uint(HIDDEN);
    uint destination_row = index / uint(HIDDEN);
    uint destination = destination_row % 4u;
    uint row = destination_row / 4u;
    float residual_sum = 0.0f;
    for (uint source = 0u; source < 4u; ++source) {
        residual_sum += combination[
            (row * 4u + source) * 4u + destination
        ] * float(residual[
            (row * 4u + source) * uint(HIDDEN) + feature
        ]);
    }
    float direct = post[row * 4u + destination]
        * float(x[row * uint(HIDDEN) + feature]);
    out[index] = half(direct + residual_sum);
"""


_HC_PRE_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_hc_pre",
    input_names=["x", "mixes", "scale", "base", "params"],
    output_names=["reduced", "post", "combination"],
    source=_HC_PRE_SOURCE,
    compile_options={"math_mode": "fast"},
)

_HC_POST_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_hc_post",
    input_names=["x", "residual", "post", "combination"],
    output_names=["out"],
    source=_HC_POST_SOURCE,
    compile_options={"math_mode": "fast"},
)


def _array(value: mx.array | np.ndarray, dtype: mx.Dtype) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    return mx.contiguous(result.astype(dtype))


def dsv4_hc_pre(
    x: mx.array | np.ndarray,
    mixes: mx.array | np.ndarray,
    scale: mx.array | np.ndarray,
    base: mx.array | np.ndarray,
    iterations: int = 20,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array, mx.array]:
    """Reduce four hyper-connections and produce post/Sinkhorn coefficients."""

    source = _array(x, mx.float16)
    mix_values = _array(mixes, mx.float32)
    scale_values = _array(scale, mx.float32)
    base_values = _array(base, mx.float32)
    if (
        source.ndim != 4
        or tuple(int(value) for value in source.shape[2:]) != (4, 4096)
        or tuple(int(value) for value in mix_values.shape)
        != (int(source.shape[0]), int(source.shape[1]), 24)
        or int(scale_values.size) != 3
        or int(base_values.size) != 24
        or int(iterations) != 20
        or not np.isfinite(eps)
        or float(eps) <= 0.0
    ):
        raise ValueError("invalid DSV4 HC pre input")
    batch, tokens = (int(value) for value in source.shape[:2])
    rows = batch * tokens
    params = mx.array([float(eps)], dtype=mx.float32)
    reduced, post, combination = _HC_PRE_KERNEL(
        inputs=[source, mix_values, scale_values, base_values, params],
        template=[("ROWS", rows), ("HIDDEN", 4096)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (batch, tokens, 4096),
            (batch, tokens, 4),
            (batch, tokens, 4, 4),
        ],
        output_dtypes=[mx.float16, mx.float32, mx.float32],
    )
    return reduced, post, combination


def dsv4_hc_post(
    x: mx.array | np.ndarray,
    residual: mx.array | np.ndarray,
    post: mx.array | np.ndarray,
    combination: mx.array | np.ndarray,
) -> mx.array:
    """Expand one transformed branch back into four hyper-connections."""

    source = _array(x, mx.float16)
    residual_values = _array(residual, mx.float16)
    post_values = _array(post, mx.float32)
    combination_values = _array(combination, mx.float32)
    if (
        source.ndim != 3
        or int(source.shape[2]) != 4096
        or tuple(int(value) for value in residual_values.shape)
        != (int(source.shape[0]), int(source.shape[1]), 4, 4096)
        or tuple(int(value) for value in post_values.shape)
        != (int(source.shape[0]), int(source.shape[1]), 4)
        or tuple(int(value) for value in combination_values.shape)
        != (int(source.shape[0]), int(source.shape[1]), 4, 4)
    ):
        raise ValueError("invalid DSV4 HC post input")
    batch, tokens = (int(value) for value in source.shape[:2])
    size = batch * tokens * 4 * 4096
    return _HC_POST_KERNEL(
        inputs=[
            source,
            residual_values,
            post_values,
            combination_values,
        ],
        template=[("SIZE", size), ("HIDDEN", 4096)],
        grid=(size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, tokens, 4, 4096)],
        output_dtypes=[mx.float16],
    )[0]


__all__ = ["dsv4_hc_post", "dsv4_hc_pre"]
