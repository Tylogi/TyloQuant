"""Kimi-K3/TPQ2 Metal primitives for Apple silicon.

The kernels in this module mirror the decode operators used by the TPQ2
Kimi-K3 graph: three-way causal short convolution, KDA gate preparation and
V-first recurrent state, SiTU gating, gated RMSNorm, and Attention-Residual
mixing.  Reductions and recurrent state updates remain FP32.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.kernels.metal.linear_attention import gated_delta_net
from mfq.kernels.metal.moe_ops import moe_topk

_SHORT_CONV3_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(3 * C)) {
        return;
    }
    uint stream = index / uint(C);
    uint channel = index - stream * uint(C);
    const device T* input = stream == 0u ? query : (
        stream == 1u ? key : value
    );
    const device T* state = stream == 0u ? query_state : (
        stream == 1u ? key_state : value_state
    );
    const device float* weight = stream == 0u ? query_weight : (
        stream == 1u ? key_weight : value_weight
    );
    device T* output = stream == 0u ? query_out : (
        stream == 1u ? key_out : value_out
    );
    device T* next_state = stream == 0u ? query_state_out : (
        stream == 1u ? key_state_out : value_state_out
    );

    uint state_base = channel * uint(HISTORY);
    uint weight_base = channel * uint(HISTORY + 1);
    float result = 0.0f;
    for (uint item = 0u; item < uint(HISTORY); ++item) {
        result = fma(
            float(state[state_base + item]),
            weight[weight_base + item],
            result
        );
        if (item + 1u < uint(HISTORY)) {
            next_state[state_base + item] =
                state[state_base + item + 1u];
        }
    }
    float current = float(input[channel]);
    result = fma(current, weight[weight_base + uint(HISTORY)], result);
    if (HISTORY != 0) {
        next_state[state_base + uint(HISTORY - 1)] = T(current);
    }
    output[channel] = T(result / (1.0f + exp(-result)));
"""


_KDA_PREP_SOURCE = r"""
    uint head = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    if (head >= uint(HEADS)) {
        return;
    }
    uint base = head * uint(D);
    float query_square = 0.0f;
    float key_square = 0.0f;
    for (uint dimension = lane; dimension < uint(D); dimension += 32u) {
        float qv = float(query[base + dimension]);
        float kv = float(key[base + dimension]);
        query_square += qv * qv;
        key_square += kv * kv;
    }
    query_square = simd_sum(query_square);
    key_square = simd_sum(key_square);
    float query_scale = rsqrt(query_square + 1.0e-6f);
    float key_scale = rsqrt(key_square + 1.0e-6f);
    float a = exp(a_log[head]);
    for (uint dimension = lane; dimension < uint(D); dimension += 32u) {
        uint index = base + dimension;
        query_norm[index] = float(query[index]) * query_scale;
        key_norm[index] = float(key[index]) * key_scale;
        float gate_value = float(gate[index]) + dt_bias[index];
        float log_decay;
        if (BOUNDED != 0) {
            float raw = a * gate_value;
            log_decay = params[0] / (1.0f + exp(-raw));
        } else {
            float softplus =
                max(gate_value, 0.0f)
                + log(1.0f + exp(-abs(gate_value)));
            log_decay = -a * softplus;
        }
        decay[index] = log_decay;
    }
    if (lane == 0u) {
        beta_out[head] = 1.0f / (1.0f + exp(-beta[head]));
    }
"""


_SITU_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }
    float gate_value = float(gate[index]);
    float up_value = float(up[index]);
    float activated =
        params[0] * tanh(gate_value / params[0])
        / (1.0f + exp(-gate_value));
    if (HAS_LINEAR_BOUND != 0) {
        up_value = params[1] * tanh(up_value / params[1]);
    }
    output[index] = T(activated * up_value);
"""


_GATED_RMSNORM_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x >> 5;
    if (row >= uint(ROWS)) {
        return;
    }
    uint base = row * uint(WIDTH);
    float square_sum = 0.0f;
    for (uint column = lane; column < uint(WIDTH); column += 32u) {
        float item = float(value[base + column]);
        square_sum += item * item;
    }
    square_sum = simd_sum(square_sum);
    float scale = rsqrt(square_sum / float(WIDTH) + params[0]);
    for (uint column = lane; column < uint(WIDTH); column += 32u) {
        uint index = base + column;
        float sigmoid_gate = 1.0f / (1.0f + exp(-float(gate[index])));
        output[index] = T(
            float(value[index]) * scale * weight[column] * sigmoid_gate
        );
    }
"""


_ATTENTION_RESIDUAL_SOURCE = r"""
    constexpr uint THREADS = 256u;
    constexpr uint SIMD_GROUPS = THREADS / 32u;
    uint batch = threadgroup_position_in_grid.x;
    uint local_thread = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    if (batch >= uint(BATCH)) {
        return;
    }
    threadgroup float scores[32];
    threadgroup float probabilities[32];
    threadgroup float partial[SIMD_GROUPS];

    for (
        uint row = simd_group;
        row < uint(ROWS);
        row += SIMD_GROUPS
    ) {
        bool is_prefix = row == uint(RESIDUAL_ROWS);
        uint source_base = is_prefix
            ? batch * uint(WIDTH)
            : (batch * uint(RESIDUAL_ROWS) + row) * uint(WIDTH);
        float square_sum = 0.0f;
        float score = 0.0f;
        for (uint column = lane; column < uint(WIDTH); column += 32u) {
            float item = is_prefix
                ? float(prefix[source_base + column])
                : float(residual[source_base + column]);
            square_sum += item * item;
            score += item * projection[column] * norm_weight[column];
        }
        square_sum = simd_sum(square_sum);
        float inverse = rsqrt(square_sum / float(WIDTH) + params[0]);
        score = simd_sum(score * inverse);
        if (lane == 0u) {
            scores[row] = score;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_thread == 0u) {
        float maximum = -INFINITY;
        for (uint row = 0u; row < uint(ROWS); ++row) {
            maximum = max(maximum, scores[row]);
        }
        float denominator = 0.0f;
        for (uint row = 0u; row < uint(ROWS); ++row) {
            float item = exp(scores[row] - maximum);
            probabilities[row] = item;
            denominator += item;
        }
        float inverse = 1.0f / denominator;
        for (uint row = 0u; row < uint(ROWS); ++row) {
            probabilities[row] *= inverse;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (
        uint column = local_thread;
        column < uint(WIDTH);
        column += THREADS
    ) {
        float mixed = 0.0f;
        for (uint row = 0u; row < uint(ROWS); ++row) {
            bool is_prefix = row == uint(RESIDUAL_ROWS);
            uint source_index = is_prefix
                ? batch * uint(WIDTH) + column
                : (batch * uint(RESIDUAL_ROWS) + row)
                    * uint(WIDTH) + column;
            mixed += probabilities[row] * (
                is_prefix
                    ? float(prefix[source_index])
                    : float(residual[source_index])
            );
        }
        output[batch * uint(WIDTH) + column] = T(mixed);
    }
    if (HAS_POST_NORM == 0) {
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_square = 0.0f;
    for (
        uint column = local_thread;
        column < uint(WIDTH);
        column += THREADS
    ) {
        float item = float(output[batch * uint(WIDTH) + column]);
        local_square += item * item;
    }
    local_square = simd_sum(local_square);
    if (lane == 0u) {
        partial[simd_group] = local_square;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_thread == 0u) {
        float total = 0.0f;
        for (uint group = 0u; group < SIMD_GROUPS; ++group) {
            total += partial[group];
        }
        scores[0] = rsqrt(total / float(WIDTH) + params[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float post_scale = scores[0];
    for (
        uint column = local_thread;
        column < uint(WIDTH);
        column += THREADS
    ) {
        uint index = batch * uint(WIDTH) + column;
        output[index] = T(
            float(output[index]) * post_scale * post_norm_weight[column]
        );
    }
"""


_SHORT_CONV3_KERNEL = mx.fast.metal_kernel(
    name="mfq_kimi_short_conv3",
    input_names=[
        "query",
        "key",
        "value",
        "query_state",
        "key_state",
        "value_state",
        "query_weight",
        "key_weight",
        "value_weight",
    ],
    output_names=[
        "query_out",
        "key_out",
        "value_out",
        "query_state_out",
        "key_state_out",
        "value_state_out",
    ],
    source=_SHORT_CONV3_SOURCE,
    compile_options={"math_mode": "fast"},
)

_KDA_PREP_KERNEL = mx.fast.metal_kernel(
    name="mfq_kimi_kda_prepare",
    input_names=[
        "query",
        "key",
        "gate",
        "beta",
        "a_log",
        "dt_bias",
        "params",
    ],
    output_names=["query_norm", "key_norm", "decay", "beta_out"],
    source=_KDA_PREP_SOURCE,
    compile_options={"math_mode": "fast"},
)

_SITU_KERNEL = mx.fast.metal_kernel(
    name="mfq_kimi_situ",
    input_names=["gate", "up", "params"],
    output_names=["output"],
    source=_SITU_SOURCE,
    compile_options={"math_mode": "fast"},
)

_GATED_RMSNORM_KERNEL = mx.fast.metal_kernel(
    name="mfq_kimi_gated_rmsnorm",
    input_names=["value", "gate", "weight", "params"],
    output_names=["output"],
    source=_GATED_RMSNORM_SOURCE,
    compile_options={"math_mode": "fast"},
)

_ATTENTION_RESIDUAL_KERNEL = mx.fast.metal_kernel(
    name="mfq_kimi_attention_residual",
    input_names=[
        "prefix",
        "residual",
        "projection",
        "norm_weight",
        "post_norm_weight",
        "params",
    ],
    output_names=["output"],
    source=_ATTENTION_RESIDUAL_SOURCE,
    compile_options={"math_mode": "fast"},
)


def _floating(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if result.dtype not in (mx.float16, mx.float32):
        result = result.astype(mx.float16)
    return mx.contiguous(result)


def _float32(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    return mx.contiguous(result.astype(mx.float32))


def _conv_state(
    value: mx.array | np.ndarray,
    channels: int,
    dtype,
) -> mx.array:
    state = _floating(value).astype(dtype)
    if state.ndim != 2 or int(state.shape[0]) != channels:
        raise ValueError("Kimi short-convolution state must have [channels,history]")
    return state


def _conv_weight(
    value: mx.array | np.ndarray,
    channels: int,
    history: int,
) -> mx.array:
    weight = _float32(value)
    if weight.ndim == 3 and tuple(map(int, weight.shape[:2])) == (channels, 1):
        weight = weight.reshape((channels, int(weight.shape[2])))
    if tuple(map(int, weight.shape)) != (channels, history + 1):
        raise ValueError("Kimi short-convolution weight must have [channels,history+1]")
    return weight


def kimi_short_conv3(
    query: mx.array | np.ndarray,
    key: mx.array | np.ndarray,
    value: mx.array | np.ndarray,
    states: tuple[
        mx.array | np.ndarray,
        mx.array | np.ndarray,
        mx.array | np.ndarray,
    ],
    weights: tuple[
        mx.array | np.ndarray,
        mx.array | np.ndarray,
        mx.array | np.ndarray,
    ],
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    tuple[mx.array, mx.array, mx.array],
]:
    """Update Kimi's Q/K/V depthwise-convolution streams in one dispatch."""

    query_value = _floating(query)
    key_value = _floating(key)
    value_value = _floating(value)
    if query_value.shape != key_value.shape or query_value.shape != value_value.shape:
        raise ValueError("Kimi short-convolution Q/K/V shapes must match")
    if query_value.ndim < 1:
        raise ValueError("Kimi short-convolution input cannot be scalar")
    channels = int(query_value.size)
    query_value = query_value.reshape((channels,))
    key_value = key_value.astype(query_value.dtype).reshape((channels,))
    value_value = value_value.astype(query_value.dtype).reshape((channels,))
    packed_states = tuple(_conv_state(item, channels, query_value.dtype) for item in states)
    history = int(packed_states[0].shape[1])
    if any(tuple(item.shape) != (channels, history) for item in packed_states):
        raise ValueError("Kimi Q/K/V convolution states must have matching shapes")
    packed_weights = tuple(_conv_weight(item, channels, history) for item in weights)
    outputs = _SHORT_CONV3_KERNEL(
        inputs=[
            query_value,
            key_value,
            value_value,
            *packed_states,
            *packed_weights,
        ],
        template=[
            ("T", query_value.dtype),
            ("C", channels),
            ("HISTORY", history),
        ],
        grid=(3 * channels, 1, 1),
        threadgroup=(min(256, max(1, 3 * channels)), 1, 1),
        output_shapes=[
            (channels,),
            (channels,),
            (channels,),
            (channels, history),
            (channels, history),
            (channels, history),
        ],
        output_dtypes=[query_value.dtype] * 6,
    )
    return outputs[0], outputs[1], outputs[2], tuple(outputs[3:])  # type: ignore[return-value]


def kimi_kda_recurrent(
    query: mx.array | np.ndarray,
    key: mx.array | np.ndarray,
    value: mx.array | np.ndarray,
    gate: mx.array | np.ndarray,
    beta: mx.array | np.ndarray,
    a_log: mx.array | np.ndarray,
    dt_bias: mx.array | np.ndarray,
    state: mx.array | np.ndarray,
    *,
    lower_bound: float | None = -5.0,
) -> tuple[mx.array, mx.array]:
    """Run one Kimi KDA update with a persistent V-first FP32 state."""

    query_value = _floating(query)
    key_value = _floating(key).astype(query_value.dtype)
    value_value = _floating(value).astype(query_value.dtype)
    gate_value = _floating(gate).astype(query_value.dtype)
    if query_value.ndim != 2 or key_value.shape != query_value.shape:
        raise ValueError("Kimi KDA query/key must have [heads,key_dim] shape")
    heads, dimension = map(int, query_value.shape)
    if dimension not in (32, 64, 128):
        raise ValueError("Kimi KDA Metal supports key_dim in {32,64,128}")
    if tuple(value_value.shape) != (heads, dimension):
        raise ValueError("Kimi KDA currently requires value_dim == key_dim")
    if gate_value.shape != query_value.shape:
        raise ValueError("Kimi KDA gate must match query shape")
    beta_value = _float32(beta).reshape((-1,))
    a_value = _float32(a_log).reshape((-1,))
    dt_value = _float32(dt_bias).reshape((-1,))
    if (
        int(beta_value.size) < heads
        or int(a_value.size) < heads
        or int(dt_value.size) < heads * dimension
    ):
        raise ValueError("Kimi KDA beta/A_log/dt_bias buffers are too small")
    state_value = _float32(state)
    if tuple(state_value.shape) != (heads, dimension, dimension):
        raise ValueError("Kimi KDA state must have [heads,value_dim,key_dim] shape")
    bounded = lower_bound is not None
    lower = -5.0 if lower_bound is None else float(lower_bound)
    if not math.isfinite(lower):
        raise ValueError("Kimi KDA lower_bound must be finite")
    params = mx.array([lower], dtype=mx.float32)

    prepared = _KDA_PREP_KERNEL(
        inputs=[
            query_value,
            key_value,
            gate_value,
            beta_value,
            a_value,
            dt_value,
            params,
        ],
        template=[
            ("T", query_value.dtype),
            ("HEADS", heads),
            ("D", dimension),
            ("BOUNDED", int(bounded)),
        ],
        grid=(heads * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (1, heads, 1, dimension),
            (1, heads, 1, dimension),
            (1, heads, 1, dimension),
            (1, heads, 1),
        ],
        output_dtypes=[mx.float32] * 4,
    )
    attended, next_state = gated_delta_net(
        prepared[0],
        prepared[1],
        value_value.reshape((1, heads, 1, dimension)),
        prepared[2],
        prepared[3],
        state_value.reshape((1, heads, dimension, dimension)),
        transposed_state=True,
    )
    return (
        attended.reshape((heads, dimension)).astype(query_value.dtype),
        next_state.reshape((heads, dimension, dimension)),
    )


def situ_mul(
    gate: mx.array | np.ndarray,
    up: mx.array | np.ndarray,
    *,
    beta: float,
    linear_beta: float | None = None,
) -> mx.array:
    """Apply Kimi's SiTU gate and optional bounded-linear up projection."""

    gate_value = _floating(gate)
    up_value = _floating(up).astype(gate_value.dtype)
    if gate_value.shape != up_value.shape:
        raise ValueError("SiTU gate and up shapes must match")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("SiTU beta must be finite and positive")
    if linear_beta is not None and (not math.isfinite(linear_beta) or linear_beta <= 0.0):
        raise ValueError("SiTU linear_beta must be finite and positive")
    size = int(gate_value.size)
    params = mx.array(
        [float(beta), -1.0 if linear_beta is None else float(linear_beta)],
        dtype=mx.float32,
    )
    return _SITU_KERNEL(
        inputs=[gate_value, up_value, params],
        template=[
            ("T", gate_value.dtype),
            ("SIZE", size),
            ("HAS_LINEAR_BOUND", int(linear_beta is not None)),
        ],
        grid=(size, 1, 1),
        threadgroup=(min(256, max(1, size)), 1, 1),
        output_shapes=[gate_value.shape],
        output_dtypes=[gate_value.dtype],
    )[0]


def situ_split(
    gate_up: mx.array | np.ndarray,
    *,
    beta: float,
    linear_beta: float | None = None,
) -> mx.array:
    """Split a concatenated gate/up projection and apply SiTU."""

    values = _floating(gate_up)
    if values.ndim < 1 or int(values.shape[-1]) % 2:
        raise ValueError("SiTU gate_up must end in an even dimension")
    gate, up = mx.split(values, 2, axis=-1)
    return situ_mul(gate, up, beta=beta, linear_beta=linear_beta)


def kimi_gated_rmsnorm(
    value: mx.array | np.ndarray,
    gate: mx.array | np.ndarray,
    weight: mx.array | np.ndarray,
    eps: float,
) -> mx.array:
    """Fuse Kimi RMSNorm, sigmoid output gate, and multiplication."""

    source = _floating(value)
    gate_value = _floating(gate).astype(source.dtype)
    if source.ndim < 1 or gate_value.shape != source.shape:
        raise ValueError("Kimi gated RMSNorm value/gate shapes must match")
    width = int(source.shape[-1])
    rows = int(source.size) // width
    weight_value = _float32(weight).reshape((-1,))
    if int(weight_value.size) != width:
        raise ValueError("Kimi gated RMSNorm weight width is incompatible")
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("Kimi gated RMSNorm epsilon must be non-negative")
    params = mx.array([float(eps)], dtype=mx.float32)
    return _GATED_RMSNORM_KERNEL(
        inputs=[source.reshape((rows, width)), gate_value, weight_value, params],
        template=[
            ("T", source.dtype),
            ("ROWS", rows),
            ("WIDTH", width),
        ],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[source.shape],
        output_dtypes=[source.dtype],
    )[0]


def kimi_attention_residual(
    prefix: mx.array | np.ndarray,
    residual: mx.array | np.ndarray,
    projection: mx.array | np.ndarray,
    norm_weight: mx.array | np.ndarray,
    eps: float,
    *,
    post_norm_weight: mx.array | np.ndarray | None = None,
) -> mx.array:
    """Apply Kimi's Attention-Residual mixture and optional following RMSNorm."""

    prefix_value = _floating(prefix)
    residual_value = _floating(residual).astype(prefix_value.dtype)
    if prefix_value.ndim != 2 or residual_value.ndim != 3:
        raise ValueError("Kimi residual expects prefix [B,W] and residual [B,R,W]")
    batch, width = map(int, prefix_value.shape)
    residual_rows = int(residual_value.shape[1])
    if tuple(map(int, residual_value.shape)) != (batch, residual_rows, width) or residual_rows > 31:
        raise ValueError("Kimi Attention-Residual supports at most 31 saved rows")
    projection_value = _float32(projection).reshape((-1,))
    norm_value = _float32(norm_weight).reshape((-1,))
    if int(projection_value.size) != width or int(norm_value.size) != width:
        raise ValueError("Kimi residual projection/norm widths are incompatible")
    if post_norm_weight is None:
        post_value = mx.zeros((width,), dtype=mx.float32)
        has_post = False
    else:
        post_value = _float32(post_norm_weight).reshape((-1,))
        if int(post_value.size) != width:
            raise ValueError("Kimi residual post-norm width is incompatible")
        has_post = True
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("Kimi residual epsilon must be non-negative")
    params = mx.array([float(eps)], dtype=mx.float32)
    return _ATTENTION_RESIDUAL_KERNEL(
        inputs=[
            prefix_value,
            residual_value,
            projection_value,
            norm_value,
            post_value,
            params,
        ],
        template=[
            ("T", prefix_value.dtype),
            ("BATCH", batch),
            ("WIDTH", width),
            ("RESIDUAL_ROWS", residual_rows),
            ("ROWS", residual_rows + 1),
            ("HAS_POST_NORM", int(has_post)),
        ],
        grid=(batch * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[prefix_value.shape],
        output_dtypes=[prefix_value.dtype],
    )[0]


def kimi_route_experts(
    logits: mx.array | np.ndarray,
    correction_bias: mx.array | np.ndarray,
    available: mx.array | np.ndarray,
    *,
    top_k: int,
    normalize: bool,
    scaling: float,
    n_group: int = 1,
    topk_group: int = 1,
) -> tuple[mx.array, mx.array]:
    """Kimi sigmoid routing with correction, availability, and group masking."""

    values = _floating(logits)
    if values.ndim < 2:
        raise ValueError("Kimi router logits must end in an expert dimension")
    experts = int(values.shape[-1])
    bias = _float32(correction_bias).reshape((-1,))
    mask = (available if isinstance(available, mx.array) else mx.array(available)).astype(mx.bool_)
    mask = mx.contiguous(mask.reshape((-1,)))
    if int(bias.size) != experts or int(mask.size) != experts:
        raise ValueError("Kimi router correction/mask widths are incompatible")
    if n_group <= 1 or n_group <= topk_group:
        ids, weights = moe_topk(
            values,
            top_k,
            use_sigmoid=True,
            normalize=normalize,
            bias=bias,
            available=mask,
            scale=scaling,
        )
        return weights, ids
    if experts % int(n_group):
        raise ValueError("Kimi expert count must be divisible by n_group")
    if not 1 <= int(topk_group) <= int(n_group):
        raise ValueError("Kimi topk_group must be in [1,n_group]")

    scores = mx.sigmoid(values.astype(mx.float32))
    choice = mx.where(mask, scores + bias, -mx.inf)
    grouped = choice.reshape((*map(int, choice.shape[:-1]), int(n_group), experts // int(n_group)))
    group_scores = mx.sum(mx.topk(grouped, 2, axis=-1), axis=-1)
    selected_groups = mx.argsort(group_scores, axis=-1)[..., -int(topk_group) :]
    group_range = mx.arange(int(n_group), dtype=mx.int32)
    group_mask = mx.any(
        selected_groups[..., :, None] == group_range,
        axis=-2,
    )
    choice = mx.where(
        mx.broadcast_to(group_mask[..., :, None], grouped.shape),
        grouped,
        -mx.inf,
    ).reshape(values.shape)
    ids = mx.argsort(choice, axis=-1)[..., -int(top_k) :][..., ::-1]
    weights = mx.take_along_axis(scores, ids, axis=-1)
    if normalize and int(top_k) > 1:
        weights = weights / mx.maximum(
            mx.sum(weights, axis=-1, keepdims=True),
            1.0e-20,
        )
    return weights * float(scaling), ids.astype(mx.int32)


__all__ = [
    "kimi_attention_residual",
    "kimi_gated_rmsnorm",
    "kimi_kda_recurrent",
    "kimi_route_experts",
    "kimi_short_conv3",
    "situ_mul",
    "situ_split",
]
