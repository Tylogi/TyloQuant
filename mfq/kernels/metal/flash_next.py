"""Shared MLX primitives for Qwen4-Exp and GLM-5-Next decoder graphs."""

from __future__ import annotations

import math

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc


_GLM5_SPARSE_MLA_SOURCE = r"""
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint VALUE_SLICES = (uint(D) + 255u) / 256u;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint local_thread = thread_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint head = workgroup % uint(HEADS);
    uint query_row = workgroup / uint(HEADS);
    uint query = query_row % uint(TOKENS);
    uint batch = query_row / uint(TOKENS);
    if (batch >= uint(B)) {
        return;
    }

    threadgroup float partials[SIMD_GROUPS];
    threadgroup float online[4];
    if (local_thread == 0u) {
        online[0] = -INFINITY;
        online[1] = 0.0f;
        online[2] = 0.0f;
        online[3] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulators[VALUE_SLICES];
    for (uint slice = 0u; slice < VALUE_SLICES; ++slice) {
        accumulators[slice] = 0.0f;
    }
    uint query_base =
        ((batch * uint(HEADS) + head) * uint(TOKENS) + query) * uint(D);
    uint index_base =
        (batch * uint(TOKENS) + query) * uint(TOPK);
    for (uint selected = 0u; selected < uint(TOPK); ++selected) {
        int cache_row = indices[index_base + selected];
        bool valid = cache_row >= 0 && cache_row < int(CACHE_LEN);
        uint safe_row = valid ? uint(cache_row) : 0u;
        uint cache_base =
            (batch * uint(CACHE_LEN) + safe_row) * uint(D);
        float dot = 0.0f;
        if (valid) {
            for (
                uint dimension = local_thread;
                dimension < uint(D);
                dimension += 256u
            ) {
                dot += query_values[query_base + dimension]
                    * float(cache_values[cache_base + dimension]);
            }
        }
        dot = simd_sum(dot);
        if (lane == 0u) {
            partials[simd_group] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_thread == 0u) {
            float score = 0.0f;
            for (uint group = 0u; group < SIMD_GROUPS; ++group) {
                score += partials[group];
            }
            score = valid ? score * params[0] : -INFINITY;
            float old_max = online[0];
            float next_max = max(old_max, score);
            float old_scale = isfinite(old_max)
                ? exp(old_max - next_max)
                : 0.0f;
            float next_scale = valid ? exp(score - next_max) : 0.0f;
            online[0] = next_max;
            online[1] = online[1] * old_scale + next_scale;
            online[2] = old_scale;
            online[3] = next_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint slice = 0u; slice < VALUE_SLICES; ++slice) {
            uint dimension = local_thread + slice * 256u;
            float value = valid && dimension < uint(D)
                ? float(cache_values[cache_base + dimension])
                : 0.0f;
            accumulators[slice] = accumulators[slice] * online[2]
                + value * online[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inverse = online[1] > 0.0f ? 1.0f / online[1] : 0.0f;
    uint output_base =
        ((batch * uint(TOKENS) + query) * uint(HEADS) + head) * uint(D);
    for (uint slice = 0u; slice < VALUE_SLICES; ++slice) {
        uint dimension = local_thread + slice * 256u;
        if (dimension < uint(D)) {
            output[output_base + dimension] = accumulators[slice] * inverse;
        }
    }
"""


_GLM5_SPARSE_MLA_KERNEL = mx.fast.metal_kernel(
    name="mfq_glm5_next_sparse_mla",
    input_names=["query_values", "cache_values", "indices", "params"],
    output_names=["output"],
    source=_GLM5_SPARSE_MLA_SOURCE,
    compile_options={"math_mode": "fast"},
)


_QWEN4_SPARSE_GQA_SOURCE = r"""
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint VALUE_SLICES = (uint(D) + 255u) / 256u;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint local_thread = thread_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint head = workgroup % uint(HEADS);
    uint query_row = workgroup / uint(HEADS);
    uint query = query_row % uint(TOKENS);
    uint batch = query_row / uint(TOKENS);
    if (batch >= uint(B)) {
        return;
    }
    uint kv_head = head / uint(HEADS / KV_HEADS);

    threadgroup float partials[SIMD_GROUPS];
    threadgroup float online[4];
    if (local_thread == 0u) {
        online[0] = -INFINITY;
        online[1] = 0.0f;
        online[2] = 0.0f;
        online[3] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulators[VALUE_SLICES];
    for (uint slice = 0u; slice < VALUE_SLICES; ++slice) {
        accumulators[slice] = 0.0f;
    }
    uint query_base =
        ((batch * uint(HEADS) + head) * uint(TOKENS) + query) * uint(D);
    uint index_base =
        (batch * uint(TOKENS) + query) * uint(TOPK);
    for (uint selected = 0u; selected < uint(TOPK); ++selected) {
        int cache_row = indices[index_base + selected];
        bool valid = cache_row >= 0 && cache_row < int(CACHE_LEN);
        uint safe_row = valid ? uint(cache_row) : 0u;
        uint cache_base =
            (((batch * uint(KV_HEADS) + kv_head) * uint(CACHE_LEN)
                + safe_row) * uint(D));
        float dot = 0.0f;
        if (valid) {
            for (
                uint dimension = local_thread;
                dimension < uint(D);
                dimension += 256u
            ) {
                dot += query_values[query_base + dimension]
                    * float(key_values[cache_base + dimension]);
            }
        }
        dot = simd_sum(dot);
        if (lane == 0u) {
            partials[simd_group] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_thread == 0u) {
            float score = 0.0f;
            for (uint group = 0u; group < SIMD_GROUPS; ++group) {
                score += partials[group];
            }
            score = valid ? score * params[0] : -INFINITY;
            float old_max = online[0];
            float next_max = max(old_max, score);
            float old_scale = isfinite(old_max)
                ? exp(old_max - next_max)
                : 0.0f;
            float next_scale = valid ? exp(score - next_max) : 0.0f;
            online[0] = next_max;
            online[1] = online[1] * old_scale + next_scale;
            online[2] = old_scale;
            online[3] = next_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint slice = 0u; slice < VALUE_SLICES; ++slice) {
            uint dimension = local_thread + slice * 256u;
            float value = valid && dimension < uint(D)
                ? float(value_values[cache_base + dimension])
                : 0.0f;
            accumulators[slice] = accumulators[slice] * online[2]
                + value * online[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inverse = online[1] > 0.0f ? 1.0f / online[1] : 0.0f;
    uint output_base =
        ((batch * uint(TOKENS) + query) * uint(HEADS) + head) * uint(D);
    for (uint slice = 0u; slice < VALUE_SLICES; ++slice) {
        uint dimension = local_thread + slice * 256u;
        if (dimension < uint(D)) {
            output[output_base + dimension] = accumulators[slice] * inverse;
        }
    }
"""


_QWEN4_SPARSE_GQA_KERNEL = mx.fast.metal_kernel(
    name="mfq_qwen4_exp_sparse_gqa",
    input_names=[
        "query_values",
        "key_values",
        "value_values",
        "indices",
        "params",
    ],
    output_names=["output"],
    source=_QWEN4_SPARSE_GQA_SOURCE,
    compile_options={"math_mode": "fast"},
)


def qwen4_grouped_rms_norm(
    value: mx.array,
    weight: mx.array,
    group_size: int,
    eps: float = 1e-6,
) -> mx.array:
    """Qwen4-Exp zero-centered RMSNorm, independently per residual stream."""

    width = int(value.shape[-1])
    group = int(group_size)
    if group <= 0 or width % group or tuple(weight.shape) != (width,):
        raise ValueError("Qwen4-Exp grouped RMSNorm dimensions disagree")
    source_dtype = value.dtype
    reshaped = value.astype(mx.float32).reshape((*value.shape[:-1], width // group, group))
    normalized = reshaped * mx.rsqrt(mx.mean(reshaped * reshaped, axis=-1, keepdims=True) + eps)
    normalized = normalized.reshape(value.shape) * (1.0 + weight.astype(mx.float32))
    return normalized.astype(source_dtype)


def qwen4_gated_residual_pre(
    hyper_input: mx.array,
    norm_weight: mx.array,
    down_weight: mx.array,
    up_weight: mx.array,
    inject_weight: mx.array | None,
    *,
    hidden_size: int,
    hc_count: int = 4,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array, mx.array | None]:
    """Collapse Qwen's GR streams and produce optional branch injection gates."""

    hidden = int(hidden_size)
    streams = int(hc_count)
    width = hidden * streams
    if int(hyper_input.shape[-1]) != width:
        raise ValueError("Qwen4-Exp gated-residual input width mismatch")
    if int(down_weight.shape[1]) != width or int(up_weight.shape[0]) != width:
        raise ValueError("Qwen4-Exp gated-residual low-rank projection mismatch")
    if int(up_weight.shape[1]) != int(down_weight.shape[0]):
        raise ValueError("Qwen4-Exp gated-residual bottleneck width mismatch")
    normalized = qwen4_grouped_rms_norm(
        hyper_input,
        norm_weight,
        hidden,
        eps,
    )
    low = mx.matmul(normalized, mx.swapaxes(down_weight, -1, -2)) / streams
    low = low * mx.sigmoid(low)
    mixing = mx.sigmoid(mx.matmul(low, mx.swapaxes(up_weight, -1, -2)))
    mixed = mx.mean(
        mixing.reshape((*mixing.shape[:-1], streams, hidden))
        * normalized.reshape((*normalized.shape[:-1], streams, hidden)),
        axis=-2,
    )
    if inject_weight is None:
        return mixed, hyper_input, None
    if tuple(inject_weight.shape) != (streams, width):
        raise ValueError("Qwen4-Exp gated-residual injection projection mismatch")
    injection = 2.0 * mx.sigmoid(
        mx.matmul(normalized, mx.swapaxes(inject_weight, -1, -2)) / streams
    )
    return mixed, hyper_input, injection


def qwen4_gated_residual_post(
    branch: mx.array,
    residual: mx.array,
    injection: mx.array,
    *,
    hc_count: int = 4,
) -> mx.array:
    """Inject one Qwen branch result back into its flattened GR streams."""

    streams = int(hc_count)
    hidden = int(branch.shape[-1])
    if int(residual.shape[-1]) != streams * hidden or tuple(injection.shape) != (
        *branch.shape[:-1],
        streams,
    ):
        raise ValueError("Qwen4-Exp gated-residual post dimensions disagree")
    update = branch[..., None, :] * injection[..., :, None]
    return residual + update.reshape(residual.shape)


def glm5_mhc_pre(
    hidden_streams: mx.array,
    function_weight: mx.array,
    base: mx.array,
    scale: mx.array,
    *,
    sinkhorn_iterations: int = 20,
    hc_eps: float = 1e-6,
    rms_eps: float = 1e-5,
) -> tuple[mx.array, mx.array, mx.array]:
    """Reference-correct GLM-5.3 manifold hyper-connection collapse."""

    if hidden_streams.ndim != 4:
        raise ValueError("GLM-5-Next mHC input must have [B,T,C,H] shape")
    streams = int(hidden_streams.shape[-2])
    hidden = int(hidden_streams.shape[-1])
    mix_width = (2 + streams) * streams
    if (
        tuple(function_weight.shape) != (mix_width, streams * hidden)
        or tuple(base.shape) != (mix_width,)
        or tuple(scale.shape) != (3,)
        or sinkhorn_iterations <= 0
    ):
        raise ValueError("GLM-5-Next mHC parameter dimensions disagree")
    flat = hidden_streams.astype(mx.float32).reshape((*hidden_streams.shape[:-2], -1))
    flat = flat * mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + rms_eps)
    logits = mx.matmul(flat, mx.swapaxes(function_weight.astype(mx.float32), -1, -2))
    pre_raw, post_raw, combination_raw = mx.split(
        logits,
        [streams, 2 * streams],
        axis=-1,
    )
    pre = mx.sigmoid(pre_raw * scale[0] + base[:streams]) + hc_eps
    post = 2.0 * mx.sigmoid(post_raw * scale[1] + base[streams : 2 * streams])
    combination_logits = combination_raw.reshape(
        (*combination_raw.shape[:-1], streams, streams)
    ) * scale[2] + base[2 * streams :].reshape((streams, streams))
    combination = mx.softmax(combination_logits, axis=-1) + hc_eps
    combination = combination / (mx.sum(combination, axis=-2, keepdims=True) + hc_eps)
    for _ in range(sinkhorn_iterations - 1):
        combination = combination / (mx.sum(combination, axis=-1, keepdims=True) + hc_eps)
        combination = combination / (mx.sum(combination, axis=-2, keepdims=True) + hc_eps)
    collapsed = mx.sum(
        pre[..., :, None] * hidden_streams,
        axis=-2,
    ).astype(hidden_streams.dtype)
    return post, combination, collapsed


def glm5_mhc_post(
    branch: mx.array,
    residual: mx.array,
    post: mx.array,
    combination: mx.array,
) -> mx.array:
    """Expand a GLM branch through post gates plus the Sinkhorn stream mix."""

    if residual.ndim != 4 or branch.shape != residual.shape[:-2] + residual.shape[-1:]:
        raise ValueError("GLM-5-Next mHC branch/residual dimensions disagree")
    streams = int(residual.shape[-2])
    if tuple(post.shape) != (*branch.shape[:-1], streams) or tuple(combination.shape) != (
        *branch.shape[:-1],
        streams,
        streams,
    ):
        raise ValueError("GLM-5-Next mHC post metadata dimensions disagree")
    mixed = mx.matmul(mx.swapaxes(combination, -1, -2), residual)
    return post.astype(residual.dtype)[..., :, None] * branch[..., None, :] + mixed


def glm5_kda_forget_gate(
    hidden_states: mx.array,
    f_a_weight: mx.array,
    f_b_weight: mx.array,
    dt_bias: mx.array,
    a_log: mx.array,
    *,
    num_heads: int,
    head_dim: int,
    lower_bound: float = -5.0,
) -> mx.array:
    """Build GLM-5.3's per-channel log-decay tensor for the KDA recurrence."""

    heads = int(num_heads)
    dimension = int(head_dim)
    if (
        tuple(f_a_weight.shape) != (dimension, int(hidden_states.shape[-1]))
        or tuple(f_b_weight.shape) != (heads * dimension, dimension)
        or tuple(dt_bias.shape) != (heads * dimension,)
        or tuple(a_log.shape) != (heads,)
    ):
        raise ValueError("GLM-5-Next KDA forget-gate dimensions disagree")
    reduced = mx.matmul(hidden_states, mx.swapaxes(f_a_weight, -1, -2))
    gate = mx.matmul(reduced, mx.swapaxes(f_b_weight, -1, -2)).astype(mx.float32)
    gate = (gate + dt_bias.astype(mx.float32)).reshape(
        (*hidden_states.shape[:-1], heads, dimension)
    )
    decay_rate = mx.exp(a_log.astype(mx.float32)).reshape(*((1,) * (gate.ndim - 2)), heads, 1)
    return float(lower_bound) * mx.sigmoid(decay_rate * gate)


def qsa_block_scores(query: mx.array, pooled_keys: mx.array) -> mx.array:
    """Qwen sparse-attention score: sum of ReLU head dots per 4-token pool."""

    if query.ndim != 4 or pooled_keys.ndim != 3:
        raise ValueError("QSA query/pooled keys must have [B,T,H,D]/[B,P,D] shape")
    if int(query.shape[0]) != int(pooled_keys.shape[0]) or int(query.shape[-1]) != int(
        pooled_keys.shape[-1]
    ):
        raise ValueError("QSA query/key dimensions disagree")
    keys = mx.swapaxes(pooled_keys.astype(mx.float32), -1, -2)[:, None, :, :]
    scores = mx.matmul(query.astype(mx.float32), keys)
    return mx.sum(mx.maximum(scores, 0.0), axis=-2) / math.sqrt(int(query.shape[-1]))


def qwen4_dense_gqa_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    query_offset: int,
) -> mx.array:
    """Run dense causal Qwen GQA with an explicit cached-query offset."""

    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("Qwen4 dense GQA expects [B,H,T,D] tensors")
    batch, heads, tokens, dimension = (int(item) for item in query.shape)
    if int(key.shape[0]) != batch or heads % int(key.shape[1]) or int(key.shape[-1]) != dimension:
        raise ValueError("Qwen4 dense GQA dimensions disagree")
    keys = int(key.shape[2])
    offset = int(query_offset)
    if keys <= 0 or offset < 0 or offset + tokens > keys:
        raise ValueError("Qwen4 dense GQA query range is outside the cache")
    key_positions = mx.arange(keys, dtype=mx.int32)[None, :]
    query_positions = mx.arange(tokens, dtype=mx.int32)[:, None] + offset
    mask = (key_positions <= query_positions)[None, None, :, :]
    output = mx.fast.scaled_dot_product_attention(
        query,
        key.astype(query.dtype),
        value.astype(query.dtype),
        scale=1.0 / math.sqrt(dimension),
        mask=mask,
    )
    return mx.transpose(output, (0, 2, 1, 3))


def qwen4_sparse_gqa_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    indices: mx.array,
) -> mx.array:
    """Run Qwen QSA over explicit per-query token indices."""

    query_values = mx.contiguous(query.astype(mx.float32))
    key_values = mx.contiguous(key.astype(mx.float16))
    value_values = mx.contiguous(value.astype(mx.float16))
    selected = mx.contiguous(indices.astype(mx.int32))
    if (
        query_values.ndim != 4
        or key_values.ndim != 4
        or value_values.shape != key_values.shape
        or selected.ndim != 3
    ):
        raise ValueError("Qwen4 sparse GQA tensors have invalid ranks")
    batch, heads, tokens, dimension = (int(item) for item in query_values.shape)
    kv_heads = int(key_values.shape[1])
    if (
        int(key_values.shape[0]) != batch
        or heads % kv_heads
        or int(key_values.shape[-1]) != dimension
        or tuple(selected.shape[:2]) != (batch, tokens)
        or int(selected.shape[-1]) <= 0
    ):
        raise ValueError("Qwen4 sparse GQA dimensions disagree")
    topk = int(selected.shape[-1])
    cache_length = int(key_values.shape[2])
    params = mx.array([1.0 / math.sqrt(dimension)], dtype=mx.float32)
    return _QWEN4_SPARSE_GQA_KERNEL(
        inputs=[query_values, key_values, value_values, selected, params],
        template=[
            ("B", batch),
            ("HEADS", heads),
            ("KV_HEADS", kv_heads),
            ("TOKENS", tokens),
            ("D", dimension),
            ("CACHE_LEN", cache_length),
            ("TOPK", topk),
        ],
        grid=(batch * heads * tokens * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, tokens, heads, dimension)],
        output_dtypes=[mx.float32],
    )[0]


def qwen4_ple_dilated_conv_silu(
    value: mx.array,
    weight: mx.array,
    state: mx.array | None,
    *,
    dilation: int,
) -> tuple[mx.array, mx.array]:
    """Apply PLE's causal dilated depthwise convolution and update its state."""

    if value.ndim != 3:
        raise ValueError("Qwen4 PLE convolution expects [B,T,C]")
    batch, tokens, channels = (int(item) for item in value.shape)
    packed_weight = weight
    if packed_weight.ndim == 3:
        if tuple(packed_weight.shape[:2]) != (channels, 1):
            raise ValueError("Qwen4 PLE [C,1,K] weight dimensions disagree")
        packed_weight = packed_weight[:, 0]
    if packed_weight.ndim != 2 or int(packed_weight.shape[0]) != channels:
        raise ValueError("Qwen4 PLE convolution weight must have [C,K] shape")
    kernel = int(packed_weight.shape[1])
    selected_dilation = int(dilation)
    if kernel <= 0 or selected_dilation <= 0:
        raise ValueError("Qwen4 PLE kernel and dilation must be positive")
    state_length = (kernel - 1) * selected_dilation
    state_values = (
        mx.zeros((batch, state_length, channels), dtype=value.dtype)
        if state is None
        else state.astype(value.dtype)
    )
    if tuple(state_values.shape) != (batch, state_length, channels):
        raise ValueError("Qwen4 PLE convolution state dimensions disagree")
    combined = mx.concatenate((state_values, value), axis=1)
    output = mx.zeros((batch, tokens, channels), dtype=mx.float32)
    for tap in range(kernel):
        start = tap * selected_dilation
        output = (
            output
            + combined[:, start : start + tokens].astype(mx.float32)
            * (packed_weight[:, tap].astype(mx.float32)[None, None])
        )
    output = output * mx.sigmoid(output)
    next_state = combined[:, -state_length:] if state_length else combined[:, :0]
    return output.astype(value.dtype), mx.contiguous(next_state)


def glm5_kpool_states(
    keys: mx.array,
    gate_scores: mx.array,
    ape: mx.array,
    *,
    pool_size: int = 4,
) -> mx.array:
    """Pool complete GLM indexer groups with its learned channel-wise softmax."""

    if keys.ndim != 3 or tuple(gate_scores.shape) != tuple(keys.shape):
        raise ValueError("GLM k-pool keys and gates must share [B,T,D] shape")
    pool = int(pool_size)
    complete = int(keys.shape[1]) // pool
    if tuple(ape.shape) != (pool, int(keys.shape[-1])):
        raise ValueError("GLM k-pool APE dimensions disagree")
    if complete == 0:
        return mx.zeros((int(keys.shape[0]), 0, int(keys.shape[-1])), keys.dtype)
    grouped_keys = keys[:, : complete * pool].reshape(
        (int(keys.shape[0]), complete, pool, int(keys.shape[-1]))
    )
    grouped_gates = gate_scores[:, : complete * pool].reshape(grouped_keys.shape)
    probabilities = mx.softmax(grouped_gates.astype(mx.float32) + ape[None, None], axis=-2)
    return mx.sum(probabilities.astype(keys.dtype) * grouped_keys, axis=-2)


def glm5_kpool_scores(
    query: mx.array,
    pooled_keys: mx.array,
    head_weights: mx.array,
) -> mx.array:
    """Score GLM's compressed index pools and reduce the 32 index heads."""

    if query.ndim != 4 or pooled_keys.ndim != 3 or head_weights.ndim != 3:
        raise ValueError("GLM k-pool score tensors have invalid ranks")
    batch, tokens, heads, dimension = (int(item) for item in query.shape)
    if (
        tuple(pooled_keys.shape[:1]) != (batch,)
        or int(pooled_keys.shape[-1]) != dimension
        or tuple(head_weights.shape) != (batch, tokens, heads)
    ):
        raise ValueError("GLM k-pool score dimensions disagree")
    keys = mx.swapaxes(pooled_keys.astype(mx.float32), -1, -2)[:, None, :, :]
    scores = mx.maximum(mx.matmul(query.astype(mx.float32), keys), 0.0)
    scores = scores / math.sqrt(dimension)
    weights = head_weights.astype(mx.float32) / math.sqrt(heads)
    return mx.sum(scores * weights[..., :, None], axis=-2)


def glm5_dense_mla_attention(
    query: mx.array,
    cache: mx.array,
    *,
    query_offset: int,
    scale: float | None = None,
) -> mx.array:
    """Run dense NoPE MLA over a shared latent K/V cache."""

    if query.ndim != 4 or cache.ndim != 3:
        raise ValueError("GLM dense MLA expects [B,H,T,D] and [B,K,D]")
    batch, _heads, tokens, dimension = (int(item) for item in query.shape)
    if tuple(cache.shape[:1]) != (batch,) or int(cache.shape[-1]) != dimension:
        raise ValueError("GLM dense MLA query/cache dimensions disagree")
    keys = int(cache.shape[1])
    offset = int(query_offset)
    if keys <= 0 or offset < 0 or offset + tokens > keys:
        raise ValueError("GLM dense MLA query range is outside the cache")
    key_value = cache.astype(query.dtype)[:, None, :, :]
    key_positions = mx.arange(keys, dtype=mx.int32)[None, :]
    query_positions = mx.arange(tokens, dtype=mx.int32)[:, None] + offset
    mask = (key_positions <= query_positions)[None, None, :, :]
    selected_scale = 1.0 / math.sqrt(dimension) if scale is None else float(scale)
    output = mx.fast.scaled_dot_product_attention(
        query,
        key_value,
        key_value,
        scale=selected_scale,
        mask=mask,
    )
    return mx.transpose(output, (0, 2, 1, 3))


def glm5_sparse_mla_attention(
    query: mx.array,
    cache: mx.array,
    indices: mx.array,
    *,
    scale: float | None = None,
) -> mx.array:
    """Run GLM-5-Next NoPE MLA over explicit per-query cache indices."""

    query_values = mx.contiguous(query.astype(mx.float32))
    cache_values = mx.contiguous(cache.astype(mx.float16))
    selected = mx.contiguous(indices.astype(mx.int32))
    if query_values.ndim != 4 or cache_values.ndim != 3 or selected.ndim != 3:
        raise ValueError("GLM sparse MLA tensors have invalid ranks")
    batch, heads, tokens, dimension = (int(item) for item in query_values.shape)
    if (
        tuple(cache_values.shape[:1]) != (batch,)
        or int(cache_values.shape[-1]) != dimension
        or tuple(selected.shape[:2]) != (batch, tokens)
        or int(selected.shape[-1]) <= 0
    ):
        raise ValueError("GLM sparse MLA dimensions disagree")
    topk = int(selected.shape[-1])
    cache_length = int(cache_values.shape[1])
    selected_scale = 1.0 / math.sqrt(dimension) if scale is None else float(scale)
    params = mx.array([selected_scale], dtype=mx.float32)
    return _GLM5_SPARSE_MLA_KERNEL(
        inputs=[query_values, cache_values, selected, params],
        template=[
            ("B", batch),
            ("HEADS", heads),
            ("TOKENS", tokens),
            ("D", dimension),
            ("CACHE_LEN", cache_length),
            ("TOPK", topk),
        ],
        grid=(batch * heads * tokens * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, tokens, heads, dimension)],
        output_dtypes=[mx.float32],
    )[0]


__all__ = [
    "glm5_kda_forget_gate",
    "glm5_dense_mla_attention",
    "glm5_kpool_scores",
    "glm5_kpool_states",
    "glm5_mhc_post",
    "glm5_mhc_pre",
    "glm5_sparse_mla_attention",
    "qsa_block_scores",
    "qwen4_dense_gqa_attention",
    "qwen4_gated_residual_post",
    "qwen4_gated_residual_pre",
    "qwen4_grouped_rms_norm",
    "qwen4_ple_dilated_conv_silu",
    "qwen4_sparse_gqa_attention",
]
