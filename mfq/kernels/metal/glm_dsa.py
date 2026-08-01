"""Metal equivalents of the GLM DSA and sparse-MLA CUDA kernels.

The indexer uses an 8-SIMD-group 32x64x128 matrix tile, matching the CUDA
kernel's on-chip head/key reduction.  Sparse MLA uses one threadgroup per
query head and performs an online-softmax reduction over the selected cache
rows without materializing a dense attention matrix.
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


_INTERLEAVED_ROPE_SOURCE = r"""
    uint pair = thread_position_in_grid.x;
    if (pair >= uint(PAIRS)) {
        return;
    }
    uint pair_dimension = pair % uint(D / 2);
    uint row = pair / uint(D / 2);
    uint token = row % uint(TOKENS);
    uint base = pair * 2u;
    float first = float(x[base]);
    float second = float(x[base + 1u]);
    if (pair_dimension * 2u < uint(ROTARY_DIM)) {
        int position = positions[token];
        position = max(0, min(position, TABLE_LEN - 1));
        uint table_index = uint(position) * uint(TABLE_STRIDE) + pair_dimension;
        float cosine = cos_table[table_index];
        float sine = sin_table[table_index];
        out[base] = T(first * cosine - second * sine);
        out[base + 1u] = T(second * cosine + first * sine);
    } else {
        out[base] = x[base];
        out[base + 1u] = x[base + 1u];
    }
"""


_INDEXER_LAYER_NORM_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint row = threadgroup_position_in_grid.x;
    if (row >= uint(ROWS)) {
        return;
    }
    uint offset = row * 128u;
    float values[4];
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (uint item = 0u; item < 4u; ++item) {
        uint dimension = lane + item * 32u;
        float value = float(x[offset + dimension]);
        values[item] = value;
        sum += value;
        square_sum += value * value;
    }
    sum = simd_sum(sum);
    square_sum = simd_sum(square_sum);
    float mean = sum / 128.0f;
    float variance = max(square_sum / 128.0f - mean * mean, 0.0f);
    float inverse = rsqrt(variance + params[0]);
    for (uint item = 0u; item < 4u; ++item) {
        uint dimension = lane + item * 32u;
        out[offset + dimension] = half(
            (values[item] - mean) * inverse * weight[dimension]
                + bias[dimension]
        );
    }
"""


_CACHE_WRITE_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= uint(CACHE_SIZE)) {
        return;
    }
    uint dimension = index % uint(D);
    uint row = index / uint(D);
    uint position = row % uint(MAX_SEQ);
    uint batch = row / uint(MAX_SEQ);
    T value = cache[index];
    for (uint token = 0u; token < uint(TOKENS); ++token) {
        int target = positions[token];
        if (target >= 0 && uint(target) == position) {
            value = values[
                (batch * uint(TOKENS) + token) * uint(D) + dimension
            ];
        }
    }
    out[index] = value;
"""


_INDEXER_SCORES_SOURCE = r"""
    constexpr uint HEADS = 32u;
    constexpr uint DIM = 128u;
    constexpr uint KEY_TILE = 64u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint local_thread = thread_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint key_tile_index = workgroup % uint(KEY_TILES);
    uint query_row = workgroup / uint(KEY_TILES);
    uint query = query_row % uint(M);
    uint batch = query_row / uint(M);
    if (batch >= uint(B)) {
        return;
    }
    uint key_base = key_tile_index * KEY_TILE;

    threadgroup half key_tile[DIM * KEY_TILE];
    threadgroup float score_tile[HEADS * KEY_TILE];
    for (
        uint index = local_thread;
        index < DIM * KEY_TILE;
        index += 256u
    ) {
        uint dimension = index / KEY_TILE;
        uint local_key = index - dimension * KEY_TILE;
        uint key = key_base + local_key;
        key_tile[index] = key < uint(K)
            ? k[
                (batch * uint(K_STRIDE) + key) * DIM + dimension
            ]
            : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint head_base = (simd_group / 4u) * 16u;
    uint local_key_base = (simd_group & 3u) * 16u;
    uint quadrant = lane / 4u;
    uint fragment_row = (quadrant & 4u) + ((lane / 2u) & 3u);
    uint fragment_col = (quadrant & 2u) * 2u + (lane & 1u) * 2u;
    metal::simdgroup_matrix<float, 8, 8> c00;
    metal::simdgroup_matrix<float, 8, 8> c01;
    metal::simdgroup_matrix<float, 8, 8> c10;
    metal::simdgroup_matrix<float, 8, 8> c11;
    c00.thread_elements()[0] = 0.0f;
    c00.thread_elements()[1] = 0.0f;
    c01.thread_elements()[0] = 0.0f;
    c01.thread_elements()[1] = 0.0f;
    c10.thread_elements()[0] = 0.0f;
    c10.thread_elements()[1] = 0.0f;
    c11.thread_elements()[0] = 0.0f;
    c11.thread_elements()[1] = 0.0f;
    uint q_base = (batch * uint(M) + query) * HEADS * DIM;
    for (uint dimension = 0u; dimension < DIM; dimension += 8u) {
        metal::simdgroup_matrix<half, 8, 8> a0;
        metal::simdgroup_matrix<half, 8, 8> a1;
        metal::simdgroup_matrix<half, 8, 8> b0;
        metal::simdgroup_matrix<half, 8, 8> b1;
        a0.thread_elements()[0] = q[
            q_base + (head_base + fragment_row) * DIM
                + dimension + fragment_col
        ];
        a0.thread_elements()[1] = q[
            q_base + (head_base + fragment_row) * DIM
                + dimension + fragment_col + 1u
        ];
        a1.thread_elements()[0] = q[
            q_base + (head_base + 8u + fragment_row) * DIM
                + dimension + fragment_col
        ];
        a1.thread_elements()[1] = q[
            q_base + (head_base + 8u + fragment_row) * DIM
                + dimension + fragment_col + 1u
        ];
        b0.thread_elements()[0] = key_tile[
            (dimension + fragment_row) * KEY_TILE
                + local_key_base + fragment_col
        ];
        b0.thread_elements()[1] = key_tile[
            (dimension + fragment_row) * KEY_TILE
                + local_key_base + fragment_col + 1u
        ];
        b1.thread_elements()[0] = key_tile[
            (dimension + fragment_row) * KEY_TILE
                + local_key_base + 8u + fragment_col
        ];
        b1.thread_elements()[1] = key_tile[
            (dimension + fragment_row) * KEY_TILE
                + local_key_base + 8u + fragment_col + 1u
        ];
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }

    uint head0 = head_base + fragment_row;
    uint head1 = head0 + 8u;
    uint key0 = local_key_base + fragment_col;
    uint key1 = key0 + 8u;
    score_tile[head0 * KEY_TILE + key0] = c00.thread_elements()[0];
    score_tile[head0 * KEY_TILE + key0 + 1u] = c00.thread_elements()[1];
    score_tile[head0 * KEY_TILE + key1] = c01.thread_elements()[0];
    score_tile[head0 * KEY_TILE + key1 + 1u] = c01.thread_elements()[1];
    score_tile[head1 * KEY_TILE + key0] = c10.thread_elements()[0];
    score_tile[head1 * KEY_TILE + key0 + 1u] = c10.thread_elements()[1];
    score_tile[head1 * KEY_TILE + key1] = c11.thread_elements()[0];
    score_tile[head1 * KEY_TILE + key1 + 1u] = c11.thread_elements()[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_thread < KEY_TILE) {
        uint key = key_base + local_thread;
        if (key < uint(K)) {
            float sum = 0.0f;
            uint weight_base = (batch * uint(M) + query) * HEADS;
            for (uint head = 0u; head < HEADS; ++head) {
                float dot = score_tile[head * KEY_TILE + local_thread]
                    * 0.08838834764831845f;
                sum += max(dot, 0.0f) * weights[weight_base + head];
            }
            uint visible_keys = uint(K);
            uint absolute_query = uint(QUERY_OFFSET) + query;
            if (USE_SEQ_LENS != 0) {
                visible_keys = min(uint(K), uint(max(seq_lens[batch], 0)));
                absolute_query = visible_keys >= uint(M)
                    ? visible_keys - uint(M) + query
                    : query;
            }
            out[(batch * uint(M) + query) * uint(K) + key] =
                key < visible_keys && key <= absolute_query
                ? sum * 0.1767766952966369f
                : -INFINITY;
        }
    }
"""


_SPARSE_MLA_SOURCE = r"""
    constexpr uint HEADS = 64u;
    constexpr uint DQ = 576u;
    constexpr uint DV = 512u;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint local_thread = thread_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint head = workgroup % HEADS;
    uint query_row = workgroup / HEADS;
    uint query = query_row % uint(M);
    uint batch = query_row / uint(M);
    if (batch >= uint(B)) {
        return;
    }

    threadgroup float partials[8];
    threadgroup float state[4];
    if (local_thread == 0u) {
        state[0] = -INFINITY;
        state[1] = 0.0f;
        state[2] = 0.0f;
        state[3] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;
    uint query_base = ((batch * HEADS + head) * uint(M) + query) * DQ;
    uint index_base = (batch * uint(M) + query) * uint(TOPK);
    for (uint selected = 0u; selected < uint(TOPK); ++selected) {
        int cache_row = indices[index_base + selected];
        bool valid = cache_row >= 0 && cache_row < int(MAX_SEQ);
        uint safe_row = valid ? uint(cache_row) : 0u;
        uint cache_base = (batch * uint(MAX_SEQ) + safe_row) * DQ;
        float dot = 0.0f;
        if (valid) {
            for (
                uint dimension = local_thread;
                dimension < DQ;
                dimension += 256u
            ) {
                dot += q[query_base + dimension]
                    * float(kv[cache_base + dimension]);
            }
        }
        dot = simd_sum(dot);
        if (lane == 0u) {
            partials[simd_group] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_thread == 0u) {
            float score = 0.0f;
            for (uint group = 0u; group < 8u; ++group) {
                score += partials[group];
            }
            score = valid ? score * params[0] : -INFINITY;
            float old_max = state[0];
            float new_max = max(old_max, score);
            float old_scale = isfinite(old_max) ? exp(old_max - new_max) : 0.0f;
            float new_scale = valid ? exp(score - new_max) : 0.0f;
            state[0] = new_max;
            state[1] = state[1] * old_scale + new_scale;
            state[2] = old_scale;
            state[3] = new_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float old_scale = state[2];
        float new_scale = state[3];
        uint value_dimension0 = local_thread;
        uint value_dimension1 = local_thread + 256u;
        float value0 = valid
            ? float(kv[cache_base + value_dimension0])
            : 0.0f;
        float value1 = valid
            ? float(kv[cache_base + value_dimension1])
            : 0.0f;
        accumulator0 = accumulator0 * old_scale + value0 * new_scale;
        accumulator1 = accumulator1 * old_scale + value1 * new_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inverse = state[1] > 0.0f ? 1.0f / state[1] : 0.0f;
    uint output_base = ((batch * uint(M) + query) * HEADS + head) * DV;
    out[output_base + local_thread] = accumulator0 * inverse;
    out[output_base + local_thread + 256u] = accumulator1 * inverse;
"""


_INTERLEAVED_ROPE_KERNEL = mx.fast.metal_kernel(
    name="mfq_glm_interleaved_rope",
    input_names=["x", "positions", "cos_table", "sin_table"],
    output_names=["out"],
    source=_INTERLEAVED_ROPE_SOURCE,
    compile_options={"math_mode": "fast"},
)

_INDEXER_LAYER_NORM_KERNEL = mx.fast.metal_kernel(
    name="mfq_glm_indexer_layer_norm",
    input_names=["x", "weight", "bias", "params"],
    output_names=["out"],
    source=_INDEXER_LAYER_NORM_SOURCE,
    compile_options={"math_mode": "fast"},
)

_CACHE_WRITE_KERNEL = mx.fast.metal_kernel(
    name="mfq_glm_dsa_cache_write",
    input_names=["cache", "values", "positions"],
    output_names=["out"],
    source=_CACHE_WRITE_SOURCE,
    compile_options={"math_mode": "fast"},
)

_INDEXER_SCORES_KERNEL = mx.fast.metal_kernel(
    name="mfq_glm_dsa_indexer_scores",
    input_names=["q", "k", "weights", "seq_lens"],
    output_names=["out"],
    source=_INDEXER_SCORES_SOURCE,
    compile_options={"math_mode": "fast"},
)

_SPARSE_MLA_KERNEL = mx.fast.metal_kernel(
    name="mfq_glm_sparse_mla",
    input_names=["q", "kv", "indices", "params"],
    output_names=["out"],
    source=_SPARSE_MLA_SOURCE,
    compile_options={"math_mode": "fast"},
)


def _array(value: mx.array | np.ndarray, dtype: mx.Dtype) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    return mx.contiguous(result.astype(dtype))


def glm_interleaved_rope(
    x: mx.array | np.ndarray,
    positions: mx.array | np.ndarray,
    cos: mx.array | np.ndarray,
    sin: mx.array | np.ndarray,
    rotary_dim: int,
) -> mx.array:
    """Apply GLM's adjacent-pair RoPE to ``[B,H,T,D]``."""

    source = x if isinstance(x, mx.array) else mx.array(x)
    if source.dtype not in (mx.float16, mx.float32):
        source = source.astype(mx.float16)
    source = mx.contiguous(source)
    position_ids = _array(positions, mx.int32)
    cosine = _array(cos, mx.float32)
    sine = _array(sin, mx.float32)
    if source.ndim != 4:
        raise ValueError("GLM interleaved RoPE expects [B,H,T,D]")
    _, _, tokens, dimension = (int(value) for value in source.shape)
    rotary = int(rotary_dim)
    if (
        dimension % 2
        or rotary <= 0
        or rotary > dimension
        or rotary % 2
        or position_ids.ndim != 1
        or int(position_ids.size) != tokens
        or cosine.ndim != 2
        or sine.shape != cosine.shape
        or int(cosine.shape[1]) < rotary // 2
    ):
        raise ValueError("GLM interleaved RoPE shape mismatch")
    pairs = int(source.size) // 2
    return _INTERLEAVED_ROPE_KERNEL(
        inputs=[source, position_ids, cosine, sine],
        template=[
            ("T", source.dtype),
            ("PAIRS", pairs),
            ("TOKENS", tokens),
            ("D", dimension),
            ("ROTARY_DIM", rotary),
            ("TABLE_LEN", int(cosine.shape[0])),
            ("TABLE_STRIDE", int(cosine.shape[1])),
        ],
        grid=(pairs, 1, 1),
        threadgroup=(min(256, max(1, pairs)), 1, 1),
        output_shapes=[tuple(int(value) for value in source.shape)],
        output_dtypes=[source.dtype],
    )[0]


def glm_dsa_indexer_layer_norm(
    x: mx.array | np.ndarray,
    weight: mx.array | np.ndarray,
    bias: mx.array | np.ndarray,
    eps: float = 1e-5,
) -> mx.array:
    """Layer-normalize the fixed 128-wide GLM indexer representation."""

    source = _array(x, mx.float16)
    norm_weight = _array(weight, mx.float32)
    norm_bias = _array(bias, mx.float32)
    if (
        source.ndim < 1
        or int(source.shape[-1]) != 128
        or int(norm_weight.size) != 128
        or int(norm_bias.size) != 128
        or source.size == 0
    ):
        raise ValueError("GLM indexer layer norm expects [...,128], [128], [128]")
    rows = int(source.size) // 128
    params = mx.array([float(eps)], dtype=mx.float32)
    return _INDEXER_LAYER_NORM_KERNEL(
        inputs=[source, norm_weight, norm_bias, params],
        template=[("ROWS", rows)],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[tuple(int(value) for value in source.shape)],
        output_dtypes=[mx.float16],
    )[0]


def glm_dsa_cache_write(
    cache: mx.array | np.ndarray,
    values: mx.array | np.ndarray,
    positions: mx.array | np.ndarray,
) -> mx.array:
    """Return a GLM cache with ``[B,T,D]`` values written at device positions."""

    target = cache if isinstance(cache, mx.array) else mx.array(cache)
    if target.dtype not in (mx.float16, mx.float32):
        target = target.astype(mx.float16)
    target = mx.contiguous(target)
    updates = _array(values, target.dtype)
    position_ids = _array(positions, mx.int32)
    if (
        target.ndim != 3
        or updates.ndim != 3
        or int(target.shape[0]) != int(updates.shape[0])
        or int(target.shape[2]) != int(updates.shape[2])
        or int(position_ids.size) != int(updates.shape[1])
    ):
        raise ValueError("GLM cache write expects [B,S,D], [B,T,D], and [T]")
    batch, max_seq, dimension = (int(value) for value in target.shape)
    tokens = int(updates.shape[1])
    size = int(target.size)
    return _CACHE_WRITE_KERNEL(
        inputs=[target, updates, position_ids],
        template=[
            ("T", target.dtype),
            ("CACHE_SIZE", size),
            ("MAX_SEQ", max_seq),
            ("TOKENS", tokens),
            ("D", dimension),
        ],
        grid=(size, 1, 1),
        threadgroup=(min(256, max(1, size)), 1, 1),
        output_shapes=[(batch, max_seq, dimension)],
        output_dtypes=[target.dtype],
    )[0]


def _glm_dsa_indexer_scores(
    q: mx.array | np.ndarray,
    k: mx.array | np.ndarray,
    weights: mx.array | np.ndarray,
    *,
    query_offset: int,
    logical_k: int,
    seq_lens: mx.array | np.ndarray | None,
) -> mx.array:
    query = _array(q, mx.float16)
    key = _array(k, mx.float16)
    head_weights = _array(weights, mx.float32)
    if (
        query.ndim != 4
        or key.ndim != 3
        or head_weights.ndim != 3
        or int(query.shape[2]) != 32
        or int(query.shape[3]) != 128
        or int(key.shape[0]) != int(query.shape[0])
        or int(key.shape[2]) != 128
        or tuple(int(value) for value in head_weights.shape)
        != (int(query.shape[0]), int(query.shape[1]), 32)
    ):
        raise ValueError("GLM indexer score shape mismatch")
    batch, queries = (int(value) for value in query.shape[:2])
    key_stride = int(key.shape[1])
    keys = int(logical_k)
    if keys <= 0 or keys > key_stride:
        raise ValueError("logical_k must be in [1,key cache length]")
    use_seq_lens = seq_lens is not None
    if use_seq_lens:
        lengths = _array(seq_lens, mx.int32)
        if int(lengths.size) != batch:
            raise ValueError("seq_lens must contain one length per batch")
    else:
        offset = int(query_offset)
        if offset < 0 or offset + queries > keys:
            raise ValueError("invalid causal query offset")
        lengths = mx.zeros((batch,), dtype=mx.int32)
    key_tiles = (keys + 63) // 64
    return _INDEXER_SCORES_KERNEL(
        inputs=[query, key, head_weights, lengths],
        template=[
            ("B", batch),
            ("M", queries),
            ("K", keys),
            ("K_STRIDE", key_stride),
            ("KEY_TILES", key_tiles),
            ("QUERY_OFFSET", int(query_offset)),
            ("USE_SEQ_LENS", int(use_seq_lens)),
        ],
        grid=(batch * queries * key_tiles * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, queries, keys)],
        output_dtypes=[mx.float32],
    )[0]


def glm_dsa_indexer_scores(
    q: mx.array | np.ndarray,
    k: mx.array | np.ndarray,
    weights: mx.array | np.ndarray,
    query_offset: int,
    logical_k: int | None = None,
) -> mx.array:
    """Compute weighted-ReLU DSA index scores with a causal prefill mask."""

    key_stride = int(k.shape[1])
    return _glm_dsa_indexer_scores(
        q,
        k,
        weights,
        query_offset=int(query_offset),
        logical_k=key_stride if logical_k is None else int(logical_k),
        seq_lens=None,
    )


def glm_dsa_indexer_scores_decode(
    q: mx.array | np.ndarray,
    k: mx.array | np.ndarray,
    weights: mx.array | np.ndarray,
    seq_len: mx.array | np.ndarray,
    planned_k: int,
) -> mx.array:
    """Decode variant of the GLM indexer with per-batch visible lengths."""

    if int(q.shape[1]) != 1:
        raise ValueError("GLM decode indexer requires one query token")
    return _glm_dsa_indexer_scores(
        q,
        k,
        weights,
        query_offset=0,
        logical_k=int(planned_k),
        seq_lens=seq_len,
    )


def attention_glm_mla_sparse(
    q: mx.array | np.ndarray,
    kv: mx.array | np.ndarray,
    indices: mx.array | np.ndarray,
    meta: mx.array | np.ndarray | None = None,
    scale: float | None = None,
) -> mx.array:
    """Run sparse GLM MLA over selected cache rows.

    ``meta`` is accepted for CUDA API compatibility; Metal's single-workgroup
    online softmax does not require a stream-K fixup workspace.
    """

    del meta
    query = _array(q, mx.float32)
    cache = _array(kv, mx.float16)
    selected = _array(indices, mx.int32)
    if (
        query.ndim != 4
        or tuple(int(value) for value in (query.shape[1], query.shape[3])) != (64, 576)
        or cache.ndim != 3
        or int(cache.shape[0]) != int(query.shape[0])
        or int(cache.shape[2]) != 576
        or selected.ndim != 3
        or int(selected.shape[0]) != int(query.shape[0])
        or int(selected.shape[1]) != int(query.shape[2])
        or int(selected.shape[2]) <= 0
        or int(selected.shape[2]) % 32
    ):
        raise ValueError("GLM sparse MLA shape mismatch")
    batch = int(query.shape[0])
    queries = int(query.shape[2])
    max_seq = int(cache.shape[1])
    topk = int(selected.shape[2])
    selected_scale = 1.0 / math.sqrt(576.0) if scale is None else float(scale)
    params = mx.array([selected_scale], dtype=mx.float32)
    return _SPARSE_MLA_KERNEL(
        inputs=[query, cache, selected, params],
        template=[
            ("B", batch),
            ("M", queries),
            ("MAX_SEQ", max_seq),
            ("TOPK", topk),
        ],
        grid=(batch * queries * 64 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, queries, 64, 512)],
        output_dtypes=[mx.float32],
    )[0]


def attention_glm_mla_dense(
    q: mx.array | np.ndarray,
    kv: mx.array | np.ndarray,
    logical_len: int | None = None,
    scale: float | None = None,
) -> mx.array:
    """Run dense causal GLM MLA with 576-wide keys and 512-wide values.

    MLX's fused SDPA accepts different key and value widths, so the dense
    prefix stays on the system Metal attention kernel without materializing
    the ``[heads,queries,keys]`` score tensor.
    """

    query = _array(q, mx.float32)
    cache = _array(kv, mx.float16)
    if (
        query.ndim != 4
        or tuple(int(value) for value in (query.shape[1], query.shape[3])) != (64, 576)
        or cache.ndim != 3
        or int(cache.shape[0]) != int(query.shape[0])
        or int(cache.shape[2]) != 576
    ):
        raise ValueError("GLM dense MLA shape mismatch")
    keys = int(cache.shape[1]) if logical_len is None else int(logical_len)
    if keys <= 0 or keys > int(cache.shape[1]) or int(query.shape[2]) > keys:
        raise ValueError("GLM dense MLA logical length is invalid")
    selected_scale = 1.0 / math.sqrt(256.0) if scale is None else float(scale)
    key = cache[:, None, :keys, :].astype(mx.float32)
    value = key[..., :512]
    output = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=selected_scale,
        mask="causal",
    )
    return mx.transpose(output, (0, 2, 1, 3))


__all__ = [
    "attention_glm_mla_dense",
    "attention_glm_mla_sparse",
    "glm_dsa_cache_write",
    "glm_dsa_indexer_layer_norm",
    "glm_dsa_indexer_scores",
    "glm_dsa_indexer_scores_decode",
    "glm_interleaved_rope",
]
