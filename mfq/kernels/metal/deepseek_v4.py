"""DeepSeek-V4 indexer, sparse-attention, and cache-quantization kernels."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc


_FP4_HEADER = r"""
    METAL_FUNC float mfq_dsv4_pow2_ceil(float value) {
        uint bits = as_type<uint>(value);
        uint exponent = (bits >> 23u) & 0xffu;
        bool has_mantissa = (bits & 0x7fffffu) != 0u;
        return as_type<float>((exponent + uint(has_mantissa)) << 23u);
    }

    METAL_FUNC float mfq_dsv4_fp4(float value, float scale) {
        float normalized = clamp(value / scale, -6.0f, 6.0f);
        float magnitude = abs(normalized);
        float quantized;
        if (magnitude <= 0.25f) {
            quantized = 0.0f;
        } else if (magnitude < 0.75f) {
            quantized = 0.5f;
        } else if (magnitude <= 1.25f) {
            quantized = 1.0f;
        } else if (magnitude < 1.75f) {
            quantized = 1.5f;
        } else if (magnitude <= 2.5f) {
            quantized = 2.0f;
        } else if (magnitude < 3.5f) {
            quantized = 3.0f;
        } else if (magnitude <= 5.0f) {
            quantized = 4.0f;
        } else {
            quantized = 6.0f;
        }
        return copysign(quantized * scale, normalized);
    }
"""


_COMPRESS_HEADER = (
    _FP4_HEADER
    + r"""
    METAL_FUNC float mfq_dsv4_bf16_round(float value) {
        uint bits = as_type<uint>(value);
        uint rounding = 0x7fffu + ((bits >> 16u) & 1u);
        return as_type<float>((bits + rounding) & 0xffff0000u);
    }

    METAL_FUNC float mfq_dsv4_fp8_e4m3(float value, float scale) {
        float normalized = clamp(value / scale, -448.0f, 448.0f);
        float magnitude = abs(normalized);
        float quantized;
        if (magnitude < 0x1p-6f) {
            quantized = rint(magnitude * 512.0f) / 512.0f;
        } else {
            float exponent = floor(log2(magnitude));
            float step = exp2(exponent - 3.0f);
            quantized = min(rint(magnitude / step) * step, 448.0f);
        }
        return copysign(quantized * scale, normalized);
    }
"""
)


_COMPRESS_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x;
    uint dimension = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    if (row >= uint(B * W) || dimension >= uint(HEAD_DIM)) {
        return;
    }
    uint batch = row / uint(W);
    uint window_index = row - batch * uint(W);
    constexpr uint OUT_DIM = OVERLAP != 0 ? 2u * uint(HEAD_DIM) : uint(HEAD_DIM);
    constexpr uint CANDIDATES = OVERLAP != 0 ? 2u * uint(RATIO) : uint(RATIO);
    threadgroup float compressed[HEAD_DIM];
    threadgroup float quant_scales[HEAD_DIM / 32];
    threadgroup float warp_sums[HEAD_DIM / 32];
    threadgroup float inverse_rms;

    float maximum = -INFINITY;
    for (uint candidate = 0u; candidate < CANDIDATES; ++candidate) {
        float score = -INFINITY;
        if (OVERLAP == 0) {
            uint index = (
                ((batch * uint(W) + window_index) * uint(RATIO) + candidate)
                    * OUT_DIM + dimension
            );
            score = float(gate[index])
                + ape[candidate * OUT_DIM + dimension];
        } else if (candidate < uint(RATIO)) {
            if (window_index > 0u) {
                uint index = (
                    ((batch * uint(W) + window_index - 1u) * uint(RATIO)
                        + candidate) * OUT_DIM + dimension
                );
                score = float(gate[index])
                    + ape[candidate * OUT_DIM + dimension];
            } else if (HAS_PREV != 0) {
                uint index = (
                    (batch * uint(RATIO) + candidate) * uint(HEAD_DIM)
                        + dimension
                );
                score = float(prev_gate[index])
                    + ape[candidate * OUT_DIM + dimension];
            }
        } else {
            uint ratio_index = candidate - uint(RATIO);
            uint index = (
                ((batch * uint(W) + window_index) * uint(RATIO)
                    + ratio_index) * OUT_DIM + uint(HEAD_DIM) + dimension
            );
            score = float(gate[index])
                + ape[ratio_index * OUT_DIM + uint(HEAD_DIM) + dimension];
        }
        maximum = max(maximum, score);
    }

    float weighted = 0.0f;
    float denominator = 0.0f;
    for (uint candidate = 0u; candidate < CANDIDATES; ++candidate) {
        float value = 0.0f;
        float score = -INFINITY;
        if (OVERLAP == 0) {
            uint index = (
                ((batch * uint(W) + window_index) * uint(RATIO) + candidate)
                    * OUT_DIM + dimension
            );
            value = float(kv[index]);
            score = float(gate[index])
                + ape[candidate * OUT_DIM + dimension];
        } else if (candidate < uint(RATIO)) {
            if (window_index > 0u) {
                uint index = (
                    ((batch * uint(W) + window_index - 1u) * uint(RATIO)
                        + candidate) * OUT_DIM + dimension
                );
                value = float(kv[index]);
                score = float(gate[index])
                    + ape[candidate * OUT_DIM + dimension];
            } else if (HAS_PREV != 0) {
                uint index = (
                    (batch * uint(RATIO) + candidate) * uint(HEAD_DIM)
                        + dimension
                );
                value = float(prev_kv[index]);
                score = float(prev_gate[index])
                    + ape[candidate * OUT_DIM + dimension];
            }
        } else {
            uint ratio_index = candidate - uint(RATIO);
            uint index = (
                ((batch * uint(W) + window_index) * uint(RATIO)
                    + ratio_index) * OUT_DIM + uint(HEAD_DIM) + dimension
            );
            value = float(kv[index]);
            score = float(gate[index])
                + ape[ratio_index * OUT_DIM + uint(HEAD_DIM) + dimension];
        }
        if (isfinite(score)) {
            float candidate_weight = exp(score - maximum);
            weighted += candidate_weight * value;
            denominator += candidate_weight;
        }
    }
    float value = weighted / denominator;
    compressed[dimension] = value;

    float square_sum = simd_sum(value * value);
    if (lane == 0u) {
        warp_sums[simd_group] = square_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dimension == 0u) {
        square_sum = 0.0f;
        for (uint group = 0u; group < uint(HEAD_DIM / 32); ++group) {
            square_sum += warp_sums[group];
        }
        inverse_rms = rsqrt(
            square_sum / float(HEAD_DIM) + params[0]
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    compressed[dimension] = mfq_dsv4_bf16_round(
        value * inverse_rms * norm[dimension]
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint NOPE_DIM = uint(HEAD_DIM) - 64u;
    if (dimension >= NOPE_DIM && ((dimension - NOPE_DIM) & 1u) == 0u) {
        uint pair = (dimension - NOPE_DIM) / 2u;
        int position = positions[row];
        position = max(0, min(position, TABLE_LEN - 1));
        float cosine = cos_table[
            uint(position) * uint(TABLE_STRIDE) + pair
        ];
        float sine = sin_table[
            uint(position) * uint(TABLE_STRIDE) + pair
        ];
        float first = compressed[dimension];
        float second = compressed[dimension + 1u];
        compressed[dimension] = mfq_dsv4_bf16_round(
            first * cosine - second * sine
        );
        compressed[dimension + 1u] = mfq_dsv4_bf16_round(
            second * cosine + first * sine
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (QUANT_MODE == 2) {
        for (
            uint half_width = 1u;
            half_width < uint(HEAD_DIM);
            half_width <<= 1u
        ) {
            if (dimension < uint(HEAD_DIM / 2)) {
                uint group = dimension / half_width;
                uint within = dimension - group * half_width;
                uint first = group * (2u * half_width) + within;
                uint second = first + half_width;
                float first_value = compressed[first];
                float second_value = compressed[second];
                compressed[first] = first_value + second_value;
                compressed[second] = first_value - second_value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        compressed[dimension] = mfq_dsv4_bf16_round(
            compressed[dimension] * rsqrt(float(HEAD_DIM))
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((dimension & 31u) == 0u) {
            float absolute_maximum = 6.0f * 0x1p-126f;
            for (uint item = 0u; item < 32u; ++item) {
                absolute_maximum = max(
                    absolute_maximum,
                    abs(compressed[dimension + item])
                );
            }
            quant_scales[dimension / 32u] = mfq_dsv4_pow2_ceil(
                absolute_maximum / 6.0f
            );
        }
    }
    if (
        QUANT_MODE == 1
        && dimension < NOPE_DIM
        && (dimension & 63u) == 0u
    ) {
        float absolute_maximum = 1e-4f;
        for (uint item = 0u; item < 64u; ++item) {
            absolute_maximum = max(
                absolute_maximum,
                abs(compressed[dimension + item])
            );
        }
        quant_scales[dimension / 64u] = absolute_maximum / 448.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float result = compressed[dimension];
    if (QUANT_MODE == 1 && dimension < NOPE_DIM) {
        result = mfq_dsv4_fp8_e4m3(
            result,
            quant_scales[dimension / 64u]
        );
    } else if (QUANT_MODE == 2) {
        result = mfq_dsv4_fp4(
            result,
            quant_scales[dimension / 32u]
        );
    }
    out[row * uint(HEAD_DIM) + dimension] = half(
        mfq_dsv4_bf16_round(result)
    );
"""


_DECODE_POOL_UPDATE_SOURCE = r"""
    uint batch = threadgroup_position_in_grid.x;
    uint dimension = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    if (batch >= uint(B) || dimension >= uint(HEAD_DIM)) {
        return;
    }
    constexpr uint OUT_DIM = OVERLAP != 0 ? 2u * uint(HEAD_DIM) : uint(HEAD_DIM);
    int length = seq_len[batch];
    bool has_token = length > 0;
    uint slot = has_token
        ? uint(length - 1) % uint(RATIO)
        : 0u;

    for (
        uint index = dimension;
        index < uint(RATIO) * OUT_DIM;
        index += uint(HEAD_DIM)
    ) {
        uint state_slot = index / OUT_DIM;
        uint feature = index - state_slot * OUT_DIM;
        uint state_index = batch * uint(RATIO) * OUT_DIM + index;
        bool update = has_token && state_slot == slot;
        state_kv_out[state_index] = update
            ? kv_token[batch * OUT_DIM + feature]
            : state_kv[state_index];
        state_gate_out[state_index] = update
            ? gate_token[batch * OUT_DIM + feature]
            : state_gate[state_index];
    }
    for (
        uint index = dimension;
        index < uint(RATIO * HEAD_DIM);
        index += uint(HEAD_DIM)
    ) {
        uint previous_index = batch * uint(RATIO * HEAD_DIM) + index;
        if (OVERLAP != 0) {
            prev_kv_out[previous_index] = prev_kv[previous_index];
            prev_gate_out[previous_index] = prev_gate[previous_index];
        } else {
            prev_kv_out[previous_index] = T(0.0f);
            prev_gate_out[previous_index] = T(0.0f);
        }
    }
    for (
        uint index = dimension;
        index < uint(POOL_CAPACITY * HEAD_DIM);
        index += uint(HEAD_DIM)
    ) {
        uint pool_index = batch * uint(POOL_CAPACITY * HEAD_DIM) + index;
        pool_out[pool_index] = pool[pool_index];
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (!has_token || slot != uint(RATIO - 1)) {
        return;
    }
    uint output_row = uint(length - 1) / uint(RATIO);
    if (output_row >= uint(POOL_CAPACITY)) {
        return;
    }

    threadgroup float compressed[HEAD_DIM];
    threadgroup float quant_scales[HEAD_DIM / 32];
    threadgroup float warp_sums[HEAD_DIM / 32];
    threadgroup float inverse_rms;
    constexpr uint CANDIDATES = OVERLAP != 0 ? 2u * uint(RATIO) : uint(RATIO);
    bool have_previous = output_row > 0u;
    float maximum = -INFINITY;
    for (uint candidate = 0u; candidate < CANDIDATES; ++candidate) {
        float score = -INFINITY;
        if (OVERLAP == 0) {
            uint index = (
                (batch * uint(RATIO) + candidate) * OUT_DIM + dimension
            );
            score = float(state_gate_out[index])
                + ape[candidate * OUT_DIM + dimension];
        } else if (candidate < uint(RATIO)) {
            if (have_previous) {
                uint index = (
                    (batch * uint(RATIO) + candidate) * uint(HEAD_DIM)
                        + dimension
                );
                score = float(prev_gate_out[index])
                    + ape[candidate * OUT_DIM + dimension];
            }
        } else {
            uint ratio_index = candidate - uint(RATIO);
            uint index = (
                (batch * uint(RATIO) + ratio_index) * OUT_DIM
                    + uint(HEAD_DIM) + dimension
            );
            score = float(state_gate_out[index])
                + ape[ratio_index * OUT_DIM + uint(HEAD_DIM) + dimension];
        }
        maximum = max(maximum, score);
    }

    float weighted = 0.0f;
    float denominator = 0.0f;
    for (uint candidate = 0u; candidate < CANDIDATES; ++candidate) {
        float value = 0.0f;
        float score = -INFINITY;
        if (OVERLAP == 0) {
            uint index = (
                (batch * uint(RATIO) + candidate) * OUT_DIM + dimension
            );
            value = float(state_kv_out[index]);
            score = float(state_gate_out[index])
                + ape[candidate * OUT_DIM + dimension];
        } else if (candidate < uint(RATIO)) {
            if (have_previous) {
                uint index = (
                    (batch * uint(RATIO) + candidate) * uint(HEAD_DIM)
                        + dimension
                );
                value = float(prev_kv_out[index]);
                score = float(prev_gate_out[index])
                    + ape[candidate * OUT_DIM + dimension];
            }
        } else {
            uint ratio_index = candidate - uint(RATIO);
            uint index = (
                (batch * uint(RATIO) + ratio_index) * OUT_DIM
                    + uint(HEAD_DIM) + dimension
            );
            value = float(state_kv_out[index]);
            score = float(state_gate_out[index])
                + ape[ratio_index * OUT_DIM + uint(HEAD_DIM) + dimension];
        }
        if (isfinite(score)) {
            float candidate_weight = exp(score - maximum);
            weighted += candidate_weight * value;
            denominator += candidate_weight;
        }
    }
    float value = weighted / denominator;
    compressed[dimension] = value;
    float square_sum = simd_sum(value * value);
    if (lane == 0u) {
        warp_sums[simd_group] = square_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dimension == 0u) {
        square_sum = 0.0f;
        for (uint group = 0u; group < uint(HEAD_DIM / 32); ++group) {
            square_sum += warp_sums[group];
        }
        inverse_rms = rsqrt(
            square_sum / float(HEAD_DIM) + params[0]
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    compressed[dimension] = mfq_dsv4_bf16_round(
        value * inverse_rms * norm[dimension]
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint NOPE_DIM = uint(HEAD_DIM) - 64u;
    if (dimension >= NOPE_DIM && ((dimension - NOPE_DIM) & 1u) == 0u) {
        uint pair = (dimension - NOPE_DIM) / 2u;
        uint table_row = min(output_row, uint(TABLE_LEN - 1));
        float cosine = cos_table[
            table_row * uint(TABLE_STRIDE) + pair
        ];
        float sine = sin_table[
            table_row * uint(TABLE_STRIDE) + pair
        ];
        float first = compressed[dimension];
        float second = compressed[dimension + 1u];
        compressed[dimension] = mfq_dsv4_bf16_round(
            first * cosine - second * sine
        );
        compressed[dimension + 1u] = mfq_dsv4_bf16_round(
            second * cosine + first * sine
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (QUANT_MODE == 2) {
        for (
            uint half_width = 1u;
            half_width < uint(HEAD_DIM);
            half_width <<= 1u
        ) {
            if (dimension < uint(HEAD_DIM / 2)) {
                uint group = dimension / half_width;
                uint within = dimension - group * half_width;
                uint first = group * (2u * half_width) + within;
                uint second = first + half_width;
                float first_value = compressed[first];
                float second_value = compressed[second];
                compressed[first] = first_value + second_value;
                compressed[second] = first_value - second_value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        compressed[dimension] = mfq_dsv4_bf16_round(
            compressed[dimension] * rsqrt(float(HEAD_DIM))
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((dimension & 31u) == 0u) {
            float absolute_maximum = 6.0f * 0x1p-126f;
            for (uint item = 0u; item < 32u; ++item) {
                absolute_maximum = max(
                    absolute_maximum,
                    abs(compressed[dimension + item])
                );
            }
            quant_scales[dimension / 32u] = mfq_dsv4_pow2_ceil(
                absolute_maximum / 6.0f
            );
        }
    }
    if (
        QUANT_MODE == 1
        && dimension < NOPE_DIM
        && (dimension & 63u) == 0u
    ) {
        float absolute_maximum = 1e-4f;
        for (uint item = 0u; item < 64u; ++item) {
            absolute_maximum = max(
                absolute_maximum,
                abs(compressed[dimension + item])
            );
        }
        quant_scales[dimension / 64u] = absolute_maximum / 448.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = compressed[dimension];
    if (QUANT_MODE == 1 && dimension < NOPE_DIM) {
        result = mfq_dsv4_fp8_e4m3(
            result,
            quant_scales[dimension / 64u]
        );
    } else if (QUANT_MODE == 2) {
        result = mfq_dsv4_fp4(
            result,
            quant_scales[dimension / 32u]
        );
    }
    uint pool_index = (
        (batch * uint(POOL_CAPACITY) + output_row) * uint(HEAD_DIM)
            + dimension
    );
    pool_out[pool_index] = half(mfq_dsv4_bf16_round(result));
    threadgroup_barrier(mem_flags::mem_device);
    if (OVERLAP != 0) {
        for (uint ratio_index = 0u; ratio_index < uint(RATIO); ++ratio_index) {
            uint state_index = (
                (batch * uint(RATIO) + ratio_index) * OUT_DIM + dimension
            );
            uint previous_index = (
                (batch * uint(RATIO) + ratio_index) * uint(HEAD_DIM)
                    + dimension
            );
            prev_kv_out[previous_index] = state_kv_out[state_index];
            prev_gate_out[previous_index] = state_gate_out[state_index];
        }
    }
"""


# The legacy update kernel above mirrors CUDA's capacity-backed mutation API,
# but an mx.fast.metal_kernel output can never alias an input.  Returning
# ``pool_out`` therefore copies the complete long-context pool every token.
# Build the graph-safe step variant from the same compressor body while
# replacing the unbounded pool output with one fixed-size emitted row.
_DECODE_POOL_STEP_SOURCE = (
    _DECODE_POOL_UPDATE_SOURCE.replace(
        r"""    for (
        uint index = dimension;
        index < uint(POOL_CAPACITY * HEAD_DIM);
        index += uint(HEAD_DIM)
    ) {
        uint pool_index = batch * uint(POOL_CAPACITY * HEAD_DIM) + index;
        pool_out[pool_index] = pool[pool_index];
    }
    threadgroup_barrier(mem_flags::mem_device);
""",
        r"""    emitted[batch * uint(HEAD_DIM) + dimension] = half(0.0f);
    if (dimension == 0u) {
        emit_rows[batch] = -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
""",
    )
    .replace(
        r"""    uint output_row = uint(length - 1) / uint(RATIO);
    if (output_row >= uint(POOL_CAPACITY)) {
        return;
    }
""",
        r"""    uint output_row = uint(length - 1) / uint(RATIO);
    if (dimension == 0u) {
        emit_rows[batch] = int(output_row);
    }
""",
    )
    .replace(
        r"""    uint pool_index = (
        (batch * uint(POOL_CAPACITY) + output_row) * uint(HEAD_DIM)
            + dimension
    );
    pool_out[pool_index] = half(mfq_dsv4_bf16_round(result));
    threadgroup_barrier(mem_flags::mem_device);
""",
        r"""    emitted[batch * uint(HEAD_DIM) + dimension] =
        half(mfq_dsv4_bf16_round(result));
    threadgroup_barrier(mem_flags::mem_threadgroup);
""",
    )
)


_FP4_SIM_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint group = threadgroup_position_in_grid.x;
    uint offset = group * 32u + lane;
    float value = float(x[offset]);
    float maximum = simd_max(abs(value));
    float scale = mfq_dsv4_pow2_ceil(
        max(maximum, 6.0f * 0x1p-126f) / 6.0f
    );
    out[offset] = half(mfq_dsv4_fp4(value, scale));
"""


_INDEXER_SCORES_SOURCE = r"""
    constexpr uint HEADS = 64u;
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
    threadgroup float score_tile[32u * KEY_TILE];
    threadgroup float weighted_sum[KEY_TILE];
    for (
        uint index = local_thread;
        index < DIM * KEY_TILE;
        index += 256u
    ) {
        uint dimension = index / KEY_TILE;
        uint local_key = index - dimension * KEY_TILE;
        uint key_index = key_base + local_key;
        key_tile[index] = key_index < uint(K)
            ? k[(batch * uint(K) + key_index) * DIM + dimension]
            : half(0.0f);
    }
    if (local_thread < KEY_TILE) {
        weighted_sum[local_thread] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint quadrant = lane / 4u;
    uint fragment_row = (quadrant & 4u) + ((lane / 2u) & 3u);
    uint fragment_col = (quadrant & 2u) * 2u + (lane & 1u) * 2u;
    uint local_key_base = (simd_group & 3u) * 16u;
    for (uint head_pass = 0u; head_pass < 2u; ++head_pass) {
        uint head_base = head_pass * 32u + (simd_group / 4u) * 16u;
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

        uint local_head0 = (simd_group / 4u) * 16u + fragment_row;
        uint local_head1 = local_head0 + 8u;
        uint key0 = local_key_base + fragment_col;
        uint key1 = key0 + 8u;
        score_tile[local_head0 * KEY_TILE + key0] =
            c00.thread_elements()[0];
        score_tile[local_head0 * KEY_TILE + key0 + 1u] =
            c00.thread_elements()[1];
        score_tile[local_head0 * KEY_TILE + key1] =
            c01.thread_elements()[0];
        score_tile[local_head0 * KEY_TILE + key1 + 1u] =
            c01.thread_elements()[1];
        score_tile[local_head1 * KEY_TILE + key0] =
            c10.thread_elements()[0];
        score_tile[local_head1 * KEY_TILE + key0 + 1u] =
            c10.thread_elements()[1];
        score_tile[local_head1 * KEY_TILE + key1] =
            c11.thread_elements()[0];
        score_tile[local_head1 * KEY_TILE + key1 + 1u] =
            c11.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (local_thread < KEY_TILE) {
            float sum = 0.0f;
            uint weight_base = (batch * uint(M) + query) * HEADS;
            for (uint local_head = 0u; local_head < 32u; ++local_head) {
                float dot = score_tile[local_head * KEY_TILE + local_thread];
                sum += max(dot, 0.0f)
                    * float(weights[
                        weight_base + head_pass * 32u + local_head
                    ]);
            }
            weighted_sum[local_thread] += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_thread < KEY_TILE) {
        uint key_index = key_base + local_thread;
        if (key_index < uint(K)) {
            uint visible = min(
                uint(K),
                (uint(QUERY_OFFSET) + query + 1u) / uint(RATIO)
            );
            out[(batch * uint(M) + query) * uint(K) + key_index] =
                key_index < visible
                ? half(weighted_sum[local_thread] * params[0])
                : half(-INFINITY);
        }
    }
"""


_INDEXER_DECODE_SCORES_SOURCE = r"""
    constexpr uint HEADS = 64u;
    constexpr uint HEAD_BLOCK = 32u;
    constexpr uint DIM = 128u;
    constexpr uint DIM4 = DIM / 4u;
    constexpr uint THREAD_COUNT = uint(THREADS);

    uint local_thread = thread_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint key_tile = workgroup % uint(KEY_TILES);
    uint batch = workgroup / uint(KEY_TILES);
    if (batch >= uint(B)) {
        return;
    }

    threadgroup half query_shared[HEADS * DIM];
    threadgroup half weight_shared[HEADS];
    uint query_base = batch * HEADS * DIM;
    uint weight_base = batch * HEADS;
    for (
        uint index = local_thread;
        index < HEADS * DIM;
        index += THREAD_COUNT
    ) {
        query_shared[index] = q[query_base + index];
    }
    if (local_thread < HEADS) {
        weight_shared[local_thread] = weights[weight_base + local_thread];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint key_index = key_tile * THREAD_COUNT + local_thread;
    if (key_index >= uint(K)) {
        return;
    }
    const device half4* key4 = (const device half4*)(
        k + (batch * uint(K) + key_index) * DIM
    );
    const threadgroup half4* query4 =
        (const threadgroup half4*)query_shared;

    float total = 0.0f;
    for (uint head_pass = 0u; head_pass < 2u; ++head_pass) {
        float accumulators[HEAD_BLOCK];
        for (uint local_head = 0u; local_head < HEAD_BLOCK; ++local_head) {
            accumulators[local_head] = 0.0f;
        }
        for (uint chunk = 0u; chunk < DIM4; ++chunk) {
            float4 key_value = float4(key4[chunk]);
            for (uint local_head = 0u; local_head < HEAD_BLOCK; ++local_head) {
                uint head = head_pass * HEAD_BLOCK + local_head;
                accumulators[local_head] += metal::dot(
                    float4(query4[head * DIM4 + chunk]),
                    key_value
                );
            }
        }
        for (uint local_head = 0u; local_head < HEAD_BLOCK; ++local_head) {
            uint head = head_pass * HEAD_BLOCK + local_head;
            total += max(accumulators[local_head], 0.0f)
                * float(weight_shared[head]);
        }
    }

    uint visible = min(
        uint(K),
        (uint(QUERY_OFFSET) + 1u) / uint(RATIO)
    );
    out[batch * uint(K) + key_index] = key_index < visible
        ? half(total * params[0])
        : half(-INFINITY);
"""


_TOPK_HEADER = r"""
    METAL_FUNC uint mfq_dsv4_ordered_half(half value) {
        uint bits = uint(as_type<ushort>(value));
        return (bits & 0x8000u) != 0u
            ? ((~bits) & 0xffffu)
            : (bits | 0x8000u);
    }
"""


_TOPK_SOURCE = r"""
    constexpr uint TOPK = 512u;
    constexpr uint THREADS = 1024u;
    constexpr uint SIMD_GROUPS = THREADS / 32u;
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    uint input_base = row * uint(K);
    uint output_base = row * TOPK;
    if (uint(K) <= TOPK) {
        for (uint index = tid; index < TOPK; index += THREADS) {
            out[output_base + index] = index < uint(K) ? int(index) : 0;
        }
        return;
    }

    threadgroup atomic_uint histogram[256];
    threadgroup atomic_uint counters[2];
    threadgroup uint state[4];
    if (tid < 256u) {
        atomic_store_explicit(
            &histogram[tid], 0u, memory_order_relaxed
        );
    }
    if (tid < 2u) {
        atomic_store_explicit(
            &counters[tid], 0u, memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < uint(K); index += THREADS) {
        uint key = mfq_dsv4_ordered_half(x[input_base + index]);
        atomic_fetch_add_explicit(
            &histogram[key >> 8u], 1u, memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint greater = 0u;
        uint threshold_high = 0u;
        for (int high = 255; high >= 0; --high) {
            uint count = atomic_load_explicit(
                &histogram[uint(high)], memory_order_relaxed
            );
            if (greater + count >= TOPK) {
                threshold_high = uint(high);
                break;
            }
            greater += count;
        }
        state[0] = threshold_high;
        state[1] = greater;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 256u) {
        atomic_store_explicit(
            &histogram[tid], 0u, memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint threshold_high = state[0];
    for (uint index = tid; index < uint(K); index += THREADS) {
        uint key = mfq_dsv4_ordered_half(x[input_base + index]);
        if ((key >> 8u) == threshold_high) {
            atomic_fetch_add_explicit(
                &histogram[key & 0xffu], 1u, memory_order_relaxed
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint greater = state[1];
        uint threshold_low = 0u;
        for (int low = 255; low >= 0; --low) {
            uint count = atomic_load_explicit(
                &histogram[uint(low)], memory_order_relaxed
            );
            if (greater + count >= TOPK) {
                threshold_low = uint(low);
                break;
            }
            greater += count;
        }
        state[2] = (threshold_high << 8u) | threshold_low;
        state[3] = greater;
        atomic_store_explicit(
            &counters[0], 0u, memory_order_relaxed
        );
        atomic_store_explicit(
            &counters[1], greater, memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint threshold = state[2];
    if (DETERMINISTIC == 0) {
        for (uint base = 0u; base < uint(K); base += THREADS) {
            uint index = base + tid;
            if (index < uint(K)) {
                uint key = mfq_dsv4_ordered_half(x[input_base + index]);
                if (key > threshold) {
                    uint position = atomic_fetch_add_explicit(
                        &counters[0], 1u, memory_order_relaxed
                    );
                    if (position < TOPK) {
                        out[output_base + position] = int(index);
                    }
                } else if (key == threshold) {
                    uint position = atomic_fetch_add_explicit(
                        &counters[1], 1u, memory_order_relaxed
                    );
                    if (position < TOPK) {
                        out[output_base + position] = int(index);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }

    threadgroup uint greater_partials[SIMD_GROUPS];
    threadgroup uint tie_partials[SIMD_GROUPS];
    uint simd_group = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint segment = (uint(K) + THREADS - 1u) / THREADS;
    uint segment_begin = tid * segment;
    uint segment_end = min(segment_begin + segment, uint(K));
    uint local_greater = 0u;
    uint local_ties = 0u;
    for (uint index = segment_begin; index < segment_end; ++index) {
        uint key = mfq_dsv4_ordered_half(x[input_base + index]);
        local_greater += key > threshold ? 1u : 0u;
        local_ties += key == threshold ? 1u : 0u;
    }
    uint prefix_greater =
        metal::simd_prefix_exclusive_sum(local_greater);
    uint prefix_ties =
        metal::simd_prefix_exclusive_sum(local_ties);
    if (lane == 31u) {
        greater_partials[simd_group] =
            prefix_greater + local_greater;
        tie_partials[simd_group] = prefix_ties + local_ties;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        uint group_greater =
            lane < SIMD_GROUPS ? greater_partials[lane] : 0u;
        uint group_ties =
            lane < SIMD_GROUPS ? tie_partials[lane] : 0u;
        uint group_greater_prefix =
            metal::simd_prefix_exclusive_sum(group_greater);
        uint group_tie_prefix =
            metal::simd_prefix_exclusive_sum(group_ties);
        if (lane < SIMD_GROUPS) {
            greater_partials[lane] = group_greater_prefix;
            tie_partials[lane] = group_tie_prefix;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint greater_position =
        greater_partials[simd_group] + prefix_greater;
    uint tie_position =
        state[3] + tie_partials[simd_group] + prefix_ties;
    for (uint index = segment_begin; index < segment_end; ++index) {
        uint key = mfq_dsv4_ordered_half(x[input_base + index]);
        if (key > threshold) {
            out[output_base + greater_position++] = int(index);
        } else if (key == threshold) {
            if (tie_position < TOPK) {
                out[output_base + tie_position] = int(index);
            }
            tie_position++;
        }
    }
"""


_PREFILL_PLAN_SOURCE = r"""
    uint linear = thread_position_in_grid.x;
    if (linear >= uint(TOTAL)) {
        return;
    }
    uint slot = linear % uint(SELECTED);
    uint row = linear / uint(SELECTED);
    uint query = row % uint(M);
    uint raw_length = uint(LOCAL_HISTORY + M);
    int index = 0;
    bool valid = false;
    if (slot < uint(WINDOW)) {
        uint local_end = uint(LOCAL_HISTORY) + query + 1u;
        uint local_count = min(uint(WINDOW), local_end);
        if (slot < local_count) {
            index = int(local_end - local_count + slot);
            valid = true;
        }
    } else if (slot < uint(WINDOW + TOPK_COUNT)) {
        int pooled = topk[row * uint(TOPK_COUNT) + slot - uint(WINDOW)];
        uint visible = min(
            uint(POOL_LEN),
            (uint(QUERY_OFFSET) + query + 1u) / uint(RATIO)
        );
        if (pooled >= 0 && uint(pooled) < visible) {
            index = int(raw_length) + pooled;
            valid = true;
        }
    }
    indices[linear] = index;
    mask[linear] = valid ? half(0.0f) : half(-INFINITY);
"""


_DECODE_PLAN_SOURCE = r"""
    uint linear = thread_position_in_grid.x;
    if (linear >= uint(TOTAL)) {
        return;
    }
    uint slot = linear % uint(SELECTED);
    uint batch = linear / uint(SELECTED);
    int length = max(seq_len[batch], 0);
    int index = 0;
    bool valid = false;
    if (slot < uint(WINDOW)) {
        int local_count = min(length, int(WINDOW));
        if (int(slot) < local_count) {
            int absolute = length - local_count + int(slot);
            index = absolute % int(WINDOW);
            valid = true;
        }
    } else if (slot < uint(WINDOW + TOPK_COUNT)) {
        int pooled = topk[
            batch * uint(TOPK_COUNT) + slot - uint(WINDOW)
        ];
        int visible = min(length / int(RATIO), int(POOL_LEN));
        if (pooled >= 0 && pooled < visible) {
            index = int(WINDOW) + pooled;
            valid = true;
        }
    }
    indices[linear] = index;
    mask[linear] = valid ? half(0.0f) : half(-INFINITY);
"""


_SPARSE_ATTENTION_SOURCE = r"""
    constexpr uint HEADS = 64u;
    constexpr uint DIM = 512u;
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
        state[0] = sinks[head];
        state[1] = 1.0f;
        state[2] = 0.0f;
        state[3] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;
    uint query_base = ((batch * HEADS + head) * uint(M) + query) * DIM;
    uint selected_base = (batch * uint(M) + query) * uint(SELECTED);
    for (uint selected = 0u; selected < uint(SELECTED); ++selected) {
        float mask_value = float(mask[selected_base + selected]);
        int cache_row = indices[selected_base + selected];
        bool valid = isfinite(mask_value)
            && cache_row >= 0
            && cache_row < int(MAX_SEQ);
        uint safe_row = valid ? uint(cache_row) : 0u;
        uint cache_base = (batch * uint(MAX_SEQ) + safe_row) * DIM;
        float dot = 0.0f;
        if (valid) {
            dot = q[query_base + local_thread]
                * float(kv[cache_base + local_thread]);
            dot += q[query_base + local_thread + 256u]
                * float(kv[cache_base + local_thread + 256u]);
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
            score = valid ? score * params[0] + mask_value : -INFINITY;
            float old_max = state[0];
            float new_max = max(old_max, score);
            float old_scale = exp(old_max - new_max);
            float new_scale = valid ? exp(score - new_max) : 0.0f;
            state[0] = new_max;
            state[1] = state[1] * old_scale + new_scale;
            state[2] = old_scale;
            state[3] = new_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float old_scale = state[2];
        float new_scale = state[3];
        float value0 = valid
            ? float(kv[cache_base + local_thread])
            : 0.0f;
        float value1 = valid
            ? float(kv[cache_base + local_thread + 256u])
            : 0.0f;
        accumulator0 = accumulator0 * old_scale + value0 * new_scale;
        accumulator1 = accumulator1 * old_scale + value1 * new_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inverse = 1.0f / state[1];
    uint output_base = ((batch * uint(M) + query) * HEADS + head) * DIM;
    out[output_base + local_thread] = accumulator0 * inverse;
    out[output_base + local_thread + 256u] = accumulator1 * inverse;
"""


_SPARSE_ATTENTION_PREFILL_MMA_SOURCE = r"""
    constexpr uint HEADS = 64u;
    constexpr uint DIM = 512u;
    constexpr uint KEY_TILE = 8u;
    constexpr uint DIM_TILES = DIM / 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint local_thread = thread_index_in_threadgroup;
    uint query_row = threadgroup_position_in_grid.x;
    uint query = query_row % uint(M);
    uint batch = query_row / uint(M);
    if (batch >= uint(B)) {
        return;
    }

    uint quadrant = lane / 4u;
    uint fragment_row = (quadrant & 4u) + ((lane / 2u) & 3u);
    uint fragment_col = (quadrant & 2u) * 2u + (lane & 1u) * 2u;
    uint head_base = simd_group * 8u;
    uint fragment_head = head_base + fragment_row;

    threadgroup half key_tile[KEY_TILE * DIM];
    threadgroup int selected_rows[KEY_TILE];
    threadgroup float selected_masks[KEY_TILE];
    threadgroup float score_tile[HEADS * KEY_TILE];
    threadgroup half probability_tile[HEADS * KEY_TILE];
    threadgroup float head_maximum[HEADS];
    threadgroup float head_sum[HEADS];
    threadgroup float head_rescale[HEADS];

    if (local_thread < HEADS) {
        head_maximum[local_thread] = sinks[local_thread];
        head_sum[local_thread] = 1.0f;
        head_rescale[local_thread] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    metal::simdgroup_matrix<float, 8, 8> output_tiles[DIM_TILES];
    for (uint dimension_tile = 0u; dimension_tile < DIM_TILES; ++dimension_tile) {
        output_tiles[dimension_tile].thread_elements()[0] = 0.0f;
        output_tiles[dimension_tile].thread_elements()[1] = 0.0f;
    }

    uint query_base =
        ((batch * HEADS + head_base) * uint(M) + query) * DIM;
    uint selected_base = (batch * uint(M) + query) * uint(SELECTED);
    for (
        uint selected_tile = 0u;
        selected_tile < uint(SELECTED);
        selected_tile += KEY_TILE
    ) {
        if (local_thread < KEY_TILE) {
            uint selected = selected_tile + local_thread;
            float mask_value = selected < uint(SELECTED)
                ? float(mask[selected_base + selected])
                : -INFINITY;
            int cache_row = selected < uint(SELECTED)
                ? indices[selected_base + selected]
                : -1;
            bool valid = isfinite(mask_value)
                && cache_row >= 0
                && cache_row < int(MAX_SEQ);
            selected_rows[local_thread] = valid ? cache_row : 0;
            selected_masks[local_thread] = valid ? mask_value : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (
            uint index = local_thread;
            index < KEY_TILE * DIM;
            index += 256u
        ) {
            uint local_key = index / DIM;
            uint dimension = index - local_key * DIM;
            int cache_row = selected_rows[local_key];
            key_tile[index] = isfinite(selected_masks[local_key])
                ? kv[(batch * uint(MAX_SEQ) + uint(cache_row)) * DIM + dimension]
                : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        metal::simdgroup_matrix<float, 8, 8> scores;
        scores.thread_elements()[0] = 0.0f;
        scores.thread_elements()[1] = 0.0f;
        for (uint dimension = 0u; dimension < DIM; dimension += 8u) {
            metal::simdgroup_matrix<half, 8, 8> query_matrix;
            metal::simdgroup_matrix<half, 8, 8> key_matrix;
            query_matrix.thread_elements()[0] = q[
                query_base + fragment_row * uint(M) * DIM
                    + dimension + fragment_col
            ];
            query_matrix.thread_elements()[1] = q[
                query_base + fragment_row * uint(M) * DIM
                    + dimension + fragment_col + 1u
            ];
            key_matrix.thread_elements()[0] = key_tile[
                fragment_col * DIM + dimension + fragment_row
            ];
            key_matrix.thread_elements()[1] = key_tile[
                (fragment_col + 1u) * DIM + dimension + fragment_row
            ];
            simdgroup_multiply_accumulate(
                scores,
                query_matrix,
                key_matrix,
                scores
            );
        }
        score_tile[fragment_head * KEY_TILE + fragment_col] =
            scores.thread_elements()[0] * params[0]
                + selected_masks[fragment_col];
        score_tile[fragment_head * KEY_TILE + fragment_col + 1u] =
            scores.thread_elements()[1] * params[0]
                + selected_masks[fragment_col + 1u];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < 8u) {
            uint head = head_base + lane;
            float tile_maximum = -INFINITY;
            for (uint local_key = 0u; local_key < KEY_TILE; ++local_key) {
                tile_maximum = max(
                    tile_maximum,
                    score_tile[head * KEY_TILE + local_key]
                );
            }
            float old_maximum = head_maximum[head];
            float new_maximum = max(old_maximum, tile_maximum);
            float rescale = metal::fast::exp(old_maximum - new_maximum);
            float tile_sum = 0.0f;
            for (uint local_key = 0u; local_key < KEY_TILE; ++local_key) {
                float score = score_tile[head * KEY_TILE + local_key];
                float probability = isfinite(score)
                    ? metal::fast::exp(score - new_maximum)
                    : 0.0f;
                probability_tile[head * KEY_TILE + local_key] =
                    half(probability);
                tile_sum += probability;
            }
            head_maximum[head] = new_maximum;
            head_sum[head] = head_sum[head] * rescale + tile_sum;
            head_rescale[head] = rescale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rescale = head_rescale[fragment_head];
        for (uint dimension_tile = 0u; dimension_tile < DIM_TILES; ++dimension_tile) {
            output_tiles[dimension_tile].thread_elements()[0] *= rescale;
            output_tiles[dimension_tile].thread_elements()[1] *= rescale;

            metal::simdgroup_matrix<half, 8, 8> probability_matrix;
            metal::simdgroup_matrix<half, 8, 8> value_matrix;
            probability_matrix.thread_elements()[0] = probability_tile[
                fragment_head * KEY_TILE + fragment_col
            ];
            probability_matrix.thread_elements()[1] = probability_tile[
                fragment_head * KEY_TILE + fragment_col + 1u
            ];
            uint dimension = dimension_tile * 8u;
            value_matrix.thread_elements()[0] = key_tile[
                fragment_row * DIM + dimension + fragment_col
            ];
            value_matrix.thread_elements()[1] = key_tile[
                fragment_row * DIM + dimension + fragment_col + 1u
            ];
            simdgroup_multiply_accumulate(
                output_tiles[dimension_tile],
                probability_matrix,
                value_matrix,
                output_tiles[dimension_tile]
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inverse_sum = 1.0f / head_sum[fragment_head];
    uint output_base =
        ((batch * uint(M) + query) * HEADS + fragment_head) * DIM;
    for (uint dimension_tile = 0u; dimension_tile < DIM_TILES; ++dimension_tile) {
        uint dimension = dimension_tile * 8u + fragment_col;
        out[output_base + dimension] =
            output_tiles[dimension_tile].thread_elements()[0] * inverse_sum;
        out[output_base + dimension + 1u] =
            output_tiles[dimension_tile].thread_elements()[1] * inverse_sum;
    }
"""


_SPARSE_ATTENTION_DECODE_SOURCE = r"""
    constexpr uint HEADS = 64u;
    constexpr uint HEADS_PER_GROUP = 4u;
    constexpr uint HEAD_GROUPS = HEADS / HEADS_PER_GROUP;
    constexpr uint DIM = 512u;
    constexpr uint VALUES_PER_LANE = DIM / 32u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint head_group = workgroup % HEAD_GROUPS;
    uint query_row = workgroup / HEAD_GROUPS;
    uint query = query_row % uint(M);
    uint batch = query_row / uint(M);
    if (batch >= uint(B)) {
        return;
    }
    uint head = head_group * HEADS_PER_GROUP + simd_group;

    uint query_base = ((batch * HEADS + head) * uint(M) + query) * DIM;
    uint selected_base = (batch * uint(M) + query) * uint(SELECTED);
    float maximum = sinks[head];
    float denominator = 1.0f;
    float accumulators[VALUES_PER_LANE];
    for (uint item = 0u; item < VALUES_PER_LANE; ++item) {
        accumulators[item] = 0.0f;
    }

    for (uint selected = 0u; selected < uint(SELECTED); ++selected) {
        float mask_value = float(mask[selected_base + selected]);
        int cache_row = indices[selected_base + selected];
        bool valid = isfinite(mask_value)
            && cache_row >= 0
            && cache_row < int(MAX_SEQ);
        uint safe_row = valid ? uint(cache_row) : 0u;
        uint cache_base = (batch * uint(MAX_SEQ) + safe_row) * DIM;

        float dot = 0.0f;
        for (uint item = 0u; item < VALUES_PER_LANE; ++item) {
            uint dimension = lane + item * 32u;
            dot += q[query_base + dimension]
                * float(kv[cache_base + dimension]);
        }
        float score = simd_sum(dot);
        score = valid ? score * params[0] + mask_value : -INFINITY;
        float new_maximum = max(maximum, score);
        float old_scale = metal::fast::exp(maximum - new_maximum);
        float new_scale = valid
            ? metal::fast::exp(score - new_maximum)
            : 0.0f;
        denominator = denominator * old_scale + new_scale;
        maximum = new_maximum;

        for (uint item = 0u; item < VALUES_PER_LANE; ++item) {
            uint dimension = lane + item * 32u;
            float value = valid
                ? float(kv[cache_base + dimension])
                : 0.0f;
            accumulators[item] =
                accumulators[item] * old_scale + value * new_scale;
        }
    }

    float inverse_denominator = 1.0f / denominator;
    uint output_base =
        ((batch * uint(M) + query) * HEADS + head) * DIM;
    for (uint item = 0u; item < VALUES_PER_LANE; ++item) {
        uint dimension = lane + item * 32u;
        out[output_base + dimension] =
            accumulators[item] * inverse_denominator;
    }
"""


_FP4_SIM_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_fp4_sim",
    input_names=["x"],
    output_names=["out"],
    header=_FP4_HEADER,
    source=_FP4_SIM_SOURCE,
    compile_options={"math_mode": "fast"},
)

_COMPRESS_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_compress",
    input_names=[
        "kv",
        "gate",
        "ape",
        "norm",
        "prev_kv",
        "prev_gate",
        "positions",
        "cos_table",
        "sin_table",
        "params",
    ],
    output_names=["out"],
    header=_COMPRESS_HEADER,
    source=_COMPRESS_SOURCE,
    compile_options={"math_mode": "fast"},
)

_DECODE_POOL_STEP_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_decode_pool_step",
    input_names=[
        "kv_token",
        "gate_token",
        "ape",
        "norm",
        "state_kv",
        "state_gate",
        "prev_kv",
        "prev_gate",
        "seq_len",
        "cos_table",
        "sin_table",
        "params",
    ],
    output_names=[
        "state_kv_out",
        "state_gate_out",
        "prev_kv_out",
        "prev_gate_out",
        "emitted",
        "emit_rows",
    ],
    header=_COMPRESS_HEADER,
    source=_DECODE_POOL_STEP_SOURCE,
    compile_options={"math_mode": "fast"},
)

_INDEXER_SCORES_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_indexer_scores",
    input_names=["q", "k", "weights", "params"],
    output_names=["out"],
    source=_INDEXER_SCORES_SOURCE,
    compile_options={"math_mode": "fast"},
)

_INDEXER_DECODE_SCORES_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_indexer_decode_scores",
    input_names=["q", "k", "weights", "params"],
    output_names=["out"],
    source=_INDEXER_DECODE_SCORES_SOURCE,
    compile_options={"math_mode": "fast"},
)

_TOPK_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_topk512",
    input_names=["x"],
    output_names=["out"],
    header=_TOPK_HEADER,
    source=_TOPK_SOURCE,
    compile_options={"math_mode": "fast"},
)

_PREFILL_PLAN_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_prefill_plan",
    input_names=["topk"],
    output_names=["indices", "mask"],
    source=_PREFILL_PLAN_SOURCE,
)

_DECODE_PLAN_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_decode_plan",
    input_names=["topk", "seq_len"],
    output_names=["indices", "mask"],
    source=_DECODE_PLAN_SOURCE,
)

_SPARSE_ATTENTION_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_sparse_attention",
    input_names=["q", "kv", "indices", "mask", "sinks", "params"],
    output_names=["out"],
    source=_SPARSE_ATTENTION_SOURCE,
    compile_options={"math_mode": "fast"},
)

_SPARSE_ATTENTION_PREFILL_MMA_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_sparse_attention_prefill_mma",
    input_names=["q", "kv", "indices", "mask", "sinks", "params"],
    output_names=["out"],
    source=_SPARSE_ATTENTION_PREFILL_MMA_SOURCE,
    compile_options={"math_mode": "fast"},
)

_SPARSE_ATTENTION_DECODE_KERNEL = mx.fast.metal_kernel(
    name="mfq_dsv4_sparse_attention_decode",
    input_names=["q", "kv", "indices", "mask", "sinks", "params"],
    output_names=["out"],
    source=_SPARSE_ATTENTION_DECODE_SOURCE,
    compile_options={"math_mode": "fast"},
)


def _array(value: mx.array | np.ndarray, dtype: mx.Dtype) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    return mx.contiguous(result.astype(dtype))


class Dsv4PoolUpdate(NamedTuple):
    """Functional state returned by :func:`dsv4_decode_pool_update`."""

    pool: mx.array
    state_kv: mx.array
    state_gate: mx.array
    prev_kv: mx.array | None
    prev_gate: mx.array | None


class Dsv4PoolStep(NamedTuple):
    """Bounded state delta returned by :func:`dsv4_decode_pool_step`.

    ``emitted`` has shape ``[B, 1, D]`` and is valid only for rows whose
    ``emit_rows`` entry is non-negative.  A cache owner can append/scatter
    those rows at a ratio boundary without making the complete pool a custom
    kernel output.
    """

    emitted: mx.array
    emit_rows: mx.array
    state_kv: mx.array
    state_gate: mx.array
    prev_kv: mx.array | None
    prev_gate: mx.array | None


def dsv4_fp4_sim(input: mx.array | np.ndarray) -> mx.array:
    """Simulate DeepSeek-V4's power-of-two-scaled E2M1 cache groups."""

    source = _array(input, mx.float16)
    if source.ndim < 1 or int(source.shape[-1]) % 32 or source.size == 0:
        raise ValueError("DSV4 FP4 simulation expects nonempty f16 [...,32*n]")
    groups = int(source.size) // 32
    return _FP4_SIM_KERNEL(
        inputs=[source],
        grid=(groups * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[tuple(int(value) for value in source.shape)],
        output_dtypes=[mx.float16],
    )[0]


def dsv4_compress(
    kv: mx.array | np.ndarray,
    gate: mx.array | np.ndarray,
    ape: mx.array | np.ndarray,
    norm: mx.array | np.ndarray,
    prev_kv: mx.array | np.ndarray | None,
    prev_gate: mx.array | np.ndarray | None,
    positions: mx.array | np.ndarray,
    cos: mx.array | np.ndarray,
    sin: mx.array | np.ndarray,
    ratio: int,
    overlap: bool,
    quant_mode: int = 0,
    eps: float = 1e-6,
) -> mx.array:
    """Compress raw DSV4 windows into the 512-D or 128-D cache."""

    source = kv if isinstance(kv, mx.array) else mx.array(kv)
    if source.dtype not in (mx.float16, mx.float32):
        source = source.astype(mx.float16)
    source = mx.contiguous(source)
    gate_values = _array(gate, source.dtype)
    ape_values = _array(ape, mx.float32)
    norm_values = _array(norm, mx.float32)
    position_ids = _array(positions, mx.int32)
    cosine = _array(cos, mx.float32)
    sine = _array(sin, mx.float32)
    head_dim = int(norm_values.size)
    ratio_value = int(ratio)
    overlap_value = bool(overlap)
    output_dim = head_dim * (2 if overlap_value else 1)
    if (
        head_dim not in (128, 512)
        or source.ndim != 4
        or gate_values.shape != source.shape
        or int(source.shape[2]) != ratio_value
        or ratio_value <= 0
        or ratio_value > 128
        or int(source.shape[3]) != output_dim
        or tuple(int(value) for value in ape_values.shape) != (ratio_value, output_dim)
        or position_ids.size != int(source.shape[0]) * int(source.shape[1])
        or cosine.ndim != 2
        or sine.shape != cosine.shape
        or int(cosine.shape[1]) < 32
        or quant_mode not in (0, 1, 2)
        or (quant_mode == 1 and head_dim != 512)
        or (quant_mode == 2 and head_dim != 128)
    ):
        raise ValueError("invalid DSV4 compressor input")
    has_prev = prev_kv is not None and int(prev_kv.size) != 0
    if has_prev:
        if prev_gate is None:
            raise ValueError("previous KV and gate state must be provided together")
        previous_kv = _array(prev_kv, source.dtype)
        previous_gate = _array(prev_gate, source.dtype)
        expected_previous = (
            int(source.shape[0]),
            ratio_value,
            head_dim,
        )
        if (
            not overlap_value
            or tuple(int(value) for value in previous_kv.shape) != expected_previous
            or previous_gate.shape != previous_kv.shape
        ):
            raise ValueError("invalid DSV4 overlap history")
    else:
        previous_kv = mx.zeros((1,), dtype=source.dtype)
        previous_gate = mx.zeros((1,), dtype=source.dtype)
    batch, windows = (int(value) for value in source.shape[:2])
    rows = batch * windows
    params = mx.array([float(eps)], dtype=mx.float32)
    return _COMPRESS_KERNEL(
        inputs=[
            source,
            gate_values,
            ape_values,
            norm_values,
            previous_kv,
            previous_gate,
            position_ids,
            cosine,
            sine,
            params,
        ],
        template=[
            ("T", source.dtype),
            ("B", batch),
            ("W", windows),
            ("HEAD_DIM", head_dim),
            ("RATIO", ratio_value),
            ("OVERLAP", int(overlap_value)),
            ("HAS_PREV", int(has_prev)),
            ("TABLE_LEN", int(cosine.shape[0])),
            ("TABLE_STRIDE", int(cosine.shape[1])),
            ("QUANT_MODE", int(quant_mode)),
        ],
        grid=(rows * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[(batch, windows, head_dim)],
        output_dtypes=[mx.float16],
    )[0]


def dsv4_decode_pool_step(
    kv_token: mx.array | np.ndarray,
    gate_token: mx.array | np.ndarray,
    ape: mx.array | np.ndarray,
    norm: mx.array | np.ndarray,
    state_kv: mx.array | np.ndarray,
    state_gate: mx.array | np.ndarray,
    prev_kv: mx.array | np.ndarray | None,
    prev_gate: mx.array | np.ndarray | None,
    seq_len: mx.array | np.ndarray,
    cos: mx.array | np.ndarray,
    sin: mx.array | np.ndarray,
    ratio: int,
    overlap: bool,
    quant_mode: int = 0,
    eps: float = 1e-6,
) -> Dsv4PoolStep:
    """Update bounded compressor state and return only a possible pool row.

    Unlike :func:`dsv4_decode_pool_update`, this operation never returns the
    complete compressed cache.  The amount of output memory is therefore
    independent of context length.
    """

    token = kv_token if isinstance(kv_token, mx.array) else mx.array(kv_token)
    if token.dtype not in (mx.float16, mx.float32):
        token = token.astype(mx.float16)
    token = mx.contiguous(token)
    gate_values = _array(gate_token, token.dtype)
    ape_values = _array(ape, mx.float32)
    norm_values = _array(norm, mx.float32)
    state_values = _array(state_kv, token.dtype)
    state_gate_values = _array(state_gate, token.dtype)
    lengths = _array(seq_len, mx.int32)
    cosine = _array(cos, mx.float32)
    sine = _array(sin, mx.float32)
    head_dim = int(norm_values.size)
    ratio_value = int(ratio)
    overlap_value = bool(overlap)
    output_dim = head_dim * (2 if overlap_value else 1)
    batch = int(token.shape[0]) if token.ndim > 0 else 0
    if (
        head_dim not in (128, 512)
        or token.ndim != 3
        or int(token.shape[1]) != 1
        or int(token.shape[2]) != output_dim
        or gate_values.shape != token.shape
        or tuple(int(value) for value in state_values.shape) != (batch, ratio_value, output_dim)
        or state_gate_values.shape != state_values.shape
        or tuple(int(value) for value in ape_values.shape) != (ratio_value, output_dim)
        or int(lengths.size) != batch
        or ratio_value <= 0
        or ratio_value > 128
        or cosine.ndim != 2
        or sine.shape != cosine.shape
        or int(cosine.shape[1]) < 32
        or quant_mode not in (0, 1, 2)
        or (quant_mode == 1 and head_dim != 512)
        or (quant_mode == 2 and head_dim != 128)
    ):
        raise ValueError("invalid DSV4 decode pool step input")
    expected_previous = (batch, ratio_value, head_dim)
    if overlap_value:
        if prev_kv is None or prev_gate is None:
            raise ValueError("overlap decode requires previous window state")
        previous_kv = _array(prev_kv, token.dtype)
        previous_gate = _array(prev_gate, token.dtype)
        if (
            tuple(int(value) for value in previous_kv.shape) != expected_previous
            or previous_gate.shape != previous_kv.shape
        ):
            raise ValueError("invalid DSV4 previous window state")
    else:
        previous_kv = mx.zeros(expected_previous, dtype=token.dtype)
        previous_gate = mx.zeros(expected_previous, dtype=token.dtype)
    params = mx.array([float(eps)], dtype=mx.float32)
    (
        next_state_kv,
        next_state_gate,
        next_prev_kv,
        next_prev_gate,
        emitted,
        emit_rows,
    ) = _DECODE_POOL_STEP_KERNEL(
        inputs=[
            token,
            gate_values,
            ape_values,
            norm_values,
            state_values,
            state_gate_values,
            previous_kv,
            previous_gate,
            lengths,
            cosine,
            sine,
            params,
        ],
        template=[
            ("T", token.dtype),
            ("B", batch),
            ("HEAD_DIM", head_dim),
            ("RATIO", ratio_value),
            ("OVERLAP", int(overlap_value)),
            ("TABLE_LEN", int(cosine.shape[0])),
            ("TABLE_STRIDE", int(cosine.shape[1])),
            ("QUANT_MODE", int(quant_mode)),
        ],
        grid=(batch * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[
            tuple(int(value) for value in state_values.shape),
            tuple(int(value) for value in state_gate_values.shape),
            expected_previous,
            expected_previous,
            (batch, 1, head_dim),
            (batch,),
        ],
        output_dtypes=[
            token.dtype,
            token.dtype,
            token.dtype,
            token.dtype,
            mx.float16,
            mx.int32,
        ],
    )
    return Dsv4PoolStep(
        emitted,
        emit_rows,
        next_state_kv,
        next_state_gate,
        next_prev_kv if overlap_value else None,
        next_prev_gate if overlap_value else None,
    )


def dsv4_decode_pool_update(
    kv_token: mx.array | np.ndarray,
    gate_token: mx.array | np.ndarray,
    ape: mx.array | np.ndarray,
    norm: mx.array | np.ndarray,
    state_kv: mx.array | np.ndarray,
    state_gate: mx.array | np.ndarray,
    prev_kv: mx.array | np.ndarray | None,
    prev_gate: mx.array | np.ndarray | None,
    pool: mx.array | np.ndarray,
    seq_len: mx.array | np.ndarray,
    cos: mx.array | np.ndarray,
    sin: mx.array | np.ndarray,
    ratio: int,
    overlap: bool,
    quant_mode: int = 0,
    eps: float = 1e-6,
) -> Dsv4PoolUpdate:
    """Compatibility API for a capacity-backed pool.

    New cache owners should use :func:`dsv4_decode_pool_step`; a custom-kernel
    output cannot alias ``pool``, so this legacy CUDA-shaped API necessarily
    exposes a complete pool array.  Internally it uses an indexed MLX update,
    so the complete pool is no longer emitted by a custom Metal kernel.
    """

    pool_values = _array(pool, mx.float16)
    step = dsv4_decode_pool_step(
        kv_token,
        gate_token,
        ape,
        norm,
        state_kv,
        state_gate,
        prev_kv,
        prev_gate,
        seq_len,
        cos,
        sin,
        ratio,
        overlap,
        quant_mode,
        eps,
    )
    batch = int(step.emitted.shape[0])
    head_dim = int(step.emitted.shape[2])
    if (
        pool_values.ndim != 3
        or int(pool_values.shape[0]) != batch
        or int(pool_values.shape[1]) <= 0
        or int(pool_values.shape[2]) != head_dim
    ):
        raise ValueError("invalid DSV4 decode pool update input")
    pool_capacity = int(pool_values.shape[1])
    valid_rows = (step.emit_rows >= 0) & (step.emit_rows < pool_capacity)
    safe_rows = mx.minimum(
        mx.maximum(step.emit_rows, mx.array(0, dtype=mx.int32)),
        mx.array(pool_capacity - 1, dtype=mx.int32),
    )
    batch_indices = mx.arange(batch, dtype=mx.int32)
    replacement = mx.where(
        valid_rows[:, None],
        step.emitted[:, 0],
        pool_values[batch_indices, safe_rows],
    )
    next_pool = pool_values
    next_pool[batch_indices, safe_rows] = replacement
    return Dsv4PoolUpdate(
        next_pool,
        step.state_kv,
        step.state_gate,
        step.prev_kv,
        step.prev_gate,
    )


def dsv4_indexer_scores_decode(
    q: mx.array | np.ndarray,
    k: mx.array | np.ndarray,
    weights: mx.array | np.ndarray,
    query_offset: int,
    ratio: int,
) -> mx.array:
    """Stream a single 64-head DSV4 query across the pooled index cache."""

    query = _array(q, mx.float16)
    key = _array(k, mx.float16)
    head_weights = _array(weights, mx.float16)
    if (
        query.ndim != 4
        or key.ndim != 3
        or head_weights.ndim != 3
        or int(query.shape[1]) != 1
        or tuple(int(value) for value in query.shape[2:]) != (64, 128)
        or int(key.shape[0]) != int(query.shape[0])
        or int(key.shape[2]) != 128
        or tuple(int(value) for value in head_weights.shape) != (int(query.shape[0]), 1, 64)
        or int(query_offset) < 0
        or int(ratio) <= 0
    ):
        raise ValueError("DSV4 decode indexer score shape mismatch")
    batch = int(query.shape[0])
    keys = int(key.shape[1])
    threads = 128
    key_tiles = (keys + threads - 1) // threads
    params = mx.array(
        [1.0 / math.sqrt(128.0 * 64.0)],
        dtype=mx.float32,
    )
    return _INDEXER_DECODE_SCORES_KERNEL(
        inputs=[query, key, head_weights, params],
        template=[
            ("B", batch),
            ("K", keys),
            ("KEY_TILES", key_tiles),
            ("THREADS", threads),
            ("QUERY_OFFSET", int(query_offset)),
            ("RATIO", int(ratio)),
        ],
        grid=(batch * key_tiles * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(batch, 1, keys)],
        output_dtypes=[mx.float16],
    )[0]


def dsv4_indexer_scores(
    q: mx.array | np.ndarray,
    k: mx.array | np.ndarray,
    weights: mx.array | np.ndarray,
    query_offset: int,
    ratio: int,
) -> mx.array:
    """Compute the 64-head DeepSeek-V4 pooled-token index score."""

    query = _array(q, mx.float16)
    key = _array(k, mx.float16)
    head_weights = _array(weights, mx.float16)
    if (
        query.ndim != 4
        or key.ndim != 3
        or head_weights.ndim != 3
        or tuple(int(value) for value in query.shape[2:]) != (64, 128)
        or int(key.shape[0]) != int(query.shape[0])
        or int(key.shape[2]) != 128
        or tuple(int(value) for value in head_weights.shape)
        != (int(query.shape[0]), int(query.shape[1]), 64)
        or int(query_offset) < 0
        or int(ratio) <= 0
    ):
        raise ValueError("DSV4 indexer score shape mismatch")
    batch, queries = (int(value) for value in query.shape[:2])
    keys = int(key.shape[1])
    # The streaming schedule avoids MMA setup and wins for the first sparse
    # decode band.  Past 1K pooled rows, the 64-key MMA tiles have enough
    # parallelism to outperform the register-heavy 64-head scalar scan.
    if queries == 1 and keys <= 1024:
        return dsv4_indexer_scores_decode(
            query,
            key,
            head_weights,
            query_offset,
            ratio,
        )
    key_tiles = (keys + 63) // 64
    params = mx.array(
        [1.0 / math.sqrt(128.0 * 64.0)],
        dtype=mx.float32,
    )
    return _INDEXER_SCORES_KERNEL(
        inputs=[query, key, head_weights, params],
        template=[
            ("B", batch),
            ("M", queries),
            ("K", keys),
            ("KEY_TILES", key_tiles),
            ("QUERY_OFFSET", int(query_offset)),
            ("RATIO", int(ratio)),
        ],
        grid=(batch * queries * key_tiles * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, queries, keys)],
        output_dtypes=[mx.float16],
    )[0]


def dsv4_topk512(
    scores: mx.array | np.ndarray,
    deterministic: bool = True,
) -> mx.array:
    """Return fixed-width top-512 indices using a two-level half histogram.

    Deterministic mode resolves threshold ties by ascending key index.  The
    optional atomic append mode has lower final-pass overhead but does not
    guarantee tie membership or output order across launches.
    """

    source = _array(scores, mx.float16)
    if source.ndim != 3 or int(source.shape[2]) <= 0:
        raise ValueError("DSV4 top-k expects f16 [B,M,K]")
    batch, queries, keys = (int(value) for value in source.shape)
    rows = batch * queries
    return _TOPK_KERNEL(
        inputs=[source],
        template=[
            ("ROWS", rows),
            ("K", keys),
            ("DETERMINISTIC", int(bool(deterministic))),
        ],
        grid=(rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(batch, queries, 512)],
        output_dtypes=[mx.int32],
    )[0]


def dsv4_build_prefill_plan(
    topk: mx.array | np.ndarray,
    query_offset: int,
    local_history: int,
    pool_len: int,
    ratio: int,
    window: int,
) -> tuple[mx.array, mx.array]:
    """Build circular-local plus pooled sparse indices for prefill."""

    selected_topk = _array(topk, mx.int32)
    if (
        selected_topk.ndim != 3
        or int(selected_topk.shape[1]) <= 0
        or min(query_offset, local_history, pool_len) < 0
        or ratio <= 0
        or window <= 0
    ):
        raise ValueError("invalid DSV4 prefill plan input")
    batch, queries, topk_count = (int(value) for value in selected_topk.shape)
    selected = ((window + topk_count + 31) // 32) * 32
    total = batch * queries * selected
    indices, mask = _PREFILL_PLAN_KERNEL(
        inputs=[selected_topk],
        template=[
            ("TOTAL", total),
            ("M", queries),
            ("TOPK_COUNT", topk_count),
            ("SELECTED", selected),
            ("QUERY_OFFSET", int(query_offset)),
            ("LOCAL_HISTORY", int(local_history)),
            ("POOL_LEN", int(pool_len)),
            ("RATIO", int(ratio)),
            ("WINDOW", int(window)),
        ],
        grid=(total, 1, 1),
        threadgroup=(min(256, max(1, total)), 1, 1),
        output_shapes=[
            (batch, queries, selected),
            (batch, queries, selected),
        ],
        output_dtypes=[mx.int32, mx.float16],
    )
    return indices, mask


def dsv4_build_decode_plan(
    topk: mx.array | np.ndarray,
    seq_len: mx.array | np.ndarray,
    pool_len: int,
    ratio: int,
    window: int,
) -> tuple[mx.array, mx.array]:
    """Build circular-local plus pooled sparse indices for decode."""

    selected_topk = _array(topk, mx.int32)
    lengths = _array(seq_len, mx.int32)
    if (
        selected_topk.ndim != 3
        or int(selected_topk.shape[1]) != 1
        or int(lengths.size) != int(selected_topk.shape[0])
        or pool_len < 0
        or ratio <= 0
        or window <= 0
    ):
        raise ValueError("invalid DSV4 decode plan input")
    batch = int(selected_topk.shape[0])
    topk_count = int(selected_topk.shape[2])
    selected = ((window + topk_count + 31) // 32) * 32
    total = batch * selected
    indices, mask = _DECODE_PLAN_KERNEL(
        inputs=[selected_topk, lengths],
        template=[
            ("TOTAL", total),
            ("TOPK_COUNT", topk_count),
            ("SELECTED", selected),
            ("POOL_LEN", int(pool_len)),
            ("RATIO", int(ratio)),
            ("WINDOW", int(window)),
        ],
        grid=(total, 1, 1),
        threadgroup=(min(256, max(1, total)), 1, 1),
        output_shapes=[(batch, 1, selected), (batch, 1, selected)],
        output_dtypes=[mx.int32, mx.float16],
    )
    return indices, mask


def attention_dsv4_sparse(
    q: mx.array | np.ndarray,
    kv: mx.array | np.ndarray,
    indices: mx.array | np.ndarray,
    mask: mx.array | np.ndarray,
    sinks: mx.array | np.ndarray,
    meta: mx.array | np.ndarray | None = None,
    scale: float | None = None,
) -> mx.array:
    """Run selected-row DSV4 attention with per-head attention sinks."""

    del meta
    query = _array(q, mx.float32)
    cache = _array(kv, mx.float16)
    selected_indices = _array(indices, mx.int32)
    selected_mask = _array(mask, mx.float16)
    sink_logits = _array(sinks, mx.float32)
    if (
        query.ndim != 4
        or tuple(int(value) for value in (query.shape[1], query.shape[3])) != (64, 512)
        or cache.ndim != 3
        or int(cache.shape[0]) != int(query.shape[0])
        or int(cache.shape[2]) != 512
        or selected_indices.ndim != 3
        or selected_mask.shape != selected_indices.shape
        or int(selected_indices.shape[0]) != int(query.shape[0])
        or int(selected_indices.shape[1]) != int(query.shape[2])
        or int(selected_indices.shape[2]) <= 0
        or int(selected_indices.shape[2]) % 32
        or int(sink_logits.size) != 64
    ):
        raise ValueError("DSV4 sparse attention shape mismatch")
    batch = int(query.shape[0])
    queries = int(query.shape[2])
    max_seq = int(cache.shape[1])
    selected = int(selected_indices.shape[2])
    selected_scale = 1.0 / math.sqrt(512.0) if scale is None else float(scale)
    params = mx.array([selected_scale], dtype=mx.float32)
    # One all-head workgroup is under-occupied for short verify/prefill
    # batches.  Isolated M5 Max measurements put the crossover at M=32.
    if queries >= 32:
        prefill_query = _array(q, mx.float16)
        return _SPARSE_ATTENTION_PREFILL_MMA_KERNEL(
            inputs=[
                prefill_query,
                cache,
                selected_indices,
                selected_mask,
                sink_logits,
                params,
            ],
            template=[
                ("B", batch),
                ("M", queries),
                ("MAX_SEQ", max_seq),
                ("SELECTED", selected),
            ],
            grid=(batch * queries * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, queries, 64, 512)],
            output_dtypes=[mx.float32],
        )[0]
    if queries == 1:
        return _SPARSE_ATTENTION_DECODE_KERNEL(
            inputs=[
                query,
                cache,
                selected_indices,
                selected_mask,
                sink_logits,
                params,
            ],
            template=[
                ("B", batch),
                ("M", queries),
                ("MAX_SEQ", max_seq),
                ("SELECTED", selected),
            ],
            grid=(batch * queries * 16 * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(batch, queries, 64, 512)],
            output_dtypes=[mx.float32],
        )[0]
    return _SPARSE_ATTENTION_KERNEL(
        inputs=[
            query,
            cache,
            selected_indices,
            selected_mask,
            sink_logits,
            params,
        ],
        template=[
            ("B", batch),
            ("M", queries),
            ("MAX_SEQ", max_seq),
            ("SELECTED", selected),
        ],
        grid=(batch * queries * 64 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, queries, 64, 512)],
        output_dtypes=[mx.float32],
    )[0]


__all__ = [
    "Dsv4PoolStep",
    "Dsv4PoolUpdate",
    "attention_dsv4_sparse",
    "dsv4_build_decode_plan",
    "dsv4_build_prefill_plan",
    "dsv4_compress",
    "dsv4_decode_pool_step",
    "dsv4_decode_pool_update",
    "dsv4_fp4_sim",
    "dsv4_indexer_scores",
    "dsv4_indexer_scores_decode",
    "dsv4_topk512",
]
