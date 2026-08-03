// Vendored Metal programs from the optimized DSV4 implementation.  This
// file is compiled into the native C++ runtime; it is not loaded at runtime.

constexpr const char* kFp4Header = R"MFQ_METAL(
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
)MFQ_METAL";
constexpr const char* kCompressHeader = R"MFQ_METAL(
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
)MFQ_METAL";

constexpr const char* kCompressSource = R"MFQ_METAL(
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
)MFQ_METAL";

constexpr const char* kDecodePoolStepSource = R"MFQ_METAL(
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
    emitted[batch * uint(HEAD_DIM) + dimension] = half(0.0f);
    if (dimension == 0u) {
        emit_rows[batch] = -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!has_token || slot != uint(RATIO - 1)) {
        return;
    }
    uint output_row = uint(length - 1) / uint(RATIO);
    if (dimension == 0u) {
        emit_rows[batch] = int(output_row);
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
    emitted[batch * uint(HEAD_DIM) + dimension] =
        half(mfq_dsv4_bf16_round(result));
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
)MFQ_METAL";

constexpr const char* kFp4SimSource = R"MFQ_METAL(
    uint lane = thread_index_in_simdgroup;
    uint group = threadgroup_position_in_grid.x;
    uint offset = group * 32u + lane;
    float value = float(x[offset]);
    float maximum = simd_max(abs(value));
    float scale = mfq_dsv4_pow2_ceil(
        max(maximum, 6.0f * 0x1p-126f) / 6.0f
    );
    out[offset] = half(mfq_dsv4_fp4(value, scale));
)MFQ_METAL";

constexpr const char* kIndexerScoresSource = R"MFQ_METAL(
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
)MFQ_METAL";

constexpr const char* kIndexerDecodeScoresSource = R"MFQ_METAL(
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
)MFQ_METAL";

constexpr const char* kTopkHeader = R"MFQ_METAL(
    METAL_FUNC uint mfq_dsv4_ordered_half(half value) {
        uint bits = uint(as_type<ushort>(value));
        return (bits & 0x8000u) != 0u
            ? ((~bits) & 0xffffu)
            : (bits | 0x8000u);
    }
)MFQ_METAL";

constexpr const char* kTopkSource = R"MFQ_METAL(
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
)MFQ_METAL";

constexpr const char* kPrefillPlanSource = R"MFQ_METAL(
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
)MFQ_METAL";

constexpr const char* kDecodePlanSource = R"MFQ_METAL(
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
)MFQ_METAL";

constexpr const char* kSparseAttentionSource = R"MFQ_METAL(
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
)MFQ_METAL";

constexpr const char* kSparseAttentionPrefillMmaSource = R"MFQ_METAL(
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
)MFQ_METAL";

constexpr const char* kSparseAttentionDecodeSource = R"MFQ_METAL(
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
)MFQ_METAL";

constexpr const char* kSparseAttentionDirectDecodeSource = R"MFQ_METAL(
    constexpr uint HEADS = 64u;
    constexpr uint HEADS_PER_GROUP = 4u;
    constexpr uint HEAD_GROUPS = HEADS / HEADS_PER_GROUP;
    constexpr uint DIM = 512u;
    constexpr uint VALUES_PER_LANE = DIM / 32u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint head_group = workgroup % HEAD_GROUPS;
    uint batch = workgroup / HEAD_GROUPS;
    if (batch >= uint(B)) {
        return;
    }
    uint head = head_group * HEADS_PER_GROUP + simd_group;
    uint query_base = (batch * HEADS + head) * DIM;
    uint seq_len = uint(max(decode_params[0], 0));
    uint pool_len = uint(max(decode_params[1], 0));
    uint topk_count = uint(max(decode_params[2], 0));
    uint topk_base = batch * topk_count;
    uint local_count = min(seq_len, uint(WINDOW));
    float maximum = sinks[head];
    float denominator = 1.0f;
    float accumulators[VALUES_PER_LANE];
    for (uint item = 0u; item < VALUES_PER_LANE; ++item) {
        accumulators[item] = 0.0f;
    }

    for (uint selected = 0u; selected < local_count; ++selected) {
        uint absolute = seq_len - local_count + selected;
        uint cache_row = absolute % uint(WINDOW);
        uint cache_base =
            (batch * uint(WINDOW) + cache_row) * DIM;
        float dot = 0.0f;
        for (uint item = 0u; item < VALUES_PER_LANE; ++item) {
            uint dimension = lane + item * 32u;
            dot += q[query_base + dimension]
                * float(local_kv[cache_base + dimension]);
        }
        float score = simd_sum(dot) * params[0];
        float new_maximum = max(maximum, score);
        float old_scale = metal::fast::exp(maximum - new_maximum);
        float new_scale = metal::fast::exp(score - new_maximum);
        denominator = denominator * old_scale + new_scale;
        maximum = new_maximum;

        for (uint item = 0u; item < VALUES_PER_LANE; ++item) {
            uint dimension = lane + item * 32u;
            float value = float(local_kv[cache_base + dimension]);
            accumulators[item] =
                accumulators[item] * old_scale + value * new_scale;
        }
    }

    uint visible = min(pool_len, seq_len / uint(RATIO));
    for (uint selected = 0u; selected < topk_count; ++selected) {
        int pooled = topk[topk_base + selected];
        bool valid = pooled >= 0 && uint(pooled) < visible;
        uint cache_row = valid ? uint(pooled) : 0u;
        uint cache_base =
            (batch * uint(POOL_CAPACITY) + cache_row) * DIM;
        float dot = 0.0f;
        for (uint item = 0u; item < VALUES_PER_LANE; ++item) {
            uint dimension = lane + item * 32u;
            dot += q[query_base + dimension]
                * float(pooled_kv[cache_base + dimension]);
        }
        float score = simd_sum(dot);
        score = valid ? score * params[0] : -INFINITY;
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
                ? float(pooled_kv[cache_base + dimension])
                : 0.0f;
            accumulators[item] =
                accumulators[item] * old_scale + value * new_scale;
        }
    }

    float inverse_denominator = 1.0f / denominator;
    uint output_base = (batch * HEADS + head) * DIM;
    for (uint item = 0u; item < VALUES_PER_LANE; ++item) {
        uint dimension = lane + item * 32u;
        out[output_base + dimension] =
            accumulators[item] * inverse_denominator;
    }
)MFQ_METAL";
