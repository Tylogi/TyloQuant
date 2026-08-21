#include "mlx_nint.h"

#include "mlx_staging_allocator.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr const char* kNintHeader = R"METAL(
#include <metal_simdgroup_matrix>

template <typename Stream>
inline uint mfq_nint_read_bits(
    Stream stream,
    uint value_index,
    uint bits
) {
    // Form floor(value_index * bits / 8) without first materializing the
    // bit offset.  Qwen3.6's 248320 x 5120 embedding/lm_head crosses 2^32
    // bits even though its packed byte stream is still well within uint.
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3) * bits + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1u]) << 8;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

template <typename Stream>
inline uint mfq_nint_read_value(
    Stream stream,
    uint value_index,
    uint bits,
    uint group_size,
    uint q5_exec
) {
    if (q5_exec != 0u && bits == 5u) {
        uint metadata_index = value_index / group_size;
        uint element = value_index - metadata_index * group_size;
        uint low_bytes = (group_size + 1u) >> 1;
        uint high_bytes = (group_size + 7u) >> 3;
        uint group_offset = metadata_index * (low_bytes + high_bytes);
        uint low_packed = uint(stream[group_offset + (element >> 1)]);
        uint low = (low_packed >> ((element & 1u) * 4u)) & 15u;
        uint high = (
            uint(stream[group_offset + low_bytes + (element >> 3)])
            >> (element & 7u)
        ) & 1u;
        return low | (high << 4u);
    }
    return mfq_nint_read_bits(stream, value_index, bits);
}
)METAL";

constexpr const char* kNintMatmul = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5;
    uint output = workgroup % uint(OUT);
    uint row_tile = workgroup / uint(OUT);
    uint first_row = row_tile * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }

    float accumulators[TILE_M];
    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        accumulators[local_row] = 0.0f;
    }
    for (uint input_index = lane; input_index < uint(K); input_index += 32u) {
        uint group = input_index / uint(GS);
        uint element = input_index - group * uint(GS);
        uint metadata_index = output * uint(NG) + group;
        float scale = neuron_scale[output] * float(sub_scale[metadata_index]);
        float minimum = neuron_min[output] * float(sub_min[metadata_index]);
        uint quantized_index = metadata_index * uint(GS) + element;
        uint quantized = BITS == 2 && Q5_EXEC == 0
            ? (
                uint(q_packed[quantized_index >> 2u])
                >> ((quantized_index & 3u) * 2u)
              ) & 3u
            : (BITS == 4 && Q5_EXEC == 0
            ? (
                uint(q_packed[quantized_index >> 1u])
                >> ((quantized_index & 1u) * 4u)
              ) & 15u
            : (BITS == 8 && Q5_EXEC == 0
            ? uint(q_packed[quantized_index])
            : mfq_nint_read_value(
                q_packed,
                quantized_index,
                uint(BITS),
                uint(GS),
                uint(Q5_EXEC))));
        float weight = scale * float(quantized) - minimum;
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            uint row = first_row + local_row;
            if (row < uint(M)) {
                accumulators[local_row] +=
                    float(x[row * uint(K) + input_index]) * weight;
            }
        }
    }

    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        float total = simd_sum(accumulators[local_row]);
        uint row = first_row + local_row;
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] = T(total);
        }
    }
)METAL";

constexpr const char* kNint4Matmul = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5;
    uint output = workgroup % uint(OUT);
    uint row_tile = workgroup / uint(OUT);
    uint first_row = row_tile * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }

    float accumulators[TILE_M];
    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        accumulators[local_row] = 0.0f;
    }
    constexpr uint pairs = (K + 1) / 2;
    for (uint pair = lane; pair < pairs; pair += 32u) {
        uint input_index = pair * 2u;
        uint group = input_index / uint(GS);
        uint element = input_index - group * uint(GS);
        uint metadata_index = output * uint(NG) + group;
        float scale =
            neuron_scale[output] * float(sub_scale[metadata_index]);
        float minimum =
            neuron_min[output] * float(sub_min[metadata_index]);
        uint quantized_index = metadata_index * uint(GS) + element;
        uint packed = uint(q_packed[quantized_index >> 1]);
        float weight0 = scale * float(packed & 15u) - minimum;
        float weight1 = scale * float(packed >> 4) - minimum;
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            uint row = first_row + local_row;
            if (row < uint(M)) {
                accumulators[local_row] +=
                    float(x[row * uint(K) + input_index]) * weight0;
                if (input_index + 1u < uint(K)) {
                    accumulators[local_row] +=
                        float(x[row * uint(K) + input_index + 1u])
                        * weight1;
                }
            }
        }
    }

    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        float total = simd_sum(accumulators[local_row]);
        uint row = first_row + local_row;
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] = T(total);
        }
    }
)METAL";

// Single-row homogeneous NINT4 gate/up projection fused with SwiGLU.
//
// Unlike the generic paired kernel, each SIMD group computes two output
// rows. Complete quantization groups are assigned to lanes so gate/up share
// every activation load and scale/min metadata is loaded only once per GS
// group. Eight SIMD groups per threadgroup produce sixteen FFN rows while
// keeping the paired gate/up accumulator set compact.
constexpr const char* kNint4SwiGlu = R"METAL(
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint OUTPUTS_PER_SIMD = 2u;
    constexpr uint OUTPUTS_PER_TG =
        SIMD_GROUPS * OUTPUTS_PER_SIMD;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD;

    float gate_acc[OUTPUTS_PER_SIMD] = {0.0f};
    float up_acc[OUTPUTS_PER_SIMD] = {0.0f};
    uint outputs[OUTPUTS_PER_SIMD];
    for (uint row = 0u; row < OUTPUTS_PER_SIMD; ++row) {
        outputs[row] = min(output_base + row, uint(OUT) - 1u);
    }

    for (uint group = lane; group < uint(NG); group += 32u) {
        uint column_base = group * uint(GS);
        float activation_sum = 0.0f;
        float gate_quantized_dots[OUTPUTS_PER_SIMD] = {0.0f};
        float up_quantized_dots[OUTPUTS_PER_SIMD] = {0.0f};
#if defined(GS) && (GS % 8) == 0
        for (uint element = 0u; element < uint(GS); element += 8u) {
            uint column = column_base + element;
            float4 activation0 = float4(0.0f);
            float4 activation1 = float4(0.0f);
            if (column + 7u < uint(K)) {
                activation0 =
                    float4(*(device const half4*)(x + column));
                activation1 =
                    float4(*(device const half4*)(x + column + 4u));
            } else {
                activation0.x =
                    column < uint(K) ? float(x[column]) : 0.0f;
                activation0.y = column + 1u < uint(K)
                    ? float(x[column + 1u]) : 0.0f;
                activation0.z = column + 2u < uint(K)
                    ? float(x[column + 2u]) : 0.0f;
                activation0.w = column + 3u < uint(K)
                    ? float(x[column + 3u]) : 0.0f;
                activation1.x = column + 4u < uint(K)
                    ? float(x[column + 4u]) : 0.0f;
                activation1.y = column + 5u < uint(K)
                    ? float(x[column + 5u]) : 0.0f;
                activation1.z = column + 6u < uint(K)
                    ? float(x[column + 6u]) : 0.0f;
                activation1.w = column + 7u < uint(K)
                    ? float(x[column + 7u]) : 0.0f;
            }
            activation_sum += activation0.x + activation0.y;
            activation_sum += activation0.z + activation0.w;
            activation_sum += activation1.x + activation1.y;
            activation_sum += activation1.z + activation1.w;
            for (uint row = 0u; row < OUTPUTS_PER_SIMD; ++row) {
                uint metadata_index =
                    outputs[row] * uint(NG) + group;
                uint quantized_index =
                    metadata_index * uint(GS) + element;
                uint gate_packed = *(device const uint*)(
                    gate_q + (quantized_index >> 1));
                uint up_packed = *(device const uint*)(
                    up_q + (quantized_index >> 1));
                gate_quantized_dots[row] +=
                    activation0.x * float(gate_packed & 15u)
                    + activation0.y * float((gate_packed >> 4u) & 15u);
                gate_quantized_dots[row] +=
                    activation0.z * float((gate_packed >> 8u) & 15u)
                    + activation0.w * float((gate_packed >> 12u) & 15u);
                gate_quantized_dots[row] +=
                    activation1.x * float((gate_packed >> 16u) & 15u)
                    + activation1.y * float((gate_packed >> 20u) & 15u);
                gate_quantized_dots[row] +=
                    activation1.z * float((gate_packed >> 24u) & 15u)
                    + activation1.w * float(gate_packed >> 28u);
                up_quantized_dots[row] +=
                    activation0.x * float(up_packed & 15u)
                    + activation0.y * float((up_packed >> 4u) & 15u);
                up_quantized_dots[row] +=
                    activation0.z * float((up_packed >> 8u) & 15u)
                    + activation0.w * float((up_packed >> 12u) & 15u);
                up_quantized_dots[row] +=
                    activation1.x * float((up_packed >> 16u) & 15u)
                    + activation1.y * float((up_packed >> 20u) & 15u);
                up_quantized_dots[row] +=
                    activation1.z * float((up_packed >> 24u) & 15u)
                    + activation1.w * float(up_packed >> 28u);
            }
        }
#else
        for (uint element = 0u; element < uint(GS); element += 2u) {
            uint column = column_base + element;
            float activation0 =
                column < uint(K) ? float(x[column]) : 0.0f;
            float activation1 =
                column + 1u < uint(K) ? float(x[column + 1u]) : 0.0f;
            activation_sum += activation0 + activation1;
            for (uint row = 0u; row < OUTPUTS_PER_SIMD; ++row) {
                uint metadata_index =
                    outputs[row] * uint(NG) + group;
                uint quantized_index =
                    metadata_index * uint(GS) + element;
                uint gate_packed =
                    uint(gate_q[quantized_index >> 1]);
                uint up_packed =
                    uint(up_q[quantized_index >> 1]);
                gate_quantized_dots[row] +=
                    activation0 * float(gate_packed & 15u)
                    + activation1 * float(gate_packed >> 4);
                up_quantized_dots[row] +=
                    activation0 * float(up_packed & 15u)
                    + activation1 * float(up_packed >> 4);
            }
        }
#endif
        for (uint row = 0u; row < OUTPUTS_PER_SIMD; ++row) {
            uint output = outputs[row];
            uint metadata_index = output * uint(NG) + group;
            float gate_scale =
                gate_neuron_scale[output]
                * float(gate_sub_scale[metadata_index]);
            float gate_minimum =
                gate_neuron_min[output]
                * float(gate_sub_min[metadata_index]);
            float up_scale =
                up_neuron_scale[output]
                * float(up_sub_scale[metadata_index]);
            float up_minimum =
                up_neuron_min[output]
                * float(up_sub_min[metadata_index]);
            gate_acc[row] = fma(
                gate_scale,
                gate_quantized_dots[row],
                fma(
                    -gate_minimum,
                    activation_sum,
                    gate_acc[row]));
            up_acc[row] = fma(
                up_scale,
                up_quantized_dots[row],
                fma(
                    -up_minimum,
                    activation_sum,
                    up_acc[row]));
        }
    }

    for (uint row = 0u; row < OUTPUTS_PER_SIMD; ++row) {
        // Preserve the established separate-GEMV semantics: each projection
        // is rounded to the activation dtype before the elementwise SwiGLU.
        float gate = float(T(simd_sum(gate_acc[row])));
        float up = float(T(simd_sum(up_acc[row])));
        uint output = output_base + row;
        if (lane == 0u && output < uint(OUT)) {
            float silu = gate / (1.0f + metal::exp(-gate));
            y[output] = T(silu * up);
        }
    }
)METAL";


constexpr const char* kNintGemvFast = R"METAL(
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;

    float accumulators[ROWS_PER_SIMD] = {0.0f};

    // Assign complete quantization groups to lanes.  Metadata is loaded once
    // per group and every activation is reused across four output rows.
    for (uint group = lane; group < uint(NG); group += 32u) {
        uint outputs[ROWS_PER_SIMD];
        float scales[ROWS_PER_SIMD];
        float minimums[ROWS_PER_SIMD];
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            uint output = min(output_base + row, uint(OUT) - 1u);
            uint metadata_index = output * uint(NG) + group;
            outputs[row] = output;
            scales[row] =
                neuron_scale[output] * float(sub_scale[metadata_index]);
            minimums[row] =
                neuron_min[output] * float(sub_min[metadata_index]);
        }

        if (BITS == 5 && Q5_EXEC != 0) {
            constexpr uint LOW_BYTES = (uint(GS) + 1u) / 2u;
            constexpr uint EXEC_BYTES =
                LOW_BYTES + (uint(GS) + 7u) / 8u;
            for (uint element = 0u; element < uint(GS); element += 8u) {
                float activations[8];
                for (uint component = 0u; component < 8u; ++component) {
                    uint column = group * uint(GS) + element + component;
                    activations[component] =
                        element + component < uint(GS) &&
                            column < uint(K)
                        ? float(x[column])
                        : 0.0f;
                }
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    uint metadata_index =
                        outputs[row] * uint(NG) + group;
                    uint group_offset = metadata_index * EXEC_BYTES;
                    uint high = uint(q_packed[
                        group_offset + LOW_BYTES + (element >> 3)
                    ]);
                    for (uint component = 0u;
                         component < 8u;
                         ++component) {
                        if (element + component >= uint(GS)) {
                            break;
                        }
                        uint low_packed = uint(q_packed[
                            group_offset
                            + ((element + component) >> 1)
                        ]);
                        uint low = (
                            low_packed
                            >> (((element + component) & 1u) * 4u)
                        ) & 15u;
                        uint quantized =
                            low | (((high >> component) & 1u) << 4u);
                        accumulators[row] += activations[component] * (
                            scales[row] * float(quantized)
                            - minimums[row]);
                    }
                }
            }
        } else if (BITS == 6 && (GS % 4) == 0) {
            for (uint element = 0u; element < uint(GS); element += 4u) {
                uint column_base = group * uint(GS) + element;
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    uint quantized_index =
                        (outputs[row] * uint(NG) + group) * uint(GS)
                        + element;
                    uint byte_index =
                        (quantized_index >> 3) * 6u
                        + (((quantized_index & 7u) * 6u) >> 3);
                    uint packed = uint(q_packed[byte_index])
                        | (uint(q_packed[byte_index + 1u]) << 8)
                        | (uint(q_packed[byte_index + 2u]) << 16);
                    for (uint component = 0u;
                         component < 4u;
                         ++component) {
                        uint column = column_base + component;
                        if (column < uint(K)) {
                            uint quantized =
                                (packed >> (component * 6u)) & 63u;
                            accumulators[row] += float(x[column]) * (
                                scales[row] * float(quantized)
                                - minimums[row]);
                        }
                    }
                }
            }
        } else if (BITS == 2 && (GS % 4) == 0) {
            for (uint element = 0u; element < uint(GS); element += 4u) {
                uint column_base = group * uint(GS) + element;
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    uint quantized_index =
                        (outputs[row] * uint(NG) + group) * uint(GS)
                        + element;
                    uint packed = uint(q_packed[quantized_index >> 2]);
                    for (uint component = 0u;
                         component < 4u;
                         ++component) {
                        uint column = column_base + component;
                        if (column < uint(K)) {
                            uint quantized =
                                (packed >> (component * 2u)) & 3u;
                            accumulators[row] += float(x[column]) * (
                                scales[row] * float(quantized)
                                - minimums[row]);
                        }
                    }
                }
            }
        } else if (BITS == 3 && (GS % 8) == 0) {
            for (uint element = 0u; element < uint(GS); element += 8u) {
                uint column_base = group * uint(GS) + element;
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    uint quantized_index =
                        (outputs[row] * uint(NG) + group) * uint(GS)
                        + element;
                    uint byte_index =
                        (quantized_index >> 3) * 3u
                        + (((quantized_index & 7u) * 3u) >> 3);
                    uint packed = uint(q_packed[byte_index])
                        | (uint(q_packed[byte_index + 1u]) << 8)
                        | (uint(q_packed[byte_index + 2u]) << 16);
                    for (uint component = 0u;
                         component < 8u;
                         ++component) {
                        uint column = column_base + component;
                        if (column < uint(K)) {
                            uint quantized =
                                (packed >> (component * 3u)) & 7u;
                            accumulators[row] += float(x[column]) * (
                                scales[row] * float(quantized)
                                - minimums[row]);
                        }
                    }
                }
            }
        } else if (BITS == 8) {
            for (uint element = 0u; element < uint(GS); ++element) {
                uint column = group * uint(GS) + element;
                if (column >= uint(K)) {
                    break;
                }
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    uint quantized_index =
                        (outputs[row] * uint(NG) + group) * uint(GS)
                        + element;
                    accumulators[row] += float(x[column]) * (
                        scales[row] * float(q_packed[quantized_index])
                        - minimums[row]);
                }
            }
        } else {
            for (uint element = 0u; element < uint(GS); ++element) {
                uint column = group * uint(GS) + element;
                if (column >= uint(K)) {
                    break;
                }
                float activation = float(x[column]);
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    uint quantized_index =
                        (outputs[row] * uint(NG) + group) * uint(GS)
                        + element;
                    uint quantized = mfq_nint_read_value(
                        q_packed,
                        quantized_index,
                        uint(BITS),
                        uint(GS),
                        uint(Q5_EXEC));
                    accumulators[row] += activation * (
                        scales[row] * float(quantized)
                        - minimums[row]);
                }
            }
        }
    }

    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        float total = simd_sum(accumulators[row]);
        uint output = output_base + row;
        if (lane == 0u && output < uint(OUT)) {
            y[output] = T(total);
        }
    }
)METAL";

// FP16 single-token NINT5/GS28 decode.
//
// Each SIMD group is split into four independent eight-lane reductions, with
// one output assigned to each subgroup. A lane owns complete quantization
// groups, loads the four high-bit bytes once, decodes each low byte into two
// values, and applies the two-level affine metadata once per group:
//
//   dot(x, scale * q - minimum)
//       = scale * dot(x, q) - minimum * sum(x)
//
// Four SIMD groups produce eight output rows per threadgroup. The specialized
// dispatch is restricted to FP16, NINT5, GS28, and the low4/high1 execution
// layout; all other shapes and dtypes retain kNintGemvFast/kNintMatmul.
constexpr const char* kNint5Gs28Gemv = R"METAL(
    constexpr uint SIMD_GROUPS = 4u;
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD = 32u / K_LANES;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint EXEC_BYTES = 18u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_row = lane / K_LANES;
    uint output_index =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD + simd_row;
    uint output = min(output_index, uint(OUT) - 1u);
    float neuron_s = neuron_scale[output];
    float neuron_m = neuron_min[output];
    float accumulator = 0.0f;

    for (uint group = k_lane; group < uint(NG); group += K_LANES) {
        uint metadata_index = output * uint(NG) + group;
        uint group_offset = metadata_index * EXEC_BYTES;
        uint high_bits =
            uint(q_packed[group_offset + 14u])
            | (uint(q_packed[group_offset + 15u]) << 8u)
            | (uint(q_packed[group_offset + 16u]) << 16u)
            | (uint(q_packed[group_offset + 17u]) << 24u);
        uint column_base = group * 28u;
        float activation_sum = 0.0f;
        float quantized_dot = 0.0f;

        for (uint element = 0u; element < 28u; element += 4u) {
            uint column = column_base + element;
            float4 activation = float4(0.0f);
            if (column + 3u < uint(K)) {
                activation =
                    float4(*(device const half4*)(x + column));
            } else {
                activation.x =
                    column < uint(K) ? float(x[column]) : 0.0f;
                activation.y =
                    column + 1u < uint(K)
                    ? float(x[column + 1u]) : 0.0f;
                activation.z =
                    column + 2u < uint(K)
                    ? float(x[column + 2u]) : 0.0f;
                activation.w =
                    column + 3u < uint(K)
                    ? float(x[column + 3u]) : 0.0f;
            }
            activation_sum +=
                activation.x + activation.y
                + activation.z + activation.w;

            uint low0 =
                uint(q_packed[group_offset + (element >> 1u)]);
            uint low1 =
                uint(q_packed[
                    group_offset + (element >> 1u) + 1u
                ]);
            uint high = high_bits >> element;
            float4 quantized = float4(
                float((low0 & 15u) | ((high & 1u) << 4u)),
                float(
                    (low0 >> 4u)
                    | (((high >> 1u) & 1u) << 4u)),
                float(
                    (low1 & 15u)
                    | (((high >> 2u) & 1u) << 4u)),
                float(
                    (low1 >> 4u)
                    | (((high >> 3u) & 1u) << 4u)));
            quantized_dot += dot(activation, quantized);
        }

        float scale =
            neuron_s * float(sub_scale[metadata_index]);
        float minimum =
            neuron_m * float(sub_min[metadata_index]);
        accumulator +=
            scale * quantized_dot - minimum * activation_sum;
    }

    accumulator += simd_shuffle_down(accumulator, 4);
    accumulator += simd_shuffle_down(accumulator, 2);
    accumulator += simd_shuffle_down(accumulator, 1);
    if (k_lane == 0u && output_index < uint(OUT)) {
        y[output_index] = T(accumulator);
    }
)METAL";

// FP16 single-token NINT4/GS24 decode.
//
// GS24 is byte aligned: four 4-bit values occupy two bytes and one complete
// quantization group occupies 12 bytes. Each SIMD group computes two output
// rows while lanes own complete groups. Applying affine metadata after each
// local group dot avoids repeating scale/minimum arithmetic for every value:
//
//   dot(x, scale * q - minimum)
//       = scale * dot(x, q) - minimum * sum(x)
//
// Eight SIMD groups produce 16 output rows per threadgroup. The specialized
// dispatch is restricted to FP16, one input row, NINT4, and GS24; every other
// NINT4 profile retains kNint4Matmul.
constexpr const char* kNint4Gs24Gemv = R"METAL(
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint ROWS_PER_SIMD = 2u;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint GROUP_BYTES = 12u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;

    uint metadata_bases[ROWS_PER_SIMD];
    float neuron_scales[ROWS_PER_SIMD];
    float neuron_minimums[ROWS_PER_SIMD];
    float accumulators[ROWS_PER_SIMD] = {0.0f};
    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        uint output = min(output_base + row, uint(OUT) - 1u);
        metadata_bases[row] = output * uint(NG);
        neuron_scales[row] = neuron_scale[output];
        neuron_minimums[row] = neuron_min[output];
    }

    for (uint group = lane; group < uint(NG); group += 32u) {
        float activation_sum = 0.0f;
        float quantized_dots[ROWS_PER_SIMD] = {0.0f};
        uint packed_words[ROWS_PER_SIMD][3];
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            uint metadata_index = metadata_bases[row] + group;
            device const uint* words = (device const uint*)(
                q_packed + metadata_index * GROUP_BYTES);
            packed_words[row][0] = words[0];
            packed_words[row][1] = words[1];
            packed_words[row][2] = words[2];
        }
        for (uint chunk = 0u; chunk < 6u; ++chunk) {
            uint column = group * 24u + chunk * 4u;
            float4 activation = float4(0.0f);
            if (column + 3u < uint(K)) {
                activation =
                    float4(*(device const half4*)(x + column));
            } else {
                activation.x =
                    column < uint(K) ? float(x[column]) : 0.0f;
                activation.y =
                    column + 1u < uint(K)
                    ? float(x[column + 1u]) : 0.0f;
                activation.z =
                    column + 2u < uint(K)
                    ? float(x[column + 2u]) : 0.0f;
                activation.w =
                    column + 3u < uint(K)
                    ? float(x[column + 3u]) : 0.0f;
            }
            activation_sum +=
                activation.x + activation.y
                + activation.z + activation.w;

            for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                uint packed = packed_words[row][chunk >> 1u]
                    >> ((chunk & 1u) * 16u);
                float4 quantized = float4(
                    float(packed & 15u),
                    float((packed >> 4u) & 15u),
                    float((packed >> 8u) & 15u),
                    float((packed >> 12u) & 15u));
                quantized_dots[row] += dot(activation, quantized);
            }
        }

        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            uint metadata_index = metadata_bases[row] + group;
            float scale =
                neuron_scales[row] *
                float(sub_scale[metadata_index]);
            float minimum =
                neuron_minimums[row] *
                float(sub_min[metadata_index]);
            accumulators[row] = fma(
                scale,
                quantized_dots[row],
                fma(
                    -minimum,
                    activation_sum,
                    accumulators[row]));
        }
    }

    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        float total = simd_sum(accumulators[row]);
        uint output = output_base + row;
        if (lane == 0u && output < uint(OUT)) {
            y[output] = T(total);
        }
    }
)METAL";

// FP16 single-token NINT6/GS24 decode.
//
// GS24 is byte aligned: four 6-bit values occupy three bytes and one complete
// quantization group occupies 18 bytes.  Each SIMD group computes two output
// rows while lanes own complete groups.  Applying the affine metadata after
// each local group dot removes the per-value scale/minimum arithmetic:
//
//   dot(x, scale * q - minimum)
//       = scale * dot(x, q) - minimum * sum(x)
//
// Eight SIMD groups produce 16 output rows per threadgroup.  The specialized
// dispatch is restricted to FP16, one input row, NINT6, and GS24; every other
// profile retains the established specialized or generic path.
constexpr const char* kNint6Gs24Gemv = R"METAL(
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint ROWS_PER_SIMD = 2u;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint GROUP_BYTES = 18u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;

    uint metadata_bases[ROWS_PER_SIMD];
    float neuron_scales[ROWS_PER_SIMD];
    float neuron_minimums[ROWS_PER_SIMD];
    float accumulators[ROWS_PER_SIMD] = {0.0f};
    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        uint output = min(output_base + row, uint(OUT) - 1u);
        metadata_bases[row] = output * uint(NG);
        neuron_scales[row] = neuron_scale[output];
        neuron_minimums[row] = neuron_min[output];
    }

    for (uint group = lane; group < uint(NG); group += 32u) {
        float activation_sum = 0.0f;
        float quantized_dots[ROWS_PER_SIMD] = {0.0f};
        ushort packed_words[ROWS_PER_SIMD][9];
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            uint metadata_index = metadata_bases[row] + group;
            device const ushort* words = (device const ushort*)(
                q_packed + metadata_index * GROUP_BYTES);
            for (uint word = 0u; word < 9u; ++word) {
                packed_words[row][word] = words[word];
            }
        }
        for (uint chunk = 0u; chunk < 6u; ++chunk) {
            uint column = group * 24u + chunk * 4u;
            float4 activation = float4(0.0f);
            if (column + 3u < uint(K)) {
                activation =
                    float4(*(device const half4*)(x + column));
            } else {
                activation.x =
                    column < uint(K) ? float(x[column]) : 0.0f;
                activation.y =
                    column + 1u < uint(K)
                    ? float(x[column + 1u]) : 0.0f;
                activation.z =
                    column + 2u < uint(K)
                    ? float(x[column + 2u]) : 0.0f;
                activation.w =
                    column + 3u < uint(K)
                    ? float(x[column + 3u]) : 0.0f;
            }
            activation_sum +=
                activation.x + activation.y
                + activation.z + activation.w;

            for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                uint byte_index = chunk * 3u;
                uint word_index = byte_index >> 1u;
                uint shift = (byte_index & 1u) * 8u;
                uint packed =
                    (uint(packed_words[row][word_index]) >> shift)
                    | (uint(packed_words[row][word_index + 1u])
                       << (16u - shift));
                float4 quantized = float4(
                    float(packed & 63u),
                    float((packed >> 6u) & 63u),
                    float((packed >> 12u) & 63u),
                    float((packed >> 18u) & 63u));
                quantized_dots[row] += dot(activation, quantized);
            }
        }

        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            uint metadata_index = metadata_bases[row] + group;
            float scale =
                neuron_scales[row] *
                float(sub_scale[metadata_index]);
            float minimum =
                neuron_minimums[row] *
                float(sub_min[metadata_index]);
            accumulators[row] = fma(
                scale,
                quantized_dots[row],
                fma(
                    -minimum,
                    activation_sum,
                    accumulators[row]));
        }
    }

    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        float total = simd_sum(accumulators[row]);
        uint output = output_base + row;
        if (lane == 0u && output < uint(OUT)) {
            y[output] = T(total);
        }
    }
)METAL";

// NINT6/GS24 LM-head decode with an in-kernel first-stage argmax. The
// projection still rounds each logit to T before comparison, exactly matching
// the regular GEMV followed by the greedy sampler. Only one candidate per
// 16-output threadgroup is written to device memory.
constexpr const char* kNint6Gs24GreedyPartial = R"METAL(
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint ROWS_PER_SIMD = 2u;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint GROUP_BYTES = 18u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint part = threadgroup_position_in_grid.x;
    uint output_base =
        part * ROWS_PER_TG + simd_group * ROWS_PER_SIMD;

    uint metadata_bases[ROWS_PER_SIMD];
    float neuron_scales[ROWS_PER_SIMD];
    float neuron_minimums[ROWS_PER_SIMD];
    float accumulators[ROWS_PER_SIMD] = {0.0f};
    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        uint output = min(output_base + row, uint(OUT) - 1u);
        metadata_bases[row] = output * uint(NG);
        neuron_scales[row] = neuron_scale[output];
        neuron_minimums[row] = neuron_min[output];
    }

    for (uint group = lane; group < uint(NG); group += 32u) {
        float activation_sum = 0.0f;
        float quantized_dots[ROWS_PER_SIMD] = {0.0f};
        ushort packed_words[ROWS_PER_SIMD][9];
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            uint metadata_index = metadata_bases[row] + group;
            device const ushort* words = (device const ushort*)(
                q_packed + metadata_index * GROUP_BYTES);
            for (uint word = 0u; word < 9u; ++word) {
                packed_words[row][word] = words[word];
            }
        }
        for (uint chunk = 0u; chunk < 6u; ++chunk) {
            uint column = group * 24u + chunk * 4u;
            float4 activation = float4(0.0f);
            if (column + 3u < uint(K)) {
                activation = float4(*(device const half4*)(x + column));
            } else {
                activation.x =
                    column < uint(K) ? float(x[column]) : 0.0f;
                activation.y = column + 1u < uint(K)
                    ? float(x[column + 1u]) : 0.0f;
                activation.z = column + 2u < uint(K)
                    ? float(x[column + 2u]) : 0.0f;
                activation.w = column + 3u < uint(K)
                    ? float(x[column + 3u]) : 0.0f;
            }
            activation_sum +=
                activation.x + activation.y + activation.z + activation.w;

            for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                uint byte_index = chunk * 3u;
                uint word_index = byte_index >> 1u;
                uint shift = (byte_index & 1u) * 8u;
                uint packed =
                    (uint(packed_words[row][word_index]) >> shift)
                    | (uint(packed_words[row][word_index + 1u])
                       << (16u - shift));
                float4 quantized = float4(
                    float(packed & 63u),
                    float((packed >> 6u) & 63u),
                    float((packed >> 12u) & 63u),
                    float((packed >> 18u) & 63u));
                quantized_dots[row] += dot(activation, quantized);
            }
        }

        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            uint metadata_index = metadata_bases[row] + group;
            float scale = neuron_scales[row] * float(sub_scale[metadata_index]);
            float minimum =
                neuron_minimums[row] * float(sub_min[metadata_index]);
            accumulators[row] = fma(
                scale,
                quantized_dots[row],
                fma(-minimum, activation_sum, accumulators[row]));
        }
    }

    threadgroup float candidates[ROWS_PER_TG];
    threadgroup int candidate_indices[ROWS_PER_TG];
    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        float total = simd_sum(accumulators[row]);
        if (lane == 0u) {
            uint slot = simd_group * ROWS_PER_SIMD + row;
            uint output = output_base + row;
            float rounded = output < uint(OUT)
                ? float(T(total))
                : -FLT_MAX;
            candidates[slot] = isnan(rounded) ? -FLT_MAX : rounded;
            candidate_indices[slot] =
                output < uint(OUT) ? int(output) : int(OUT);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u) {
        float best = lane < ROWS_PER_TG ? candidates[lane] : -FLT_MAX;
        int best_index = lane < ROWS_PER_TG
            ? candidate_indices[lane]
            : int(OUT);
        for (uint stride = 16u; stride > 0u; stride >>= 1u) {
            float other = simd_shuffle_down(best, stride);
            int other_index = simd_shuffle_down(best_index, stride);
            if (lane < stride && (
                other > best ||
                (other == best && other_index < best_index))) {
                best = other;
                best_index = other_index;
            }
        }
        if (lane == 0u) {
            partial_values[part] = best;
            partial_indices[part] = best_index;
        }
    }
)METAL";

constexpr const char* kNintGreedyReduce = R"METAL(
    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    float best = -FLT_MAX;
    int best_index = int(OUT);
    for (uint part = tid; part < uint(PARTS); part += 256u) {
        float value = partial_values[part];
        int index = partial_indices[part];
        if (value > best || (value == best && index < best_index)) {
            best = value;
            best_index = index;
        }
    }
    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        float other = simd_shuffle_down(best, stride);
        int other_index = simd_shuffle_down(best_index, stride);
        if (lane < stride && (
            other > best ||
            (other == best && other_index < best_index))) {
            best = other;
            best_index = other_index;
        }
    }
    threadgroup float simd_values[8];
    threadgroup int simd_indices[8];
    if (lane == 0u) {
        simd_values[simd_group] = best;
        simd_indices[simd_group] = best_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        best = lane < 8u ? simd_values[lane] : -FLT_MAX;
        best_index = lane < 8u ? simd_indices[lane] : int(OUT);
        for (uint stride = 16u; stride > 0u; stride >>= 1u) {
            float other = simd_shuffle_down(best, stride);
            int other_index = simd_shuffle_down(best_index, stride);
            if (lane < stride && (
                other > best ||
                (other == best && other_index < best_index))) {
                best = other;
                best_index = other_index;
            }
        }
        if (lane == 0u) output[0] = best_index;
    }
)METAL";

constexpr const char* kNintEmbedding = R"METAL(
    uint output_index = thread_position_in_grid.x;
    if (output_index >= uint(COUNT * K)) {
        return;
    }

    uint token_position = output_index / uint(K);
    uint input_index = output_index - token_position * uint(K);
    uint output = uint(token_ids[token_position]);
    uint group = input_index / uint(GS);
    uint element = input_index - group * uint(GS);

    uint metadata_index = output * uint(NG) + group;
    uint quantized_index = metadata_index * uint(GS) + element;
    uint quantized = mfq_nint_read_value(
        q_packed,
        quantized_index,
        uint(BITS),
        uint(GS),
        uint(Q5_EXEC));
    float scale = neuron_scale[output] * float(sub_scale[metadata_index]);
    float minimum = neuron_min[output] * float(sub_min[metadata_index]);
    y[output_index] = T(scale * float(quantized) - minimum);
)METAL";

class BlobCursor {
public:
    explicit BlobCursor(std::span<const std::uint8_t> blob)
        : blob_(blob) {}

    template <typename T>
    T scalar(const char* name) {
        require(sizeof(T), name);
        T value{};
        std::memcpy(&value, blob_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    detail::StagingVector<std::uint8_t> bytes(
        std::size_t count,
        const char* name) {
        require(count, name);
        detail::StagingVector<std::uint8_t> result(
            blob_.begin() + static_cast<std::ptrdiff_t>(offset_),
            blob_.begin() + static_cast<std::ptrdiff_t>(offset_ + count));
        offset_ += count;
        return result;
    }

    std::size_t offset() const noexcept {
        return offset_;
    }
    std::size_t remaining() const noexcept {
        return blob_.size() - offset_;
    }

private:
    void require(std::size_t count, const char* name) {
        if (count > blob_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated NINT ") + name);
        }
    }

    std::span<const std::uint8_t> blob_;
    std::size_t offset_ = 0;
};

std::size_t packed_size(std::size_t count, int bits) {
    if (bits <= 0 || bits > 8 ||
        count > (std::numeric_limits<std::size_t>::max() - 7) /
            static_cast<std::size_t>(bits)) {
        throw std::runtime_error("invalid NINT packed bit count");
    }
    return (count * static_cast<std::size_t>(bits) + 7) / 8;
}

std::uint8_t packed_value(
    std::span<const std::uint8_t> data,
    std::size_t index,
    int bits) {
    const auto bit_index = index * static_cast<std::size_t>(bits);
    const auto byte_index = bit_index / 8;
    const auto shift = static_cast<unsigned>(bit_index & 7);
    std::uint32_t value = data[byte_index];
    if (shift + static_cast<unsigned>(bits) > 8) {
        value |= static_cast<std::uint32_t>(data[byte_index + 1]) << 8;
    }
    return static_cast<std::uint8_t>(
        (value >> shift) & ((1u << bits) - 1u));
}

detail::StagingVector<std::uint8_t> unpack_values(
    std::span<const std::uint8_t> data,
    std::size_t count,
    int bits) {
    detail::StagingVector<std::uint8_t> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        result[index] = packed_value(data, index, bits);
    }
    return result;
}

detail::StagingVector<std::uint8_t> pack_values(
    std::span<const std::uint8_t> values,
    int bits) {
    detail::StagingVector<std::uint8_t> result(
        packed_size(values.size(), bits),
        0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto bit_index = index * static_cast<std::size_t>(bits);
        for (int bit = 0; bit < bits; ++bit) {
            if ((values[index] >> bit) & 1u) {
                const auto target = bit_index + static_cast<std::size_t>(bit);
                result[target / 8] |=
                    static_cast<std::uint8_t>(1u << (target & 7));
            }
        }
    }
    return result;
}

detail::StagingVector<std::uint8_t> pack_q5_execution_layout(
    std::span<const std::uint8_t> values,
    std::size_t rows,
    std::size_t group_size) {
    const auto low_bytes = (group_size + 1) / 2;
    const auto high_bytes = (group_size + 7) / 8;
    detail::StagingVector<std::uint8_t> result(
        rows * (low_bytes + high_bytes),
        0);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto source = row * group_size;
        const auto target = row * (low_bytes + high_bytes);
        for (std::size_t element = 0; element < group_size; ++element) {
            const auto value = values[source + element];
            result[target + element / 2] |= static_cast<std::uint8_t>(
                (value & 15u) << ((element & 1u) * 4u));
            result[target + low_bytes + element / 8] |=
                static_cast<std::uint8_t>(
                    ((value >> 4u) & 1u) << (element & 7u));
        }
    }
    return result;
}

detail::StagingVector<std::uint8_t> read_old_values(
    BlobCursor& cursor,
    std::size_t count,
    int storage_bytes,
    const char* name) {
    detail::StagingVector<std::uint8_t> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        std::uint32_t value = storage_bytes == 1
            ? cursor.scalar<std::uint8_t>(name)
            : cursor.scalar<std::uint16_t>(name);
        if (value > 255) {
            throw std::runtime_error(
                std::string("NINT ") + name +
                " cannot be represented by Metal uint8 metadata");
        }
        result[index] = static_cast<std::uint8_t>(value);
    }
    return result;
}

float half_to_float(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const auto exponent = static_cast<unsigned>((bits >> 10) & 0x1fu);
    const auto mantissa = static_cast<unsigned>(bits & 0x03ffu);
    float value = 0.0f;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa == 0
            ? std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(
            1.0f + static_cast<float>(mantissa) / 1024.0f,
            static_cast<int>(exponent) - 15);
    }
    return negative ? -value : value;
}

template <typename T, typename Allocator>
array make_array(
    const std::vector<T, Allocator>& values,
    Shape shape) {
    return array(values.begin(), std::move(shape));
}

mlx::core::fast::CustomKernelFunction make_nint_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_packed_matmul",
        {
            "q_packed",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
            "x",
        },
        {"y"},
        kNintMatmul,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& nint_kernel() {
    static const auto kernel = make_nint_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_nint4_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint4_packed_matmul",
        {
            "q_packed",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
            "x",
        },
        {"y"},
        kNint4Matmul,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& nint4_kernel() {
    static const auto kernel = make_nint4_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_nint4_swiglu_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint4_packed_swiglu_decode",
        {
            "gate_q",
            "gate_sub_scale",
            "gate_sub_min",
            "gate_neuron_scale",
            "gate_neuron_min",
            "up_q",
            "up_sub_scale",
            "up_sub_min",
            "up_neuron_scale",
            "up_neuron_min",
            "x",
        },
        {"y"},
        kNint4SwiGlu,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
nint4_swiglu_kernel() {
    static const auto kernel = make_nint4_swiglu_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_nint_gemv_fast_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_packed_gemv_fast_v2",
        {
            "q_packed",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
            "x",
        },
        {"y"},
        kNintGemvFast,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
nint_gemv_fast_kernel() {
    static const auto kernel = make_nint_gemv_fast_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_nint5_gs28_gemv_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint5_gs28_packed_gemv",
        {
            "q_packed",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
            "x",
        },
        {"y"},
        kNint5Gs28Gemv,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
nint5_gs28_gemv_kernel() {
    static const auto kernel = make_nint5_gs28_gemv_kernel();
    return kernel;
}

std::string nint4_gs24_gemv_source(
    bool add_residual) {
    std::string source(kNint4Gs24Gemv);
    if (!add_residual) return source;

    constexpr std::string_view assignment =
        "y[output] = T(total);";
    constexpr std::string_view fused_assignment =
        "y[output] = T(float(T(total)) + float(residual[output]));";
    const auto position = source.find(assignment);
    if (position == std::string::npos) {
        throw std::runtime_error(
            "NINT4/GS24 residual fusion source is inconsistent");
    }
    source.replace(position, assignment.size(), fused_assignment);
    return source;
}

mlx::core::fast::CustomKernelFunction make_nint4_gs24_gemv_kernel(
    bool add_residual) {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    std::vector<std::string> inputs{
        "q_packed",
        "sub_scale",
        "sub_min",
        "neuron_scale",
        "neuron_min",
        "x",
    };
    if (add_residual) inputs.emplace_back("residual");
    return mlx::core::fast::metal_kernel(
        add_residual
            ? "mfq_cpp_nint4_gs24_packed_gemv_add_vec"
            : "mfq_cpp_nint4_gs24_packed_gemv_vec",
        std::move(inputs),
        {"y"},
        nint4_gs24_gemv_source(add_residual),
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
nint4_gs24_gemv_kernel(bool add_residual = false) {
    static const auto kernel = make_nint4_gs24_gemv_kernel(false);
    static const auto add_kernel = make_nint4_gs24_gemv_kernel(true);
    return add_residual ? add_kernel : kernel;
}

std::string fp16_shape_defines(
    int groups,
    int group_size,
    int input_size,
    int output_size) {
    return "#define T half\n#define NG " + std::to_string(groups) +
        "\n#define GS " + std::to_string(group_size) +
        "\n#define K " + std::to_string(input_size) +
        "\n#define OUT " + std::to_string(output_size) + "\n";
}

const mlx::core::fast::CustomKernelFunction&
specialized_nint4_gs24_gemv_kernel(
    int groups,
    int input_size,
    int output_size,
    bool add_residual) {
    using Kernel = mlx::core::fast::CustomKernelFunction;
    struct LocalEntry {
        int groups;
        int input_size;
        int output_size;
        bool add_residual;
        const Kernel* kernel;
    };
    thread_local std::vector<LocalEntry> local_cache;
    for (const auto& entry : local_cache) {
        if (entry.groups == groups &&
            entry.input_size == input_size &&
            entry.output_size == output_size &&
            entry.add_residual == add_residual) {
            return *entry.kernel;
        }
    }
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        mlx::core::fast::CustomKernelFunction> kernels;
    const auto key = std::to_string(groups) + "_" +
        std::to_string(input_size) + "_" +
        std::to_string(output_size) +
        (add_residual ? "_add" : "");
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = kernels.find(key); found != kernels.end()) {
        local_cache.push_back({
            groups, input_size, output_size, add_residual, &found->second});
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    std::vector<std::string> inputs{
        "q_packed",
        "sub_scale",
        "sub_min",
        "neuron_scale",
        "neuron_min",
        "x",
    };
    if (add_residual) inputs.emplace_back("residual");
    auto header = fp16_shape_defines(groups, 24, input_size, output_size) +
        kNintHeader;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_nint4_gs24_shape_v6_" + key,
        std::move(inputs),
        {"y"},
        nint4_gs24_gemv_source(add_residual),
        std::move(header),
        true,
        false,
        options);
    const auto [inserted, unused] = kernels.emplace(key, std::move(kernel));
    (void)unused;
    local_cache.push_back({
        groups, input_size, output_size, add_residual, &inserted->second});
    return inserted->second;
}

const mlx::core::fast::CustomKernelFunction&
specialized_nint4_swiglu_kernel(
    int groups,
    int group_size,
    int input_size,
    int output_size) {
    using Kernel = mlx::core::fast::CustomKernelFunction;
    struct LocalEntry {
        int groups;
        int group_size;
        int input_size;
        int output_size;
        const Kernel* kernel;
    };
    thread_local std::vector<LocalEntry> local_cache;
    for (const auto& entry : local_cache) {
        if (entry.groups == groups &&
            entry.group_size == group_size &&
            entry.input_size == input_size &&
            entry.output_size == output_size) {
            return *entry.kernel;
        }
    }
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        mlx::core::fast::CustomKernelFunction> kernels;
    const auto key = std::to_string(groups) + "_" +
        std::to_string(group_size) + "_" +
        std::to_string(input_size) + "_" +
        std::to_string(output_size);
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = kernels.find(key); found != kernels.end()) {
        local_cache.push_back({
            groups,
            group_size,
            input_size,
            output_size,
            &found->second,
        });
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto header = fp16_shape_defines(
        groups, group_size, input_size, output_size) +
        kNintHeader;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_nint4_swiglu_shape_v10_" + key,
        {
            "gate_q",
            "gate_sub_scale",
            "gate_sub_min",
            "gate_neuron_scale",
            "gate_neuron_min",
            "up_q",
            "up_sub_scale",
            "up_sub_min",
            "up_neuron_scale",
            "up_neuron_min",
            "x",
        },
        {"y"},
        kNint4SwiGlu,
        std::move(header),
        true,
        false,
        options);
    const auto [inserted, unused] = kernels.emplace(key, std::move(kernel));
    (void)unused;
    local_cache.push_back({
        groups,
        group_size,
        input_size,
        output_size,
        &inserted->second,
    });
    return inserted->second;
}

mlx::core::fast::CustomKernelFunction make_nint6_gs24_gemv_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint6_gs24_packed_gemv",
        {
            "q_packed",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
            "x",
        },
        {"y"},
        kNint6Gs24Gemv,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
nint6_gs24_gemv_kernel() {
    static const auto kernel = make_nint6_gs24_gemv_kernel();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
specialized_nint6_gs24_gemv_kernel(
    int groups,
    int input_size,
    int output_size) {
    using Kernel = mlx::core::fast::CustomKernelFunction;
    struct LocalEntry {
        int groups;
        int input_size;
        int output_size;
        const Kernel* kernel;
    };
    thread_local std::vector<LocalEntry> local_cache;
    for (const auto& entry : local_cache) {
        if (entry.groups == groups &&
            entry.input_size == input_size &&
            entry.output_size == output_size) {
            return *entry.kernel;
        }
    }
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        mlx::core::fast::CustomKernelFunction> kernels;
    const auto key = std::to_string(groups) + "_" +
        std::to_string(input_size) + "_" +
        std::to_string(output_size);
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = kernels.find(key); found != kernels.end()) {
        local_cache.push_back({
            groups, input_size, output_size, &found->second});
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto header = fp16_shape_defines(groups, 24, input_size, output_size) +
        kNintHeader;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_nint6_gs24_shape_" + key,
        {
            "q_packed",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
            "x",
        },
        {"y"},
        kNint6Gs24Gemv,
        std::move(header),
        true,
        false,
        options);
    const auto [inserted, unused] = kernels.emplace(key, std::move(kernel));
    (void)unused;
    local_cache.push_back({
        groups, input_size, output_size, &inserted->second});
    return inserted->second;
}

mlx::core::fast::CustomKernelFunction
make_nint6_gs24_greedy_partial_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint6_gs24_greedy_partial",
        {
            "q_packed",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
            "x",
        },
        {"partial_values", "partial_indices"},
        kNint6Gs24GreedyPartial,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
nint6_gs24_greedy_partial_kernel() {
    static const auto kernel = make_nint6_gs24_greedy_partial_kernel();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
specialized_nint6_gs24_greedy_partial_kernel(
    int groups,
    int input_size,
    int output_size) {
    using Kernel = mlx::core::fast::CustomKernelFunction;
    struct LocalEntry {
        int groups;
        int input_size;
        int output_size;
        const Kernel* kernel;
    };
    thread_local std::vector<LocalEntry> local_cache;
    for (const auto& entry : local_cache) {
        if (entry.groups == groups &&
            entry.input_size == input_size &&
            entry.output_size == output_size) {
            return *entry.kernel;
        }
    }
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        mlx::core::fast::CustomKernelFunction> kernels;
    const auto key = std::to_string(groups) + "_" +
        std::to_string(input_size) + "_" +
        std::to_string(output_size);
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = kernels.find(key); found != kernels.end()) {
        local_cache.push_back({
            groups, input_size, output_size, &found->second});
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto header = fp16_shape_defines(groups, 24, input_size, output_size) +
        kNintHeader;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_nint6_greedy_shape_" + key,
        {
            "q_packed",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
            "x",
        },
        {"partial_values", "partial_indices"},
        kNint6Gs24GreedyPartial,
        std::move(header),
        true,
        false,
        options);
    const auto [inserted, unused] = kernels.emplace(key, std::move(kernel));
    (void)unused;
    local_cache.push_back({
        groups, input_size, output_size, &inserted->second});
    return inserted->second;
}

mlx::core::fast::CustomKernelFunction make_nint_greedy_reduce_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_greedy_reduce",
        {"partial_values", "partial_indices"},
        {"output"},
        kNintGreedyReduce,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
nint_greedy_reduce_kernel() {
    static const auto kernel = make_nint_greedy_reduce_kernel();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
specialized_nint_greedy_reduce_kernel(
    int output_size,
    int parts) {
    using Kernel = mlx::core::fast::CustomKernelFunction;
    struct LocalEntry {
        int output_size;
        int parts;
        const Kernel* kernel;
    };
    thread_local std::vector<LocalEntry> local_cache;
    for (const auto& entry : local_cache) {
        if (entry.output_size == output_size && entry.parts == parts) {
            return *entry.kernel;
        }
    }
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        mlx::core::fast::CustomKernelFunction> kernels;
    const auto key = std::to_string(output_size) + "_" +
        std::to_string(parts);
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = kernels.find(key); found != kernels.end()) {
        local_cache.push_back({output_size, parts, &found->second});
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    const auto header = "#define OUT " + std::to_string(output_size) +
        "\n#define PARTS " + std::to_string(parts) + "\n";
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_greedy_reduce_shape_" + key,
        {"partial_values", "partial_indices"},
        {"output"},
        kNintGreedyReduce,
        header,
        true,
        false,
        options);
    const auto [inserted, unused] = kernels.emplace(key, std::move(kernel));
    (void)unused;
    local_cache.push_back({output_size, parts, &inserted->second});
    return inserted->second;
}

mlx::core::fast::CustomKernelFunction make_nint_embedding_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_packed_embedding",
        {
            "q_packed",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
            "token_ids",
        },
        {"y"},
        kNintEmbedding,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& nint_embedding_kernel() {
    static const auto kernel = make_nint_embedding_kernel();
    return kernel;
}

std::int32_t checked_shape(std::int64_t value, const char* name) {
    if (value <= 0 ||
        value > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error(
            std::string("invalid NINT ") + name);
    }
    return static_cast<std::int32_t>(value);
}

} // namespace

bool is_nint_dtype(std::string_view dtype) noexcept {
    if (dtype == "NINT") {
        return true;
    }
    return dtype.size() == 5 &&
        dtype.substr(0, 4) == "NINT" &&
        dtype[4] >= '1' &&
        dtype[4] <= '8';
}

MlxNintWeight::MlxNintWeight(
    array q_packed,
    array sub_scale,
    array sub_min,
    array neuron_scale,
    array neuron_min,
    int bits,
    int group_size,
    int groups,
    int input_size,
    int output_size,
    bool q5_execution_layout)
    : q_packed_(std::move(q_packed)),
      sub_scale_(std::move(sub_scale)),
      sub_min_(std::move(sub_min)),
      neuron_scale_(std::move(neuron_scale)),
      neuron_min_(std::move(neuron_min)),
      bits_(bits),
      group_size_(group_size),
      groups_(groups),
      input_size_(input_size),
      output_size_(output_size),
      q5_execution_layout_(q5_execution_layout) {}

MlxNintWeight MlxNintWeight::from_blob(
    std::span<const std::uint8_t> blob) {
    BlobCursor cursor(blob);
    const auto bits = static_cast<int>(cursor.scalar<std::uint8_t>("bits"));
    const auto sub_bits =
        static_cast<int>(cursor.scalar<std::uint8_t>("sub bits"));
    const auto group_size = cursor.scalar<std::int32_t>("group size");
    const auto axis = cursor.scalar<std::int32_t>("axis");
    const auto input_size = cursor.scalar<std::int32_t>("input size");
    const auto dimensions = cursor.scalar<std::uint32_t>("dimension count");
    if (bits <= 0 || bits > 8 || sub_bits <= 0 || sub_bits > 8 ||
        group_size <= 0 || input_size <= 0 ||
        dimensions != 2 || axis != 0) {
        throw std::runtime_error("unsupported NINT Metal dimensions");
    }

    std::vector<std::int64_t> shape(dimensions);
    for (auto& value : shape) {
        value = cursor.scalar<std::int64_t>("shape");
    }
    const auto output_size =
        cursor.scalar<std::uint32_t>("output size");
    const auto groups = cursor.scalar<std::uint32_t>("group count");
    if (shape[0] != output_size ||
        shape[1] != input_size ||
        groups == 0 ||
        static_cast<std::uint64_t>(input_size) >
            static_cast<std::uint64_t>(groups) *
                static_cast<std::uint64_t>(group_size)) {
        throw std::runtime_error("inconsistent NINT Metal dimensions");
    }

    detail::StagingVector<float> neuron_scale(output_size);
    detail::StagingVector<float> neuron_min(output_size);
    for (auto& value : neuron_scale) {
        value = half_to_float(
            cursor.scalar<std::uint16_t>("neuron scale"));
    }
    for (auto& value : neuron_min) {
        value = half_to_float(
            cursor.scalar<std::uint16_t>("neuron minimum"));
    }

    const auto metadata_count =
        static_cast<std::size_t>(output_size) * groups;
    const auto q_count =
        metadata_count * static_cast<std::size_t>(group_size);
    const auto packed_metadata_bytes =
        packed_size(metadata_count, sub_bits);
    const auto packed_q_bytes = packed_size(q_count, bits);
    const auto packed_tail =
        2 * packed_metadata_bytes + packed_q_bytes;
    const int old_sub_bytes = sub_bits <= 8 ? 1 : 2;
    const int old_q_bytes = bits <= 8 ? 1 : 2;
    const auto old_tail =
        2 * metadata_count * static_cast<std::size_t>(old_sub_bytes) +
        q_count * static_cast<std::size_t>(old_q_bytes);

    detail::StagingVector<std::uint8_t> sub_scale;
    detail::StagingVector<std::uint8_t> sub_min;
    detail::StagingVector<std::uint8_t> q_values;
    detail::StagingVector<std::uint8_t> q_packed;
    if (cursor.remaining() == packed_tail) {
        sub_scale = unpack_values(
            cursor.bytes(packed_metadata_bytes, "sub scale"),
            metadata_count,
            sub_bits);
        sub_min = unpack_values(
            cursor.bytes(packed_metadata_bytes, "sub minimum"),
            metadata_count,
            sub_bits);
        q_packed = cursor.bytes(packed_q_bytes, "packed values");
        if (bits == 5) {
            q_values = unpack_values(q_packed, q_count, bits);
        }
    } else if (cursor.remaining() == old_tail) {
        sub_scale = read_old_values(
            cursor,
            metadata_count,
            old_sub_bytes,
            "sub scale");
        sub_min = read_old_values(
            cursor,
            metadata_count,
            old_sub_bytes,
            "sub minimum");
        q_values = read_old_values(
            cursor,
            q_count,
            old_q_bytes,
            "quantized value");
        q_packed = pack_values(q_values, bits);
    } else {
        throw std::runtime_error("invalid NINT packed payload length");
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error("trailing bytes in NINT tensor");
    }

    const bool q5_execution_layout = bits == 5;
    if (q5_execution_layout) {
        q_packed = pack_q5_execution_layout(
            q_values,
            metadata_count,
            static_cast<std::size_t>(group_size));
    }

    return MlxNintWeight(
        make_array(
            q_packed,
            Shape{checked_shape(q_packed.size(), "packed byte count")}),
        make_array(
            sub_scale,
            Shape{
                checked_shape(output_size, "output size"),
                checked_shape(groups, "group count"),
            }),
        make_array(
            sub_min,
            Shape{
                checked_shape(output_size, "output size"),
                checked_shape(groups, "group count"),
            }),
        make_array(
            neuron_scale,
            Shape{checked_shape(output_size, "output size")}),
        make_array(
            neuron_min,
            Shape{checked_shape(output_size, "output size")}),
        bits,
        group_size,
        static_cast<int>(groups),
        input_size,
        static_cast<int>(output_size),
        q5_execution_layout);
}

array MlxNintWeight::matmul(const array& input) const {
    return matmul_impl(input, nullptr);
}

array MlxNintWeight::matmul_add(
    const array& input,
    const array& residual) const {
    return matmul_impl(input, &residual);
}

std::optional<array> MlxNintWeight::greedy_argmax(
    const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error(
            "NINT greedy input width does not match packed weight");
    }
    std::int64_t rows = 1;
    for (std::size_t index = 0; index + 1 < input.ndim(); ++index) {
        rows *= input.shape(static_cast<int>(index));
    }
    if (rows != 1 || bits_ != 6 || group_size_ != 24 ||
        q5_execution_layout_) {
        return std::nullopt;
    }

    auto source = input;
    if (source.dtype() != mlx::core::float16) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::contiguous(mlx::core::reshape(
        source, Shape{1, input_size_}));
    const int parts = (output_size_ + 15) / 16;
    auto partials = specialized_nint6_gs24_greedy_partial_kernel(
        groups_, input_size_, output_size_)(
        {
            q_packed_,
            sub_scale_,
            sub_min_,
            neuron_scale_,
            neuron_min_,
            source,
        },
        {Shape{parts}, Shape{parts}},
        {mlx::core::float32, mlx::core::int32},
        {parts * 256, 1, 1},
        {256, 1, 1},
        {},
        std::nullopt,
        false,
        {});
    auto output = specialized_nint_greedy_reduce_kernel(
        output_size_, parts)(
        {partials.at(0), partials.at(1)},
        {Shape{1}},
        {mlx::core::int32},
        {256, 1, 1},
        {256, 1, 1},
        {},
        std::nullopt,
        false,
        {});
    return output.front();
}

array MlxNintWeight::matmul_impl(
    const array& input,
    const array* residual) const {
    if (input.ndim() == 0 ||
        input.shape(-1) != input_size_) {
        throw std::runtime_error(
            "NINT input width does not match packed weight");
    }
    std::int64_t rows = 1;
    Shape output_shape = input.shape();
    for (std::size_t index = 0; index + 1 < input.ndim(); ++index) {
        rows *= input.shape(static_cast<int>(index));
    }
    if (rows <= 0 ||
        rows > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("unsupported NINT input row count");
    }
    output_shape.back() = output_size_;
    if (residual != nullptr && residual->shape() != output_shape) {
        throw std::runtime_error(
            "NINT residual shape does not match matmul output");
    }

    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::reshape(
        source,
        Shape{
            static_cast<std::int32_t>(rows),
            input_size_,
        });
    if (rows >= 64 &&
        source.dtype() == mlx::core::float16) {
        const auto token_ids = mlx::core::arange(
            output_size_,
            mlx::core::int32);
        const auto dense = embedding(
            token_ids,
            mlx::core::float16);
        auto result = mlx::core::matmul(
            source,
            mlx::core::transpose(dense));
        result = mlx::core::reshape(
            std::move(result),
            std::move(output_shape));
        return residual != nullptr ? result + *residual : result;
    }

    const int tile_rows =
        rows == 1 ? 1 : (rows <= 16 ? static_cast<int>(rows) : 8);
    const auto row_tiles =
        (rows + tile_rows - 1) / tile_rows;
    const bool use_nint4_gs24_gemv =
        rows == 1 &&
        bits_ == 4 &&
        group_size_ == 24 &&
        !q5_execution_layout_ &&
        source.dtype() == mlx::core::float16;
    const bool fuse_residual =
        residual != nullptr &&
        use_nint4_gs24_gemv &&
        residual->dtype() == source.dtype();
    const bool use_nint4_gemv =
        rows == 1 &&
        bits_ == 4 &&
        !use_nint4_gs24_gemv;
    const bool use_nint5_gs28_gemv =
        rows == 1 &&
        bits_ == 5 &&
        group_size_ == 28 &&
        q5_execution_layout_ &&
        source.dtype() == mlx::core::float16;
    const bool use_nint6_gs24_gemv =
        rows == 1 &&
        bits_ == 6 &&
        group_size_ == 24 &&
        !q5_execution_layout_ &&
        source.dtype() == mlx::core::float16;
    const bool use_fast_gemv =
        rows == 1 &&
        bits_ != 4 &&
        !use_nint5_gs28_gemv &&
        !use_nint6_gs24_gemv;
    const auto grid_x = use_nint4_gs24_gemv
        ? static_cast<std::int64_t>(
              (output_size_ + 15) / 16) * 256
        : (use_nint5_gs28_gemv
               ? static_cast<std::int64_t>(
                     (output_size_ + 15) / 16) * 128
               : (use_nint6_gs24_gemv
                      ? static_cast<std::int64_t>(
                            (output_size_ + 15) / 16) * 256
                      : (use_fast_gemv
                             ? static_cast<std::int64_t>(
                                   (output_size_ + 7) / 8) * 64
                             : row_tiles *
                                   static_cast<std::int64_t>(
                                       output_size_) * 32)));
    if (grid_x > std::numeric_limits<int>::max()) {
        throw std::runtime_error("NINT Metal grid exceeds MLX limits");
    }

    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
        templates{
            {"T", source.dtype()},
            {"BITS", bits_},
            {"GS", group_size_},
            {"NG", groups_},
            {"K", input_size_},
            {"OUT", output_size_},
            {"M", static_cast<int>(rows)},
            {"TILE_M", tile_rows},
            {"Q5_EXEC", static_cast<int>(q5_execution_layout_)},
        };
    const mlx::core::fast::CustomKernelFunction* kernel = nullptr;
    if (use_nint4_gs24_gemv) {
        kernel = &specialized_nint4_gs24_gemv_kernel(
            groups_, input_size_, output_size_, fuse_residual);
    } else if (use_nint5_gs28_gemv) {
        kernel = &nint5_gs28_gemv_kernel();
    } else if (use_nint6_gs24_gemv) {
        kernel = &specialized_nint6_gs24_gemv_kernel(
            groups_, input_size_, output_size_);
    } else if (use_nint4_gemv) {
        kernel = &nint4_kernel();
    } else if (use_fast_gemv) {
        kernel = &nint_gemv_fast_kernel();
    } else {
        kernel = &nint_kernel();
    }
    if (use_nint4_gs24_gemv || use_nint6_gs24_gemv) {
        templates.clear();
    }
    const int threadgroup =
        use_nint4_gs24_gemv
        ? 256
        : (use_nint5_gs28_gemv
               ? 128
               : (use_nint6_gs24_gemv
                      ? 256
                      : (use_fast_gemv ? 64 : 32)));
    std::vector<array> inputs{
        q_packed_,
        sub_scale_,
        sub_min_,
        neuron_scale_,
        neuron_min_,
        source,
    };
    if (fuse_residual) {
        inputs.push_back(mlx::core::contiguous(
            mlx::core::reshape(
                *residual,
                Shape{
                    static_cast<std::int32_t>(rows),
                    output_size_,
                })));
    }
    auto outputs = (*kernel)(
        std::move(inputs),
        {
            Shape{
                static_cast<std::int32_t>(rows),
                output_size_,
            },
        },
        {source.dtype()},
        {static_cast<int>(grid_x), 1, 1},
        {threadgroup, 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {});
    auto result = mlx::core::reshape(
        outputs.front(),
        std::move(output_shape));
    return residual != nullptr && !fuse_residual
        ? result + *residual
        : result;
}

bool MlxNintWeight::can_fuse_swiglu(
    const MlxNintWeight& up) const noexcept {
    // The specialized kernel reads two packed nibbles per byte. Requiring an
    // even GS keeps every quantization-group boundary byte aligned. NINT5's
    // execution layout and other bit widths keep their established GEMV path.
    return bits_ == 4 &&
        up.bits_ == 4 &&
        group_size_ > 0 &&
        (group_size_ % 2) == 0 &&
        group_size_ == up.group_size_ &&
        groups_ == up.groups_ &&
        input_size_ == up.input_size_ &&
        output_size_ == up.output_size_ &&
        !q5_execution_layout_ &&
        !up.q5_execution_layout_;
}

array MlxNintWeight::swiglu(
    const MlxNintWeight& up,
    const array& input) const {
    if (!can_fuse_swiglu(up)) {
        throw std::runtime_error(
            "fused NINT SwiGLU requires compatible even-GS NINT4 weights");
    }
    if (input.ndim() == 0 ||
        input.shape(-1) != input_size_) {
        throw std::runtime_error(
            "fused NINT SwiGLU input width mismatch");
    }
    const auto rows =
        input.size() / static_cast<std::size_t>(input_size_);
    if (rows != 1) {
        throw std::runtime_error(
            "fused NINT SwiGLU is restricted to one decode row");
    }

    Shape output_shape = input.shape();
    output_shape.back() = output_size_;
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::reshape(
        source,
        Shape{1, input_size_});

    constexpr int outputs_per_threadgroup = 16;
    const auto threadgroups =
        (static_cast<std::int64_t>(output_size_) +
         outputs_per_threadgroup - 1) /
        outputs_per_threadgroup;
    const auto grid_x = threadgroups * 256;
    if (grid_x > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "fused NINT SwiGLU Metal grid exceeds MLX limits");
    }

    auto outputs = specialized_nint4_swiglu_kernel(
        groups_,
        group_size_,
        input_size_,
        output_size_)(
        {
            q_packed_,
            sub_scale_,
            sub_min_,
            neuron_scale_,
            neuron_min_,
            up.q_packed_,
            up.sub_scale_,
            up.sub_min_,
            up.neuron_scale_,
            up.neuron_min_,
            source,
        },
        {
            Shape{1, output_size_},
        },
        {source.dtype()},
        {static_cast<int>(grid_x), 1, 1},
        {256, 1, 1},
        {},
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        outputs.front(),
        std::move(output_shape));
}

array MlxNintWeight::embedding(
    const array& token_ids,
    Dtype dtype) const {
    if (dtype != mlx::core::float16 &&
        dtype != mlx::core::float32) {
        throw std::runtime_error(
            "NINT embedding output must be float16 or float32");
    }
    auto ids = token_ids;
    if (ids.dtype() != mlx::core::int32 &&
        ids.dtype() != mlx::core::uint32) {
        ids = mlx::core::astype(ids, mlx::core::int32);
    }
    const auto count = ids.size();
    Shape output_shape = ids.shape();
    output_shape.push_back(input_size_);
    if (count == 0) {
        return mlx::core::zeros(output_shape, dtype);
    }
    if (count >
            static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max()) ||
        count > static_cast<std::size_t>(
                    std::numeric_limits<int>::max() / input_size_)) {
        throw std::runtime_error("NINT embedding input is too large");
    }
    ids = mlx::core::reshape(
        ids,
        Shape{static_cast<std::int32_t>(count)});
    const int grid = static_cast<int>(count) * input_size_;
    const int threadgroup = std::min(256, std::max(1, grid));
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
        templates{
            {"T", dtype},
            {"BITS", bits_},
            {"GS", group_size_},
            {"NG", groups_},
            {"K", input_size_},
            {"COUNT", static_cast<int>(count)},
            {"Q5_EXEC", static_cast<int>(q5_execution_layout_)},
        };
    auto outputs = nint_embedding_kernel()(
        {
            q_packed_,
            sub_scale_,
            sub_min_,
            neuron_scale_,
            neuron_min_,
            ids,
        },
        {
            Shape{
                static_cast<std::int32_t>(count),
                input_size_,
            },
        },
        {dtype},
        {grid, 1, 1},
        {threadgroup, 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(outputs.front(), std::move(output_shape));
}

std::size_t MlxNintWeight::packed_nbytes() const noexcept {
    return q_packed_.nbytes() +
        sub_scale_.nbytes() +
        sub_min_.nbytes() +
        neuron_scale_.nbytes() +
        neuron_min_.nbytes();
}

} // namespace mfq::metal
