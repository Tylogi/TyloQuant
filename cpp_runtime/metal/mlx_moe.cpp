#include "mlx_moe.h"

#include "mfq_container.h"
#include "mfq_nintm_prefill_embedded.h"
#include "mlx_nint.h"
#include "mlx_nint8_zero.h"
#include "mlx_staging_allocator.h"
#include "mlx_vq.h"

#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/memory.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kDescriptorSize = 32;

constexpr int kFamily = 0;
constexpr int kLocalExpert = 1;
constexpr int kOut = 2;
constexpr int kInput = 3;

constexpr int kFamilyNint = 0;
constexpr int kNintBits = 4;
constexpr int kNintGroupSize = 5;
constexpr int kNintGroups = 6;
constexpr int kNintQOffset = 7;
constexpr int kNintSubOffset = 8;
constexpr int kNintAnchorOffset = 9;
constexpr int kNintQ5Execution = 10;

// Keep the same family value and fields as the Python Metal descriptor.  The
// unused value 1 remains available for the VQ-family extension.
constexpr int kFamilyNint8Zero = 2;
constexpr int kQ8Groups = 4;
constexpr int kQ8QOffset = 5;
constexpr int kQ8ScaleOffset = 6;

constexpr int kFamilyVq = 1;
constexpr int kVqGroupSize = 4;
constexpr int kVqGroups = 5;
constexpr int kVqVectorSize = 6;
constexpr int kVqVectors = 7;
constexpr int kVqIndexBits = 8;
constexpr int kVqStateBits = 9;
constexpr int kVqStates = 10;
constexpr int kVqEntries = 11;
constexpr int kVqCodeBanks = 12;
constexpr int kVqAuxMode = 13;
constexpr int kVqCodeBankMode = 14;
constexpr int kVqHasTableBanks = 15;
constexpr int kVqGroupsPerSuper = 16;
constexpr int kVqSupergroups = 17;
constexpr int kVqIndicesOffset = 18;
constexpr int kVqStateOffset = 19;
constexpr int kVqAuxOffset = 20;
constexpr int kVqAnchorOffset = 21;
constexpr int kVqCodebookOffset = 22;
constexpr int kVqScaleOffset = 23;
constexpr int kVqStateBankOffset = 24;
constexpr int kVqBankOffset = 25;
constexpr int kVqParameterOffset = 26;
constexpr int kVqRotationVariant = 27;
constexpr int kVqProfile = 28;
constexpr int kVqJscExecution = 29;
constexpr int kVqProfileGeneric = 0;
constexpr int kVqProfileJsc4 = 1;
constexpr int kVqProfileNpqS = 2;
constexpr int kVqProfileNvq1L = 3;
constexpr int kVqProfileJsc8 = 4;
constexpr int kVqProfileNpqL = 5;

constexpr const char* kMoeHeader = R"METAL(
template <typename Stream>
inline uint mfq_moe_read_bits(
    Stream stream,
    uint value_index,
    uint bits
) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3) * bits + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1u]) << 8;
    }
    if (shift + bits > 16u) {
        packed |= uint(stream[byte_index + 2u]) << 16;
    }
    return (packed >> shift)
        & ((1u << bits) - 1u);
}

template <uint BITS, typename Stream>
inline uint3 mfq_moe_read_npq_group_indices(
    Stream stream,
    uint row,
    uint group,
    uint vectors
) {
    const uint first = group * 3u;
    if (first + 2u >= vectors) {
        return uint3(
            first < vectors
                ? mfq_moe_read_bits(
                      stream,
                      row * vectors + first,
                      BITS)
                : 0u,
            first + 1u < vectors
                ? mfq_moe_read_bits(
                      stream,
                      row * vectors + first + 1u,
                      BITS)
                : 0u,
            0u);
    }
    const uint linear = row * vectors + first;
    const uint bit = linear * BITS;
    const uint byte = bit >> 3;
    const uint shift = bit & 7u;
    uint packed = uint(stream[byte])
        | (uint(stream[byte + 1u]) << 8)
        | (uint(stream[byte + 2u]) << 16);
    if (shift + 3u * BITS > 24u) {
        packed |= uint(stream[byte + 3u]) << 24;
    }
    const uint mask = (1u << BITS) - 1u;
    return uint3(
        (packed >> shift) & mask,
        (packed >> (shift + BITS)) & mask,
        (packed >> (shift + 2u * BITS)) & mask);
}

template <typename Stream>
inline uint mfq_moe_nint_read_bits(
    Stream stream,
    uint value_index,
    uint bits
) {
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
inline uint mfq_moe_nint_read_value(
    Stream stream,
    uint value_index,
    uint bits,
    uint group_size,
    uint q5_execution
) {
    if (q5_execution != 0u && bits == 5u) {
        uint metadata_index = value_index / group_size;
        uint element = value_index - metadata_index * group_size;
        uint low_bytes = (group_size + 1u) >> 1;
        uint high_bytes = (group_size + 7u) >> 3;
        uint group_offset =
            metadata_index * (low_bytes + high_bytes);
        uint low_packed =
            uint(stream[group_offset + (element >> 1)]);
        uint low =
            (low_packed >> ((element & 1u) * 4u)) & 15u;
        uint high = (
            uint(stream[
                group_offset + low_bytes + (element >> 3)])
            >> (element & 7u)
        ) & 1u;
        return low | (high << 4u);
    }
    return mfq_moe_nint_read_bits(
        stream,
        value_index,
        bits);
}

inline float4 mfq_moe_load_code4(
    device const int8_t* stream,
    uint offset
) {
    return float4(
        *(device const char4*)(stream + offset));
}

inline float4 mfq_moe_load_code4(
    constant const int8_t* stream,
    uint offset
) {
    return float4(
        *(constant const char4*)(stream + offset));
}

template <
    uint VECTOR_SIZE,
    uint MATRIX_ROWS,
    uint EXECUTION_LAYOUT,
    typename XStream,
    typename IndexStream,
    typename StateStream,
    typename AuxStream,
    typename ScaleStream,
    typename StateBankStream,
    typename CodebookStream
>
inline void mfq_moe_jsc_profile(
    XStream x,
    IndexStream indices_stream,
    StateStream state_stream,
    AuxStream aux_stream,
    ScaleStream scale_stream,
    StateBankStream state_bank_stream,
    CodebookStream codebook_stream,
    uint x_offset,
    thread const uint* outputs,
    thread const float* row_anchors,
    thread float* accumulators,
    uint groups,
    uint vectors,
    uint indices_offset,
    uint state_offset,
    uint aux_offset,
    uint codebook_offset,
    uint scale_offset,
    uint state_bank_offset,
    uint signs,
    uint k_lane,
    uint k_lanes,
    uint k_size
) {
    constexpr uint VECTOR_SHIFT =
        VECTOR_SIZE == 4u ? 2u : 3u;
    constexpr uint BYTES_PER_SIGN =
        VECTOR_SIZE == 4u ? 3u : 2u;
    for (
        uint group = k_lane;
        group < groups;
        group += k_lanes
    ) {
        uint selected_code_banks[MATRIX_ROWS];
        float weight_scales[MATRIX_ROWS];
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            uint state_index = outputs[row] * groups + group;
            uint state_byte = uint(state_stream[
                state_offset + (state_index >> 1)
            ]);
            uint state = (
                state_byte >> ((state_index & 1u) * 4u)
            ) & 15u;
            selected_code_banks[row] = uint(
                state_bank_stream[state_bank_offset + state]);
            weight_scales[row] =
                row_anchors[row]
                * scale_stream[scale_offset + state];
        }
        for (
            uint sign_block = 0u;
            sign_block < 3u;
            ++sign_block
        ) {
            uint column_base = group * 24u + sign_block * 8u;
            if (column_base >= k_size) {
                break;
            }
            float4 activation0 = float4(
                float(x[x_offset + column_base]),
                float(x[x_offset + column_base + 1u]),
                float(x[x_offset + column_base + 2u]),
                float(x[x_offset + column_base + 3u]));
            float4 activation1 = float4(
                float(x[x_offset + column_base + 4u]),
                float(x[x_offset + column_base + 5u]),
                float(x[x_offset + column_base + 6u]),
                float(x[x_offset + column_base + 7u]));
            uint first_vector = column_base >> VECTOR_SHIFT;
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                uint index0;
                uint index1 = 0u;
                uint sign_value;
                if (EXECUTION_LAYOUT != 0u) {
                    uint execution_offset = indices_offset + (
                        outputs[row] * signs + column_base / 8u
                    ) * BYTES_PER_SIGN;
                    index0 = uint(indices_stream[execution_offset]);
                    if (VECTOR_SIZE == 4u) {
                        index1 = uint(indices_stream[
                            execution_offset + 1u]);
                    }
                    sign_value = uint(indices_stream[
                        execution_offset + BYTES_PER_SIGN - 1u]);
                } else {
                    index0 = uint(indices_stream[
                        indices_offset
                        + outputs[row] * vectors
                        + first_vector]);
                    if (VECTOR_SIZE == 4u) {
                        index1 = uint(indices_stream[
                            indices_offset
                            + outputs[row] * vectors
                            + first_vector + 1u]);
                    }
                    sign_value = mfq_moe_read_bits(
                        aux_stream + aux_offset,
                        outputs[row] * signs + column_base / 8u,
                        7u);
                }
                uint bank_base = selected_code_banks[row] * 256u;
                uint code0_offset =
                    (bank_base + index0) * VECTOR_SIZE;
                uint code1_offset = VECTOR_SIZE == 4u
                    ? (bank_base + index1) * 4u
                    : code0_offset + 4u;
                bool4 negative0 = bool4(
                    (sign_value & 1u) != 0u,
                    (sign_value & 2u) != 0u,
                    (sign_value & 4u) != 0u,
                    (sign_value & 8u) != 0u);
                bool4 negative1 = bool4(
                    (sign_value & 16u) != 0u,
                    (sign_value & 32u) != 0u,
                    (sign_value & 64u) != 0u,
                    EXECUTION_LAYOUT != 0u
                        ? (sign_value & 128u) != 0u
                        : (popcount(sign_value) & 1u) != 0u);
                float4 code0 = mfq_moe_load_code4(
                    codebook_stream,
                    codebook_offset + code0_offset);
                float4 code1 = mfq_moe_load_code4(
                    codebook_stream,
                    codebook_offset + code1_offset);
                code0 = select(code0, -code0, negative0);
                code1 = select(code1, -code1, negative1);
                float code_dot =
                    dot(activation0, code0)
                    + dot(activation1, code1);
                accumulators[row] = fma(
                    weight_scales[row],
                    code_dot,
                    accumulators[row]);
            }
        }
    }
}
)METAL";

constexpr const char* kMoeSource = R"METAL(
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint K_LANES = uint(K_LANES_VALUE);
    constexpr uint LANE_GROUPS = 32u / K_LANES;
    constexpr uint ROWS_PER_SIMD = 1u;
    constexpr uint MATRIX_ROWS =
        uint(FUSED_SWIGLU) != 0u
            ? 2u * ROWS_PER_SIMD
            : ROWS_PER_SIMD;
    constexpr uint ROWS_PER_PHYSICAL_SIMD =
        LANE_GROUPS * ROWS_PER_SIMD;
    constexpr uint ROWS_PER_TG =
        SIMD_GROUPS * ROWS_PER_PHYSICAL_SIMD;
    constexpr uint OUTPUT_TILES =
        (uint(OUT) + ROWS_PER_TG - 1u) / ROWS_PER_TG;

    uint lane = thread_index_in_simdgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint lane_group = lane / K_LANES;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint output_tile = workgroup % OUTPUT_TILES;
    uint projection_index = workgroup / OUTPUT_TILES;
    uint projection =
        projection_index % uint(PROJECTIONS);
    uint sorted_slot =
        projection_index / uint(PROJECTIONS);
    uint route_index = uint(SORTED_ROUTES) != 0u
        ? uint(route_order[sorted_slot])
        : sorted_slot;
    uint route = route_index % uint(ROUTES);
    uint token = route_index / uint(ROUTES);
    if (token >= uint(TOKENS)) {
        return;
    }

    uint output_base =
        output_tile * ROWS_PER_TG
        + simd_group * ROWS_PER_PHYSICAL_SIMD
        + lane_group * ROWS_PER_SIMD;
    int expert =
        int(expert_ids[token * uint(ROUTES) + route]);
    if (expert < 0 || expert >= int(EXPERTS)) {
        for (
            uint row = 0u;
            row < ROWS_PER_SIMD;
            ++row
        ) {
            uint output = output_base + row;
            if (k_lane == 0u && output < uint(OUT)) {
                y[
                    (
                        (
                            token * uint(ROUTES) + route
                        ) * uint(PROJECTIONS)
                        + projection
                    ) * uint(OUT)
                    + output
                ] = T(0.0f);
            }
        }
        return;
    }

    uint descriptor_base = (
        uint(expert) * uint(PROJECTIONS)
        + projection
    ) * uint(DESCRIPTOR_SIZE);
    uint family =
        uint(descriptors[descriptor_base]);
    uint local_expert =
        uint(descriptors[descriptor_base + 1u]);
    uint rotation_variant =
        uint(descriptors[descriptor_base + 27u]);
    uint x_offset = (
        rotation_variant * uint(VARIANT_STRIDE)
        + (
            uint(SHARED_INPUT) != 0u
                ? token
                : token * uint(ROUTES) + route
        )
    ) * uint(K);
    float accumulators[MATRIX_ROWS] = {0.0f};

    if (
        (uint(FAMILY_MASK) & 1u) != 0u
        && family == 0u
    ) {
        uint bits =
            uint(descriptors[descriptor_base + 4u]);
        uint group_size =
            uint(descriptors[descriptor_base + 5u]);
        uint groups =
            uint(descriptors[descriptor_base + 6u]);
        uint q_offset =
            uint(descriptors[descriptor_base + 7u]);
        uint sub_offset =
            uint(descriptors[descriptor_base + 8u]);
        uint anchor_offset =
            uint(descriptors[descriptor_base + 9u]);
        uint q5_execution =
            uint(descriptors[descriptor_base + 10u]);

        for (
            uint group = k_lane;
            group < groups;
            group += K_LANES
        ) {
            uint outputs[MATRIX_ROWS];
            float scales[MATRIX_ROWS];
            float minimums[MATRIX_ROWS];
            for (
                uint row = 0u;
                row < MATRIX_ROWS;
                ++row
            ) {
                uint output = min(
                    output_base + (
                        uint(FUSED_SWIGLU) != 0u
                            ? (row / ROWS_PER_SIMD) * uint(OUT)
                                + row % ROWS_PER_SIMD
                            : row
                    ),
                    uint(MATRIX_OUT) - 1u);
                uint pool_output =
                    local_expert * uint(MATRIX_OUT) + output;
                uint metadata =
                    pool_output * groups + group;
                outputs[row] = pool_output;
                scales[row] =
                    nint_anchor_scale[
                        anchor_offset + pool_output]
                    * float(
                        nint_sub_scale[
                            sub_offset + metadata]);
                minimums[row] =
                    nint_anchor_min[
                        anchor_offset + pool_output]
                    * float(
                        nint_sub_min[
                            sub_offset + metadata]);
            }

            if (
                bits == 2u
                && (group_size % 4u) == 0u
            ) {
                for (
                    uint element = 0u;
                    element < group_size;
                    element += 4u
                ) {
                    uint column_base =
                        group * group_size + element;
                    float activations[4];
                    for (
                        uint component = 0u;
                        component < 4u;
                        ++component
                    ) {
                        uint column =
                            column_base + component;
                        activations[component] =
                            column < uint(K)
                                ? float(
                                    x[x_offset + column])
                                : 0.0f;
                    }
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint quantized_index = (
                            outputs[row] * groups + group
                        ) * group_size + element;
                        uint packed = uint(nint_q[
                            q_offset
                            + (quantized_index >> 2)
                        ]);
                        for (
                            uint component = 0u;
                            component < 4u;
                            ++component
                        ) {
                            uint quantized =
                                (
                                    packed
                                    >> (component * 2u)
                                ) & 3u;
                            accumulators[row] = fma(
                                activations[component],
                                scales[row]
                                    * float(quantized)
                                    - minimums[row],
                                accumulators[row]);
                        }
                    }
                }
            } else if (
                bits == 3u
                && (group_size % 8u) == 0u
            ) {
                for (
                    uint element = 0u;
                    element < group_size;
                    element += 8u
                ) {
                    uint column_base =
                        group * group_size + element;
                    float activations[8];
                    for (
                        uint component = 0u;
                        component < 8u;
                        ++component
                    ) {
                        uint column =
                            column_base + component;
                        activations[component] =
                            column < uint(K)
                                ? float(
                                    x[x_offset + column])
                                : 0.0f;
                    }
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint quantized_index = (
                            outputs[row] * groups + group
                        ) * group_size + element;
                        uint byte_index =
                            (quantized_index >> 3) * 3u
                            + (((quantized_index & 7u) * 3u) >> 3);
                        uint packed =
                            uint(nint_q[
                                q_offset + byte_index])
                            | (
                                uint(nint_q[
                                    q_offset
                                    + byte_index + 1u])
                                << 8
                            )
                            | (
                                uint(nint_q[
                                    q_offset
                                    + byte_index + 2u])
                                << 16
                            );
                        for (
                            uint component = 0u;
                            component < 8u;
                            ++component
                        ) {
                            uint quantized =
                                (
                                    packed
                                    >> (component * 3u)
                                ) & 7u;
                            accumulators[row] = fma(
                                activations[component],
                                scales[row]
                                    * float(quantized)
                                    - minimums[row],
                                accumulators[row]);
                        }
                    }
                }
            } else if (
                bits == 4u
                && (group_size % 2u) == 0u
            ) {
                for (
                    uint element = 0u;
                    element < group_size;
                    element += 2u
                ) {
                    uint column =
                        group * group_size + element;
                    float activation0 =
                        column < uint(K)
                            ? float(x[x_offset + column])
                            : 0.0f;
                    float activation1 =
                        column + 1u < uint(K)
                            ? float(
                                x[x_offset + column + 1u])
                            : 0.0f;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint quantized_index = (
                            outputs[row] * groups + group
                        ) * group_size + element;
                        uint packed = uint(nint_q[
                            q_offset
                            + (quantized_index >> 1)
                        ]);
                        accumulators[row] = fma(
                            activation0,
                            scales[row]
                                * float(packed & 15u)
                                - minimums[row],
                            accumulators[row]);
                        accumulators[row] = fma(
                            activation1,
                            scales[row]
                                * float(packed >> 4)
                                - minimums[row],
                            accumulators[row]);
                    }
                }
            } else if (
                bits == 5u
                && q5_execution != 0u
            ) {
                uint low_bytes =
                    (group_size + 1u) >> 1;
                uint execution_bytes =
                    low_bytes
                    + ((group_size + 7u) >> 3);
                for (
                    uint element = 0u;
                    element < group_size;
                    element += 8u
                ) {
                    float activations[8];
                    for (
                        uint component = 0u;
                        component < 8u;
                        ++component
                    ) {
                        uint column =
                            group * group_size
                            + element + component;
                        activations[component] =
                            element + component
                                        < group_size
                                    && column < uint(K)
                                ? float(
                                    x[x_offset + column])
                                : 0.0f;
                    }
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint metadata =
                            outputs[row] * groups + group;
                        uint group_offset =
                            q_offset
                            + metadata * execution_bytes;
                        uint high = uint(nint_q[
                            group_offset
                            + low_bytes
                            + (element >> 3)
                        ]);
                        for (
                            uint component = 0u;
                            component < 8u;
                            ++component
                        ) {
                            if (
                                element + component
                                >= group_size
                            ) {
                                break;
                            }
                            uint low_packed =
                                uint(nint_q[
                                    group_offset
                                    + (
                                        (
                                            element
                                            + component
                                        ) >> 1
                                    )
                                ]);
                            uint low = (
                                low_packed
                                >> (
                                    (
                                        (
                                            element
                                            + component
                                        ) & 1u
                                    ) * 4u
                                )
                            ) & 15u;
                            uint quantized =
                                low
                                | (
                                    (
                                        (
                                            high
                                            >> component
                                        ) & 1u
                                    ) << 4u
                                );
                            accumulators[row] = fma(
                                activations[component],
                                scales[row]
                                    * float(quantized)
                                    - minimums[row],
                                accumulators[row]);
                        }
                    }
                }
            } else if (
                bits == 6u
                && (group_size % 4u) == 0u
            ) {
                for (
                    uint element = 0u;
                    element < group_size;
                    element += 4u
                ) {
                    uint column_base =
                        group * group_size + element;
                    float activations[4];
                    for (
                        uint component = 0u;
                        component < 4u;
                        ++component
                    ) {
                        uint column =
                            column_base + component;
                        activations[component] =
                            column < uint(K)
                                ? float(
                                    x[x_offset + column])
                                : 0.0f;
                    }
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint quantized_index = (
                            outputs[row] * groups + group
                        ) * group_size + element;
                        uint byte_index =
                            (quantized_index >> 3) * 6u
                            + (((quantized_index & 7u) * 6u) >> 3);
                        uint packed =
                            uint(nint_q[
                                q_offset + byte_index])
                            | (
                                uint(nint_q[
                                    q_offset
                                    + byte_index + 1u])
                                << 8
                            )
                            | (
                                uint(nint_q[
                                    q_offset
                                    + byte_index + 2u])
                                << 16
                            );
                        for (
                            uint component = 0u;
                            component < 4u;
                            ++component
                        ) {
                            uint quantized =
                                (
                                    packed
                                    >> (component * 6u)
                                ) & 63u;
                            accumulators[row] = fma(
                                activations[component],
                                scales[row]
                                    * float(quantized)
                                    - minimums[row],
                                accumulators[row]);
                        }
                    }
                }
            } else if (bits == 8u) {
                for (
                    uint element = 0u;
                    element < group_size;
                    ++element
                ) {
                    uint column =
                        group * group_size + element;
                    float activation =
                        column < uint(K)
                            ? float(x[x_offset + column])
                            : 0.0f;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint quantized_index = (
                            outputs[row] * groups + group
                        ) * group_size + element;
                        uint quantized = uint(nint_q[
                            q_offset + quantized_index
                        ]);
                        accumulators[row] = fma(
                            activation,
                            scales[row]
                                * float(quantized)
                                - minimums[row],
                            accumulators[row]);
                    }
                }
            } else {
                for (
                    uint element = 0u;
                    element < group_size;
                    ++element
                ) {
                    uint column =
                        group * group_size + element;
                    float activation =
                        column < uint(K)
                            ? float(x[x_offset + column])
                            : 0.0f;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint quantized_index = (
                            outputs[row] * groups + group
                        ) * group_size + element;
                        uint quantized =
                            mfq_moe_nint_read_value(
                                nint_q + q_offset,
                                quantized_index,
                                bits,
                                group_size,
                                q5_execution);
                        accumulators[row] = fma(
                            activation,
                            scales[row]
                                * float(quantized)
                                - minimums[row],
                            accumulators[row]);
                    }
                }
            }
        }
    } else if (
        (uint(FAMILY_MASK) & 2u) != 0u
        && family == 1u
    ) {
        uint group_size =
            uint(descriptors[descriptor_base + 4u]);
        uint groups =
            uint(descriptors[descriptor_base + 5u]);
        uint vector_size =
            uint(descriptors[descriptor_base + 6u]);
        uint vectors =
            uint(descriptors[descriptor_base + 7u]);
        uint index_bits =
            uint(descriptors[descriptor_base + 8u]);
        uint state_bits =
            uint(descriptors[descriptor_base + 9u]);
        uint states =
            uint(descriptors[descriptor_base + 10u]);
        uint entries =
            uint(descriptors[descriptor_base + 11u]);
        uint code_banks =
            uint(descriptors[descriptor_base + 12u]);
        uint aux_mode =
            uint(descriptors[descriptor_base + 13u]);
        uint code_bank_mode =
            uint(descriptors[descriptor_base + 14u]);
        uint has_table_banks =
            uint(descriptors[descriptor_base + 15u]);
        uint groups_per_super =
            uint(descriptors[descriptor_base + 16u]);
        uint supergroups =
            uint(descriptors[descriptor_base + 17u]);
        uint indices_offset =
            uint(descriptors[descriptor_base + 18u]);
        uint state_offset =
            uint(descriptors[descriptor_base + 19u]);
        uint aux_offset =
            uint(descriptors[descriptor_base + 20u]);
        uint anchor_offset =
            uint(descriptors[descriptor_base + 21u]);
        uint codebook_offset =
            uint(descriptors[descriptor_base + 22u]);
        uint scale_offset =
            uint(descriptors[descriptor_base + 23u]);
        uint state_bank_offset =
            uint(descriptors[descriptor_base + 24u]);
        uint bank_offset =
            uint(descriptors[descriptor_base + 25u]);
        uint parameter_offset =
            uint(descriptors[descriptor_base + 26u]);
        uint vectors_per_group =
            (
                group_size + vector_size - 1u
            ) / vector_size;
        uint signs = (uint(K) + 7u) / 8u;

        uint outputs[MATRIX_ROWS];
        float row_anchors[MATRIX_ROWS];
        for (
            uint row = 0u;
            row < MATRIX_ROWS;
            ++row
        ) {
            uint output = min(
                output_base + (
                    uint(FUSED_SWIGLU) != 0u
                        ? (row / ROWS_PER_SIMD) * uint(OUT)
                            + row % ROWS_PER_SIMD
                        : row
                ),
                uint(MATRIX_OUT) - 1u);
            uint pool_output =
                local_expert * uint(MATRIX_OUT) + output;
            outputs[row] = pool_output;
            row_anchors[row] =
                vq_anchors[anchor_offset + pool_output];
        }

        uint profile =
            uint(descriptors[descriptor_base + 28u]);
        if (
            (uint(VQ_PROFILE_MASK) & 2u) != 0u
            && profile == 1u
        ) {
            for (
                uint group = k_lane;
                group < groups;
                group += K_LANES
            ) {
                uint selected_code_banks[MATRIX_ROWS];
                float weight_scales[MATRIX_ROWS];
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint state_index =
                        outputs[row] * groups + group;
                    uint state_byte = uint(vq_state[
                        state_offset + (state_index >> 1)
                    ]);
                    uint state = (
                        state_byte
                        >> ((state_index & 1u) * 4u)
                    ) & 15u;
                    selected_code_banks[row] = uint(
                        vq_state_to_codebank[
                            state_bank_offset + state
                        ]);
                    weight_scales[row] =
                        row_anchors[row]
                        * vq_scales[scale_offset + state];
                }
                constexpr uint vector_shift = 2u;
                constexpr uint vectors_per_sign = 2u;
                for (
                    uint sign_block = 0u;
                    sign_block < 3u;
                    ++sign_block
                ) {
                    uint column_base =
                        group * 24u + sign_block * 8u;
                    if (column_base >= uint(K)) {
                        break;
                    }
                    float4 activation0 = float4(
                        float(x[x_offset + column_base]),
                        float(x[x_offset + column_base + 1u]),
                        float(x[x_offset + column_base + 2u]),
                        float(x[x_offset + column_base + 3u]));
                    float4 activation1 = float4(
                        float(x[x_offset + column_base + 4u]),
                        float(x[x_offset + column_base + 5u]),
                        float(x[x_offset + column_base + 6u]),
                        float(x[x_offset + column_base + 7u]));
                    uint first_vector =
                        column_base >> vector_shift;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint indices[2] = {0u, 0u};
                        uint sign_value;
                        if (uint(JSC_EXECUTION_LAYOUT) != 0u) {
                            constexpr uint bytes_per_sign = 3u;
                            uint execution_offset =
                                indices_offset
                                + (
                                    outputs[row] * signs
                                    + column_base / 8u
                                ) * bytes_per_sign;
                            indices[0] = uint(vq_indices[
                                execution_offset]);
                            indices[1] = uint(vq_indices[
                                execution_offset + 1u]);
                            sign_value = uint(vq_indices[
                                execution_offset
                                + bytes_per_sign - 1u]);
                        } else {
                            for (
                                uint local_vector = 0u;
                                local_vector < vectors_per_sign;
                                ++local_vector
                            ) {
                                indices[local_vector] = uint(vq_indices[
                                    indices_offset
                                    + outputs[row] * vectors
                                    + first_vector + local_vector
                                ]);
                            }
                            sign_value = mfq_moe_read_bits(
                                vq_aux + aux_offset,
                                outputs[row] * signs
                                    + column_base / 8u,
                                7u);
                        }
                        uint bank_base =
                            selected_code_banks[row] * 256u;
                        uint code0_offset =
                            (bank_base + indices[0]) * 4u;
                        uint code1_offset =
                            (bank_base + indices[1]) * 4u;
                        bool4 negative0 = bool4(
                            (sign_value & 1u) != 0u,
                            (sign_value & 2u) != 0u,
                            (sign_value & 4u) != 0u,
                            (sign_value & 8u) != 0u);
                        bool4 negative1 = bool4(
                            (sign_value & 16u) != 0u,
                            (sign_value & 32u) != 0u,
                            (sign_value & 64u) != 0u,
                            uint(JSC_EXECUTION_LAYOUT) != 0u
                                ? (sign_value & 128u) != 0u
                                : (popcount(sign_value) & 1u) != 0u);
                        float4 code0 = mfq_moe_load_code4(
                            vq_codebooks,
                            codebook_offset + code0_offset);
                        float4 code1 = mfq_moe_load_code4(
                            vq_codebooks,
                            codebook_offset + code1_offset);
                        code0 = select(code0, -code0, negative0);
                        code1 = select(code1, -code1, negative1);
                        float code_dot =
                            dot(activation0, code0)
                            + dot(activation1, code1);
                        accumulators[row] = fma(
                            weight_scales[row],
                            code_dot,
                            accumulators[row]);
                    }
                }
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 16u) != 0u
            && profile == 4u
        ) {
            mfq_moe_jsc_profile<
                8u,
                MATRIX_ROWS,
                JSC_EXECUTION_LAYOUT
            >(
                x,
                vq_indices,
                vq_state,
                vq_aux,
                vq_scales,
                vq_state_to_codebank,
                vq_codebooks,
                x_offset,
                outputs,
                row_anchors,
                accumulators,
                groups,
                vectors,
                indices_offset,
                state_offset,
                aux_offset,
                codebook_offset,
                scale_offset,
                state_bank_offset,
                signs,
                k_lane,
                K_LANES,
                uint(K));
        } else if (
            (uint(VQ_PROFILE_MASK) & 4u) != 0u
            && profile == 2u
        ) {
            for (
                uint group = k_lane;
                group < groups;
                group += K_LANES
            ) {
                uint states_by_row[MATRIX_ROWS];
                float weight_scales[MATRIX_ROWS];
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint state_index =
                        outputs[row] * groups + group;
                    uint state = mfq_moe_read_bits(
                        vq_state + state_offset,
                        state_index,
                        2u);
                    states_by_row[row] = state;
                    weight_scales[row] =
                        row_anchors[row]
                        * vq_scales[scale_offset + state];
                }
                uint3 indices_by_row[MATRIX_ROWS];
                if (uint(NPQ_GROUPED_INDICES) != 0u) {
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        indices_by_row[row] =
                            mfq_moe_read_npq_group_indices<6u>(
                                vq_indices + indices_offset,
                                outputs[row],
                                group,
                                vectors);
                    }
                }

                for (
                    uint local_vector = 0u;
                    local_vector < 3u;
                    ++local_vector
                ) {
                    uint column_base =
                        group * 24u + local_vector * 8u;
                    if (column_base >= uint(K)) {
                        break;
                    }
                    float4 activation0 = float4(
                        float(x[x_offset + column_base]),
                        float(x[x_offset + column_base + 1u]),
                        float(x[x_offset + column_base + 2u]),
                        float(x[x_offset + column_base + 3u]));
                    float4 activation1 = float4(
                        float(x[x_offset + column_base + 4u]),
                        float(x[x_offset + column_base + 5u]),
                        float(x[x_offset + column_base + 6u]),
                        float(x[x_offset + column_base + 7u]));
                    uint vector = column_base >> 3;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint index =
                            uint(NPQ_GROUPED_INDICES) != 0u
                            ? indices_by_row[row][local_vector]
                            : mfq_moe_read_bits(
                                  vq_indices + indices_offset,
                                  outputs[row] * vectors + vector,
                                  6u);
                        uint code_base = codebook_offset + (
                            states_by_row[row] * 64u + index
                        ) * 8u;
                        float4 code0 = mfq_moe_load_code4(
                            vq_codebooks, code_base);
                        float4 code1 = mfq_moe_load_code4(
                            vq_codebooks, code_base + 4u);
                        float code_dot =
                            dot(activation0, code0)
                            + dot(activation1, code1);
                        accumulators[row] = fma(
                            weight_scales[row],
                            code_dot,
                            accumulators[row]);
                    }
                }
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 32u) != 0u
            && profile == 5u
        ) {
            for (
                uint group = k_lane;
                group < groups;
                group += K_LANES
            ) {
                uint states_by_row[MATRIX_ROWS];
                float weight_scales[MATRIX_ROWS];
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint state_index =
                        outputs[row] * groups + group;
                    uint state = mfq_moe_read_bits(
                        vq_state + state_offset,
                        state_index,
                        3u);
                    states_by_row[row] = state;
                    weight_scales[row] =
                        row_anchors[row]
                        * vq_scales[scale_offset + state];
                }
                uint3 indices_by_row[MATRIX_ROWS];
                if (uint(NPQ_GROUPED_INDICES) != 0u) {
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        indices_by_row[row] =
                            mfq_moe_read_npq_group_indices<7u>(
                                vq_indices + indices_offset,
                                outputs[row],
                                group,
                                vectors);
                    }
                }

                for (
                    uint local_vector = 0u;
                    local_vector < 3u;
                    ++local_vector
                ) {
                    uint column_base =
                        group * 24u + local_vector * 8u;
                    if (column_base >= uint(K)) {
                        break;
                    }
                    float4 activation0 = float4(
                        float(x[x_offset + column_base]),
                        float(x[x_offset + column_base + 1u]),
                        float(x[x_offset + column_base + 2u]),
                        float(x[x_offset + column_base + 3u]));
                    float4 activation1 = float4(
                        float(x[x_offset + column_base + 4u]),
                        float(x[x_offset + column_base + 5u]),
                        float(x[x_offset + column_base + 6u]),
                        float(x[x_offset + column_base + 7u]));
                    uint vector = column_base >> 3;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint index =
                            uint(NPQ_GROUPED_INDICES) != 0u
                            ? indices_by_row[row][local_vector]
                            : mfq_moe_read_bits(
                                  vq_indices + indices_offset,
                                  outputs[row] * vectors + vector,
                                  7u);
                        uint code_base = codebook_offset + (
                            states_by_row[row] * 128u + index
                        ) * 8u;
                        float4 code0 = mfq_moe_load_code4(
                            vq_codebooks, code_base);
                        float4 code1 = mfq_moe_load_code4(
                            vq_codebooks, code_base + 4u);
                        float code_dot =
                            dot(activation0, code0)
                            + dot(activation1, code1);
                        accumulators[row] = fma(
                            weight_scales[row],
                            code_dot,
                            accumulators[row]);
                    }
                }
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 8u) != 0u
            && profile == 3u
        ) {
            float delta = vq_parameters[parameter_offset];
            for (
                uint group = k_lane;
                group < groups;
                group += K_LANES
            ) {
                uint delta_by_row[MATRIX_ROWS];
                float weight_scales[MATRIX_ROWS];
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint state_index =
                        outputs[row] * groups + group;
                    uint state = mfq_moe_read_bits(
                        vq_state + state_offset,
                        state_index,
                        state_bits);
                    delta_by_row[row] = mfq_moe_read_bits(
                        vq_aux + aux_offset,
                        state_index,
                        1u);
                    weight_scales[row] =
                        row_anchors[row]
                        * vq_scales[scale_offset + state];
                }

                for (
                    uint local_vector = 0u;
                    local_vector < 3u;
                    ++local_vector
                ) {
                    uint column_base =
                        group * 24u + local_vector * 8u;
                    if (column_base >= uint(K)) {
                        break;
                    }
                    float4 activation0 = float4(
                        float(x[x_offset + column_base]),
                        float(x[x_offset + column_base + 1u]),
                        float(x[x_offset + column_base + 2u]),
                        float(x[x_offset + column_base + 3u]));
                    float4 activation1 = float4(
                        float(x[x_offset + column_base + 4u]),
                        float(x[x_offset + column_base + 5u]),
                        float(x[x_offset + column_base + 6u]),
                        float(x[x_offset + column_base + 7u]));
                    uint vector = column_base >> 3;
                    for (
                        uint row = 0u;
                        row < MATRIX_ROWS;
                        ++row
                    ) {
                        uint index = mfq_moe_read_bits(
                            vq_indices + indices_offset,
                            outputs[row] * vectors + vector,
                            11u);
                        float signed_delta =
                            delta_by_row[row] != 0u
                            ? -delta
                            : delta;
                        uint code_base =
                            codebook_offset + index * 8u;
                        float4 code0 = mfq_moe_load_code4(
                            vq_codebooks, code_base)
                            + float4(signed_delta);
                        float4 code1 = mfq_moe_load_code4(
                            vq_codebooks, code_base + 4u)
                            + float4(signed_delta);
                        float code_dot =
                            dot(activation0, code0)
                            + dot(activation1, code1);
                        accumulators[row] = fma(
                            weight_scales[row],
                            code_dot,
                            accumulators[row]);
                    }
                }
            }
        } else if (
            (uint(VQ_PROFILE_MASK) & 1u) != 0u
        ) {
        for (
            uint group = k_lane;
            group < groups;
            group += K_LANES
        ) {
            uint table_banks[MATRIX_ROWS];
            uint selected_code_banks[MATRIX_ROWS];
            uint delta_values[MATRIX_ROWS];
            float weight_scales[MATRIX_ROWS];
            for (
                uint row = 0u;
                row < MATRIX_ROWS;
                ++row
            ) {
                uint pool_output = outputs[row];
                uint state_index =
                    pool_output * groups + group;
                uint state = mfq_moe_read_bits(
                    vq_state + state_offset,
                    state_index,
                    state_bits);
                uint table_bank =
                    has_table_banks != 0u
                    ? uint(vq_banks[
                          bank_offset
                          + pool_output * supergroups
                          + group / groups_per_super
                      ])
                    : 0u;
                uint delta_value =
                    aux_mode == 3u
                    ? mfq_moe_read_bits(
                          vq_aux + aux_offset,
                          state_index,
                          1u)
                    : 0u;
                uint selected_code_bank = 0u;
                if (code_bank_mode == 1u) {
                    selected_code_bank = uint(
                        vq_state_to_codebank[
                            state_bank_offset + state
                        ]);
                } else if (code_bank_mode == 2u) {
                    selected_code_bank = delta_value;
                }
                table_banks[row] = table_bank;
                selected_code_banks[row] =
                    selected_code_bank;
                delta_values[row] = delta_value;
                weight_scales[row] =
                    row_anchors[row] * vq_scales[
                        scale_offset
                        + table_bank * states + state
                    ];
            }

            for (
                uint local_vector = 0u;
                local_vector < vectors_per_group;
                ++local_vector
            ) {
                uint column_base =
                    group * group_size
                    + local_vector * vector_size;
                if (column_base >= uint(K)) {
                    break;
                }
                float activations[8];
                for (
                    uint component = 0u;
                    component < 8u;
                    ++component
                ) {
                    uint column =
                        column_base + component;
                    activations[component] =
                        component < vector_size
                                && column < uint(K)
                        ? float(x[x_offset + column])
                        : 0.0f;
                }
                uint vector =
                    column_base / vector_size;
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint index = mfq_moe_read_bits(
                        vq_indices + indices_offset,
                        outputs[row] * vectors + vector,
                        index_bits);
                    uint aux_value = 0u;
                    if (
                        aux_mode == 1u
                        || aux_mode == 2u
                    ) {
                        aux_value =
                            mfq_moe_read_bits(
                                vq_aux + aux_offset,
                                outputs[row] * signs
                                    + column_base / 8u,
                                7u);
                    }
                    for (
                        uint component = 0u;
                        component < 8u;
                        ++component
                    ) {
                        if (component >= vector_size) {
                            break;
                        }
                        uint column =
                            column_base + component;
                        if (column >= uint(K)) {
                            break;
                        }
                        uint code_offset = (
                            (
                                (
                                    table_banks[row]
                                        * code_banks
                                    + selected_code_banks[
                                        row]
                                )
                                * entries + index
                            )
                            * vector_size + component
                        );
                        float code = float(vq_codebooks[
                            codebook_offset
                            + code_offset
                        ]);
                        if (
                            aux_mode == 1u
                            || aux_mode == 2u
                        ) {
                            uint sign_position =
                                column & 7u;
                            uint negative =
                                sign_position < 7u
                                ? (
                                    (
                                        aux_value
                                        >> sign_position
                                    ) & 1u
                                )
                                : (
                                    popcount(aux_value)
                                    & 1u
                                );
                            if (
                                aux_mode == 2u
                                && sign_position == 7u
                            ) {
                                negative ^=
                                    (index >> 7u) & 1u;
                            }
                            code = negative != 0u
                                ? -code
                                : code;
                        } else if (aux_mode == 3u) {
                            float delta = vq_parameters[
                                parameter_offset
                            ];
                            code +=
                                delta_values[row] != 0u
                                ? -delta
                                : delta;
                        }
                        accumulators[row] = fma(
                            activations[component],
                            weight_scales[row] * code,
                            accumulators[row]);
                    }
                }
            }
        }
        }
    } else if (
        (uint(FAMILY_MASK) & 4u) != 0u
        && family == 2u
    ) {
        uint groups =
            uint(descriptors[descriptor_base + 4u]);
        uint q_offset =
            uint(descriptors[descriptor_base + 5u]);
        uint scale_offset =
            uint(descriptors[descriptor_base + 6u]);
        for (
            uint group = k_lane;
            group < groups;
            group += K_LANES
        ) {
            uint column_base = group * 32u;
            uint outputs[MATRIX_ROWS];
            float scales[MATRIX_ROWS];
            for (
                uint row = 0u;
                row < MATRIX_ROWS;
                ++row
            ) {
                uint output = min(
                    output_base + (
                        uint(FUSED_SWIGLU) != 0u
                            ? (row / ROWS_PER_SIMD) * uint(OUT)
                                + row % ROWS_PER_SIMD
                            : row
                    ),
                    uint(MATRIX_OUT) - 1u);
                uint pool_output =
                    local_expert * uint(MATRIX_OUT) + output;
                outputs[row] = pool_output;
                scales[row] = float(q8_scales[
                    scale_offset
                    + pool_output * groups + group
                ]);
            }
            for (
                uint component = 0u;
                component < 32u;
                ++component
            ) {
                uint column =
                    column_base + component;
                float activation =
                    column < uint(K)
                        ? float(x[x_offset + column])
                        : 0.0f;
                for (
                    uint row = 0u;
                    row < MATRIX_ROWS;
                    ++row
                ) {
                    uint quantized_index = (
                        outputs[row] * groups + group
                    ) * 32u + component;
                    accumulators[row] = fma(
                        activation,
                        scales[row]
                            * float(q8_q[
                                q_offset
                                + quantized_index
                            ]),
                        accumulators[row]);
                }
            }
        }
    }

    if (uint(FUSED_SWIGLU) != 0u) {
        for (
            uint row = 0u;
            row < ROWS_PER_SIMD;
            ++row
        ) {
            float gate = accumulators[row];
            float up = accumulators[ROWS_PER_SIMD + row];
            for (
                uint offset = K_LANES >> 1;
                offset > 0u;
                offset >>= 1
            ) {
                gate += simd_shuffle_down(gate, offset);
                up += simd_shuffle_down(up, offset);
            }
            uint output = output_base + row;
            if (k_lane == 0u && output < uint(OUT)) {
                // Match the unfused graph: both projections round to the
                // activation dtype before the elementwise SwiGLU.
                gate = float(T(gate));
                up = float(T(up));
                if (params[0] > 0.0f) {
                    gate = min(gate, params[0]);
                    up = clamp(up, -params[0], params[0]);
                }
                float activated = gate / (1.0f + exp(-gate));
                y[
                    (token * uint(ROUTES) + route)
                        * uint(OUT)
                    + output
                ] = T(activated * up);
            }
        }
    } else {
        for (
            uint row = 0u;
            row < ROWS_PER_SIMD;
            ++row
        ) {
            float total = accumulators[row];
            for (
                uint offset = K_LANES >> 1;
                offset > 0u;
                offset >>= 1
            ) {
                total += simd_shuffle_down(total, offset);
            }
            uint output = output_base + row;
            if (k_lane == 0u && output < uint(OUT)) {
                y[
                    (
                        (
                            token * uint(ROUTES) + route
                        ) * uint(PROJECTIONS)
                        + projection
                    ) * uint(OUT)
                    + output
                ] = T(total);
            }
        }
    }
)METAL";

constexpr int kCccpDescriptorSize = 5;
constexpr int kCccpBits = 0;
constexpr int kCccpIndexOffset = 1;
constexpr int kCccpCodebookOffset = 2;
constexpr int kCccpVectorSize = 3;
constexpr int kCccpBlocks = 4;

constexpr const char* kCccpMoeHeader = R"METAL(
inline uint mfq_cccp_moe_read_index(
    device const uchar* indices,
    uint byte_base,
    uint value_index,
    uint bits
) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_offset = byte_base
        + (value_index >> 3) * bits
        + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed =
        uint(indices[byte_offset])
        | (uint(indices[byte_offset + 1u]) << 8u)
        | (uint(indices[byte_offset + 2u]) << 16u);
    return (packed >> shift)
        & ((1u << bits) - 1u);
}
)METAL";

constexpr const char* kCccpMoeSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint task = threadgroup_position_in_grid.x;
    uint output = task % uint(OUT);
    uint pair = task / uint(OUT);
    uint route = pair % uint(ROUTES);
    uint token = pair / uint(ROUTES);
    if (token >= uint(TOKENS)) {
        return;
    }

    int expert =
        int(expert_ids[token * uint(ROUTES) + route]);
    uint destination = pair * uint(OUT) + output;
    if (expert < 0 || expert >= int(EXPERTS)) {
        if (lane == 0u) {
            y[destination] = T(0.0f);
        }
        return;
    }
    uint descriptor_base =
        uint(expert) * uint(DESCRIPTOR_SIZE);
    uint bits =
        uint(descriptors[descriptor_base]);
    if (bits == 0u) {
        if (lane == 0u) {
            y[destination] = T(0.0f);
        }
        return;
    }
    uint index_offset =
        uint(descriptors[descriptor_base + 1u]);
    uint codebook_offset =
        uint(descriptors[descriptor_base + 2u]);
    uint vector_size =
        uint(descriptors[descriptor_base + 3u]);
    uint blocks =
        uint(descriptors[descriptor_base + 4u]);
    uint input_base = (
        uint(SHARED_INPUT) != 0u
            ? token
            : token * uint(ROUTES) + route
    ) * uint(K);
    uint row_base = output * blocks;
    float accumulator = 0.0f;
    for (
        uint block = lane;
        block < blocks;
        block += 32u
    ) {
        uint code = mfq_cccp_moe_read_index(
            indices,
            index_offset,
            row_base + block,
            bits);
        uint code_base =
            codebook_offset + code * vector_size;
        uint column_base = block * vector_size;
        for (
            uint component = 0u;
            component < vector_size;
            ++component
        ) {
            accumulator = fma(
                float(x[
                    input_base
                    + column_base
                    + component]),
                float(codebooks[
                    code_base + component]),
                accumulator);
        }
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) {
        y[destination] = T(accumulator);
    }
)METAL";

constexpr const char* kMoeHadamardSource = R"METAL(
    uint row = thread_position_in_grid.x / 256u;
    uint lane = thread_index_in_threadgroup;
    if (row >= uint(M)) {
        return;
    }

    threadgroup float values[BLOCK];
    for (
        uint local_block = 0u;
        local_block < uint(K) / uint(BLOCK);
        ++local_block
    ) {
        uint column_base =
            local_block * uint(BLOCK);
        for (
            uint index = lane;
            index < uint(BLOCK);
            index += 256u
        ) {
            uint column = column_base + index;
            values[index] =
                float(x[row * uint(K) + column])
                * float(signs[column]);
        }
        threadgroup_barrier(
            mem_flags::mem_threadgroup);
        for (
            uint stride = 1u;
            stride < uint(BLOCK);
            stride <<= 1u
        ) {
            for (
                uint pair = lane;
                pair < uint(BLOCK) / 2u;
                pair += 256u
            ) {
                uint pair_block = pair / stride;
                uint within =
                    pair - pair_block * stride;
                uint first =
                    pair_block * (stride << 1u)
                    + within;
                uint second = first + stride;
                float first_value = values[first];
                float second_value = values[second];
                values[first] =
                    first_value + second_value;
                values[second] =
                    first_value - second_value;
            }
            threadgroup_barrier(
                mem_flags::mem_threadgroup);
        }
        float inverse = rsqrt(float(BLOCK));
        for (
            uint index = lane;
            index < uint(BLOCK);
            index += 256u
        ) {
            uint column = column_base + index;
            y[row * uint(K) + column] =
                T(values[index] * inverse);
        }
        threadgroup_barrier(
            mem_flags::mem_threadgroup);
    }
)METAL";

class BlobCursor {
public:
    explicit BlobCursor(
        std::span<const std::uint8_t> blob)
        : blob_(blob) {}

    template <typename T>
    T scalar(const char* name) {
        require(sizeof(T), name);
        T value{};
        std::memcpy(
            &value,
            blob_.data() + offset_,
            sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    std::span<const std::uint8_t> bytes(
        std::size_t count,
        const char* name) {
        require(count, name);
        auto result = blob_.subspan(offset_, count);
        offset_ += count;
        return result;
    }

    std::size_t remaining() const noexcept {
        return blob_.size() - offset_;
    }

private:
    void require(
        std::size_t count,
        const char* name) const {
        if (count > blob_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated NINTM ") + name);
        }
    }

    std::span<const std::uint8_t> blob_;
    std::size_t offset_ = 0;
};

std::size_t checked_size(
    std::uint64_t value,
    const char* name) {
    if (
        value
        > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())
    ) {
        throw std::runtime_error(
            std::string("NINTM ") + name
            + " exceeds addressable memory");
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* name) {
    if (
        right != 0
        && left
            > std::numeric_limits<std::size_t>::max()
                / right
    ) {
        throw std::runtime_error(
            std::string("NINTM ") + name
            + " overflows");
    }
    return left * right;
}

std::size_t checked_add(
    std::size_t left,
    std::size_t right,
    const char* name) {
    if (
        right
        > std::numeric_limits<std::size_t>::max()
            - left
    ) {
        throw std::runtime_error(
            std::string("NINTM ") + name
            + " overflows");
    }
    return left + right;
}

std::size_t checked_packed_size(
    std::size_t count,
    int bits,
    const char* name) {
    const auto bit_count = checked_product(
        count,
        static_cast<std::size_t>(bits),
        name);
    return checked_add(bit_count, 7, name) / 8;
}

std::int32_t checked_int(
    std::size_t value,
    const char* name) {
    if (
        value
        > static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max())
    ) {
        throw std::runtime_error(
            std::string("NINTM ") + name
            + " exceeds int32 range");
    }
    return static_cast<std::int32_t>(value);
}

int checked_positive(
    std::uint32_t value,
    const char* name) {
    if (
        value == 0
        || value
            > static_cast<std::uint32_t>(
                std::numeric_limits<std::int32_t>::max())
    ) {
        throw std::runtime_error(
            std::string("invalid NINTM ") + name);
    }
    return static_cast<int>(value);
}

template <typename Allocator>
void append_raw(
    std::vector<std::uint8_t, Allocator>& target,
    const array& source,
    Dtype expected,
    const char* name) {
    if (
        source.dtype() != expected
        || !source.flags().row_contiguous
    ) {
        throw std::runtime_error(
            std::string("invalid NINTM packed ") + name);
    }
    auto evaluated = source;
    evaluated.eval();
    if (
        evaluated.nbytes()
        > std::numeric_limits<std::size_t>::max()
            - target.size()
    ) {
        throw std::runtime_error(
            "NINTM packed stream size overflows");
    }
    const auto previous = target.size();
    target.resize(previous + evaluated.nbytes());
    std::memcpy(
        target.data() + previous,
        evaluated.data<std::uint8_t>(),
        evaluated.nbytes());
}

template <typename Allocator>
array make_raw_array(
    std::vector<std::uint8_t, Allocator> bytes,
    Dtype dtype) {
    if (bytes.empty()) {
        bytes.resize(dtype.size(), 0);
    }
    if (bytes.size() % dtype.size() != 0) {
        throw std::runtime_error(
            "NINTM packed stream is misaligned");
    }
    const auto elements = checked_int(
        bytes.size() / dtype.size(),
        "packed stream");
    auto result = array(
        mlx::core::allocator::malloc(bytes.size()),
        Shape{elements},
        dtype);
    std::memcpy(
        result.data<std::uint8_t>(),
        bytes.data(),
        bytes.size());
    return result;
}

array make_int32_array(
    const std::vector<std::int32_t>& values,
    Shape shape) {
    return array(values.begin(), std::move(shape));
}

struct NativeMoeConfig {
    Dtype dtype;
    Shape output_shape;
    int tokens = 0;
    int routes = 0;
    int experts = 0;
    int output_width = 0;
    int matrix_output_width = 0;
    int projections = 0;
    int fused_swiglu = 0;
    int input_width = 0;
    int k_lanes = 0;
    int descriptor_size = 0;
    int variant_stride = 0;
    int shared_input = 0;
    int jsc_execution_layout = 0;
    int family_mask = 0;
    int vq_profile_mask = 0;
    int npq_grouped_indices = 0;
    int sorted_routes = 0;
    int workgroups = 0;
};

struct GroupedVqMmqConfig {
    Shape output_shape;
    int route_count = 0;
    int max_blocks = 0;
    int tokens = 0;
    int routes = 0;
    int experts = 0;
    int output_width = 0;
    int matrix_output_width = 0;
    int input_width = 0;
    int descriptor_size = 0;
    int variant_stride = 0;
    int shared_input = 0;
    int input_sorted = 0;
    int fused_swiglu = 0;
    float swiglu_limit = 0.0f;
};

// Adapted from oMLX's DeepSeek-V4 block-list builder.  The expert IDs have
// already been sorted, so one GPU thread can find each expert's contiguous
// range and split it into independently schedulable BM-row blocks.
const mlx::core::fast::CustomKernelFunction&
grouped_vq_block_builder() {
    static const auto builder = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_grouped_vq_block_builder_bm32",
            {"indices"},
            {"block_meta", "block_count"},
            R"METAL(
                const uint expert = thread_index_in_threadgroup;

                threadgroup atomic_int local_count;
                if (expert == 0) {
                    atomic_store_explicit(
                        &local_count,
                        0,
                        memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (expert >= NUM_EXPERTS) {
                    return;
                }

                int lo = 0;
                int hi = M;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (indices[mid] < int(expert)) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                const int start = lo;

                hi = M;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (indices[mid] <= int(expert)) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                const int end = lo;

                for (int row = start; row < end; row += BM) {
                    const int rows = min(BM, end - row);
                    const int slot = atomic_fetch_add_explicit(
                        &local_count,
                        1,
                        memory_order_relaxed);
                    if (slot < MAX_BLOCKS) {
                        block_meta[slot * 3 + 0] = row;
                        block_meta[slot * 3 + 1] = int(expert);
                        block_meta[slot * 3 + 2] = rows;
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (expert == 0) {
                    block_count[0] = atomic_load_explicit(
                        &local_count,
                        memory_order_relaxed);
                }
            )METAL",
            "",
            true,
            false,
            options);
    }();
    return builder;
}

MlxGroupedVqMmqPlan make_grouped_vq_mmq_plan(
    const array& expert_ids,
    const array& route_order_value,
    int experts) {
    constexpr int block_rows = 32;
    if (experts <= 0 || experts > 1024) {
        throw std::invalid_argument(
            "grouped VQ MMQ block builder requires 1..1024 experts");
    }
    auto ids = mlx::core::contiguous(
        mlx::core::astype(expert_ids, mlx::core::int32));
    if (ids.ndim() != 2) {
        throw std::invalid_argument(
            "routed expert IDs must have [tokens,routes] shape");
    }
    const int route_count = checked_int(
        checked_product(
            static_cast<std::size_t>(ids.shape(0)),
            static_cast<std::size_t>(ids.shape(1)),
            "route count"),
        "route count");
    auto route_order = mlx::core::contiguous(
        mlx::core::astype(route_order_value, mlx::core::int32));
    if (
        route_order.ndim() != 1
        || route_order.size()
            != static_cast<std::size_t>(route_count)
    ) {
        throw std::invalid_argument(
            "route order must contain one index per routed row");
    }
    const int max_blocks = checked_int(
        static_cast<std::size_t>(
            (route_count + block_rows - 1) / block_rows)
            + static_cast<std::size_t>(experts),
        "grouped VQ MMQ max blocks");
    auto sorted_ids = mlx::core::contiguous(
        mlx::core::take(
            mlx::core::reshape(ids, Shape{route_count}),
            route_order,
            0));
    auto outputs = grouped_vq_block_builder()(
        {std::move(sorted_ids)},
        {
            Shape{max_blocks, 3},
            Shape{1},
        },
        {
            mlx::core::int32,
            mlx::core::int32,
        },
        {experts, 1, 1},
        {experts, 1, 1},
        {
            {"NUM_EXPERTS", experts},
            {"BM", block_rows},
            {"M", route_count},
            {"MAX_BLOCKS", max_blocks},
        },
        std::nullopt,
        false,
        {});
    return MlxGroupedVqMmqPlan{
        .block_meta = std::move(outputs.at(0)),
        .block_count = std::move(outputs.at(1)),
        .max_blocks = max_blocks,
        .block_rows = block_rows,
        .route_count = route_count,
        .experts = experts,
    };
}

std::string native_moe_kernel_name(
    const NativeMoeConfig& config) {
    std::ostringstream name;
    name
        << "mfq_native_nintm_"
        << (config.dtype == mlx::core::float16 ? "f16" : "f32")
        << "_t" << config.tokens
        << "_r" << config.routes
        << "_e" << config.experts
        << "_o" << config.output_width
        << "_mo" << config.matrix_output_width
        << "_p" << config.projections
        << "_sw" << config.fused_swiglu
        << "_k" << config.input_width
        << "_kl" << config.k_lanes
        << "_vs" << config.variant_stride
        << "_si" << config.shared_input
        << "_jl" << config.jsc_execution_layout
        << "_fm" << config.family_mask
        << "_vm" << config.vq_profile_mask
        << "_ni" << config.npq_grouped_indices;
    name << "_sr" << config.sorted_routes;
    return name.str();
}

std::string make_native_moe_source(
    const NativeMoeConfig& config,
    const std::string& kernel_name) {
    std::ostringstream source;
    source
        << "#include <metal_stdlib>\n"
        << "using namespace metal;\n"
        << "using T = "
        << (config.dtype == mlx::core::float16 ? "half" : "float")
        << ";\n"
        << kMoeHeader
        << "\nkernel void " << kernel_name << "(\n"
        << "device const int* descriptors [[buffer(0)]],\n"
        << "device const uchar* nint_q [[buffer(1)]],\n"
        << "device const uchar* nint_sub_scale [[buffer(2)]],\n"
        << "device const uchar* nint_sub_min [[buffer(3)]],\n"
        << "device const float* nint_anchor_scale [[buffer(4)]],\n"
        << "device const float* nint_anchor_min [[buffer(5)]],\n"
        << "device const int8_t* q8_q [[buffer(6)]],\n"
        << "device const half* q8_scales [[buffer(7)]],\n"
        << "device const uchar* vq_indices [[buffer(8)]],\n"
        << "device const uchar* vq_state [[buffer(9)]],\n"
        << "device const uchar* vq_aux [[buffer(10)]],\n"
        << "device const float* vq_anchors [[buffer(11)]],\n"
        << "device const int8_t* vq_codebooks [[buffer(12)]],\n"
        << "device const float* vq_scales [[buffer(13)]],\n"
        << "device const uchar* vq_state_to_codebank [[buffer(14)]],\n"
        << "device const uchar* vq_banks [[buffer(15)]],\n"
        << "device const float* vq_parameters [[buffer(16)]],\n"
        << "device const T* x [[buffer(17)]],\n"
        << "device const int* expert_ids [[buffer(18)]],\n"
        << "device const int* route_order [[buffer(19)]],\n"
        << "device const float* params [[buffer(20)]],\n"
        << "device T* y [[buffer(21)]],\n"
        << "uint thread_index_in_simdgroup "
           "[[thread_index_in_simdgroup]],\n"
        << "uint simdgroup_index_in_threadgroup "
           "[[simdgroup_index_in_threadgroup]],\n"
        << "uint3 threadgroup_position_in_grid "
           "[[threadgroup_position_in_grid]]) {\n"
        << "constexpr int TOKENS = " << config.tokens << ";\n"
        << "constexpr int ROUTES = " << config.routes << ";\n"
        << "constexpr int EXPERTS = " << config.experts << ";\n"
        << "constexpr int OUT = " << config.output_width << ";\n"
        << "constexpr int MATRIX_OUT = "
        << config.matrix_output_width << ";\n"
        << "constexpr int PROJECTIONS = "
        << config.projections << ";\n"
        << "constexpr int FUSED_SWIGLU = "
        << config.fused_swiglu << ";\n"
        << "constexpr int K = " << config.input_width << ";\n"
        << "constexpr int K_LANES_VALUE = "
        << config.k_lanes << ";\n"
        << "constexpr int DESCRIPTOR_SIZE = "
        << config.descriptor_size << ";\n"
        << "constexpr int VARIANT_STRIDE = "
        << config.variant_stride << ";\n"
        << "constexpr int SHARED_INPUT = "
        << config.shared_input << ";\n"
        << "constexpr int JSC_EXECUTION_LAYOUT = "
        << config.jsc_execution_layout << ";\n"
        << "constexpr int FAMILY_MASK = "
        << config.family_mask << ";\n"
        << "constexpr int VQ_PROFILE_MASK = "
        << config.vq_profile_mask << ";\n"
        << "constexpr int NPQ_GROUPED_INDICES = "
        << config.npq_grouped_indices << ";\n"
        << "constexpr int SORTED_ROUTES = "
        << config.sorted_routes << ";\n"
        << kMoeSource
        << "\n}\n";
    return source.str();
}

class NativeNintMoePrimitive final
    : public mlx::core::UnaryPrimitive {
public:
    NativeNintMoePrimitive(
        mlx::core::Stream stream,
        NativeMoeConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)),
          kernel_name_(native_moe_kernel_name(config_)) {}

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "native NINTM primitive has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 21) {
            throw std::logic_error(
                "native NINTM primitive input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_,
            compile_options,
            [config = config_, name = kernel_name_] {
                return make_native_moe_source(
                    config,
                    name);
            });
        auto* kernel = device.get_kernel(
            kernel_name_,
            library);
        auto& encoder =
            mlx::core::metal::get_command_encoder(
                selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 21; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)],
                index);
        }
        encoder.set_output_array(output, 21);
        encoder.dispatch_threadgroups(
            MTL::Size(config_.workgroups, 1, 1),
            MTL::Size(64, 1, 1));
    }

    const char* name() const override {
        return "NativeNintMoePrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const NativeNintMoePrimitive*>(&other);
        return primitive != nullptr
            && primitive->kernel_name_ == kernel_name_;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    NativeMoeConfig config_;
    std::string kernel_name_;
};

array native_moe_dispatch(
    std::vector<array> inputs,
    NativeMoeConfig config) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "native NINTM primitive requires the Metal device");
    }
    auto shape = config.output_shape;
    auto dtype = config.dtype;
    return array(
        std::move(shape),
        dtype,
        std::make_shared<NativeNintMoePrimitive>(
            stream,
            std::move(config)),
        std::move(inputs));
}

class GroupedVqMmqPrimitive final
    : public mlx::core::UnaryPrimitive {
public:
    GroupedVqMmqPrimitive(
        mlx::core::Stream stream,
        GroupedVqMmqConfig config)
        : UnaryPrimitive(stream),
          config_(std::move(config)) {}

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "grouped VQ MMQ primitive has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 23) {
            throw std::logic_error(
                "grouped VQ MMQ primitive input count mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(
            selected_stream.device);
        CompileOptions compile_options;
        compile_options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            "mfq_grouped_vq_mmq_v8",
            compile_options,
            [] {
                std::string source;
                source.reserve(
                    sizeof(detail::kSteelMmaSource)
                    + sizeof(detail::kNintmPrefillSource)
                    + 128);
                source += "#include <metal_stdlib>\n";
                source += "#include <metal_simdgroup>\n";
                source += "#include <metal_simdgroup_matrix>\n";
                source += "using namespace metal;\n";
                source += "using bfloat16_t = bfloat;\n";
                source += detail::kSteelMmaSource;
                source += detail::kNintmPrefillSource;
                return source;
            });
        auto& encoder =
            mlx::core::metal::get_command_encoder(
                selected_stream);
        encoder.set_input_array(inputs[0], 0);
        for (int source = 8; source <= 19; ++source) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(source)],
                source - 7);
        }
        encoder.set_input_array(inputs[21], 26);
        encoder.set_input_array(inputs[22], 27);
        encoder.set_output_array(output, 13);
        encoder.set_bytes(config_.route_count, 14);
        encoder.set_bytes(config_.tokens, 15);
        encoder.set_bytes(config_.routes, 16);
        encoder.set_bytes(config_.experts, 17);
        encoder.set_bytes(config_.output_width, 18);
        encoder.set_bytes(config_.matrix_output_width, 19);
        encoder.set_bytes(config_.input_width, 20);
        encoder.set_bytes(config_.descriptor_size, 21);
        encoder.set_bytes(config_.variant_stride, 22);
        encoder.set_bytes(config_.shared_input, 23);
        encoder.set_bytes(config_.input_sorted, 24);
        encoder.set_bytes(config_.swiglu_limit, 25);
        const char* kernel_name = config_.fused_swiglu != 0
            ? "mfq_grouped_vq_swiglu_f16_bm32_bn64_bk96"
            : "mfq_grouped_vq_mmq_f16_bm32_bn64_bk96";
        auto* kernel = device.get_kernel(
            kernel_name,
            library);
        encoder.set_compute_pipeline_state(kernel);
        encoder.dispatch_threadgroups(
            MTL::Size(
                (config_.output_width + 63) / 64,
                config_.max_blocks,
                1),
            MTL::Size(256, 1, 1));
    }

    const char* name() const override {
        return "GroupedVqMmqPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const GroupedVqMmqPrimitive*>(&other);
        return primitive != nullptr
            && primitive->config_.route_count == config_.route_count
            && primitive->config_.max_blocks == config_.max_blocks
            && primitive->config_.experts == config_.experts
            && primitive->config_.output_width == config_.output_width
            && primitive->config_.input_width == config_.input_width
            && primitive->config_.variant_stride == config_.variant_stride
            && primitive->config_.shared_input == config_.shared_input
            && primitive->config_.input_sorted == config_.input_sorted
            && primitive->config_.fused_swiglu == config_.fused_swiglu
            && primitive->config_.swiglu_limit == config_.swiglu_limit;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {config_.output_shape};
    }

private:
    GroupedVqMmqConfig config_;
};

array grouped_vq_mmq_dispatch(
    std::vector<array> inputs,
    GroupedVqMmqConfig config) {
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "grouped VQ MMQ requires the Metal device");
    }
    auto shape = config.output_shape;
    return array(
        std::move(shape),
        mlx::core::float16,
        std::make_shared<GroupedVqMmqPrimitive>(
            stream,
            std::move(config)),
        std::move(inputs));
}

mlx::core::fast::CustomKernelFunction
make_moe_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_heterogeneous_nint_moe",
        {
            "descriptors",
            "nint_q",
            "nint_sub_scale",
            "nint_sub_min",
            "nint_anchor_scale",
            "nint_anchor_min",
            "q8_q",
            "q8_scales",
            "vq_indices",
            "vq_state",
            "vq_aux",
            "vq_anchors",
            "vq_codebooks",
            "vq_scales",
            "vq_state_to_codebank",
            "vq_banks",
            "vq_parameters",
            "x",
            "expert_ids",
            "route_order",
            "params",
        },
        {"y"},
        kMoeSource,
        kMoeHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
moe_kernel() {
    static const auto kernel = make_moe_kernel();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
cccp_moe_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_streamed_cccp_moe",
            {
                "descriptors",
                "indices",
                "codebooks",
                "x",
                "expert_ids",
            },
            {"y"},
            kCccpMoeSource,
            kCccpMoeHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

mlx::core::fast::CustomKernelFunction
make_moe_hadamard_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_moe_signed_hadamard",
        {"x", "signs"},
        {"y"},
        kMoeHadamardSource,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
moe_hadamard_kernel() {
    static const auto kernel =
        make_moe_hadamard_kernel();
    return kernel;
}

struct RotationSpec {
    array signs;
    std::vector<std::int8_t> sign_values;
    int block = 0;
    std::uint64_t seed = 0;
};

struct PackedStreams {
    detail::StagingVector<std::uint8_t> nint_q;
    detail::StagingVector<std::uint8_t> nint_sub_scale;
    detail::StagingVector<std::uint8_t> nint_sub_min;
    detail::StagingVector<std::uint8_t> nint_anchor_scale;
    detail::StagingVector<std::uint8_t> nint_anchor_min;
    detail::StagingVector<std::uint8_t> q8_q;
    detail::StagingVector<std::uint8_t> q8_scales;
    detail::StagingVector<std::uint8_t> vq_indices;
    detail::StagingVector<std::uint8_t> vq_state;
    detail::StagingVector<std::uint8_t> vq_aux;
    detail::StagingVector<std::uint8_t> vq_anchors;
    detail::StagingVector<std::uint8_t> vq_codebooks;
    detail::StagingVector<std::uint8_t> vq_scales;
    detail::StagingVector<std::uint8_t>
        vq_state_to_codebank;
    detail::StagingVector<std::uint8_t> vq_banks;
    detail::StagingVector<std::uint8_t> vq_parameters;
};

void validate_nint_payload_shape(
    std::span<const std::uint8_t> payload,
    int expected_rows,
    int expected_columns) {
    BlobCursor cursor(payload);
    const int bits = static_cast<int>(
        cursor.scalar<std::uint8_t>("NINT bits"));
    const int sub_bits = static_cast<int>(
        cursor.scalar<std::uint8_t>(
            "NINT sub bits"));
    const int group_size =
        cursor.scalar<std::int32_t>(
            "NINT group size");
    const int axis =
        cursor.scalar<std::int32_t>("NINT axis");
    const int columns =
        cursor.scalar<std::int32_t>(
            "NINT neuron length");
    const auto dimensions =
        cursor.scalar<std::uint32_t>(
            "NINT dimension count");
    if (
        bits < 1
        || bits > 8
        || sub_bits < 1
        || sub_bits > 8
        || group_size <= 0
        || axis != 0
        || columns != expected_columns
        || dimensions != 2
    ) {
        throw std::runtime_error(
            "NINTM NINT cohort shape is inconsistent");
    }
    const auto rows =
        cursor.scalar<std::int64_t>(
            "NINT output shape");
    const auto shape_columns =
        cursor.scalar<std::int64_t>(
            "NINT input shape");
    const auto output_size =
        cursor.scalar<std::uint32_t>(
            "NINT output size");
    const auto groups =
        cursor.scalar<std::uint32_t>(
            "NINT group count");
    const auto expected_groups =
        (
            static_cast<std::uint64_t>(
                expected_columns)
            + static_cast<std::uint64_t>(
                group_size)
            - 1
        ) / static_cast<std::uint64_t>(
            group_size);
    if (
        rows != expected_rows
        || shape_columns != expected_columns
        || output_size
            != static_cast<std::uint32_t>(
                expected_rows)
        || groups != expected_groups
    ) {
        throw std::runtime_error(
            "NINTM NINT cohort shape is inconsistent");
    }

    const auto metadata_count = checked_product(
        static_cast<std::size_t>(output_size),
        static_cast<std::size_t>(groups),
        "NINT metadata count");
    const auto value_count = checked_product(
        metadata_count,
        static_cast<std::size_t>(group_size),
        "NINT value count");
    const auto anchor_bytes = checked_product(
        static_cast<std::size_t>(output_size),
        2 * sizeof(std::uint16_t),
        "NINT anchor bytes");
    const auto packed_metadata =
        checked_packed_size(
            metadata_count,
            sub_bits,
            "NINT metadata bytes");
    const auto packed_values =
        checked_packed_size(
            value_count,
            bits,
            "NINT value bytes");
    const auto packed_tail = checked_add(
        checked_product(
            packed_metadata,
            2,
            "NINT metadata bytes"),
        packed_values,
        "NINT packed tail");
    const auto old_tail = checked_add(
        checked_product(
            metadata_count,
            2,
            "legacy NINT metadata bytes"),
        value_count,
        "legacy NINT tail");
    const auto remaining = cursor.remaining();
    if (
        remaining
            != checked_add(
                anchor_bytes,
                packed_tail,
                "NINT payload bytes")
        && remaining
            != checked_add(
                anchor_bytes,
                old_tail,
                "legacy NINT payload bytes")
    ) {
        throw std::runtime_error(
            "invalid NINTM NINT cohort payload length");
    }
}

void claim_experts(
    const std::vector<std::int32_t>& expert_ids,
    std::vector<int>& owners,
    int pool) {
    for (const auto expert : expert_ids) {
        if (
            expert < 0
            || expert
                >= static_cast<std::int32_t>(
                    owners.size())
        ) {
            throw std::runtime_error(
                "NINTM pool contains an invalid expert id");
        }
        if (owners[static_cast<std::size_t>(expert)] >= 0) {
            throw std::runtime_error(
                "an expert belongs to multiple NINTM pools");
        }
        owners[static_cast<std::size_t>(expert)] = pool;
    }
}

void add_nint_pool(
    std::span<const std::uint8_t> payload,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<std::int32_t>& descriptors) {
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "NINT cohort row count");
    const int checked_rows = checked_int(
        expected_rows,
        "NINT cohort row count");
    validate_nint_payload_shape(
        payload,
        checked_rows,
        neuron_len);
    auto weight = MlxNintWeight::from_blob(payload);
    if (
        weight.output_size() != checked_rows
        || weight.input_size() != neuron_len
        || weight.groups()
            != (
                neuron_len + weight.group_size() - 1
            ) / weight.group_size()
    ) {
        throw std::runtime_error(
            "NINTM NINT cohort shape is inconsistent");
    }

    const int q_offset =
        checked_int(streams.nint_q.size(), "NINT q offset");
    const int sub_offset =
        checked_int(
            streams.nint_sub_scale.size(),
            "NINT sub offset");
    const int anchor_offset =
        checked_int(
            streams.nint_anchor_scale.size()
                / sizeof(float),
            "NINT anchor offset");

    for (
        std::size_t local_expert = 0;
        local_expert < expert_ids.size();
        ++local_expert
    ) {
        const int expert =
            expert_ids[local_expert];
        const auto base =
            checked_product(
                static_cast<std::size_t>(expert),
                static_cast<std::size_t>(
                    kDescriptorSize),
                "descriptor offset");
        descriptors[base + kFamily] = kFamilyNint;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kNintBits] = weight.bits();
        descriptors[base + kNintGroupSize] =
            weight.group_size();
        descriptors[base + kNintGroups] =
            weight.groups();
        descriptors[base + kNintQOffset] = q_offset;
        descriptors[base + kNintSubOffset] =
            sub_offset;
        descriptors[base + kNintAnchorOffset] =
            anchor_offset;
        descriptors[base + kNintQ5Execution] =
            static_cast<int>(
                weight.q5_execution_layout());
    }

    append_raw(
        streams.nint_q,
        weight.packed_values(),
        mlx::core::uint8,
        "NINT values");
    // Fast 3/6-bit paths load up to two bytes past the logical packet.
    streams.nint_q.insert(streams.nint_q.end(), 2, 0);
    append_raw(
        streams.nint_sub_scale,
        weight.sub_scales(),
        mlx::core::uint8,
        "NINT sub scales");
    append_raw(
        streams.nint_sub_min,
        weight.sub_mins(),
        mlx::core::uint8,
        "NINT sub minima");
    append_raw(
        streams.nint_anchor_scale,
        weight.neuron_scales(),
        mlx::core::float32,
        "NINT neuron scales");
    append_raw(
        streams.nint_anchor_min,
        weight.neuron_mins(),
        mlx::core::float32,
        "NINT neuron minima");

}

void add_q8_pool(
    std::span<const std::uint8_t> payload,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<std::int32_t>& descriptors) {
    auto weight =
        MlxNint8ZeroWeight::from_blob(payload);
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "NINT8-0 cohort row count");
    if (
        weight.output_size()
            != checked_int(
                expected_rows,
                "NINT8-0 cohort row count")
        || weight.input_size() != neuron_len
        || weight.groups() != neuron_len / 32
    ) {
        throw std::runtime_error(
            "NINTM NINT8-0 cohort shape is inconsistent");
    }

    const int q_offset =
        checked_int(
            streams.q8_q.size(),
            "NINT8-0 q offset");
    const int scale_offset =
        checked_int(
            streams.q8_scales.size()
                / sizeof(std::uint16_t),
            "NINT8-0 scale offset");
    for (
        std::size_t local_expert = 0;
        local_expert < expert_ids.size();
        ++local_expert
    ) {
        const int expert =
            expert_ids[local_expert];
        const auto base =
            checked_product(
                static_cast<std::size_t>(expert),
                static_cast<std::size_t>(
                    kDescriptorSize),
                "descriptor offset");
        descriptors[base + kFamily] =
            kFamilyNint8Zero;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kQ8Groups] =
            weight.groups();
        descriptors[base + kQ8QOffset] = q_offset;
        descriptors[base + kQ8ScaleOffset] =
            scale_offset;
    }

    append_raw(
        streams.q8_q,
        weight.quantized_values(),
        mlx::core::int8,
        "NINT8-0 values");
    append_raw(
        streams.q8_scales,
        weight.scales(),
        mlx::core::float16,
        "NINT8-0 scales");
}

std::vector<std::int8_t> int8_values(
    const array& source,
    const char* name) {
    if (
        source.dtype() != mlx::core::int8
        || !source.flags().row_contiguous
    ) {
        throw std::runtime_error(
            std::string("invalid NINTM packed ") + name);
    }
    auto evaluated = source;
    evaluated.eval();
    return {
        evaluated.data<std::int8_t>(),
        evaluated.data<std::int8_t>()
            + static_cast<std::ptrdiff_t>(
                evaluated.size()),
    };
}

int rotation_variant(
    const MlxVqWeight& weight,
    std::vector<RotationSpec>& rotations) {
    if (weight.rotation_block() == 0) {
        return 0;
    }
    auto values = int8_values(
        weight.rotation_signs(),
        "VQ rotation signs");
    if (
        values.size()
        != static_cast<std::size_t>(
            weight.input_size())
    ) {
        throw std::runtime_error(
            "rotated NINTM VQ sign width mismatch");
    }
    for (
        std::size_t index = 0;
        index < rotations.size();
        ++index
    ) {
        const auto& rotation = rotations[index];
        if (
            rotation.block == weight.rotation_block()
            && rotation.seed == weight.rotation_seed()
        ) {
            if (rotation.sign_values != values) {
                throw std::runtime_error(
                    "conflicting NINTM HSG1 sign "
                    "vectors share one rotation key");
            }
            return checked_int(
                index + 1,
                "rotation variant");
        }
    }
    rotations.push_back({
        weight.rotation_signs(),
        std::move(values),
        weight.rotation_block(),
        weight.rotation_seed(),
    });
    return checked_int(
        rotations.size(),
        "rotation variant");
}

int rotation_variant(
    const RotationSpec& source,
    std::vector<RotationSpec>& rotations) {
    for (
        std::size_t index = 0;
        index < rotations.size();
        ++index
    ) {
        const auto& rotation = rotations[index];
        if (
            rotation.block == source.block
            && rotation.seed == source.seed
        ) {
            if (
                rotation.sign_values
                != source.sign_values
            ) {
                throw std::runtime_error(
                    "conflicting NINTM HSG1 sign "
                    "vectors share one rotation key");
            }
            return checked_int(
                index + 1,
                "rotation variant");
        }
    }
    rotations.push_back(source);
    return checked_int(
        rotations.size(),
        "rotation variant");
}

void add_vq_pool(
    std::string_view dtype,
    std::span<const std::uint8_t> payload,
    std::span<const std::uint8_t> runtime,
    const std::vector<std::int32_t>& expert_ids,
    int out_per_expert,
    int neuron_len,
    PackedStreams& streams,
    std::vector<RotationSpec>& rotations,
    std::vector<std::int32_t>& descriptors) {
    const auto expected_rows = checked_product(
        expert_ids.size(),
        static_cast<std::size_t>(out_per_expert),
        "VQ cohort row count");
    const int checked_rows = checked_int(
        expected_rows,
        "VQ cohort row count");
    const auto metadata = inspect_vq_blob(
        dtype,
        payload,
        runtime);
    const auto& output_shape = metadata.output_shape;
    const bool cross_expert =
        output_shape.size() == 2;
    if (
        metadata.output_size != checked_rows
        || metadata.input_size != neuron_len
        || (
            cross_expert
            && (
                output_shape[0]
                    != static_cast<int>(
                        expert_ids.size())
                || output_shape[1]
                    != out_per_expert
            )
        )
        || (
            !cross_expert
            && (
                output_shape.size() != 1
                || output_shape.front()
                    != checked_rows
            )
        )
    ) {
        throw std::runtime_error(
            "NINTM VQ cohort shape is inconsistent");
    }
    // Every supported VQ profile stores one FP16 anchor per flattened output
    // row.  Reject impossible dimensions before the full parser allocates
    // canonical execution arrays from an attacker-controlled header.
    if (
        expected_rows
        > payload.size() / sizeof(std::uint16_t)
    ) {
        throw std::runtime_error(
            "NINTM VQ cohort dimensions exceed "
            "its payload");
    }
    auto weight = MlxVqWeight::from_blob(
        dtype,
        payload,
        runtime);
    if (
        weight.output_size() != metadata.output_size
        || weight.input_size() != metadata.input_size
        || weight.output_shape() != metadata.output_shape
        || weight.rotation_block()
            != metadata.rotation_block
        || weight.rotation_seed()
            != metadata.rotation_seed
    ) {
        throw std::runtime_error(
            "NINTM VQ header/full parse mismatch");
    }

    // CUDA's NVQ2 execution layout merges one byte index and one byte sign
    // mask.  The same idea also applies to NVQ3J by placing its two D4
    // indices beside the parity-completed sign mask.  This trades at most one
    // padding bit per eight weights for one sequential read in the decode
    // kernel instead of independent packed index and 7-bit auxiliary reads.
    const char* jsc_exec_env = std::getenv(
        "MFQ_METAL_NINTM_JSC_EXEC");
    const bool jsc_execution =
        (dtype == "NVQ2J" || dtype == "NVQ3J")
        && (
            jsc_exec_env == nullptr
            || std::string_view(jsc_exec_env) != "0"
        );
    detail::StagingVector<std::uint8_t>
        jsc_execution_indices;
    if (jsc_execution) {
        auto packed_indices = weight.packed_indices();
        auto packed_aux = weight.packed_auxiliary();
        packed_indices.eval();
        packed_aux.eval();
        const auto* index_data =
            packed_indices.data<std::uint8_t>();
        const auto* aux_data =
            packed_aux.data<std::uint8_t>();
        const int signs = (neuron_len + 7) / 8;
        const int bytes_per_sign =
            weight.vector_size() == 4 ? 3 : 2;
        const auto execution_bytes = checked_product(
            checked_product(
                expected_rows,
                static_cast<std::size_t>(signs),
                "VQ JSC execution stream size"),
            static_cast<std::size_t>(bytes_per_sign),
            "VQ JSC execution stream size");
        jsc_execution_indices.resize(
            execution_bytes);
        const auto read_aux = [&](std::size_t linear) {
            const auto bit = linear * 7;
            const auto byte = bit >> 3;
            const int shift = bit & 7;
            std::uint32_t packed = aux_data[byte];
            if (byte + 1 < packed_aux.size()) {
                packed |= static_cast<std::uint32_t>(
                    aux_data[byte + 1]) << 8;
            }
            return (packed >> shift) & 0x7fu;
        };
        const auto pack_rows = [&](std::size_t begin, std::size_t end) {
            for (std::size_t row = begin; row < end; ++row) {
                for (int sign = 0; sign < signs; ++sign) {
                    const auto source_sign =
                        row * signs + sign;
                    const auto target =
                        source_sign * bytes_per_sign;
                    const auto first_vector =
                        static_cast<std::size_t>(
                            sign)
                        * (weight.vector_size() == 4 ? 2 : 1);
                    jsc_execution_indices[target] =
                        index_data[
                            row * weight.vectors()
                            + first_vector];
                    if (weight.vector_size() == 4) {
                        jsc_execution_indices[target + 1] =
                            index_data[
                                row * weight.vectors()
                                + first_vector + 1];
                    }
                    const auto mask7 = read_aux(
                        source_sign);
                    const auto parity =
                        std::popcount(mask7) & 1u;
                    jsc_execution_indices[
                        target + bytes_per_sign - 1] =
                        static_cast<std::uint8_t>(
                            mask7 | (parity << 7));
                }
            }
        };
        constexpr std::size_t kParallelThreshold =
            8u * 1024u * 1024u;
        const auto worker_count = execution_bytes
                >= kParallelThreshold
            ? std::min<std::size_t>(8, expected_rows)
            : 1;
        if (worker_count == 1) {
            pack_rows(0, expected_rows);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            for (
                std::size_t worker = 0;
                worker < worker_count;
                ++worker
            ) {
                const auto begin =
                    expected_rows * worker / worker_count;
                const auto end =
                    expected_rows * (worker + 1) / worker_count;
                workers.emplace_back(pack_rows, begin, end);
            }
            for (auto& worker : workers) {
                worker.join();
            }
        }
    }

    const int indices_offset = checked_int(
        streams.vq_indices.size(),
        "VQ index offset");
    const int state_offset = checked_int(
        streams.vq_state.size(),
        "VQ state offset");
    const int aux_offset = checked_int(
        streams.vq_aux.size(),
        "VQ auxiliary offset");
    const int anchor_offset = checked_int(
        streams.vq_anchors.size() / sizeof(float),
        "VQ anchor offset");
    const int codebook_offset = checked_int(
        streams.vq_codebooks.size(),
        "VQ codebook offset");
    const int scale_offset = checked_int(
        streams.vq_scales.size() / sizeof(float),
        "VQ scale offset");
    const int state_bank_offset = checked_int(
        streams.vq_state_to_codebank.size(),
        "VQ state-bank offset");
    const int bank_offset = checked_int(
        streams.vq_banks.size(),
        "VQ bank offset");
    const int parameter_offset = checked_int(
        streams.vq_parameters.size()
            / sizeof(float),
        "VQ parameter offset");
    const int rotation =
        rotation_variant(weight, rotations);

    for (
        std::size_t local_expert = 0;
        local_expert < expert_ids.size();
        ++local_expert
    ) {
        const int expert = expert_ids[local_expert];
        const auto base = checked_product(
            static_cast<std::size_t>(expert),
            static_cast<std::size_t>(kDescriptorSize),
            "descriptor offset");
        descriptors[base + kFamily] = kFamilyVq;
        descriptors[base + kLocalExpert] =
            checked_int(local_expert, "local expert");
        descriptors[base + kOut] = out_per_expert;
        descriptors[base + kInput] = neuron_len;
        descriptors[base + kVqGroupSize] =
            weight.group_size();
        descriptors[base + kVqGroups] =
            weight.groups();
        descriptors[base + kVqVectorSize] =
            weight.vector_size();
        descriptors[base + kVqVectors] =
            weight.vectors();
        descriptors[base + kVqIndexBits] =
            weight.index_bits();
        descriptors[base + kVqStateBits] =
            weight.state_bits();
        descriptors[base + kVqStates] =
            weight.states();
        descriptors[base + kVqEntries] =
            weight.entries();
        descriptors[base + kVqCodeBanks] =
            weight.code_banks();
        descriptors[base + kVqAuxMode] =
            weight.aux_mode();
        descriptors[base + kVqCodeBankMode] =
            weight.code_bank_mode();
        descriptors[base + kVqHasTableBanks] =
            static_cast<int>(
                weight.table_banks() > 1);
        descriptors[base + kVqGroupsPerSuper] =
            weight.groups_per_supergroup();
        descriptors[base + kVqSupergroups] =
            weight.supergroups();
        descriptors[base + kVqIndicesOffset] =
            indices_offset;
        descriptors[base + kVqStateOffset] =
            state_offset;
        descriptors[base + kVqAuxOffset] =
            aux_offset;
        descriptors[base + kVqAnchorOffset] =
            anchor_offset;
        descriptors[base + kVqCodebookOffset] =
            codebook_offset;
        descriptors[base + kVqScaleOffset] =
            scale_offset;
        descriptors[base + kVqStateBankOffset] =
            state_bank_offset;
        descriptors[base + kVqBankOffset] =
            bank_offset;
        descriptors[base + kVqParameterOffset] =
            parameter_offset;
        descriptors[base + kVqRotationVariant] =
            rotation;
        descriptors[base + kVqProfile] =
            dtype == "NVQ2J" || dtype == "NVQ3J"
                ? (
                    weight.vector_size() == 4
                        ? kVqProfileJsc4
                        : kVqProfileJsc8
                )
                : (
                    dtype == "NPQ0-S" || dtype == "NPQ0-L"
                        ? (
                            dtype == "NPQ0-S"
                                ? kVqProfileNpqS
                                : kVqProfileNpqL
                        )
                        : (
                            dtype == "NVQ1-L"
                                    && weight.group_size() == 24
                                    && weight.vector_size() == 8
                                    && weight.index_bits() == 11
                                ? kVqProfileNvq1L
                                : kVqProfileGeneric
                        )
                );
        descriptors[base + kVqJscExecution] =
            static_cast<int>(jsc_execution);
    }

    if (jsc_execution) {
        streams.vq_indices.insert(
            streams.vq_indices.end(),
            jsc_execution_indices.begin(),
            jsc_execution_indices.end());
    } else {
        append_raw(
            streams.vq_indices,
            weight.packed_indices(),
            mlx::core::uint8,
            "VQ indices");
    }
    append_raw(
        streams.vq_state,
        weight.packed_states(),
        mlx::core::uint8,
        "VQ states");
    if (!jsc_execution) {
        append_raw(
            streams.vq_aux,
            weight.packed_auxiliary(),
            mlx::core::uint8,
            "VQ auxiliary values");
    }
    append_raw(
        streams.vq_anchors,
        weight.anchors(),
        mlx::core::float32,
        "VQ anchors");
    append_raw(
        streams.vq_codebooks,
        weight.codebooks(),
        mlx::core::int8,
        "VQ codebooks");
    append_raw(
        streams.vq_scales,
        weight.scale_lut(),
        mlx::core::float32,
        "VQ scales");
    append_raw(
        streams.vq_state_to_codebank,
        weight.state_to_codebank(),
        mlx::core::uint8,
        "VQ state banks");
    append_raw(
        streams.vq_banks,
        weight.bank_ids(),
        mlx::core::uint8,
        "VQ table banks");
    append_raw(
        streams.vq_parameters,
        weight.parameters(),
        mlx::core::float32,
        "VQ parameters");
}

bool is_ascii(
    std::span<const std::uint8_t> value) {
    return std::all_of(
        value.begin(),
        value.end(),
        [](std::uint8_t byte) {
            return byte <= 0x7fu;
        });
}

array concatenate_1d(
    std::vector<array> values) {
    return mlx::core::contiguous(
        mlx::core::concatenate(
            std::move(values),
            0));
}

array apply_rotation(
    const array& source,
    const RotationSpec& rotation,
    int rows,
    int width) {
    if (
        source.ndim() != 2
        || source.shape(0) != rows
        || source.shape(1) != width
        || rotation.block <= 0
        || rotation.block > 8192
        || (
            rotation.block
            & (rotation.block - 1)
        ) != 0
        || width % rotation.block != 0
        || rotation.signs.dtype()
            != mlx::core::int8
        || rotation.signs.size()
            != static_cast<std::size_t>(width)
    ) {
        throw std::runtime_error(
            "invalid NINTM HSG1 rotation layout");
    }
    const auto grid = checked_product(
        static_cast<std::size_t>(rows),
        256,
        "HSG1 Metal grid");
    if (
        grid
        > static_cast<std::size_t>(
            std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "NINTM HSG1 Metal grid exceeds MLX limits");
    }
    auto outputs = moe_hadamard_kernel()(
        {source, rotation.signs},
        {Shape{rows, width}},
        {source.dtype()},
        {
            static_cast<int>(grid),
            1,
            1,
        },
        {256, 1, 1},
        {
            {"T", source.dtype()},
            {"M", rows},
            {"K", width},
            {"BLOCK", rotation.block},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

int descriptor_with_offset(
    int value,
    std::size_t offset,
    const char* name) {
    if (value < 0) {
        throw std::runtime_error(
            std::string("invalid NINTM ") + name);
    }
    return checked_int(
        checked_add(
            static_cast<std::size_t>(value),
            offset,
            name),
        name);
}

class CccpStreamUnsupported final
    : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct CccpTierLayout {
    int tier = 0;
    int vector_size = 0;
    int entries = 0;
};

CccpTierLayout cccp_tier_layout(
    std::string_view dtype) {
    if (dtype == "TPQ-X" || dtype == "CCCP-X") {
        return {1, 8, 256};
    }
    if (dtype == "TPQ-W" || dtype == "CCCP-W") {
        return {2, 8, 4096};
    }
    if (dtype == "TPQ-V" || dtype == "CCCP-V") {
        return {3, 4, 256};
    }
    if (dtype == "TPQ-VV" || dtype == "CCCP-VV") {
        return {4, 4, 4096};
    }
    if (dtype.rfind("TPQ-", 0) == 0 ||
        dtype.rfind("CCCP-", 0) == 0) {
        throw std::runtime_error(
            "unsupported streamed CCCP cohort "
            "dtype: " + std::string(dtype));
    }
    throw CccpStreamUnsupported(
        "NINTM contains a non-CCCP cohort");
}

bool cccp_index_layout_allowed(
    int entries,
    int bits) {
    if (entries == 256) {
        return bits == 8
            || bits == 12
            || bits == 14;
    }
    if (entries == 4096) {
        return bits == 12
            || bits == 14
            || bits == 16;
    }
    return false;
}

std::uint64_t checked_range_add(
    std::uint64_t left,
    std::uint64_t right,
    const char* name) {
    if (
        left
        > std::numeric_limits<std::uint64_t>::max()
            - right
    ) {
        throw std::runtime_error(
            std::string("CCCP ") + name
            + " overflows");
    }
    return left + right;
}

std::uint64_t checked_range_product(
    std::uint64_t left,
    std::uint64_t right,
    const char* name) {
    if (
        left != 0
        && right
            > std::numeric_limits<std::uint64_t>::max()
                / left
    ) {
        throw std::runtime_error(
            std::string("CCCP ") + name
            + " overflows");
    }
    return left * right;
}

template <typename T>
T cccp_scalar(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset,
    const char* name) {
    if (
        offset > bytes.size()
        || sizeof(T) > bytes.size() - offset
    ) {
        throw std::runtime_error(
            std::string("truncated CCCP ") + name);
    }
    T value{};
    std::memcpy(
        &value,
        bytes.data() + offset,
        sizeof(T));
    return value;
}

std::string cccp_ascii(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset,
    std::size_t count,
    const char* name) {
    if (
        offset > bytes.size()
        || count > bytes.size() - offset
    ) {
        throw std::runtime_error(
            std::string("truncated CCCP ") + name);
    }
    const auto begin =
        bytes.begin()
        + static_cast<std::ptrdiff_t>(offset);
    const auto end =
        begin + static_cast<std::ptrdiff_t>(count);
    if (
        std::any_of(
            begin,
            end,
            [](std::uint8_t value) {
                return value > 0x7fu;
            })
    ) {
        throw std::runtime_error(
            std::string("CCCP ") + name
            + " is not ASCII");
    }
    return {
        reinterpret_cast<const char*>(
            bytes.data() + offset),
        count,
    };
}

array make_cccp_codebook(
    const std::vector<std::uint8_t>& bytes,
    int entries,
    int vector_size,
    const std::string& name) {
    const auto elements = checked_product(
        static_cast<std::size_t>(entries),
        static_cast<std::size_t>(vector_size),
        "CCCP codebook elements");
    if (
        bytes.size()
        != checked_product(
            elements,
            sizeof(float),
            "CCCP codebook bytes")
    ) {
        throw std::runtime_error(
            "CCCP codebook byte count mismatch: "
            + name);
    }
    std::vector<float> values(elements);
    if (!values.empty()) {
        std::memcpy(
            values.data(),
            bytes.data(),
            bytes.size());
    }
    if (
        std::any_of(
            values.begin(),
            values.end(),
            [](float value) {
                return !std::isfinite(value);
            })
    ) {
        throw std::runtime_error(
            "CCCP codebook contains non-finite values: "
            + name);
    }
    const array source(
        values.begin(),
        Shape{checked_int(
            elements,
            "CCCP codebook elements")});
    return mlx::core::contiguous(
        mlx::core::astype(
            source,
            mlx::core::float16));
}

struct CccpStreamPool {
    std::string dtype;
    int vector_size = 0;
    int entries = 0;
    int index_bits = 0;
    int rows_per_expert = 0;
    int columns = 0;
    int blocks = 0;
    int expert_count = 0;
    int codebook_offset = 0;
    std::uint64_t indices_offset = 0;
    std::size_t indices_per_expert = 0;
};

struct CccpExpertLocation {
    std::shared_ptr<const CccpStreamPool> pool;
    int local_expert = 0;
};

struct CccpStreamProjection {
    CccpStreamProjection(
        int expert_count,
        int output_width,
        int input_width,
        array tables,
        std::size_t table_bytes,
        std::vector<
            std::optional<CccpExpertLocation>>
            locations)
        : experts(expert_count),
          out_per_expert(output_width),
          neuron_len(input_width),
          codebooks(std::move(tables)),
          codebook_nbytes(table_bytes),
          experts_by_id(std::move(locations)) {}

    int experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    array codebooks;
    std::size_t codebook_nbytes = 0;
    std::vector<
        std::optional<CccpExpertLocation>>
        experts_by_id;
};

std::uint32_t cccp_read_packed(
    const std::vector<std::uint8_t>& bytes,
    std::size_t bit_offset,
    int bits) {
    const auto byte_offset = bit_offset / 8;
    const auto shift =
        static_cast<unsigned>(bit_offset & 7);
    std::uint32_t packed = 0;
    for (unsigned index = 0; index < 3; ++index) {
        const auto source = byte_offset + index;
        if (source < bytes.size()) {
            packed |=
                static_cast<std::uint32_t>(
                    bytes[source])
                << (8 * index);
        }
    }
    return (
        packed >> shift
    ) & ((std::uint32_t{1} << bits) - 1u);
}

void cccp_write_packed(
    std::vector<std::uint8_t>& bytes,
    std::size_t value_index,
    int bits,
    std::uint32_t value) {
    const auto bit_offset =
        value_index * static_cast<std::size_t>(bits);
    for (int bit = 0; bit < bits; ++bit) {
        if (((value >> bit) & 1u) == 0u) {
            continue;
        }
        const auto target =
            bit_offset
            + static_cast<std::size_t>(bit);
        bytes[target / 8] |=
            static_cast<std::uint8_t>(
                1u << (target & 7));
    }
}

struct CccpCachedExpert {
    CccpCachedExpert(
        std::int32_t global,
        std::shared_ptr<const CccpStreamPool> source,
        array packed,
        std::size_t bytes)
        : expert(global),
          pool(std::move(source)),
          indices(std::move(packed)),
          packed_nbytes(bytes) {}

    std::int32_t expert = 0;
    std::shared_ptr<const CccpStreamPool> pool;
    array indices;
    std::size_t packed_nbytes = 0;
};

} // namespace

struct MlxCccpRoutedWeight::Impl {
    Impl(
        array descriptor_array,
        array index_array,
        array codebook_array,
        int expert_count,
        int output_width,
        int input_width,
        std::size_t active_bytes,
        std::size_t table_bytes)
        : descriptors(std::move(descriptor_array)),
          indices(std::move(index_array)),
          codebooks(std::move(codebook_array)),
          experts(expert_count),
          out_per_expert(output_width),
          neuron_len(input_width),
          packed_bytes(active_bytes),
          codebook_bytes(table_bytes) {}

    array descriptors;
    array indices;
    array codebooks;
    int experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::size_t packed_bytes = 0;
    std::size_t codebook_bytes = 0;
};

struct MlxNintMoeOffloadCache::Impl {
    struct Key {
        std::string name;
        std::int32_t expert = 0;

        bool operator==(const Key& other) const noexcept {
            return expert == other.expert
                && name == other.name;
        }
    };

    struct KeyHash {
        std::size_t operator()(
            const Key& value) const noexcept {
            const auto first =
                std::hash<std::string>{}(value.name);
            const auto second =
                std::hash<std::int32_t>{}(
                    value.expert);
            return first
                ^ (
                    second
                    + std::size_t{
                        0x9e3779b97f4a7c15ull}
                    + (first << 6)
                    + (first >> 2)
                );
        }
    };

    struct CacheValue {
        Key key;
        std::shared_ptr<const CccpCachedExpert>
            weight;
    };

    using Lru = std::list<CacheValue>;
    using ProjectionCache = std::unordered_map<
        std::string,
        std::shared_ptr<
            const CccpStreamProjection>>;
    using ExpertCache = std::unordered_map<
        Key,
        Lru::iterator,
        KeyHash>;

    Impl(
        const MfqContainer& source,
        std::size_t limit,
        int expert_count)
        : model(source),
          cache_limit(limit),
          experts(expert_count) {
        if (experts <= 0) {
            throw std::invalid_argument(
                "NINTM offload expert count must "
                "be positive");
        }
    }

    std::shared_ptr<const CccpStreamProjection>
    parse_projection(
        const std::string& name) {
        const auto& record = model.record(name);
        if (record.dtype != "NINTM") {
            throw CccpStreamUnsupported(
                "expert record is not NINTM: "
                + name);
        }
        constexpr std::uint64_t header_size = 20;
        constexpr std::uint64_t pool_header_size = 24;
        constexpr std::uint64_t pq_prefix_size = 44;
        if (record.nbytes < header_size) {
            throw std::runtime_error(
                "truncated streamed NINTM header: "
                + name);
        }
        const auto header =
            model.read_range(
                name,
                0,
                header_size);
        const std::string_view magic(
            reinterpret_cast<const char*>(
                header.data()),
            4);
        if (magic == "NIM1") {
            throw CccpStreamUnsupported(
                "NIM1 expert records are not "
                "streamable");
        }
        if (magic != "NIM2") {
            throw std::runtime_error(
                "invalid streamed NINTM magic: "
                + name);
        }
        const auto record_experts =
            cccp_scalar<std::uint32_t>(
                header,
                4,
                "NINTM expert count");
        const auto rows_per_expert =
            cccp_scalar<std::uint32_t>(
                header,
                8,
                "NINTM output width");
        const auto columns =
            cccp_scalar<std::uint32_t>(
                header,
                12,
                "NINTM input width");
        const auto pool_count =
            cccp_scalar<std::uint32_t>(
                header,
                16,
                "NINTM pool count");
        if (
            record_experts
                != static_cast<std::uint32_t>(
                    experts)
            || rows_per_expert == 0
            || columns == 0
            || rows_per_expert
                > static_cast<std::uint32_t>(
                    std::numeric_limits<int>::max())
            || columns
                > static_cast<std::uint32_t>(
                    std::numeric_limits<int>::max())
            || pool_count == 0
            || pool_count > record_experts
        ) {
            throw std::runtime_error(
                "invalid streamed NINTM dimensions: "
                + name);
        }

        std::vector<
            std::optional<CccpExpertLocation>>
            locations(
                static_cast<std::size_t>(experts));
        std::vector<array> codebooks;
        codebooks.reserve(pool_count);
        std::size_t codebook_elements = 0;
        std::uint64_t offset = header_size;

        for (
            std::uint32_t pool_index = 0;
            pool_index < pool_count;
            ++pool_index
        ) {
            if (
                offset > record.nbytes
                || pool_header_size
                    > record.nbytes - offset
            ) {
                throw std::runtime_error(
                    "truncated streamed NINTM pool "
                    "header: " + name);
            }
            const auto pool_header =
                model.read_range(
                    name,
                    offset,
                    pool_header_size);
            const auto pool_experts =
                cccp_scalar<std::uint32_t>(
                    pool_header,
                    0,
                    "pool expert count");
            const auto dtype_bytes =
                cccp_scalar<std::uint32_t>(
                    pool_header,
                    4,
                    "pool dtype length");
            const auto payload_bytes =
                cccp_scalar<std::uint64_t>(
                    pool_header,
                    8,
                    "pool payload length");
            const auto runtime_bytes =
                cccp_scalar<std::uint64_t>(
                    pool_header,
                    16,
                    "pool runtime length");
            if (
                pool_experts == 0
                || pool_experts > record_experts
                || dtype_bytes == 0
                || dtype_bytes > 32
            ) {
                throw std::runtime_error(
                    "invalid streamed NINTM pool "
                    "metadata: " + name);
            }
            offset = checked_range_add(
                offset,
                pool_header_size,
                "pool offset");
            const auto ids_bytes =
                checked_range_product(
                    pool_experts,
                    sizeof(std::int32_t),
                    "expert ID bytes");
            const auto metadata_bytes =
                checked_range_add(
                    ids_bytes,
                    dtype_bytes,
                    "pool metadata bytes");
            const auto metadata_end =
                checked_range_add(
                    offset,
                    metadata_bytes,
                    "pool metadata end");
            const auto payload_start =
                checked_range_add(
                    metadata_end,
                    runtime_bytes,
                    "pool payload offset");
            const auto payload_end =
                checked_range_add(
                    payload_start,
                    payload_bytes,
                    "pool payload end");
            if (payload_end > record.nbytes) {
                throw std::runtime_error(
                    "truncated streamed NINTM pool: "
                    + name);
            }
            const auto metadata =
                model.read_range(
                    name,
                    offset,
                    metadata_bytes);
            const auto dtype = cccp_ascii(
                metadata,
                static_cast<std::size_t>(
                    ids_bytes),
                dtype_bytes,
                "pool dtype");
            const auto layout =
                cccp_tier_layout(dtype);
            if (
                runtime_bytes != 0
                || payload_bytes < pq_prefix_size
            ) {
                throw std::runtime_error(
                    "CCCP pool has invalid runtime/"
                    "payload metadata: " + name);
            }

            const auto prefix =
                model.read_range(
                    name,
                    payload_start,
                    pq_prefix_size);
            const std::string_view pq_magic(
                reinterpret_cast<const char*>(
                    prefix.data()),
                4);
            const int version =
                prefix.at(4);
            const int tier =
                prefix.at(5);
            const int vector_size =
                prefix.at(6);
            const int index_bits =
                prefix.at(7);
            const auto axis =
                cccp_scalar<std::int32_t>(
                    prefix,
                    8,
                    "PQ axis");
            const auto neuron_len =
                cccp_scalar<std::int32_t>(
                    prefix,
                    12,
                    "PQ neuron length");
            const auto dimensions =
                cccp_scalar<std::uint32_t>(
                    prefix,
                    16,
                    "PQ dimension count");
            const auto entries =
                cccp_scalar<std::uint32_t>(
                    prefix,
                    20,
                    "PQ codebook entries");
            const auto shape_rows =
                cccp_scalar<std::int64_t>(
                    prefix,
                    24,
                    "PQ row shape");
            const auto shape_columns =
                cccp_scalar<std::int64_t>(
                    prefix,
                    32,
                    "PQ column shape");
            const auto row_tail =
                cccp_scalar<std::uint32_t>(
                    prefix,
                    40,
                    "PQ row count");
            const auto expected_rows =
                checked_range_product(
                    pool_experts,
                    rows_per_expert,
                    "PQ rows");
            if (
                pq_magic != "CPQ1"
                || version != 1
                || tier != layout.tier
                || vector_size
                    != layout.vector_size
                || entries
                    != static_cast<std::uint32_t>(
                        layout.entries)
                || !cccp_index_layout_allowed(
                    layout.entries,
                    index_bits)
                || axis != 0
                || neuron_len
                    != static_cast<std::int32_t>(
                        columns)
                || dimensions != 2
                || shape_rows
                    != static_cast<std::int64_t>(
                        expected_rows)
                || shape_columns
                    != static_cast<std::int64_t>(
                        columns)
                || row_tail != expected_rows
                || columns
                    % static_cast<std::uint32_t>(
                        vector_size)
                    != 0
            ) {
                throw std::runtime_error(
                    "inconsistent streamed CCCP pool "
                    "header: " + name);
            }
            const int blocks =
                static_cast<int>(columns)
                / vector_size;
            const auto table_elements =
                checked_range_product(
                    entries,
                    vector_size,
                    "codebook elements");
            const auto table_bytes =
                checked_range_product(
                    table_elements,
                    sizeof(float),
                    "codebook bytes");
            const auto index_count =
                checked_range_product(
                    expected_rows,
                    static_cast<std::uint64_t>(
                        blocks),
                    "index count");
            const auto index_bits_total =
                checked_range_product(
                    index_count,
                    static_cast<std::uint64_t>(
                        index_bits),
                    "index bits");
            const auto index_bytes =
                checked_range_add(
                    index_bits_total,
                    7,
                    "index rounding")
                / 8;
            const auto expected_payload =
                checked_range_add(
                    checked_range_add(
                        pq_prefix_size,
                        table_bytes,
                        "payload table end"),
                    index_bytes,
                    "payload index end");
            if (payload_bytes != expected_payload) {
                throw std::runtime_error(
                    "streamed CCCP pool payload length "
                    "mismatch: " + name);
            }
            const auto table_start =
                checked_range_add(
                    payload_start,
                    pq_prefix_size,
                    "codebook offset");
            const auto index_start =
                checked_range_add(
                    table_start,
                    table_bytes,
                    "index offset");
            const auto raw_table =
                model.read_range(
                    name,
                    table_start,
                    table_bytes);
            codebooks.push_back(
                make_cccp_codebook(
                    raw_table,
                    layout.entries,
                    vector_size,
                    name));
            const auto table_offset =
                checked_int(
                    codebook_elements,
                    "CCCP codebook offset");
            codebook_elements = checked_add(
                codebook_elements,
                static_cast<std::size_t>(
                    table_elements),
                "CCCP codebook elements");

            if (
                index_bits_total % 8 != 0
                && index_bytes != 0
            ) {
                const auto last =
                    model.read_range(
                        name,
                        index_start
                            + index_bytes - 1,
                        1);
                const unsigned used =
                    static_cast<unsigned>(
                        index_bits_total & 7u);
                const auto padding_mask =
                    static_cast<std::uint8_t>(
                        0xffu << used);
                if ((last.front() & padding_mask) != 0) {
                    throw std::runtime_error(
                        "streamed CCCP index padding is "
                        "non-zero: " + name);
                }
            }

            auto pool =
                std::make_shared<CccpStreamPool>();
            pool->dtype = dtype;
            pool->vector_size = vector_size;
            pool->entries = layout.entries;
            pool->index_bits = index_bits;
            pool->rows_per_expert =
                static_cast<int>(
                    rows_per_expert);
            pool->columns =
                static_cast<int>(columns);
            pool->blocks = blocks;
            pool->expert_count =
                static_cast<int>(pool_experts);
            pool->codebook_offset =
                table_offset;
            pool->indices_offset =
                index_start;
            pool->indices_per_expert =
                checked_product(
                    static_cast<std::size_t>(
                        rows_per_expert),
                    static_cast<std::size_t>(
                        blocks),
                    "CCCP expert index count");

            for (
                std::uint32_t local = 0;
                local < pool_experts;
                ++local
            ) {
                const auto expert =
                    cccp_scalar<std::int32_t>(
                        metadata,
                        static_cast<std::size_t>(
                            local)
                            * sizeof(std::int32_t),
                        "global expert ID");
                if (
                    expert < 0
                    || expert >= experts
                    || locations[
                        static_cast<std::size_t>(
                            expert)].has_value()
                ) {
                    throw std::runtime_error(
                        "invalid or duplicate streamed "
                        "CCCP global expert ID: "
                        + name);
                }
                locations[
                    static_cast<std::size_t>(
                        expert)] =
                    CccpExpertLocation{
                        pool,
                        static_cast<int>(local),
                    };
            }
            offset = payload_end;
        }
        if (offset != record.nbytes) {
            throw std::runtime_error(
                "invalid streamed NINTM tail: "
                + name);
        }
        array combined = codebooks.size() == 1
            ? codebooks.front()
            : mlx::core::contiguous(
                mlx::core::concatenate(
                    std::move(codebooks),
                    0));
        return std::make_shared<
            CccpStreamProjection>(
                experts,
                static_cast<int>(
                    rows_per_expert),
                static_cast<int>(columns),
                std::move(combined),
                checked_product(
                    codebook_elements,
                    sizeof(std::uint16_t),
                    "resident CCCP codebook bytes"),
                std::move(locations));
    }

    std::shared_ptr<const CccpStreamProjection>
    projection_locked(
        const std::string& name) {
        const auto found =
            projections.find(name);
        if (found != projections.end()) {
            return found->second;
        }
        auto result = parse_projection(name);
        projections.emplace(name, result);
        return result;
    }

    std::shared_ptr<const CccpCachedExpert>
    load_expert(
        const std::string& name,
        std::int32_t expert,
        const CccpExpertLocation& location) {
        const auto& pool = *location.pool;
        if (
            location.local_expert < 0
            || location.local_expert
                >= pool.expert_count
        ) {
            throw std::runtime_error(
                "streamed CCCP local expert is "
                "out of range: " + name);
        }
        const auto expert_bits =
            checked_range_product(
                pool.indices_per_expert,
                static_cast<std::uint64_t>(
                    pool.index_bits),
                "expert index bits");
        const auto source_bit =
            checked_range_product(
                static_cast<std::uint64_t>(
                    location.local_expert),
                expert_bits,
                "expert index offset");
        const auto source_byte =
            source_bit / 8;
        const auto source_shift =
            static_cast<std::size_t>(
                source_bit & 7u);
        const auto source_span_bits =
            checked_range_add(
                source_shift,
                expert_bits,
                "expert source bits");
        const auto source_bytes =
            checked_range_add(
                source_span_bits,
                7,
                "expert source bytes")
            / 8;
        const auto raw = model.read_range(
            name,
            checked_range_add(
                pool.indices_offset,
                source_byte,
                "expert file offset"),
            source_bytes);
        const auto packed_nbytes =
            checked_range_add(
                expert_bits,
                7,
                "expert packed bytes")
            / 8;
        if (
            packed_nbytes
            > static_cast<std::uint64_t>(
                std::numeric_limits<
                    std::size_t>::max())
        ) {
            throw std::runtime_error(
                "streamed CCCP expert index stream "
                "is too large: " + name);
        }
        std::vector<std::uint8_t> packed(
            static_cast<std::size_t>(
                packed_nbytes),
            0);
        for (
            std::size_t index = 0;
            index < pool.indices_per_expert;
            ++index
        ) {
            const auto value =
                cccp_read_packed(
                    raw,
                    source_shift
                        + index
                            * static_cast<
                                std::size_t>(
                                pool.index_bits),
                    pool.index_bits);
            if (
                value
                >= static_cast<std::uint32_t>(
                    pool.entries)
            ) {
                throw std::runtime_error(
                    "streamed CCCP expert references "
                    "a missing codeword: " + name);
            }
            cccp_write_packed(
                packed,
                index,
                pool.index_bits,
                value);
        }
        auto indices =
            make_raw_array(
                std::move(packed),
                mlx::core::uint8);
        return std::make_shared<
            CccpCachedExpert>(
                expert,
                location.pool,
                std::move(indices),
                static_cast<std::size_t>(
                    packed_nbytes));
    }

    // Own the record table and source paths.  Streamed layers frequently
    // outlive the MfqContainer object used by their load call.
    MfqContainer model;
    const std::size_t cache_limit = 0;
    const int experts = 0;
    mutable std::mutex mutex;
    ProjectionCache projections;
    Lru lru;
    ExpertCache cache;
    std::size_t resident_bytes = 0;
};

MlxCccpRoutedWeight::MlxCccpRoutedWeight(
    std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {
    if (!impl_) {
        throw std::invalid_argument(
            "streamed CCCP implementation cannot "
            "be null");
    }
}

array MlxCccpRoutedWeight::routed_matmul(
    const array& input,
    const array& expert_ids) const {
    auto ids = mlx::core::contiguous(
        mlx::core::astype(
            expert_ids,
            mlx::core::int32));
    if (ids.ndim() != 2) {
        throw std::invalid_argument(
            "streamed CCCP expert IDs must have "
            "[tokens,routes] shape");
    }
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    const bool shared_input =
        input.ndim() == 2
        && input.shape(0) == tokens
        && input.shape(1) == impl_->neuron_len;
    if (
        !shared_input
        && (
            input.ndim() != 3
            || input.shape(0) != tokens
            || input.shape(1) != routes
            || input.shape(2)
                != impl_->neuron_len
        )
    ) {
        throw std::invalid_argument(
            "streamed CCCP input must have "
            "[tokens,K] or [tokens,routes,K] shape");
    }
    auto source = input;
    if (
        source.dtype() != mlx::core::float16
        && source.dtype() != mlx::core::float32
    ) {
        source = mlx::core::astype(
            source,
            mlx::core::float16);
    }
    source = mlx::core::contiguous(source);
    const Shape output_shape{
        tokens,
        routes,
        impl_->out_per_expert,
    };
    if (tokens == 0 || routes == 0) {
        return mlx::core::zeros(
            output_shape,
            source.dtype());
    }
    auto tasks = checked_product(
        checked_product(
            static_cast<std::size_t>(tokens),
            static_cast<std::size_t>(routes),
            "streamed CCCP route count"),
        static_cast<std::size_t>(
            impl_->out_per_expert),
        "streamed CCCP task count");
    const auto grid = checked_product(
        tasks,
        32,
        "streamed CCCP Metal grid");
    if (
        grid
        > static_cast<std::size_t>(
            std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "streamed CCCP Metal grid exceeds "
            "MLX limits");
    }
    auto outputs = cccp_moe_kernel()(
        {
            impl_->descriptors,
            impl_->indices,
            impl_->codebooks,
            source,
            ids,
        },
        {output_shape},
        {source.dtype()},
        {
            static_cast<int>(grid),
            1,
            1,
        },
        {32, 1, 1},
        {
            {"T", source.dtype()},
            {"TOKENS", tokens},
            {"ROUTES", routes},
            {"EXPERTS", impl_->experts},
            {"OUT", impl_->out_per_expert},
            {"K", impl_->neuron_len},
            {
                "DESCRIPTOR_SIZE",
                kCccpDescriptorSize,
            },
            {
                "SHARED_INPUT",
                static_cast<int>(
                    shared_input),
            },
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

int MlxCccpRoutedWeight::experts() const noexcept {
    return impl_->experts;
}

int MlxCccpRoutedWeight::out_per_expert() const noexcept {
    return impl_->out_per_expert;
}

int MlxCccpRoutedWeight::neuron_len() const noexcept {
    return impl_->neuron_len;
}

std::size_t
MlxCccpRoutedWeight::packed_nbytes() const noexcept {
    return impl_->packed_bytes;
}

std::size_t
MlxCccpRoutedWeight::shared_codebook_nbytes() const noexcept {
    return impl_->codebook_bytes;
}

MlxNintMoeOffloadCache::MlxNintMoeOffloadCache(
    const MfqContainer& model,
    std::size_t cache_limit_bytes,
    int experts)
    : impl_(std::make_unique<Impl>(
          model,
          cache_limit_bytes,
          experts)) {}

MlxNintMoeOffloadCache::~MlxNintMoeOffloadCache() =
    default;

bool MlxNintMoeOffloadCache::can_offload(
    const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    try {
        (void)impl_->projection_locked(name);
        return true;
    } catch (const CccpStreamUnsupported&) {
        return false;
    }
}

MlxNintMoeProjectionInfo
MlxNintMoeOffloadCache::projection_info(
    const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    const auto projection =
        impl_->projection_locked(name);
    MlxNintMoeProjectionInfo result;
    result.experts = projection->experts;
    result.out_per_expert =
        projection->out_per_expert;
    result.neuron_len =
        projection->neuron_len;
    result.shared_codebook_nbytes =
        projection->codebook_nbytes;
    for (
        int expert = 0;
        expert < projection->experts;
        ++expert
    ) {
        if (
            projection->experts_by_id[
                static_cast<std::size_t>(
                    expert)].has_value()
        ) {
            result.available_experts.push_back(
                expert);
        }
    }
    return result;
}

std::vector<std::uint8_t>
MlxNintMoeOffloadCache::availability(
    const std::string& name) {
    std::lock_guard lock(impl_->mutex);
    const auto projection =
        impl_->projection_locked(name);
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(
            projection->experts),
        0);
    for (
        int expert = 0;
        expert < projection->experts;
        ++expert
    ) {
        result[static_cast<std::size_t>(
            expert)] =
            static_cast<std::uint8_t>(
                projection->experts_by_id[
                    static_cast<std::size_t>(
                        expert)].has_value());
    }
    return result;
}

MlxCccpRoutedWeight
MlxNintMoeOffloadCache::grouped(
    const std::string& name,
    const std::vector<std::int32_t>&
        active_experts) {
    std::lock_guard lock(impl_->mutex);
    const auto parsed =
        impl_->projections.find(name);
    const bool projection_is_new =
        parsed == impl_->projections.end();
    const auto projection =
        projection_is_new
        ? impl_->parse_projection(name)
        : parsed->second;

    // Validate the entire active set before reading or staging a single
    // expert.  In particular, a late invalid/unavailable ID must not touch
    // the LRU position of an earlier cached ID.
    std::vector<Impl::Key> ordered_keys;
    ordered_keys.reserve(active_experts.size());
    std::vector<std::shared_ptr<
        const CccpCachedExpert>> active;
    active.reserve(active_experts.size());
    std::unordered_set<
        Impl::Key,
        Impl::KeyHash>
        active_keys;
    active_keys.reserve(active_experts.size());
    for (const auto expert : active_experts) {
        if (
            expert < 0
            || expert >= projection->experts
        ) {
            throw std::out_of_range(
                "streamed CCCP global expert ID "
                "is out of range");
        }
        const auto& location =
            projection->experts_by_id[
                static_cast<std::size_t>(
                    expert)];
        if (!location.has_value()) {
            throw std::runtime_error(
                "streamed CCCP global expert "
                + std::to_string(expert)
                + " is unavailable in " + name);
        }
        Impl::Key key{name, expert};
        if (!active_keys.emplace(key).second) {
            continue;
        }
        ordered_keys.push_back(std::move(key));
    }

    // New expert arrays and their map/list nodes live outside the residency
    // until every fallible operation, including the returned MLX graph
    // construction, has succeeded.
    Impl::Lru staged_lru;
    Impl::ExpertCache staged_cache;
    staged_cache.reserve(ordered_keys.size());
    std::size_t staged_bytes = 0;
    for (const auto& key : ordered_keys) {
        const auto cached =
            impl_->cache.find(key);
        if (cached != impl_->cache.end()) {
            active.push_back(
                cached->second->weight);
            continue;
        }
        const auto& location =
            projection->experts_by_id[
                static_cast<std::size_t>(
                    key.expert)];
        auto weight = impl_->load_expert(
            name,
            key.expert,
            *location);
        staged_bytes = checked_add(
            staged_bytes,
            weight->packed_nbytes,
            "staged CCCP packed bytes");
        staged_lru.push_back({
            key,
            weight,
        });
        const auto inserted =
            std::prev(staged_lru.end());
        const auto cached_insert =
            staged_cache.emplace(
                inserted->key,
                inserted);
        if (!cached_insert.second) {
            throw std::logic_error(
                "duplicate staged CCCP expert");
        }
        active.push_back(std::move(weight));
    }

    std::vector<std::int32_t> descriptors(
        checked_product(
            static_cast<std::size_t>(
                projection->experts),
            static_cast<std::size_t>(
                kCccpDescriptorSize),
            "CCCP descriptor count"),
        0);
    std::vector<array> index_arrays;
    index_arrays.reserve(active.size() + 1);
    std::size_t index_offset = 0;
    for (const auto& weight : active) {
        const auto base = checked_product(
            static_cast<std::size_t>(
                weight->expert),
            static_cast<std::size_t>(
                kCccpDescriptorSize),
            "CCCP descriptor offset");
        descriptors[
            base + kCccpBits] =
            weight->pool->index_bits;
        descriptors[
            base + kCccpIndexOffset] =
            checked_int(
                index_offset,
                "CCCP active index offset");
        descriptors[
            base + kCccpCodebookOffset] =
            weight->pool->codebook_offset;
        descriptors[
            base + kCccpVectorSize] =
            weight->pool->vector_size;
        descriptors[
            base + kCccpBlocks] =
            weight->pool->blocks;
        index_arrays.push_back(
            weight->indices);
        index_offset = checked_add(
            index_offset,
            weight->packed_nbytes,
            "CCCP active index bytes");
    }
    // The bit reader may issue a three-byte load for the final value.
    index_arrays.push_back(
        mlx::core::zeros(
            Shape{2},
            mlx::core::uint8));
    auto combined_indices =
        mlx::core::contiguous(
            mlx::core::concatenate(
                std::move(index_arrays),
                0));
    auto descriptor_array =
        make_int32_array(
            descriptors,
            Shape{
                projection->experts,
                kCccpDescriptorSize,
            });
    const auto packed_bytes = checked_add(
        combined_indices.nbytes(),
        descriptor_array.nbytes(),
        "CCCP active packed bytes");
    MlxCccpRoutedWeight result(
        std::make_shared<
            MlxCccpRoutedWeight::Impl>(
                std::move(descriptor_array),
                std::move(combined_indices),
                projection->codebooks,
                projection->experts,
                projection->out_per_expert,
                projection->neuron_len,
                packed_bytes,
                projection->codebook_nbytes));
    auto committed_bytes = checked_add(
        impl_->resident_bytes,
        staged_bytes,
        "resident CCCP packed bytes");

    // Preallocate both destination hash tables before publishing any staged
    // node.  reserve() may throw, but it cannot change the logical
    // cache/LRU/resident-byte state.
    impl_->cache.reserve(
        checked_add(
            impl_->cache.size(),
            staged_cache.size(),
            "resident CCCP cache entries"));
    Impl::ProjectionCache staged_projections;
    if (projection_is_new) {
        staged_projections.emplace(
            name,
            projection);
        impl_->projections.reserve(
            checked_add(
                impl_->projections.size(),
                std::size_t{1},
                "resident CCCP projections"));
    }

    // Transferring an unordered_map node does not allocate after reserve.
    // Keep exact published-node addresses so an implementation-level
    // exception from insertion can still roll the transaction back before
    // the LRU is touched.
    std::vector<const Impl::Key*> published_keys;
    published_keys.reserve(staged_cache.size());
    bool projection_published = false;
    try {
        while (!staged_cache.empty()) {
            auto node =
                staged_cache.extract(
                    staged_cache.begin());
            auto published =
                impl_->cache.insert(
                    std::move(node));
            if (!published.inserted) {
                throw std::logic_error(
                    "CCCP staged cache key already "
                    "exists");
            }
            published_keys.push_back(
                &published.position->first);
        }
        if (projection_is_new) {
            auto node =
                staged_projections.extract(
                    staged_projections.begin());
            auto published =
                impl_->projections.insert(
                    std::move(node));
            if (!published.inserted) {
                throw std::logic_error(
                    "CCCP staged projection already "
                    "exists");
            }
            projection_published = true;
        }
    } catch (...) {
        if (projection_published) {
            impl_->projections.erase(name);
        }
        for (const auto* key : published_keys) {
            const auto found =
                impl_->cache.find(*key);
            if (found != impl_->cache.end()) {
                impl_->cache.erase(found);
            }
        }
        throw;
    }

    // From this point onward all operations are non-allocating.  Publish the
    // staged list nodes, reproduce the request-order LRU touches, evict only
    // inactive entries, and expose resident_bytes once at the final value.
    impl_->lru.splice(
        impl_->lru.end(),
        staged_lru);
    for (const auto& key : ordered_keys) {
        const auto cached =
            impl_->cache.find(key);
        impl_->lru.splice(
            impl_->lru.end(),
            impl_->lru,
            cached->second);
    }
    while (
        committed_bytes > impl_->cache_limit
        && !impl_->lru.empty()
    ) {
        const auto candidate =
            std::find_if(
                impl_->lru.begin(),
                impl_->lru.end(),
                [&](const auto& item) {
                    return active_keys.find(
                        item.key)
                        == active_keys.end();
                });
        if (candidate == impl_->lru.end()) {
            break;
        }
        committed_bytes -=
            candidate->weight->packed_nbytes;
        const auto cached =
            impl_->cache.find(candidate->key);
        impl_->cache.erase(cached);
        impl_->lru.erase(candidate);
    }
    impl_->resident_bytes = committed_bytes;
    return result;
}

std::size_t
MlxNintMoeOffloadCache::cache_limit_bytes() const noexcept {
    return impl_->cache_limit;
}

std::size_t
MlxNintMoeOffloadCache::resident_packed_bytes() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->resident_bytes;
}

std::size_t
MlxNintMoeOffloadCache::cached_expert_count() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->cache.size();
}

void MlxNintMoeOffloadCache::discard_record(
    const std::string& name) noexcept {
    std::lock_guard lock(impl_->mutex);
    for (
        auto item = impl_->lru.begin();
        item != impl_->lru.end();
    ) {
        if (item->key.name != name) {
            ++item;
            continue;
        }
        impl_->resident_bytes -=
            item->weight->packed_nbytes;
        const auto cached =
            impl_->cache.find(item->key);
        if (cached != impl_->cache.end()) {
            impl_->cache.erase(cached);
        }
        item = impl_->lru.erase(item);
    }
    impl_->projections.erase(name);
}

void MlxNintMoeOffloadCache::clear() {
    std::lock_guard lock(impl_->mutex);
    impl_->cache.clear();
    impl_->lru.clear();
    impl_->projections.clear();
    impl_->resident_bytes = 0;
}

struct MlxNintMoeWeight::Impl {
    array descriptors;
    array nint_q;
    array nint_sub_scale;
    array nint_sub_min;
    array nint_anchor_scale;
    array nint_anchor_min;
    array q8_q;
    array q8_scales;
    array vq_indices;
    array vq_state;
    array vq_aux;
    array vq_anchors;
    array vq_codebooks;
    array vq_scales;
    array vq_state_to_codebank;
    array vq_banks;
    array vq_parameters;
    std::vector<RotationSpec> rotations;
    std::vector<std::int32_t> descriptor_values;
    int experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    int projections = 0;
    std::uint32_t family_mask = 0;
    std::uint32_t vq_profile_mask = 0;
    bool jsc_execution_layout = false;
    bool npq_grouped_indices = true;
    bool native_primitive = true;
    bool grouped_vq_mmq = false;
    int k_lanes_override = 0;
    std::size_t packed_bytes = 0;

    Impl(
        array descriptor_array,
        array nint_q_array,
        array nint_sub_scale_array,
        array nint_sub_min_array,
        array nint_anchor_scale_array,
        array nint_anchor_min_array,
        array q8_q_array,
        array q8_scale_array,
        array vq_indices_array,
        array vq_state_array,
        array vq_aux_array,
        array vq_anchor_array,
        array vq_codebook_array,
        array vq_scale_array,
        array vq_state_bank_array,
        array vq_bank_array,
        array vq_parameter_array,
        std::vector<RotationSpec> rotation_values,
        std::vector<std::int32_t> values,
        int expert_count,
        int output_width,
        int input_width,
        int projection_count)
        : descriptors(std::move(descriptor_array)),
          nint_q(std::move(nint_q_array)),
          nint_sub_scale(
              std::move(nint_sub_scale_array)),
          nint_sub_min(
              std::move(nint_sub_min_array)),
          nint_anchor_scale(
              std::move(nint_anchor_scale_array)),
          nint_anchor_min(
              std::move(nint_anchor_min_array)),
          q8_q(std::move(q8_q_array)),
          q8_scales(std::move(q8_scale_array)),
          vq_indices(std::move(vq_indices_array)),
          vq_state(std::move(vq_state_array)),
          vq_aux(std::move(vq_aux_array)),
          vq_anchors(std::move(vq_anchor_array)),
          vq_codebooks(std::move(vq_codebook_array)),
          vq_scales(std::move(vq_scale_array)),
          vq_state_to_codebank(
              std::move(vq_state_bank_array)),
          vq_banks(std::move(vq_bank_array)),
          vq_parameters(
              std::move(vq_parameter_array)),
          rotations(std::move(rotation_values)),
          descriptor_values(std::move(values)),
          experts(expert_count),
          out_per_expert(output_width),
          neuron_len(input_width),
          projections(projection_count) {
        grouped_vq_mmq = !descriptor_values.empty();
        for (
            std::size_t base = 0;
            base + kDescriptorSize
                <= descriptor_values.size();
            base += kDescriptorSize
        ) {
            const auto family = descriptor_values[
                base + kFamily];
            if (family >= 0 && family < 3) {
                family_mask |= std::uint32_t{1}
                    << static_cast<unsigned>(family);
            }
            if (family == kFamilyVq) {
                const auto profile = descriptor_values[
                    base + kVqProfile];
                if (profile >= 0 && profile < 6) {
                    vq_profile_mask |= std::uint32_t{1}
                        << static_cast<unsigned>(profile);
                }
            }
            if (
                family != kFamilyVq
                || descriptor_values[base + kVqGroupSize] != 24
                || (
                    descriptor_values[base + kVqVectorSize] != 4
                    && descriptor_values[base + kVqVectorSize] != 8
                )
            ) {
                grouped_vq_mmq = false;
            }
            if (
                descriptor_values[
                    base + kVqJscExecution] != 0
            ) {
                jsc_execution_layout = true;
            }
        }
        const char* specialize_env = std::getenv(
            "MFQ_METAL_NINTM_SPECIALIZE");
        if (
            specialize_env != nullptr
            && std::string_view(specialize_env) == "0"
        ) {
            family_mask = 7;
            vq_profile_mask = 63;
        }
        const char* npq_indices_env = std::getenv(
            "MFQ_METAL_NINTM_NPQ_INDICES");
        npq_grouped_indices =
            npq_indices_env == nullptr
            || std::string_view(npq_indices_env) != "0";
        const char* native_env = std::getenv(
            "MFQ_METAL_NINTM_NATIVE_PRIMITIVE");
        native_primitive = native_env == nullptr
            || std::string_view(native_env) != "0";
        const char* k_lanes_env = std::getenv(
            "MFQ_METAL_NINTM_K_LANES");
        if (k_lanes_env != nullptr) {
            const auto value = std::string_view(k_lanes_env);
            if (value == "8") {
                k_lanes_override = 8;
            } else if (value == "16") {
                k_lanes_override = 16;
            } else if (value == "32") {
                k_lanes_override = 32;
            }
        }
        packed_bytes =
            descriptors.nbytes()
            + nint_q.nbytes()
            + nint_sub_scale.nbytes()
            + nint_sub_min.nbytes()
            + nint_anchor_scale.nbytes()
            + nint_anchor_min.nbytes()
            + q8_q.nbytes()
            + q8_scales.nbytes()
            + vq_indices.nbytes()
            + vq_state.nbytes()
            + vq_aux.nbytes()
            + vq_anchors.nbytes()
            + vq_codebooks.nbytes()
            + vq_scales.nbytes()
            + vq_state_to_codebank.nbytes()
            + vq_banks.nbytes()
            + vq_parameters.nbytes();
        for (const auto& rotation : rotations) {
            packed_bytes += rotation.signs.nbytes();
        }
    }
};

MlxNintMoeWeight::MlxNintMoeWeight(
    std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {
    if (!impl_) {
        throw std::invalid_argument(
            "NINTM implementation cannot be null");
    }
}

MlxNintMoeWeight MlxNintMoeWeight::from_blob(
    std::span<const std::uint8_t> blob) {
    BlobCursor cursor(blob);
    const auto magic_bytes = cursor.bytes(4, "header");
    const std::string_view magic(
        reinterpret_cast<const char*>(
            magic_bytes.data()),
        magic_bytes.size());
    if (magic != "NIM1" && magic != "NIM2") {
        throw std::runtime_error("invalid NINTM magic");
    }

    const int expert_count = checked_positive(
        cursor.scalar<std::uint32_t>("expert count"),
        "expert count");
    const int output_width = checked_positive(
        cursor.scalar<std::uint32_t>(
            "output width"),
        "output width");
    const int input_width = checked_positive(
        cursor.scalar<std::uint32_t>(
            "neuron length"),
        "neuron length");
    const auto pool_count =
        cursor.scalar<std::uint32_t>("pool count");
    if (
        pool_count == 0
        || pool_count
            > static_cast<std::uint32_t>(
                expert_count)
    ) {
        throw std::runtime_error(
            "invalid NINTM pool count");
    }
    if (
        static_cast<std::size_t>(expert_count)
        > cursor.remaining() / sizeof(std::int32_t)
    ) {
        throw std::runtime_error(
            "NINTM expert count exceeds its payload");
    }

    std::vector<std::int32_t> descriptors(
        checked_product(
            static_cast<std::size_t>(expert_count),
            static_cast<std::size_t>(kDescriptorSize),
            "descriptor count"),
        0);
    std::vector<int> owners(
        static_cast<std::size_t>(expert_count),
        -1);
    PackedStreams streams;
    std::vector<RotationSpec> rotations;

    for (
        std::uint32_t pool = 0;
        pool < pool_count;
        ++pool
    ) {
        std::uint32_t count = 0;
        std::uint32_t dtype_bytes = 0;
        std::uint64_t payload_bytes = 0;
        std::uint64_t runtime_bytes = 0;
        if (magic == "NIM1") {
            count =
                cursor.scalar<std::uint32_t>(
                    "pool header");
            payload_bytes =
                cursor.scalar<std::uint64_t>(
                    "pool header");
        } else {
            count =
                cursor.scalar<std::uint32_t>(
                    "v2 pool header");
            dtype_bytes =
                cursor.scalar<std::uint32_t>(
                    "v2 pool header");
            payload_bytes =
                cursor.scalar<std::uint64_t>(
                    "v2 pool header");
            runtime_bytes =
                cursor.scalar<std::uint64_t>(
                    "v2 pool header");
        }
        if (
            count == 0
            || count
                > static_cast<std::uint32_t>(
                    expert_count)
            || (
                magic == "NIM2"
                && (
                    dtype_bytes == 0
                    || dtype_bytes > 32
                )
            )
        ) {
            throw std::runtime_error(
                "invalid NINTM pool metadata");
        }

        std::vector<std::int32_t> expert_ids(count);
        for (auto& expert : expert_ids) {
            expert =
                cursor.scalar<std::int32_t>(
                    "expert IDs");
        }
        claim_experts(
            expert_ids,
            owners,
            static_cast<int>(pool));

        std::string dtype = "NINT";
        if (magic == "NIM2") {
            const auto raw_dtype =
                cursor.bytes(
                    dtype_bytes,
                    "cohort dtype");
            if (!is_ascii(raw_dtype)) {
                throw std::runtime_error(
                    "NINTM cohort dtype must be ASCII");
            }
            dtype.assign(
                reinterpret_cast<const char*>(
                    raw_dtype.data()),
                raw_dtype.size());
        }

        auto runtime = cursor.bytes(
            checked_size(
                runtime_bytes,
                "cohort runtime metadata"),
            "cohort runtime metadata");
        auto payload = cursor.bytes(
            checked_size(
                payload_bytes,
                "cohort payload"),
            "cohort payload");

        if (is_nint_dtype(dtype)) {
            if (!runtime.empty()) {
                throw std::runtime_error(
                    "unexpected NINTM NINT "
                    "runtime metadata");
            }
            add_nint_pool(
                payload,
                expert_ids,
                output_width,
                input_width,
                streams,
                descriptors);
        } else if (is_nint8_zero_dtype(dtype)) {
            if (!runtime.empty()) {
                throw std::runtime_error(
                    "unexpected NINTM NINT8-0 "
                    "runtime metadata");
            }
            add_q8_pool(
                payload,
                expert_ids,
                output_width,
                input_width,
                streams,
                descriptors);
        } else if (is_vq_dtype(dtype)) {
            add_vq_pool(
                dtype,
                payload,
                runtime,
                expert_ids,
                output_width,
                input_width,
                streams,
                rotations,
                descriptors);
        } else {
            throw std::runtime_error(
                "unsupported nested NINTM cohort dtype: "
                + dtype);
        }
    }

    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "trailing bytes in NINTM tensor");
    }
    const auto missing = std::find(
        owners.begin(),
        owners.end(),
        -1);
    if (missing != owners.end()) {
        throw std::runtime_error(
            "NINTM pools do not cover every expert");
    }

    const Shape descriptor_shape{
        expert_count,
        kDescriptorSize,
    };
    auto impl = std::make_shared<Impl>(
        make_int32_array(
            descriptors,
            descriptor_shape),
        make_raw_array(
            std::move(streams.nint_q),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.nint_sub_scale),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.nint_sub_min),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.nint_anchor_scale),
            mlx::core::float32),
        make_raw_array(
            std::move(streams.nint_anchor_min),
            mlx::core::float32),
        make_raw_array(
            std::move(streams.q8_q),
            mlx::core::int8),
        make_raw_array(
            std::move(streams.q8_scales),
            mlx::core::float16),
        make_raw_array(
            std::move(streams.vq_indices),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.vq_state),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.vq_aux),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.vq_anchors),
            mlx::core::float32),
        make_raw_array(
            std::move(streams.vq_codebooks),
            mlx::core::int8),
        make_raw_array(
            std::move(streams.vq_scales),
            mlx::core::float32),
        make_raw_array(
            std::move(
                streams.vq_state_to_codebank),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.vq_banks),
            mlx::core::uint8),
        make_raw_array(
            std::move(streams.vq_parameters),
            mlx::core::float32),
        std::move(rotations),
        std::move(descriptors),
        expert_count,
        output_width,
        input_width,
        1);
    // NINTM cohorts are parsed through the standalone NINT/VQ loaders and
    // then repacked into the heterogeneous execution streams above. Those
    // temporary MLX arrays are dead now, but the Metal allocator otherwise
    // caches their buffers and can retain almost another model-sized copy
    // while loading a fully resident MoE. Active arrays are unaffected.
    mlx::core::clear_cache();
    return MlxNintMoeWeight(std::move(impl));
}

MlxNintMoeWeight MlxNintMoeWeight::concatenate_projections(
    const std::vector<MlxNintMoeWeight>& weights) {
    if (weights.empty()) {
        throw std::invalid_argument(
            "at least one NINTM projection is required");
    }
    const auto& first = *weights.front().impl_;
    for (const auto& weight : weights) {
        const auto& current = *weight.impl_;
        if (
            current.projections != 1
            || current.experts != first.experts
            || current.out_per_expert
                != first.out_per_expert
            || current.neuron_len != first.neuron_len
        ) {
            throw std::invalid_argument(
                "NINTM projections have incompatible shapes");
        }
    }

    const auto descriptor_count = checked_product(
        checked_product(
            static_cast<std::size_t>(first.experts),
            weights.size(),
            "projection descriptor count"),
        static_cast<std::size_t>(kDescriptorSize),
        "projection descriptor count");
    std::vector<std::int32_t> descriptors(
        descriptor_count,
        0);

    std::size_t nint_q_offset = 0;
    std::size_t nint_sub_offset = 0;
    std::size_t nint_anchor_offset = 0;
    std::size_t q8_q_offset = 0;
    std::size_t q8_scale_offset = 0;
    std::size_t vq_indices_offset = 0;
    std::size_t vq_state_offset = 0;
    std::size_t vq_aux_offset = 0;
    std::size_t vq_anchor_offset = 0;
    std::size_t vq_codebook_offset = 0;
    std::size_t vq_scale_offset = 0;
    std::size_t vq_state_bank_offset = 0;
    std::size_t vq_bank_offset = 0;
    std::size_t vq_parameter_offset = 0;

    std::vector<array> nint_q_arrays;
    std::vector<array> nint_sub_scale_arrays;
    std::vector<array> nint_sub_min_arrays;
    std::vector<array> nint_anchor_scale_arrays;
    std::vector<array> nint_anchor_min_arrays;
    std::vector<array> q8_q_arrays;
    std::vector<array> q8_scale_arrays;
    std::vector<array> vq_indices_arrays;
    std::vector<array> vq_state_arrays;
    std::vector<array> vq_aux_arrays;
    std::vector<array> vq_anchor_arrays;
    std::vector<array> vq_codebook_arrays;
    std::vector<array> vq_scale_arrays;
    std::vector<array> vq_state_bank_arrays;
    std::vector<array> vq_bank_arrays;
    std::vector<array> vq_parameter_arrays;
    std::vector<RotationSpec> combined_rotations;
    for (
        std::size_t projection = 0;
        projection < weights.size();
        ++projection
    ) {
        const auto& source = *weights[projection].impl_;
        std::vector<int> rotation_map(
            source.rotations.size() + 1,
            0);
        for (
            std::size_t index = 0;
            index < source.rotations.size();
            ++index
        ) {
            rotation_map[index + 1] =
                rotation_variant(
                    source.rotations[index],
                    combined_rotations);
        }
        for (
            int expert = 0;
            expert < first.experts;
            ++expert
        ) {
            const auto source_base =
                static_cast<std::size_t>(expert)
                * kDescriptorSize;
            const auto target_base = (
                static_cast<std::size_t>(expert)
                    * weights.size()
                + projection
            ) * kDescriptorSize;
            std::copy_n(
                source.descriptor_values.begin()
                    + static_cast<std::ptrdiff_t>(
                        source_base),
                kDescriptorSize,
                descriptors.begin()
                    + static_cast<std::ptrdiff_t>(
                        target_base));
            auto* descriptor =
                descriptors.data() + target_base;
            if (descriptor[kFamily] == kFamilyNint) {
                descriptor[kNintQOffset] =
                    descriptor_with_offset(
                        descriptor[kNintQOffset],
                        nint_q_offset,
                        "NINT q offset");
                descriptor[kNintSubOffset] =
                    descriptor_with_offset(
                        descriptor[kNintSubOffset],
                        nint_sub_offset,
                        "NINT sub offset");
                descriptor[kNintAnchorOffset] =
                    descriptor_with_offset(
                        descriptor[
                            kNintAnchorOffset],
                        nint_anchor_offset,
                        "NINT anchor offset");
            } else if (
                descriptor[kFamily]
                == kFamilyNint8Zero
            ) {
                descriptor[kQ8QOffset] =
                    descriptor_with_offset(
                        descriptor[kQ8QOffset],
                        q8_q_offset,
                        "NINT8-0 q offset");
                descriptor[kQ8ScaleOffset] =
                    descriptor_with_offset(
                        descriptor[kQ8ScaleOffset],
                        q8_scale_offset,
                        "NINT8-0 scale offset");
            } else if (
                descriptor[kFamily] == kFamilyVq
            ) {
                descriptor[kVqIndicesOffset] =
                    descriptor_with_offset(
                        descriptor[kVqIndicesOffset],
                        vq_indices_offset,
                        "VQ index offset");
                descriptor[kVqStateOffset] =
                    descriptor_with_offset(
                        descriptor[kVqStateOffset],
                        vq_state_offset,
                        "VQ state offset");
                descriptor[kVqAuxOffset] =
                    descriptor_with_offset(
                        descriptor[kVqAuxOffset],
                        vq_aux_offset,
                        "VQ auxiliary offset");
                descriptor[kVqAnchorOffset] =
                    descriptor_with_offset(
                        descriptor[kVqAnchorOffset],
                        vq_anchor_offset,
                        "VQ anchor offset");
                descriptor[kVqCodebookOffset] =
                    descriptor_with_offset(
                        descriptor[kVqCodebookOffset],
                        vq_codebook_offset,
                        "VQ codebook offset");
                descriptor[kVqScaleOffset] =
                    descriptor_with_offset(
                        descriptor[kVqScaleOffset],
                        vq_scale_offset,
                        "VQ scale offset");
                descriptor[kVqStateBankOffset] =
                    descriptor_with_offset(
                        descriptor[
                            kVqStateBankOffset],
                        vq_state_bank_offset,
                        "VQ state-bank offset");
                descriptor[kVqBankOffset] =
                    descriptor_with_offset(
                        descriptor[kVqBankOffset],
                        vq_bank_offset,
                        "VQ bank offset");
                descriptor[kVqParameterOffset] =
                    descriptor_with_offset(
                        descriptor[
                            kVqParameterOffset],
                        vq_parameter_offset,
                        "VQ parameter offset");
                const int local_rotation =
                    descriptor[
                        kVqRotationVariant];
                if (
                    local_rotation < 0
                    || static_cast<std::size_t>(
                           local_rotation)
                        >= rotation_map.size()
                ) {
                    throw std::runtime_error(
                        "invalid NINTM VQ rotation "
                        "variant");
                }
                descriptor[kVqRotationVariant] =
                    rotation_map[
                        static_cast<std::size_t>(
                            local_rotation)];
            } else {
                throw std::runtime_error(
                    "unsupported NINTM descriptor family");
            }
        }

        nint_q_arrays.push_back(source.nint_q);
        nint_sub_scale_arrays.push_back(
            source.nint_sub_scale);
        nint_sub_min_arrays.push_back(
            source.nint_sub_min);
        nint_anchor_scale_arrays.push_back(
            source.nint_anchor_scale);
        nint_anchor_min_arrays.push_back(
            source.nint_anchor_min);
        q8_q_arrays.push_back(source.q8_q);
        q8_scale_arrays.push_back(source.q8_scales);
        vq_indices_arrays.push_back(
            source.vq_indices);
        vq_state_arrays.push_back(source.vq_state);
        vq_aux_arrays.push_back(source.vq_aux);
        vq_anchor_arrays.push_back(
            source.vq_anchors);
        vq_codebook_arrays.push_back(
            source.vq_codebooks);
        vq_scale_arrays.push_back(source.vq_scales);
        vq_state_bank_arrays.push_back(
            source.vq_state_to_codebank);
        vq_bank_arrays.push_back(source.vq_banks);
        vq_parameter_arrays.push_back(
            source.vq_parameters);

        nint_q_offset = checked_add(
            nint_q_offset,
            source.nint_q.size(),
            "NINT q stream size");
        nint_sub_offset = checked_add(
            nint_sub_offset,
            source.nint_sub_scale.size(),
            "NINT sub stream size");
        nint_anchor_offset = checked_add(
            nint_anchor_offset,
            source.nint_anchor_scale.size(),
            "NINT anchor stream size");
        q8_q_offset = checked_add(
            q8_q_offset,
            source.q8_q.size(),
            "NINT8-0 q stream size");
        q8_scale_offset = checked_add(
            q8_scale_offset,
            source.q8_scales.size(),
            "NINT8-0 scale stream size");
        vq_indices_offset = checked_add(
            vq_indices_offset,
            source.vq_indices.size(),
            "VQ index stream size");
        vq_state_offset = checked_add(
            vq_state_offset,
            source.vq_state.size(),
            "VQ state stream size");
        vq_aux_offset = checked_add(
            vq_aux_offset,
            source.vq_aux.size(),
            "VQ auxiliary stream size");
        vq_anchor_offset = checked_add(
            vq_anchor_offset,
            source.vq_anchors.size(),
            "VQ anchor stream size");
        vq_codebook_offset = checked_add(
            vq_codebook_offset,
            source.vq_codebooks.size(),
            "VQ codebook stream size");
        vq_scale_offset = checked_add(
            vq_scale_offset,
            source.vq_scales.size(),
            "VQ scale stream size");
        vq_state_bank_offset = checked_add(
            vq_state_bank_offset,
            source.vq_state_to_codebank.size(),
            "VQ state-bank stream size");
        vq_bank_offset = checked_add(
            vq_bank_offset,
            source.vq_banks.size(),
            "VQ bank stream size");
        vq_parameter_offset = checked_add(
            vq_parameter_offset,
            source.vq_parameters.size(),
            "VQ parameter stream size");
    }

    const int projection_count =
        checked_int(weights.size(), "projection count");
    auto impl = std::make_shared<Impl>(
        make_int32_array(
            descriptors,
            Shape{
                first.experts * projection_count,
                kDescriptorSize,
            }),
        concatenate_1d(std::move(nint_q_arrays)),
        concatenate_1d(
            std::move(nint_sub_scale_arrays)),
        concatenate_1d(
            std::move(nint_sub_min_arrays)),
        concatenate_1d(
            std::move(nint_anchor_scale_arrays)),
        concatenate_1d(
            std::move(nint_anchor_min_arrays)),
        concatenate_1d(std::move(q8_q_arrays)),
        concatenate_1d(std::move(q8_scale_arrays)),
        concatenate_1d(
            std::move(vq_indices_arrays)),
        concatenate_1d(
            std::move(vq_state_arrays)),
        concatenate_1d(std::move(vq_aux_arrays)),
        concatenate_1d(
            std::move(vq_anchor_arrays)),
        concatenate_1d(
            std::move(vq_codebook_arrays)),
        concatenate_1d(
            std::move(vq_scale_arrays)),
        concatenate_1d(
            std::move(vq_state_bank_arrays)),
        concatenate_1d(
            std::move(vq_bank_arrays)),
        concatenate_1d(
            std::move(vq_parameter_arrays)),
        std::move(combined_rotations),
        std::move(descriptors),
        first.experts,
        first.out_per_expert,
        first.neuron_len,
        projection_count);
    return MlxNintMoeWeight(std::move(impl));
}

array MlxNintMoeWeight::routed_matmul(
    const array& input,
    const array& expert_ids) const {
    return routed_matmul_impl(
        input,
        expert_ids,
        false,
        0.0f);
}

array MlxNintMoeWeight::routed_swiglu(
    const array& input,
    const array& expert_ids,
    float limit) const {
    if (
        impl_->projections != 1
        || impl_->out_per_expert <= 0
        || (impl_->out_per_expert & 1) != 0
    ) {
        throw std::invalid_argument(
            "fused NINTM SwiGLU requires one even-width gate/up projection");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "NINTM SwiGLU limit must be finite and non-negative");
    }
    return routed_matmul_impl(
        input,
        expert_ids,
        true,
        limit);
}

bool MlxNintMoeWeight::supports_grouped_vq_mmq() const noexcept {
    return impl_->projections == 1 && impl_->grouped_vq_mmq;
}

MlxGroupedVqMmqPlan MlxNintMoeWeight::build_grouped_vq_mmq_plan(
    const array& expert_ids,
    const array& route_order) const {
    if (!supports_grouped_vq_mmq()) {
        throw std::invalid_argument(
            "weight does not support grouped VQ MMQ");
    }
    return make_grouped_vq_mmq_plan(
        expert_ids,
        route_order,
        impl_->experts);
}

array MlxNintMoeWeight::routed_matmul_sorted(
    const array& input,
    const array& expert_ids,
    const array& route_order_value,
    bool input_is_sorted,
    bool fused_swiglu,
    float swiglu_limit,
    const MlxGroupedVqMmqPlan* plan) const {
    if (!supports_grouped_vq_mmq()) {
        throw std::invalid_argument(
            "weight does not support grouped VQ MMQ");
    }
    if (
        fused_swiglu
        && (
            impl_->out_per_expert <= 0
            || (impl_->out_per_expert & 1) != 0
            || !std::isfinite(swiglu_limit)
            || swiglu_limit < 0.0f
        )
    ) {
        throw std::invalid_argument(
            "fused grouped VQ SwiGLU parameters are invalid");
    }
    auto ids = mlx::core::contiguous(
        mlx::core::astype(
            expert_ids,
            mlx::core::int32));
    if (ids.ndim() != 2) {
        throw std::invalid_argument(
            "routed expert IDs must have [tokens,routes] shape");
    }
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    const auto route_count_size = checked_product(
        static_cast<std::size_t>(tokens),
        static_cast<std::size_t>(routes),
        "route count");
    const int route_count = checked_int(
        route_count_size,
        "route count");
    auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            route_order_value,
            mlx::core::int32));
    if (
        route_order.ndim() != 1
        || route_order.size() != route_count_size
    ) {
        throw std::invalid_argument(
            "route order must contain one index per routed row");
    }

    bool shared_input = false;
    if (input_is_sorted) {
        if (
            input.ndim() != 2
            || input.shape(0) != route_count
            || input.shape(1) != impl_->neuron_len
        ) {
            throw std::invalid_argument(
                "sorted routed input must have [tokens*routes,K] shape");
        }
    } else if (
        input.ndim() == 2
        && input.shape(0) == tokens
        && input.shape(1) == impl_->neuron_len
    ) {
        shared_input = true;
    } else if (
        input.ndim() != 3
        || input.shape(0) != tokens
        || input.shape(1) != routes
        || input.shape(2) != impl_->neuron_len
    ) {
        throw std::invalid_argument(
            "routed input must have [tokens,K] or [tokens,routes,K] shape");
    }

    auto source = input.dtype() == mlx::core::float16
        ? input
        : mlx::core::astype(input, mlx::core::float16);
    source = mlx::core::contiguous(source);
    const int variant_stride = route_count;
    if (!impl_->rotations.empty()) {
        if (shared_input) {
            source = mlx::core::broadcast_to(
                mlx::core::reshape(
                    source,
                    Shape{tokens, 1, impl_->neuron_len}),
                Shape{tokens, routes, impl_->neuron_len});
        }
        source = mlx::core::contiguous(
            mlx::core::reshape(
                source,
                Shape{route_count, impl_->neuron_len}));
        std::vector<array> variants;
        variants.reserve(impl_->rotations.size() + 1);
        variants.push_back(source);
        for (const auto& rotation : impl_->rotations) {
            variants.push_back(
                apply_rotation(
                    source,
                    rotation,
                    variant_stride,
                    impl_->neuron_len));
        }
        source = mlx::core::contiguous(
            mlx::core::concatenate(
                std::move(variants),
                0));
        shared_input = false;
    }

    const array params({0.0f}, mlx::core::float32);
    std::vector<array> kernel_inputs{
        impl_->descriptors,
        impl_->nint_q,
        impl_->nint_sub_scale,
        impl_->nint_sub_min,
        impl_->nint_anchor_scale,
        impl_->nint_anchor_min,
        impl_->q8_q,
        impl_->q8_scales,
        impl_->vq_indices,
        impl_->vq_state,
        impl_->vq_aux,
        impl_->vq_anchors,
        impl_->vq_codebooks,
        impl_->vq_scales,
        impl_->vq_state_to_codebank,
        impl_->vq_banks,
        impl_->vq_parameters,
        source,
        ids,
        route_order,
        params,
    };
    const int output_width = fused_swiglu
        ? impl_->out_per_expert / 2
        : impl_->out_per_expert;
    auto owned_plan = plan == nullptr
        ? std::optional<MlxGroupedVqMmqPlan>(
            make_grouped_vq_mmq_plan(
                ids,
                route_order,
                impl_->experts))
        : std::nullopt;
    const auto& selected_plan = plan != nullptr
        ? *plan
        : *owned_plan;
    if (
        selected_plan.route_count != route_count
        || selected_plan.experts != impl_->experts
        || selected_plan.block_rows != 32
        || selected_plan.max_blocks <= 0
    ) {
        throw std::invalid_argument(
            "grouped VQ MMQ block plan does not match routed projection");
    }
    kernel_inputs.push_back(selected_plan.block_meta);
    kernel_inputs.push_back(selected_plan.block_count);
    return grouped_vq_mmq_dispatch(
        std::move(kernel_inputs),
        GroupedVqMmqConfig{
            .output_shape = Shape{
                route_count,
                output_width,
            },
            .route_count = route_count,
            .max_blocks = selected_plan.max_blocks,
            .tokens = tokens,
            .routes = routes,
            .experts = impl_->experts,
            .output_width = output_width,
            .matrix_output_width = impl_->out_per_expert,
            .input_width = impl_->neuron_len,
            .descriptor_size = kDescriptorSize,
            .variant_stride = variant_stride,
            .shared_input = static_cast<int>(shared_input),
            .input_sorted = static_cast<int>(input_is_sorted),
            .fused_swiglu = static_cast<int>(fused_swiglu),
            .swiglu_limit = swiglu_limit,
        });
}

array MlxNintMoeWeight::routed_matmul_impl(
    const array& input,
    const array& expert_ids,
    bool fused_swiglu,
    float swiglu_limit) const {
    auto ids = mlx::core::contiguous(
        mlx::core::astype(
            expert_ids,
            mlx::core::int32));
    if (ids.ndim() != 2) {
        throw std::invalid_argument(
            "routed expert IDs must have "
            "[tokens,routes] shape");
    }
    const int tokens = ids.shape(0);
    const int routes = ids.shape(1);
    if (tokens < 0 || routes < 0) {
        throw std::invalid_argument(
            "routed expert dimensions cannot be negative");
    }

    bool shared_input = false;
    if (
        input.ndim() == 2
        && input.shape(0) == tokens
        && input.shape(1) == impl_->neuron_len
    ) {
        shared_input = true;
    } else if (
        input.ndim() != 3
        || input.shape(0) != tokens
        || input.shape(1) != routes
        || input.shape(2) != impl_->neuron_len
    ) {
        throw std::invalid_argument(
            "routed input must have [tokens,K] or "
            "[tokens,routes,K] shape");
    }

    auto source = input;
    if (
        source.dtype() != mlx::core::float16
        && source.dtype() != mlx::core::float32
    ) {
        source =
            mlx::core::astype(
                source,
                mlx::core::float16);
    }
    source = mlx::core::contiguous(source);
    const int logical_output_width = fused_swiglu
        ? impl_->out_per_expert / 2
        : impl_->out_per_expert;
    const auto output_width = fused_swiglu
        ? static_cast<std::size_t>(logical_output_width)
        : checked_product(
              static_cast<std::size_t>(impl_->projections),
              static_cast<std::size_t>(impl_->out_per_expert),
              "routed output width");
    const Shape output_shape{
        tokens,
        routes,
        checked_int(
            output_width,
            "routed output width"),
    };
    if (tokens == 0 || routes == 0) {
        return mlx::core::zeros(
            output_shape,
            source.dtype());
    }

    const auto route_count_size = checked_product(
        static_cast<std::size_t>(tokens),
        static_cast<std::size_t>(routes),
        "route count");
    const int variant_stride = checked_int(
        route_count_size,
        "rotation variant stride");
    const bool sorted_routes = tokens > 1;
    auto route_order = mlx::core::zeros(
        Shape{1},
        mlx::core::int32);
    if (sorted_routes) {
        route_order = mlx::core::contiguous(
            mlx::core::astype(
                mlx::core::argsort(
                    mlx::core::reshape(
                        ids,
                        Shape{variant_stride})),
                mlx::core::int32));
    }
    if (!impl_->rotations.empty()) {
        if (shared_input) {
            source = mlx::core::broadcast_to(
                mlx::core::reshape(
                    source,
                    Shape{
                        tokens,
                        1,
                        impl_->neuron_len,
                    }),
                Shape{
                    tokens,
                    routes,
                    impl_->neuron_len,
                });
        }
        source = mlx::core::contiguous(
            mlx::core::reshape(
                source,
                Shape{
                    variant_stride,
                    impl_->neuron_len,
                }));
        std::vector<array> variants;
        variants.reserve(
            impl_->rotations.size() + 1);
        variants.push_back(source);
        for (const auto& rotation : impl_->rotations) {
            variants.push_back(
                apply_rotation(
                    source,
                    rotation,
                    variant_stride,
                    impl_->neuron_len));
        }
        source = mlx::core::contiguous(
            mlx::core::concatenate(
                std::move(variants),
                0));
        shared_input = false;
    }

    const int k_lanes = impl_->k_lanes_override != 0
        ? impl_->k_lanes_override
        : (fused_swiglu ? 16 : 8);
    const auto rows_per_workgroup =
        static_cast<std::size_t>(64 / k_lanes);
    const auto output_tiles =
        (
            static_cast<std::size_t>(
                logical_output_width)
            + rows_per_workgroup - 1
        ) / rows_per_workgroup;
    auto workgroups = route_count_size;
    workgroups = checked_product(
        workgroups,
        static_cast<std::size_t>(
            fused_swiglu ? 1 : impl_->projections),
        "routed projection count");
    workgroups = checked_product(
        workgroups,
        output_tiles,
        "routed workgroup count");
    const auto grid = checked_product(
        workgroups,
        64,
        "Metal grid");
    if (
        grid
        > static_cast<std::size_t>(
            std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "NINTM Metal grid exceeds MLX limits");
    }

    const array params(
        {swiglu_limit},
        mlx::core::float32);
    std::vector<array> kernel_inputs{
        impl_->descriptors,
        impl_->nint_q,
        impl_->nint_sub_scale,
        impl_->nint_sub_min,
        impl_->nint_anchor_scale,
        impl_->nint_anchor_min,
        impl_->q8_q,
        impl_->q8_scales,
        impl_->vq_indices,
        impl_->vq_state,
        impl_->vq_aux,
        impl_->vq_anchors,
        impl_->vq_codebooks,
        impl_->vq_scales,
        impl_->vq_state_to_codebank,
        impl_->vq_banks,
        impl_->vq_parameters,
        source,
        ids,
        route_order,
        params,
    };
    if (
        sorted_routes
        && tokens >= 32
        && source.dtype() == mlx::core::float16
        && impl_->projections == 1
        && impl_->grouped_vq_mmq
        && !fused_swiglu
    ) {
        auto plan = make_grouped_vq_mmq_plan(
            ids,
            route_order,
            impl_->experts);
        kernel_inputs.push_back(plan.block_meta);
        kernel_inputs.push_back(plan.block_count);
        auto sorted_outputs = grouped_vq_mmq_dispatch(
            std::move(kernel_inputs),
            GroupedVqMmqConfig{
                .output_shape = Shape{
                    variant_stride,
                    logical_output_width,
                },
                .route_count = variant_stride,
                .max_blocks = plan.max_blocks,
                .tokens = tokens,
                .routes = routes,
                .experts = impl_->experts,
                .output_width = logical_output_width,
                .matrix_output_width = impl_->out_per_expert,
                .input_width = impl_->neuron_len,
                .descriptor_size = kDescriptorSize,
                .variant_stride = variant_stride,
                .shared_input = static_cast<int>(shared_input),
            });
        auto inverse_order = mlx::core::argsort(route_order);
        auto restored = mlx::core::take(
            std::move(sorted_outputs),
            inverse_order,
            0);
        auto valid = mlx::core::expand_dims(
            mlx::core::greater_equal(
                ids,
                array(0, mlx::core::int32)),
            -1);
        return mlx::core::where(
            valid,
            mlx::core::reshape(
                std::move(restored),
                output_shape),
            mlx::core::zeros(
                output_shape,
                source.dtype()));
    }
    if (impl_->native_primitive) {
        return native_moe_dispatch(
            std::move(kernel_inputs),
            NativeMoeConfig{
                .dtype = source.dtype(),
                .output_shape = output_shape,
                .tokens = tokens,
                .routes = routes,
                .experts = impl_->experts,
                .output_width = logical_output_width,
                .matrix_output_width = impl_->out_per_expert,
                .projections = impl_->projections,
                .fused_swiglu = static_cast<int>(fused_swiglu),
                .input_width = impl_->neuron_len,
                .k_lanes = k_lanes,
                .descriptor_size = kDescriptorSize,
                .variant_stride = variant_stride,
                .shared_input = static_cast<int>(shared_input),
                .jsc_execution_layout = static_cast<int>(
                    impl_->jsc_execution_layout),
                .family_mask = static_cast<int>(impl_->family_mask),
                .vq_profile_mask = static_cast<int>(
                    impl_->vq_profile_mask),
                .npq_grouped_indices = static_cast<int>(
                    impl_->npq_grouped_indices),
                .sorted_routes = static_cast<int>(
                    sorted_routes),
                .workgroups = static_cast<int>(workgroups),
            });
    }
    auto outputs = moe_kernel()(
        std::move(kernel_inputs),
        {output_shape},
        {source.dtype()},
        {
            static_cast<int>(grid),
            1,
            1,
        },
        {64, 1, 1},
        {
            {"T", source.dtype()},
            {"TOKENS", tokens},
            {"ROUTES", routes},
            {"EXPERTS", impl_->experts},
            {"OUT", logical_output_width},
            {"MATRIX_OUT", impl_->out_per_expert},
            {"PROJECTIONS", impl_->projections},
            {
                "FUSED_SWIGLU",
                static_cast<int>(fused_swiglu),
            },
            {"K", impl_->neuron_len},
            {"K_LANES_VALUE", k_lanes},
            {"DESCRIPTOR_SIZE", kDescriptorSize},
            {"VARIANT_STRIDE", variant_stride},
            {
                "SHARED_INPUT",
                static_cast<int>(shared_input),
            },
            {
                "JSC_EXECUTION_LAYOUT",
                static_cast<int>(
                    impl_->jsc_execution_layout),
            },
            {
                "FAMILY_MASK",
                static_cast<int>(impl_->family_mask),
            },
            {
                "VQ_PROFILE_MASK",
                static_cast<int>(
                    impl_->vq_profile_mask),
            },
            {
                "NPQ_GROUPED_INDICES",
                static_cast<int>(
                    impl_->npq_grouped_indices),
            },
            {
                "SORTED_ROUTES",
                static_cast<int>(sorted_routes),
            },
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

int MlxNintMoeWeight::experts() const noexcept {
    return impl_->experts;
}

int MlxNintMoeWeight::out_per_expert() const noexcept {
    return impl_->out_per_expert;
}

int MlxNintMoeWeight::neuron_len() const noexcept {
    return impl_->neuron_len;
}

int MlxNintMoeWeight::projections() const noexcept {
    return impl_->projections;
}

std::size_t MlxNintMoeWeight::packed_nbytes() const noexcept {
    return impl_->packed_bytes;
}

MlxRoutedLinear::MlxRoutedLinear(
    MlxMoeWeight weight)
    : weight_(std::move(weight)) {
    if (weight_.projections() != 1) {
        throw std::invalid_argument(
            "routed linear requires one NINTM projection");
    }
}

MlxRoutedLinear MlxRoutedLinear::from_blob(
    std::span<const std::uint8_t> blob) {
    return MlxRoutedLinear(
        MlxMoeWeight::from_blob(blob));
}

array MlxRoutedLinear::forward(
    const array& input,
    const array& expert_ids) const {
    return weight_.routed_matmul(input, expert_ids);
}

array MlxRoutedLinear::swiglu(
    const array& input,
    const array& expert_ids,
    float limit) const {
    return weight_.routed_swiglu(
        input,
        expert_ids,
        limit);
}

array MlxRoutedLinear::combine(
    const array& input,
    const array& expert_ids,
    const array& route_weights) const {
    return moe_weighted_reduce(
        forward(input, expert_ids),
        route_weights);
}

MlxRoutedSwiGluFfn::MlxRoutedSwiGluFfn(
    MlxMoeWeight gate,
    MlxMoeWeight up,
    MlxMoeWeight down)
    : gate_up_(
          MlxMoeWeight::concatenate_projections(
              {std::move(gate), std::move(up)})),
      down_(std::move(down)) {
    if (
        gate_up_.experts() != down_.experts()
        || gate_up_.out_per_expert()
            != down_.neuron_len()
        || gate_up_.neuron_len()
            != down_.out_per_expert()
        || down_.projections() != 1
    ) {
        throw std::invalid_argument(
            "routed SwiGLU gate/up/down shapes "
            "are incompatible");
    }
}

MlxRoutedSwiGluFfn
MlxRoutedSwiGluFfn::from_blobs(
    std::span<const std::uint8_t> gate,
    std::span<const std::uint8_t> up,
    std::span<const std::uint8_t> down) {
    return MlxRoutedSwiGluFfn(
        MlxMoeWeight::from_blob(gate),
        MlxMoeWeight::from_blob(up),
        MlxMoeWeight::from_blob(down));
}

array MlxRoutedSwiGluFfn::forward(
    const array& input,
    const array& expert_ids,
    const array& route_weights) const {
    auto gate_up =
        gate_up_.routed_matmul(
            input,
            expert_ids);
    auto hidden =
        moe_swiglu_split(gate_up);
    auto pair_output =
        down_.routed_matmul(
            hidden,
            expert_ids);
    return moe_weighted_reduce(
        pair_output,
        route_weights);
}

MlxRoutedFfnResult
MlxRoutedSwiGluFfn::forward_from_logits(
    const array& input,
    const array& router_logits,
    int top_k,
    bool use_sigmoid,
    bool use_sqrt_softplus,
    bool normalize,
    bool delayed_softmax,
    const std::optional<array>& bias,
    const std::optional<array>& available,
    float norm_floor,
    float scale) const {
    auto routes = moe_topk(
        router_logits,
        top_k,
        use_sigmoid,
        use_sqrt_softplus,
        normalize,
        delayed_softmax,
        bias,
        available,
        norm_floor,
        scale);
    auto output = forward(
        input,
        routes.ids,
        routes.weights);
    return {
        std::move(output),
        std::move(routes.ids),
        std::move(routes.weights),
    };
}

} // namespace mfq::metal
