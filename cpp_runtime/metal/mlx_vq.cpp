#include "mlx_vq.h"

#include "../nvq_codebooks.generated.h"
#include "mlx_staging_allocator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kAuxNone = 0;
constexpr int kAuxSignEven = 1;
constexpr int kAuxSignIndexParity = 2;
constexpr int kAuxDelta = 3;

constexpr int kCodeBankFixed = 0;
constexpr int kCodeBankState = 1;
constexpr int kCodeBankAux = 2;

constexpr int kExecutionStreams = 0;
constexpr int kExecutionGroup64 = 1;

constexpr const char* kBitstreamHeader = R"METAL(
#include <metal_simdgroup_matrix>

inline uint mfq_vq_read_bits(
    device const uchar* stream,
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
    return (packed >> shift) & ((1u << bits) - 1u);
}

inline uint mfq_vq_read_bits(
    constant const uchar* stream,
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
    return (packed >> shift) & ((1u << bits) - 1u);
}

template <typename Stream>
inline uint mfq_vq_read_4(
    Stream stream,
    uint value_index
) {
    uint packed = uint(stream[value_index >> 1u]);
    return (packed >> ((value_index & 1u) * 4u)) & 15u;
}

template <typename Stream>
inline uint mfq_vq_read_8(
    Stream stream,
    uint value_index
) {
    return uint(stream[value_index]);
}

template <typename Stream>
inline uint mfq_vq_read_12(
    Stream stream,
    uint value_index
) {
    uint odd = value_index & 1u;
    uint byte_index = (value_index >> 1u) * 3u + odd;
    uint packed = uint(stream[byte_index])
        | (uint(stream[byte_index + 1u]) << 8u);
    return (packed >> (odd * 4u)) & 4095u;
}

inline uint2 mfq_vq_read_group64(
    device const uchar* stream,
    uint record_index
) {
    return *(device const uint2*)(stream + record_index * 8u);
}

inline uint mfq_vq_group64_segment(
    uint2 record,
    uint local_vector
) {
    if (local_vector == 0u) {
        return record.x & 0xfffffu;
    }
    if (local_vector == 1u) {
        return ((record.x >> 20u) | (record.y << 12u)) & 0xfffffu;
    }
    return (record.y >> 8u) & 0xfffffu;
}

template <
    typename IndicesPtr,
    typename StatePtr,
    typename AuxPtr,
    typename AnchorPtr,
    typename CodebookPtr,
    typename ScalePtr,
    typename StateBankPtr,
    typename BankPtr,
    typename ParameterPtr
>
inline float mfq_vq_decode_weight(
    IndicesPtr indices_packed,
    StatePtr state_packed,
    AuxPtr aux_packed,
    AnchorPtr anchors,
    CodebookPtr codebooks,
    ScalePtr scale_lut,
    StateBankPtr state_to_codebank,
    BankPtr bank_ids,
    ParameterPtr parameters,
    uint output,
    uint column,
    uint groupsize,
    uint groups,
    uint vector_size,
    uint vectors,
    uint index_bits,
    uint state_bits,
    uint states,
    uint entries,
    uint code_banks,
    uint aux_mode,
    uint code_bank_mode,
    uint execution_layout,
    uint has_table_banks,
    uint groups_per_super,
    uint supergroups,
    uint signs
) {
    uint group = column / groupsize;
    uint vector = column / vector_size;
    uint component = column - vector * vector_size;
    uint state_index = output * groups + group;
    uint state;
    uint index;
    uint aux_value = 0u;
    if (execution_layout == 1u) {
        uint2 record = mfq_vq_read_group64(
            indices_packed,
            state_index);
        uint segment = mfq_vq_group64_segment(
            record,
            (column - group * groupsize) / vector_size);
        state = record.y >> 28u;
        index = segment & 4095u;
        aux_value = segment >> 12u;
    } else {
        state = mfq_vq_read_bits(
            state_packed, state_index, state_bits);
        index = mfq_vq_read_bits(
            indices_packed,
            output * vectors + vector,
            index_bits);
        if (aux_mode == 1u || aux_mode == 2u) {
            aux_value = mfq_vq_read_bits(
                aux_packed, output * signs + column / 8u, 7u);
        } else if (aux_mode == 3u) {
            aux_value = mfq_vq_read_bits(
                aux_packed, state_index, 1u);
        }
    }
    uint table_bank = 0u;
    if (has_table_banks != 0u) {
        table_bank = uint(bank_ids[
            output * supergroups + group / groups_per_super
        ]);
    }
    uint code_bank = 0u;
    if (code_bank_mode == 1u) {
        code_bank = uint(state_to_codebank[state]);
    } else if (code_bank_mode == 2u) {
        code_bank = aux_value;
    }
    uint code_offset = (
        (
            (table_bank * code_banks + code_bank)
            * entries + index
        )
        * vector_size + component
    );
    float code = float(codebooks[code_offset]);
    if (aux_mode == 1u || aux_mode == 2u) {
        uint sign_position = column & 7u;
        uint negative = execution_layout == 1u
            ? ((aux_value >> sign_position) & 1u)
            : (sign_position < 7u
                ? ((aux_value >> sign_position) & 1u)
                : (popcount(aux_value) & 1u));
        if (aux_mode == 2u && sign_position == 7u) {
            negative ^= (index >> 7u) & 1u;
        }
        code = negative != 0u ? -code : code;
    } else if (aux_mode == 3u) {
        code += aux_value != 0u
            ? -parameters[0] : parameters[0];
    }
    return anchors[output]
        * scale_lut[table_bank * states + state]
        * code;
}
)METAL";

constexpr const char* kMatmulSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5;
    uint output = workgroup % uint(OUT);
    uint row_tile = workgroup / uint(OUT);
    uint first_row = row_tile * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }

    float accumulators[TILE_M];
    for (
        uint local_row = 0u;
        local_row < uint(TILE_M);
        ++local_row
    ) {
        accumulators[local_row] = 0.0f;
    }
    for (uint column = lane; column < uint(K); column += 32u) {
        float weight = mfq_vq_decode_weight(
            indices_packed,
            state_packed,
            aux_packed,
            anchors,
            codebooks,
            scale_lut,
            state_to_codebank,
            bank_ids,
            parameters,
            output,
            column,
            uint(GS),
            uint(NG),
            uint(VECTOR_SIZE),
            uint(NVEC),
            uint(INDEX_BITS),
            uint(STATE_BITS),
            uint(STATES),
            uint(ENTRIES),
            uint(CODE_BANKS),
            uint(AUX_MODE),
            uint(CODE_BANK_MODE),
            uint(EXECUTION_LAYOUT),
            uint(HAS_TABLE_BANKS),
            uint(GROUPS_PER_SUPER),
            uint(NSUPER),
            uint(NSIGN)
        );
        for (
            uint local_row = 0u;
            local_row < uint(TILE_M);
            ++local_row
        ) {
            uint row = first_row + local_row;
            if (row < uint(M)) {
                accumulators[local_row] = fma(
                    float(x[row * uint(K) + column]),
                    weight,
                    accumulators[local_row]);
            }
        }
    }
    for (
        uint local_row = 0u;
        local_row < uint(TILE_M);
        ++local_row
    ) {
        float total = simd_sum(accumulators[local_row]);
        uint row = first_row + local_row;
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] = T(total);
        }
    }
)METAL";

constexpr const char* kGemvSource = R"METAL(
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG =
        SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint VECTORS_PER_GROUP =
        (uint(GS) + uint(VECTOR_SIZE) - 1u)
        / uint(VECTOR_SIZE);

    uint lane = thread_index_in_simdgroup;
    uint simd_group =
        simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;
    float accumulators[ROWS_PER_SIMD] = {0.0f};

    for (uint group = lane;
         group < uint(NG);
         group += 32u) {
        uint outputs[ROWS_PER_SIMD];
        uint table_banks[ROWS_PER_SIMD];
        uint code_banks[ROWS_PER_SIMD];
        uint delta_values[ROWS_PER_SIMD];
        float weight_scales[ROWS_PER_SIMD];

        for (uint row = 0u;
             row < ROWS_PER_SIMD;
             ++row) {
            uint output =
                min(output_base + row, uint(OUT) - 1u);
            uint state_index =
                output * uint(NG) + group;
            uint state = mfq_vq_read_bits(
                state_packed,
                state_index,
                uint(STATE_BITS));
            uint table_bank = 0u;
            if (HAS_TABLE_BANKS != 0) {
                table_bank = uint(bank_ids[
                    output * uint(NSUPER)
                    + group / uint(GROUPS_PER_SUPER)
                ]);
            }
            uint delta_value = AUX_MODE == 3
                ? mfq_vq_read_bits(
                      aux_packed,
                      state_index,
                      1u)
                : 0u;
            uint code_bank = 0u;
            if (CODE_BANK_MODE == 1) {
                code_bank =
                    uint(state_to_codebank[state]);
            } else if (CODE_BANK_MODE == 2) {
                code_bank = delta_value;
            }
            outputs[row] = output;
            table_banks[row] = table_bank;
            code_banks[row] = code_bank;
            delta_values[row] = delta_value;
            weight_scales[row] = anchors[output]
                * scale_lut[
                    table_bank * uint(STATES)
                    + state
                ];
        }

        for (uint local_vector = 0u;
             local_vector < VECTORS_PER_GROUP;
             ++local_vector) {
            uint column_base =
                group * uint(GS)
                + local_vector * uint(VECTOR_SIZE);
            if (column_base >= uint(K)) {
                break;
            }
            float activations[VECTOR_SIZE];
            for (uint component = 0u;
                 component < uint(VECTOR_SIZE);
                 ++component) {
                uint column = column_base + component;
                activations[component] =
                    column < uint(K)
                    ? float(x[column])
                    : 0.0f;
            }

            uint vector =
                column_base / uint(VECTOR_SIZE);
            for (uint row = 0u;
                 row < ROWS_PER_SIMD;
                 ++row) {
                uint output = outputs[row];
                uint index = mfq_vq_read_bits(
                    indices_packed,
                    output * uint(NVEC) + vector,
                    uint(INDEX_BITS));
                uint sign_value = 0u;
                if (AUX_MODE == 1 || AUX_MODE == 2) {
                    sign_value = mfq_vq_read_bits(
                        aux_packed,
                        output * uint(NSIGN)
                            + column_base / 8u,
                        7u);
                }
                for (uint component = 0u;
                     component < uint(VECTOR_SIZE);
                     ++component) {
                    uint column =
                        column_base + component;
                    if (column >= uint(K)) {
                        break;
                    }
                    uint code_offset = (
                        (
                            (
                                table_banks[row]
                                    * uint(CODE_BANKS)
                                + code_banks[row]
                            )
                            * uint(ENTRIES) + index
                        )
                        * uint(VECTOR_SIZE) + component
                    );
                    float code =
                        float(codebooks[code_offset]);
                    if (AUX_MODE == 1 ||
                        AUX_MODE == 2) {
                        uint sign_position =
                            column & 7u;
                        uint negative =
                            sign_position < 7u
                            ? (
                                (sign_value
                                    >> sign_position)
                                & 1u
                            )
                            : (popcount(sign_value) & 1u);
                        if (
                            AUX_MODE == 2
                            && sign_position == 7u
                        ) {
                            negative ^=
                                (index >> 7u) & 1u;
                        }
                        code = negative != 0u
                            ? -code
                            : code;
                    } else if (AUX_MODE == 3) {
                        code +=
                            delta_values[row] != 0u
                            ? -parameters[0]
                            : parameters[0];
                    }
                    accumulators[row] +=
                        activations[component]
                        * weight_scales[row]
                        * code;
                }
            }
        }
    }

    for (uint row = 0u;
         row < ROWS_PER_SIMD;
         ++row) {
        float total = simd_sum(accumulators[row]);
        uint output = output_base + row;
        if (lane == 0u && output < uint(OUT)) {
            y[output] = T(total);
        }
    }
)METAL";

constexpr const char* kMmqSource = R"METAL(
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD =
        32u / K_LANES;
    constexpr uint ROWS_PER_TG =
        SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint VECTORS_PER_GROUP =
        (uint(GS) + uint(VECTOR_SIZE) - 1u)
        / uint(VECTOR_SIZE);

    uint lane = thread_index_in_simdgroup;
    uint simd_group =
        simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_row = lane / K_LANES;
    uint output_index =
        threadgroup_position_in_grid.y * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD
        + simd_row;
    uint output =
        min(output_index, uint(OUT) - 1u);
    uint first_row =
        threadgroup_position_in_grid.x
        * uint(TILE_M);
    uint output_group_base = output * uint(NG);
    uint output_vector_base = output * uint(NVEC);
    uint output_sign_base = output * uint(NSIGN);
    uint output_super_base = output * uint(NSUPER);
    float output_anchor = anchors[output];

    float accumulators[TILE_M];
    for (uint row = 0u;
         row < uint(TILE_M);
         ++row) {
        accumulators[row] = 0.0f;
    }

    for (uint group = k_lane;
         group < uint(NG);
         group += K_LANES) {
        uint state_index = output_group_base + group;
        uint state = STATE_BITS == 4
            ? mfq_vq_read_4(state_packed, state_index)
            : (STATE_BITS == 8
                ? mfq_vq_read_8(state_packed, state_index)
                : mfq_vq_read_bits(
                    state_packed,
                    state_index,
                    uint(STATE_BITS)));
        uint table_bank = 0u;
        if (HAS_TABLE_BANKS != 0) {
            table_bank = uint(bank_ids[
                output_super_base
                + group / uint(GROUPS_PER_SUPER)
            ]);
        }
        uint delta_value = AUX_MODE == 3
            ? mfq_vq_read_bits(
                  aux_packed,
                  state_index,
                  1u)
            : 0u;
        uint code_bank = 0u;
        if (CODE_BANK_MODE == 1) {
            code_bank =
                uint(state_to_codebank[state]);
        } else if (CODE_BANK_MODE == 2) {
            code_bank = delta_value;
        }
        float weight_scale = output_anchor
            * scale_lut[
                table_bank * uint(STATES)
                + state
            ];

        for (uint local_vector = 0u;
             local_vector < VECTORS_PER_GROUP;
             ++local_vector) {
            uint column_base =
                group * uint(GS)
                + local_vector * uint(VECTOR_SIZE);
            if (column_base >= uint(K)) {
                break;
            }
            uint vector =
                column_base / uint(VECTOR_SIZE);
            uint index_position = output_vector_base + vector;
            uint index = INDEX_BITS == 4
                ? mfq_vq_read_4(indices_packed, index_position)
                : (INDEX_BITS == 8
                    ? mfq_vq_read_8(indices_packed, index_position)
                    : (INDEX_BITS == 12
                        ? mfq_vq_read_12(
                            indices_packed,
                            index_position)
                        : mfq_vq_read_bits(
                            indices_packed,
                            index_position,
                            uint(INDEX_BITS))));
            uint sign_value = 0u;
            if (AUX_MODE == 1 || AUX_MODE == 2) {
                sign_value = mfq_vq_read_bits(
                    aux_packed,
                    output_sign_base + column_base / 8u,
                    7u);
            }
            uint code_base = (
                (
                    (
                        table_bank * uint(CODE_BANKS)
                        + code_bank
                    )
                    * uint(ENTRIES) + index
                )
                * uint(VECTOR_SIZE)
            );
            for (uint component = 0u;
                 component < uint(VECTOR_SIZE);
                 ++component) {
                uint column =
                    column_base + component;
                if (column >= uint(K)) {
                    break;
                }
                float code =
                    float(codebooks[code_base + component]);
                if (AUX_MODE == 1 ||
                    AUX_MODE == 2) {
                    uint sign_position =
                        column & 7u;
                    uint negative =
                        sign_position < 7u
                        ? (
                            (sign_value
                                >> sign_position)
                            & 1u
                        )
                        : (popcount(sign_value) & 1u);
                    if (
                        AUX_MODE == 2
                        && sign_position == 7u
                    ) {
                        negative ^=
                            (index >> 7u) & 1u;
                    }
                    code = negative != 0u
                        ? -code
                        : code;
                } else if (AUX_MODE == 3) {
                    code += delta_value != 0u
                        ? -parameters[0]
                        : parameters[0];
                }
                float weight =
                    weight_scale * code;
                for (uint input = 0u;
                     input < uint(TILE_M);
                     ++input) {
                    uint row = min(
                        first_row + input,
                        uint(M) - 1u);
                    accumulators[input] +=
                        float(
                            x[
                                row * uint(K)
                                + column
                            ]
                        )
                        * weight;
                }
            }
        }
    }

    for (uint input = 0u;
         input < uint(TILE_M);
         ++input) {
        accumulators[input] +=
            simd_shuffle_down(
                accumulators[input],
                4);
        accumulators[input] +=
            simd_shuffle_down(
                accumulators[input],
                2);
        accumulators[input] +=
            simd_shuffle_down(
                accumulators[input],
                1);
        uint row = first_row + input;
        if (
            k_lane == 0u
            && output_index < uint(OUT)
            && row < uint(M)
        ) {
            y[
                row * uint(OUT) + output_index
            ] = T(accumulators[input]);
        }
    }
)METAL";

constexpr const char* kDequantizeSource = R"METAL(
    uint linear = thread_position_in_grid.x;
    if (linear >= uint(OUT) * uint(K)) {
        return;
    }
    uint output = linear / uint(K);
    uint column = linear - output * uint(K);
    y[linear] = T(mfq_vq_decode_weight(
        indices_packed,
        state_packed,
        aux_packed,
        anchors,
        codebooks,
        scale_lut,
        state_to_codebank,
        bank_ids,
        parameters,
        output,
        column,
        uint(GS),
        uint(NG),
        uint(VECTOR_SIZE),
        uint(NVEC),
        uint(INDEX_BITS),
        uint(STATE_BITS),
        uint(STATES),
        uint(ENTRIES),
        uint(CODE_BANKS),
        uint(AUX_MODE),
        uint(CODE_BANK_MODE),
        uint(EXECUTION_LAYOUT),
        uint(HAS_TABLE_BANKS),
        uint(GROUPS_PER_SUPER),
        uint(NSUPER),
        uint(NSIGN)
    ));
)METAL";

constexpr const char* kEmbeddingSource = R"METAL(
    uint linear = thread_position_in_grid.x;
    if (linear >= uint(COUNT) * uint(K)) {
        return;
    }
    uint token_position = linear / uint(K);
    uint column = linear - token_position * uint(K);
    uint output = uint(token_ids[token_position]);
    if (output >= uint(OUT)) {
        y[linear] = T(0.0f);
        return;
    }
    y[linear] = T(mfq_vq_decode_weight(
        indices_packed,
        state_packed,
        aux_packed,
        anchors,
        codebooks,
        scale_lut,
        state_to_codebank,
        bank_ids,
        parameters,
        output,
        column,
        uint(GS),
        uint(NG),
        uint(VECTOR_SIZE),
        uint(NVEC),
        uint(INDEX_BITS),
        uint(STATE_BITS),
        uint(STATES),
        uint(ENTRIES),
        uint(CODE_BANKS),
        uint(AUX_MODE),
        uint(CODE_BANK_MODE),
        uint(EXECUTION_LAYOUT),
        uint(HAS_TABLE_BANKS),
        uint(GROUPS_PER_SUPER),
        uint(NSUPER),
        uint(NSIGN)
    ));
)METAL";

constexpr const char* kHadamardSource = R"METAL(
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
        uint column_base = local_block * uint(BLOCK);
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
                uint within = pair - pair_block * stride;
                uint first =
                    pair_block * (stride << 1u) + within;
                uint second = first + stride;
                float a = values[first];
                float b = values[second];
                values[first] = a + b;
                values[second] = a - b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
)METAL";

constexpr const char* kResidualMatmulSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint logical = threadgroup_position_in_grid.x * 4u + simd_group;
    uint total = uint(M) * uint(OUT);
    if (logical >= total) {
        return;
    }
    uint input_row = logical / uint(OUT);
    uint output = logical - input_row * uint(OUT);
    float residual = 0.0f;
    for (uint block = lane; block < uint(RESIDUAL_BLOCKS); block += 32u) {
        uint record_index = output * uint(RESIDUAL_BLOCKS) + block;
        short records[2] = {
            residual_first[record_index],
            residual_second[record_index],
        };
        for (uint stream = 0u; stream < 2u; ++stream) {
            int record = int(records[stream]);
            if (record < 0) {
                continue;
            }
            uint position = uint(record) & ((1u << uint(POSITION_BITS)) - 1u);
            uint dictionary_id = uint(record) >> uint(POSITION_BITS);
            uint vector = block * uint(BLOCK_VECTORS) + position;
            if (dictionary_id >= 1024u || vector >= uint(NVEC)) {
                continue;
            }
            uint input_offset = input_row * uint(K) + vector * 8u;
            uint dictionary_offset = dictionary_id * 8u;
            for (uint component = 0u; component < 8u; ++component) {
                residual = fma(
                    float(x[input_offset + component]),
                    float(residual_codebook[dictionary_offset + component]),
                    residual
                );
            }
        }
    }
    residual = simd_sum(residual);
    if (lane == 0u) {
        y[logical] = T(float(base[logical]) + residual);
    }
)METAL";

constexpr const char* kResidualDequantizeSource = R"METAL(
    uint logical = thread_position_in_grid.x;
    uint total = uint(OUT) * uint(K);
    if (logical >= total) {
        return;
    }
    uint output = logical / uint(K);
    uint column = logical - output * uint(K);
    uint vector = column >> 3u;
    uint component = column & 7u;
    uint block = vector / uint(BLOCK_VECTORS);
    uint position_in_block = vector - block * uint(BLOCK_VECTORS);
    uint record_index = output * uint(RESIDUAL_BLOCKS) + block;
    short records[2] = {
        residual_first[record_index],
        residual_second[record_index],
    };
    float value = float(base[logical]);
    for (uint stream = 0u; stream < 2u; ++stream) {
        int record = int(records[stream]);
        if (record < 0) {
            continue;
        }
        uint position = uint(record) & ((1u << uint(POSITION_BITS)) - 1u);
        uint dictionary_id = uint(record) >> uint(POSITION_BITS);
        if (position == position_in_block && dictionary_id < 1024u) {
            value += float(residual_codebook[dictionary_id * 8u + component]);
        }
    }
    y[logical] = T(value);
)METAL";

class BlobCursor {
public:
    explicit BlobCursor(std::span<const std::uint8_t> blob)
        : blob_(blob) {}

    template <typename T>
    T scalar(const char* name) {
        require(sizeof(T), name);
        T result{};
        std::memcpy(&result, blob_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return result;
    }

    std::array<char, 4> magic(const char* name) {
        require(4, name);
        std::array<char, 4> result{};
        std::memcpy(result.data(), blob_.data() + offset_, 4);
        offset_ += 4;
        return result;
    }

    detail::StagingVector<std::uint8_t> bytes(
        std::size_t count,
        const char* name) {
        require(count, name);
        detail::StagingVector<std::uint8_t> result(
            blob_.begin() + static_cast<std::ptrdiff_t>(offset_),
            blob_.begin()
                + static_cast<std::ptrdiff_t>(offset_ + count));
        offset_ += count;
        return result;
    }

    const std::uint8_t* view(
        std::size_t count,
        const char* name) {
        require(count, name);
        const auto* result = blob_.data() + offset_;
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
    void require(std::size_t count, const char* name) const {
        if (count > blob_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated VQ ") + name);
        }
    }

    std::span<const std::uint8_t> blob_;
    std::size_t offset_ = 0;
};

struct MatrixHeader {
    std::uint8_t profile = 0;
    std::uint8_t bits = 0;
    std::uint16_t group_size = 0;
    std::int32_t axis = 0;
    std::int32_t input_size = 0;
    std::vector<std::int64_t> shape;
    std::uint32_t output_size = 0;
};

struct CanonicalVq {
    detail::StagingVector<std::uint8_t> indices;
    detail::StagingVector<std::uint8_t> states_packed;
    detail::StagingVector<std::uint8_t> auxiliary;
    detail::StagingVector<float> anchors;
    detail::StagingVector<std::int8_t> codebooks;
    detail::StagingVector<float> scales;
    detail::StagingVector<std::uint8_t> state_to_codebank;
    detail::StagingVector<std::uint8_t> bank_ids;
    detail::StagingVector<std::int8_t> rotation_signs;
    detail::StagingVector<float> residual_codebook;
    detail::StagingVector<std::int16_t> residual_first;
    detail::StagingVector<std::int16_t> residual_second;
    float parameter = 0.0f;
    std::string label;
    std::vector<int> output_shape;
    int input_size = 0;
    int output_size = 0;
    int group_size = 0;
    int groups = 0;
    int vector_size = 0;
    int vectors = 0;
    int index_bits = 0;
    int state_bits = 0;
    int states = 0;
    int entries = 0;
    int code_banks = 0;
    int aux_mode = kAuxNone;
    int code_bank_mode = kCodeBankFixed;
    int execution_layout = kExecutionStreams;
    int table_banks = 1;
    int groups_per_supergroup = 0;
    int supergroups = 1;
    int rotation_block = 0;
    std::uint64_t rotation_seed = 0;
    int residual_position_bits = 0;
    int residual_block_vectors = 0;
    int residual_blocks_per_row = 0;
};

bool magic_is(
    const std::array<char, 4>& value,
    std::string_view expected) {
    return expected.size() == value.size() &&
        std::memcmp(
            value.data(),
            expected.data(),
            value.size()) == 0;
}

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* name) {
    if (right != 0 &&
        left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::runtime_error(
            std::string("VQ ") + name + " overflows");
    }
    return left * right;
}

std::size_t checked_add(
    std::size_t left,
    std::size_t right,
    const char* name) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::runtime_error(
            std::string("VQ ") + name + " overflows");
    }
    return left + right;
}

int checked_int(std::uint64_t value, const char* name) {
    if (value == 0 ||
        value >
            static_cast<std::uint64_t>(
                std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string("invalid VQ ") + name);
    }
    return static_cast<int>(value);
}

std::size_t packed_nbytes(
    std::size_t count,
    int bits) {
    if (bits == 0) {
        return 0;
    }
    if (bits < 0 || bits > 16 ||
        count >
            (std::numeric_limits<std::size_t>::max() - 7)
                / static_cast<std::size_t>(bits)) {
        throw std::runtime_error("invalid VQ packed bit count");
    }
    return (
        count * static_cast<std::size_t>(bits) + 7
    ) / 8;
}

detail::StagingVector<std::uint8_t> padded_stream(
    BlobCursor& cursor,
    std::size_t count,
    int bits,
    const char* name) {
    auto result = cursor.bytes(
        packed_nbytes(count, bits),
        name);
    result.insert(result.end(), 2, 0);
    if (result.size() < 3) {
        result.resize(3, 0);
    }
    return result;
}

detail::StagingVector<std::uint16_t> unpack_u16_stream(
    BlobCursor& cursor,
    std::size_t count,
    int bits,
    const char* name) {
    const auto bytes = cursor.bytes(
        packed_nbytes(count, bits),
        name);
    detail::StagingVector<std::uint16_t> result(count, 0);
    for (std::size_t index = 0; index < count; ++index) {
        const auto first_bit = index * static_cast<std::size_t>(bits);
        std::uint16_t value = 0;
        for (int bit = 0; bit < bits; ++bit) {
            const auto source_bit = first_bit + static_cast<std::size_t>(bit);
            value |= static_cast<std::uint16_t>(
                ((bytes[source_bit / 8] >> (source_bit & 7)) & 1u)
                << bit);
        }
        result[index] = value;
    }
    return result;
}

float half_to_float(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const auto exponent =
        static_cast<unsigned>((bits >> 10) & 0x1fu);
    const auto mantissa =
        static_cast<unsigned>(bits & 0x03ffu);
    float value = 0.0f;
    if (exponent == 0) {
        value = std::ldexp(
            static_cast<float>(mantissa),
            -24);
    } else if (exponent == 31) {
        value = mantissa == 0
            ? std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(
            1.0f
                + static_cast<float>(mantissa) / 1024.0f,
            static_cast<int>(exponent) - 15);
    }
    return negative ? -value : value;
}

detail::StagingVector<float> read_half_values(
    BlobCursor& cursor,
    std::size_t count,
    const char* name,
    bool require_nonnegative = true) {
    detail::StagingVector<float> result(count);
    for (auto& value : result) {
        value = half_to_float(
            cursor.scalar<std::uint16_t>(name));
        if (!std::isfinite(value) ||
            (require_nonnegative && value < 0.0f)) {
            throw std::runtime_error(
                std::string("invalid VQ ") + name);
        }
    }
    return result;
}

MatrixHeader read_matrix_header(
    BlobCursor& cursor,
    const std::array<char, 4>& magic) {
    MatrixHeader result;
    result.profile = cursor.scalar<std::uint8_t>("profile");
    result.bits = cursor.scalar<std::uint8_t>("state bits");
    result.group_size =
        cursor.scalar<std::uint16_t>("group size");
    result.axis = cursor.scalar<std::int32_t>("axis");
    result.input_size =
        cursor.scalar<std::int32_t>("neuron length");
    const auto dimensions =
        cursor.scalar<std::uint32_t>("dimension count");
    if (dimensions == 0 || dimensions > 8 ||
        result.input_size <= 0) {
        throw std::runtime_error(
            "invalid VQ matrix header dimensions");
    }
    result.shape.resize(dimensions);
    for (auto& dimension : result.shape) {
        dimension = cursor.scalar<std::int64_t>("shape");
        if (dimension <= 0) {
            throw std::runtime_error(
                "invalid VQ matrix shape");
        }
    }
    result.output_size =
        cursor.scalar<std::uint32_t>("output size");
    if (result.output_size == 0) {
        throw std::runtime_error(
            "invalid VQ matrix output size");
    }
    if (result.axis != 0 || dimensions != 2 ||
        result.shape[0] != result.output_size ||
        result.shape[1] != result.input_size) {
        throw std::runtime_error(
            std::string("unsupported ")
            + std::string(magic.data(), magic.size())
            + " VQ matrix layout");
    }
    return result;
}

detail::StagingVector<std::int8_t> decode_lattice_codebook(
    const std::uint16_t* words,
    int entries,
    int vector_size) {
    if (entries <= 0 ||
        (vector_size != 4 && vector_size != 8)) {
        throw std::runtime_error(
            "invalid VQ lattice codebook dimensions");
    }
    const int digit_bits = vector_size == 8 ? 2 : 3;
    detail::StagingVector<std::int8_t> result(
        checked_product(
            static_cast<std::size_t>(entries),
            static_cast<std::size_t>(vector_size),
            "codebook size"));
    for (int entry = 0; entry < entries; ++entry) {
        const auto word = words[entry];
        for (int component = 0;
             component < vector_size;
             ++component) {
            const auto digit = static_cast<unsigned>(
                (word >> (digit_bits * component))
                & ((1u << digit_bits) - 1u));
            result[
                static_cast<std::size_t>(entry) * vector_size
                    + component
            ] = static_cast<std::int8_t>(2 * digit + 1);
        }
    }
    return result;
}

detail::StagingVector<std::int8_t> decode_lattice_codebook(
    const std::uint8_t* bytes,
    int entries,
    int vector_size) {
    std::vector<std::uint16_t> words(
        static_cast<std::size_t>(entries));
    std::memcpy(
        words.data(),
        bytes,
        checked_product(
            words.size(),
            sizeof(std::uint16_t),
            "codebook bytes"));
    return decode_lattice_codebook(
        words.data(),
        entries,
        vector_size);
}

detail::StagingVector<std::int8_t> decode_ternary_codebook(
    const std::uint16_t* words,
    int entries) {
    detail::StagingVector<std::int8_t> result(
        checked_product(
            static_cast<std::size_t>(entries),
            std::size_t{8},
            "ternary codebook size"));
    for (int entry = 0; entry < entries; ++entry) {
        const auto word = words[entry];
        for (int component = 0; component < 8; ++component) {
            const auto digit =
                static_cast<unsigned>(
                    (word >> (2 * component)) & 3u);
            if (digit > 2) {
                throw std::runtime_error(
                    "invalid ternary VQ codebook digit");
            }
            result[
                static_cast<std::size_t>(entry) * 8
                    + component
            ] = static_cast<std::int8_t>(
                static_cast<int>(digit) - 1);
        }
    }
    return result;
}

detail::StagingVector<std::int8_t> decode_ternary_codebook(
    const std::uint8_t* bytes,
    int entries) {
    std::vector<std::uint16_t> words(
        static_cast<std::size_t>(entries));
    std::memcpy(
        words.data(),
        bytes,
        checked_product(
            words.size(),
            sizeof(std::uint16_t),
            "ternary codebook bytes"));
    return decode_ternary_codebook(
        words.data(),
        entries);
}

void append_product_codebook(
    CanonicalVq& result,
    const std::uint8_t* table,
    bool short_profile) {
    const int states = short_profile ? 4 : 8;
    const int first_entries = 8;
    const int second_entries = short_profile ? 8 : 16;
    const int table_version = short_profile ? 2 : 1;
    const std::array<int, 6> expected{
        table_version,
        states,
        3,
        short_profile ? 3 : 4,
        24,
        8,
    };
    for (std::size_t index = 0;
         index < expected.size();
         ++index) {
        if (table[index] != expected[index]) {
            throw std::runtime_error(
                "unsupported NPQ table profile");
        }
    }
    if (table[6] != 0 || table[7] != 0) {
        throw std::runtime_error(
            "invalid NPQ reserved table bytes");
    }
    for (int state = 0; state < states; ++state) {
        std::uint16_t bits{};
        std::memcpy(
            &bits,
            table + 8 + state * 2,
            sizeof(bits));
        const float scale = half_to_float(bits);
        if (!std::isfinite(scale) || scale < 0.0f) {
            throw std::runtime_error(
                "invalid NPQ scale LUT");
        }
        result.scales.push_back(scale);
    }
    const auto* first = reinterpret_cast<const std::int8_t*>(
        table + 64);
    const auto first_count =
        states * first_entries * 4;
    const auto* second = first + first_count;
    for (int state = 0; state < states; ++state) {
        for (int second_index = 0;
             second_index < second_entries;
             ++second_index) {
            for (int first_index = 0;
                 first_index < first_entries;
                 ++first_index) {
                const auto first_base =
                    (state * first_entries + first_index) * 4;
                const auto second_base =
                    (state * second_entries + second_index) * 4;
                result.codebooks.insert(
                    result.codebooks.end(),
                    first + first_base,
                    first + first_base + 4);
                result.codebooks.insert(
                    result.codebooks.end(),
                    second + second_base,
                    second + second_base + 4);
            }
        }
    }
}

void validate_canonical(const CanonicalVq& value) {
    if (value.input_size <= 0 ||
        value.output_size <= 0 ||
        value.group_size <= 0 ||
        value.groups <= 0 ||
        value.vector_size <= 0 ||
        value.vectors <= 0 ||
        value.index_bits <= 0 ||
        value.index_bits > 16 ||
        value.state_bits <= 0 ||
        value.state_bits > 8 ||
        value.states != (1 << value.state_bits) ||
        value.entries != (1 << value.index_bits) ||
        value.code_banks <= 0 ||
        value.table_banks <= 0 ||
        value.groups_per_supergroup <= 0 ||
        value.supergroups <= 0 ||
        value.anchors.size() !=
            static_cast<std::size_t>(value.output_size) ||
        value.scales.size() !=
            static_cast<std::size_t>(
                value.table_banks * value.states) ||
        value.codebooks.size() !=
            static_cast<std::size_t>(
                value.table_banks
                * value.code_banks
                * value.entries
                * value.vector_size) ||
        value.state_to_codebank.size() !=
            static_cast<std::size_t>(value.states) ||
        value.bank_ids.size() !=
            static_cast<std::size_t>(
                value.output_size * value.supergroups)) {
        throw std::runtime_error(
            "inconsistent canonical VQ execution layout");
    }
    for (const auto bank : value.bank_ids) {
        if (bank >= value.table_banks) {
            throw std::runtime_error(
                "VQ bank selector references a missing table");
        }
    }
    for (const auto bank : value.state_to_codebank) {
        if (bank >= value.code_banks) {
            throw std::runtime_error(
                "VQ state references a missing code bank");
        }
    }
}

struct NvqCodebookProfile {
    int id = 0;
    int vector_size = 0;
    int index_bits = 0;
    const char* name = "";
};

NvqCodebookProfile nvq_profile(int id) {
    switch (id) {
        case 1:
            return {1, 8, 8, "e8_256"};
        case 2:
            return {2, 4, 8, "d4_256"};
        case 3:
            return {3, 4, 9, "d4_512"};
        case 4:
            return {4, 8, 10, "e8_1024"};
        case 5:
            return {5, 8, 12, "e8_4096"};
        case 6:
            return {6, 4, 10, "d4_1024"};
        default:
            throw std::runtime_error(
                "unknown NVQ codebook profile");
    }
}

std::string expected_nvq_dtype(
    const NvqCodebookProfile& profile,
    bool jsc) {
    if (!jsc) {
        if (profile.id == 1) {
            return "NVQ2";
        }
        if (profile.id == 2) {
            return "NVQ3";
        }
        throw std::runtime_error(
            "extended plain NVQ profile has no public dtype");
    }
    switch (profile.id) {
        case 1:
            return "NVQ2J";
        case 2:
            return "NVQ3J";
        case 3:
            return "NVQ3J-512";
        case 4:
            return "NVQ2J-L";
        case 5:
            return "NVQ2J-XL";
        case 6:
            return "NVQ3J-L";
        default:
            throw std::runtime_error(
                "unsupported NVQ-JSC public dtype");
    }
}

CanonicalVq parse_nvq(
    std::string_view dtype,
    std::span<const std::uint8_t> blob) {
    constexpr std::uint8_t kIndexParity = 0x80;
    constexpr std::uint8_t kCustomCodebook = 0x40;
    constexpr std::uint8_t kJsc = 0x20;

    BlobCursor cursor(blob);
    const auto magic = cursor.magic("NVQ magic");
    if (!magic_is(magic, "NVQ1") &&
        !magic_is(magic, "NIQ1")) {
        throw std::runtime_error("invalid NVQ magic");
    }
    const auto header = read_matrix_header(cursor, magic);
    const bool index_parity =
        (header.profile & kIndexParity) != 0;
    const bool custom =
        (header.profile & kCustomCodebook) != 0;
    const bool jsc = (header.profile & kJsc) != 0;
    const int id = header.profile
        & ~(kIndexParity | kCustomCodebook | kJsc);
    const auto profile = nvq_profile(id);
    const auto expected_dtype =
        expected_nvq_dtype(profile, jsc);
    if (dtype != expected_dtype) {
        throw std::runtime_error(
            "MFQ dtype/blob mismatch: "
            + std::string(dtype)
            + " contains " + expected_dtype);
    }
    if (index_parity && profile.id != 1) {
        throw std::runtime_error(
            "NVQ index-parity signs require e8_256");
    }
    if (header.bits <= 0 || header.bits > 8 ||
        header.group_size == 0 ||
        header.group_size % 8 != 0 ||
        (jsc && header.input_size % profile.vector_size != 0)) {
        throw std::runtime_error(
            "unsupported NVQ group profile");
    }

    CanonicalVq result;
    result.label = expected_dtype;
    result.output_shape = {
        checked_int(header.output_size, "output size"),
    };
    result.output_size = result.output_shape.front();
    result.input_size =
        checked_int(header.input_size, "input size");
    result.group_size = header.group_size;
    result.groups =
        (result.input_size + result.group_size - 1)
        / result.group_size;
    result.vector_size = profile.vector_size;
    result.vectors =
        (result.input_size + result.vector_size - 1)
        / result.vector_size;
    result.index_bits = profile.index_bits;
    result.state_bits = header.bits;
    result.states = 1 << result.state_bits;
    result.entries = 1 << result.index_bits;
    result.groups_per_supergroup = result.groups;
    result.supergroups = 1;
    result.table_banks = 1;
    result.bank_ids.assign(
        static_cast<std::size_t>(result.output_size),
        0);

    if (jsc) {
        if (custom || index_parity ||
            result.group_size != 24 ||
            result.state_bits != 4) {
            throw std::runtime_error(
                "invalid NVQ-JSC profile flags");
        }
        const auto version =
            cursor.scalar<std::uint8_t>(
                "NVQ-JSC metadata version");
        const auto banks =
            cursor.scalar<std::uint8_t>(
                "NVQ-JSC bank count");
        const auto states =
            cursor.scalar<std::uint8_t>(
                "NVQ-JSC state count");
        const auto state_mode =
            cursor.scalar<std::uint8_t>(
                "NVQ-JSC state mode");
        if ((version != 1 && version != 2) ||
            (banks != 1 && banks != 2 && banks != 4) ||
            states != 16 ||
            state_mode > 1) {
            throw std::runtime_error(
                "unsupported NVQ-JSC metadata");
        }
        result.scales = read_half_values(
            cursor,
            16,
            "NVQ-JSC scale LUT");
        result.state_to_codebank =
            cursor.bytes(16, "NVQ-JSC bank map");
        const auto storage_layout =
            cursor.scalar<std::uint8_t>(
                "NVQ-JSC storage layout");
        const auto reserved =
            cursor.bytes(11, "NVQ-JSC reserved metadata");
        if ((version == 1 && storage_layout != 0) ||
            (version == 2 && storage_layout != 1)) {
            throw std::runtime_error(
                "unsupported NVQ-JSC storage layout");
        }
        if (std::any_of(
                reserved.begin(),
                reserved.end(),
                [](std::uint8_t value) {
                    return value != 0;
                })) {
            throw std::runtime_error(
                "NVQ-JSC reserved metadata is nonzero");
        }
        result.code_banks = banks;
        for (const auto bank : result.state_to_codebank) {
            if (bank >= banks) {
                throw std::runtime_error(
                    "NVQ-JSC state bank is out of range");
            }
        }
        const auto codebook_size = checked_product(
            static_cast<std::size_t>(banks),
            checked_product(
                static_cast<std::size_t>(result.entries),
                static_cast<std::size_t>(
                    result.vector_size),
                "JSC codebook size"),
            "JSC banked codebook size");
        const auto* codebook =
            cursor.view(codebook_size, "NVQ-JSC codebooks");
        result.codebooks.assign(
            reinterpret_cast<const std::int8_t*>(codebook),
            reinterpret_cast<const std::int8_t*>(codebook)
                + static_cast<std::ptrdiff_t>(codebook_size));
        result.aux_mode = kAuxSignEven;
        result.code_bank_mode = kCodeBankState;
        result.execution_layout = version == 2
            ? kExecutionGroup64
            : kExecutionStreams;
        if (result.execution_layout == kExecutionGroup64 &&
            profile.id != 5) {
            throw std::runtime_error(
                "NVQ-JSC group64 storage requires NVQ2J-XL");
        }
    } else {
        result.code_banks = 1;
        result.state_to_codebank.assign(
            static_cast<std::size_t>(result.states),
            0);
        result.scales.resize(
            static_cast<std::size_t>(result.states));
        for (int state = 0; state < result.states; ++state) {
            result.scales[state] =
                static_cast<float>(state);
        }
        if (custom) {
            const auto* packed = cursor.view(
                checked_product(
                    static_cast<std::size_t>(result.entries),
                    sizeof(std::uint16_t),
                    "custom codebook size"),
                "NVQ custom codebook");
            result.codebooks = decode_lattice_codebook(
                packed,
                result.entries,
                result.vector_size);
        } else if (profile.id == 1) {
            result.codebooks = decode_lattice_codebook(
                nvq_codebooks::kNvq2CodebookPacked,
                result.entries,
                result.vector_size);
        } else if (profile.id == 2) {
            result.codebooks = decode_lattice_codebook(
                nvq_codebooks::kNvq3CodebookPacked,
                result.entries,
                result.vector_size);
        } else {
            throw std::runtime_error(
                "missing builtin extended NVQ codebook");
        }
        result.aux_mode = index_parity
            ? kAuxSignIndexParity
            : kAuxSignEven;
        result.code_bank_mode = kCodeBankFixed;
    }

    result.anchors = read_half_values(
        cursor,
        static_cast<std::size_t>(result.output_size),
        "neuron anchors");
    if (result.execution_layout == kExecutionGroup64) {
        const auto records = checked_product(
            static_cast<std::size_t>(result.output_size),
            static_cast<std::size_t>(result.groups),
            "group64 record count");
        result.indices = cursor.bytes(
            checked_product(records, std::size_t{8},
                "group64 stream size"),
            "group64 stream");
        for (std::size_t record = 0; record < records; ++record) {
            std::uint64_t packed{};
            std::memcpy(
                &packed,
                result.indices.data() + record * 8,
                sizeof(packed));
            const auto group = record
                % static_cast<std::size_t>(result.groups);
            for (int local = 0; local < 3; ++local) {
                const auto segment = static_cast<std::uint32_t>(
                    (packed >> (local * 20)) & 0xfffffu);
                const auto sign = (segment >> 12) & 0xffu;
                const auto sign7 = sign & 0x7fu;
                if ((sign >> 7) !=
                    (std::popcount(sign7) & 1u)) {
                    throw std::runtime_error(
                        "invalid NVQ-JSC group64 parity bit");
                }
                if (group * 3 + static_cast<std::size_t>(local) >=
                        static_cast<std::size_t>(result.vectors) &&
                    segment != 0) {
                    throw std::runtime_error(
                        "NVQ-JSC group64 padding must be zero");
                }
            }
        }
        result.indices.insert(result.indices.end(), 2, 0);
        result.states_packed.assign(3, 0);
        result.auxiliary.assign(3, 0);
    } else {
        result.states_packed = padded_stream(
            cursor,
            checked_product(
                static_cast<std::size_t>(result.output_size),
                static_cast<std::size_t>(result.groups),
                "state count"),
            result.state_bits,
            "state stream");
        result.indices = padded_stream(
            cursor,
            checked_product(
                static_cast<std::size_t>(result.output_size),
                static_cast<std::size_t>(result.vectors),
                "index count"),
            result.index_bits,
            "index stream");
        const int signs =
            (result.input_size + 7) / 8;
        result.auxiliary = padded_stream(
            cursor,
            checked_product(
                static_cast<std::size_t>(result.output_size),
                static_cast<std::size_t>(signs),
                "sign count"),
            7,
            "sign stream");
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "invalid NVQ tensor tail");
    }
    validate_canonical(result);
    return result;
}

CanonicalVq parse_nvq1_l(
    std::string_view dtype,
    std::span<const std::uint8_t> blob) {
    if (dtype != "NVQ1-L") {
        throw std::runtime_error(
            "MFQ dtype/blob mismatch for NVQ1-L");
    }
    BlobCursor cursor(blob);
    const auto magic = cursor.magic("NVQ1-L magic");
    if (!magic_is(magic, "NQ1L")) {
        throw std::runtime_error("invalid NVQ1-L magic");
    }
    const auto header = read_matrix_header(cursor, magic);
    if ((header.profile != 1 && header.profile != 2) ||
        header.bits == 0 || header.bits > 8 ||
        header.group_size == 0 ||
        header.group_size % 8 != 0) {
        throw std::runtime_error(
            "unsupported NVQ1-L stream profile");
    }

    CanonicalVq result;
    result.label = "NVQ1-L";
    result.output_size =
        checked_int(header.output_size, "output size");
    result.output_shape = {result.output_size};
    result.input_size =
        checked_int(header.input_size, "input size");
    result.group_size = header.group_size;
    result.groups =
        (result.input_size + result.group_size - 1)
        / result.group_size;
    result.vector_size = 8;
    result.vectors =
        (result.input_size + 7) / 8;
    result.index_bits = 11;
    result.state_bits = header.bits;
    result.states = 1 << result.state_bits;
    result.entries = 2048;
    result.code_banks = 1;
    result.aux_mode = kAuxDelta;
    result.code_bank_mode = kCodeBankFixed;
    result.table_banks = 1;
    result.groups_per_supergroup = result.groups;
    result.supergroups = 1;
    result.parameter = 0.125f;
    result.scales.resize(
        static_cast<std::size_t>(result.states));
    for (int state = 0; state < result.states; ++state) {
        result.scales[state] = static_cast<float>(state);
    }
    result.state_to_codebank.assign(
        static_cast<std::size_t>(result.states),
        0);
    result.bank_ids.assign(
        static_cast<std::size_t>(result.output_size),
        0);
    if (header.profile == 2) {
        const auto* packed = cursor.view(
            4096,
            "NVQ1-L custom codebook");
        result.codebooks = decode_ternary_codebook(
            packed,
            result.entries);
    } else {
        result.codebooks = decode_ternary_codebook(
            nvq_codebooks::kNvq1LCodebookPacked,
            result.entries);
    }

    result.anchors = read_half_values(
        cursor,
        static_cast<std::size_t>(result.output_size),
        "NVQ1-L anchors");
    const auto state_count = checked_product(
        static_cast<std::size_t>(result.output_size),
        static_cast<std::size_t>(result.groups),
        "NVQ1-L state count");
    result.states_packed = padded_stream(
        cursor,
        state_count,
        result.state_bits,
        "NVQ1-L state stream");
    result.indices = padded_stream(
        cursor,
        checked_product(
            static_cast<std::size_t>(result.output_size),
            static_cast<std::size_t>(result.vectors),
            "NVQ1-L index count"),
        result.index_bits,
        "NVQ1-L index stream");
    result.auxiliary = padded_stream(
        cursor,
        state_count,
        1,
        "NVQ1-L delta stream");
    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "invalid NVQ1-L tensor tail");
    }
    validate_canonical(result);
    return result;
}

CanonicalVq parse_nvq1_s(
    std::string_view dtype,
    std::span<const std::uint8_t> blob) {
    if (dtype != "NVQ1-S") {
        throw std::runtime_error(
            "MFQ dtype/blob mismatch for NVQ1-S");
    }
    BlobCursor cursor(blob);
    const auto magic = cursor.magic("NVQ1-S magic");
    if (!magic_is(magic, "NQ1S")) {
        throw std::runtime_error("invalid NVQ1-S magic");
    }
    const auto header = read_matrix_header(cursor, magic);
    if (header.profile != 1 ||
        header.bits != 4 ||
        header.group_size != 24 ||
        header.input_size % 8 != 0) {
        throw std::runtime_error(
            "unsupported NVQ1-S stream profile");
    }

    CanonicalVq result;
    result.label = "NVQ1-S";
    result.output_size =
        checked_int(header.output_size, "output size");
    result.output_shape = {result.output_size};
    result.input_size =
        checked_int(header.input_size, "input size");
    result.group_size = 24;
    result.groups = (result.input_size + 23) / 24;
    result.vector_size = 8;
    result.vectors = result.input_size / 8;
    result.index_bits = 9;
    result.state_bits = 4;
    result.states = 16;
    result.entries = 512;
    result.code_banks = 2;
    result.aux_mode = kAuxDelta;
    result.code_bank_mode = kCodeBankAux;
    result.table_banks = 1;
    result.groups_per_supergroup = result.groups;
    result.supergroups = 1;
    result.parameter = 0.15625f;
    result.scales.resize(16);
    for (int state = 0; state < 16; ++state) {
        result.scales[state] = static_cast<float>(state);
    }
    result.state_to_codebank.assign(16, 0);
    result.bank_ids.assign(
        static_cast<std::size_t>(result.output_size),
        0);

    const auto* table = cursor.view(
        2048,
        "NVQ1-S codebooks");
    result.codebooks.reserve(2 * 512 * 8);
    for (int bank = 0; bank < 2; ++bank) {
        auto decoded = decode_ternary_codebook(
            table + bank * 1024,
            512);
        result.codebooks.insert(
            result.codebooks.end(),
            decoded.begin(),
            decoded.end());
    }
    result.anchors = read_half_values(
        cursor,
        static_cast<std::size_t>(result.output_size),
        "NVQ1-S anchors");
    const auto state_count = checked_product(
        static_cast<std::size_t>(result.output_size),
        static_cast<std::size_t>(result.groups),
        "NVQ1-S state count");
    result.states_packed = padded_stream(
        cursor,
        state_count,
        4,
        "NVQ1-S state stream");
    result.indices = padded_stream(
        cursor,
        checked_product(
            static_cast<std::size_t>(result.output_size),
            static_cast<std::size_t>(result.vectors),
            "NVQ1-S index count"),
        9,
        "NVQ1-S index stream");
    result.auxiliary = padded_stream(
        cursor,
        state_count,
        1,
        "NVQ1-S delta stream");
    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "invalid NVQ1-S tensor tail");
    }
    validate_canonical(result);
    return result;
}

CanonicalVq parse_npq(
    std::string_view dtype,
    std::span<const std::uint8_t> blob,
    bool short_profile) {
    const auto expected_dtype =
        short_profile ? "NPQ0-S" : "NPQ0-L";
    if (dtype != expected_dtype) {
        throw std::runtime_error(
            "MFQ dtype/blob mismatch for "
            + std::string(expected_dtype));
    }
    BlobCursor cursor(blob);
    const auto magic = cursor.magic("NPQ magic");
    if (!magic_is(
            magic,
            short_profile ? "NPQS" : "NPQL")) {
        throw std::runtime_error("invalid NPQ magic");
    }
    const auto header = read_matrix_header(cursor, magic);
    const int states = short_profile ? 4 : 8;
    const int state_bits = short_profile ? 2 : 3;
    const int index_bits = short_profile ? 6 : 7;
    const std::size_t table_bytes =
        short_profile ? 320 : 832;
    if (header.profile !=
            static_cast<std::uint8_t>(
                short_profile ? 2 : 1) ||
        header.bits != state_bits ||
        header.group_size != 24 ||
        header.input_size % 8 != 0) {
        throw std::runtime_error(
            "unsupported NPQ stream profile");
    }

    CanonicalVq result;
    result.label = expected_dtype;
    result.output_size =
        checked_int(header.output_size, "output size");
    result.output_shape = {result.output_size};
    result.input_size =
        checked_int(header.input_size, "input size");
    result.group_size = 24;
    result.groups = (result.input_size + 23) / 24;
    result.vector_size = 8;
    result.vectors = result.input_size / 8;
    result.index_bits = index_bits;
    result.state_bits = state_bits;
    result.states = states;
    result.entries = 1 << index_bits;
    result.code_banks = states;
    result.aux_mode = kAuxNone;
    result.code_bank_mode = kCodeBankState;
    result.table_banks = 1;
    result.groups_per_supergroup = result.groups;
    result.supergroups = 1;
    result.state_to_codebank.resize(
        static_cast<std::size_t>(states));
    for (int state = 0; state < states; ++state) {
        result.state_to_codebank[state] =
            static_cast<std::uint8_t>(state);
    }
    result.bank_ids.assign(
        static_cast<std::size_t>(result.output_size),
        0);
    const auto* table =
        cursor.view(table_bytes, "NPQ tables");
    append_product_codebook(
        result,
        table,
        short_profile);
    result.anchors = read_half_values(
        cursor,
        static_cast<std::size_t>(result.output_size),
        "NPQ anchors");
    result.states_packed = padded_stream(
        cursor,
        checked_product(
            static_cast<std::size_t>(result.output_size),
            static_cast<std::size_t>(result.groups),
            "NPQ state count"),
        state_bits,
        "NPQ state stream");
    result.indices = padded_stream(
        cursor,
        checked_product(
            static_cast<std::size_t>(result.output_size),
            static_cast<std::size_t>(result.vectors),
            "NPQ index count"),
        index_bits,
        "NPQ index stream");
    result.auxiliary.assign(3, 0);
    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "invalid NPQ tensor tail");
    }
    validate_canonical(result);
    return result;
}

struct NepqProfile {
    int id = 0;
    const char* label = "";
    int base_id = 0;
    int state_bits = 0;
    int index_bits = 0;
    int auxiliary_bits = 0;
    std::size_t table_bytes = 0;
    int residual_block_vectors = 0;
    bool residual_second = false;
};

NepqProfile nepq_profile(int id) {
    switch (id) {
        case 0:
            return {0, "NEPQ0-S", 0, 2, 6, 0, 320, 0, false};
        case 1:
            return {1, "NEPQ0-L", 1, 3, 7, 0, 832, 0, false};
        case 2:
            return {2, "NEPQ1-S", 2, 4, 9, 1, 2048, 0, false};
        case 3:
            return {3, "NEPQ1-L", 3, 3, 11, 1, 4096, 0, false};
        case 4:
            return {4, "NEPQ0-A", 0, 2, 6, 0, 320, 24, false};
        case 5:
            return {5, "NEPQ1-A", 2, 4, 9, 1, 2048, 16, true};
        default:
            throw std::runtime_error(
                "unknown NEPQ cohort profile");
    }
}

detail::StagingVector<std::int8_t> parse_rotation_payload(
    std::span<const std::uint8_t> payload,
    int input_size,
    int rotation_block,
    std::uint64_t rotation_seed) {
    if (rotation_block == 0) {
        if (!payload.empty()) {
            throw std::runtime_error(
                "unexpected NEPQ rotation metadata");
        }
        return {};
    }
    const auto expected_size = checked_add(
        std::size_t{20},
        static_cast<std::size_t>(input_size),
        "rotation payload");
    if (payload.size() != expected_size) {
        throw std::runtime_error(
            "rotated NEPQ cohort lacks its HSG1 sign vector");
    }
    BlobCursor cursor(payload);
    if (!magic_is(
            cursor.magic("rotation magic"),
            "HSG1")) {
        throw std::runtime_error(
            "invalid NEPQ rotation metadata magic");
    }
    const auto width =
        cursor.scalar<std::uint32_t>("rotation width");
    const auto block =
        cursor.scalar<std::uint32_t>("rotation block");
    const auto seed =
        cursor.scalar<std::uint64_t>("rotation seed");
    if (width != static_cast<std::uint32_t>(input_size) ||
        block != static_cast<std::uint32_t>(rotation_block) ||
        seed != rotation_seed) {
        throw std::runtime_error(
            "NEPQ rotation metadata does not match its payload");
    }
    const auto raw = cursor.bytes(
        static_cast<std::size_t>(input_size),
        "rotation signs");
    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "invalid NEPQ rotation metadata tail");
    }
    detail::StagingVector<std::int8_t> signs(raw.size());
    std::memcpy(signs.data(), raw.data(), raw.size());
    for (const auto sign : signs) {
        if (sign != -1 && sign != 1) {
            throw std::runtime_error(
                "invalid NEPQ rotation sign");
        }
    }
    return signs;
}

CanonicalVq parse_nepq(
    std::string_view dtype,
    std::span<const std::uint8_t> blob,
    std::span<const std::uint8_t> runtime_payload) {
    BlobCursor cursor(blob);
    if (!magic_is(cursor.magic("NEPQ magic"), "NEP1")) {
        throw std::runtime_error("invalid NEPQ cohort magic");
    }
    const auto version =
        cursor.scalar<std::uint8_t>("NEPQ version");
    const auto profile_id =
        cursor.scalar<std::uint8_t>("NEPQ profile");
    const auto groups_per_super =
        cursor.scalar<std::uint8_t>(
            "NEPQ groups per super-group");
    const auto flags =
        cursor.scalar<std::uint8_t>("NEPQ flags");
    const auto n_experts_raw =
        cursor.scalar<std::uint32_t>("NEPQ expert count");
    const auto out_per_expert_raw =
        cursor.scalar<std::uint32_t>(
            "NEPQ output rows per expert");
    const auto input_size_raw =
        cursor.scalar<std::uint32_t>("NEPQ input size");
    const auto table_banks_raw =
        cursor.scalar<std::uint32_t>("NEPQ table-bank count");
    const auto rotation_block_raw =
        cursor.scalar<std::uint32_t>("NEPQ rotation block");
    const auto rotation_seed =
        cursor.scalar<std::uint64_t>("NEPQ rotation seed");

    const auto profile = nepq_profile(profile_id);
    if (dtype != profile.label) {
        throw std::runtime_error(
            "MFQ dtype/blob mismatch: "
            + std::string(dtype)
            + " contains " + profile.label);
    }
    if (version != 1 ||
        groups_per_super != 4 ||
        (flags & ~std::uint8_t{1}) != 0 ||
        ((flags & 1u) != 0) != (rotation_block_raw != 0)) {
        throw std::runtime_error(
            "unsupported NEPQ cohort header");
    }
    const int n_experts =
        checked_int(n_experts_raw, "NEPQ expert count");
    const int out_per_expert = checked_int(
        out_per_expert_raw,
        "NEPQ output rows per expert");
    const int input_size =
        checked_int(input_size_raw, "NEPQ input size");
    const int table_banks =
        checked_int(table_banks_raw, "NEPQ table-bank count");
    const int rotation_block = rotation_block_raw == 0
        ? 0
        : checked_int(
              rotation_block_raw,
              "NEPQ rotation block");
    if (input_size % 8 != 0 ||
        table_banks > 256) {
        throw std::runtime_error(
            "invalid NEPQ cohort dimensions");
    }
    if (rotation_block != 0 &&
        ((rotation_block & (rotation_block - 1)) != 0 ||
         input_size % rotation_block != 0 ||
         rotation_block > 8192)) {
        throw std::runtime_error(
            "invalid NEPQ Metal Hadamard block");
    }
    if (rotation_block == 0 && rotation_seed != 0) {
        throw std::runtime_error(
            "NEPQ rotation seed requires a nonzero block");
    }
    if (profile.residual_block_vectors != 0 && rotation_block == 0) {
        throw std::runtime_error(
            "NEPQ-A requires a Hadamard rotation");
    }

    const auto output_count = checked_product(
        static_cast<std::size_t>(n_experts),
        static_cast<std::size_t>(out_per_expert),
        "NEPQ output size");
    CanonicalVq result;
    result.label = profile.label;
    result.output_size = checked_int(
        output_count,
        "NEPQ flattened output size");
    result.output_shape = {n_experts, out_per_expert};
    result.input_size = input_size;
    result.group_size = 24;
    result.groups = (input_size + 23) / 24;
    result.vector_size = 8;
    result.vectors = input_size / 8;
    result.index_bits = profile.index_bits;
    result.state_bits = profile.state_bits;
    result.states = 1 << result.state_bits;
    result.entries = 1 << result.index_bits;
    result.table_banks = table_banks;
    result.groups_per_supergroup = 4;
    result.supergroups = (result.groups + 3) / 4;
    result.rotation_block = rotation_block;
    result.rotation_seed = rotation_seed;
    result.rotation_signs = parse_rotation_payload(
        runtime_payload,
        input_size,
        rotation_block,
        rotation_seed);

    if (profile.base_id == 0 || profile.base_id == 1) {
        result.code_banks = result.states;
        result.code_bank_mode = kCodeBankState;
        result.aux_mode = kAuxNone;
        result.state_to_codebank.resize(
            static_cast<std::size_t>(result.states));
        for (int state = 0; state < result.states; ++state) {
            result.state_to_codebank[state] =
                static_cast<std::uint8_t>(state);
        }
    } else if (profile.base_id == 2) {
        result.code_banks = 2;
        result.code_bank_mode = kCodeBankAux;
        result.aux_mode = kAuxDelta;
        result.parameter = 0.15625f;
        result.state_to_codebank.assign(
            static_cast<std::size_t>(result.states),
            0);
    } else {
        result.code_banks = 1;
        result.code_bank_mode = kCodeBankFixed;
        result.aux_mode = kAuxDelta;
        result.parameter = 0.125f;
        result.state_to_codebank.assign(
            static_cast<std::size_t>(result.states),
            0);
    }

    for (int bank = 0; bank < table_banks; ++bank) {
        const auto* table = cursor.view(
            profile.table_bytes,
            "NEPQ table bank");
        if (profile.base_id == 0 || profile.base_id == 1) {
            append_product_codebook(
                result,
                table,
                profile.base_id == 0);
            continue;
        }
        for (int state = 0;
             state < result.states;
             ++state) {
            result.scales.push_back(
                static_cast<float>(state));
        }
        if (profile.base_id == 2) {
            for (int code_bank = 0;
                 code_bank < 2;
                 ++code_bank) {
                auto decoded = decode_ternary_codebook(
                    table + code_bank * 1024,
                    512);
                result.codebooks.insert(
                    result.codebooks.end(),
                    decoded.begin(),
                    decoded.end());
            }
        } else {
            auto decoded = decode_ternary_codebook(
                table,
                2048);
            result.codebooks.insert(
                result.codebooks.end(),
                decoded.begin(),
                decoded.end());
        }
    }

    result.anchors = read_half_values(
        cursor,
        output_count,
        "NEPQ neuron anchors");
    const auto state_count = checked_product(
        output_count,
        static_cast<std::size_t>(result.groups),
        "NEPQ state count");
    result.states_packed = padded_stream(
        cursor,
        state_count,
        result.state_bits,
        "NEPQ state stream");
    result.indices = padded_stream(
        cursor,
        checked_product(
            output_count,
            static_cast<std::size_t>(result.vectors),
            "NEPQ index count"),
        result.index_bits,
        "NEPQ index stream");
    result.auxiliary = padded_stream(
        cursor,
        state_count,
        profile.auxiliary_bits,
        "NEPQ auxiliary stream");
    result.bank_ids = cursor.bytes(
        checked_product(
            output_count,
            static_cast<std::size_t>(result.supergroups),
            "NEPQ bank-selector count"),
        "NEPQ bank selectors");
    if (profile.residual_block_vectors != 0) {
        const auto header_offset = cursor.offset();
        if (!magic_is(
                cursor.magic("NEPQ-A residual magic"),
                "NRA1")) {
            throw std::runtime_error(
                "invalid NEPQ-A residual header");
        }
        const auto residual_version =
            cursor.scalar<std::uint8_t>("NEPQ-A residual version");
        const auto record_bits =
            cursor.scalar<std::uint8_t>("NEPQ-A residual record bits");
        const auto position_bits =
            cursor.scalar<std::uint8_t>("NEPQ-A residual position bits");
        const auto residual_flags =
            cursor.scalar<std::uint8_t>("NEPQ-A residual flags");
        const auto dictionary_entries =
            cursor.scalar<std::uint32_t>("NEPQ-A dictionary entries");
        const auto block_count =
            cursor.scalar<std::uint32_t>("NEPQ-A block count");
        const auto second_count =
            cursor.scalar<std::uint32_t>("NEPQ-A second count");
        const auto padding_nbytes =
            cursor.scalar<std::uint32_t>("NEPQ-A padding size");
        const auto reserved =
            cursor.scalar<std::uint64_t>("NEPQ-A reserved field");
        const auto consumed_header = cursor.offset() - header_offset;
        if (consumed_header > 64) {
            throw std::runtime_error("invalid NEPQ-A residual header size");
        }
        const auto header_padding = cursor.bytes(
            64 - consumed_header,
            "NEPQ-A residual header padding");
        if (std::any_of(
                header_padding.begin(),
                header_padding.end(),
                [](std::uint8_t value) { return value != 0; })) {
            throw std::runtime_error(
                "NEPQ-A residual header padding must be zero");
        }
        int expected_position_bits = 0;
        for (int capacity = 1;
             capacity < profile.residual_block_vectors;
             capacity <<= 1) {
            ++expected_position_bits;
        }
        const int expected_record_bits = expected_position_bits + 10;
        result.residual_block_vectors = profile.residual_block_vectors;
        result.residual_position_bits = expected_position_bits;
        result.residual_blocks_per_row =
            (result.vectors + result.residual_block_vectors - 1)
            / result.residual_block_vectors;
        const auto expected_blocks = checked_product(
            output_count,
            static_cast<std::size_t>(result.residual_blocks_per_row),
            "NEPQ-A residual block count");
        const auto expected_flags =
            static_cast<std::uint8_t>(profile.residual_second ? 1 : 0);
        if (residual_version != 1 ||
            record_bits != expected_record_bits ||
            position_bits != expected_position_bits ||
            residual_flags != expected_flags ||
            dictionary_entries != 1024 ||
            block_count != expected_blocks ||
            reserved != 0 ||
            (!profile.residual_second && second_count != 0)) {
            throw std::runtime_error(
                "unsupported NEPQ-A residual profile");
        }
        result.residual_codebook = read_half_values(
            cursor,
            1024 * 8,
            "NEPQ-A residual dictionary",
            false);
        const auto first = unpack_u16_stream(
            cursor,
            expected_blocks,
            expected_record_bits,
            "NEPQ-A first residual stream");
        result.residual_first.assign(first.begin(), first.end());
        result.residual_second.assign(expected_blocks, -1);
        if (profile.residual_second) {
            const auto mask = unpack_u16_stream(
                cursor,
                expected_blocks,
                1,
                "NEPQ-A second residual bitmap");
            const auto second = unpack_u16_stream(
                cursor,
                second_count,
                expected_record_bits,
                "NEPQ-A second residual stream");
            std::size_t compact = 0;
            for (std::size_t block = 0; block < expected_blocks; ++block) {
                if (mask[block] == 0) {
                    continue;
                }
                if (compact >= second.size()) {
                    throw std::runtime_error(
                        "NEPQ-A second residual count mismatch");
                }
                result.residual_second[block] =
                    static_cast<std::int16_t>(second[compact++]);
            }
            if (compact != second.size()) {
                throw std::runtime_error(
                    "NEPQ-A second residual count mismatch");
            }
        }
        const auto position_mask =
            (1u << result.residual_position_bits) - 1u;
        auto validate_record = [&](std::uint16_t record, std::size_t block) {
            const auto position = record & position_mask;
            const auto dictionary_id =
                record >> result.residual_position_bits;
            const auto block_in_row = static_cast<int>(
                block % static_cast<std::size_t>(
                    result.residual_blocks_per_row));
            const int available = std::min(
                result.residual_block_vectors,
                result.vectors
                    - block_in_row * result.residual_block_vectors);
            if (dictionary_id >= 1024 ||
                position >= static_cast<std::uint32_t>(available)) {
                throw std::runtime_error(
                    "NEPQ-A residual record is out of range");
            }
        };
        for (std::size_t block = 0; block < expected_blocks; ++block) {
            validate_record(
                static_cast<std::uint16_t>(result.residual_first[block]),
                block);
            if (result.residual_second[block] >= 0) {
                validate_record(
                    static_cast<std::uint16_t>(
                        result.residual_second[block]),
                    block);
            }
        }
        const auto padding = cursor.bytes(
            padding_nbytes,
            "NEPQ-A residual padding");
        if (std::any_of(
                padding.begin(),
                padding.end(),
                [](std::uint8_t value) { return value != 0; })) {
            throw std::runtime_error(
                "NEPQ-A residual padding must be zero");
        }
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "invalid NEPQ cohort tail");
    }
    validate_canonical(result);
    return result;
}

VqTensorMetadata inspect_matrix_vq_header(
    std::string_view dtype,
    std::span<const std::uint8_t> blob,
    std::span<const std::uint8_t> runtime_payload) {
    if (!runtime_payload.empty()) {
        throw std::runtime_error(
            "unexpected matrix VQ runtime metadata");
    }
    BlobCursor cursor(blob);
    const auto magic = cursor.magic("matrix VQ magic");
    const auto header = read_matrix_header(
        cursor,
        magic);

    if (dtype == "NVQ1-L") {
        if (!magic_is(magic, "NQ1L") ||
            (header.profile != 1 &&
             header.profile != 2) ||
            header.bits == 0 ||
            header.bits > 8 ||
            header.group_size == 0 ||
            header.group_size % 8 != 0) {
            throw std::runtime_error(
                "unsupported NVQ1-L header");
        }
    } else if (dtype == "NVQ1-S") {
        if (!magic_is(magic, "NQ1S") ||
            header.profile != 1 ||
            header.bits != 4 ||
            header.group_size != 24 ||
            header.input_size % 8 != 0) {
            throw std::runtime_error(
                "unsupported NVQ1-S header");
        }
    } else if (dtype == "NPQ0-S" ||
               dtype == "NPQ0-L") {
        const bool short_profile =
            dtype == "NPQ0-S";
        if (!magic_is(
                magic,
                short_profile ? "NPQS" : "NPQL") ||
            header.profile !=
                static_cast<std::uint8_t>(
                    short_profile ? 2 : 1) ||
            header.bits !=
                (short_profile ? 2 : 3) ||
            header.group_size != 24 ||
            header.input_size % 8 != 0) {
            throw std::runtime_error(
                "unsupported NPQ header");
        }
    } else {
        constexpr std::uint8_t kIndexParity = 0x80;
        constexpr std::uint8_t kCustomCodebook = 0x40;
        constexpr std::uint8_t kJsc = 0x20;
        if (!magic_is(magic, "NVQ1") &&
            !magic_is(magic, "NIQ1")) {
            throw std::runtime_error(
                "invalid NVQ matrix magic");
        }
        const bool index_parity =
            (header.profile & kIndexParity) != 0;
        const bool custom =
            (header.profile & kCustomCodebook) != 0;
        const bool jsc =
            (header.profile & kJsc) != 0;
        const int id = header.profile
            & ~(kIndexParity | kCustomCodebook | kJsc);
        const auto profile = nvq_profile(id);
        const auto expected =
            expected_nvq_dtype(profile, jsc);
        if (dtype != expected ||
            (index_parity && profile.id != 1) ||
            header.bits == 0 ||
            header.bits > 8 ||
            header.group_size == 0 ||
            header.group_size % 8 != 0 ||
            (
                jsc &&
                (
                    custom ||
                    index_parity ||
                    header.bits != 4 ||
                    header.group_size != 24
                )
            )) {
            throw std::runtime_error(
                "unsupported NVQ matrix header");
        }
    }

    const int output_size =
        checked_int(header.output_size, "output size");
    return {
        std::string(dtype),
        {output_size},
        checked_int(
            static_cast<std::uint64_t>(
                header.input_size),
            "input size"),
        output_size,
        0,
        0,
    };
}

VqTensorMetadata inspect_nepq_header(
    std::string_view dtype,
    std::span<const std::uint8_t> blob,
    std::span<const std::uint8_t> runtime_payload) {
    BlobCursor cursor(blob);
    if (!magic_is(cursor.magic("NEPQ magic"), "NEP1")) {
        throw std::runtime_error(
            "invalid NEPQ cohort magic");
    }
    const auto version =
        cursor.scalar<std::uint8_t>("NEPQ version");
    const auto profile_id =
        cursor.scalar<std::uint8_t>("NEPQ profile");
    const auto groups_per_super =
        cursor.scalar<std::uint8_t>(
            "NEPQ groups per super-group");
    const auto flags =
        cursor.scalar<std::uint8_t>("NEPQ flags");
    const auto n_experts_raw =
        cursor.scalar<std::uint32_t>("NEPQ expert count");
    const auto out_per_expert_raw =
        cursor.scalar<std::uint32_t>(
            "NEPQ output rows per expert");
    const auto input_size_raw =
        cursor.scalar<std::uint32_t>("NEPQ input size");
    const auto table_banks_raw =
        cursor.scalar<std::uint32_t>("NEPQ table-bank count");
    const auto rotation_block_raw =
        cursor.scalar<std::uint32_t>("NEPQ rotation block");
    const auto rotation_seed =
        cursor.scalar<std::uint64_t>("NEPQ rotation seed");
    const auto profile = nepq_profile(profile_id);
    if (dtype != profile.label ||
        version != 1 ||
        groups_per_super != 4 ||
        (flags & ~std::uint8_t{1}) != 0 ||
        ((flags & 1u) != 0) !=
            (rotation_block_raw != 0)) {
        throw std::runtime_error(
            "unsupported NEPQ cohort header");
    }
    const int n_experts =
        checked_int(n_experts_raw, "NEPQ expert count");
    const int out_per_expert = checked_int(
        out_per_expert_raw,
        "NEPQ output rows per expert");
    const int input_size =
        checked_int(input_size_raw, "NEPQ input size");
    const int table_banks =
        checked_int(
            table_banks_raw,
            "NEPQ table-bank count");
    const int rotation_block = rotation_block_raw == 0
        ? 0
        : checked_int(
              rotation_block_raw,
              "NEPQ rotation block");
    if (input_size % 8 != 0 ||
        table_banks > 256 ||
        (
            profile.residual_block_vectors != 0
            && rotation_block == 0
        ) ||
        (
            rotation_block != 0 &&
            (
                (rotation_block
                    & (rotation_block - 1)) != 0 ||
                input_size % rotation_block != 0 ||
                rotation_block > 8192
            )
        ) ||
        (rotation_block == 0 && rotation_seed != 0)) {
        throw std::runtime_error(
            "invalid NEPQ cohort dimensions");
    }
    (void)parse_rotation_payload(
        runtime_payload,
        input_size,
        rotation_block,
        rotation_seed);
    const auto output_count = checked_product(
        static_cast<std::size_t>(n_experts),
        static_cast<std::size_t>(out_per_expert),
        "NEPQ output size");
    return {
        profile.label,
        {n_experts, out_per_expert},
        input_size,
        checked_int(
            output_count,
            "NEPQ flattened output size"),
        rotation_block,
        rotation_seed,
    };
}

CanonicalVq parse_vq(
    std::string_view dtype,
    std::span<const std::uint8_t> blob,
    std::span<const std::uint8_t> runtime_payload) {
    if (dtype == "NVQ1-L") {
        if (!runtime_payload.empty()) {
            throw std::runtime_error(
                "unexpected NVQ1-L runtime metadata");
        }
        return parse_nvq1_l(dtype, blob);
    }
    if (dtype == "NVQ1-S") {
        if (!runtime_payload.empty()) {
            throw std::runtime_error(
                "unexpected NVQ1-S runtime metadata");
        }
        return parse_nvq1_s(dtype, blob);
    }
    if (dtype == "NPQ0-S" || dtype == "NPQ0-L") {
        if (!runtime_payload.empty()) {
            throw std::runtime_error(
                "unexpected NPQ runtime metadata");
        }
        return parse_npq(dtype, blob, dtype == "NPQ0-S");
    }
    if (dtype == "NEPQ0-S" || dtype == "NEPQ0-L" ||
        dtype == "NEPQ1-S" || dtype == "NEPQ1-L" ||
        dtype == "NEPQ0-A" || dtype == "NEPQ1-A") {
        return parse_nepq(
            dtype,
            blob,
            runtime_payload);
    }
    if (dtype == "NVQ2" || dtype == "NVQ3" ||
        dtype == "NVQ2J" || dtype == "NVQ2J-L" ||
        dtype == "NVQ2J-XL" || dtype == "NVQ3J" ||
        dtype == "NVQ3J-512" || dtype == "NVQ3J-L") {
        if (!runtime_payload.empty()) {
            throw std::runtime_error(
                "unexpected NVQ runtime metadata");
        }
        return parse_nvq(dtype, blob);
    }
    throw std::runtime_error(
        "unsupported native Metal VQ dtype: "
        + std::string(dtype));
}

template <typename T, typename Allocator>
array make_array(
    const std::vector<T, Allocator>& values,
    Shape shape) {
    return array(values.begin(), std::move(shape));
}

mlx::core::fast::CustomKernelFunction make_vq_matmul_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_vq_packed_matmul",
        {
            "indices_packed",
            "state_packed",
            "aux_packed",
            "anchors",
            "codebooks",
            "scale_lut",
            "state_to_codebank",
            "bank_ids",
            "parameters",
            "x",
        },
        {"y"},
        kMatmulSource,
        kBitstreamHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
vq_matmul_kernel() {
    static const auto kernel = make_vq_matmul_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_vq_gemv_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_vq_packed_gemv",
        {
            "indices_packed",
            "state_packed",
            "aux_packed",
            "anchors",
            "codebooks",
            "scale_lut",
            "state_to_codebank",
            "bank_ids",
            "parameters",
            "x",
        },
        {"y"},
        kGemvSource,
        kBitstreamHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
vq_gemv_kernel() {
    static const auto kernel = make_vq_gemv_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_vq_mmq_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_vq_packed_mmq",
        {
            "indices_packed",
            "state_packed",
            "aux_packed",
            "anchors",
            "codebooks",
            "scale_lut",
            "state_to_codebank",
            "bank_ids",
            "parameters",
            "x",
        },
        {"y"},
        kMmqSource,
        kBitstreamHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
vq_mmq_kernel() {
    static const auto kernel = make_vq_mmq_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction
make_vq_dequantize_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_vq_dequantize",
        {
            "indices_packed",
            "state_packed",
            "aux_packed",
            "anchors",
            "codebooks",
            "scale_lut",
            "state_to_codebank",
            "bank_ids",
            "parameters",
        },
        {"y"},
        kDequantizeSource,
        kBitstreamHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
vq_dequantize_kernel() {
    static const auto kernel = make_vq_dequantize_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction
make_vq_embedding_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_vq_embedding",
        {
            "indices_packed",
            "state_packed",
            "aux_packed",
            "anchors",
            "codebooks",
            "scale_lut",
            "state_to_codebank",
            "bank_ids",
            "parameters",
            "token_ids",
        },
        {"y"},
        kEmbeddingSource,
        kBitstreamHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
vq_embedding_kernel() {
    static const auto kernel = make_vq_embedding_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction
make_hadamard_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_vq_signed_hadamard",
        {"x", "signs"},
        {"y"},
        kHadamardSource,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
hadamard_kernel() {
    static const auto kernel = make_hadamard_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction
make_residual_matmul_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nepq_sparse_residual_matmul",
        {
            "base",
            "x",
            "residual_codebook",
            "residual_first",
            "residual_second",
        },
        {"y"},
        kResidualMatmulSource,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
residual_matmul_kernel() {
    static const auto kernel = make_residual_matmul_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction
make_residual_dequantize_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nepq_sparse_residual_dequantize",
        {
            "base",
            "residual_codebook",
            "residual_first",
            "residual_second",
        },
        {"y"},
        kResidualDequantizeSource,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
residual_dequantize_kernel() {
    static const auto kernel = make_residual_dequantize_kernel();
    return kernel;
}

std::vector<std::pair<
    std::string,
    mlx::core::fast::TemplateArg>>
common_templates(
    Dtype dtype,
    int input_size,
    int output_size,
    int group_size,
    int groups,
    int vector_size,
    int vectors,
    int index_bits,
    int state_bits,
    int states,
    int entries,
    int code_banks,
    int aux_mode,
    int code_bank_mode,
    int execution_layout,
    int table_banks,
    int groups_per_supergroup,
    int supergroups) {
    return {
        {"T", dtype},
        {"OUT", output_size},
        {"K", input_size},
        {"GS", group_size},
        {"NG", groups},
        {"VECTOR_SIZE", vector_size},
        {"NVEC", vectors},
        {"INDEX_BITS", index_bits},
        {"STATE_BITS", state_bits},
        {"STATES", states},
        {"ENTRIES", entries},
        {"CODE_BANKS", code_banks},
        {"AUX_MODE", aux_mode},
        {"CODE_BANK_MODE", code_bank_mode},
        {"EXECUTION_LAYOUT", execution_layout},
        {"HAS_TABLE_BANKS", static_cast<int>(table_banks > 1)},
        {"GROUPS_PER_SUPER", groups_per_supergroup},
        {"NSUPER", supergroups},
        {"NSIGN", (input_size + 7) / 8},
    };
}

} // namespace

VqTensorMetadata inspect_vq_blob(
    std::string_view dtype,
    std::span<const std::uint8_t> blob,
    std::span<const std::uint8_t> runtime_payload) {
    if (!is_vq_dtype(dtype)) {
        throw std::runtime_error(
            "unsupported native Metal VQ dtype: "
            + std::string(dtype));
    }
    if (dtype == "NEPQ0-S" || dtype == "NEPQ0-L" ||
        dtype == "NEPQ1-S" || dtype == "NEPQ1-L" ||
        dtype == "NEPQ0-A" || dtype == "NEPQ1-A") {
        return inspect_nepq_header(
            dtype,
            blob,
            runtime_payload);
    }
    return inspect_matrix_vq_header(
        dtype,
        blob,
        runtime_payload);
}

bool is_vq_dtype(std::string_view dtype) noexcept {
    constexpr std::array<std::string_view, 18> dtypes{
        "NVQ2",
        "NVQ2J",
        "NVQ2J-L",
        "NVQ2J-XL",
        "NVQ3",
        "NVQ3J",
        "NVQ3J-512",
        "NVQ3J-L",
        "NVQ1-L",
        "NVQ1-S",
        "NPQ0-L",
        "NPQ0-S",
        "NEPQ0-L",
        "NEPQ0-S",
        "NEPQ1-L",
        "NEPQ1-S",
        "NEPQ0-A",
        "NEPQ1-A",
    };
    return std::find(
        dtypes.begin(),
        dtypes.end(),
        dtype) != dtypes.end();
}

MlxVqWeight::MlxVqWeight(
    array indices_packed,
    array state_packed,
    array aux_packed,
    array anchors,
    array codebooks,
    array scale_lut,
    array state_to_codebank,
    array bank_ids,
    array rotation_signs,
    array parameters,
    array residual_codebook,
    array residual_first,
    array residual_second,
    std::string format_label,
    std::vector<int> output_shape,
    int input_size,
    int output_size,
    int group_size,
    int groups,
    int vector_size,
    int vectors,
    int index_bits,
    int state_bits,
    int states,
    int entries,
    int code_banks,
    int aux_mode,
    int code_bank_mode,
    int execution_layout,
    int table_banks,
    int groups_per_supergroup,
    int supergroups,
    int rotation_block,
    std::uint64_t rotation_seed,
    int residual_position_bits,
    int residual_block_vectors,
    int residual_blocks_per_row)
    : indices_packed_(std::move(indices_packed)),
      state_packed_(std::move(state_packed)),
      aux_packed_(std::move(aux_packed)),
      anchors_(std::move(anchors)),
      codebooks_(std::move(codebooks)),
      scale_lut_(std::move(scale_lut)),
      state_to_codebank_(std::move(state_to_codebank)),
      bank_ids_(std::move(bank_ids)),
      rotation_signs_(std::move(rotation_signs)),
      parameters_(std::move(parameters)),
      residual_codebook_(std::move(residual_codebook)),
      residual_first_(std::move(residual_first)),
      residual_second_(std::move(residual_second)),
      format_label_(std::move(format_label)),
      output_shape_(std::move(output_shape)),
      input_size_(input_size),
      output_size_(output_size),
      group_size_(group_size),
      groups_(groups),
      vector_size_(vector_size),
      vectors_(vectors),
      index_bits_(index_bits),
      state_bits_(state_bits),
      states_(states),
      entries_(entries),
      code_banks_(code_banks),
      aux_mode_(aux_mode),
      code_bank_mode_(code_bank_mode),
      execution_layout_(execution_layout),
      table_banks_(table_banks),
      groups_per_supergroup_(groups_per_supergroup),
      supergroups_(supergroups),
      rotation_block_(rotation_block),
      rotation_seed_(rotation_seed),
      residual_position_bits_(residual_position_bits),
      residual_block_vectors_(residual_block_vectors),
      residual_blocks_per_row_(residual_blocks_per_row) {}

MlxVqWeight MlxVqWeight::from_blob(
    std::string_view dtype,
    std::span<const std::uint8_t> blob,
    std::span<const std::uint8_t> runtime_payload) {
    auto parsed = parse_vq(
        dtype,
        blob,
        runtime_payload);
    auto rotations = parsed.rotation_signs;
    if (rotations.empty()) {
        rotations.push_back(1);
    }
    auto residual_codebook = parsed.residual_codebook;
    auto residual_first = parsed.residual_first;
    auto residual_second = parsed.residual_second;
    if (residual_codebook.empty()) {
        residual_codebook.push_back(0.0f);
    }
    if (residual_first.empty()) {
        residual_first.push_back(-1);
    }
    if (residual_second.empty()) {
        residual_second.push_back(-1);
    }
    return MlxVqWeight(
        make_array(
            parsed.indices,
            Shape{
                static_cast<int>(parsed.indices.size()),
            }),
        make_array(
            parsed.states_packed,
            Shape{
                static_cast<int>(
                    parsed.states_packed.size()),
            }),
        make_array(
            parsed.auxiliary,
            Shape{
                static_cast<int>(
                    parsed.auxiliary.size()),
            }),
        make_array(
            parsed.anchors,
            Shape{parsed.output_size}),
        make_array(
            parsed.codebooks,
            Shape{
                parsed.table_banks,
                parsed.code_banks,
                parsed.entries,
                parsed.vector_size,
            }),
        make_array(
            parsed.scales,
            Shape{
                parsed.table_banks,
                parsed.states,
            }),
        make_array(
            parsed.state_to_codebank,
            Shape{parsed.states}),
        make_array(
            parsed.bank_ids,
            Shape{
                parsed.output_size,
                parsed.supergroups,
            }),
        make_array(
            rotations,
            Shape{
                static_cast<int>(rotations.size()),
            }),
        make_array(
            std::vector<float>{parsed.parameter},
            Shape{1}),
        make_array(
            residual_codebook,
            Shape{static_cast<int>(residual_codebook.size())}),
        make_array(
            residual_first,
            Shape{static_cast<int>(residual_first.size())}),
        make_array(
            residual_second,
            Shape{static_cast<int>(residual_second.size())}),
        std::move(parsed.label),
        std::move(parsed.output_shape),
        parsed.input_size,
        parsed.output_size,
        parsed.group_size,
        parsed.groups,
        parsed.vector_size,
        parsed.vectors,
        parsed.index_bits,
        parsed.state_bits,
        parsed.states,
        parsed.entries,
        parsed.code_banks,
        parsed.aux_mode,
        parsed.code_bank_mode,
        parsed.execution_layout,
        parsed.table_banks,
        parsed.groups_per_supergroup,
        parsed.supergroups,
        parsed.rotation_block,
        parsed.rotation_seed,
        parsed.residual_position_bits,
        parsed.residual_block_vectors,
        parsed.residual_blocks_per_row);
}

std::size_t MlxVqWeight::packed_nbytes() const noexcept {
    return indices_packed_.nbytes()
        + state_packed_.nbytes()
        + aux_packed_.nbytes()
        + anchors_.nbytes()
        + codebooks_.nbytes()
        + scale_lut_.nbytes()
        + state_to_codebank_.nbytes()
        + bank_ids_.nbytes()
        + rotation_signs_.nbytes()
        + parameters_.nbytes()
        + residual_codebook_.nbytes()
        + residual_first_.nbytes()
        + residual_second_.nbytes();
}

array MlxVqWeight::dequantize(Dtype dtype) const {
    if (dtype != mlx::core::float16 &&
        dtype != mlx::core::float32) {
        throw std::runtime_error(
            "VQ dequantization output must be float16 or float32");
    }
    const auto size = checked_product(
        static_cast<std::size_t>(output_size_),
        static_cast<std::size_t>(input_size_),
        "dequantized matrix size");
    if (size >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "VQ dequantization grid exceeds MLX limits");
    }
    auto templates = common_templates(
        dtype,
        input_size_,
        output_size_,
        group_size_,
        groups_,
        vector_size_,
        vectors_,
        index_bits_,
        state_bits_,
        states_,
        entries_,
        code_banks_,
        aux_mode_,
        code_bank_mode_,
        execution_layout_,
        table_banks_,
        groups_per_supergroup_,
        supergroups_);
    auto outputs = vq_dequantize_kernel()(
        {
            indices_packed_,
            state_packed_,
            aux_packed_,
            anchors_,
            codebooks_,
            scale_lut_,
            state_to_codebank_,
            bank_ids_,
            parameters_,
        },
        {Shape{output_size_, input_size_}},
        {dtype},
        {static_cast<int>(size), 1, 1},
        {
            std::min(
                256,
                std::max(1, static_cast<int>(size))),
            1,
            1,
        },
        std::move(templates),
        std::nullopt,
        false,
        {});
    auto result = std::move(outputs.front());
    if (residual_position_bits_ == 0) {
        return result;
    }
    auto residual_outputs = residual_dequantize_kernel()(
        {
            result,
            residual_codebook_,
            residual_first_,
            residual_second_,
        },
        {Shape{output_size_, input_size_}},
        {dtype},
        {static_cast<int>(size), 1, 1},
        {
            std::min(256, static_cast<int>(size)),
            1,
            1,
        },
        {
            {"T", dtype},
            {"OUT", output_size_},
            {"K", input_size_},
            {"RESIDUAL_BLOCKS", residual_blocks_per_row_},
            {"POSITION_BITS", residual_position_bits_},
            {"BLOCK_VECTORS", residual_block_vectors_},
        },
        std::nullopt,
        false,
        {});
    return std::move(residual_outputs.front());
}

array MlxVqWeight::embedding(
    const array& token_ids,
    Dtype dtype) const {
    if (rotation_block_ != 0 ||
        output_shape_.size() != 1) {
        throw std::runtime_error(
            "VQ embedding requires a non-rotated matrix weight");
    }
    if (dtype != mlx::core::float16 &&
        dtype != mlx::core::float32) {
        throw std::runtime_error(
            "VQ embedding output must be float16 or float32");
    }
    auto ids = token_ids;
    if (ids.dtype() != mlx::core::int32 &&
        ids.dtype() != mlx::core::uint32) {
        ids = mlx::core::astype(
            ids,
            mlx::core::int32);
    }
    const auto count = ids.size();
    Shape output_shape = ids.shape();
    output_shape.push_back(input_size_);
    if (count == 0) {
        return mlx::core::zeros(
            std::move(output_shape),
            dtype);
    }
    const auto output_elements = checked_product(
        count,
        static_cast<std::size_t>(input_size_),
        "embedding output size");
    if (count >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max()) ||
        output_elements >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "VQ embedding grid exceeds MLX limits");
    }
    ids = mlx::core::contiguous(
        mlx::core::reshape(
            ids,
            Shape{static_cast<int>(count)}));
    auto templates = common_templates(
        dtype,
        input_size_,
        output_size_,
        group_size_,
        groups_,
        vector_size_,
        vectors_,
        index_bits_,
        state_bits_,
        states_,
        entries_,
        code_banks_,
        aux_mode_,
        code_bank_mode_,
        execution_layout_,
        table_banks_,
        groups_per_supergroup_,
        supergroups_);
    templates.emplace_back(
        "COUNT",
        static_cast<int>(count));
    auto outputs = vq_embedding_kernel()(
        {
            indices_packed_,
            state_packed_,
            aux_packed_,
            anchors_,
            codebooks_,
            scale_lut_,
            state_to_codebank_,
            bank_ids_,
            parameters_,
            ids,
        },
        {
            Shape{
                static_cast<int>(count),
                input_size_,
            },
        },
        {dtype},
        {
            static_cast<int>(output_elements),
            1,
            1,
        },
        {
            std::min(
                256,
                static_cast<int>(output_elements)),
            1,
            1,
        },
        std::move(templates),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        std::move(outputs.front()),
        std::move(output_shape));
}

array MlxVqWeight::prepare_input(
    const array& input,
    Shape& prefix,
    int& rows) const {
    if (input.ndim() == 0 ||
        input.shape(-1) != input_size_) {
        throw std::runtime_error(
            "VQ input width does not match packed weight");
    }
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(
            source,
            mlx::core::float16);
    }
    prefix.clear();
    std::size_t row_count = 1;
    for (std::size_t dimension = 0;
         dimension + 1 < source.ndim();
         ++dimension) {
        const auto extent =
            source.shape(static_cast<int>(dimension));
        prefix.push_back(extent);
        row_count = checked_product(
            row_count,
            static_cast<std::size_t>(extent),
            "input row count");
    }
    if (row_count >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "VQ input row count exceeds MLX limits");
    }
    rows = static_cast<int>(row_count);
    source = mlx::core::contiguous(
        mlx::core::reshape(
            source,
            Shape{rows, input_size_}));
    if (rotation_block_ == 0 || rows == 0) {
        return source;
    }
    auto outputs = hadamard_kernel()(
        {source, rotation_signs_},
        {Shape{rows, input_size_}},
        {source.dtype()},
        {rows * 256, 1, 1},
        {256, 1, 1},
        {
            {"T", source.dtype()},
            {"M", rows},
            {"K", input_size_},
            {"BLOCK", rotation_block_},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array MlxVqWeight::reshape_output(
    array value,
    const Shape& prefix) const {
    Shape shape = prefix;
    shape.insert(
        shape.end(),
        output_shape_.begin(),
        output_shape_.end());
    return mlx::core::reshape(
        std::move(value),
        std::move(shape));
}

array MlxVqWeight::packed_matmul(
    const array& source,
    const Shape& prefix,
    int rows,
    int tile_rows) const {
    if (rows == 0) {
        Shape shape = prefix;
        shape.insert(
            shape.end(),
            output_shape_.begin(),
            output_shape_.end());
        return mlx::core::zeros(
            std::move(shape),
            source.dtype());
    }
    if (tile_rows <= 0 || tile_rows > 16) {
        throw std::runtime_error(
            "invalid VQ packed row tile");
    }
    const bool fast_gemv =
        rows == 1 && tile_rows == 1
        && execution_layout_ == kExecutionStreams;
    const bool wide_mmq =
        rows >= 2 && rows <= 16
        && tile_rows == rows
        && execution_layout_ == kExecutionStreams;
    int effective_tile_rows = tile_rows;
    std::tuple<int, int, int> grid;
    std::tuple<int, int, int> threadgroup;
    if (fast_gemv) {
        const auto output_tiles =
            (
                static_cast<std::size_t>(output_size_)
                + 7
            ) / 8;
        const auto grid_x = checked_product(
            output_tiles,
            std::size_t{64},
            "GEMV grid");
        if (grid_x >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "VQ GEMV grid exceeds MLX limits");
        }
        grid = {
            static_cast<int>(grid_x),
            1,
            1,
        };
        threadgroup = {64, 1, 1};
    } else if (wide_mmq) {
        const int mmq_simd_groups = index_bits_ <= 8 ? 2 : 4;
        constexpr int mmq_rows_per_simd = 4;
        const int row_tiles = (rows + 4) / 5;
        effective_tile_rows =
            (rows + row_tiles - 1) / row_tiles;
        const auto grid_x = checked_product(
            static_cast<std::size_t>(row_tiles),
            static_cast<std::size_t>(mmq_simd_groups * 32),
            "MMQ row grid");
        const auto grid_y =
            (
                static_cast<std::size_t>(output_size_)
                + mmq_simd_groups * mmq_rows_per_simd - 1
            ) / (mmq_simd_groups * mmq_rows_per_simd);
        if (grid_x >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()) ||
            grid_y >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "VQ MMQ grid exceeds MLX limits");
        }
        grid = {
            static_cast<int>(grid_x),
            static_cast<int>(grid_y),
            1,
        };
        threadgroup = {mmq_simd_groups * 32, 1, 1};
    } else {
        const auto row_tiles =
            (rows + tile_rows - 1) / tile_rows;
        const auto workgroups = checked_product(
            static_cast<std::size_t>(row_tiles),
            static_cast<std::size_t>(output_size_),
            "packed matmul workgroup count");
        const auto grid_size = checked_product(
            workgroups,
            std::size_t{32},
            "packed matmul grid");
        if (grid_size >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "VQ packed matmul grid exceeds MLX limits");
        }
        grid = {
            static_cast<int>(grid_size),
            1,
            1,
        };
        threadgroup = {32, 1, 1};
    }
    auto templates = common_templates(
        source.dtype(),
        input_size_,
        output_size_,
        group_size_,
        groups_,
        vector_size_,
        vectors_,
        index_bits_,
        state_bits_,
        states_,
        entries_,
        code_banks_,
        aux_mode_,
        code_bank_mode_,
        execution_layout_,
        table_banks_,
        groups_per_supergroup_,
        supergroups_);
    templates.emplace_back("M", rows);
    templates.emplace_back(
        "TILE_M",
        effective_tile_rows);
    if (wide_mmq) {
        templates.emplace_back(
            "SIMD_GROUPS",
            index_bits_ <= 8 ? 2 : 4);
    }
    const auto& kernel = fast_gemv
        ? vq_gemv_kernel()
        : (
            wide_mmq
            ? vq_mmq_kernel()
            : vq_matmul_kernel()
        );
    auto outputs = kernel(
        {
            indices_packed_,
            state_packed_,
            aux_packed_,
            anchors_,
            codebooks_,
            scale_lut_,
            state_to_codebank_,
            bank_ids_,
            parameters_,
            source,
        },
        {Shape{rows, output_size_}},
        {source.dtype()},
        grid,
        threadgroup,
        std::move(templates),
        std::nullopt,
        false,
        {});
    auto result = std::move(outputs.front());
    if (residual_position_bits_ != 0) {
        const auto total = checked_product(
            static_cast<std::size_t>(rows),
            static_cast<std::size_t>(output_size_),
            "NEPQ-A residual output size");
        const auto workgroups = (total + 3) / 4;
        const auto residual_grid = checked_product(
            workgroups,
            std::size_t{128},
            "NEPQ-A residual grid");
        if (residual_grid > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "NEPQ-A residual grid exceeds MLX limits");
        }
        auto residual_outputs = residual_matmul_kernel()(
            {
                result,
                source,
                residual_codebook_,
                residual_first_,
                residual_second_,
            },
            {Shape{rows, output_size_}},
            {source.dtype()},
            {static_cast<int>(residual_grid), 1, 1},
            {128, 1, 1},
            {
                {"T", source.dtype()},
                {"M", rows},
                {"OUT", output_size_},
                {"K", input_size_},
                {"NVEC", vectors_},
                {"RESIDUAL_BLOCKS", residual_blocks_per_row_},
                {"POSITION_BITS", residual_position_bits_},
                {"BLOCK_VECTORS", residual_block_vectors_},
            },
            std::nullopt,
            false,
            {});
        result = std::move(residual_outputs.front());
    }
    return reshape_output(std::move(result), prefix);
}

array MlxVqWeight::gemv(const array& input) const {
    Shape prefix;
    int rows = 0;
    auto source = prepare_input(
        input,
        prefix,
        rows);
    if (rows != 1) {
        throw std::runtime_error(
            "VQ GEMV requires exactly one input row");
    }
    return packed_matmul(
        source,
        prefix,
        rows,
        1);
}

array MlxVqWeight::mmq(const array& input) const {
    Shape prefix;
    int rows = 0;
    auto source = prepare_input(
        input,
        prefix,
        rows);
    if (rows < 2 || rows > 16) {
        throw std::runtime_error(
            "VQ MMQ requires 2 to 16 input rows");
    }
    return packed_matmul(
        source,
        prefix,
        rows,
        rows);
}

array MlxVqWeight::gemm(const array& input) const {
    Shape prefix;
    int rows = 0;
    auto source = prepare_input(
        input,
        prefix,
        rows);
    return packed_matmul(
        source,
        prefix,
        rows,
        8);
}

array MlxVqWeight::matmul(const array& input) const {
    Shape prefix;
    int rows = 0;
    auto source = prepare_input(
        input,
        prefix,
        rows);
    if (rows == 0) {
        return packed_matmul(
            source,
            prefix,
            rows,
            1);
    }
    if (rows >= 64 &&
        source.dtype() == mlx::core::float16) {
        auto dense = dequantize(mlx::core::float16);
        auto result = mlx::core::matmul(
            source,
            mlx::core::transpose(dense));
        return reshape_output(
            std::move(result),
            prefix);
    }
    if (rows == 1) {
        return packed_matmul(
            source,
            prefix,
            rows,
            1);
    }
    if (rows <= 16) {
        return packed_matmul(
            source,
            prefix,
            rows,
            rows);
    }
    return packed_matmul(
        source,
        prefix,
        rows,
        8);
}

} // namespace mfq::metal
