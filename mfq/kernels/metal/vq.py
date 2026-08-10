"""Packed NVQ, NPQ, and NEPQ Metal matrix kernels.

The execution layout keeps per-weight indices, group states, signs, and
delta selectors in continuous little-endian bit streams. Only the small
shared int8 decode tables are expanded into a read-optimized Metal LUT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats.nepq import (
    NEPQ0_L,
    NEPQ0_S,
    NEPQ1_L,
    NEPQ1_S,
    NepqTensor,
    nepq_base_spec,
    rotation_signs,
    validate_nepq,
)
from mfq.formats.npq0_l import Npq0LTensor, unpack_npq0_l_tables
from mfq.formats.npq0_s import Npq0STensor, unpack_npq0_s_tables
from mfq.formats.nvq import (
    NvqJscTensor,
    NvqTensor,
    codebook_for,
)
from mfq.formats.nvq1_l import IQ1S_TERNARY_2048, Nvq1LTensor, unpack_ternary_codebook
from mfq.formats.nvq1_s import (
    NVQ1_S_SYNTHETIC_BANKS,
    Nvq1STensor,
    unpack_nvq1_s_banked_codebook,
)

VqTensor: TypeAlias = (
    NvqTensor | NvqJscTensor | Nvq1LTensor | Nvq1STensor | Npq0LTensor | Npq0STensor | NepqTensor
)

_AUX_NONE = 0
_AUX_SIGN_EVEN = 1
_AUX_SIGN_INDEX_PARITY = 2
_AUX_DELTA = 3

_CODE_BANK_FIXED = 0
_CODE_BANK_STATE = 1
_CODE_BANK_AUX = 2

_BITSTREAM_HEADER = r"""
#include <metal_simdgroup_matrix>

inline uint mfq_read_bits(
    device const uchar* stream,
    uint value_index,
    uint bits
) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3) * bits + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(stream[byte_index])
        | (uint(stream[byte_index + 1u]) << 8)
        | (uint(stream[byte_index + 2u]) << 16);
    return (packed >> shift) & ((1u << bits) - 1u);
}

inline uint mfq_read_bits(
    constant const uchar* stream,
    uint value_index,
    uint bits
) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3) * bits + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(stream[byte_index])
        | (uint(stream[byte_index + 1u]) << 8)
        | (uint(stream[byte_index + 2u]) << 16);
    return (packed >> shift) & ((1u << bits) - 1u);
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
    uint has_table_banks,
    uint groups_per_super,
    uint supergroups,
    uint signs
) {
    uint group = column / groupsize;
    uint vector = column / vector_size;
    uint component = column - vector * vector_size;
    uint state_index = output * groups + group;
    uint state = mfq_read_bits(state_packed, state_index, state_bits);
    uint table_bank = 0u;
    if (has_table_banks != 0u) {
        table_bank = uint(bank_ids[
            output * supergroups + group / groups_per_super
        ]);
    }
    uint index = mfq_read_bits(
        indices_packed, output * vectors + vector, index_bits);
    uint aux_value = 0u;
    if (aux_mode == 1u || aux_mode == 2u) {
        aux_value = mfq_read_bits(
            aux_packed, output * signs + column / 8u, 7u);
    } else if (aux_mode == 3u) {
        aux_value = mfq_read_bits(aux_packed, state_index, 1u);
    }
    uint code_bank = 0u;
    if (code_bank_mode == 1u) {
        code_bank = uint(state_to_codebank[state]);
    } else if (code_bank_mode == 2u) {
        code_bank = aux_value;
    }
    uint code_offset = (
        (
            (table_bank * code_banks + code_bank) * entries + index
        )
        * vector_size + component
    );
    float code = float(codebooks[code_offset]);
    if (aux_mode == 1u || aux_mode == 2u) {
        uint sign_position = column & 7u;
        uint negative = sign_position < 7u
            ? ((aux_value >> sign_position) & 1u)
            : (popcount(aux_value) & 1u);
        if (aux_mode == 2u && sign_position == 7u) {
            negative ^= (index >> 7u) & 1u;
        }
        code = negative != 0u ? -code : code;
    } else if (aux_mode == 3u) {
        code += aux_value != 0u ? -parameters[0] : parameters[0];
    }
    return anchors[output]
        * scale_lut[table_bank * states + state]
        * code;
}
"""

_VQ_GEMV_FAST_SOURCE = r"""
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint VECTORS_PER_GROUP =
        (uint(GS) + uint(VECTOR_SIZE) - 1u) / uint(VECTOR_SIZE);

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;

    float accumulators[ROWS_PER_SIMD] = {0.0f};

    // A lane owns complete gs24 groups.  State/bank metadata is decoded once,
    // activation vectors are shared by four output rows, and codebook indices
    // are consumed a vector at a time as in the CUDA vec8 path.
    for (uint group = lane; group < uint(NG); group += 32u) {
        uint outputs[ROWS_PER_SIMD];
        uint table_banks[ROWS_PER_SIMD];
        uint code_banks[ROWS_PER_SIMD];
        uint delta_values[ROWS_PER_SIMD];
        float weight_scales[ROWS_PER_SIMD];

        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            uint output = min(output_base + row, uint(OUT) - 1u);
            uint state_index = output * uint(NG) + group;
            uint state = mfq_read_bits(
                state_packed, state_index, uint(STATE_BITS));
            uint table_bank = 0u;
            if (HAS_TABLE_BANKS != 0) {
                table_bank = uint(bank_ids[
                    output * uint(NSUPER)
                    + group / uint(GROUPS_PER_SUPER)
                ]);
            }
            uint delta_value = AUX_MODE == 3
                ? mfq_read_bits(aux_packed, state_index, 1u)
                : 0u;
            uint code_bank = 0u;
            if (CODE_BANK_MODE == 1) {
                code_bank = uint(state_to_codebank[state]);
            } else if (CODE_BANK_MODE == 2) {
                code_bank = delta_value;
            }
            outputs[row] = output;
            table_banks[row] = table_bank;
            code_banks[row] = code_bank;
            delta_values[row] = delta_value;
            weight_scales[row] = anchors[output]
                * scale_lut[table_bank * uint(STATES) + state];
        }

        for (
            uint local_vector = 0u;
            local_vector < VECTORS_PER_GROUP;
            ++local_vector
        ) {
            uint column_base =
                group * uint(GS) + local_vector * uint(VECTOR_SIZE);
            if (column_base >= uint(K)) {
                break;
            }
            float activations[VECTOR_SIZE];
            for (
                uint component = 0u;
                component < uint(VECTOR_SIZE);
                ++component
            ) {
                uint column = column_base + component;
                activations[component] =
                    column < uint(K) ? float(x[column]) : 0.0f;
            }

            uint vector = column_base / uint(VECTOR_SIZE);
            for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                uint output = outputs[row];
                uint index = mfq_read_bits(
                    indices_packed,
                    output * uint(NVEC) + vector,
                    uint(INDEX_BITS)
                );
                uint sign_value = 0u;
                if (AUX_MODE == 1 || AUX_MODE == 2) {
                    sign_value = mfq_read_bits(
                        aux_packed,
                        output * uint(NSIGN) + column_base / 8u,
                        7u
                    );
                }
                for (
                    uint component = 0u;
                    component < uint(VECTOR_SIZE);
                    ++component
                ) {
                    uint column = column_base + component;
                    if (column >= uint(K)) {
                        break;
                    }
                    uint code_offset = (
                        (
                            (
                                table_banks[row] * uint(CODE_BANKS)
                                + code_banks[row]
                            )
                            * uint(ENTRIES) + index
                        )
                        * uint(VECTOR_SIZE) + component
                    );
                    float code = float(codebooks[code_offset]);
                    if (AUX_MODE == 1 || AUX_MODE == 2) {
                        uint sign_position = column & 7u;
                        uint negative = sign_position < 7u
                            ? ((sign_value >> sign_position) & 1u)
                            : (popcount(sign_value) & 1u);
                        if (AUX_MODE == 2 && sign_position == 7u) {
                            negative ^= (index >> 7u) & 1u;
                        }
                        code = negative != 0u ? -code : code;
                    } else if (AUX_MODE == 3) {
                        code += delta_values[row] != 0u
                            ? -parameters[0] : parameters[0];
                    }
                    accumulators[row] +=
                        activations[component] * weight_scales[row] * code;
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
"""

_VQ_MMQ_WIDE_SOURCE = r"""
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD = 32u / K_LANES;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint VECTORS_PER_GROUP =
        (uint(GS) + uint(VECTOR_SIZE) - 1u) / uint(VECTOR_SIZE);

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_row = lane / K_LANES;
    uint output_index =
        threadgroup_position_in_grid.y * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD + simd_row;
    uint output = min(output_index, uint(OUT) - 1u);
    uint first_vector = threadgroup_position_in_grid.x * uint(TILE_M);

    float accumulators[TILE_M];
    for (uint vector = 0u; vector < uint(TILE_M); ++vector) {
        accumulators[vector] = 0.0f;
    }

    for (uint group = k_lane; group < uint(NG); group += K_LANES) {
        uint state_index = output * uint(NG) + group;
        uint state = mfq_read_bits(
            state_packed, state_index, uint(STATE_BITS));
        uint table_bank = 0u;
        if (HAS_TABLE_BANKS != 0) {
            table_bank = uint(bank_ids[
                output * uint(NSUPER) + group / uint(GROUPS_PER_SUPER)
            ]);
        }
        uint delta_value = AUX_MODE == 3
            ? mfq_read_bits(aux_packed, state_index, 1u)
            : 0u;
        uint code_bank = 0u;
        if (CODE_BANK_MODE == 1) {
            code_bank = uint(state_to_codebank[state]);
        } else if (CODE_BANK_MODE == 2) {
            code_bank = delta_value;
        }
        float weight_scale = anchors[output]
            * scale_lut[table_bank * uint(STATES) + state];

        for (
            uint local_vector = 0u;
            local_vector < VECTORS_PER_GROUP;
            ++local_vector
        ) {
            uint column_base =
                group * uint(GS) + local_vector * uint(VECTOR_SIZE);
            if (column_base >= uint(K)) {
                break;
            }
            uint vector = column_base / uint(VECTOR_SIZE);
            uint index = mfq_read_bits(
                indices_packed,
                output * uint(NVEC) + vector,
                uint(INDEX_BITS)
            );
            uint sign_value = 0u;
            if (AUX_MODE == 1 || AUX_MODE == 2) {
                sign_value = mfq_read_bits(
                    aux_packed,
                    output * uint(NSIGN) + column_base / 8u,
                    7u
                );
            }
            for (
                uint component = 0u;
                component < uint(VECTOR_SIZE);
                ++component
            ) {
                uint column = column_base + component;
                if (column >= uint(K)) {
                    break;
                }
                uint code_offset = (
                    (
                        (table_bank * uint(CODE_BANKS) + code_bank)
                        * uint(ENTRIES) + index
                    )
                    * uint(VECTOR_SIZE) + component
                );
                float code = float(codebooks[code_offset]);
                if (AUX_MODE == 1 || AUX_MODE == 2) {
                    uint sign_position = column & 7u;
                    uint negative = sign_position < 7u
                        ? ((sign_value >> sign_position) & 1u)
                        : (popcount(sign_value) & 1u);
                    if (AUX_MODE == 2 && sign_position == 7u) {
                        negative ^= (index >> 7u) & 1u;
                    }
                    code = negative != 0u ? -code : code;
                } else if (AUX_MODE == 3) {
                    code += delta_value != 0u
                        ? -parameters[0] : parameters[0];
                }
                float weight = weight_scale * code;
                for (uint input = 0u; input < uint(TILE_M); ++input) {
                    uint row = min(first_vector + input, uint(M) - 1u);
                    accumulators[input] +=
                        float(x[row * uint(K) + column]) * weight;
                }
            }
        }
    }

    for (uint input = 0u; input < uint(TILE_M); ++input) {
        accumulators[input] += simd_shuffle_down(accumulators[input], 4);
        accumulators[input] += simd_shuffle_down(accumulators[input], 2);
        accumulators[input] += simd_shuffle_down(accumulators[input], 1);
        uint row = first_vector + input;
        if (
            k_lane == 0u
            && output_index < uint(OUT)
            && row < uint(M)
        ) {
            y[row * uint(OUT) + output_index] = T(accumulators[input]);
        }
    }
"""

_VQ_SWIGLU_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint output = thread_position_in_grid.x >> 5;
    if (output >= uint(OUT)) {
        return;
    }

    float gate_acc[TILE_M];
    float up_acc[TILE_M];
    for (uint row = 0u; row < uint(TILE_M); ++row) {
        gate_acc[row] = 0.0f;
        up_acc[row] = 0.0f;
    }

    for (uint column = lane; column < uint(K); column += 32u) {
        float gate_weight = mfq_vq_decode_weight(
            gate_indices,
            gate_state,
            gate_aux,
            gate_anchors,
            gate_codebooks,
            gate_scales,
            gate_state_to_codebank,
            gate_banks,
            gate_parameters,
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
            uint(HAS_TABLE_BANKS),
            uint(GROUPS_PER_SUPER),
            uint(NSUPER),
            uint(NSIGN)
        );
        float up_weight = mfq_vq_decode_weight(
            up_indices,
            up_state,
            up_aux,
            up_anchors,
            up_codebooks,
            up_scales,
            up_state_to_codebank,
            up_banks,
            up_parameters,
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
            uint(HAS_TABLE_BANKS),
            uint(GROUPS_PER_SUPER),
            uint(NSUPER),
            uint(NSIGN)
        );
        for (uint row = 0u; row < uint(TILE_M); ++row) {
            float activation = float(x[row * uint(K) + column]);
            gate_acc[row] += activation * gate_weight;
            up_acc[row] += activation * up_weight;
        }
    }

    for (uint row = 0u; row < uint(TILE_M); ++row) {
        float gate_value = simd_sum(gate_acc[row]);
        float up_value = simd_sum(up_acc[row]);
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] =
                T(gate_value / (1.0f + exp(-gate_value)) * up_value);
        }
    }
"""

_VQ_MATMUL_SOURCE = r"""
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

    for (uint column = lane; column < uint(K); column += 32u) {
        uint group = column / uint(GS);
        uint vector = column / uint(VECTOR_SIZE);
        uint component = column - vector * uint(VECTOR_SIZE);
        uint state_index = output * uint(NG) + group;
        uint state = mfq_read_bits(state_packed, state_index, uint(STATE_BITS));
        uint table_bank = 0u;
        if (HAS_TABLE_BANKS != 0) {
            table_bank = uint(bank_ids[
                output * uint(NSUPER) + group / uint(GROUPS_PER_SUPER)
            ]);
        }

        uint index = mfq_read_bits(
            indices_packed,
            output * uint(NVEC) + vector,
            uint(INDEX_BITS)
        );
        uint aux_value = 0u;
        if (AUX_MODE == 1 || AUX_MODE == 2) {
            aux_value = mfq_read_bits(
                aux_packed,
                output * uint(NSIGN) + column / 8u,
                7u
            );
        } else if (AUX_MODE == 3) {
            aux_value = mfq_read_bits(aux_packed, state_index, 1u);
        }

        uint code_bank = 0u;
        if (CODE_BANK_MODE == 1) {
            code_bank = uint(state_to_codebank[state]);
        } else if (CODE_BANK_MODE == 2) {
            code_bank = aux_value;
        }
        uint code_offset = (
            (
                (table_bank * uint(CODE_BANKS) + code_bank)
                * uint(ENTRIES)
                + index
            )
            * uint(VECTOR_SIZE)
            + component
        );
        float code = float(codebooks[code_offset]);

        if (AUX_MODE == 1 || AUX_MODE == 2) {
            uint sign_position = column & 7u;
            uint negative = sign_position < 7u
                ? ((aux_value >> sign_position) & 1u)
                : (popcount(aux_value) & 1u);
            if (AUX_MODE == 2 && sign_position == 7u) {
                negative ^= (index >> 7u) & 1u;
            }
            code = negative != 0u ? -code : code;
        } else if (AUX_MODE == 3) {
            code += aux_value != 0u ? -parameters[0] : parameters[0];
        }

        float weight = anchors[output]
            * scale_lut[table_bank * uint(STATES) + state]
            * code;
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            uint row = first_row + local_row;
            if (row < uint(M)) {
                accumulators[local_row] += float(x[row * uint(K) + column]) * weight;
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
"""

_VQ_DEQUANT_SOURCE = r"""
    uint linear = thread_position_in_grid.x;
    if (linear >= uint(OUT) * uint(K)) {
        return;
    }
    uint output = linear / uint(K);
    uint column = linear - output * uint(K);
    uint group = column / uint(GS);
    uint vector = column / uint(VECTOR_SIZE);
    uint component = column - vector * uint(VECTOR_SIZE);
    uint state_index = output * uint(NG) + group;
    uint state = mfq_read_bits(
        state_packed, state_index, uint(STATE_BITS));
    uint table_bank = 0u;
    if (HAS_TABLE_BANKS != 0) {
        table_bank = uint(bank_ids[
            output * uint(NSUPER) + group / uint(GROUPS_PER_SUPER)
        ]);
    }

    uint index = mfq_read_bits(
        indices_packed,
        output * uint(NVEC) + vector,
        uint(INDEX_BITS)
    );
    uint aux_value = 0u;
    if (AUX_MODE == 1 || AUX_MODE == 2) {
        aux_value = mfq_read_bits(
            aux_packed,
            output * uint(NSIGN) + column / 8u,
            7u
        );
    } else if (AUX_MODE == 3) {
        aux_value = mfq_read_bits(aux_packed, state_index, 1u);
    }

    uint code_bank = 0u;
    if (CODE_BANK_MODE == 1) {
        code_bank = uint(state_to_codebank[state]);
    } else if (CODE_BANK_MODE == 2) {
        code_bank = aux_value;
    }
    uint code_offset = (
        (
            (table_bank * uint(CODE_BANKS) + code_bank)
            * uint(ENTRIES) + index
        )
        * uint(VECTOR_SIZE) + component
    );
    float code = float(codebooks[code_offset]);
    if (AUX_MODE == 1 || AUX_MODE == 2) {
        uint sign_position = column & 7u;
        uint negative = sign_position < 7u
            ? ((aux_value >> sign_position) & 1u)
            : (popcount(aux_value) & 1u);
        if (AUX_MODE == 2 && sign_position == 7u) {
            negative ^= (index >> 7u) & 1u;
        }
        code = negative != 0u ? -code : code;
    } else if (AUX_MODE == 3) {
        code += aux_value != 0u ? -parameters[0] : parameters[0];
    }
    y[linear] = T(
        anchors[output]
        * scale_lut[table_bank * uint(STATES) + state]
        * code
    );
"""

_VQ_EMBEDDING_SOURCE = r"""
    uint linear = thread_position_in_grid.x;
    if (linear >= uint(COUNT) * uint(K)) {
        return;
    }
    uint token_position = linear / uint(K);
    uint column = linear - token_position * uint(K);
    uint output = uint(token_ids[token_position]);
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
        uint(HAS_TABLE_BANKS),
        uint(GROUPS_PER_SUPER),
        uint(NSUPER),
        uint(NSIGN)
    );
    y[linear] = T(weight);
"""

_VQ_GEMM_MATRIX_SOURCE = r"""
    constexpr uint BM = uint(BM_TILE);
    constexpr uint BN = 64u;
    constexpr uint GPC = 4u;
    constexpr uint BK = uint(GS) * GPC;
    constexpr uint BK_PAD = BK + 8u;
    constexpr uint BN_PAD = BN + 8u;
    constexpr uint VECTORS_PER_GROUP =
        (uint(GS) + uint(VECTOR_SIZE) - 1u) / uint(VECTOR_SIZE);

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint local_thread = thread_index_in_threadgroup;
    uint row_base = threadgroup_position_in_grid.y * BM;
    uint output_base = threadgroup_position_in_grid.x * BN;

    threadgroup half activation_tile[BM * BK_PAD];
    threadgroup half weight_tile[BK * BN_PAD];

    metal::simdgroup_matrix<float, 8, 8> c00;
    metal::simdgroup_matrix<float, 8, 8> c01;
    metal::simdgroup_matrix<float, 8, 8> c10;
    metal::simdgroup_matrix<float, 8, 8> c11;
    metal::simdgroup_matrix<float, 8, 8> c20;
    metal::simdgroup_matrix<float, 8, 8> c21;
    metal::simdgroup_matrix<float, 8, 8> c30;
    metal::simdgroup_matrix<float, 8, 8> c31;
    c00.thread_elements()[0] = 0.0f;
    c00.thread_elements()[1] = 0.0f;
    c01.thread_elements()[0] = 0.0f;
    c01.thread_elements()[1] = 0.0f;
    c10.thread_elements()[0] = 0.0f;
    c10.thread_elements()[1] = 0.0f;
    c11.thread_elements()[0] = 0.0f;
    c11.thread_elements()[1] = 0.0f;
    if (BM == 64u) {
        c20.thread_elements()[0] = 0.0f;
        c20.thread_elements()[1] = 0.0f;
        c21.thread_elements()[0] = 0.0f;
        c21.thread_elements()[1] = 0.0f;
        c30.thread_elements()[0] = 0.0f;
        c30.thread_elements()[1] = 0.0f;
        c31.thread_elements()[0] = 0.0f;
        c31.thread_elements()[1] = 0.0f;
    }

    uint quadrant = lane / 4u;
    uint fragment_row = (quadrant & 4u) + ((lane / 2u) & 3u);
    uint fragment_col = (quadrant & 2u) * 2u + (lane & 1u) * 2u;
    uint simd_row = (simd_group / 4u) * (BM / 2u);
    uint simd_col = (simd_group & 3u) * 16u;

    uint chunks = (uint(NG) + GPC - 1u) / GPC;
    for (uint chunk = 0u; chunk < chunks; ++chunk) {
        uint group_base = chunk * GPC;
        uint column_base = group_base * uint(GS);

        for (
            uint index = local_thread;
            index < BM * BK;
            index += 256u
        ) {
            uint local_row = index / BK;
            uint local_column = index - local_row * BK;
            uint row = row_base + local_row;
            uint column = column_base + local_column;
            activation_tile[local_row * BK_PAD + local_column] =
                row < uint(M) && column < uint(K)
                ? half(x[row * uint(K) + column])
                : half(0.0f);
        }

        // Exactly BN*GPC threads each expand one output/group pair.  The
        // packed matrix is never materialized beyond this transient K96 tile.
        for (
            uint task = local_thread;
            task < BN * GPC;
            task += 256u
        ) {
            uint local_output = task / GPC;
            uint local_group = task - local_output * GPC;
            uint output = output_base + local_output;
            uint group = group_base + local_group;
            bool valid = output < uint(OUT) && group < uint(NG);

            uint state = 0u;
            uint table_bank = 0u;
            uint code_bank = 0u;
            uint delta_value = 0u;
            float weight_scale = 0.0f;
            if (valid) {
                uint state_index = output * uint(NG) + group;
                state = mfq_read_bits(
                    state_packed, state_index, uint(STATE_BITS));
                if (HAS_TABLE_BANKS != 0) {
                    table_bank = uint(bank_ids[
                        output * uint(NSUPER)
                        + group / uint(GROUPS_PER_SUPER)
                    ]);
                }
                if (AUX_MODE == 3) {
                    delta_value = mfq_read_bits(
                        aux_packed, state_index, 1u);
                }
                if (CODE_BANK_MODE == 1) {
                    code_bank = uint(state_to_codebank[state]);
                } else if (CODE_BANK_MODE == 2) {
                    code_bank = delta_value;
                }
                weight_scale = anchors[output]
                    * scale_lut[table_bank * uint(STATES) + state];
            }

            for (
                uint local_vector = 0u;
                local_vector < VECTORS_PER_GROUP;
                ++local_vector
            ) {
                uint local_column_base =
                    local_group * uint(GS)
                    + local_vector * uint(VECTOR_SIZE);
                uint global_column =
                    column_base + local_column_base;
                uint vector = global_column / uint(VECTOR_SIZE);
                uint index = valid && global_column < uint(K)
                    ? mfq_read_bits(
                        indices_packed,
                        output * uint(NVEC) + vector,
                        uint(INDEX_BITS))
                    : 0u;
                uint sign_value = 0u;
                if (
                    valid
                    && global_column < uint(K)
                    && (AUX_MODE == 1 || AUX_MODE == 2)
                ) {
                    sign_value = mfq_read_bits(
                        aux_packed,
                        output * uint(NSIGN) + global_column / 8u,
                        7u
                    );
                }
                for (
                    uint component = 0u;
                    component < uint(VECTOR_SIZE);
                    ++component
                ) {
                    uint local_column = local_column_base + component;
                    uint column = column_base + local_column;
                    float value = 0.0f;
                    if (valid && column < uint(K)) {
                        uint code_offset = (
                            (
                                (
                                    table_bank * uint(CODE_BANKS)
                                    + code_bank
                                )
                                * uint(ENTRIES) + index
                            )
                            * uint(VECTOR_SIZE) + component
                        );
                        float code = float(codebooks[code_offset]);
                        if (AUX_MODE == 1 || AUX_MODE == 2) {
                            uint sign_position = column & 7u;
                            uint negative = sign_position < 7u
                                ? ((sign_value >> sign_position) & 1u)
                                : (popcount(sign_value) & 1u);
                            if (
                                AUX_MODE == 2
                                && sign_position == 7u
                            ) {
                                negative ^= (index >> 7u) & 1u;
                            }
                            code = negative != 0u ? -code : code;
                        } else if (AUX_MODE == 3) {
                            code += delta_value != 0u
                                ? -parameters[0] : parameters[0];
                        }
                        value = weight_scale * code;
                    }
                    weight_tile[
                        local_column * BN_PAD + local_output
                    ] = half(value);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < BK; kk += 8u) {
            metal::simdgroup_matrix<half, 8, 8> a0;
            metal::simdgroup_matrix<half, 8, 8> a1;
            metal::simdgroup_matrix<half, 8, 8> a2;
            metal::simdgroup_matrix<half, 8, 8> a3;
            metal::simdgroup_matrix<half, 8, 8> b0;
            metal::simdgroup_matrix<half, 8, 8> b1;

            a0.thread_elements()[0] = activation_tile[
                (simd_row + fragment_row) * BK_PAD + kk + fragment_col];
            a0.thread_elements()[1] = activation_tile[
                (simd_row + fragment_row) * BK_PAD + kk + fragment_col + 1u];
            a1.thread_elements()[0] = activation_tile[
                (simd_row + 8u + fragment_row) * BK_PAD + kk + fragment_col];
            a1.thread_elements()[1] = activation_tile[
                (simd_row + 8u + fragment_row) * BK_PAD
                + kk + fragment_col + 1u];
            if (BM == 64u) {
                a2.thread_elements()[0] = activation_tile[
                    (simd_row + 16u + fragment_row) * BK_PAD
                    + kk + fragment_col];
                a2.thread_elements()[1] = activation_tile[
                    (simd_row + 16u + fragment_row) * BK_PAD
                    + kk + fragment_col + 1u];
                a3.thread_elements()[0] = activation_tile[
                    (simd_row + 24u + fragment_row) * BK_PAD
                    + kk + fragment_col];
                a3.thread_elements()[1] = activation_tile[
                    (simd_row + 24u + fragment_row) * BK_PAD
                    + kk + fragment_col + 1u];
            }

            b0.thread_elements()[0] = weight_tile[
                (kk + fragment_row) * BN_PAD + simd_col + fragment_col];
            b0.thread_elements()[1] = weight_tile[
                (kk + fragment_row) * BN_PAD + simd_col + fragment_col + 1u];
            b1.thread_elements()[0] = weight_tile[
                (kk + fragment_row) * BN_PAD
                + simd_col + 8u + fragment_col];
            b1.thread_elements()[1] = weight_tile[
                (kk + fragment_row) * BN_PAD
                + simd_col + 8u + fragment_col + 1u];

            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
            if (BM == 64u) {
                simdgroup_multiply_accumulate(c20, a2, b0, c20);
                simdgroup_multiply_accumulate(c21, a2, b1, c21);
                simdgroup_multiply_accumulate(c30, a3, b0, c30);
                simdgroup_multiply_accumulate(c31, a3, b1, c31);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint row0 = row_base + simd_row + fragment_row;
    uint row1 = row0 + 8u;
    uint col0 = output_base + simd_col + fragment_col;
    uint col1 = col0 + 8u;
    if (row0 < uint(M)) {
        if (col0 < uint(OUT)) {
            y[row0 * uint(OUT) + col0] = T(c00.thread_elements()[0]);
        }
        if (col0 + 1u < uint(OUT)) {
            y[row0 * uint(OUT) + col0 + 1u] = T(c00.thread_elements()[1]);
        }
        if (col1 < uint(OUT)) {
            y[row0 * uint(OUT) + col1] = T(c01.thread_elements()[0]);
        }
        if (col1 + 1u < uint(OUT)) {
            y[row0 * uint(OUT) + col1 + 1u] = T(c01.thread_elements()[1]);
        }
    }
    if (row1 < uint(M)) {
        if (col0 < uint(OUT)) {
            y[row1 * uint(OUT) + col0] = T(c10.thread_elements()[0]);
        }
        if (col0 + 1u < uint(OUT)) {
            y[row1 * uint(OUT) + col0 + 1u] = T(c10.thread_elements()[1]);
        }
        if (col1 < uint(OUT)) {
            y[row1 * uint(OUT) + col1] = T(c11.thread_elements()[0]);
        }
        if (col1 + 1u < uint(OUT)) {
            y[row1 * uint(OUT) + col1 + 1u] = T(c11.thread_elements()[1]);
        }
    }
    if (BM == 64u) {
        uint row2 = row0 + 16u;
        uint row3 = row0 + 24u;
        if (row2 < uint(M)) {
            if (col0 < uint(OUT)) {
                y[row2 * uint(OUT) + col0] = T(c20.thread_elements()[0]);
            }
            if (col0 + 1u < uint(OUT)) {
                y[row2 * uint(OUT) + col0 + 1u] =
                    T(c20.thread_elements()[1]);
            }
            if (col1 < uint(OUT)) {
                y[row2 * uint(OUT) + col1] = T(c21.thread_elements()[0]);
            }
            if (col1 + 1u < uint(OUT)) {
                y[row2 * uint(OUT) + col1 + 1u] =
                    T(c21.thread_elements()[1]);
            }
        }
        if (row3 < uint(M)) {
            if (col0 < uint(OUT)) {
                y[row3 * uint(OUT) + col0] = T(c30.thread_elements()[0]);
            }
            if (col0 + 1u < uint(OUT)) {
                y[row3 * uint(OUT) + col0 + 1u] =
                    T(c30.thread_elements()[1]);
            }
            if (col1 < uint(OUT)) {
                y[row3 * uint(OUT) + col1] = T(c31.thread_elements()[0]);
            }
            if (col1 + 1u < uint(OUT)) {
                y[row3 * uint(OUT) + col1 + 1u] =
                    T(c31.thread_elements()[1]);
            }
        }
    }
"""

_HADAMARD_SOURCE = r"""
    uint row = thread_position_in_grid.x / 256u;
    uint lane = thread_index_in_threadgroup;
    if (row >= uint(M)) {
        return;
    }

    threadgroup float values[BLOCK];
    for (uint local_block = 0u; local_block < uint(K) / uint(BLOCK); ++local_block) {
        uint column_base = local_block * uint(BLOCK);
        for (uint index = lane; index < uint(BLOCK); index += 256u) {
            uint column = column_base + index;
            values[index] = float(x[row * uint(K) + column]) * float(signs[column]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 1u; stride < uint(BLOCK); stride <<= 1u) {
            for (uint pair = lane; pair < uint(BLOCK) / 2u; pair += 256u) {
                uint pair_block = pair / stride;
                uint within = pair - pair_block * stride;
                uint first = pair_block * (stride << 1u) + within;
                uint second = first + stride;
                float a = values[first];
                float b = values[second];
                values[first] = a + b;
                values[second] = a - b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float inverse = rsqrt(float(BLOCK));
        for (uint index = lane; index < uint(BLOCK); index += 256u) {
            uint column = column_base + index;
            y[row * uint(K) + column] = T(values[index] * inverse);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


def _vq_kernel(name: str, source: str):
    return mx.fast.metal_kernel(
        name=name,
        input_names=[
            "indices_packed",
            "state_packed",
            "aux_packed",
            "anchors",
            "codebooks",
            "scale_lut",
            "state_to_codebank",
            "bank_ids",
            "x",
            "parameters",
        ],
        output_names=["y"],
        header=_BITSTREAM_HEADER,
        source=source,
        compile_options={"math_mode": "fast"},
    )


_VQ_GEMV_KERNEL = _vq_kernel("mfq_vq_gemv", _VQ_MATMUL_SOURCE)
_VQ_GEMV_FAST_KERNEL = _vq_kernel("mfq_vq_gemv_fast", _VQ_GEMV_FAST_SOURCE)
_VQ_MMQ_KERNEL = _vq_kernel("mfq_vq_mmq", _VQ_MATMUL_SOURCE)
_VQ_MMQ_WIDE_KERNEL = _vq_kernel("mfq_vq_mmq_wide", _VQ_MMQ_WIDE_SOURCE)
_VQ_GEMM_KERNEL = _vq_kernel("mfq_vq_gemm", _VQ_MATMUL_SOURCE)
_VQ_GEMM_MATRIX_KERNEL = _vq_kernel(
    "mfq_vq_gemm_matrix",
    _VQ_GEMM_MATRIX_SOURCE,
)
_VQ_DEQUANT_KERNEL = mx.fast.metal_kernel(
    name="mfq_vq_dequant",
    input_names=[
        "indices_packed",
        "state_packed",
        "aux_packed",
        "anchors",
        "codebooks",
        "scale_lut",
        "state_to_codebank",
        "bank_ids",
        "parameters",
    ],
    output_names=["y"],
    header=_BITSTREAM_HEADER,
    source=_VQ_DEQUANT_SOURCE,
    compile_options={"math_mode": "fast"},
)
_VQ_EMBEDDING_KERNEL = mx.fast.metal_kernel(
    name="mfq_vq_packed_embedding",
    input_names=[
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
    ],
    output_names=["y"],
    header=_BITSTREAM_HEADER,
    source=_VQ_EMBEDDING_SOURCE,
    compile_options={"math_mode": "fast"},
)
_VQ_SWIGLU_KERNEL = mx.fast.metal_kernel(
    name="mfq_vq_packed_swiglu",
    input_names=[
        "gate_indices",
        "gate_state",
        "gate_aux",
        "gate_anchors",
        "gate_codebooks",
        "gate_scales",
        "gate_state_to_codebank",
        "gate_banks",
        "gate_parameters",
        "up_indices",
        "up_state",
        "up_aux",
        "up_anchors",
        "up_codebooks",
        "up_scales",
        "up_state_to_codebank",
        "up_banks",
        "up_parameters",
        "x",
    ],
    output_names=["y"],
    header=_BITSTREAM_HEADER,
    source=_VQ_SWIGLU_SOURCE,
    compile_options={"math_mode": "fast"},
)

_HADAMARD_KERNEL = mx.fast.metal_kernel(
    name="mfq_signed_hadamard",
    input_names=["x", "signs"],
    output_names=["y"],
    source=_HADAMARD_SOURCE,
    compile_options={"math_mode": "fast"},
)

_NEPQ_RESIDUAL_MATMUL_SOURCE = r"""
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
"""

_NEPQ_RESIDUAL_DEQUANT_SOURCE = r"""
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
"""

_NEPQ_RESIDUAL_MATMUL_KERNEL = mx.fast.metal_kernel(
    name="mfq_nepq_sparse_residual_matmul",
    input_names=[
        "base",
        "x",
        "residual_codebook",
        "residual_first",
        "residual_second",
    ],
    output_names=["y"],
    source=_NEPQ_RESIDUAL_MATMUL_SOURCE,
    compile_options={"math_mode": "fast"},
)

_NEPQ_RESIDUAL_DEQUANT_KERNEL = mx.fast.metal_kernel(
    name="mfq_nepq_sparse_residual_dequant",
    input_names=[
        "base",
        "residual_codebook",
        "residual_first",
        "residual_second",
    ],
    output_names=["y"],
    source=_NEPQ_RESIDUAL_DEQUANT_SOURCE,
    compile_options={"math_mode": "fast"},
)


def _pack_bits(values: np.ndarray, bits: int) -> np.ndarray:
    source = np.ascontiguousarray(values).reshape(-1)
    if bits == 0:
        return np.zeros(3, dtype=np.uint8)
    if not 1 <= int(bits) <= 16:
        raise ValueError(f"unsupported packed width: {bits}")
    unsigned = source.astype(np.uint16, copy=False)
    if np.any(unsigned >= (1 << int(bits))):
        raise ValueError(f"value exceeds {bits}-bit packed width")
    shifts = np.arange(int(bits), dtype=np.uint16)
    rows = ((unsigned[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    packed = np.packbits(rows.reshape(-1), bitorder="little")
    return np.concatenate((packed, np.zeros(2, dtype=np.uint8)))


def _product_codebooks(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.int8)
    second = np.asarray(second, dtype=np.int8)
    if first.ndim != 3 or second.ndim != 3 or first.shape[0] != second.shape[0]:
        raise ValueError("invalid NPQ product codebooks")
    states, first_entries, width = first.shape
    second_entries = second.shape[1]
    if width != 4 or second.shape[2] != 4:
        raise ValueError("NPQ product codebook subvectors must have width four")
    first_product = np.broadcast_to(
        first[:, None, :, :],
        (states, second_entries, first_entries, 4),
    )
    second_product = np.broadcast_to(
        second[:, :, None, :],
        (states, second_entries, first_entries, 4),
    )
    return np.ascontiguousarray(
        np.concatenate((first_product, second_product), axis=-1).reshape(
            states, first_entries * second_entries, 8
        ),
        dtype=np.int8,
    )


def _matrix_shape(tensor: VqTensor) -> tuple[int, int]:
    if isinstance(tensor, NepqTensor):
        return tensor.n_experts * tensor.out_per_expert, tensor.neuron_len
    if tensor.axis != 0 or len(tensor.shape) != 2:
        raise ValueError("Metal NVQ/NPQ linear weights must be rank-2 and quantized on axis 0")
    out = int(np.asarray(tensor.neuron_scale).size)
    if tuple(tensor.shape) != (out, int(tensor.neuron_len)):
        raise ValueError("Metal NVQ/NPQ matrix dimensions are inconsistent")
    return out, int(tensor.neuron_len)


@dataclass(frozen=True)
class MetalVqWeight:
    """Canonical packed Metal layout shared by NVQ, NPQ, and NEPQ."""

    indices_packed: mx.array
    state_packed: mx.array
    aux_packed: mx.array
    anchors: mx.array
    codebooks: mx.array
    scale_lut: mx.array
    state_to_codebank: mx.array
    bank_ids: mx.array
    rotation_signs: mx.array
    parameters: mx.array
    residual_codebook: mx.array
    residual_first: mx.array
    residual_second: mx.array
    format_label: str
    out: int
    neuron_len: int
    groupsize: int
    groups: int
    vector_size: int
    vectors: int
    index_bits: int
    state_bits: int
    states: int
    entries: int
    code_banks: int
    aux_mode: int
    code_bank_mode: int
    table_banks: int
    groups_per_super: int
    supergroups: int
    output_shape: tuple[int, ...]
    rotation_block: int
    rotation_seed: int
    residual_position_bits: int
    residual_block_vectors: int
    residual_blocks_per_row: int

    @classmethod
    def from_tensor(cls, tensor: VqTensor) -> MetalVqWeight:
        out, neuron_len = _matrix_shape(tensor)
        groupsize = int(tensor.spec.groupsize)
        groups = math.ceil(neuron_len / groupsize)
        vector_size = int(tensor.spec.vector_size)
        vectors = math.ceil(neuron_len / vector_size)
        output_shape = (out,)
        table_banks = 1
        groups_per_super = groups
        supergroups = 1
        bank_ids = np.zeros((out, 1), dtype=np.uint8)
        rotation = np.empty(0, dtype=np.int8)
        rotation_block = 0
        rotation_seed = 0
        residual_codebook = np.zeros((1, 8), dtype=np.float16)
        residual_first = np.zeros((1, 1), dtype=np.int16)
        residual_second = np.full((1, 1), -1, dtype=np.int16)
        residual_position_bits = 0
        residual_block_vectors = 0
        residual_blocks_per_row = 0

        if isinstance(tensor, NepqTensor):
            _, vectors, supergroups, table_banks, _ = validate_nepq(tensor)
            output_shape = (tensor.n_experts, tensor.out_per_expert)
            groups_per_super = tensor.spec.groups_per_supergroup
            bank_ids = np.ascontiguousarray(tensor.bank_ids, dtype=np.uint8).reshape(
                out, supergroups
            )
            rotation_block = int(tensor.rotation_block)
            rotation_seed = int(tensor.rotation_seed)
            rotation = (
                rotation_signs(neuron_len, rotation_block, int(tensor.rotation_seed))
                if rotation_block
                else np.empty(0, dtype=np.int8)
            )
            if tensor.spec.is_residual:
                residual_blocks_per_row = int(tensor.residual_blocks_per_row)
                residual_position_bits = int(tensor.spec.residual_position_bits)
                residual_block_vectors = int(tensor.spec.residual_block_vectors)
                residual_codebook = np.ascontiguousarray(
                    tensor.residual_codebook,
                    dtype=np.float16,
                )
                residual_first = np.ascontiguousarray(
                    tensor.residual_first,
                    dtype=np.int16,
                ).reshape(out, residual_blocks_per_row)
                residual_second = np.full(
                    (out, residual_blocks_per_row),
                    -1,
                    dtype=np.int16,
                )
                if tensor.spec.residual_second:
                    mask = np.asarray(
                        tensor.residual_second_mask,
                        dtype=np.uint8,
                    ).reshape(-1)
                    compact = np.asarray(
                        tensor.residual_second_records,
                        dtype=np.int16,
                    ).reshape(-1)
                    residual_second.reshape(-1)[np.flatnonzero(mask)] = compact

        aux_mode = _AUX_NONE
        code_bank_mode = _CODE_BANK_FIXED
        delta = 0.0
        state_to_codebank: np.ndarray

        if isinstance(tensor, NvqJscTensor):
            states = 16
            entries = tensor.spec.codebook_entries
            code_banks = int(tensor.codebooks.shape[0])
            codebooks = np.asarray(tensor.codebooks, dtype=np.int8)[None, ...]
            scale_lut = np.asarray(tensor.scale_lut, dtype=np.float32)[None, :]
            state_to_codebank = np.asarray(tensor.bank_for_state, dtype=np.uint8)
            state = tensor.state
            indices = tensor.indices
            aux = tensor.signs
            aux_mode = _AUX_SIGN_EVEN
            code_bank_mode = _CODE_BANK_STATE
            label = {
                "e8_256": "NVQ2J",
                "e8_1024": "NVQ2J-L",
                "e8_4096": "NVQ2J-XL",
                "d4_256": "NVQ3J",
                "d4_512": "NVQ3J-512",
                "d4_1024": "NVQ3J-L",
            }.get(tensor.spec.codebook, tensor.spec.label)
        elif isinstance(tensor, NvqTensor):
            states = 1 << int(tensor.spec.sub_bits)
            entries = tensor.spec.codebook_entries
            code_banks = 1
            table = tensor.codebook if tensor.codebook is not None else codebook_for(tensor.spec)
            codebooks = np.asarray(table, dtype=np.int8)[None, None, ...]
            scale_lut = np.arange(states, dtype=np.float32)[None, :]
            state_to_codebank = np.zeros(states, dtype=np.uint8)
            state = tensor.sub_scale
            indices = tensor.indices
            aux = tensor.signs
            aux_mode = (
                _AUX_SIGN_INDEX_PARITY
                if tensor.spec.sign_mode == "index_parity"
                else _AUX_SIGN_EVEN
            )
            label = tensor.spec.label
        elif isinstance(tensor, Nvq1STensor):
            states = 1 << int(tensor.spec.sub_bits)
            entries = 1 << int(tensor.spec.index_bits)
            code_banks = 2
            table = NVQ1_S_SYNTHETIC_BANKS if tensor.codebook is None else tensor.codebook
            codebooks = np.asarray(table, dtype=np.int8)[None, ...]
            scale_lut = np.arange(states, dtype=np.float32)[None, :]
            state_to_codebank = np.zeros(states, dtype=np.uint8)
            state = tensor.sub_scale
            indices = tensor.indices
            aux = tensor.delta_sign
            aux_mode = _AUX_DELTA
            code_bank_mode = _CODE_BANK_AUX
            delta = float(tensor.spec.delta)
            label = tensor.spec.label
        elif isinstance(tensor, Nvq1LTensor):
            states = 1 << int(tensor.spec.sub_bits)
            entries = 1 << int(tensor.spec.index_bits)
            code_banks = 1
            table = IQ1S_TERNARY_2048 if tensor.codebook is None else tensor.codebook
            codebooks = np.asarray(table, dtype=np.int8)[None, None, ...]
            scale_lut = np.arange(states, dtype=np.float32)[None, :]
            state_to_codebank = np.zeros(states, dtype=np.uint8)
            state = tensor.sub_scale
            indices = tensor.indices
            aux = tensor.delta_sign
            aux_mode = _AUX_DELTA
            delta = float(tensor.spec.delta)
            label = "NVQ1-L"
        elif isinstance(tensor, Npq0STensor | Npq0LTensor):
            states = 1 << int(tensor.spec.state_bits)
            entries = 1 << int(tensor.spec.index_bits)
            code_banks = states
            product = _product_codebooks(
                tensor.first_codebooks,
                tensor.second_codebooks,
            )
            codebooks = product[None, ...]
            scale_lut = np.asarray(tensor.scale_lut, dtype=np.float32)[None, :]
            state_to_codebank = np.arange(states, dtype=np.uint8)
            state = tensor.state
            indices = tensor.indices
            aux = np.empty(0, dtype=np.uint8)
            code_bank_mode = _CODE_BANK_STATE
            label = tensor.spec.label
        elif isinstance(tensor, NepqTensor):
            base_spec = nepq_base_spec(tensor.spec)
            states = 1 << int(tensor.spec.state_bits)
            entries = 1 << int(tensor.spec.index_bits)
            state = tensor.state
            indices = tensor.indices
            aux = np.empty(0, dtype=np.uint8) if tensor.aux is None else tensor.aux
            labels = []
            scales = []
            tables = []
            for payload in np.asarray(tensor.table_payloads, dtype=np.uint8):
                raw = payload.tobytes()
                if base_spec is NEPQ0_S:
                    scale, first, second, _ = unpack_npq0_s_tables(raw)
                    table = _product_codebooks(first, second)
                    labels.append(tensor.spec.label)
                elif base_spec is NEPQ0_L:
                    scale, first, second, _ = unpack_npq0_l_tables(raw)
                    table = _product_codebooks(first, second)
                    labels.append(tensor.spec.label)
                elif base_spec is NEPQ1_S:
                    scale = np.arange(states, dtype=np.float32)
                    table = unpack_nvq1_s_banked_codebook(raw)
                    labels.append(tensor.spec.label)
                elif base_spec is NEPQ1_L:
                    scale = np.arange(states, dtype=np.float32)
                    table = unpack_ternary_codebook(raw)[None, ...]
                    labels.append(tensor.spec.label)
                else:  # pragma: no cover - validated above
                    raise ValueError(f"unsupported NEPQ profile: {tensor.spec.label}")
                scales.append(scale)
                tables.append(table)
            scale_lut = np.ascontiguousarray(np.stack(scales), dtype=np.float32)
            codebooks = np.ascontiguousarray(np.stack(tables), dtype=np.int8)
            label = labels[0]
            if base_spec in (NEPQ0_S, NEPQ0_L):
                code_banks = states
                code_bank_mode = _CODE_BANK_STATE
                state_to_codebank = np.arange(states, dtype=np.uint8)
            elif base_spec is NEPQ1_S:
                code_banks = 2
                code_bank_mode = _CODE_BANK_AUX
                state_to_codebank = np.zeros(states, dtype=np.uint8)
                aux_mode = _AUX_DELTA
                delta = 0.15625
            else:
                code_banks = 1
                state_to_codebank = np.zeros(states, dtype=np.uint8)
                aux_mode = _AUX_DELTA
                delta = 0.125
        else:  # pragma: no cover - exhaustive TypeAlias
            raise TypeError(f"unsupported Metal VQ tensor: {type(tensor).__name__}")

        index_bits = int(tensor.spec.index_bits)
        state_bits = int(
            tensor.spec.state_bits
            if isinstance(tensor, (Npq0STensor, Npq0LTensor, NepqTensor))
            else tensor.spec.sub_bits
        )
        return cls(
            indices_packed=mx.array(_pack_bits(np.asarray(indices), index_bits)),
            state_packed=mx.array(_pack_bits(np.asarray(state), state_bits)),
            aux_packed=mx.array(
                _pack_bits(
                    np.asarray(aux),
                    7
                    if aux_mode in (_AUX_SIGN_EVEN, _AUX_SIGN_INDEX_PARITY)
                    else (1 if aux_mode == _AUX_DELTA else 0),
                )
            ),
            anchors=mx.array(
                np.ascontiguousarray(tensor.neuron_scale, dtype=np.float32).reshape(-1)
            ),
            codebooks=mx.array(np.ascontiguousarray(codebooks, dtype=np.int8)),
            scale_lut=mx.array(np.ascontiguousarray(scale_lut, dtype=np.float32)),
            state_to_codebank=mx.array(np.ascontiguousarray(state_to_codebank, dtype=np.uint8)),
            bank_ids=mx.array(bank_ids),
            rotation_signs=mx.array(
                np.ascontiguousarray(rotation if rotation.size else np.zeros(1), dtype=np.int8)
            ),
            parameters=mx.array([delta], dtype=mx.float32),
            residual_codebook=mx.array(residual_codebook),
            residual_first=mx.array(residual_first),
            residual_second=mx.array(residual_second),
            format_label=label,
            out=out,
            neuron_len=neuron_len,
            groupsize=groupsize,
            groups=groups,
            vector_size=vector_size,
            vectors=vectors,
            index_bits=index_bits,
            state_bits=state_bits,
            states=states,
            entries=entries,
            code_banks=code_banks,
            aux_mode=aux_mode,
            code_bank_mode=code_bank_mode,
            table_banks=table_banks,
            groups_per_super=groups_per_super,
            supergroups=supergroups,
            output_shape=output_shape,
            rotation_block=rotation_block,
            rotation_seed=rotation_seed,
            residual_position_bits=residual_position_bits,
            residual_block_vectors=residual_block_vectors,
            residual_blocks_per_row=residual_blocks_per_row,
        )

    @classmethod
    def from_blob(
        cls,
        dtype: str,
        blob: bytes | memoryview,
    ) -> MetalVqWeight:
        """Upload a packed MFQ VQ payload without materializing dense weights."""

        from mfq.formats.io import unpack_tensor_payload

        tensor = unpack_tensor_payload(dtype, blob)
        if not isinstance(
            tensor,
            (
                NvqTensor,
                NvqJscTensor,
                Nvq1LTensor,
                Nvq1STensor,
                Npq0LTensor,
                Npq0STensor,
                NepqTensor,
            ),
        ):
            raise TypeError(f"MFQ payload {dtype!r} is not an NVQ/NPQ/NEPQ tensor")
        return cls.from_tensor(tensor)

    @property
    def packed_nbytes(self) -> int:
        arrays = (
            self.indices_packed,
            self.state_packed,
            self.aux_packed,
            self.anchors,
            self.codebooks,
            self.scale_lut,
            self.state_to_codebank,
            self.bank_ids,
            self.rotation_signs,
            self.parameters,
            self.residual_codebook,
            self.residual_first,
            self.residual_second,
        )
        return sum(int(array.nbytes) for array in arrays)


def _floating(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if result.dtype not in (mx.float16, mx.float32):
        result = result.astype(mx.float16)
    return mx.contiguous(result)


def signed_hadamard(
    x: mx.array | np.ndarray,
    signs: mx.array | np.ndarray,
    block: int,
) -> mx.array:
    """Apply the normalized signed block Hadamard transform used by NEPQ."""

    source = _floating(x)
    if source.ndim != 2:
        raise ValueError("signed Hadamard input must be rank-2 [M,K]")
    rows, width = (int(value) for value in source.shape)
    block = int(block)
    if block <= 0 or block & (block - 1) or width % block:
        raise ValueError("Hadamard block must be a power of two dividing K")
    if block > 8192:
        raise ValueError("Hadamard block exceeds Metal threadgroup memory")
    diagonal = signs if isinstance(signs, mx.array) else mx.array(signs)
    if diagonal.ndim != 1 or int(diagonal.size) != width:
        raise ValueError(f"Hadamard sign diagonal must have shape ({width},)")
    diagonal = mx.contiguous(diagonal.astype(mx.int8))
    if rows == 0:
        return mx.zeros(source.shape, dtype=source.dtype)
    return _HADAMARD_KERNEL(
        inputs=[source, diagonal],
        template=[
            ("T", source.dtype),
            ("M", rows),
            ("K", width),
            ("BLOCK", block),
        ],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[source.shape],
        output_dtypes=[source.dtype],
    )[0]


def _prepare_input(
    weight: MetalVqWeight,
    x: mx.array | np.ndarray,
) -> tuple[mx.array, tuple[int, ...], int]:
    source = _floating(x)
    if source.ndim < 1:
        raise ValueError("Metal VQ matmul input must have at least one dimension")
    if int(source.shape[-1]) != weight.neuron_len:
        raise ValueError(
            f"Metal VQ input width {source.shape[-1]} != weight width {weight.neuron_len}"
        )
    prefix = tuple(int(value) for value in source.shape[:-1])
    rows = int(np.prod(prefix, dtype=np.int64)) if prefix else 1
    source = source.reshape((rows, weight.neuron_len))
    if weight.rotation_block:
        source = signed_hadamard(source, weight.rotation_signs, weight.rotation_block)
    return source, prefix, rows


def _has_sparse_residual(weight: MetalVqWeight) -> bool:
    return weight.residual_position_bits > 0


def _add_sparse_residual_matmul(
    weight: MetalVqWeight,
    source: mx.array,
    base: mx.array,
) -> mx.array:
    if not _has_sparse_residual(weight):
        return base
    rows = int(source.shape[0])
    total = rows * weight.out
    workgroups = (total + 3) // 4
    return _NEPQ_RESIDUAL_MATMUL_KERNEL(
        inputs=[
            base,
            source,
            weight.residual_codebook,
            weight.residual_first,
            weight.residual_second,
        ],
        template=[
            ("T", source.dtype),
            ("M", rows),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("NVEC", weight.vectors),
            ("RESIDUAL_BLOCKS", weight.residual_blocks_per_row),
            ("POSITION_BITS", weight.residual_position_bits),
            ("BLOCK_VECTORS", weight.residual_block_vectors),
        ],
        grid=(workgroups * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, weight.out)],
        output_dtypes=[source.dtype],
    )[0]


def _add_sparse_residual_dequant(
    weight: MetalVqWeight,
    base: mx.array,
) -> mx.array:
    if not _has_sparse_residual(weight):
        return base
    total = weight.out * weight.neuron_len
    return _NEPQ_RESIDUAL_DEQUANT_KERNEL(
        inputs=[
            base,
            weight.residual_codebook,
            weight.residual_first,
            weight.residual_second,
        ],
        template=[
            ("T", base.dtype),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("RESIDUAL_BLOCKS", weight.residual_blocks_per_row),
            ("POSITION_BITS", weight.residual_position_bits),
            ("BLOCK_VECTORS", weight.residual_block_vectors),
        ],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
        output_shapes=[(weight.out, weight.neuron_len)],
        output_dtypes=[base.dtype],
    )[0]


def _matmul(
    weight: MetalVqWeight,
    x: mx.array | np.ndarray,
    *,
    path: str,
) -> mx.array:
    source, prefix, rows = _prepare_input(weight, x)
    if rows == 0:
        return mx.zeros((*prefix, *weight.output_shape), dtype=source.dtype)
    if path == "gemv":
        if rows != 1:
            raise ValueError("VQ GEMV requires exactly one input row")
        kernel = _VQ_GEMV_FAST_KERNEL
        tile_rows = 1
        grid = (((weight.out + 7) // 8) * 64, 1, 1)
        threadgroup = (64, 1, 1)
    elif path == "mmq":
        if not 2 <= rows <= 16:
            raise ValueError("VQ MMQ requires 2 to 16 input rows")
        vector_tiles = (rows + 4) // 5
        tile_rows = (rows + vector_tiles - 1) // vector_tiles
        kernel = _VQ_MMQ_WIDE_KERNEL
        grid = (
            vector_tiles * 64,
            (weight.out + 7) // 8,
            1,
        )
        threadgroup = (64, 1, 1)
    elif path == "gemm":
        if source.dtype == mx.float16:
            kernel = _VQ_GEMM_MATRIX_KERNEL
            tile_rows = 64 if rows >= 64 else 32
            grid = (
                ((weight.out + 63) // 64) * 256,
                (rows + tile_rows - 1) // tile_rows,
                1,
            )
            threadgroup = (256, 1, 1)
        else:
            kernel = _VQ_GEMM_KERNEL
            tile_rows = 8
            row_tiles = (rows + tile_rows - 1) // tile_rows
            grid = (row_tiles * weight.out * 32, 1, 1)
            threadgroup = (32, 1, 1)
    else:
        raise ValueError(f"unknown Metal VQ matmul path: {path}")

    output = kernel(
        inputs=[
            weight.indices_packed,
            weight.state_packed,
            weight.aux_packed,
            weight.anchors,
            weight.codebooks,
            weight.scale_lut,
            weight.state_to_codebank,
            weight.bank_ids,
            source,
            weight.parameters,
        ],
        template=[
            ("T", source.dtype),
            ("M", rows),
            ("TILE_M", tile_rows),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("GS", weight.groupsize),
            ("NG", weight.groups),
            ("VECTOR_SIZE", weight.vector_size),
            ("NVEC", weight.vectors),
            ("INDEX_BITS", weight.index_bits),
            ("STATE_BITS", weight.state_bits),
            ("STATES", weight.states),
            ("ENTRIES", weight.entries),
            ("CODE_BANKS", weight.code_banks),
            ("AUX_MODE", weight.aux_mode),
            ("CODE_BANK_MODE", weight.code_bank_mode),
            ("HAS_TABLE_BANKS", int(weight.table_banks > 1)),
            ("GROUPS_PER_SUPER", weight.groups_per_super),
            ("NSUPER", weight.supergroups),
            ("NSIGN", math.ceil(weight.neuron_len / 8)),
            ("BM_TILE", tile_rows if kernel is _VQ_GEMM_MATRIX_KERNEL else 32),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(rows, weight.out)],
        output_dtypes=[source.dtype],
    )[0]
    output = _add_sparse_residual_matmul(weight, source, output)
    return output.reshape((*prefix, *weight.output_shape))


def vq_gemv(weight: MetalVqWeight, x: mx.array | np.ndarray) -> mx.array:
    """Single-row packed vector-quantized matrix-vector multiply."""

    return _matmul(weight, x, path="gemv")


def vq_mmq(weight: MetalVqWeight, x: mx.array | np.ndarray) -> mx.array:
    """Small-M qmv_wide multiply sharing decoded weights across row tiles."""

    return _matmul(weight, x, path="mmq")


def vq_gemm(weight: MetalVqWeight, x: mx.array | np.ndarray) -> mx.array:
    """Online-decode simdgroup-matrix prefill multiply for packed weights."""

    return _matmul(weight, x, path="gemm")


def vq_matmul(
    weight: MetalVqWeight,
    x: mx.array | np.ndarray,
    *,
    dequantize_threshold: int | None = 64,
) -> mx.array:
    """Dispatch VQ matmul across packed and temporary-dense paths."""

    source = x if isinstance(x, mx.array) else mx.array(x)
    if source.ndim < 1:
        raise ValueError("Metal VQ matmul input must have at least one dimension")
    rows = (
        int(np.prod(tuple(int(value) for value in source.shape[:-1]), dtype=np.int64))
        if source.ndim > 1
        else 1
    )
    if (
        dequantize_threshold is not None
        and rows >= int(dequantize_threshold)
        and source.dtype == mx.float16
    ):
        return vq_dequantize_matmul(weight, source)
    if rows == 1:
        return vq_gemv(weight, source)
    if rows <= 16:
        return vq_mmq(weight, source)
    return vq_gemm(weight, source)


def vq_dequantize(
    weight: MetalVqWeight,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Decode a packed VQ-family matrix to a temporary dense Metal array."""

    size = weight.out * weight.neuron_len
    if size == 0:
        return mx.zeros((weight.out, weight.neuron_len), dtype=dtype)
    output = _VQ_DEQUANT_KERNEL(
        inputs=[
            weight.indices_packed,
            weight.state_packed,
            weight.aux_packed,
            weight.anchors,
            weight.codebooks,
            weight.scale_lut,
            weight.state_to_codebank,
            weight.bank_ids,
            weight.parameters,
        ],
        template=[
            ("T", dtype),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("GS", weight.groupsize),
            ("NG", weight.groups),
            ("VECTOR_SIZE", weight.vector_size),
            ("NVEC", weight.vectors),
            ("INDEX_BITS", weight.index_bits),
            ("STATE_BITS", weight.state_bits),
            ("STATES", weight.states),
            ("ENTRIES", weight.entries),
            ("CODE_BANKS", weight.code_banks),
            ("AUX_MODE", weight.aux_mode),
            ("CODE_BANK_MODE", weight.code_bank_mode),
            ("HAS_TABLE_BANKS", int(weight.table_banks > 1)),
            ("GROUPS_PER_SUPER", weight.groups_per_super),
            ("NSUPER", weight.supergroups),
            ("NSIGN", math.ceil(weight.neuron_len / 8)),
        ],
        grid=(size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(weight.out, weight.neuron_len)],
        output_dtypes=[dtype],
    )[0]
    return _add_sparse_residual_dequant(weight, output)


def vq_dequantize_matmul(
    weight: MetalVqWeight,
    x: mx.array | np.ndarray,
) -> mx.array:
    """Temporarily dequantize VQ-family weights and dispatch MLX dense GEMM."""

    source, prefix, rows = _prepare_input(weight, x)
    if rows == 0:
        return mx.zeros((*prefix, *weight.output_shape), dtype=source.dtype)
    dtype = mx.float16 if source.dtype == mx.float16 else mx.float32
    dense = vq_dequantize(weight, dtype=dtype)
    result = source.astype(dtype) @ dense.T
    return result.reshape((*prefix, *weight.output_shape))


def vq_embedding(
    weight: MetalVqWeight,
    token_ids: mx.array | np.ndarray,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Decode only selected NVQ/NPQ rows into an embedding result."""

    if weight.rotation_block or weight.output_shape != (weight.out,):
        raise ValueError("VQ embedding requires a non-rotated matrix weight")
    ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
    if ids.dtype not in (mx.int32, mx.uint32):
        ids = ids.astype(mx.int32)
    shape = tuple(int(value) for value in ids.shape)
    count = int(ids.size)
    if count == 0:
        return mx.zeros((*shape, weight.neuron_len), dtype=dtype)
    ids = mx.contiguous(ids.reshape((-1,)))
    output_size = count * weight.neuron_len
    output = _VQ_EMBEDDING_KERNEL(
        inputs=[
            weight.indices_packed,
            weight.state_packed,
            weight.aux_packed,
            weight.anchors,
            weight.codebooks,
            weight.scale_lut,
            weight.state_to_codebank,
            weight.bank_ids,
            weight.parameters,
            ids,
        ],
        template=[
            ("T", dtype),
            ("COUNT", count),
            ("OUT", weight.out),
            ("K", weight.neuron_len),
            ("GS", weight.groupsize),
            ("NG", weight.groups),
            ("VECTOR_SIZE", weight.vector_size),
            ("NVEC", weight.vectors),
            ("INDEX_BITS", weight.index_bits),
            ("STATE_BITS", weight.state_bits),
            ("STATES", weight.states),
            ("ENTRIES", weight.entries),
            ("CODE_BANKS", weight.code_banks),
            ("AUX_MODE", weight.aux_mode),
            ("CODE_BANK_MODE", weight.code_bank_mode),
            ("HAS_TABLE_BANKS", int(weight.table_banks > 1)),
            ("GROUPS_PER_SUPER", weight.groups_per_super),
            ("NSUPER", weight.supergroups),
            ("NSIGN", math.ceil(weight.neuron_len / 8)),
        ],
        grid=(output_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(count, weight.neuron_len)],
        output_dtypes=[dtype],
    )[0]
    return output.reshape((*shape, weight.neuron_len))


def vq_swiglu_compatible(gate: MetalVqWeight, up: MetalVqWeight) -> bool:
    """Return whether gate/up can share one packed VQ SwiGLU dispatch."""

    if _has_sparse_residual(gate) or _has_sparse_residual(up):
        return False

    fields = (
        "format_label",
        "out",
        "neuron_len",
        "groupsize",
        "groups",
        "vector_size",
        "vectors",
        "index_bits",
        "state_bits",
        "states",
        "entries",
        "code_banks",
        "aux_mode",
        "code_bank_mode",
        "table_banks",
        "groups_per_super",
        "supergroups",
        "output_shape",
        "rotation_block",
        "rotation_seed",
    )
    return all(getattr(gate, name) == getattr(up, name) for name in fields)


def vq_swiglu(
    gate: MetalVqWeight,
    up: MetalVqWeight,
    x: mx.array | np.ndarray,
) -> mx.array:
    """Fuse compatible VQ-family gate/up projections with the SiLU product."""

    if not vq_swiglu_compatible(gate, up):
        raise ValueError("fused VQ SwiGLU requires matching gate/up execution layouts")
    source, prefix, rows = _prepare_input(gate, x)
    if rows == 0:
        return mx.zeros((*prefix, *gate.output_shape), dtype=source.dtype)
    if rows > 16:
        gate_value = vq_matmul(gate, x)
        up_value = vq_matmul(up, x)
        return mx.sigmoid(gate_value) * gate_value * up_value

    result = _VQ_SWIGLU_KERNEL(
        inputs=[
            gate.indices_packed,
            gate.state_packed,
            gate.aux_packed,
            gate.anchors,
            gate.codebooks,
            gate.scale_lut,
            gate.state_to_codebank,
            gate.bank_ids,
            gate.parameters,
            up.indices_packed,
            up.state_packed,
            up.aux_packed,
            up.anchors,
            up.codebooks,
            up.scale_lut,
            up.state_to_codebank,
            up.bank_ids,
            up.parameters,
            source,
        ],
        template=[
            ("T", source.dtype),
            ("M", rows),
            ("TILE_M", rows),
            ("OUT", gate.out),
            ("K", gate.neuron_len),
            ("GS", gate.groupsize),
            ("NG", gate.groups),
            ("VECTOR_SIZE", gate.vector_size),
            ("NVEC", gate.vectors),
            ("INDEX_BITS", gate.index_bits),
            ("STATE_BITS", gate.state_bits),
            ("STATES", gate.states),
            ("ENTRIES", gate.entries),
            ("CODE_BANKS", gate.code_banks),
            ("AUX_MODE", gate.aux_mode),
            ("CODE_BANK_MODE", gate.code_bank_mode),
            ("HAS_TABLE_BANKS", int(gate.table_banks > 1)),
            ("GROUPS_PER_SUPER", gate.groups_per_super),
            ("NSUPER", gate.supergroups),
            ("NSIGN", math.ceil(gate.neuron_len / 8)),
        ],
        grid=(gate.out * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, gate.out)],
        output_dtypes=[source.dtype],
    )[0]
    return result.reshape((*prefix, *gate.output_shape))


__all__ = [
    "MetalVqWeight",
    "VqTensor",
    "signed_hadamard",
    "vq_dequantize",
    "vq_dequantize_matmul",
    "vq_embedding",
    "vq_gemm",
    "vq_gemv",
    "vq_matmul",
    "vq_mmq",
    "vq_swiglu",
    "vq_swiglu_compatible",
]
