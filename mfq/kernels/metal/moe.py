"""Single-dispatch heterogeneous routed matmul for Apple silicon.

Each global expert owns one fixed-width descriptor.  The descriptor points
into concatenated packed NINT, VQ-family, or MXFP4 streams, so one Metal
dispatch can execute routes spanning different precision cohorts without
first evaluating every expert in every cohort.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats.moe import NintMoeTensor
from mfq.formats.mx import MXFP4_DTYPE, MxTensor
from mfq.formats.nepq import NepqTensor
from mfq.formats.nint8_zero import Nint8ZeroTensor
from mfq.formats.npq0_l import Npq0LTensor
from mfq.formats.npq0_s import Npq0STensor
from mfq.formats.nvq import NvqJscTensor, NvqTensor
from mfq.formats.nvq1_l import Nvq1LTensor
from mfq.formats.nvq1_s import Nvq1STensor
from mfq.kernels.metal.mx import MetalMxWeight
from mfq.kernels.metal.nint import MetalNintWeight
from mfq.kernels.metal.nint8_zero import MetalNint8ZeroWeight
from mfq.kernels.metal.vq import _BITSTREAM_HEADER, MetalVqWeight, signed_hadamard
from mfq.quantize.nint_quant import NintTensor

_VQ_TYPES = (
    NvqTensor,
    NvqJscTensor,
    Nvq1LTensor,
    Nvq1STensor,
    Npq0LTensor,
    Npq0STensor,
    NepqTensor,
)

_FAMILY_NINT = 0
_FAMILY_VQ = 1
_FAMILY_NINT8_ZERO = 2
_FAMILY_MXFP4 = 3
_DESCRIPTOR_SIZE = 32

# Common descriptor fields.
_FAMILY = 0
_LOCAL_EXPERT = 1
_OUT = 2
_K = 3

# NINT descriptor fields.
_NINT_BITS = 4
_NINT_GS = 5
_NINT_NG = 6
_NINT_Q_OFFSET = 7
_NINT_SUB_OFFSET = 8
_NINT_ANCHOR_OFFSET = 9
_NINT_Q5_EXEC = 10

# NINT8-0 descriptor fields.
_Q8_NG = 4
_Q8_Q_OFFSET = 5
_Q8_SCALE_OFFSET = 6

# Native MXFP4 descriptor fields. Values are packed low-nibble first and one
# E8M0 scale is stored per 32 input columns and flattened output row.
_MX_NG = 4
_MX_VALUE_OFFSET = 5
_MX_SCALE_OFFSET = 6

# VQ-family descriptor fields.
_VQ_GS = 4
_VQ_NG = 5
_VQ_VECTOR_SIZE = 6
_VQ_NVEC = 7
_VQ_INDEX_BITS = 8
_VQ_STATE_BITS = 9
_VQ_STATES = 10
_VQ_ENTRIES = 11
_VQ_CODE_BANKS = 12
_VQ_AUX_MODE = 13
_VQ_CODE_BANK_MODE = 14
_VQ_HAS_TABLE_BANKS = 15
_VQ_GROUPS_PER_SUPER = 16
_VQ_NSUPER = 17
_VQ_INDICES_OFFSET = 18
_VQ_STATE_OFFSET = 19
_VQ_AUX_OFFSET = 20
_VQ_ANCHOR_OFFSET = 21
_VQ_CODEBOOK_OFFSET = 22
_VQ_SCALE_OFFSET = 23
_VQ_STATE_BANK_OFFSET = 24
_VQ_BANK_OFFSET = 25
_VQ_PARAMETER_OFFSET = 26
_VQ_ROTATION_VARIANT = 27


class UnsupportedGroupedMoeError(ValueError):
    """The tensor is valid, but cannot safely use the grouped Metal kernel."""


_GROUPED_HEADER = (
    _BITSTREAM_HEADER
    + r"""
template <typename StreamPtr>
inline uint mfq_grouped_nint_read_bits(
    StreamPtr stream,
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

template <typename StreamPtr>
inline uint mfq_grouped_nint_read_value(
    StreamPtr stream,
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
    return mfq_grouped_nint_read_bits(stream, value_index, bits);
}

template <typename StreamPtr>
inline uint mfq_grouped_nint_read_octet(
    StreamPtr stream,
    uint value_index,
    uint bits
) {
    // Eight 2/3/4-bit values occupy exactly bits bytes.  Loading that packet
    // once avoids repeating byte-address and cross-byte work for every value
    // when an MMA weight tile is materialized in threadgroup memory.
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3) * bits + (residual_bits >> 3);
    uint packed = 0u;
    for (uint byte = 0u; byte < bits; ++byte) {
        packed |= uint(stream[byte_index + byte]) << (byte * 8u);
    }
    return packed;
}

inline float mfq_grouped_mxfp4_value(uchar code) {
    uint magnitude = uint(code & 7u);
    float value = magnitude == 0u ? 0.0f
        : (magnitude == 1u ? 0.5f
        : (magnitude == 2u ? 1.0f
        : (magnitude == 3u ? 1.5f
        : (magnitude == 4u ? 2.0f
        : (magnitude == 5u ? 3.0f
        : (magnitude == 6u ? 4.0f : 6.0f))))));
    return (code & 8u) == 0u ? value : -value;
}

inline float mfq_grouped_e8m0(uchar raw) {
    if (raw == 255u) {
        return NAN;
    }
    uint bits = raw == 0u ? 0x00400000u : uint(raw) << 23u;
    return as_type<float>(bits);
}

template <
    typename DescriptorPtr,
    typename NintPtr,
    typename NintSubScalePtr,
    typename NintSubMinPtr,
    typename NintAnchorScalePtr,
    typename NintAnchorMinPtr,
    typename Q8Ptr,
    typename Q8ScalePtr,
    typename VqIndicesPtr,
    typename VqStatePtr,
    typename VqAuxPtr,
    typename VqAnchorsPtr,
    typename VqCodebooksPtr,
    typename VqScalesPtr,
    typename VqStateBankPtr,
    typename VqBanksPtr,
    typename VqParametersPtr,
    typename MxValuePtr,
    typename MxScalePtr
>
inline float mfq_grouped_decode_weight(
    DescriptorPtr descriptors,
    NintPtr nint_q,
    NintSubScalePtr nint_sub_scale,
    NintSubMinPtr nint_sub_min,
    NintAnchorScalePtr nint_anchor_scale,
    NintAnchorMinPtr nint_anchor_min,
    Q8Ptr q8_q,
    Q8ScalePtr q8_scales,
    VqIndicesPtr vq_indices,
    VqStatePtr vq_state,
    VqAuxPtr vq_aux,
    VqAnchorsPtr vq_anchors,
    VqCodebooksPtr vq_codebooks,
    VqScalesPtr vq_scales,
    VqStateBankPtr vq_state_to_codebank,
    VqBanksPtr vq_banks,
    VqParametersPtr vq_parameters,
    MxValuePtr mx_values,
    MxScalePtr mx_scales,
    uint descriptor_base,
    uint pool_output,
    uint column,
    uint K
) {
    uint family = uint(descriptors[descriptor_base]);
    if (family == 0u) {
        uint bits = uint(descriptors[descriptor_base + 4u]);
        uint groupsize = uint(descriptors[descriptor_base + 5u]);
        uint groups = uint(descriptors[descriptor_base + 6u]);
        uint group = column / groupsize;
        uint element = column - group * groupsize;
        uint metadata_index = pool_output * groups + group;
        uint quantized_index = metadata_index * groupsize + element;
        uint quantized = mfq_grouped_nint_read_value(
            nint_q + uint(descriptors[descriptor_base + 7u]),
            quantized_index,
            bits,
            groupsize,
            uint(descriptors[descriptor_base + 10u])
        );
        float scale =
            nint_anchor_scale[
                uint(descriptors[descriptor_base + 9u]) + pool_output
            ] * float(nint_sub_scale[
                uint(descriptors[descriptor_base + 8u]) + metadata_index
            ]);
        float minimum =
            nint_anchor_min[
                uint(descriptors[descriptor_base + 9u]) + pool_output
            ] * float(nint_sub_min[
                uint(descriptors[descriptor_base + 8u]) + metadata_index
            ]);
        return scale * float(quantized) - minimum;
    }
    if (family == 2u) {
        uint groups = uint(descriptors[descriptor_base + 4u]);
        uint group = column >> 5;
        return float(q8_scales[
            uint(descriptors[descriptor_base + 6u])
                + pool_output * groups + group
        ]) * float(q8_q[
            uint(descriptors[descriptor_base + 5u])
                + pool_output * K + column
        ]);
    }
    if (family == 3u) {
        uint groups = uint(descriptors[descriptor_base + 4u]);
        uint packed = uint(mx_values[
            uint(descriptors[descriptor_base + 5u])
                + pool_output * (K >> 1u) + (column >> 1u)
        ]);
        uint code = (column & 1u) == 0u ? packed & 15u : packed >> 4u;
        uchar raw_scale = mx_scales[
            uint(descriptors[descriptor_base + 6u])
                + pool_output * groups + (column >> 5u)
        ];
        return mfq_grouped_mxfp4_value(uchar(code))
            * mfq_grouped_e8m0(raw_scale);
    }

    uint groupsize = uint(descriptors[descriptor_base + 4u]);
    uint groups = uint(descriptors[descriptor_base + 5u]);
    uint vector_size = uint(descriptors[descriptor_base + 6u]);
    return mfq_vq_decode_weight(
        vq_indices + uint(descriptors[descriptor_base + 18u]),
        vq_state + uint(descriptors[descriptor_base + 19u]),
        vq_aux + uint(descriptors[descriptor_base + 20u]),
        vq_anchors + uint(descriptors[descriptor_base + 21u]),
        vq_codebooks + uint(descriptors[descriptor_base + 22u]),
        vq_scales + uint(descriptors[descriptor_base + 23u]),
        vq_state_to_codebank + uint(descriptors[descriptor_base + 24u]),
        vq_banks + uint(descriptors[descriptor_base + 25u]),
        vq_parameters + uint(descriptors[descriptor_base + 26u]),
        pool_output,
        column,
        groupsize,
        groups,
        vector_size,
        uint(descriptors[descriptor_base + 7u]),
        uint(descriptors[descriptor_base + 8u]),
        uint(descriptors[descriptor_base + 9u]),
        uint(descriptors[descriptor_base + 10u]),
        uint(descriptors[descriptor_base + 11u]),
        uint(descriptors[descriptor_base + 12u]),
        uint(descriptors[descriptor_base + 13u]),
        uint(descriptors[descriptor_base + 14u]),
        uint(descriptors[descriptor_base + 15u]),
        uint(descriptors[descriptor_base + 16u]),
        uint(descriptors[descriptor_base + 17u]),
        (K + 7u) / 8u
    );
}
"""
)


_GROUPED_SOURCE = r"""
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint OUTPUT_TILES = (uint(OUT) + ROWS_PER_TG - 1u) / ROWS_PER_TG;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint output_tile = workgroup % OUTPUT_TILES;
    uint projection_index = workgroup / OUTPUT_TILES;
    uint projection = projection_index % uint(PROJECTIONS);
    uint route_index = projection_index / uint(PROJECTIONS);
    uint route = route_index % uint(ROUTES);
    uint token = route_index / uint(ROUTES);
    if (token >= uint(TOKENS)) {
        return;
    }
    uint output_base =
        output_tile * ROWS_PER_TG + simd_group * ROWS_PER_SIMD;

    int expert = int(expert_ids[token * uint(ROUTES) + route]);
    if (expert < 0 || expert >= int(EXPERTS)) {
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            uint output = output_base + row;
            if (lane == 0u && output < uint(OUT)) {
                y[
                    (
                        (token * uint(ROUTES) + route) * uint(PROJECTIONS)
                        + projection
                    ) * uint(OUT) + output
                ] = T(0.0f);
            }
        }
        return;
    }

    uint descriptor_base = (
        uint(expert) * uint(PROJECTIONS) + projection
    ) * uint(DESCRIPTOR_SIZE);
    uint family = uint(descriptors[descriptor_base]);
    uint local_expert = uint(descriptors[descriptor_base + 1u]);
    uint rotation_variant = uint(descriptors[descriptor_base + 27u]);
    uint x_offset = (
        rotation_variant * uint(TOKENS * ROUTES)
        + token * uint(ROUTES) + route
    ) * uint(K);
    float accumulators[ROWS_PER_SIMD] = {0.0f};

    if (family == 0u) {
        uint bits = uint(descriptors[descriptor_base + 4u]);
        uint groupsize = uint(descriptors[descriptor_base + 5u]);
        uint groups = uint(descriptors[descriptor_base + 6u]);
        uint q_offset = uint(descriptors[descriptor_base + 7u]);
        uint sub_offset = uint(descriptors[descriptor_base + 8u]);
        uint anchor_offset = uint(descriptors[descriptor_base + 9u]);
        uint q5_exec = uint(descriptors[descriptor_base + 10u]);

        for (uint group = lane; group < groups; group += 32u) {
            uint outputs[ROWS_PER_SIMD];
            float scales[ROWS_PER_SIMD];
            float minimums[ROWS_PER_SIMD];
            for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                uint output = min(output_base + row, uint(OUT) - 1u);
                uint pool_output = local_expert * uint(OUT) + output;
                uint metadata_index = pool_output * groups + group;
                outputs[row] = pool_output;
                scales[row] =
                    nint_anchor_scale[anchor_offset + pool_output]
                    * float(nint_sub_scale[sub_offset + metadata_index]);
                minimums[row] =
                    nint_anchor_min[anchor_offset + pool_output]
                    * float(nint_sub_min[sub_offset + metadata_index]);
            }

            if (bits == 2u && (groupsize % 4u) == 0u) {
                for (uint element = 0u; element < groupsize; element += 4u) {
                    uint column_base = group * groupsize + element;
                    float activations[4];
                    for (uint component = 0u; component < 4u; ++component) {
                        uint column = column_base + component;
                        activations[component] = column < uint(K)
                            ? float(x[x_offset + column]) : 0.0f;
                    }
                    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                        uint quantized_index =
                            (outputs[row] * groups + group) * groupsize
                            + element;
                        uint packed =
                            uint(nint_q[q_offset + (quantized_index >> 2)]);
                        for (
                            uint component = 0u;
                            component < 4u;
                            ++component
                        ) {
                            uint quantized =
                                (packed >> (component * 2u)) & 3u;
                            accumulators[row] += activations[component] * (
                                scales[row] * float(quantized) - minimums[row]);
                        }
                    }
                }
            } else if (bits == 3u && (groupsize % 8u) == 0u) {
                for (uint element = 0u; element < groupsize; element += 8u) {
                    uint column_base = group * groupsize + element;
                    float activations[8];
                    for (uint component = 0u; component < 8u; ++component) {
                        uint column = column_base + component;
                        activations[component] = column < uint(K)
                            ? float(x[x_offset + column]) : 0.0f;
                    }
                    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                        uint quantized_index =
                            (outputs[row] * groups + group) * groupsize
                            + element;
                        uint byte_index =
                            (quantized_index >> 3) * 3u
                            + (((quantized_index & 7u) * 3u) >> 3);
                        uint packed = uint(nint_q[q_offset + byte_index])
                            | (uint(nint_q[q_offset + byte_index + 1u]) << 8)
                            | (uint(nint_q[q_offset + byte_index + 2u]) << 16);
                        for (
                            uint component = 0u;
                            component < 8u;
                            ++component
                        ) {
                            uint quantized =
                                (packed >> (component * 3u)) & 7u;
                            accumulators[row] += activations[component] * (
                                scales[row] * float(quantized) - minimums[row]);
                        }
                    }
                }
            } else if (bits == 4u && (groupsize % 2u) == 0u) {
                for (uint element = 0u; element < groupsize; element += 2u) {
                    uint column = group * groupsize + element;
                    float activation0 = column < uint(K)
                        ? float(x[x_offset + column]) : 0.0f;
                    float activation1 = column + 1u < uint(K)
                        ? float(x[x_offset + column + 1u]) : 0.0f;
                    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                        uint quantized_index =
                            (outputs[row] * groups + group) * groupsize
                            + element;
                        uint packed =
                            uint(nint_q[q_offset + (quantized_index >> 1)]);
                        accumulators[row] += activation0 * (
                            scales[row] * float(packed & 15u) - minimums[row]);
                        accumulators[row] += activation1 * (
                            scales[row] * float(packed >> 4) - minimums[row]);
                    }
                }
            } else if (bits == 5u && q5_exec != 0u) {
                uint low_bytes = (groupsize + 1u) >> 1;
                uint exec_bytes =
                    low_bytes + ((groupsize + 7u) >> 3);
                for (uint element = 0u; element < groupsize; element += 8u) {
                    float activations[8];
                    for (uint component = 0u; component < 8u; ++component) {
                        uint column = group * groupsize + element + component;
                        activations[component] =
                            element + component < groupsize && column < uint(K)
                            ? float(x[x_offset + column]) : 0.0f;
                    }
                    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                        uint metadata_index =
                            outputs[row] * groups + group;
                        uint group_offset =
                            q_offset + metadata_index * exec_bytes;
                        uint high = uint(nint_q[
                            group_offset + low_bytes + (element >> 3)
                        ]);
                        for (
                            uint component = 0u;
                            component < 8u;
                            ++component
                        ) {
                            if (element + component >= groupsize) {
                                break;
                            }
                            uint low_packed = uint(nint_q[
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
                                scales[row] * float(quantized) - minimums[row]);
                        }
                    }
                }
            } else if (bits == 6u && (groupsize % 4u) == 0u) {
                for (uint element = 0u; element < groupsize; element += 4u) {
                    uint column_base = group * groupsize + element;
                    float activations[4];
                    for (uint component = 0u; component < 4u; ++component) {
                        uint column = column_base + component;
                        activations[component] = column < uint(K)
                            ? float(x[x_offset + column]) : 0.0f;
                    }
                    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                        uint quantized_index =
                            (outputs[row] * groups + group) * groupsize
                            + element;
                        uint byte_index =
                            (quantized_index >> 3) * 6u
                            + (((quantized_index & 7u) * 6u) >> 3);
                        uint packed = uint(nint_q[q_offset + byte_index])
                            | (uint(nint_q[q_offset + byte_index + 1u]) << 8)
                            | (uint(nint_q[q_offset + byte_index + 2u]) << 16);
                        for (
                            uint component = 0u;
                            component < 4u;
                            ++component
                        ) {
                            uint quantized =
                                (packed >> (component * 6u)) & 63u;
                            accumulators[row] += activations[component] * (
                                scales[row] * float(quantized) - minimums[row]);
                        }
                    }
                }
            } else if (bits == 8u) {
                for (uint element = 0u; element < groupsize; ++element) {
                    uint column = group * groupsize + element;
                    float activation = column < uint(K)
                        ? float(x[x_offset + column]) : 0.0f;
                    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                        uint quantized_index =
                            (outputs[row] * groups + group) * groupsize
                            + element;
                        uint quantized =
                            uint(nint_q[q_offset + quantized_index]);
                        accumulators[row] += activation * (
                            scales[row] * float(quantized) - minimums[row]);
                    }
                }
            } else {
                for (uint element = 0u; element < groupsize; ++element) {
                    uint column = group * groupsize + element;
                    float activation = column < uint(K)
                        ? float(x[x_offset + column]) : 0.0f;
                    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                        uint quantized_index =
                            (outputs[row] * groups + group) * groupsize
                            + element;
                        uint quantized = mfq_grouped_nint_read_value(
                            nint_q + q_offset,
                            quantized_index,
                            bits,
                            groupsize,
                            q5_exec);
                        accumulators[row] += activation * (
                            scales[row] * float(quantized) - minimums[row]);
                    }
                }
            }
        }
    } else if (family == 2u) {
        uint groups = uint(descriptors[descriptor_base + 4u]);
        uint q_offset = uint(descriptors[descriptor_base + 5u]);
        uint scale_offset = uint(descriptors[descriptor_base + 6u]);
        for (uint group = lane; group < groups; group += 32u) {
            uint column_base = group * 32u;
            uint outputs[ROWS_PER_SIMD];
            float scales[ROWS_PER_SIMD];
            for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                uint output = min(output_base + row, uint(OUT) - 1u);
                uint pool_output = local_expert * uint(OUT) + output;
                outputs[row] = pool_output;
                scales[row] = float(
                    q8_scales[scale_offset + pool_output * groups + group]
                );
            }
            for (uint component = 0u; component < 32u; ++component) {
                uint column = column_base + component;
                float activation = column < uint(K)
                    ? float(x[x_offset + column]) : 0.0f;
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    uint q_index =
                        (outputs[row] * groups + group) * 32u + component;
                    accumulators[row] += activation * scales[row]
                        * float(q8_q[q_offset + q_index]);
                }
            }
        }
    } else if (family == 3u) {
        uint groups = uint(descriptors[descriptor_base + 4u]);
        uint value_offset = uint(descriptors[descriptor_base + 5u]);
        uint scale_offset = uint(descriptors[descriptor_base + 6u]);
        for (uint group = lane; group < groups; group += 32u) {
            uint column_base = group * 32u;
            uint outputs[ROWS_PER_SIMD];
            float scales[ROWS_PER_SIMD];
            for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                uint output = min(output_base + row, uint(OUT) - 1u);
                uint pool_output = local_expert * uint(OUT) + output;
                outputs[row] = pool_output;
                scales[row] = mfq_grouped_e8m0(
                    mx_scales[scale_offset + pool_output * groups + group]
                );
            }
            for (uint component = 0u; component < 32u; component += 2u) {
                uint column = column_base + component;
                float activation0 = column < uint(K)
                    ? float(x[x_offset + column]) : 0.0f;
                float activation1 = column + 1u < uint(K)
                    ? float(x[x_offset + column + 1u]) : 0.0f;
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    uint packed_index = value_offset
                        + outputs[row] * (uint(K) >> 1u)
                        + (column >> 1u);
                    uchar packed = mx_values[packed_index];
                    accumulators[row] += activation0 * scales[row]
                        * mfq_grouped_mxfp4_value(packed & 15u);
                    accumulators[row] += activation1 * scales[row]
                        * mfq_grouped_mxfp4_value(packed >> 4u);
                }
            }
        }
    } else {
        uint groupsize = uint(descriptors[descriptor_base + 4u]);
        uint groups = uint(descriptors[descriptor_base + 5u]);
        uint vector_size = uint(descriptors[descriptor_base + 6u]);
        uint vectors = uint(descriptors[descriptor_base + 7u]);
        uint index_bits = uint(descriptors[descriptor_base + 8u]);
        uint state_bits = uint(descriptors[descriptor_base + 9u]);
        uint states = uint(descriptors[descriptor_base + 10u]);
        uint entries = uint(descriptors[descriptor_base + 11u]);
        uint code_banks = uint(descriptors[descriptor_base + 12u]);
        uint aux_mode = uint(descriptors[descriptor_base + 13u]);
        uint code_bank_mode = uint(descriptors[descriptor_base + 14u]);
        uint has_table_banks = uint(descriptors[descriptor_base + 15u]);
        uint groups_per_super = uint(descriptors[descriptor_base + 16u]);
        uint supergroups = uint(descriptors[descriptor_base + 17u]);
        uint indices_offset = uint(descriptors[descriptor_base + 18u]);
        uint state_offset = uint(descriptors[descriptor_base + 19u]);
        uint aux_offset = uint(descriptors[descriptor_base + 20u]);
        uint anchor_offset = uint(descriptors[descriptor_base + 21u]);
        uint codebook_offset = uint(descriptors[descriptor_base + 22u]);
        uint scale_offset = uint(descriptors[descriptor_base + 23u]);
        uint state_bank_offset = uint(descriptors[descriptor_base + 24u]);
        uint bank_offset = uint(descriptors[descriptor_base + 25u]);
        uint parameter_offset = uint(descriptors[descriptor_base + 26u]);
        uint vectors_per_group =
            (groupsize + vector_size - 1u) / vector_size;
        uint signs = (uint(K) + 7u) / 8u;

        for (uint group = lane; group < groups; group += 32u) {
            uint outputs[ROWS_PER_SIMD];
            uint table_banks[ROWS_PER_SIMD];
            uint selected_code_banks[ROWS_PER_SIMD];
            uint delta_values[ROWS_PER_SIMD];
            float weight_scales[ROWS_PER_SIMD];
            for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                uint output = min(output_base + row, uint(OUT) - 1u);
                uint pool_output = local_expert * uint(OUT) + output;
                uint state_index = pool_output * groups + group;
                uint state = mfq_read_bits(
                    vq_state + state_offset,
                    state_index,
                    state_bits);
                uint table_bank = has_table_banks != 0u
                    ? uint(vq_banks[
                        bank_offset + pool_output * supergroups
                        + group / groups_per_super
                    ]) : 0u;
                uint delta_value = aux_mode == 3u
                    ? mfq_read_bits(
                        vq_aux + aux_offset,
                        state_index,
                        1u)
                    : 0u;
                uint selected_code_bank = 0u;
                if (code_bank_mode == 1u) {
                    selected_code_bank =
                        uint(vq_state_to_codebank[state_bank_offset + state]);
                } else if (code_bank_mode == 2u) {
                    selected_code_bank = delta_value;
                }
                outputs[row] = pool_output;
                table_banks[row] = table_bank;
                selected_code_banks[row] = selected_code_bank;
                delta_values[row] = delta_value;
                weight_scales[row] =
                    vq_anchors[anchor_offset + pool_output]
                    * vq_scales[scale_offset + table_bank * states + state];
            }

            for (
                uint local_vector = 0u;
                local_vector < vectors_per_group;
                ++local_vector
            ) {
                uint column_base =
                    group * groupsize + local_vector * vector_size;
                if (column_base >= uint(K)) {
                    break;
                }
                float activations[8];
                for (uint component = 0u; component < 8u; ++component) {
                    uint column = column_base + component;
                    activations[component] =
                        component < vector_size && column < uint(K)
                        ? float(x[x_offset + column]) : 0.0f;
                }
                uint vector = column_base / vector_size;
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    uint index = mfq_read_bits(
                        vq_indices + indices_offset,
                        outputs[row] * vectors + vector,
                        index_bits);
                    uint aux_value = 0u;
                    if (aux_mode == 1u || aux_mode == 2u) {
                        aux_value = mfq_read_bits(
                            vq_aux + aux_offset,
                            outputs[row] * signs + column_base / 8u,
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
                        uint column = column_base + component;
                        if (column >= uint(K)) {
                            break;
                        }
                        uint code_offset = (
                            (
                                (
                                    table_banks[row] * code_banks
                                    + selected_code_banks[row]
                                )
                                * entries + index
                            )
                            * vector_size + component
                        );
                        float code = float(
                            vq_codebooks[codebook_offset + code_offset]);
                        if (aux_mode == 1u || aux_mode == 2u) {
                            uint sign_position = column & 7u;
                            uint negative = sign_position < 7u
                                ? ((aux_value >> sign_position) & 1u)
                                : (popcount(aux_value) & 1u);
                            if (
                                aux_mode == 2u
                                && sign_position == 7u
                            ) {
                                negative ^= (index >> 7u) & 1u;
                            }
                            code = negative != 0u ? -code : code;
                        } else if (aux_mode == 3u) {
                            float delta =
                                vq_parameters[parameter_offset];
                            code += delta_values[row] != 0u
                                ? -delta : delta;
                        }
                        accumulators[row] +=
                            activations[component]
                            * weight_scales[row]
                            * code;
                    }
                }
            }
        }
    }

    for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
        float total = simd_sum(accumulators[row]);
        uint output = output_base + row;
        if (lane == 0u && output < uint(OUT)) {
            y[
                (
                    (token * uint(ROUTES) + route) * uint(PROJECTIONS)
                    + projection
                ) * uint(OUT) + output
            ] = T(total);
        }
    }
"""

_GROUPED_COMPACT_SOURCE = r"""
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint K_LANES = 8u;
    constexpr uint OUTPUTS_PER_SIMD = 32u / K_LANES;
    constexpr uint OUTPUTS_PER_TG = SIMD_GROUPS * OUTPUTS_PER_SIMD;
    constexpr uint ROUTES_PER_TILE = 4u;
    constexpr uint OUTPUT_TILES =
        (uint(OUT) + OUTPUTS_PER_TG - 1u) / OUTPUTS_PER_TG;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_output = lane / K_LANES;
    uint workgroup = threadgroup_position_in_grid.x;
    uint output_tile = workgroup % OUTPUT_TILES;
    uint projection_index = workgroup / OUTPUT_TILES;
    uint projection = projection_index % uint(PROJECTIONS);
    uint route_tile = projection_index / uint(PROJECTIONS);
    uint first_route = route_tile * ROUTES_PER_TILE;
    uint output_index =
        output_tile * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD
        + simd_output;
    uint output = min(output_index, uint(OUT) - 1u);

    float accumulators[ROUTES_PER_TILE] = {0.0f};

    // Sorted routes usually place several requests for one expert in this
    // four-row tile. Decode each distinct expert/output weight only once and
    // apply it to every matching activation row.
    for (uint candidate = 0u; candidate < ROUTES_PER_TILE; ++candidate) {
        uint candidate_row = first_route + candidate;
        if (candidate_row >= uint(ROUTE_COUNT)) {
            break;
        }
        int expert = int(expert_ids[candidate_row]);
        bool already_seen = false;
        for (uint previous = 0u; previous < candidate; ++previous) {
            already_seen = already_seen
                || int(expert_ids[first_route + previous]) == expert;
        }
        if (
            already_seen
            || expert < 0
            || expert >= int(EXPERTS)
        ) {
            continue;
        }

        uint descriptor_base = (
            uint(expert) * uint(PROJECTIONS) + projection
        ) * uint(DESCRIPTOR_SIZE);
        uint family = uint(descriptors[descriptor_base]);
        uint local_expert = uint(descriptors[descriptor_base + 1u]);
        uint rotation_variant = uint(descriptors[descriptor_base + 27u]);
        uint pool_output = local_expert * uint(OUT) + output;
        uint original_rows[ROUTES_PER_TILE];
        bool selected_inputs[ROUTES_PER_TILE];
        for (uint input = 0u; input < ROUTES_PER_TILE; ++input) {
            uint row = first_route + input;
            bool selected = row < uint(ROUTE_COUNT)
                && int(expert_ids[row]) == expert;
            selected_inputs[input] = selected;
            original_rows[input] = selected
                ? uint(route_positions[row]) : 0u;
        }

        if (family == 0u) {
            uint bits = uint(descriptors[descriptor_base + 4u]);
            uint groupsize = uint(descriptors[descriptor_base + 5u]);
            uint groups = uint(descriptors[descriptor_base + 6u]);
            uint q_offset = uint(descriptors[descriptor_base + 7u]);
            uint sub_offset = uint(descriptors[descriptor_base + 8u]);
            uint anchor_offset = uint(descriptors[descriptor_base + 9u]);
            uint q5_exec = uint(descriptors[descriptor_base + 10u]);

            for (uint group = k_lane; group < groups; group += K_LANES) {
                uint metadata_index = pool_output * groups + group;
                float scale =
                    nint_anchor_scale[anchor_offset + pool_output]
                    * float(nint_sub_scale[sub_offset + metadata_index]);
                float minimum =
                    nint_anchor_min[anchor_offset + pool_output]
                    * float(nint_sub_min[sub_offset + metadata_index]);
                if (bits == 4u && (groupsize % 2u) == 0u) {
                    for (
                        uint element = 0u;
                        element < groupsize;
                        element += 2u
                    ) {
                        uint column = group * groupsize + element;
                        uint quantized_index =
                            metadata_index * groupsize + element;
                        uint packed =
                            uint(nint_q[q_offset + (quantized_index >> 1)]);
                        float weight0 =
                            scale * float(packed & 15u) - minimum;
                        float weight1 =
                            scale * float(packed >> 4) - minimum;
                        for (
                            uint input = 0u;
                            input < ROUTES_PER_TILE;
                            ++input
                        ) {
                            if (selected_inputs[input]) {
                                uint x_index = (
                                    rotation_variant * uint(ROUTE_COUNT)
                                    + original_rows[input]
                                ) * uint(K) + column;
                                if (column < uint(K)) {
                                    accumulators[input] +=
                                        float(x[x_index]) * weight0;
                                }
                                if (column + 1u < uint(K)) {
                                    accumulators[input] +=
                                        float(x[x_index + 1u]) * weight1;
                                }
                            }
                        }
                    }
                } else if (bits == 5u && q5_exec != 0u) {
                    uint low_bytes = (groupsize + 1u) >> 1;
                    uint exec_bytes =
                        low_bytes + ((groupsize + 7u) >> 3);
                    uint group_offset =
                        q_offset + metadata_index * exec_bytes;
                    for (
                        uint element = 0u;
                        element < groupsize;
                        element += 8u
                    ) {
                        uint high = uint(nint_q[
                            group_offset + low_bytes + (element >> 3)
                        ]);
                        for (
                            uint component = 0u;
                            component < 8u;
                            ++component
                        ) {
                            if (element + component >= groupsize) {
                                break;
                            }
                            uint column =
                                group * groupsize + element + component;
                            if (column >= uint(K)) {
                                break;
                            }
                            uint low_packed = uint(nint_q[
                                group_offset
                                + ((element + component) >> 1)
                            ]);
                            uint low = (
                                low_packed
                                >> (((element + component) & 1u) * 4u)
                            ) & 15u;
                            uint quantized =
                                low | (((high >> component) & 1u) << 4u);
                            float weight =
                                scale * float(quantized) - minimum;
                            for (
                                uint input = 0u;
                                input < ROUTES_PER_TILE;
                                ++input
                            ) {
                                if (selected_inputs[input]) {
                                    uint x_index = (
                                        rotation_variant * uint(ROUTE_COUNT)
                                        + original_rows[input]
                                    ) * uint(K) + column;
                                    accumulators[input] +=
                                        float(x[x_index]) * weight;
                                }
                            }
                        }
                    }
                } else {
                    for (
                        uint element = 0u;
                        element < groupsize;
                        ++element
                    ) {
                        uint column = group * groupsize + element;
                        if (column >= uint(K)) {
                            break;
                        }
                        uint quantized_index =
                            metadata_index * groupsize + element;
                        uint quantized = mfq_grouped_nint_read_value(
                            nint_q + q_offset,
                            quantized_index,
                            bits,
                            groupsize,
                            q5_exec);
                        float weight =
                            scale * float(quantized) - minimum;
                        for (
                            uint input = 0u;
                            input < ROUTES_PER_TILE;
                            ++input
                        ) {
                            if (selected_inputs[input]) {
                                uint x_index = (
                                    rotation_variant * uint(ROUTE_COUNT)
                                    + original_rows[input]
                                ) * uint(K) + column;
                                accumulators[input] +=
                                    float(x[x_index]) * weight;
                            }
                        }
                    }
                }
            }
        } else if (family == 2u) {
            uint groups = uint(descriptors[descriptor_base + 4u]);
            uint q_offset = uint(descriptors[descriptor_base + 5u]);
            uint scale_offset = uint(descriptors[descriptor_base + 6u]);
            for (uint group = k_lane; group < groups; group += K_LANES) {
                float scale = float(
                    q8_scales[scale_offset + pool_output * groups + group]
                );
                uint column_base = group * 32u;
                for (uint component = 0u; component < 32u; ++component) {
                    uint column = column_base + component;
                    if (column >= uint(K)) {
                        break;
                    }
                    float weight = scale * float(q8_q[
                        q_offset + (pool_output * groups + group) * 32u
                            + component
                    ]);
                    for (
                        uint input = 0u;
                        input < ROUTES_PER_TILE;
                        ++input
                    ) {
                        if (selected_inputs[input]) {
                            uint x_index = (
                                rotation_variant * uint(ROUTE_COUNT)
                                + original_rows[input]
                            ) * uint(K) + column;
                            accumulators[input] +=
                                float(x[x_index]) * weight;
                        }
                    }
                }
            }
        } else if (family == 3u) {
            uint groups = uint(descriptors[descriptor_base + 4u]);
            uint value_offset = uint(descriptors[descriptor_base + 5u]);
            uint scale_offset = uint(descriptors[descriptor_base + 6u]);
            for (uint group = k_lane; group < groups; group += K_LANES) {
                float scale = mfq_grouped_e8m0(
                    mx_scales[scale_offset + pool_output * groups + group]
                );
                uint column_base = group * 32u;
                for (uint component = 0u; component < 32u; component += 2u) {
                    uint column = column_base + component;
                    uchar packed = mx_values[
                        value_offset + pool_output * (uint(K) >> 1u)
                            + (column >> 1u)
                    ];
                    float weight0 = scale
                        * mfq_grouped_mxfp4_value(packed & 15u);
                    float weight1 = scale
                        * mfq_grouped_mxfp4_value(packed >> 4u);
                    for (uint input = 0u; input < ROUTES_PER_TILE; ++input) {
                        if (selected_inputs[input]) {
                            uint x_index = (
                                rotation_variant * uint(ROUTE_COUNT)
                                + original_rows[input]
                            ) * uint(K) + column;
                            accumulators[input] += float(x[x_index]) * weight0;
                            if (column + 1u < uint(K)) {
                                accumulators[input] += float(x[x_index + 1u]) * weight1;
                            }
                        }
                    }
                }
            }
        } else {
            uint groupsize = uint(descriptors[descriptor_base + 4u]);
            uint groups = uint(descriptors[descriptor_base + 5u]);
            uint vector_size = uint(descriptors[descriptor_base + 6u]);
            uint vectors = uint(descriptors[descriptor_base + 7u]);
            uint index_bits = uint(descriptors[descriptor_base + 8u]);
            uint state_bits = uint(descriptors[descriptor_base + 9u]);
            uint states = uint(descriptors[descriptor_base + 10u]);
            uint entries = uint(descriptors[descriptor_base + 11u]);
            uint code_banks = uint(descriptors[descriptor_base + 12u]);
            uint aux_mode = uint(descriptors[descriptor_base + 13u]);
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

            for (uint group = k_lane; group < groups; group += K_LANES) {
                uint first_column = group * groupsize;
                uint end_column = min(
                    first_column + groupsize,
                    uint(K));
                for (
                    uint column = first_column;
                    column < end_column;
                    ++column
                ) {
                    float weight = mfq_vq_decode_weight(
                        vq_indices + indices_offset,
                        vq_state + state_offset,
                        vq_aux + aux_offset,
                        vq_anchors + anchor_offset,
                        vq_codebooks + codebook_offset,
                        vq_scales + scale_offset,
                        vq_state_to_codebank + state_bank_offset,
                        vq_banks + bank_offset,
                        vq_parameters + parameter_offset,
                        pool_output,
                        column,
                        groupsize,
                        groups,
                        vector_size,
                        vectors,
                        index_bits,
                        state_bits,
                        states,
                        entries,
                        code_banks,
                        aux_mode,
                        code_bank_mode,
                        has_table_banks,
                        groups_per_super,
                        supergroups,
                        (uint(K) + 7u) / 8u);
                    for (
                        uint input = 0u;
                        input < ROUTES_PER_TILE;
                        ++input
                    ) {
                        if (selected_inputs[input]) {
                            uint x_index = (
                                rotation_variant * uint(ROUTE_COUNT)
                                + original_rows[input]
                            ) * uint(K) + column;
                            accumulators[input] +=
                                float(x[x_index]) * weight;
                        }
                    }
                }
            }
        }
    }

    for (uint input = 0u; input < ROUTES_PER_TILE; ++input) {
        accumulators[input] +=
            simd_shuffle_down(accumulators[input], 4);
        accumulators[input] +=
            simd_shuffle_down(accumulators[input], 2);
        accumulators[input] +=
            simd_shuffle_down(accumulators[input], 1);
        uint row = first_route + input;
        if (
            k_lane == 0u
            && row < uint(ROUTE_COUNT)
            && output_index < uint(OUT)
        ) {
            uint original_row = uint(route_positions[row]);
            y[
                (original_row * uint(PROJECTIONS) + projection) * uint(OUT)
                + output_index
            ] = T(accumulators[input]);
        }
    }
"""


_GROUPED_MMA_SOURCE = r"""
    constexpr uint BM = 32u;
    constexpr uint BN = 64u;
    constexpr uint GPC = 2u;
    constexpr uint BK = 96u;
    constexpr uint BK_PAD = BK + 8u;
    constexpr uint BN_PAD = BN + 8u;
    constexpr uint ROUTE_TILES = (uint(ROUTE_COUNT) + BM - 1u) / BM;
    constexpr uint OUTPUT_TILES = (uint(OUT) + BN - 1u) / BN;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint local_thread = thread_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint output_tile = workgroup % OUTPUT_TILES;
    uint projection_index = workgroup / OUTPUT_TILES;
    uint projection = projection_index % uint(PROJECTIONS);
    uint route_tile = projection_index / uint(PROJECTIONS);
    uint first_route = route_tile * BM;
    uint output_base = output_tile * BN;

    threadgroup half activation_tile[BM * BK_PAD];
    threadgroup half weight_tile[BK * BN_PAD];

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

    uint quadrant = lane / 4u;
    uint fragment_row = (quadrant & 4u) + ((lane / 2u) & 3u);
    uint fragment_col = (quadrant & 2u) * 2u + (lane & 1u) * 2u;
    uint simd_row = (simd_group / 4u) * 16u;
    uint simd_col = (simd_group & 3u) * 16u;

    // A sorted 8-route tile normally contains one expert. Boundary tiles may
    // contain several; masked activation rows let each distinct expert
    // contribute to the same MMA accumulator without a CPU-built task list.
    for (uint candidate = 0u; candidate < BM; ++candidate) {
        uint candidate_route = first_route + candidate;
        if (candidate_route >= uint(ROUTE_COUNT)) {
            break;
        }
        int expert = int(expert_ids[candidate_route]);
        bool already_seen = false;
        for (uint previous = 0u; previous < candidate; ++previous) {
            already_seen = already_seen
                || int(expert_ids[first_route + previous]) == expert;
        }
        if (already_seen || expert < 0 || expert >= int(EXPERTS)) {
            continue;
        }
        uint descriptor_base = (
            uint(expert) * uint(PROJECTIONS) + projection
        ) * uint(DESCRIPTOR_SIZE);
        uint local_expert = uint(descriptors[descriptor_base + 1u]);
        uint rotation_variant = uint(descriptors[descriptor_base + 27u]);

        uint family = uint(descriptors[descriptor_base]);
        uint groupsize = family == 0u
            ? uint(descriptors[descriptor_base + 5u])
            : ((family == 2u || family == 3u)
                ? 32u : uint(descriptors[descriptor_base + 4u]));
        uint groups = family == 0u
            ? uint(descriptors[descriptor_base + 6u])
            : ((family == 2u || family == 3u)
                ? uint(descriptors[descriptor_base + 4u])
                : uint(descriptors[descriptor_base + 5u]));
        uint chunks = (groups + GPC - 1u) / GPC;
        for (uint chunk = 0u; chunk < chunks; ++chunk) {
            uint group_base = chunk * GPC;
            uint chunk_base = group_base * groupsize;
            uint chunk_width = groupsize * GPC;
            for (
                uint index = local_thread;
                index < BM * BK;
                index += 256u
            ) {
                uint local_row = index / BK;
                uint local_column = index - local_row * BK;
                uint sorted_route = first_route + local_row;
                uint column = chunk_base + local_column;
                bool selected = sorted_route < uint(ROUTE_COUNT)
                    && int(expert_ids[sorted_route]) == expert;
                uint original_route = selected
                    ? uint(route_positions[sorted_route])
                    : 0u;
                activation_tile[local_row * BK_PAD + local_column] =
                    selected
                        && local_column < chunk_width
                        && column < uint(K)
                    ? half(x[
                        (
                            rotation_variant * uint(ROUTE_COUNT)
                            + original_route
                        ) * uint(K) + column
                    ])
                    : half(0.0f);
            }

            for (
                uint index = local_thread;
                index < BK * BN;
                index += 256u
            ) {
                uint local_column = index / BN;
                uint local_output = index - local_column * BN;
                weight_tile[
                    local_column * BN_PAD + local_output
                ] = half(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (
                uint task = local_thread;
                task < BN * GPC;
                task += 256u
            ) {
                uint local_output = task / GPC;
                uint local_group = task - local_output * GPC;
                uint group = group_base + local_group;
                uint output = output_base + local_output;
                bool valid = group < groups && output < uint(OUT);
                uint pool_output = local_expert * uint(OUT) + output;
                float scale = 0.0f;
                float minimum = 0.0f;
                uint metadata_index = pool_output * groups + group;
                if (valid && family == 0u) {
                    scale = nint_anchor_scale[
                        uint(descriptors[descriptor_base + 9u])
                            + pool_output
                    ] * float(nint_sub_scale[
                        uint(descriptors[descriptor_base + 8u])
                            + metadata_index
                    ]);
                    minimum = nint_anchor_min[
                        uint(descriptors[descriptor_base + 9u])
                            + pool_output
                    ] * float(nint_sub_min[
                        uint(descriptors[descriptor_base + 8u])
                            + metadata_index
                    ]);
                } else if (valid && family == 2u) {
                    scale = float(q8_scales[
                        uint(descriptors[descriptor_base + 6u])
                            + metadata_index
                    ]);
                }
                uint nint_bits = family == 0u
                    ? uint(descriptors[descriptor_base + 4u])
                    : 0u;
                uint quantized_base = metadata_index * groupsize;
                uint packed_octet = 0u;
                for (
                    uint element = 0u;
                    element < groupsize;
                    ++element
                ) {
                    uint local_column =
                        local_group * groupsize + element;
                    uint column = chunk_base + local_column;
                    float value = 0.0f;
                    if (valid && column < uint(K)) {
                        if (family == 0u) {
                            auto quantized_stream = nint_q + uint(
                                descriptors[descriptor_base + 7u]
                            );
                            uint quantized;
                            if (nint_bits <= 4u) {
                                if ((element & 7u) == 0u) {
                                    packed_octet =
                                        mfq_grouped_nint_read_octet(
                                            quantized_stream,
                                            quantized_base + element,
                                            nint_bits
                                        );
                                }
                                quantized = (
                                    packed_octet
                                    >> ((element & 7u) * nint_bits)
                                ) & ((1u << nint_bits) - 1u);
                            } else {
                                quantized = mfq_grouped_nint_read_value(
                                    quantized_stream,
                                    quantized_base + element,
                                    nint_bits,
                                    groupsize,
                                    uint(descriptors[
                                        descriptor_base + 10u
                                    ])
                                );
                            }
                            value =
                                scale * float(quantized) - minimum;
                        } else if (family == 2u) {
                            value = scale * float(q8_q[
                                uint(descriptors[
                                    descriptor_base + 5u
                                ]) + metadata_index * 32u + element
                            ]);
                        } else {
                            value = mfq_grouped_decode_weight(
                                descriptors,
                                nint_q,
                                nint_sub_scale,
                                nint_sub_min,
                                nint_anchor_scale,
                                nint_anchor_min,
                                q8_q,
                                q8_scales,
                                vq_indices,
                                vq_state,
                                vq_aux,
                                vq_anchors,
                                vq_codebooks,
                                vq_scales,
                                vq_state_to_codebank,
                                vq_banks,
                                vq_parameters,
                                mx_values,
                                mx_scales,
                                descriptor_base,
                                pool_output,
                                column,
                                uint(K)
                            );
                        }
                    }
                    weight_tile[
                        local_column * BN_PAD + local_output
                    ] = half(value);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0u; kk < BK; kk += 8u) {
                metal::simdgroup_matrix<half, 8, 8> a0;
                metal::simdgroup_matrix<half, 8, 8> a1;
                metal::simdgroup_matrix<half, 8, 8> b0;
                metal::simdgroup_matrix<half, 8, 8> b1;
                a0.thread_elements()[0] = activation_tile[
                    (simd_row + fragment_row) * BK_PAD
                        + kk + fragment_col
                ];
                a0.thread_elements()[1] = activation_tile[
                    (simd_row + fragment_row) * BK_PAD
                        + kk + fragment_col + 1u
                ];
                a1.thread_elements()[0] = activation_tile[
                    (simd_row + 8u + fragment_row) * BK_PAD
                        + kk + fragment_col
                ];
                a1.thread_elements()[1] = activation_tile[
                    (simd_row + 8u + fragment_row) * BK_PAD
                        + kk + fragment_col + 1u
                ];
                b0.thread_elements()[0] = weight_tile[
                    (kk + fragment_row) * BN_PAD
                        + simd_col + fragment_col
                ];
                b0.thread_elements()[1] = weight_tile[
                    (kk + fragment_row) * BN_PAD
                        + simd_col + fragment_col + 1u
                ];
                b1.thread_elements()[0] = weight_tile[
                    (kk + fragment_row) * BN_PAD
                        + simd_col + 8u + fragment_col
                ];
                b1.thread_elements()[1] = weight_tile[
                    (kk + fragment_row) * BN_PAD
                        + simd_col + 8u + fragment_col + 1u
                ];
                simdgroup_multiply_accumulate(c00, a0, b0, c00);
                simdgroup_multiply_accumulate(c01, a0, b1, c01);
                simdgroup_multiply_accumulate(c10, a1, b0, c10);
                simdgroup_multiply_accumulate(c11, a1, b1, c11);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint sorted_route0 = first_route + simd_row + fragment_row;
    uint sorted_route1 = sorted_route0 + 8u;
    uint output0 = output_base + simd_col + fragment_col;
    uint output1 = output0 + 8u;
    if (sorted_route0 < uint(ROUTE_COUNT)) {
        uint original_route = uint(route_positions[sorted_route0]);
        if (output0 < uint(OUT)) {
            y[
                (original_route * uint(PROJECTIONS) + projection)
                    * uint(OUT) + output0
            ] = T(c00.thread_elements()[0]);
        }
        if (output0 + 1u < uint(OUT)) {
            y[
                (original_route * uint(PROJECTIONS) + projection)
                    * uint(OUT) + output0 + 1u
            ] = T(c00.thread_elements()[1]);
        }
        if (output1 < uint(OUT)) {
            y[
                (original_route * uint(PROJECTIONS) + projection)
                    * uint(OUT) + output1
            ] = T(c01.thread_elements()[0]);
        }
        if (output1 + 1u < uint(OUT)) {
            y[
                (original_route * uint(PROJECTIONS) + projection)
                    * uint(OUT) + output1 + 1u
            ] = T(c01.thread_elements()[1]);
        }
    }
    if (sorted_route1 < uint(ROUTE_COUNT)) {
        uint original_route = uint(route_positions[sorted_route1]);
        if (output0 < uint(OUT)) {
            y[
                (original_route * uint(PROJECTIONS) + projection)
                    * uint(OUT) + output0
            ] = T(c10.thread_elements()[0]);
        }
        if (output0 + 1u < uint(OUT)) {
            y[
                (original_route * uint(PROJECTIONS) + projection)
                    * uint(OUT) + output0 + 1u
            ] = T(c10.thread_elements()[1]);
        }
        if (output1 < uint(OUT)) {
            y[
                (original_route * uint(PROJECTIONS) + projection)
                    * uint(OUT) + output1
            ] = T(c11.thread_elements()[0]);
        }
        if (output1 + 1u < uint(OUT)) {
            y[
                (original_route * uint(PROJECTIONS) + projection)
                    * uint(OUT) + output1 + 1u
            ] = T(c11.thread_elements()[1]);
        }
    }
"""


_GROUPED_EXPERT_MMA_SOURCE = r"""
    constexpr uint BM = 32u;
    constexpr uint BN = 64u;
    constexpr uint BK = 64u;
    constexpr uint BK_PAD = BK + 8u;
    constexpr uint BN_PAD = BN + 8u;
    constexpr uint OUTPUT_TILES = (uint(OUT) + BN - 1u) / BN;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint local_thread = thread_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint output_tile = workgroup % OUTPUT_TILES;
    uint projection_index = workgroup / OUTPUT_TILES;
    uint projection = projection_index % uint(PROJECTIONS);
    uint expert = projection_index / uint(PROJECTIONS);
    if (expert >= uint(EXPERTS)) {
        return;
    }
    uint output_base = output_tile * BN;
    uint descriptor_base = (
        expert * uint(PROJECTIONS) + projection
    ) * uint(DESCRIPTOR_SIZE);
    uint local_expert = uint(descriptors[descriptor_base + 1u]);
    uint rotation_variant = uint(descriptors[descriptor_base + 27u]);

    threadgroup uint route_bounds[2];
    if (local_thread == 0u) {
        uint low = 0u;
        uint high = uint(ROUTE_COUNT);
        while (low < high) {
            uint middle = (low + high) >> 1u;
            if (int(expert_ids[middle]) < int(expert)) {
                low = middle + 1u;
            } else {
                high = middle;
            }
        }
        route_bounds[0] = low;
        high = uint(ROUTE_COUNT);
        while (low < high) {
            uint middle = (low + high) >> 1u;
            if (int(expert_ids[middle]) <= int(expert)) {
                low = middle + 1u;
            } else {
                high = middle;
            }
        }
        route_bounds[1] = low;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint route_start = route_bounds[0];
    uint route_end = route_bounds[1];
    if (route_start >= route_end) {
        return;
    }

    threadgroup half activation_tile[BM * BK_PAD];
    threadgroup half weight_tile[BK * BN_PAD];
    uint quadrant = lane / 4u;
    uint fragment_row = (quadrant & 4u) + ((lane / 2u) & 3u);
    uint fragment_col = (quadrant & 2u) * 2u + (lane & 1u) * 2u;
    uint simd_row = (simd_group / 4u) * 16u;
    uint simd_col = (simd_group & 3u) * 16u;
    uint chunks = (uint(K) + BK - 1u) / BK;

    for (uint chunk = 0u; chunk < chunks; ++chunk) {
        uint column_base = chunk * BK;
        uint family = uint(descriptors[descriptor_base]);
        uint nint_bits = family == 0u
            ? uint(descriptors[descriptor_base + 4u])
            : 0u;
        uint nint_groupsize = family == 0u
            ? uint(descriptors[descriptor_base + 5u])
            : 1u;
        for (
            uint packet = local_thread;
            packet < (BK / 8u) * BN;
            packet += 256u
        ) {
            uint local_packet = packet / BN;
            uint local_output = packet - local_packet * BN;
            uint local_column = local_packet * 8u;
            uint first_column = column_base + local_column;
            uint output = output_base + local_output;
            uint nint_element_base = first_column % nint_groupsize;
            if (
                family == 0u
                && nint_bits <= 4u
                && nint_element_base + 8u <= nint_groupsize
                && first_column < uint(K)
                && output < uint(OUT)
            ) {
                uint groupsize = nint_groupsize;
                uint groups = uint(descriptors[descriptor_base + 6u]);
                uint group = first_column / groupsize;
                uint element_base = nint_element_base;
                uint pool_output = local_expert * uint(OUT) + output;
                uint metadata_index = pool_output * groups + group;
                float scale = nint_anchor_scale[
                    uint(descriptors[descriptor_base + 9u]) + pool_output
                ] * float(nint_sub_scale[
                    uint(descriptors[descriptor_base + 8u]) + metadata_index
                ]);
                float minimum = nint_anchor_min[
                    uint(descriptors[descriptor_base + 9u]) + pool_output
                ] * float(nint_sub_min[
                    uint(descriptors[descriptor_base + 8u]) + metadata_index
                ]);
                auto quantized_stream = nint_q + uint(
                    descriptors[descriptor_base + 7u]
                );
                uint packed_octet = mfq_grouped_nint_read_octet(
                    quantized_stream,
                    metadata_index * groupsize + element_base,
                    nint_bits
                );
                uint mask = (1u << nint_bits) - 1u;
                for (uint element = 0u; element < 8u; ++element) {
                    uint column = first_column + element;
                    float value = column < uint(K)
                        ? scale * float(
                            (packed_octet >> (element * nint_bits)) & mask
                        ) - minimum
                        : 0.0f;
                    weight_tile[
                        (local_column + element) * BN_PAD + local_output
                    ] = half(value);
                }
            } else {
                for (uint element = 0u; element < 8u; ++element) {
                    uint column = first_column + element;
                    float value = 0.0f;
                    if (column < uint(K) && output < uint(OUT)) {
                        uint pool_output = local_expert * uint(OUT) + output;
                        value = mfq_grouped_decode_weight(
                            descriptors,
                            nint_q,
                            nint_sub_scale,
                            nint_sub_min,
                            nint_anchor_scale,
                            nint_anchor_min,
                            q8_q,
                            q8_scales,
                            vq_indices,
                            vq_state,
                            vq_aux,
                            vq_anchors,
                            vq_codebooks,
                            vq_scales,
                            vq_state_to_codebank,
                            vq_banks,
                            vq_parameters,
                            mx_values,
                            mx_scales,
                            descriptor_base,
                            pool_output,
                            column,
                            uint(K)
                        );
                    }
                    weight_tile[
                        (local_column + element) * BN_PAD + local_output
                    ] = half(value);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (
            uint route_base = route_start;
            route_base < route_end;
            route_base += BM
        ) {
            for (
                uint index = local_thread;
                index < BM * BK;
                index += 256u
            ) {
                uint local_row = index / BK;
                uint local_column = index - local_row * BK;
                uint sorted_route = route_base + local_row;
                uint column = column_base + local_column;
                uint original_route = sorted_route < route_end
                    ? uint(route_positions[sorted_route])
                    : 0u;
                activation_tile[local_row * BK_PAD + local_column] =
                    sorted_route < route_end && column < uint(K)
                    ? half(x[
                        (
                            rotation_variant * uint(ROUTE_COUNT)
                            + original_route
                        ) * uint(K) + column
                    ])
                    : half(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

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
            for (uint kk = 0u; kk < BK; kk += 8u) {
                metal::simdgroup_matrix<half, 8, 8> a0;
                metal::simdgroup_matrix<half, 8, 8> a1;
                metal::simdgroup_matrix<half, 8, 8> b0;
                metal::simdgroup_matrix<half, 8, 8> b1;
                a0.thread_elements()[0] = activation_tile[
                    (simd_row + fragment_row) * BK_PAD
                        + kk + fragment_col
                ];
                a0.thread_elements()[1] = activation_tile[
                    (simd_row + fragment_row) * BK_PAD
                        + kk + fragment_col + 1u
                ];
                a1.thread_elements()[0] = activation_tile[
                    (simd_row + 8u + fragment_row) * BK_PAD
                        + kk + fragment_col
                ];
                a1.thread_elements()[1] = activation_tile[
                    (simd_row + 8u + fragment_row) * BK_PAD
                        + kk + fragment_col + 1u
                ];
                b0.thread_elements()[0] = weight_tile[
                    (kk + fragment_row) * BN_PAD
                        + simd_col + fragment_col
                ];
                b0.thread_elements()[1] = weight_tile[
                    (kk + fragment_row) * BN_PAD
                        + simd_col + fragment_col + 1u
                ];
                b1.thread_elements()[0] = weight_tile[
                    (kk + fragment_row) * BN_PAD
                        + simd_col + 8u + fragment_col
                ];
                b1.thread_elements()[1] = weight_tile[
                    (kk + fragment_row) * BN_PAD
                        + simd_col + 8u + fragment_col + 1u
                ];
                simdgroup_multiply_accumulate(c00, a0, b0, c00);
                simdgroup_multiply_accumulate(c01, a0, b1, c01);
                simdgroup_multiply_accumulate(c10, a1, b0, c10);
                simdgroup_multiply_accumulate(c11, a1, b1, c11);
            }

            uint sorted_route0 = route_base + simd_row + fragment_row;
            uint sorted_route1 = sorted_route0 + 8u;
            uint output0 = output_base + simd_col + fragment_col;
            uint output1 = output0 + 8u;
            if (sorted_route0 < route_end) {
                uint original_route = uint(route_positions[sorted_route0]);
                uint base = (
                    original_route * uint(PROJECTIONS) + projection
                ) * uint(OUT);
                if (output0 < uint(OUT)) {
                    uint index = base + output0;
                    y[index] = chunk == 0u
                        ? c00.thread_elements()[0]
                        : y[index] + c00.thread_elements()[0];
                }
                if (output0 + 1u < uint(OUT)) {
                    uint index = base + output0 + 1u;
                    y[index] = chunk == 0u
                        ? c00.thread_elements()[1]
                        : y[index] + c00.thread_elements()[1];
                }
                if (output1 < uint(OUT)) {
                    uint index = base + output1;
                    y[index] = chunk == 0u
                        ? c01.thread_elements()[0]
                        : y[index] + c01.thread_elements()[0];
                }
                if (output1 + 1u < uint(OUT)) {
                    uint index = base + output1 + 1u;
                    y[index] = chunk == 0u
                        ? c01.thread_elements()[1]
                        : y[index] + c01.thread_elements()[1];
                }
            }
            if (sorted_route1 < route_end) {
                uint original_route = uint(route_positions[sorted_route1]);
                uint base = (
                    original_route * uint(PROJECTIONS) + projection
                ) * uint(OUT);
                if (output0 < uint(OUT)) {
                    uint index = base + output0;
                    y[index] = chunk == 0u
                        ? c10.thread_elements()[0]
                        : y[index] + c10.thread_elements()[0];
                }
                if (output0 + 1u < uint(OUT)) {
                    uint index = base + output0 + 1u;
                    y[index] = chunk == 0u
                        ? c10.thread_elements()[1]
                        : y[index] + c10.thread_elements()[1];
                }
                if (output1 < uint(OUT)) {
                    uint index = base + output1;
                    y[index] = chunk == 0u
                        ? c11.thread_elements()[0]
                        : y[index] + c11.thread_elements()[0];
                }
                if (output1 + 1u < uint(OUT)) {
                    uint index = base + output1 + 1u;
                    y[index] = chunk == 0u
                        ? c11.thread_elements()[1]
                        : y[index] + c11.thread_elements()[1];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


_GROUPED_KERNEL = mx.fast.metal_kernel(
    name="mfq_heterogeneous_grouped_matmul",
    input_names=[
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
        "mx_values",
        "mx_scales",
        "x",
        "expert_ids",
    ],
    output_names=["y"],
    header=_GROUPED_HEADER,
    source=_GROUPED_SOURCE,
    compile_options={"math_mode": "fast"},
)

_GROUPED_COMPACT_KERNEL = mx.fast.metal_kernel(
    name="mfq_heterogeneous_grouped_compact_mmq",
    input_names=[
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
        "mx_values",
        "mx_scales",
        "x",
        "expert_ids",
        "route_positions",
    ],
    output_names=["y"],
    header=_GROUPED_HEADER,
    source=_GROUPED_COMPACT_SOURCE,
    compile_options={"math_mode": "fast"},
)

_GROUPED_MMA_KERNEL = mx.fast.metal_kernel(
    name="mfq_heterogeneous_grouped_compact_mma",
    input_names=[
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
        "mx_values",
        "mx_scales",
        "x",
        "expert_ids",
        "route_positions",
    ],
    output_names=["y"],
    header=_GROUPED_HEADER,
    source=_GROUPED_MMA_SOURCE,
    compile_options={"math_mode": "fast"},
)

_GROUPED_EXPERT_MMA_KERNEL = mx.fast.metal_kernel(
    name="mfq_heterogeneous_grouped_expert_mma",
    input_names=[
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
        "mx_values",
        "mx_scales",
        "x",
        "expert_ids",
        "route_positions",
    ],
    output_names=["y"],
    header=_GROUPED_HEADER,
    source=_GROUPED_EXPERT_MMA_SOURCE,
    compile_options={"math_mode": "fast"},
)


def _size(value: mx.array) -> int:
    return int(value.size)


def _join(
    arrays: list[mx.array],
    *,
    dtype: mx.Dtype,
    padding: int = 0,
) -> mx.array:
    parts: list[mx.array] = []
    for array in arrays:
        parts.append(array.reshape((-1,)).astype(dtype))
        if padding:
            parts.append(mx.zeros((padding,), dtype=dtype))
    if not parts:
        return mx.zeros((max(1, padding),), dtype=dtype)
    return mx.contiguous(mx.concatenate(parts))


@dataclass(frozen=True)
class MetalMoeWeight:
    """Concatenated packed buffers and per-expert heterogeneous descriptors."""

    descriptors: mx.array
    nint_q: mx.array
    nint_sub_scale: mx.array
    nint_sub_min: mx.array
    nint_anchor_scale: mx.array
    nint_anchor_min: mx.array
    q8_q: mx.array
    q8_scales: mx.array
    vq_indices: mx.array
    vq_state: mx.array
    vq_aux: mx.array
    vq_anchors: mx.array
    vq_codebooks: mx.array
    vq_scales: mx.array
    vq_state_to_codebank: mx.array
    vq_banks: mx.array
    vq_parameters: mx.array
    mx_values: mx.array
    mx_scales: mx.array
    descriptor_values: np.ndarray
    rotation_specs: tuple[tuple[mx.array, int, int], ...]
    experts: int
    out_per_expert: int
    neuron_len: int
    projections: int

    @classmethod
    def from_tensor(cls, tensor: NintMoeTensor) -> MetalMoeWeight:
        descriptors = np.zeros(
            (tensor.n_experts, _DESCRIPTOR_SIZE),
            dtype=np.int32,
        )

        nint_q: list[mx.array] = []
        nint_sub_scale: list[mx.array] = []
        nint_sub_min: list[mx.array] = []
        nint_anchor_scale: list[mx.array] = []
        nint_anchor_min: list[mx.array] = []
        q8_q: list[mx.array] = []
        q8_scales: list[mx.array] = []
        vq_indices: list[mx.array] = []
        vq_state: list[mx.array] = []
        vq_aux: list[mx.array] = []
        vq_anchors: list[mx.array] = []
        vq_codebooks: list[mx.array] = []
        vq_scales: list[mx.array] = []
        vq_state_to_codebank: list[mx.array] = []
        vq_banks: list[mx.array] = []
        vq_parameters: list[mx.array] = []
        mx_values: list[mx.array] = []
        mx_scales: list[mx.array] = []

        offsets = {
            "nint_q": 0,
            "nint_sub": 0,
            "nint_anchor": 0,
            "q8_q": 0,
            "q8_scale": 0,
            "vq_indices": 0,
            "vq_state": 0,
            "vq_aux": 0,
            "vq_anchor": 0,
            "vq_codebook": 0,
            "vq_scale": 0,
            "vq_state_bank": 0,
            "vq_bank": 0,
            "vq_parameter": 0,
            "mx_value": 0,
            "mx_scale": 0,
        }
        rotation_variants: dict[tuple[int, int], int] = {}
        rotation_specs: list[tuple[mx.array, int, int]] = []

        for pool in tensor.pools:
            source = pool.tensor
            expert_ids = np.asarray(pool.expert_ids, dtype=np.int32).reshape(-1)
            if isinstance(source, NintTensor):
                weight: MetalNintWeight | MetalVqWeight = MetalNintWeight.from_tensor(source)
                if weight.out != expert_ids.size * tensor.out_per_expert:
                    raise ValueError("NINTM NINT cohort row count is inconsistent")
                common = (
                    _FAMILY_NINT,
                    weight.bits,
                    weight.groupsize,
                    weight.groups,
                    offsets["nint_q"],
                    offsets["nint_sub"],
                    offsets["nint_anchor"],
                    int(weight.q5_exec),
                )
                for local_expert, expert in enumerate(expert_ids):
                    descriptor = descriptors[int(expert)]
                    descriptor[_FAMILY] = common[0]
                    descriptor[_LOCAL_EXPERT] = local_expert
                    descriptor[_OUT] = tensor.out_per_expert
                    descriptor[_K] = tensor.neuron_len
                    descriptor[_NINT_BITS] = common[1]
                    descriptor[_NINT_GS] = common[2]
                    descriptor[_NINT_NG] = common[3]
                    descriptor[_NINT_Q_OFFSET] = common[4]
                    descriptor[_NINT_SUB_OFFSET] = common[5]
                    descriptor[_NINT_ANCHOR_OFFSET] = common[6]
                    descriptor[_NINT_Q5_EXEC] = common[7]

                nint_q.append(weight.q_packed)
                nint_sub_scale.append(weight.sub_scale)
                nint_sub_min.append(weight.sub_min)
                nint_anchor_scale.append(weight.neuron_scale)
                nint_anchor_min.append(weight.neuron_min)
                offsets["nint_q"] += _size(weight.q_packed) + 2
                offsets["nint_sub"] += _size(weight.sub_scale)
                offsets["nint_anchor"] += _size(weight.neuron_scale)
                continue

            if isinstance(source, Nint8ZeroTensor):
                q8_weight = MetalNint8ZeroWeight.from_tensor(source)
                if q8_weight.out != expert_ids.size * tensor.out_per_expert:
                    raise ValueError("NINTM NINT8-0 cohort row count is inconsistent")
                for local_expert, expert in enumerate(expert_ids):
                    descriptor = descriptors[int(expert)]
                    descriptor[_FAMILY] = _FAMILY_NINT8_ZERO
                    descriptor[_LOCAL_EXPERT] = local_expert
                    descriptor[_OUT] = tensor.out_per_expert
                    descriptor[_K] = tensor.neuron_len
                    descriptor[_Q8_NG] = q8_weight.groups
                    descriptor[_Q8_Q_OFFSET] = offsets["q8_q"]
                    descriptor[_Q8_SCALE_OFFSET] = offsets["q8_scale"]
                q8_q.append(q8_weight.q)
                q8_scales.append(q8_weight.scales)
                offsets["q8_q"] += _size(q8_weight.q)
                offsets["q8_scale"] += _size(q8_weight.scales)
                continue

            if isinstance(source, MxTensor):
                if source.dtype != MXFP4_DTYPE:
                    raise TypeError(
                        "grouped Metal NINTM supports native MXFP4 cohorts, "
                        f"received {source.dtype}"
                    )
                mx_weight = MetalMxWeight.from_tensor(source)
                if mx_weight.out != expert_ids.size * tensor.out_per_expert:
                    raise ValueError("NINTM MXFP4 cohort row count is inconsistent")
                groups = tensor.neuron_len // 32
                for local_expert, expert in enumerate(expert_ids):
                    descriptor = descriptors[int(expert)]
                    descriptor[_FAMILY] = _FAMILY_MXFP4
                    descriptor[_LOCAL_EXPERT] = local_expert
                    descriptor[_OUT] = tensor.out_per_expert
                    descriptor[_K] = tensor.neuron_len
                    descriptor[_MX_NG] = groups
                    descriptor[_MX_VALUE_OFFSET] = offsets["mx_value"]
                    descriptor[_MX_SCALE_OFFSET] = offsets["mx_scale"]
                mx_values.append(mx_weight.values)
                mx_scales.append(mx_weight.scales)
                offsets["mx_value"] += _size(mx_weight.values)
                offsets["mx_scale"] += _size(mx_weight.scales)
                continue

            if not isinstance(source, _VQ_TYPES):
                raise TypeError(
                    "grouped Metal NINTM supports NINT/NVQ/NPQ/NEPQ/MXFP4 cohorts; "
                    f"received {type(source).__name__}"
                )
            weight = MetalVqWeight.from_tensor(source)
            if weight.out != expert_ids.size * tensor.out_per_expert:
                raise ValueError("NINTM VQ cohort row count is inconsistent")
            rotation_variant = 0
            if weight.rotation_block:
                key = (weight.rotation_block, weight.rotation_seed)
                rotation_variant = rotation_variants.get(key, 0)
                if rotation_variant == 0:
                    rotation_variant = len(rotation_specs) + 1
                    rotation_variants[key] = rotation_variant
                    rotation_specs.append(
                        (
                            weight.rotation_signs,
                            weight.rotation_block,
                            weight.rotation_seed,
                        )
                    )
            for local_expert, expert in enumerate(expert_ids):
                descriptor = descriptors[int(expert)]
                descriptor[_FAMILY] = _FAMILY_VQ
                descriptor[_LOCAL_EXPERT] = local_expert
                descriptor[_OUT] = tensor.out_per_expert
                descriptor[_K] = tensor.neuron_len
                descriptor[_VQ_GS] = weight.groupsize
                descriptor[_VQ_NG] = weight.groups
                descriptor[_VQ_VECTOR_SIZE] = weight.vector_size
                descriptor[_VQ_NVEC] = weight.vectors
                descriptor[_VQ_INDEX_BITS] = weight.index_bits
                descriptor[_VQ_STATE_BITS] = weight.state_bits
                descriptor[_VQ_STATES] = weight.states
                descriptor[_VQ_ENTRIES] = weight.entries
                descriptor[_VQ_CODE_BANKS] = weight.code_banks
                descriptor[_VQ_AUX_MODE] = weight.aux_mode
                descriptor[_VQ_CODE_BANK_MODE] = weight.code_bank_mode
                descriptor[_VQ_HAS_TABLE_BANKS] = int(weight.table_banks > 1)
                descriptor[_VQ_GROUPS_PER_SUPER] = weight.groups_per_super
                descriptor[_VQ_NSUPER] = weight.supergroups
                descriptor[_VQ_INDICES_OFFSET] = offsets["vq_indices"]
                descriptor[_VQ_STATE_OFFSET] = offsets["vq_state"]
                descriptor[_VQ_AUX_OFFSET] = offsets["vq_aux"]
                descriptor[_VQ_ANCHOR_OFFSET] = offsets["vq_anchor"]
                descriptor[_VQ_CODEBOOK_OFFSET] = offsets["vq_codebook"]
                descriptor[_VQ_SCALE_OFFSET] = offsets["vq_scale"]
                descriptor[_VQ_STATE_BANK_OFFSET] = offsets["vq_state_bank"]
                descriptor[_VQ_BANK_OFFSET] = offsets["vq_bank"]
                descriptor[_VQ_PARAMETER_OFFSET] = offsets["vq_parameter"]
                descriptor[_VQ_ROTATION_VARIANT] = rotation_variant

            vq_indices.append(weight.indices_packed)
            vq_state.append(weight.state_packed)
            vq_aux.append(weight.aux_packed)
            vq_anchors.append(weight.anchors)
            vq_codebooks.append(weight.codebooks)
            vq_scales.append(weight.scale_lut)
            vq_state_to_codebank.append(weight.state_to_codebank)
            vq_banks.append(weight.bank_ids)
            vq_parameters.append(weight.parameters)
            offsets["vq_indices"] += _size(weight.indices_packed) + 2
            offsets["vq_state"] += _size(weight.state_packed) + 2
            offsets["vq_aux"] += _size(weight.aux_packed) + 2
            offsets["vq_anchor"] += _size(weight.anchors)
            offsets["vq_codebook"] += _size(weight.codebooks)
            offsets["vq_scale"] += _size(weight.scale_lut)
            offsets["vq_state_bank"] += _size(weight.state_to_codebank)
            offsets["vq_bank"] += _size(weight.bank_ids)
            offsets["vq_parameter"] += _size(weight.parameters)

        return cls(
            descriptors=mx.array(descriptors),
            nint_q=_join(nint_q, dtype=mx.uint8, padding=2),
            nint_sub_scale=_join(nint_sub_scale, dtype=mx.uint8),
            nint_sub_min=_join(nint_sub_min, dtype=mx.uint8),
            nint_anchor_scale=_join(nint_anchor_scale, dtype=mx.float32),
            nint_anchor_min=_join(nint_anchor_min, dtype=mx.float32),
            q8_q=_join(q8_q, dtype=mx.int8),
            q8_scales=_join(q8_scales, dtype=mx.float16),
            vq_indices=_join(vq_indices, dtype=mx.uint8, padding=2),
            vq_state=_join(vq_state, dtype=mx.uint8, padding=2),
            vq_aux=_join(vq_aux, dtype=mx.uint8, padding=2),
            vq_anchors=_join(vq_anchors, dtype=mx.float32),
            vq_codebooks=_join(vq_codebooks, dtype=mx.int8),
            vq_scales=_join(vq_scales, dtype=mx.float32),
            vq_state_to_codebank=_join(vq_state_to_codebank, dtype=mx.uint8),
            vq_banks=_join(vq_banks, dtype=mx.uint8),
            vq_parameters=_join(vq_parameters, dtype=mx.float32),
            mx_values=_join(mx_values, dtype=mx.uint8),
            mx_scales=_join(mx_scales, dtype=mx.uint8),
            descriptor_values=descriptors,
            rotation_specs=tuple(rotation_specs),
            experts=tensor.n_experts,
            out_per_expert=tensor.out_per_expert,
            neuron_len=tensor.neuron_len,
            projections=1,
        )

    @classmethod
    def concatenate_projections(
        cls,
        weights: tuple[MetalMoeWeight, ...],
    ) -> MetalMoeWeight:
        """Combine compatible routed projections into one packed dispatch."""

        if not weights:
            raise ValueError("at least one grouped projection is required")
        first = weights[0]
        if any(
            weight.projections != 1
            or weight.experts != first.experts
            or weight.out_per_expert != first.out_per_expert
            or weight.neuron_len != first.neuron_len
            for weight in weights
        ):
            raise ValueError("grouped projections have incompatible shapes")

        buffer_fields = (
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
            "mx_values",
            "mx_scales",
        )
        offsets = {field: 0 for field in buffer_fields}
        descriptor_sets: list[np.ndarray] = []
        combined_specs: list[tuple[mx.array, int, int]] = []
        rotation_variants: dict[tuple[int, int], int] = {}

        for weight in weights:
            descriptors = np.array(weight.descriptor_values, copy=True)
            for descriptor in descriptors:
                if descriptor[_FAMILY] == _FAMILY_NINT:
                    descriptor[_NINT_Q_OFFSET] += offsets["nint_q"]
                    descriptor[_NINT_SUB_OFFSET] += offsets["nint_sub_scale"]
                    descriptor[_NINT_ANCHOR_OFFSET] += offsets["nint_anchor_scale"]
                elif descriptor[_FAMILY] == _FAMILY_NINT8_ZERO:
                    descriptor[_Q8_Q_OFFSET] += offsets["q8_q"]
                    descriptor[_Q8_SCALE_OFFSET] += offsets["q8_scales"]
                elif descriptor[_FAMILY] == _FAMILY_MXFP4:
                    descriptor[_MX_VALUE_OFFSET] += offsets["mx_values"]
                    descriptor[_MX_SCALE_OFFSET] += offsets["mx_scales"]
                else:
                    descriptor[_VQ_INDICES_OFFSET] += offsets["vq_indices"]
                    descriptor[_VQ_STATE_OFFSET] += offsets["vq_state"]
                    descriptor[_VQ_AUX_OFFSET] += offsets["vq_aux"]
                    descriptor[_VQ_ANCHOR_OFFSET] += offsets["vq_anchors"]
                    descriptor[_VQ_CODEBOOK_OFFSET] += offsets["vq_codebooks"]
                    descriptor[_VQ_SCALE_OFFSET] += offsets["vq_scales"]
                    descriptor[_VQ_STATE_BANK_OFFSET] += offsets["vq_state_to_codebank"]
                    descriptor[_VQ_BANK_OFFSET] += offsets["vq_banks"]
                    descriptor[_VQ_PARAMETER_OFFSET] += offsets["vq_parameters"]
                    local_variant = int(descriptor[_VQ_ROTATION_VARIANT])
                    if local_variant:
                        signs, block, seed = weight.rotation_specs[local_variant - 1]
                        key = (block, seed)
                        combined_variant = rotation_variants.get(key, 0)
                        if combined_variant == 0:
                            combined_variant = len(combined_specs) + 1
                            rotation_variants[key] = combined_variant
                            combined_specs.append((signs, block, seed))
                        descriptor[_VQ_ROTATION_VARIANT] = combined_variant
            descriptor_sets.append(descriptors)
            for field in buffer_fields:
                offsets[field] += _size(getattr(weight, field))

        descriptor_values = np.ascontiguousarray(
            np.stack(descriptor_sets, axis=1).reshape(
                first.experts * len(weights),
                _DESCRIPTOR_SIZE,
            ),
            dtype=np.int32,
        )

        def combine(field: str) -> mx.array:
            return mx.contiguous(
                mx.concatenate([getattr(weight, field).reshape((-1,)) for weight in weights])
            )

        return cls(
            descriptors=mx.array(descriptor_values),
            nint_q=combine("nint_q"),
            nint_sub_scale=combine("nint_sub_scale"),
            nint_sub_min=combine("nint_sub_min"),
            nint_anchor_scale=combine("nint_anchor_scale"),
            nint_anchor_min=combine("nint_anchor_min"),
            q8_q=combine("q8_q"),
            q8_scales=combine("q8_scales"),
            vq_indices=combine("vq_indices"),
            vq_state=combine("vq_state"),
            vq_aux=combine("vq_aux"),
            vq_anchors=combine("vq_anchors"),
            vq_codebooks=combine("vq_codebooks"),
            vq_scales=combine("vq_scales"),
            vq_state_to_codebank=combine("vq_state_to_codebank"),
            vq_banks=combine("vq_banks"),
            vq_parameters=combine("vq_parameters"),
            mx_values=combine("mx_values"),
            mx_scales=combine("mx_scales"),
            descriptor_values=descriptor_values,
            rotation_specs=tuple(combined_specs),
            experts=first.experts,
            out_per_expert=first.out_per_expert,
            neuron_len=first.neuron_len,
            projections=len(weights),
        )

    @property
    def packed_nbytes(self) -> int:
        arrays = (
            self.descriptors,
            self.nint_q,
            self.nint_sub_scale,
            self.nint_sub_min,
            self.nint_anchor_scale,
            self.nint_anchor_min,
            self.q8_q,
            self.q8_scales,
            self.vq_indices,
            self.vq_state,
            self.vq_aux,
            self.vq_anchors,
            self.vq_codebooks,
            self.vq_scales,
            self.vq_state_to_codebank,
            self.vq_banks,
            self.vq_parameters,
            self.mx_values,
            self.mx_scales,
            *(signs for signs, _, _ in self.rotation_specs),
        )
        return sum(int(array.nbytes) for array in arrays)


def grouped_moe_matmul(
    weight: MetalMoeWeight,
    x: mx.array | np.ndarray,
    expert_ids: mx.array | np.ndarray,
    *,
    compact_threshold: int | None = 0,
    matrix_threshold: int | None = 0,
    expert_matrix_threshold: int | None = 1,
) -> mx.array:
    """Execute routed experts with direct decode or route-compacted MMQ/MMA.

    The zero-valued auto thresholds select route compaction near four routes per
    expert and MMA near sixteen.  Pure NINT2/3 weights skip the intermediate
    compact-MMQ range because their direct packed kernel remains faster up to
    the MMA crossover.  The matrix path normally uses the expert-owned kernel,
    which decodes each weight tile once and reuses it across assigned routes.
    Setting ``expert_matrix_threshold=None`` exposes the route-owned MMA variant
    for tuning and regression tests.
    """

    source = x if isinstance(x, mx.array) else mx.array(x)
    ids = expert_ids if isinstance(expert_ids, mx.array) else mx.array(expert_ids)
    if ids.dtype not in (mx.int32, mx.uint32):
        ids = ids.astype(mx.int32)
    if ids.ndim != 2:
        raise ValueError("routed expert IDs must have [tokens,routes] shape")
    tokens, routes = (int(value) for value in ids.shape)
    if source.ndim == 2:
        if tuple(int(value) for value in source.shape) != (
            tokens,
            weight.neuron_len,
        ):
            raise ValueError("shared routed input must have [tokens,neuron_len] shape")
        source = mx.broadcast_to(
            source[:, None, :],
            (tokens, routes, weight.neuron_len),
        )
    elif source.ndim != 3 or tuple(int(value) for value in source.shape) != (
        tokens,
        routes,
        weight.neuron_len,
    ):
        raise ValueError("routed input must have [tokens,K] or [tokens,routes,K] shape")
    if source.dtype not in (mx.float16, mx.float32):
        source = source.astype(mx.float16)
    source = mx.contiguous(source)
    ids = mx.contiguous(ids)
    if tokens == 0 or routes == 0:
        return mx.zeros(
            (
                tokens,
                routes,
                weight.projections * weight.out_per_expert,
            ),
            dtype=source.dtype,
        )

    if weight.rotation_specs:
        flattened = source.reshape((tokens * routes, weight.neuron_len))
        variants = [source]
        variants.extend(
            signed_hadamard(flattened, signs, block).reshape(source.shape)
            for signs, block, _ in weight.rotation_specs
        )
        source = mx.contiguous(mx.concatenate(variants, axis=0))

    inputs = [
        weight.descriptors,
        weight.nint_q,
        weight.nint_sub_scale,
        weight.nint_sub_min,
        weight.nint_anchor_scale,
        weight.nint_anchor_min,
        weight.q8_q,
        weight.q8_scales,
        weight.vq_indices,
        weight.vq_state,
        weight.vq_aux,
        weight.vq_anchors,
        weight.vq_codebooks,
        weight.vq_scales,
        weight.vq_state_to_codebank,
        weight.vq_banks,
        weight.vq_parameters,
        weight.mx_values,
        weight.mx_scales,
    ]
    route_count = tokens * routes
    descriptor_families = weight.descriptor_values[:, _FAMILY]
    only_low_bit_nint = bool(
        np.all(descriptor_families == _FAMILY_NINT)
        and np.all(weight.descriptor_values[:, _NINT_BITS] <= 3)
    )
    effective_compact_threshold = (
        max(
            128,
            weight.experts * (16 if only_low_bit_nint else 4),
        )
        if compact_threshold == 0
        else compact_threshold
    )
    if effective_compact_threshold is not None and route_count >= int(effective_compact_threshold):
        flat_ids = ids.reshape((route_count,))
        order = mx.argsort(flat_ids, axis=0)
        sorted_ids = mx.contiguous(mx.take(flat_ids, order, axis=0))
        effective_matrix_threshold = (
            max(128, weight.experts * 16) if matrix_threshold == 0 else matrix_threshold
        )
        use_matrix = (
            source.dtype == mx.float16
            and effective_matrix_threshold is not None
            and route_count >= int(effective_matrix_threshold)
        )
        matrix_group_sizes = np.where(
            descriptor_families == _FAMILY_NINT,
            weight.descriptor_values[:, _NINT_GS],
            np.where(
                descriptor_families == _FAMILY_NINT8_ZERO,
                32,
                np.where(
                    descriptor_families == _FAMILY_MXFP4,
                    32,
                    weight.descriptor_values[:, _VQ_GS],
                ),
            ),
        )
        route_matrix_safe = bool(np.all(matrix_group_sizes <= 48))
        use_expert_matrix = use_matrix and (
            not route_matrix_safe
            or (expert_matrix_threshold is not None and route_count >= int(expert_matrix_threshold))
        )
        kernel = (
            _GROUPED_EXPERT_MMA_KERNEL
            if use_expert_matrix
            else (_GROUPED_MMA_KERNEL if use_matrix else _GROUPED_COMPACT_KERNEL)
        )
        grid = (
            (weight.experts * weight.projections * ((weight.out_per_expert + 63) // 64) * 256)
            if use_expert_matrix
            else (
                ((route_count + 31) // 32)
                * weight.projections
                * ((weight.out_per_expert + 63) // 64)
                * 256
            )
            if use_matrix
            else (
                ((route_count + 3) // 4)
                * weight.projections
                * ((weight.out_per_expert + 7) // 8)
                * 64
            )
        )
        sorted_result = kernel(
            inputs=[*inputs, source, sorted_ids, mx.contiguous(order)],
            template=[
                ("T", source.dtype),
                ("ROUTE_COUNT", route_count),
                ("EXPERTS", weight.experts),
                ("OUT", weight.out_per_expert),
                ("PROJECTIONS", weight.projections),
                ("K", weight.neuron_len),
                ("DESCRIPTOR_SIZE", _DESCRIPTOR_SIZE),
            ],
            grid=(grid, 1, 1),
            threadgroup=((256 if use_matrix else 64), 1, 1),
            output_shapes=[
                (
                    route_count,
                    weight.projections * weight.out_per_expert,
                )
            ],
            output_dtypes=[mx.float32 if use_matrix else source.dtype],
        )[0]
        if use_matrix:
            valid = (flat_ids >= 0) & (flat_ids < weight.experts)
            sorted_result = mx.where(
                valid[:, None],
                sorted_result,
                mx.zeros_like(sorted_result),
            ).astype(source.dtype)
        return sorted_result.reshape(
            (
                tokens,
                routes,
                weight.projections * weight.out_per_expert,
            )
        )

    return _GROUPED_KERNEL(
        inputs=[*inputs, source, ids],
        template=[
            ("T", source.dtype),
            ("TOKENS", tokens),
            ("ROUTES", routes),
            ("EXPERTS", weight.experts),
            ("OUT", weight.out_per_expert),
            ("PROJECTIONS", weight.projections),
            ("K", weight.neuron_len),
            ("DESCRIPTOR_SIZE", _DESCRIPTOR_SIZE),
        ],
        grid=(
            tokens * routes * weight.projections * ((weight.out_per_expert + 7) // 8) * 64,
            1,
            1,
        ),
        threadgroup=(64, 1, 1),
        output_shapes=[
            (
                tokens,
                routes,
                weight.projections * weight.out_per_expert,
            )
        ],
        output_dtypes=[source.dtype],
    )[0]


__all__ = [
    "MetalMoeWeight",
    "UnsupportedGroupedMoeError",
    "grouped_moe_matmul",
]
