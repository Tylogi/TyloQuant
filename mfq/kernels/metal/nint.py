"""Packed NINT Metal kernels for Apple silicon.

The kernels in this module use :func:`mlx.core.fast.metal_kernel` so they can
be JIT-compiled by MLX without a separate ``metal`` command-line compiler.
Weights remain packed in the MFQ deploy layout:

``q_packed``
    A continuous little-endian q-value bit stream for NINT2/3/4/6/8. NINT5 is
    repacked once into separate low4/high1 planes for branch-free hot-loop
    extraction.
``sub_scale`` / ``sub_min``
    Per-output, per-group integer scale metadata.
``neuron_scale`` / ``neuron_min``
    Per-output floating-point anchors.

GEMV and small-M schedules cooperatively consume complete quantization groups.
FP16 GEMM online-decodes only the current weight tile and feeds it to Metal
``simdgroup_matrix`` multiply-accumulate; a dense FP16 weight matrix is never
materialized.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats.nint import NintTensor

_NINT_HEADER = struct.Struct("<BBiii")

_NINT_BITSTREAM_HEADER = r"""
#include <metal_simdgroup_matrix>

inline uint mfq_nint_read_bits(
    device const uchar* stream,
    uint value_index,
    uint bits
) {
    // Avoid overflowing a 32-bit intermediate bit offset.  Large vocab
    // embedding/lm-head tensors can exceed 2^32 bits while their packed byte
    // offsets still fit comfortably in uint.
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

inline uint mfq_nint_read_value(
    device const uchar* stream,
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
"""

_NINT_MATMUL_SOURCE = r"""
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
        uint quantized = mfq_nint_read_value(
            q_packed,
            quantized_index,
            uint(BITS),
            uint(GS),
            uint(Q5_EXEC));
        float weight = scale * float(quantized) - minimum;
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            uint row = first_row + local_row;
            if (row < uint(M)) {
                accumulators[local_row] += float(x[row * uint(K) + input_index]) * weight;
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


_NINT4_MATMUL_SOURCE = r"""
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
        uint metadata_index = output * uint(NG) + group;
        float scale = neuron_scale[output] * float(sub_scale[metadata_index]);
        float minimum = neuron_min[output] * float(sub_min[metadata_index]);
        uint quantized_index = metadata_index * uint(GS) + input_index - group * uint(GS);
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
                        float(x[row * uint(K) + input_index + 1u]) * weight1;
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
"""

_NINT_SWIGLU_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5;
    uint output = workgroup % uint(OUT);
    uint first_row = (workgroup / uint(OUT)) * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }

    float gate_acc[TILE_M];
    float up_acc[TILE_M];
    for (uint row = 0u; row < uint(TILE_M); ++row) {
        gate_acc[row] = 0.0f;
        up_acc[row] = 0.0f;
    }

    for (uint column = lane; column < uint(K); column += 32u) {
        uint group = column / uint(GS);
        uint element = column - group * uint(GS);
        uint metadata_index = output * uint(NG) + group;
        float gate_scale =
            gate_neuron_scale[output] * float(gate_sub_scale[metadata_index]);
        float gate_minimum =
            gate_neuron_min[output] * float(gate_sub_min[metadata_index]);
        float up_scale =
            up_neuron_scale[output] * float(up_sub_scale[metadata_index]);
        float up_minimum =
            up_neuron_min[output] * float(up_sub_min[metadata_index]);
        uint quantized_index = metadata_index * uint(GS) + element;
        uint gate_quantized = mfq_nint_read_value(
            gate_q,
            quantized_index,
            uint(BITS),
            uint(GS),
            uint(Q5_EXEC));
        uint up_quantized = mfq_nint_read_value(
            up_q,
            quantized_index,
            uint(BITS),
            uint(GS),
            uint(Q5_EXEC));
        float gate_weight =
            gate_scale * float(gate_quantized) - gate_minimum;
        float up_weight =
            up_scale * float(up_quantized) - up_minimum;
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            uint row = first_row + local_row;
            if (row < uint(M)) {
                float activation = float(x[row * uint(K) + column]);
                gate_acc[local_row] += activation * gate_weight;
                up_acc[local_row] += activation * up_weight;
            }
        }
    }

    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        float gate = simd_sum(gate_acc[local_row]);
        float up = simd_sum(up_acc[local_row]);
        uint row = first_row + local_row;
        if (lane == 0u && row < uint(M)) {
            float silu = gate / (1.0f + metal::exp(-gate));
            y[row * uint(OUT) + output] = T(silu * up);
        }
    }
"""

_NINT4_SWIGLU_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5;
    uint output = workgroup % uint(OUT);
    uint first_row = (workgroup / uint(OUT)) * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }

    float gate_acc[TILE_M];
    float up_acc[TILE_M];
    for (uint row = 0u; row < uint(TILE_M); ++row) {
        gate_acc[row] = 0.0f;
        up_acc[row] = 0.0f;
    }

    constexpr uint pairs = (K + 1) / 2;
    for (uint pair = lane; pair < pairs; pair += 32u) {
        uint column = pair * 2u;
        uint group = column / uint(GS);
        uint metadata_index = output * uint(NG) + group;
        float gate_scale =
            gate_neuron_scale[output] * float(gate_sub_scale[metadata_index]);
        float gate_minimum =
            gate_neuron_min[output] * float(gate_sub_min[metadata_index]);
        float up_scale =
            up_neuron_scale[output] * float(up_sub_scale[metadata_index]);
        float up_minimum =
            up_neuron_min[output] * float(up_sub_min[metadata_index]);
        uint quantized_index =
            metadata_index * uint(GS) + column - group * uint(GS);
        uint gate_packed = uint(gate_q[quantized_index >> 1]);
        uint up_packed = uint(up_q[quantized_index >> 1]);
        float gate_weight0 =
            gate_scale * float(gate_packed & 15u) - gate_minimum;
        float gate_weight1 =
            gate_scale * float(gate_packed >> 4) - gate_minimum;
        float up_weight0 =
            up_scale * float(up_packed & 15u) - up_minimum;
        float up_weight1 =
            up_scale * float(up_packed >> 4) - up_minimum;
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            uint row = first_row + local_row;
            if (row < uint(M)) {
                float activation0 = float(x[row * uint(K) + column]);
                gate_acc[local_row] += activation0 * gate_weight0;
                up_acc[local_row] += activation0 * up_weight0;
                if (column + 1u < uint(K)) {
                    float activation1 =
                        float(x[row * uint(K) + column + 1u]);
                    gate_acc[local_row] += activation1 * gate_weight1;
                    up_acc[local_row] += activation1 * up_weight1;
                }
            }
        }
    }

    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        float gate = simd_sum(gate_acc[local_row]);
        float up = simd_sum(up_acc[local_row]);
        uint row = first_row + local_row;
        if (lane == 0u && row < uint(M)) {
            float silu = gate / (1.0f + metal::exp(-gate));
            y[row * uint(OUT) + output] = T(silu * up);
        }
    }
"""

_NINT_GEMV_FAST_SOURCE = r"""
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;

    float accumulators[ROWS_PER_SIMD] = {0.0f};

    // The CUDA fast path assigns complete quantization groups to lanes.  Doing
    // the same here loads scale/min once per group and shares each activation
    // value across four output rows in the SIMD group.
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

        if (BITS == 2 && (GS % 4) == 0) {
            for (uint element = 0u; element < uint(GS); element += 4u) {
                uint column_base = group * uint(GS) + element;
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    uint quantized_index =
                        (outputs[row] * uint(NG) + group) * uint(GS)
                        + element;
                    uint packed = uint(q_packed[quantized_index >> 2]);
                    for (uint component = 0u; component < 4u; ++component) {
                        uint column = column_base + component;
                        if (column < uint(K)) {
                            uint quantized = (packed >> (component * 2u)) & 3u;
                            accumulators[row] += float(x[column]) * (
                                scales[row] * float(quantized) - minimums[row]);
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
                    for (uint component = 0u; component < 8u; ++component) {
                        uint column = column_base + component;
                        if (column < uint(K)) {
                            uint quantized = (packed >> (component * 3u)) & 7u;
                            accumulators[row] += float(x[column]) * (
                                scales[row] * float(quantized) - minimums[row]);
                        }
                    }
                }
            }
        } else if (BITS == 4 && (GS % 2) == 0) {
            for (uint element = 0u; element < uint(GS); element += 2u) {
                uint column = group * uint(GS) + element;
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    uint quantized_index =
                        (outputs[row] * uint(NG) + group) * uint(GS)
                        + element;
                    uint packed = uint(q_packed[quantized_index >> 1]);
                    if (column < uint(K)) {
                        accumulators[row] += float(x[column]) * (
                            scales[row] * float(packed & 15u) - minimums[row]);
                    }
                    if (column + 1u < uint(K)) {
                        accumulators[row] += float(x[column + 1u]) * (
                            scales[row] * float(packed >> 4) - minimums[row]);
                    }
                }
            }
        } else if (BITS == 5 && Q5_EXEC != 0) {
            constexpr uint LOW_BYTES = (uint(GS) + 1u) / 2u;
            constexpr uint EXEC_BYTES = LOW_BYTES + (uint(GS) + 7u) / 8u;
            for (uint element = 0u; element < uint(GS); element += 8u) {
                float activations[8];
                for (uint component = 0u; component < 8u; ++component) {
                    uint column = group * uint(GS) + element + component;
                    activations[component] =
                        element + component < uint(GS) && column < uint(K)
                        ? float(x[column]) : 0.0f;
                }
                for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
                    uint metadata_index = outputs[row] * uint(NG) + group;
                    uint group_offset = metadata_index * EXEC_BYTES;
                    uint high = uint(q_packed[
                        group_offset + LOW_BYTES + (element >> 3)
                    ]);
                    for (uint component = 0u; component < 8u; ++component) {
                        if (element + component >= uint(GS)) {
                            break;
                        }
                        uint low_packed = uint(q_packed[
                            group_offset + ((element + component) >> 1)
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
                    for (uint component = 0u; component < 4u; ++component) {
                        uint column = column_base + component;
                        if (column < uint(K)) {
                            uint quantized =
                                (packed >> (component * 6u)) & 63u;
                            accumulators[row] += float(x[column]) * (
                                scales[row] * float(quantized) - minimums[row]);
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
                        scales[row] * float(quantized) - minimums[row]);
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

_NINT5_GS28_GEMV_SOURCE = r"""
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

    // One eight-lane subgroup computes one output.  Complete GS28 groups are
    // assigned to lanes, the high-bit plane is loaded once per group, and one
    // low byte supplies two adjacent NINT5 values. Applying affine metadata
    // after the local dot preserves MFQ's two-level scale/min representation:
    // dot(x, scale*q-min) = scale*dot(x,q)-min*sum(x).
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
            uint low1 = uint(q_packed[
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
"""

_NINT6_GS24_GEMV_SOURCE = r"""
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

    // GS24 is byte aligned: four 6-bit values occupy three bytes and one
    // complete group occupies 18 bytes. Lanes own complete groups and share
    // every activation across two output rows. Applying affine metadata once
    // per group implements scale*dot(q,x)-minimum*sum(x).
    for (uint group = lane; group < uint(NG); group += 32u) {
        float activation_sum = 0.0f;
        float quantized_dots[ROWS_PER_SIMD] = {0.0f};
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
                uint metadata_index = metadata_bases[row] + group;
                // Form a byte offset directly. Large-vocabulary tensors can
                // cross 2^32 packed bits, so value_index*6 is unsafe.
                uint byte_index =
                    metadata_index * GROUP_BYTES + chunk * 3u;
                uint packed =
                    uint(q_packed[byte_index])
                    | (uint(q_packed[byte_index + 1u]) << 8u)
                    | (uint(q_packed[byte_index + 2u]) << 16u);
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
"""

_NINT4_GS24_GEMV_SOURCE = r"""
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

    // GS24 is byte aligned: four 4-bit values occupy two bytes and one
    // complete group occupies 12 bytes. Lanes own complete groups and share
    // every activation across two output rows. Applying affine metadata once
    // per group implements scale*dot(q,x)-minimum*sum(x).
    for (uint group = lane; group < uint(NG); group += 32u) {
        float activation_sum = 0.0f;
        float quantized_dots[ROWS_PER_SIMD] = {0.0f};
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
                uint metadata_index = metadata_bases[row] + group;
                // Direct byte addressing avoids first materializing a packed
                // bit index and remains safe for large output matrices.
                uint byte_index =
                    metadata_index * GROUP_BYTES + chunk * 2u;
                uint packed =
                    uint(q_packed[byte_index])
                    | (uint(q_packed[byte_index + 1u]) << 8u);
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
"""

_NINT_MMQ_WIDE_SOURCE = r"""
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD = 32u / K_LANES;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;

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

    // Eight lanes reduce K for one output row.  A decoded weight is reused
    // across up to five activation rows, matching MLX's affine qmv_wide data
    // flow while retaining MFQ's two-level scale/min representation.
    for (uint group = k_lane; group < uint(NG); group += K_LANES) {
        uint metadata_index = output * uint(NG) + group;
        float scale =
            neuron_scale[output] * float(sub_scale[metadata_index]);
        float minimum =
            neuron_min[output] * float(sub_min[metadata_index]);
        constexpr uint Q5_LOW_BYTES = (uint(GS) + 1u) / 2u;
        constexpr uint Q5_EXEC_BYTES =
            Q5_LOW_BYTES + (uint(GS) + 7u) / 8u;
        uint q5_group_offset = metadata_index * Q5_EXEC_BYTES;
        for (uint element = 0u; element < uint(GS); ++element) {
            uint column = group * uint(GS) + element;
            if (column >= uint(K)) {
                break;
            }
            uint quantized_index = metadata_index * uint(GS) + element;
            uint quantized;
            if (BITS == 5 && Q5_EXEC != 0) {
                uint low_packed =
                    uint(q_packed[q5_group_offset + (element >> 1)]);
                uint low =
                    (low_packed >> ((element & 1u) * 4u)) & 15u;
                uint high = (
                    uint(q_packed[
                        q5_group_offset + Q5_LOW_BYTES + (element >> 3)
                    ]) >> (element & 7u)
                ) & 1u;
                quantized = low | (high << 4u);
            } else {
                quantized = mfq_nint_read_value(
                    q_packed,
                    quantized_index,
                    uint(BITS),
                    uint(GS),
                    uint(Q5_EXEC));
            }
            float weight = scale * float(quantized) - minimum;
            for (uint vector = 0u; vector < uint(TILE_M); ++vector) {
                uint input_row = min(first_vector + vector, uint(M) - 1u);
                accumulators[vector] +=
                    float(x[input_row * uint(K) + column]) * weight;
            }
        }
    }

    for (uint vector = 0u; vector < uint(TILE_M); ++vector) {
        accumulators[vector] += simd_shuffle_down(accumulators[vector], 4);
        accumulators[vector] += simd_shuffle_down(accumulators[vector], 2);
        accumulators[vector] += simd_shuffle_down(accumulators[vector], 1);
        uint input_row = first_vector + vector;
        if (
            k_lane == 0u
            && output_index < uint(OUT)
            && input_row < uint(M)
        ) {
            y[input_row * uint(OUT) + output_index] = T(accumulators[vector]);
        }
    }
"""

_NINT_GEMM_MATRIX_SOURCE = r"""
    constexpr uint BM = uint(BM_TILE);
    constexpr uint BN = 64u;
    // NINT5 gs28 with four groups nearly exhausts Apple threadgroup memory
    // at BM64. Two groups keep the tile near 16 KiB and permit dual
    // residency; the other compact profiles amortize barriers better at four.
    constexpr uint GPC =
        (BITS == 5 && BM_TILE == 64) ? 2u : (GS >= 48 ? 2u : 4u);
    constexpr uint BK = uint(GS) * GPC;
    constexpr uint BK_PAD = BK + 8u;
    constexpr uint BN_PAD = BN + 8u;

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

        // One thread expands one complete output/group pair, the same
        // cooperative online-dequant tile used by the CUDA Tensor-Core path.
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
            float scale = 0.0f;
            float minimum = 0.0f;
            uint metadata_index = 0u;
            if (valid) {
                metadata_index = output * uint(NG) + group;
                scale =
                    neuron_scale[output] * float(sub_scale[metadata_index]);
                minimum =
                    neuron_min[output] * float(sub_min[metadata_index]);
            }
            if (BITS == 2 && (GS % 4) == 0) {
                for (uint element = 0u; element < uint(GS); element += 4u) {
                    uint quantized_index =
                        metadata_index * uint(GS) + element;
                    uint packed = valid
                        ? uint(q_packed[quantized_index >> 2]) : 0u;
                    for (uint component = 0u; component < 4u; ++component) {
                        uint local_column =
                            local_group * uint(GS) + element + component;
                        uint column = column_base + local_column;
                        uint quantized = (packed >> (component * 2u)) & 3u;
                        float value = valid && column < uint(K)
                            ? scale * float(quantized) - minimum : 0.0f;
                        weight_tile[
                            local_column * BN_PAD + local_output
                        ] = half(value);
                    }
                }
            } else if (BITS == 3 && (GS % 8) == 0) {
                for (uint element = 0u; element < uint(GS); element += 8u) {
                    uint quantized_index =
                        metadata_index * uint(GS) + element;
                    uint byte_index =
                        (quantized_index >> 3) * 3u
                        + (((quantized_index & 7u) * 3u) >> 3);
                    uint packed = valid
                        ? uint(q_packed[byte_index])
                            | (uint(q_packed[byte_index + 1u]) << 8)
                            | (uint(q_packed[byte_index + 2u]) << 16)
                        : 0u;
                    for (uint component = 0u; component < 8u; ++component) {
                        uint local_column =
                            local_group * uint(GS) + element + component;
                        uint column = column_base + local_column;
                        uint quantized =
                            (packed >> (component * 3u)) & 7u;
                        float value = valid && column < uint(K)
                            ? scale * float(quantized) - minimum : 0.0f;
                        weight_tile[
                            local_column * BN_PAD + local_output
                        ] = half(value);
                    }
                }
            } else if (BITS == 4 && (GS % 2) == 0) {
                for (uint element = 0u; element < uint(GS); element += 2u) {
                    uint quantized_index =
                        metadata_index * uint(GS) + element;
                    uint packed = valid
                        ? uint(q_packed[quantized_index >> 1]) : 0u;
                    uint local_column =
                        local_group * uint(GS) + element;
                    uint column = column_base + local_column;
                    float value0 = valid && column < uint(K)
                        ? scale * float(packed & 15u) - minimum : 0.0f;
                    float value1 = valid && column + 1u < uint(K)
                        ? scale * float(packed >> 4) - minimum : 0.0f;
                    weight_tile[
                        local_column * BN_PAD + local_output
                    ] = half(value0);
                    weight_tile[
                        (local_column + 1u) * BN_PAD + local_output
                    ] = half(value1);
                }
            } else if (BITS == 5 && Q5_EXEC != 0) {
                constexpr uint LOW_BYTES = (uint(GS) + 1u) / 2u;
                constexpr uint EXEC_BYTES =
                    LOW_BYTES + (uint(GS) + 7u) / 8u;
                uint group_offset = metadata_index * EXEC_BYTES;
                for (uint element = 0u; element < uint(GS); ++element) {
                    uint local_column =
                        local_group * uint(GS) + element;
                    uint column = column_base + local_column;
                    uint low_packed = valid
                        ? uint(q_packed[group_offset + (element >> 1)]) : 0u;
                    uint low =
                        (low_packed >> ((element & 1u) * 4u)) & 15u;
                    uint high = valid
                        ? (
                            uint(q_packed[
                                group_offset + LOW_BYTES + (element >> 3)
                            ]) >> (element & 7u)
                        ) & 1u
                        : 0u;
                    uint quantized = low | (high << 4u);
                    float value = valid && column < uint(K)
                        ? scale * float(quantized) - minimum : 0.0f;
                    weight_tile[
                        local_column * BN_PAD + local_output
                    ] = half(value);
                }
            } else if (BITS == 6 && (GS % 4) == 0) {
                for (uint element = 0u; element < uint(GS); element += 4u) {
                    uint quantized_index =
                        metadata_index * uint(GS) + element;
                    uint byte_index =
                        (quantized_index >> 3) * 6u
                        + (((quantized_index & 7u) * 6u) >> 3);
                    uint packed = valid
                        ? uint(q_packed[byte_index])
                            | (uint(q_packed[byte_index + 1u]) << 8)
                            | (uint(q_packed[byte_index + 2u]) << 16)
                        : 0u;
                    for (uint component = 0u; component < 4u; ++component) {
                        uint local_column =
                            local_group * uint(GS) + element + component;
                        uint column = column_base + local_column;
                        uint quantized =
                            (packed >> (component * 6u)) & 63u;
                        float value = valid && column < uint(K)
                            ? scale * float(quantized) - minimum : 0.0f;
                        weight_tile[
                            local_column * BN_PAD + local_output
                        ] = half(value);
                    }
                }
            } else if (BITS == 8) {
                for (uint element = 0u; element < uint(GS); ++element) {
                    uint local_column =
                        local_group * uint(GS) + element;
                    uint column = column_base + local_column;
                    uint quantized_index =
                        metadata_index * uint(GS) + element;
                    float value = valid && column < uint(K)
                        ? scale * float(q_packed[quantized_index]) - minimum
                        : 0.0f;
                    weight_tile[
                        local_column * BN_PAD + local_output
                    ] = half(value);
                }
            } else {
                for (uint element = 0u; element < uint(GS); ++element) {
                    uint local_column =
                        local_group * uint(GS) + element;
                    uint column = column_base + local_column;
                    float value = 0.0f;
                    if (valid && column < uint(K)) {
                        uint quantized = mfq_nint_read_value(
                            q_packed,
                            metadata_index * uint(GS) + element,
                            uint(BITS),
                            uint(GS),
                            uint(Q5_EXEC));
                        value = scale * float(quantized) - minimum;
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


_NINT_EMBEDDING_SOURCE = r"""
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
"""


def _nint_kernel(name: str, source: str):
    return mx.fast.metal_kernel(
        name=name,
        input_names=[
            "q_packed",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
            "x",
        ],
        output_names=["y"],
        header=_NINT_BITSTREAM_HEADER,
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _nint_pair_kernel(name: str, source: str):
    return mx.fast.metal_kernel(
        name=name,
        input_names=[
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
        ],
        output_names=["y"],
        header=_NINT_BITSTREAM_HEADER,
        source=source,
        compile_options={"math_mode": "fast"},
    )


_NINT_GEMV_KERNEL = _nint_kernel("mfq_nint_packed_gemv", _NINT_MATMUL_SOURCE)
_NINT4_GEMV_KERNEL = _nint_kernel("mfq_nint4_packed_gemv", _NINT4_MATMUL_SOURCE)
_NINT_GEMV_FAST_KERNEL = _nint_kernel(
    "mfq_nint_packed_gemv_fast",
    _NINT_GEMV_FAST_SOURCE,
)
_NINT5_GS28_GEMV_KERNEL = _nint_kernel(
    "mfq_nint5_gs28_packed_gemv",
    _NINT5_GS28_GEMV_SOURCE,
)
_NINT4_GS24_GEMV_KERNEL = _nint_kernel(
    "mfq_nint4_gs24_packed_gemv",
    _NINT4_GS24_GEMV_SOURCE,
)
_NINT6_GS24_GEMV_KERNEL = _nint_kernel(
    "mfq_nint6_gs24_packed_gemv",
    _NINT6_GS24_GEMV_SOURCE,
)
_NINT_MMQ_KERNEL = _nint_kernel("mfq_nint_packed_mmq", _NINT_MATMUL_SOURCE)
_NINT4_MMQ_KERNEL = _nint_kernel("mfq_nint4_packed_mmq", _NINT4_MATMUL_SOURCE)
_NINT_MMQ_WIDE_KERNEL = _nint_kernel(
    "mfq_nint_packed_mmq_wide",
    _NINT_MMQ_WIDE_SOURCE,
)
_NINT_GEMM_KERNEL = _nint_kernel("mfq_nint_packed_gemm", _NINT_MATMUL_SOURCE)
_NINT4_GEMM_KERNEL = _nint_kernel("mfq_nint4_packed_gemm", _NINT4_MATMUL_SOURCE)
_NINT_GEMM_MATRIX_KERNEL = _nint_kernel(
    "mfq_nint_packed_gemm_matrix",
    _NINT_GEMM_MATRIX_SOURCE,
)
_NINT_SWIGLU_KERNEL = _nint_pair_kernel(
    "mfq_nint_packed_swiglu",
    _NINT_SWIGLU_SOURCE,
)
_NINT4_SWIGLU_KERNEL = _nint_pair_kernel(
    "mfq_nint4_packed_swiglu",
    _NINT4_SWIGLU_SOURCE,
)


_NINT_EMBEDDING_KERNEL = mx.fast.metal_kernel(
    name="mfq_nint_packed_embedding",
    input_names=[
        "q_packed",
        "sub_scale",
        "sub_min",
        "neuron_scale",
        "neuron_min",
        "token_ids",
    ],
    output_names=["y"],
    header=_NINT_BITSTREAM_HEADER,
    source=_NINT_EMBEDDING_SOURCE,
    compile_options={"math_mode": "fast"},
)


def _pack_qbits(values: np.ndarray, bits: int) -> np.ndarray:
    """Pack all values into the MFQ continuous little-endian bit stream."""

    source = np.ascontiguousarray(values, dtype=np.uint8)
    if source.ndim != 3:
        raise ValueError(
            f"NINT values must have [out, groups, groupsize] shape, got {source.shape}"
        )
    if not 1 <= int(bits) <= 8:
        raise ValueError(f"Metal NINT supports bit widths in [1, 8], got {bits}")
    if np.any(source > ((1 << int(bits)) - 1)):
        raise ValueError(f"NINT values exceed the {bits}-bit range")
    flat = source.reshape(-1)
    if int(bits) == 8:
        return flat

    bit_rows = np.unpackbits(flat[:, None], axis=-1, bitorder="little")[:, :bits]
    packed = np.packbits(bit_rows.reshape(-1), bitorder="little")
    return np.ascontiguousarray(packed, dtype=np.uint8)


def _pack_q5_exec(values: np.ndarray) -> np.ndarray:
    """Repack NINT5 groups into CUDA's low4/high1 execution layout."""

    source = np.ascontiguousarray(values, dtype=np.uint8)
    if source.ndim != 3:
        raise ValueError(
            f"NINT5 values must have [out, groups, groupsize] shape, got {source.shape}"
        )
    if np.any(source > 31):
        raise ValueError("NINT5 values exceed the 5-bit range")
    groupsize = int(source.shape[-1])
    low_bytes = (groupsize + 1) // 2
    low = np.zeros((*source.shape[:-1], low_bytes), dtype=np.uint8)
    low[..., : source[..., 0::2].shape[-1]] = source[..., 0::2] & 15
    odd = source[..., 1::2]
    low[..., : odd.shape[-1]] |= (odd & 15) << 4
    high = np.packbits(source >> 4, axis=-1, bitorder="little")
    return np.ascontiguousarray(
        np.concatenate((low, high), axis=-1).reshape(-1),
        dtype=np.uint8,
    )


def _unpack_metadata(
    blob: bytes | memoryview,
    offset: int,
    count: int,
    bits: int,
) -> tuple[np.ndarray, int]:
    if not 1 <= int(bits) <= 8:
        raise ValueError(f"Metal NINT metadata supports bit widths in [1, 8], got {bits}")
    nbytes = (int(count) * int(bits) + 7) // 8
    end = int(offset) + nbytes
    if end > len(blob):
        raise ValueError("truncated packed NINT metadata")
    packed = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=offset)
    if int(bits) == 8:
        return packed.copy(), end
    bitstream = np.unpackbits(packed, bitorder="little")[: count * bits]
    bitstream = bitstream.reshape(count, bits)
    shifts = 1 << np.arange(bits, dtype=np.uint16)
    values = (bitstream.astype(np.uint16) * shifts).sum(axis=1).astype(np.uint8)
    return values, end


def _metadata_u8(values: np.ndarray, name: str) -> np.ndarray:
    source = np.asarray(values)
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    if np.any(source < 0) or np.any(source > 255):
        raise ValueError(f"{name} values must fit in uint8 for the Metal deploy layout")
    return np.ascontiguousarray(source, dtype=np.uint8)


@dataclass(frozen=True)
class MetalNintWeight:
    """Execution-ready packed NINT weight resident in MLX/Metal memory."""

    q_packed: mx.array
    sub_scale: mx.array
    sub_min: mx.array
    neuron_scale: mx.array
    neuron_min: mx.array
    bits: int
    groupsize: int
    out: int
    groups: int
    neuron_len: int
    q5_exec: bool

    @classmethod
    def from_tensor(cls, tensor: NintTensor) -> MetalNintWeight:
        if len(tensor.shape) != 2 or tensor.axis != 0:
            raise ValueError("MetalNintWeight currently requires a 2D weight quantized on axis 0")
        out, groups, groupsize = (int(value) for value in tensor.q.shape)
        if groupsize != int(tensor.spec.groupsize):
            raise ValueError("NINT groupsize metadata does not match its packed values")
        if int(tensor.neuron_len) > groups * groupsize:
            raise ValueError("NINT neuron length exceeds the packed group capacity")
        if tensor.sub_scale.shape != (out, groups) or tensor.sub_min.shape != (out, groups):
            raise ValueError("NINT group metadata shape mismatch")
        if tensor.neuron_scale.shape != (out,) or tensor.neuron_min.shape != (out,):
            raise ValueError("NINT neuron metadata shape mismatch")

        bits = int(tensor.spec.bits)
        return cls(
            q_packed=mx.array(
                _pack_q5_exec(tensor.q) if bits == 5 else _pack_qbits(tensor.q, bits)
            ),
            sub_scale=mx.array(_metadata_u8(tensor.sub_scale, "sub_scale")),
            sub_min=mx.array(_metadata_u8(tensor.sub_min, "sub_min")),
            neuron_scale=mx.array(np.ascontiguousarray(tensor.neuron_scale, dtype=np.float32)),
            neuron_min=mx.array(np.ascontiguousarray(tensor.neuron_min, dtype=np.float32)),
            bits=bits,
            groupsize=groupsize,
            out=out,
            groups=groups,
            neuron_len=int(tensor.neuron_len),
            q5_exec=bits == 5,
        )

    @classmethod
    def from_blob(cls, blob: bytes | memoryview) -> MetalNintWeight:
        """Upload one packed NINT file blob without expanding its q-values."""

        if len(blob) < _NINT_HEADER.size:
            raise ValueError("truncated NINT blob")
        bits, sub_bits, groupsize, axis, neuron_len = _NINT_HEADER.unpack_from(blob, 0)
        offset = _NINT_HEADER.size
        if offset + 4 > len(blob):
            raise ValueError("truncated NINT shape header")
        ndim = struct.unpack_from("<I", blob, offset)[0]
        offset += 4
        shape_nbytes = int(ndim) * 8
        if offset + shape_nbytes + 8 > len(blob):
            raise ValueError("truncated NINT shape")
        shape = tuple(int(value) for value in struct.unpack_from(f"<{ndim}q", blob, offset))
        offset += shape_nbytes
        out, groups = (int(value) for value in struct.unpack_from("<II", blob, offset))
        offset += 8
        if len(shape) != 2 or int(axis) != 0:
            raise ValueError("MetalNintWeight requires a 2D NINT blob quantized on axis 0")
        if shape[0] != out or int(neuron_len) > groups * int(groupsize):
            raise ValueError("NINT blob dimensions are inconsistent")

        anchors_nbytes = out * np.dtype(np.float16).itemsize
        anchors_end = offset + 2 * anchors_nbytes
        if anchors_end > len(blob):
            raise ValueError("truncated NINT neuron metadata")
        neuron_scale = np.frombuffer(blob, dtype="<f2", count=out, offset=offset).astype(np.float32)
        offset += anchors_nbytes
        neuron_min = np.frombuffer(blob, dtype="<f2", count=out, offset=offset).astype(np.float32)
        offset += anchors_nbytes

        metadata_count = out * groups
        q_count = metadata_count * int(groupsize)
        packed_metadata_nbytes = (metadata_count * int(sub_bits) + 7) // 8
        packed_q_nbytes = (q_count * int(bits) + 7) // 8
        packed_tail_nbytes = 2 * packed_metadata_nbytes + packed_q_nbytes
        old_sub_dtype = np.uint8 if (1 << int(sub_bits)) - 1 <= 255 else np.uint16
        old_q_dtype = np.uint8 if (1 << int(bits)) - 1 <= 255 else np.uint16
        old_tail_nbytes = (
            2 * metadata_count * np.dtype(old_sub_dtype).itemsize
            + q_count * np.dtype(old_q_dtype).itemsize
        )
        remaining = len(blob) - offset
        if remaining == packed_tail_nbytes:
            sub_scale, offset = _unpack_metadata(blob, offset, metadata_count, int(sub_bits))
            sub_min, offset = _unpack_metadata(blob, offset, metadata_count, int(sub_bits))
            q_packed = np.frombuffer(
                blob, dtype=np.uint8, count=packed_q_nbytes, offset=offset
            ).copy()
            offset += packed_q_nbytes
        elif remaining == old_tail_nbytes:
            sub_scale = np.frombuffer(
                blob, dtype=old_sub_dtype, count=metadata_count, offset=offset
            ).copy()
            offset += metadata_count * np.dtype(old_sub_dtype).itemsize
            sub_min = np.frombuffer(
                blob, dtype=old_sub_dtype, count=metadata_count, offset=offset
            ).copy()
            offset += metadata_count * np.dtype(old_sub_dtype).itemsize
            q = np.frombuffer(blob, dtype=old_q_dtype, count=q_count, offset=offset)
            q_packed = _pack_qbits(q.reshape(out, groups, int(groupsize)), int(bits))
            offset += q_count * np.dtype(old_q_dtype).itemsize
        else:
            raise ValueError(
                "invalid NINT blob tail: "
                f"remaining={remaining}, packed={packed_tail_nbytes}, old={old_tail_nbytes}"
            )
        if offset != len(blob):
            raise ValueError("invalid trailing bytes in NINT blob")
        q5_exec = int(bits) == 5
        if q5_exec:
            q_values, q_end = _unpack_metadata(q_packed, 0, q_count, 5)
            if q_end != len(q_packed):
                raise ValueError("invalid packed NINT5 q stream")
            q_packed = _pack_q5_exec(q_values.reshape(out, groups, int(groupsize)))

        return cls(
            q_packed=mx.array(np.ascontiguousarray(q_packed, dtype=np.uint8)),
            sub_scale=mx.array(_metadata_u8(sub_scale, "sub_scale")),
            sub_min=mx.array(_metadata_u8(sub_min, "sub_min")),
            neuron_scale=mx.array(neuron_scale),
            neuron_min=mx.array(neuron_min),
            bits=int(bits),
            groupsize=int(groupsize),
            out=out,
            groups=groups,
            neuron_len=int(neuron_len),
            q5_exec=q5_exec,
        )

    @property
    def packed_nbytes(self) -> int:
        """Persistent Metal bytes used by the packed weight and metadata."""

        arrays = (
            self.q_packed,
            self.sub_scale,
            self.sub_min,
            self.neuron_scale,
            self.neuron_min,
        )
        return sum(int(array.size) * int(array.itemsize) for array in arrays)


def _floating_input(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if result.dtype not in (mx.float16, mx.float32):
        result = result.astype(mx.float16)
    return result


def _threadgroup_size(grid_size: int) -> tuple[int, int, int]:
    return (min(256, max(1, int(grid_size))), 1, 1)


def _prepare_matmul_input(
    weight: MetalNintWeight,
    x: mx.array | np.ndarray,
) -> tuple[mx.array, tuple[int, ...], int]:
    source = _floating_input(x)
    if source.ndim < 1:
        raise ValueError("NINT matmul input must have at least one dimension")
    if int(source.shape[-1]) != weight.neuron_len:
        raise ValueError(
            f"NINT matmul input width {source.shape[-1]} != weight width {weight.neuron_len}"
        )
    prefix = tuple(int(value) for value in source.shape[:-1])
    rows = int(np.prod(prefix, dtype=np.int64)) if prefix else 1
    return source.reshape((rows, weight.neuron_len)), prefix, rows


def _can_use_nint5_gs28_decode(
    weight: MetalNintWeight,
    source: mx.array,
    rows: int,
) -> bool:
    """Return whether the tuned FP16 Q5 execution-layout kernel is valid."""

    return (
        int(rows) == 1
        and weight.bits == 5
        and weight.groupsize == 28
        and weight.q5_exec
        and source.dtype == mx.float16
    )


def _can_use_nint4_gs24_decode(
    weight: MetalNintWeight,
    source: mx.array,
    rows: int,
) -> bool:
    """Return whether the tuned FP16 byte-aligned NINT4 kernel is valid."""

    return (
        int(rows) == 1
        and weight.bits == 4
        and weight.groupsize == 24
        and not weight.q5_exec
        and source.dtype == mx.float16
    )


def _can_use_nint6_gs24_decode(
    weight: MetalNintWeight,
    source: mx.array,
    rows: int,
) -> bool:
    """Return whether the tuned FP16 byte-aligned NINT6 kernel is valid."""

    return (
        int(rows) == 1
        and weight.bits == 6
        and weight.groupsize == 24
        and not weight.q5_exec
        and source.dtype == mx.float16
    )


def _nint_matmul_path(
    weight: MetalNintWeight,
    x: mx.array | np.ndarray,
    *,
    path: str,
) -> mx.array:
    source_2d, prefix, rows = _prepare_matmul_input(weight, x)
    if rows == 0:
        return mx.zeros((*prefix, weight.out), dtype=source_2d.dtype)
    if path == "gemv":
        if rows != 1:
            raise ValueError("NINT GEMV requires exactly one input row")
        tile_rows = 1
        if _can_use_nint4_gs24_decode(weight, source_2d, rows):
            # Eight SIMD groups produce two output rows each. GS24 keeps every
            # NINT4 group aligned to its own 12-byte packed interval.
            kernel = _NINT4_GS24_GEMV_KERNEL
            grid = (((weight.out + 15) // 16) * 256, 1, 1)
            threadgroup = (256, 1, 1)
        elif weight.bits == 4:
            # Preserve the established nibble-specialized fallback for other
            # NINT4 group sizes and FP32 inputs.
            kernel = _NINT4_GEMV_KERNEL
            grid = (weight.out * 32, 1, 1)
            threadgroup = (32, 1, 1)
        elif _can_use_nint5_gs28_decode(weight, source_2d, rows):
            # Four SIMD groups are split into 8-lane reductions.  This path
            # assumes GS28's 14-byte low4 + 4-byte high1 execution layout.
            kernel = _NINT5_GS28_GEMV_KERNEL
            grid = (((weight.out + 15) // 16) * 128, 1, 1)
            threadgroup = (128, 1, 1)
        elif _can_use_nint6_gs24_decode(weight, source_2d, rows):
            # Eight SIMD groups produce two output rows each. GS24 keeps every
            # NINT6 group aligned to its own 18-byte packed interval.
            kernel = _NINT6_GS24_GEMV_KERNEL
            grid = (((weight.out + 15) // 16) * 256, 1, 1)
            threadgroup = (256, 1, 1)
        else:
            kernel = _NINT_GEMV_FAST_KERNEL
            grid = (((weight.out + 7) // 8) * 64, 1, 1)
            threadgroup = (64, 1, 1)
    elif path == "mmq":
        if not 2 <= rows <= 16:
            raise ValueError("NINT MMQ requires 2 to 16 input rows")
        tile_rows = rows
        if weight.q5_exec:
            # The NINT5 low4/high1 group layout is decoded by eight K lanes and
            # reused across the complete small-M tile.
            kernel = _NINT_MMQ_WIDE_KERNEL
            grid = (64, (weight.out + 7) // 8, 1)
            threadgroup = (64, 1, 1)
        else:
            # Other NINT profiles keep every small-M row in registers and read
            # the packed weight only once.
            kernel = _NINT4_MMQ_KERNEL if weight.bits == 4 else _NINT_MMQ_KERNEL
            grid = (weight.out * 32, 1, 1)
            threadgroup = (32, 1, 1)
    elif path == "gemm":
        if source_2d.dtype == mx.float16:
            kernel = _NINT_GEMM_MATRIX_KERNEL
            tile_rows = 64 if rows >= 64 else 32
            grid = (
                ((weight.out + 63) // 64) * 256,
                (rows + tile_rows - 1) // tile_rows,
                1,
            )
            threadgroup = (256, 1, 1)
        else:
            kernel = _NINT4_GEMM_KERNEL if weight.bits == 4 else _NINT_GEMM_KERNEL
            tile_rows = 8
            row_tiles = (rows + tile_rows - 1) // tile_rows
            grid = (row_tiles * weight.out * 32, 1, 1)
            threadgroup = (32, 1, 1)
    else:
        raise ValueError(f"unknown Metal NINT matmul path: {path}")

    output = kernel(
        inputs=[
            weight.q_packed,
            weight.sub_scale,
            weight.sub_min,
            weight.neuron_scale,
            weight.neuron_min,
            source_2d,
        ],
        template=[
            ("T", source_2d.dtype),
            ("BITS", weight.bits),
            ("GS", weight.groupsize),
            ("NG", weight.groups),
            ("K", weight.neuron_len),
            ("OUT", weight.out),
            ("M", rows),
            ("TILE_M", tile_rows),
            ("Q5_EXEC", int(weight.q5_exec)),
            ("BM_TILE", tile_rows if kernel is _NINT_GEMM_MATRIX_KERNEL else 32),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(rows, weight.out)],
        output_dtypes=[source_2d.dtype],
    )[0]
    return output.reshape((*prefix, weight.out))


def nint_gemv(weight: MetalNintWeight, x: mx.array | np.ndarray) -> mx.array:
    """Single-row packed NINT matrix-vector multiply."""

    return _nint_matmul_path(weight, x, path="gemv")


def nint_mmq(weight: MetalNintWeight, x: mx.array | np.ndarray) -> mx.array:
    """Small-M packed NINT multiply with one weight decode shared across rows."""

    return _nint_matmul_path(weight, x, path="mmq")


def nint_gemm(weight: MetalNintWeight, x: mx.array | np.ndarray) -> mx.array:
    """Tiled packed NINT prefill matrix multiply."""

    return _nint_matmul_path(weight, x, path="gemm")


def nint_matmul(
    weight: MetalNintWeight,
    x: mx.array | np.ndarray,
    *,
    dequantize_threshold: int | None = 64,
) -> mx.array:
    """Dispatch ``x @ W.T`` across packed and temporary-dense NINT paths."""

    source = x if isinstance(x, mx.array) else mx.array(x)
    if source.ndim < 1:
        raise ValueError("NINT matmul input must have at least one dimension")
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
        return nint_dequantize_matmul(weight, source)
    if rows == 1:
        return nint_gemv(weight, source)
    if rows <= 16:
        return nint_mmq(weight, source)
    return nint_gemm(weight, source)


def nint_dequantize(
    weight: MetalNintWeight,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Decode a packed NINT matrix to a temporary dense Metal array."""

    ids = mx.arange(weight.out, dtype=mx.int32)
    return nint_embedding(weight, ids, dtype=dtype)


def nint_dequantize_matmul(
    weight: MetalNintWeight,
    x: mx.array | np.ndarray,
) -> mx.array:
    """Temporarily dequantize NINT weights and dispatch MLX dense GEMM."""

    source, prefix, rows = _prepare_matmul_input(weight, x)
    if rows == 0:
        return mx.zeros((*prefix, weight.out), dtype=source.dtype)
    dtype = mx.float16 if source.dtype == mx.float16 else mx.float32
    dense = nint_dequantize(weight, dtype=dtype)
    result = source.astype(dtype) @ dense.T
    return result.reshape((*prefix, weight.out))


def nint_swiglu(
    gate: MetalNintWeight,
    up: MetalNintWeight,
    x: mx.array | np.ndarray,
) -> mx.array:
    """Fuse compatible packed gate/up projections with the SiLU product."""

    layout = (
        "bits",
        "groupsize",
        "out",
        "groups",
        "neuron_len",
        "q5_exec",
    )
    mismatch = [name for name in layout if getattr(gate, name) != getattr(up, name)]
    if mismatch:
        raise ValueError(
            "fused NINT SwiGLU requires matching gate/up layouts; "
            f"different fields: {', '.join(mismatch)}"
        )
    source, prefix, rows = _prepare_matmul_input(gate, x)
    if rows == 0:
        return mx.zeros((*prefix, gate.out), dtype=source.dtype)
    if rows > 16:
        gate_value = nint_matmul(gate, source)
        up_value = nint_matmul(up, source)
        return (mx.sigmoid(gate_value) * gate_value * up_value).reshape((*prefix, gate.out))

    kernel = _NINT4_SWIGLU_KERNEL if gate.bits == 4 else _NINT_SWIGLU_KERNEL
    output = kernel(
        inputs=[
            gate.q_packed,
            gate.sub_scale,
            gate.sub_min,
            gate.neuron_scale,
            gate.neuron_min,
            up.q_packed,
            up.sub_scale,
            up.sub_min,
            up.neuron_scale,
            up.neuron_min,
            source,
        ],
        template=[
            ("T", source.dtype),
            ("BITS", gate.bits),
            ("GS", gate.groupsize),
            ("NG", gate.groups),
            ("K", gate.neuron_len),
            ("OUT", gate.out),
            ("M", rows),
            ("TILE_M", rows),
            ("Q5_EXEC", int(gate.q5_exec)),
        ],
        grid=(gate.out * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, gate.out)],
        output_dtypes=[source.dtype],
    )[0]
    return output.reshape((*prefix, gate.out))


def nint_embedding(
    weight: MetalNintWeight,
    token_ids: mx.array | np.ndarray,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Decode only selected NINT rows into an embedding result on Metal."""

    ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
    if ids.dtype not in (mx.int32, mx.uint32):
        ids = ids.astype(mx.int32)
    shape = tuple(int(value) for value in ids.shape)
    count = int(ids.size)
    if count == 0:
        return mx.zeros((*shape, weight.neuron_len), dtype=dtype)
    ids_flat = ids.reshape((count,))
    output_size = count * weight.neuron_len
    output = _NINT_EMBEDDING_KERNEL(
        inputs=[
            weight.q_packed,
            weight.sub_scale,
            weight.sub_min,
            weight.neuron_scale,
            weight.neuron_min,
            ids_flat,
        ],
        template=[
            ("T", dtype),
            ("BITS", weight.bits),
            ("GS", weight.groupsize),
            ("NG", weight.groups),
            ("K", weight.neuron_len),
            ("COUNT", count),
            ("Q5_EXEC", int(weight.q5_exec)),
        ],
        grid=(output_size, 1, 1),
        threadgroup=_threadgroup_size(output_size),
        output_shapes=[(count, weight.neuron_len)],
        output_dtypes=[dtype],
    )[0]
    return output.reshape((*shape, weight.neuron_len))


__all__ = [
    "MetalNintWeight",
    "nint_dequantize",
    "nint_dequantize_matmul",
    "nint_embedding",
    "nint_gemm",
    "nint_gemv",
    "nint_matmul",
    "nint_mmq",
    "nint_swiglu",
]
