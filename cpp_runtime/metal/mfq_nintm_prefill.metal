#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;

namespace {

inline uint read_bits(
    const device uchar* stream,
    uint value_index,
    uint bits) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3u) * bits + (residual_bits >> 3u);
    uint shift = residual_bits & 7u;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1u]) << 8u;
    }
    if (shift + bits > 16u) {
        packed |= uint(stream[byte_index + 2u]) << 16u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

inline void decode_vq_group24(
    const device int* d,
    const device uchar* indices,
    const device uchar* state_stream,
    const device uchar* aux,
    const device float* anchors,
    const device int8_t* codebooks,
    const device float* scales,
    const device uchar* state_to_bank,
    const device uchar* banks,
    const device float* parameters,
    threadgroup half* target,
    uint row,
    uint group,
    uint k_size) {
    uint groups = uint(d[5]);
    uint vector_size = uint(d[6]);
    uint vectors = uint(d[7]);
    uint index_bits = uint(d[8]);
    uint state_bits = uint(d[9]);
    uint state_count = uint(d[10]);
    uint entries = uint(d[11]);
    uint code_banks = uint(d[12]);
    uint aux_mode = uint(d[13]);
    uint code_bank_mode = uint(d[14]);
    uint has_table_banks = uint(d[15]);
    uint groups_per_super = uint(d[16]);
    uint supergroups = uint(d[17]);
    uint indices_offset = uint(d[18]);
    uint state_offset = uint(d[19]);
    uint aux_offset = uint(d[20]);
    uint anchor_offset = uint(d[21]);
    uint codebook_offset = uint(d[22]);
    uint scale_offset = uint(d[23]);
    uint state_bank_offset = uint(d[24]);
    uint bank_offset = uint(d[25]);
    uint parameter_offset = uint(d[26]);
    uint execution = uint(d[29]);
    uint profile = uint(d[28]);
    uint state_index = row * groups + group;
    uint signs = (k_size + 7u) / 8u;
    float anchor = anchors[anchor_offset + row];

    if (profile == 1u || profile == 4u) {
        uint packed_state =
            uint(state_stream[state_offset + (state_index >> 1u)]);
        uint state =
            (packed_state >> ((state_index & 1u) * 4u)) & 15u;
        uint selected_bank =
            uint(state_to_bank[state_bank_offset + state]);
        float scale = anchor * scales[scale_offset + state];
        uint jsc_vector = profile == 1u ? 4u : 8u;
        uint bytes_per_sign = profile == 1u ? 3u : 2u;
        for (uint chunk = 0u; chunk < 3u; ++chunk) {
            uint column_base = group * 24u + chunk * 8u;
            uint first_vector = column_base / jsc_vector;
            uint index0 = 0u;
            uint index1 = 0u;
            uint sign_value = 0u;
            if (execution != 0u) {
                uint offset = indices_offset +
                    (row * signs + column_base / 8u) * bytes_per_sign;
                index0 = uint(indices[offset]);
                if (profile == 1u) {
                    index1 = uint(indices[offset + 1u]);
                }
                sign_value =
                    uint(indices[offset + bytes_per_sign - 1u]);
            } else {
                index0 = uint(indices[
                    indices_offset + row * vectors + first_vector]);
                if (profile == 1u) {
                    index1 = uint(indices[
                        indices_offset + row * vectors
                        + first_vector + 1u]);
                }
                sign_value = read_bits(
                    aux + aux_offset,
                    row * signs + column_base / 8u,
                    7u);
            }
            for (uint inner = 0u; inner < 8u; ++inner) {
                uint index =
                    profile == 1u && inner >= 4u ? index1 : index0;
                uint code_component =
                    profile == 1u ? (inner & 3u) : inner;
                float code = float(codebooks[
                    codebook_offset +
                    (selected_bank * 256u + index) * jsc_vector
                    + code_component]);
                uint negative = inner < 7u
                    ? ((sign_value >> inner) & 1u)
                    : (execution != 0u
                        ? ((sign_value >> 7u) & 1u)
                        : (popcount(sign_value) & 1u));
                target[chunk * 8u + inner] = half(
                    scale * (negative != 0u ? -code : code));
            }
        }
        return;
    }

    if (profile == 2u || profile == 5u) {
        uint state_width = profile == 2u ? 2u : 3u;
        uint index_width = profile == 2u ? 6u : 7u;
        uint state = read_bits(
            state_stream + state_offset,
            state_index,
            state_width);
        uint table_size = profile == 2u ? 64u : 128u;
        float scale = anchor * scales[scale_offset + state];
        for (uint chunk = 0u; chunk < 3u; ++chunk) {
            uint index = read_bits(
                indices + indices_offset,
                row * vectors + group * 3u + chunk,
                index_width);
            uint code_base =
                codebook_offset + (state * table_size + index) * 8u;
            for (uint inner = 0u; inner < 8u; ++inner) {
                target[chunk * 8u + inner] = half(
                    scale * float(codebooks[code_base + inner]));
            }
        }
        return;
    }

    if (profile == 3u) {
        uint state = read_bits(
            state_stream + state_offset,
            state_index,
            state_bits);
        uint sign = read_bits(
            aux + aux_offset,
            state_index,
            1u);
        float delta = parameters[parameter_offset];
        float scale = anchor * scales[scale_offset + state];
        for (uint chunk = 0u; chunk < 3u; ++chunk) {
            uint index = read_bits(
                indices + indices_offset,
                row * vectors + group * 3u + chunk,
                11u);
            uint code_base = codebook_offset + index * 8u;
            for (uint inner = 0u; inner < 8u; ++inner) {
                float code = float(codebooks[code_base + inner]);
                code += sign != 0u ? -delta : delta;
                target[chunk * 8u + inner] = half(scale * code);
            }
        }
        return;
    }

    uint state = read_bits(
        state_stream + state_offset,
        state_index,
        state_bits);
    uint table_bank = has_table_banks != 0u
        ? uint(banks[
              bank_offset + row * supergroups
              + group / groups_per_super])
        : 0u;
    uint delta_value = aux_mode == 3u
        ? read_bits(aux + aux_offset, state_index, 1u)
        : 0u;
    uint selected_bank = code_bank_mode == 1u
        ? uint(state_to_bank[state_bank_offset + state])
        : (code_bank_mode == 2u ? delta_value : 0u);
    float scale = anchor * scales[
        scale_offset + table_bank * state_count + state];
    uint vectors_per_chunk = 8u / vector_size;
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        uint sign_value = 0u;
        if (aux_mode == 1u || aux_mode == 2u) {
            sign_value = read_bits(
                aux + aux_offset,
                row * signs + group * 3u + chunk,
                7u);
        }
        for (uint local = 0u; local < vectors_per_chunk; ++local) {
            uint vector =
                group * (24u / vector_size)
                + chunk * vectors_per_chunk + local;
            uint index = read_bits(
                indices + indices_offset,
                row * vectors + vector,
                index_bits);
            uint code_base = codebook_offset +
                (((table_bank * code_banks + selected_bank) * entries
                  + index) * vector_size);
            for (uint component = 0u;
                 component < vector_size;
                 ++component) {
                uint inner = local * vector_size + component;
                float code = float(codebooks[code_base + component]);
                uint negative = inner < 7u
                    ? ((sign_value >> inner) & 1u)
                    : (popcount(sign_value) & 1u);
                if (aux_mode == 2u && inner == 7u) {
                    negative ^= (index >> 7u) & 1u;
                }
                if (aux_mode == 1u || aux_mode == 2u) {
                    if (negative != 0u) {
                        code = -code;
                    }
                } else if (aux_mode == 3u) {
                    float delta = parameters[parameter_offset];
                    code += delta_value != 0u ? -delta : delta;
                }
                target[chunk * 8u + inner] = half(scale * code);
            }
        }
    }
}

} // namespace

template <bool FUSED_SWIGLU>
[[kernel]] void mfq_grouped_vq_mmq_f16_bm32_bn64_bk96(
    const device int* descriptors [[buffer(0)]],
    const device uchar* vq_indices [[buffer(1)]],
    const device uchar* vq_state [[buffer(2)]],
    const device uchar* vq_aux [[buffer(3)]],
    const device float* vq_anchors [[buffer(4)]],
    const device int8_t* vq_codebooks [[buffer(5)]],
    const device float* vq_scales [[buffer(6)]],
    const device uchar* vq_state_to_bank [[buffer(7)]],
    const device uchar* vq_banks [[buffer(8)]],
    const device float* vq_parameters [[buffer(9)]],
    const device half* x [[buffer(10)]],
    const device int* expert_ids [[buffer(11)]],
    const device int* route_order [[buffer(12)]],
    const device int* block_meta [[buffer(26)]],
    const device int* block_count [[buffer(27)]],
    device half* y [[buffer(13)]],
    constant int& route_count [[buffer(14)]],
    constant int& tokens [[buffer(15)]],
    constant int& routes [[buffer(16)]],
    constant int& experts [[buffer(17)]],
    constant int& output_width [[buffer(18)]],
    constant int& matrix_output_width [[buffer(19)]],
    constant int& input_width [[buffer(20)]],
    constant int& descriptor_size [[buffer(21)]],
    constant int& variant_stride [[buffer(22)]],
    constant int& shared_input [[buffer(23)]],
    constant int& input_sorted [[buffer(24)]],
    constant float& swiglu_limit [[buffer(25)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 96;
    constexpr int BK_padded = 104;
    constexpr uint TGP_SIZE = 256u;
    using mma_t = mlx::steel::BlockMMA<
        half,
        half,
        BM,
        BN,
        BK,
        2,
        4,
        false,
        true,
        BK_padded,
        BK_padded>;

    int output_base = int(tid.x) * BN;
    int block_id = int(tid.y);
    int nblocks = block_count[0];
    if (output_base >= output_width || block_id >= nblocks) {
        return;
    }

    int row_base = block_meta[block_id * 3 + 0];
    int expert = block_meta[block_id * 3 + 1];
    int row_count = block_meta[block_id * 3 + 2];
    if (row_count <= 0 || expert < 0 || expert >= experts) {
        return;
    }

    const device int* descriptor =
        descriptors + expert * descriptor_size;
    uint local_expert = uint(descriptor[1]);
    uint rotation = uint(descriptor[27]);
    short valid_n = short(min(BN, output_width - output_base));
    threadgroup half Xs[BM * BK_padded];
    threadgroup half Ws[BN * BK_padded];

    thread mma_t gate_mma(simd_group_id, simd_lane_id);
    thread mma_t up_mma(simd_group_id, simd_lane_id);
        for (int k_base = 0; k_base < input_width; k_base += BK) {
            for (uint item = thread_id;
                 item < uint(BM * BK_padded);
                 item += TGP_SIZE) {
                uint row = item / uint(BK_padded);
                uint column = item - row * uint(BK_padded);
                half value = half(0.0f);
                int input_column = k_base + int(column);
                if (int(row) < row_count && column < uint(BK)
                    && input_column < input_width) {
                    uint route_index =
                        uint(route_order[row_base + int(row)]);
                    uint source_row = input_sorted != 0
                        ? uint(row_base) + row
                        : (shared_input != 0
                            ? route_index / uint(routes)
                            : route_index);
                    uint source_offset =
                        (rotation * uint(variant_stride) + source_row)
                        * uint(input_width);
                    value = x[source_offset + uint(input_column)];
                }
                Xs[item] = value;
            }
            constexpr int PROJECTIONS = FUSED_SWIGLU ? 2 : 1;
            for (int projection = 0;
                 projection < PROJECTIONS;
                 ++projection) {
                for (uint item = thread_id;
                     item < uint(BN * BK_padded);
                     item += TGP_SIZE) {
                    Ws[item] = half(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint group_item = thread_id;
                     group_item < uint(BN * (BK / 24));
                     group_item += TGP_SIZE) {
                    uint output_row = group_item / uint(BK / 24);
                    uint local_group =
                        group_item - output_row * uint(BK / 24);
                    uint input_column =
                        uint(k_base) + local_group * 24u;
                    if (output_row < uint(valid_n)
                        && input_column < uint(input_width)) {
                        uint group = input_column / 24u;
                        uint pool_row =
                            local_expert * uint(matrix_output_width)
                            + uint(output_base) + output_row
                            + uint(projection * output_width);
                        decode_vq_group24(
                            descriptor,
                            vq_indices,
                            vq_state,
                            vq_aux,
                            vq_anchors,
                            vq_codebooks,
                            vq_scales,
                            vq_state_to_bank,
                            vq_banks,
                            vq_parameters,
                            Ws + output_row * uint(BK_padded)
                                + local_group * 24u,
                            pool_row,
                            group,
                            uint(input_width));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (projection == 0) {
                    gate_mma.mma(Xs, Ws);
                } else {
                    up_mma.mma(Xs, Ws);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if constexpr (FUSED_SWIGLU) {
            for (short item = 0;
                 item < decltype(gate_mma.Ctile)::kElemsPerTile;
                 ++item) {
                float gate = gate_mma.Ctile.elems()[item];
                float up = up_mma.Ctile.elems()[item];
                if (swiglu_limit > 0.0f) {
                    gate = min(gate, swiglu_limit);
                    up = clamp(up, -swiglu_limit, swiglu_limit);
                }
                gate_mma.Ctile.elems()[item] =
                    gate / (1.0f + exp(-gate)) * up;
            }
        }
        device half* destination =
            y + row_base * output_width + output_base;
        gate_mma.store_result_slice(
            destination,
            output_width,
            short2(0, 0),
            short2(valid_n, short(row_count)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#define instantiate_mfq_grouped_vq(name, fused) \
    template [[host_name(name)]] [[kernel]] \
    decltype(mfq_grouped_vq_mmq_f16_bm32_bn64_bk96<fused>) \
    mfq_grouped_vq_mmq_f16_bm32_bn64_bk96<fused>;

instantiate_mfq_grouped_vq(
    "mfq_grouped_vq_mmq_f16_bm32_bn64_bk96",
    false)
instantiate_mfq_grouped_vq(
    "mfq_grouped_vq_swiglu_f16_bm32_bn64_bk96",
    true)
