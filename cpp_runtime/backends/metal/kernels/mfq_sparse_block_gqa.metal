// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 OpenAI
//
// Adapted from oMLX's direct-index Qwen4 QSA kernel. The public ABI is model
// neutral: model adapters provide selected fixed-width blocks and cache views.

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/attn/attn.h"

using namespace metal;
using namespace mlx::steel;

#ifndef STEEL_PRAGMA_UNROLL
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif

struct MfqSparseBlockGqaParams {
    int batch;
    int query_heads;
    int kv_heads;
    int queries;
    int keys;
    int selected_blocks;
    int gqa_factor;
    int query_offset;
    int block_size;
    float scale;
    long query_strides[3];
    long key_strides[3];
    long value_strides[3];
    long block_strides[3];
};

struct MfqSparseBlockMaxOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) {
        return metal::max(x, y);
    }
};

struct MfqSparseBlockSumOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) {
        return x + y;
    }
};

struct MfqSparseBlockMulOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) {
        return x * y;
    }
};

struct MfqSparseBlockExpSubOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) {
        return fast::exp2(x - y);
    }
};

struct MfqSparseBlockDivOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) {
        return x / y;
    }
};

// One threadgroup owns one (query row, KV head). All GQA query heads sharing
// that KV head reuse the same randomly addressed K/V tile. The selected block
// list remains compact; expansion to tokens and the causal tail happen here.
template <
    typename T,
    int BK,
    int DC,
    int GQA,
    int H_PAD,
    int D,
    int WM>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]]
void mfq_sparse_block_gqa(
    const device T* query [[buffer(0)]],
    const device T* key [[buffer(1)]],
    const device T* value [[buffer(2)]],
    const device int* blocks [[buffer(3)]],
    device T* output [[buffer(4)]],
    constant MfqSparseBlockGqaParams& params [[buffer(5)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
    constexpr short frag_size = 8;
    constexpr short pad_q = 16 / sizeof(T);
    constexpr short pad_k = 16 / sizeof(T);
    constexpr short pad_v = 16 / sizeof(T);
    constexpr short ldq = DC + pad_q;
    constexpr short ldk = BK + pad_k;
    constexpr short ldv = DC + pad_v;
    constexpr int warps = WM;
    constexpr int tq = H_PAD / (warps * frag_size);
    constexpr int tk = BK / frag_size;
    constexpr int tdc = DC / frag_size;
    constexpr int dimension_chunks = D / DC;
    constexpr int threads = WM * 32;

    static_assert(GQA <= H_PAD);
    static_assert(tq == 1);
    static_assert(H_PAD % (warps * frag_size) == 0);
    static_assert(BK % frag_size == 0);
    static_assert(DC % frag_size == 0);
    static_assert(D % DC == 0);

    const int lane = int(simd_group_id * 32 + simd_lane_id);
    const int query_position = int(tid.x);
    const int kv_head = int(tid.y);
    const int batch = int(tid.z);
    if (batch >= params.batch || query_position >= params.queries ||
        kv_head >= params.kv_heads) {
        return;
    }

    threadgroup T query_tile[H_PAD * ldq];
    threadgroup T kv_tile[(BK * ldv > DC * ldk) ? BK * ldv : DC * ldk];
    threadgroup int selected[BK];

    using Frag = BaseMMAFrag<float, frag_size, frag_size>;
    MMATile<float, tq, 1, Frag> q_fragment;
    MMATile<float, 1, tk, Frag> k_fragment;
    MMATile<float, tq, tk, Frag> score_fragment;
    MMATile<float, 1, 1, Frag> v_fragment;
    MMATile<float, tq, dimension_chunks * tdc, Frag> output_fragment;
    output_fragment.clear();

    const short2 simd_coord = Frag::get_coord(simd_lane_id);
    const short sm = simd_coord.y;
    const short sn = simd_coord.x;
    const short tm = frag_size * tq * simd_group_id;
    const short query_smem_offset = (tm + sm) * ldq + sn;
    const short key_smem_offset = sm * ldk + sn;
    const short value_smem_offset = sm * ldv + sn;
    const float score_scale = params.scale * M_LOG2E_F;

    constexpr short rows_per_thread = decltype(score_fragment)::kRowsPerThread;
    float maximum[rows_per_thread];
    float denominator[rows_per_thread] = {0};
    STEEL_PRAGMA_UNROLL
    for (short row = 0; row < rows_per_thread; ++row) {
        maximum[row] = Limits<float>::finite_min;
    }

    const int query_head_base = kv_head * GQA;
    const device T* query_base = query
        + size_t(batch) * size_t(params.query_strides[0])
        + size_t(query_head_base) * size_t(params.query_strides[1])
        + size_t(query_position) * size_t(params.query_strides[2]);
    const device T* key_base = key
        + size_t(batch) * size_t(params.key_strides[0])
        + size_t(kv_head) * size_t(params.key_strides[1]);
    const device T* value_base = value
        + size_t(batch) * size_t(params.value_strides[0])
        + size_t(kv_head) * size_t(params.value_strides[1]);
    const device int* block_base = blocks
        + size_t(batch) * size_t(params.block_strides[0])
        + size_t(query_position) * size_t(params.block_strides[1]);

    const int absolute_query = params.query_offset + query_position;
    const int tail_width = params.block_size - 1;
    const int selected_tokens =
        params.selected_blocks * params.block_size + tail_width;
    const int complete_blocks =
        (absolute_query + 1) / params.block_size;
    const int valid_blocks = metal::min(
        params.selected_blocks, complete_blocks);
    const int key_tiles = (selected_tokens + BK - 1) / BK;

    for (int key_tile_index = 0;
         key_tile_index < key_tiles;
         ++key_tile_index) {
        const int token_base = key_tile_index * BK;
        for (int item = lane; item < BK; item += threads) {
            const int slot = token_base + item;
            int key_position = -1;
            const int block_token_count =
                params.selected_blocks * params.block_size;
            if (slot < block_token_count) {
                const int block_slot = slot / params.block_size;
                if (block_slot < valid_blocks) {
                    const int raw_block = block_base[block_slot];
                    const long candidate =
                        long(raw_block) * long(params.block_size)
                        + long(slot % params.block_size);
                    if (raw_block >= 0 && candidate < long(params.keys) &&
                        candidate <= long(absolute_query)) {
                        key_position = int(candidate);
                    }
                }
            } else if (slot < selected_tokens) {
                const int tail_offset = slot - block_token_count;
                const int candidate =
                    complete_blocks * params.block_size + tail_offset;
                if (candidate < params.keys && candidate <= absolute_query) {
                    key_position = candidate;
                }
            }
            selected[item] = key_position;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        score_fragment.clear();
        STEEL_PRAGMA_UNROLL
        for (short chunk = 0; chunk < dimension_chunks; ++chunk) {
            const int dimension_base = int(chunk) * DC;
            for (int element = lane;
                 element < H_PAD * (DC / 8);
                 element += threads) {
                const int head = element / (DC / 8);
                const int vector = element - head * (DC / 8);
                uint4 word = uint4(0);
                if (head < GQA) {
                    word = *((const device uint4*)(query_base
                        + size_t(head) * size_t(params.query_strides[1])
                        + dimension_base) + vector);
                }
                *((threadgroup uint4*)(query_tile + head * ldq) + vector) = word;
            }
            for (int element = lane;
                 element < BK * (DC / 8);
                 element += threads) {
                const int row = element / (DC / 8);
                const int vector = element - row * (DC / 8);
                const int key_position = selected[row];
                uint4 word = uint4(0);
                if (key_position >= 0) {
                    word = *((const device uint4*)(key_base
                        + size_t(key_position) * size_t(params.key_strides[2])
                        + dimension_base) + vector);
                }
                thread T* values = (thread T*)&word;
                const int dimension = vector * 8;
                STEEL_PRAGMA_UNROLL
                for (short item = 0; item < 8; ++item) {
                    kv_tile[row + (dimension + item) * ldk] = values[item];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            STEEL_PRAGMA_UNROLL
            for (short block = 0; block < tdc; ++block) {
                simdgroup_barrier(mem_flags::mem_none);
                q_fragment.template load<T, 1, 1, ldq, 1>(
                    &query_tile[query_smem_offset + block * frag_size]);
                k_fragment.template load<T, 1, 1, ldk, 1>(
                    &kv_tile[key_smem_offset + block * frag_size * ldk]);
                simdgroup_barrier(mem_flags::mem_none);
                tile_matmad(
                    score_fragment,
                    q_fragment,
                    k_fragment,
                    score_fragment);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        STEEL_PRAGMA_UNROLL
        for (short item = 0;
             item < decltype(score_fragment)::kElemsPerTile;
             ++item) {
            score_fragment.elems()[item] *= score_scale;
        }
        using ScoreTile = decltype(score_fragment);
        STEEL_PRAGMA_UNROLL
        for (short row = 0; row < ScoreTile::kTileRows; ++row) {
            STEEL_PRAGMA_UNROLL
            for (short column_tile = 0;
                 column_tile < ScoreTile::kTileCols;
                 ++column_tile) {
                const short column = sn + column_tile * ScoreTile::kFragCols;
                STEEL_PRAGMA_UNROLL
                for (short item = 0;
                     item < ScoreTile::MMAFrag_t::kElemCols;
                     ++item) {
                    if (selected[column + item] < 0) {
                        score_fragment.frag_at(row, column_tile)[item] =
                            -INFINITY;
                    }
                }
            }
        }

        float next_maximum[rows_per_thread];
        float rescale[rows_per_thread];
        STEEL_PRAGMA_UNROLL
        for (short row = 0; row < rows_per_thread; ++row) {
            next_maximum[row] = maximum[row];
        }
        score_fragment.template row_reduce<MfqSparseBlockMaxOp>(next_maximum);
        score_fragment.template row_bin_op<MfqSparseBlockExpSubOp>(next_maximum);
        STEEL_PRAGMA_UNROLL
        for (short row = 0; row < rows_per_thread; ++row) {
            rescale[row] = fast::exp2(maximum[row] - next_maximum[row]);
            maximum[row] = next_maximum[row];
        }
        float tile_sum[rows_per_thread] = {0};
        score_fragment.template row_reduce<MfqSparseBlockSumOp>(tile_sum);
        STEEL_PRAGMA_UNROLL
        for (short row = 0; row < rows_per_thread; ++row) {
            denominator[row] = denominator[row] * rescale[row] + tile_sum[row];
        }
        output_fragment.template row_bin_op<MfqSparseBlockMulOp>(rescale);

        STEEL_PRAGMA_UNROLL
        for (short chunk = 0; chunk < dimension_chunks; ++chunk) {
            const int dimension_base = int(chunk) * DC;
            for (int element = lane;
                 element < BK * (DC / 8);
                 element += threads) {
                const int row = element / (DC / 8);
                const int vector = element - row * (DC / 8);
                const int key_position = selected[row];
                uint4 word = uint4(0);
                if (key_position >= 0) {
                    word = *((const device uint4*)(value_base
                        + size_t(key_position) * size_t(params.value_strides[2])
                        + dimension_base) + vector);
                }
                *((threadgroup uint4*)(kv_tile + row * ldv) + vector) = word;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            STEEL_PRAGMA_UNROLL
            for (short query_tile_index = 0;
                 query_tile_index < tq;
                 ++query_tile_index) {
                STEEL_PRAGMA_UNROLL
                for (short dimension_tile = 0;
                     dimension_tile < tdc;
                     ++dimension_tile) {
                    STEEL_PRAGMA_UNROLL
                    for (short key_fragment_index = 0;
                         key_fragment_index < tk;
                         ++key_fragment_index) {
                        const short key_offset = key_fragment_index * frag_size;
                        const short dimension_offset = dimension_tile * frag_size;
                        v_fragment.template load<T, 1, 1, ldv, 1>(
                            &kv_tile[value_smem_offset
                                + key_offset * ldv + dimension_offset]);
                        Frag::mma(
                            output_fragment.frag_at(
                                query_tile_index,
                                chunk * tdc + dimension_tile),
                            score_fragment.frag_at(
                                query_tile_index,
                                key_fragment_index),
                            v_fragment.frag_at(0, 0),
                            output_fragment.frag_at(
                                query_tile_index,
                                chunk * tdc + dimension_tile));
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    output_fragment.template row_bin_op<MfqSparseBlockDivOp>(denominator);
    device T* output_base = output
        + (size_t(batch) * size_t(params.queries) * size_t(params.query_heads)
           + size_t(query_position) * size_t(params.query_heads)
           + size_t(query_head_base + tm + sm)) * size_t(D) + size_t(sn);
    const short rows_left = short(GQA - (tm + sm));
    if (rows_left > 0) {
        output_fragment.template store_safe<T, 1, 1>(
            output_base,
            D,
            short2(D - sn, rows_left));
    }
}

template [[host_name("mfq_sparse_block_gqa_f16_bk64_dc64_gqa12_d256_wm2")]]
[[kernel]] decltype(mfq_sparse_block_gqa<half, 64, 64, 12, 16, 256, 2>)
    mfq_sparse_block_gqa<half, 64, 64, 12, 16, 256, 2>;

template [[host_name("mfq_sparse_block_gqa_bf16_bk64_dc64_gqa12_d256_wm2")]]
[[kernel]] decltype(mfq_sparse_block_gqa<bfloat, 64, 64, 12, 16, 256, 2>)
    mfq_sparse_block_gqa<bfloat, 64, 64, 12, 16, 256, 2>;
