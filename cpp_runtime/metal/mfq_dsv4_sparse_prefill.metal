#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/attn/attn.h"

using namespace metal;
using namespace mlx::steel;

#ifndef STEEL_PRAGMA_UNROLL
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif

struct MfqDsv4SparseParams {
    int batch;
    int queries;
    int keys;
    int selected;
    float scale;
};

struct MfqSparseMaxOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) {
        return metal::max(x, y);
    }
};

struct MfqSparseSumOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) {
        return x + y;
    }
};

struct MfqSparseMulOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) {
        return x * y;
    }
};

struct MfqSparseExpSubOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) {
        return fast::exp2(x - y);
    }
};

struct MfqSparseDivOp {
    template <typename T>
    METAL_FUNC static constexpr T apply(T x, T y) {
        return x / y;
    }
};

template <typename T, int BK, int DC, int H, int D, int WM>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]]
void mfq_dsv4_sparse_prefill(
    const device T* q [[buffer(0)]],
    const device T* kv [[buffer(1)]],
    const device int* indices [[buffer(2)]],
    const device T* mask [[buffer(3)]],
    const device T* sinks [[buffer(4)]],
    device T* output [[buffer(5)]],
    constant MfqDsv4SparseParams& params [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
    constexpr short kFragSize = 8;
    constexpr short padQ = 16 / sizeof(T);
    constexpr short padK = 16 / sizeof(T);
    constexpr short padV = 16 / sizeof(T);
    constexpr short LDQ = DC + padQ;
    constexpr short LDK = BK + padK;
    constexpr short LDV = DC + padV;
    constexpr int TQ = H / (WM * kFragSize);
    constexpr int TK = BK / kFragSize;
    constexpr int TDC = DC / kFragSize;
    constexpr int D_CHUNKS = D / DC;
    constexpr int tgp_size = WM * 32;

    static_assert(TQ >= 1);
    static_assert(H % (WM * kFragSize) == 0);
    static_assert(BK % kFragSize == 0);
    static_assert(DC % kFragSize == 0);
    static_assert(D % DC == 0);

    const int lane = int(simd_group_id * 32 + simd_lane_id);
    const int query_position = int(tid.x);
    const int batch = int(tid.y);
    if (batch >= params.batch || query_position >= params.queries) {
        return;
    }

    threadgroup T Qs[H * LDQ];
    threadgroup T KVs[(BK * LDV > DC * LDK) ? BK * LDV : DC * LDK];
    threadgroup int selected_rows[BK];
    threadgroup T selected_masks[BK];

    using Frag = BaseMMAFrag<float, kFragSize, kFragSize>;
    MMATile<float, TQ, 1, Frag> Qtile;
    MMATile<float, 1, TK, Frag> Ktile;
    MMATile<float, TQ, TK, Frag> Stile;
    MMATile<float, 1, 1, Frag> Vtile;
    MMATile<float, TQ, D_CHUNKS * TDC, Frag> Otile;
    Otile.clear();

    const short2 simd_coord = Frag::get_coord(simd_lane_id);
    const short sm = simd_coord.y;
    const short sn = simd_coord.x;
    const short tm = kFragSize * TQ * simd_group_id;
    const short Qs_offset = (tm + sm) * LDQ + sn;
    const short Ks_offset = sm * LDK + sn;
    const short Vs_offset = sm * LDV + sn;
    const float score_scale = params.scale * M_LOG2E_F;

    constexpr short rows_per_thread = decltype(Stile)::kRowsPerThread;
    float maximum[rows_per_thread];
    float denominator[rows_per_thread] = {0};
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
        const int head = int(tm + sm + i * kFragSize);
        if (head < H) {
            maximum[i] = M_LOG2E_F * float(sinks[head]);
            denominator[i] = 1.0f;
        } else {
            maximum[i] = Limits<float>::finite_min;
        }
    }

    const device T* q_base = q
        + (size_t(batch) * H * params.queries + query_position) * D;
    const device T* kv_base = kv + size_t(batch) * params.keys * D;
    const size_t selected_base =
        (size_t(batch) * params.queries + query_position) * params.selected;
    const int key_tiles = (params.selected + BK - 1) / BK;

    for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
        const int slot_base = key_tile * BK;
        for (int k = lane; k < BK; k += tgp_size) {
            const int slot = slot_base + k;
            int row = -1;
            T mask_value = T(-INFINITY);
            if (slot < params.selected) {
                row = indices[selected_base + slot];
                mask_value = mask[selected_base + slot];
                if (row < 0 || row >= params.keys || !isfinite(mask_value)) {
                    row = -1;
                    mask_value = T(-INFINITY);
                }
            }
            selected_rows[k] = row;
            selected_masks[k] = mask_value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        Stile.clear();
        STEEL_PRAGMA_UNROLL
        for (short chunk = 0; chunk < D_CHUNKS; ++chunk) {
            const int dimension_base = int(chunk) * DC;
            for (int element = lane; element < H * DC; element += tgp_size) {
                const int head = element / DC;
                const int dimension = element - head * DC;
                Qs[head * LDQ + dimension] = q_base[
                    size_t(head) * params.queries * D
                    + dimension_base + dimension];
            }
            for (int element = lane; element < BK * DC; element += tgp_size) {
                const int k = element / DC;
                const int dimension = element - k * DC;
                const int row = selected_rows[k];
                KVs[k + dimension * LDK] = row >= 0
                    ? kv_base[size_t(row) * D + dimension_base + dimension]
                    : T(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            STEEL_PRAGMA_UNROLL
            for (short block = 0; block < TDC; ++block) {
                simdgroup_barrier(mem_flags::mem_none);
                Qtile.template load<T, 1, 1, LDQ, 1>(
                    &Qs[Qs_offset + block * kFragSize]);
                Ktile.template load<T, 1, 1, LDK, 1>(
                    &KVs[Ks_offset + block * kFragSize * LDK]);
                simdgroup_barrier(mem_flags::mem_none);
                tile_matmad(Stile, Qtile, Ktile, Stile);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        using ScoreTile = decltype(Stile);
        constexpr float negative_infinity = Limits<float>::finite_min;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < ScoreTile::kTileRows; ++i) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < ScoreTile::kTileCols; ++j) {
                const short column = sn + j * ScoreTile::kFragCols;
                STEEL_PRAGMA_UNROLL
                for (short jj = 0; jj < ScoreTile::MMAFrag_t::kElemCols; ++jj) {
                    const int key = int(column + jj);
                    Stile.frag_at(i, j)[jj] = selected_rows[key] < 0
                        ? negative_infinity
                        : Stile.frag_at(i, j)[jj] * score_scale
                            + float(selected_masks[key]) * M_LOG2E_F;
                }
            }
        }

        float new_maximum[rows_per_thread];
        float rescale[rows_per_thread];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < rows_per_thread; ++i) {
            new_maximum[i] = maximum[i];
        }
        Stile.template row_reduce<MfqSparseMaxOp>(new_maximum);
        Stile.template row_bin_op<MfqSparseExpSubOp>(new_maximum);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < rows_per_thread; ++i) {
            rescale[i] = fast::exp2(maximum[i] - new_maximum[i]);
            maximum[i] = new_maximum[i];
        }
        float tile_sum[rows_per_thread] = {0};
        Stile.template row_reduce<MfqSparseSumOp>(tile_sum);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < rows_per_thread; ++i) {
            denominator[i] = denominator[i] * rescale[i] + tile_sum[i];
        }
        Otile.template row_bin_op<MfqSparseMulOp>(rescale);

        STEEL_PRAGMA_UNROLL
        for (short chunk = 0; chunk < D_CHUNKS; ++chunk) {
            const int dimension_base = int(chunk) * DC;
            for (int element = lane; element < BK * DC; element += tgp_size) {
                const int k = element / DC;
                const int dimension = element - k * DC;
                const int row = selected_rows[k];
                KVs[k * LDV + dimension] = row >= 0
                    ? kv_base[size_t(row) * D + dimension_base + dimension]
                    : T(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            STEEL_PRAGMA_UNROLL
            for (short query_tile = 0; query_tile < TQ; ++query_tile) {
                STEEL_PRAGMA_UNROLL
                for (short dimension_tile = 0;
                     dimension_tile < TDC;
                     ++dimension_tile) {
                    STEEL_PRAGMA_UNROLL
                    for (short key = 0; key < TK; ++key) {
                        const short kk = key * kFragSize;
                        const short dd = dimension_tile * kFragSize;
                        Vtile.template load<T, 1, 1, LDV, 1>(
                            &KVs[Vs_offset + kk * LDV + dd]);
                        Frag::mma(
                            Otile.frag_at(
                                query_tile,
                                chunk * TDC + dimension_tile),
                            Stile.frag_at(query_tile, key),
                            Vtile.frag_at(0, 0),
                            Otile.frag_at(
                                query_tile,
                                chunk * TDC + dimension_tile));
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    Otile.template row_bin_op<MfqSparseDivOp>(denominator);
    device T* output_base = output
        + (size_t(batch) * params.queries * H
           + size_t(query_position) * H + size_t(tm + sm)) * D + sn;
    Otile.template store<T, 1, 1>(output_base, D);
}

template [[host_name("mfq_dsv4_sparse_prefill_f16_bk256_dc32")]]
[[kernel]] decltype(mfq_dsv4_sparse_prefill<half, 256, 32, 64, 512, 8>)
    mfq_dsv4_sparse_prefill<half, 256, 32, 64, 512, 8>;
