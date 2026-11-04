#pragma once

#ifndef MFQ_FATTN_KERNEL_ONLY
#define MFQ_FATTN_KERNEL_ONLY
#define MFQ_SPARSE_ATTN_UNDEF_KERNEL_ONLY
#endif
#include "mfq_fattn_mma_f16.cuh"
#ifdef MFQ_SPARSE_ATTN_UNDEF_KERNEL_ONLY
#undef MFQ_SPARSE_ATTN_UNDEF_KERNEL_ONLY
#undef MFQ_FATTN_KERNEL_ONLY
#endif

#include "mfq_tensor_backend.h"
#include "selected_attention.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace mfq_sparse_attn_mma {

using Tensor = mfq_tensor_backend::Tensor;

struct RuntimeInput {
    const void* positions = nullptr;
    int64_t query_batch_stride = 0;
    int64_t query_head_stride = 0;
    int64_t query_token_stride = 0;
    int64_t query_column_stride = 0;
    int causal_columns = 0;
    bool half_query = false;
    bool wide_positions = false;
    bool direct_query = false;
    bool bf16_query = false;
    // Inline selection: indices may hold sentinels (negative, out of range or
    // padding) and no separate FP16 logit mask is supplied.
    bool inline_indices = false;
    int index_limit = 0;
    mfq::cuda::HotKvView hot{};
    mfq::cuda::IndirectKvView indirect{};
};

inline uint3 sparse_fastdiv_values(uint32_t divisor) {
    uint32_t shift = 0;
    while (shift < 32 && (uint32_t{1} << shift) < divisor) {
        ++shift;
    }
    const uint32_t multiplier = static_cast<uint32_t>(
        (uint64_t{1} << 32) * ((uint64_t{1} << shift) - divisor) /
        divisor + 1);
    return make_uint3(multiplier, shift, divisor);
}

// Shared selected-token attention kernel. Model-specific indexers own the
// selection order and optional validity mask; this kernel owns KV gathering,
// MMA, online softmax and stream-K reduction.
template<int DKQ, int DV, int ncols1, int ncols2, bool V_is_K_view,
         bool needs_fixup, bool is_fixup, bool use_hot>
static __device__ __forceinline__ void process_sparse_attention_tile(
    const void * __restrict__ q,
    const half * __restrict__ k,
    const half * __restrict__ v,
    const int * __restrict__ indices,
    const half * __restrict__ mask,
    const float * __restrict__ sinks,
    float * __restrict__ out,
    float2 * __restrict__ meta,
    float scale,
    int M,
    int heads,
    int kv_heads,
    int max_seq,
    int selected,
    uint3 ne01,
    int64_t work,
    int64_t work_per_sequence,
    int iter_k,
    int iter_j,
    int iter_z_gqa,
    int kb0_start,
    int kb0_stop,
    RuntimeInput input) {
    constexpr int ncols = ncols1 * ncols2;
    constexpr int nthreads =
        ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
    constexpr int nwarps = nthreads / 32;
    constexpr bool use_logit_softcap = false;
    constexpr bool use_indirect = true;

    const int gqa_ratio = heads / kv_heads;
    const int sequence = static_cast<int>(work / work_per_sequence);
    const int64_t sequence_work = work -
        static_cast<int64_t>(sequence) * work_per_sequence;
    const int z_kv = static_cast<int>(sequence_work /
        (static_cast<int64_t>(iter_k) * iter_j * iter_z_gqa));
    const int64_t head_work = sequence_work -
        static_cast<int64_t>(z_kv) * iter_k * iter_j * iter_z_gqa;
    const int zt_gqa = static_cast<int>(head_work /
        (static_cast<int64_t>(iter_k) * iter_j));
    const int64_t query_work = head_work -
        static_cast<int64_t>(zt_gqa) * iter_k * iter_j;
    const int jt = static_cast<int>(query_work / iter_k);
    const int zt_q = z_kv * gqa_ratio + zt_gqa * ncols2;

    const float2 * q_f2 = input.direct_query ? nullptr : reinterpret_cast<const float2 *>(q) +
        (static_cast<int64_t>(sequence) * heads * M +
         static_cast<int64_t>(zt_q) * M) * (DKQ / 2);
    const bool row_cache = input.indirect.states && input.indirect.active(sequence);
    const int64_t physical_seq = row_cache ? input.indirect.capacity(sequence) :
        (use_hot && input.hot.cold_mapping ? input.hot.cold_cells : max_seq);
    if (row_cache) {
        k = static_cast<const half*>(input.indirect.address(sequence));
        v = static_cast<const half*>(input.indirect.address(sequence, 1));
    }
    const int cache_batch = input.indirect.states ? 0 : sequence;
    const half2 * k_h2 = reinterpret_cast<const half2 *>(k) +
        (static_cast<int64_t>(cache_batch) * kv_heads * physical_seq +
         static_cast<int64_t>(z_kv) * physical_seq) * (DKQ / 2);
    const half2 * v_h2 = V_is_K_view
        ? k_h2
        : reinterpret_cast<const half2 *>(v) +
            (static_cast<int64_t>(cache_batch) * kv_heads * physical_seq +
             static_cast<int64_t>(z_kv) * physical_seq) * (DV / 2);
    float2 * dst = reinterpret_cast<float2 *>(out) +
        (static_cast<int64_t>(sequence) * M * heads + zt_q) * (DV / 2);
    const int * row_indices = indices ? indices +
        (static_cast<int64_t>(sequence) * M + jt) * selected : nullptr;
    const half * sequence_mask = mask == nullptr ? nullptr :
        mask + static_cast<int64_t>(sequence) * M * selected;
    const float * tile_sinks = sinks == nullptr ? nullptr : sinks + zt_q;
    mfq_fattn_runtime_input tile_input{};
    if (input.direct_query) {
        const int64_t query_offset = int64_t(sequence)*input.query_batch_stride +
            int64_t(zt_q)*input.query_head_stride;
        const int64_t query_element = input.half_query || input.bf16_query
            ? sizeof(half) : sizeof(float);
        tile_input.query = static_cast<const char*>(q) + query_offset*query_element;
        tile_input.query_token_stride = input.query_token_stride;
        tile_input.query_head_stride = input.query_head_stride;
        tile_input.query_column_stride = input.query_column_stride;
        tile_input.half_query = input.half_query;
        tile_input.bf16_query = input.bf16_query;
    }
    if (input.inline_indices) {
        tile_input.index_count = selected;
        tile_input.index_limit = input.index_limit;
    }
    if (input.positions) {
        int64_t end = input.wide_positions ? static_cast<const int64_t*>(input.positions)[jt]
            : static_cast<const int32_t*>(input.positions)[jt];
        end = end < -1 ? -1 : end;
        end = end >= max_seq ? max_seq - 1 : end;
        end = end >= input.causal_columns ? input.causal_columns - 1 : end;
        tile_input.causal_query_offset = static_cast<int>(end) - jt;
        tile_input.causal_row_count = static_cast<int>(end + 1);
    }

    if constexpr(use_hot) {
        tile_input.hot_keys=reinterpret_cast<const half2*>(input.hot.keys)+(int64_t(sequence)*kv_heads+z_kv)*input.hot.cells*(DKQ/2);
        tile_input.hot_values=reinterpret_cast<const half2*>(input.hot.values)+(int64_t(sequence)*kv_heads+z_kv)*input.hot.cells*(DV/2);
        tile_input.hot_mapping=input.hot.mapping;
        tile_input.cold_mapping=input.hot.cold_mapping;
    }
    flash_attn_ext_f16_process_tile<
        DKQ, DV, ncols1, ncols2, nwarps,
        use_logit_softcap, V_is_K_view, needs_fixup, is_fixup,
        use_indirect, use_hot>(
        q_f2, k_h2, v_h2, sequence_mask, tile_sinks, dst, meta,
        scale, 1.0f, 0.0f, ne01, heads, gqa_ratio, selected,
        DKQ / 2, M * (DKQ / 2), DKQ / 2,
        V_is_K_view ? DKQ / 2 : DV / 2,
        sequence_mask == nullptr ? 0 : selected,
        jt, zt_gqa, kb0_start, kb0_stop, row_indices,
        input.positions != nullptr, 0, tile_input);
}

template<int DKQ, int DV, int ncols1, int ncols2, bool V_is_K_view, bool use_hot = false>
__launch_bounds__(
    ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1 * ncols2),
    ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1 * ncols2))
__global__ void sparse_attention_kernel(
    const void * __restrict__ q,
    const half * __restrict__ k,
    const half * __restrict__ v,
    const int * __restrict__ indices,
    const half * __restrict__ mask,
    const float * __restrict__ sinks,
    float * __restrict__ out,
    float2 * __restrict__ meta,
    float scale,
    int B,
    int M,
    int heads,
    int kv_heads,
    int max_seq,
    int selected,
    uint3 ne01,
    RuntimeInput input) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE)
    static_assert(ncols1 == 1,
        "selected-token tiles require one index row per query tile");
    static_assert(DKQ % 2 == 0 && DV % 2 == 0, "attention widths must be even");
    static_assert(!V_is_K_view || DV <= DKQ, "V view exceeds K width");

    constexpr int ncols = ncols1 * ncols2;
    constexpr int nbatch_fa =
        ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    const int gqa_ratio = heads / kv_heads;
    const int iter_k = (selected + nbatch_fa - 1) / nbatch_fa;
    const int iter_j = (M + ncols1 - 1) / ncols1;
    const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const int64_t work_per_sequence =
        static_cast<int64_t>(iter_k) * iter_j * iter_z_gqa * kv_heads;
    const int64_t total_work = work_per_sequence * B;
    int64_t kbc = static_cast<int64_t>(blockIdx.x) * total_work / gridDim.x;
    const int64_t kbc_stop =
        static_cast<int64_t>(blockIdx.x + 1) * total_work / gridDim.x;

    int kb0_start = static_cast<int>(kbc % iter_k);
    int kb0_stop = min(iter_k, kb0_start + static_cast<int>(kbc_stop - kbc));

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        if (kb0_start == 0) {
            process_sparse_attention_tile<
                DKQ, DV, ncols1, ncols2, V_is_K_view, false, false, use_hot>(
                q, k, v, indices, mask, sinks, out, meta, scale,
                M, heads, kv_heads, max_seq, selected, ne01, kbc,
                work_per_sequence, iter_k, iter_j, iter_z_gqa,
                kb0_start, kb0_stop, input);
        } else {
            process_sparse_attention_tile<
                DKQ, DV, ncols1, ncols2, V_is_K_view, true, false, use_hot>(
                q, k, v, indices, mask, sinks, out, meta, scale,
                M, heads, kv_heads, max_seq, selected, ne01, kbc,
                work_per_sequence, iter_k, iter_j, iter_z_gqa,
                kb0_start, kb0_stop, input);
        }
        kbc += iter_k;
        kbc -= kbc % iter_k;
        kb0_start = 0;
        kb0_stop = min(iter_k, static_cast<int>(kbc_stop - kbc));
    }

    if (kbc < kbc_stop) {
        process_sparse_attention_tile<
            DKQ, DV, ncols1, ncols2, V_is_K_view, false, true, use_hot>(
            q, k, v, indices, mask, sinks, out, meta, scale,
            M, heads, kv_heads, max_seq, selected, ne01, kbc,
            work_per_sequence, iter_k, iter_j, iter_z_gqa,
            kb0_start, kb0_stop, input);
    }
#else
    (void)q; (void)k; (void)v; (void)indices; (void)mask; (void)sinks;
    (void)out; (void)meta; (void)scale; (void)B; (void)M; (void)heads;
    (void)kv_heads; (void)max_seq; (void)selected; (void)ne01; (void)input;
#endif
}

template<int DKQ, int DV, int ncols1, int ncols2, bool V_is_K_view, bool use_hot = false>
Tensor launch(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& indices,
    const Tensor& mask,
    const Tensor& sinks,
    Tensor meta,
    double scale,
    const char * name,
    const Tensor& gate = Tensor{},
    bool half_output = false,
    const Tensor& positions = Tensor{},
    int64_t causal_columns = 0, mfq::cuda::HotKvView hot = {},
    mfq::cuda::IndirectKvView indirect = {},
    bool inline_indices = false, int64_t index_limit = 0) {
    static_assert(ncols1 == 1,
        "selected-token tiles require one index row per query tile");
    constexpr int ncols = ncols1 * ncols2;
    MFQ_RUNTIME_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 &&
        q.size(0) <= INT_MAX && q.size(1) <= INT_MAX && q.size(2) <= INT_MAX &&
        k.size(1) <= INT_MAX && k.size(2) <= INT_MAX,
        name, ": attention dimensions exceed 32-bit indexing");
    const int B = static_cast<int>(q.size(0));
    const int heads = static_cast<int>(q.size(1));
    const int M = static_cast<int>(q.size(2));
    const int kv_heads = static_cast<int>(k.size(1));
    const int max_seq = static_cast<int>(k.size(2));
    const bool causal = positions.defined();
    MFQ_RUNTIME_CHECK(!causal || (causal_columns > 0 && causal_columns <= INT_MAX - 32),
        name, ": causal selection exceeds 32-bit indexing");
    const int selected = causal ? static_cast<int>(((causal_columns + 31) / 32) * 32)
        : static_cast<int>(indices.size(2));
    MFQ_RUNTIME_CHECK(!inline_indices || (!causal && !mask.defined() &&
        index_limit > 0 && index_limit <= max_seq),
        name, ": invalid inline selection");

    MFQ_RUNTIME_CHECK(B > 0 && M > 0 && heads > 0 && kv_heads > 0 &&
        heads % kv_heads == 0 && max_seq > 0 && selected > 0,
        name, ": invalid attention geometry");
    MFQ_RUNTIME_CHECK(q.dim() == 4 && q.size(3) == DKQ &&
        k.dim() == 4 && (k.size(0) == B || indirect.states) && k.size(3) == DKQ &&
        v.dim() == 4 && (v.size(0) == B || indirect.states) && v.size(1) == kv_heads &&
        v.size(2) == max_seq &&
        v.size(3) == (V_is_K_view ? DKQ : DV) &&
        (causal || (indices.dim() == 3 && indices.size(0) == B && indices.size(1) == M)),
        name, ": tensor shapes disagree");
    MFQ_RUNTIME_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() &&
        (causal || indices.is_cuda()) && q.device() == k.device() &&
        q.device() == v.device() && (causal || q.device() == indices.device()),
        name, ": tensors must share one CUDA device");
    MFQ_RUNTIME_CHECK(k.is_contiguous() &&
        v.is_contiguous() && (causal || indices.is_contiguous()),
        name, ": tensors must be contiguous");
    const bool bf16_query = q.scalar_type() == mfq_tensor_backend::kBFloat16;
    MFQ_RUNTIME_CHECK((q.scalar_type() == mfq_tensor_backend::kFloat32 ||
        q.scalar_type() == mfq_tensor_backend::kFloat16 ||
        (bf16_query && inline_indices)) &&
        k.scalar_type() == mfq_tensor_backend::kFloat16 &&
        v.scalar_type() == mfq_tensor_backend::kFloat16 &&
        (causal || indices.scalar_type() == mfq_tensor_backend::kInt32),
        name, ": expected f32/f16/f16/i32 tensors");
    if (causal) {
        MFQ_RUNTIME_CHECK(positions.is_cuda() && positions.device() == q.device() &&
            positions.is_contiguous() && positions.dim() == 1 && positions.numel() == M &&
            (positions.scalar_type() == mfq_tensor_backend::kInt32 || positions.scalar_type() == mfq_tensor_backend::kInt64) &&
            !mask.defined() && !indices.defined(), name, ": invalid implicit causal selection");
    }
    if (mask.defined()) {
        MFQ_RUNTIME_CHECK(mask.is_cuda() && mask.device() == q.device() &&
            mask.is_contiguous() &&
            mask.scalar_type() == mfq_tensor_backend::kFloat16 &&
            mask.sizes() == indices.sizes(), name, ": invalid mask");
    }
    if (sinks.defined()) {
        MFQ_RUNTIME_CHECK(sinks.is_cuda() && sinks.device() == q.device() &&
            sinks.is_contiguous() &&
            sinks.scalar_type() == mfq_tensor_backend::kFloat32 &&
            sinks.numel() == heads, name, ": invalid sinks");
    }

    const MfqCudaGuard guard(q.device());
    int device = 0;
    cudaDeviceProp properties{};
    MFQ_RUNTIME_CHECK(cudaGetDevice(&device) == cudaSuccess,
        name, ": cudaGetDevice failed");
    MFQ_RUNTIME_CHECK(cudaGetDeviceProperties(&properties, device) == cudaSuccess,
        name, ": cudaGetDeviceProperties failed");
    const int cc = properties.major * 100 + properties.minor * 10;
    MFQ_RUNTIME_CHECK(turing_mma_available(cc),
        name, ": requires Turing-class tensor cores or newer");
    const auto config = ampere_mma_available(cc)
        ? ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols)
        : ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
    // Inline selection rewrites the logit mask inside the synchronous loader
    // path; the multi-stage pipeline owns a different mask preload. This is
    // ggml_cuda_fattn_mma_get_nstages for the chosen config, without the host
    // GGML_ASSERT path that needs ggml_abort from the full ggml library.
    const int nstages = cp_async_available(cc) && ncols2 >= 2 ? config.nstages_target : 0;
    MFQ_RUNTIME_CHECK(!inline_indices || nstages <= 1,
        name, ": inline selection requires the synchronous KV loader");
    const int nthreads = config.nthreads;
    const int nwarps = nthreads / 32;
    const int nbatch_fa = config.nbatch_fa;
    const int nbatch_k2 = config.nbatch_K2;
    const int nbatch_v2 = config.nbatch_V2;
    const int nbatch_combine = config.nbatch_combine;
    const bool q_in_reg = config.Q_in_reg;
    MFQ_RUNTIME_CHECK(nthreads > 0 && nbatch_fa > 0 && nbatch_k2 > 0 &&
        nbatch_v2 > 0 && nbatch_combine > 0,
        name, ": unsupported MMA geometry");
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const size_t shared_kv = static_cast<size_t>(nbatch_fa) *
        std::max(nbatch_k2 + 4, nbatch_v2 + 4) * sizeof(half2);
    const size_t shared_q = static_cast<size_t>(ncols) *
        (DKQ / 2 + 4) * sizeof(half2);
    const size_t shared_mask = static_cast<size_t>(ncols1) *
        (nbatch_fa / 2 + 4) * sizeof(half2);
    const size_t shared_combine = static_cast<size_t>(nwarps) *
        cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
    const size_t shmem = std::max(
        shared_combine,
        q_in_reg ? std::max(shared_q, shared_kv + shared_mask)
                 : shared_q + shared_kv + shared_mask);

    using Kernel = decltype(&sparse_attention_kernel<
        DKQ, DV, ncols1, ncols2, V_is_K_view, use_hot>);
    Kernel kernel = sparse_attention_kernel<
        DKQ, DV, ncols1, ncols2, V_is_K_view, use_hot>;
    static bool shmem_set[32] = {};
    MFQ_RUNTIME_CHECK(device >= 0 && device < 32,
        name, ": unsupported CUDA device index");
    if (!shmem_set[device]) {
        const cudaError_t status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shmem));
        MFQ_RUNTIME_CHECK(status == cudaSuccess,
            name, ": shared-memory attribute failed: ", cudaGetErrorString(status));
        shmem_set[device] = true;
    }

    const int gqa_ratio = heads / kv_heads;
    const int iter_j = (M + ncols1 - 1) / ncols1;
    const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const int64_t ntiles_dst64 =
        static_cast<int64_t>(B) * kv_heads * iter_z_gqa * iter_j;
    const int ntiles_kv = (selected + nbatch_fa - 1) / nbatch_fa;
    MFQ_RUNTIME_CHECK(ntiles_dst64 > 0 &&
        ntiles_dst64 <= std::numeric_limits<int>::max(),
        name, ": CUDA grid is too large");
    const int ntiles_dst = static_cast<int>(ntiles_dst64);
    int max_blocks_per_sm = 0;
    cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, kernel, nthreads, shmem);
    MFQ_RUNTIME_CHECK(status == cudaSuccess && max_blocks_per_sm > 0,
        name, ": occupancy query failed: ", cudaGetErrorString(status));
    const int resident_blocks =
        max_blocks_per_sm * properties.multiProcessorCount;
    const int64_t total_tiles = ntiles_dst64 * ntiles_kv;
    const int raw_blocks = static_cast<int>(std::min<int64_t>(
        resident_blocks, total_tiles));
    const int rounded_blocks = std::max(
        ntiles_dst, (raw_blocks / ntiles_dst) * ntiles_dst);
    const int blocks_per_tile = rounded_blocks / ntiles_dst;

    const size_t meta_float2 = static_cast<size_t>(rounded_blocks) *
        ncols * (2 + DV / 2);
    const size_t required_meta = blocks_per_tile > 1 ? 2 * meta_float2 : 1;
    if (!meta.defined()) {
        MFQ_RUNTIME_CHECK(required_meta <=
            static_cast<size_t>(std::numeric_limits<int64_t>::max()),
            name, ": meta workspace is too large");
        meta = mfq_tensor_backend::empty(
            {static_cast<int64_t>(required_meta)},
            q.options().dtype(mfq_tensor_backend::kFloat32));
    }
    MFQ_RUNTIME_CHECK(meta.is_cuda() && meta.device() == q.device() &&
        meta.is_contiguous() &&
        meta.scalar_type() == mfq_tensor_backend::kFloat32 &&
        static_cast<size_t>(meta.numel()) >= required_meta,
        name, ": meta workspace too small, need ", required_meta,
        " float elements");

    Tensor out = mfq_tensor_backend::empty({B, M, heads, DV}, q.options().dtype(mfq_tensor_backend::kFloat32));
    const auto stream = mfq_current_cuda_stream();
    const bool direct_query = causal || inline_indices || bf16_query ||
        q.scalar_type() == mfq_tensor_backend::kFloat16 || !q.is_contiguous();
    const RuntimeInput input{causal ? positions.data_ptr() : nullptr,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), static_cast<int>(causal_columns),
        q.scalar_type() == mfq_tensor_backend::kFloat16,
        causal && positions.scalar_type() == mfq_tensor_backend::kInt64,
        direct_query, bf16_query, inline_indices, static_cast<int>(index_limit), hot, indirect};
    kernel<<<rounded_blocks, dim3(32, nwarps, 1), shmem, stream>>>(
        q.data_ptr(),
        reinterpret_cast<const half *>(k.data_ptr<mfq_half>()),
        reinterpret_cast<const half *>(v.data_ptr<mfq_half>()),
        indices.defined() ? indices.data_ptr<int>() : nullptr,
        mask.defined()
            ? reinterpret_cast<const half *>(mask.data_ptr<mfq_half>())
            : nullptr,
        sinks.defined() ? sinks.data_ptr<float>() : nullptr,
        out.data_ptr<float>(), reinterpret_cast<float2 *>(meta.data_ptr<float>()),
        static_cast<float>(scale), B, M, heads, kv_heads, max_seq, selected,
        sparse_fastdiv_values(static_cast<uint32_t>(M)), input);
    status = cudaGetLastError();
    MFQ_RUNTIME_CHECK(status == cudaSuccess,
        name, " launch failed: ", cudaGetErrorString(status));

    if (gate.defined()) {
        return mfq_selected_attention::mma_reduce_gated(out, meta, gate, half_output,
            kv_heads, rounded_blocks, blocks_per_tile, ncols1, ncols2);
    }
    if (blocks_per_tile > 1) {
        const uint3 fd0 = sparse_fastdiv_values(
            static_cast<uint32_t>(iter_j * iter_z_gqa * kv_heads));
        const uint3 fd1 = sparse_fastdiv_values(
            static_cast<uint32_t>(iter_j * iter_z_gqa));
        const uint3 fd2 = sparse_fastdiv_values(static_cast<uint32_t>(iter_j));
        flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>
            <<<dim3(ntiles_dst, ncols1, ncols2), DV, 0, stream>>>(
                out.data_ptr<float>(),
                reinterpret_cast<const float2 *>(meta.data_ptr<float>()),
                M, heads, kv_heads, rounded_blocks, gqa_ratio,
                blocks_per_tile, fd0, fd1, fd2);
        status = cudaGetLastError();
        MFQ_RUNTIME_CHECK(status == cudaSuccess,
            name, " fixup failed: ", cudaGetErrorString(status));
    }
    return out;
}

} // namespace mfq_sparse_attn_mma
