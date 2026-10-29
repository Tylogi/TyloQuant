#include <hip/hip_runtime.h>
#include "kv_cache.h"

#include "hip_execution.h"
#include "mfq_hip_cache_ops.h"
#include "mfq_hip_paged_kv.h"

#include <stdexcept>
#include <utility>

using mfq_tensor_backend::indexing::Slice;

KVCache::KVCache(
        int64_t B,
        int64_t H,
        int64_t max_seq,
        int64_t D,
        bool use_ring,
        mfq_tensor_backend::Device device,
        mfq_tensor_backend::ScalarType dtype, mfq::cuda::KvQuantization settings)
    : ring(use_ring), quantization(settings), paged_dtype(dtype) {
    auto opts = mfq_tensor_backend::TensorOptions().device(device).dtype(dtype);
    if(quantization.enabled())opts=opts.dtype(mfq_tensor_backend::kInt32);
    MFQ_RUNTIME_CHECK(!ring || !quantization.enabled(),"TurboQuant ring cache is unsupported");
    k = mfq_tensor_backend::zeros({B,H,max_seq,quantization.enabled()?mfq::cuda::cuda_kv_packed_width(int(D),quantization.key_bits()):D},opts);
    v = mfq_tensor_backend::zeros({B,H,max_seq,quantization.enabled()?mfq::cuda::cuda_kv_packed_width(int(D),quantization.value_bits()):D},opts);
}

KVCache KVCache::paged_view(
        mfq_tensor_backend::Tensor key_chunks,
        mfq_tensor_backend::Tensor value_chunks,
        mfq_tensor_backend::Tensor pages,
        int64_t batch, int64_t heads, int64_t head_dim,
        int64_t tokens_per_page, int64_t chunk_pages,
        mfq_tensor_backend::ScalarType dtype) {
    KVCache result;
    result.k_chunk_ptrs = std::move(key_chunks);
    result.v_chunk_ptrs = std::move(value_chunks);
    result.page_table = std::move(pages);
    result.paged_batch = batch;
    result.paged_heads = heads;
    result.paged_head_dim = head_dim;
    result.page_size = tokens_per_page;
    result.pages_per_chunk = chunk_pages;
    result.paged_dtype = dtype;
    return result;
}

std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> KVCache::append(
        const CudaExecutionConfig& config,
        mfq_tensor_backend::Tensor kk, mfq_tensor_backend::Tensor vv, mfq_tensor_backend::Tensor pos,
        int64_t start_pos, int64_t end_pos,
        bool contiguous_prefill_prefix) {
        (void)start_pos;
        auto kh = kk.to(scalar_type()).contiguous();
        auto vh = vv.to(scalar_type()).contiguous();
        if(quantization.enabled()) {
            auto packed_k=mfq::cuda::cuda_kv_encode(kh,quantization.key_bits(),false);
            auto packed_v=mfq::cuda::cuda_kv_encode(vh,quantization.value_bits(),true);
            if(is_paged())mfq::cuda::cuda_kv_write_paged(quantized_pages(),packed_k,packed_v,pos);
            else if(pos.dim()==1) {
                k.index_copy_(2,pos,packed_k);v.index_copy_(2,pos,packed_v);
            } else {
                for(int64_t b=0;b<kh.size(0);++b) {
                    auto at=pos.narrow(0,b,1).reshape({-1});
                    k.narrow(0,b,1).index_copy_(2,at,packed_k.narrow(0,b,1));
                    v.narrow(0,b,1).index_copy_(2,at,packed_v.narrow(0,b,1));
                }
            }
            return {k,v};
        }
        if (is_paged()) {
            MFQ_RUNTIME_CHECK(
                paged_batch == kh.size(0) && paged_heads == kh.size(1) &&
                paged_head_dim == kh.size(3) &&
                page_size > 0 && pages_per_chunk > 0,
                "Paged KV cache geometry does not match the write");
            paged_kv_cache_write_cuda(
                k_chunk_ptrs, v_chunk_ptrs, page_table,
                kh, vh, pos, page_size, pages_per_chunk);
            // Later prefill chunks must attend to the complete previous
            // prefix, including a final one-token chunk. Decode reads pages
            // directly and does not materialize a contiguous prefix.
            if (contiguous_prefill_prefix && start_pos > 0) {
                auto options = mfq_tensor_backend::TensorOptions()
                    .device(kh.device()).dtype(kh.scalar_type());
                auto prefix_k = mfq_tensor_backend::empty(
                    {paged_batch, paged_heads, end_pos, paged_head_dim}, options);
                auto prefix_v = mfq_tensor_backend::empty(
                    {paged_batch, paged_heads, end_pos, paged_head_dim}, options);
                paged_kv_cache_gather_cuda(k_chunk_ptrs, v_chunk_ptrs,
                    page_table, prefix_k, prefix_v, page_size, pages_per_chunk);
                return {prefix_k, prefix_v};
            }
            return {kh, vh};
        }
        MFQ_RUNTIME_CHECK(
            kh.dim() == 4 && vh.sizes() == kh.sizes() &&
            kh.size(0) == k.size(0) && kh.size(1) == k.size(1) &&
            kh.size(3) == k.size(3) &&
            ((pos.dim() == 1 && pos.numel() == kh.size(2)) ||
             (pos.dim() == 2 && pos.size(0) == kh.size(0) &&
              pos.size(1) == kh.size(2))),
            "KV cache write requires positions [T] or [B,T]");
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const bool aten_write = config.kv_cache_write_aten;
#else
        const bool aten_write = k.scalar_type() != mfq_tensor_backend::kFloat16 ||
            config.kv_cache_write_aten;
#endif
        if (k.is_cuda() && !aten_write && !ring) {
            auto out = kv_cache_write_cuda(k, v, kh, vh, pos);
            (void)out;
        } else {
            const int64_t batches = pos.dim() == 1 ? 1 : pos.size(0);
            for (int64_t batch = 0; batch < batches; ++batch) {
                auto cache_k = pos.dim() == 1 ? k : k.narrow(0, batch, 1);
                auto cache_v = pos.dim() == 1 ? v : v.narrow(0, batch, 1);
                auto source_k = pos.dim() == 1 ? kh : kh.narrow(0, batch, 1);
                auto source_v = pos.dim() == 1 ? vh : vh.narrow(0, batch, 1);
                auto write_pos = pos.dim() == 1
                    ? pos : pos.narrow(0, batch, 1).reshape({-1});
                if (!k.is_cuda() || aten_write) {
                    auto slots = ring
                        ? mfq_tensor_backend::remainder(write_pos, k.size(2))
                        : write_pos;
                    slots = slots.to(mfq_tensor_backend::kInt64).contiguous();
                    cache_k.index_copy_(2, slots, source_k);
                    cache_v.index_copy_(2, slots, source_v);
                } else {
                    auto out = ring
                        ? kv_cache_write_ring_positions_cuda(
                            cache_k, cache_v, source_k, source_v, write_pos)
                        : kv_cache_write_cuda(
                            cache_k, cache_v, source_k, source_v, write_pos);
                    (void)out;
                }
            }
        }
        if (ring) return {k, v};
        return {k.index({Slice(), Slice(), Slice(0, end_pos), Slice()}),
                v.index({Slice(), Slice(), Slice(0, end_pos), Slice()})};
    }
