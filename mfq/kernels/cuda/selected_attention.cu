// Shared dense and index-selected attention mechanics.
#include <math_constants.h>
#include "selected_attention.h"
#include "mfq_sparse_attn_mma_f16.cuh"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <cub/block/block_radix_sort.cuh>

namespace mfq_selected_attention {
namespace tb = mfq_tensor_backend;

void values(std::initializer_list<const Tensor*> tensors) {
    const auto* first = *tensors.begin();
    for (const auto* value : tensors) {
        MFQ_RUNTIME_CHECK(value->defined() && value->is_cuda(),
            "selected attention requires CUDA tensors");
        MFQ_RUNTIME_CHECK(value->device() == first->device(),
            "selected-attention tensors must share one CUDA device");
        const auto dtype = value->scalar_type();
        MFQ_RUNTIME_CHECK(dtype == tb::kFloat32 || dtype == tb::kFloat16 ||
            dtype == tb::kBFloat16, "selected attention requires F32/F16/BF16 values");
    }
}

namespace {
template<int Items>
__device__ void live_topk_sort(const float* source, const int64_t* ids, float* output, int64_t* out_ids,
        int row, int columns, int begin, int live, int pools, int count, int64_t out, unsigned char* temporary) {
    using Sort = cub::BlockRadixSort<float, 256, Items, int64_t>;
    float keys[Items];
    int64_t indices[Items];
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int column = begin + threadIdx.x * Items + i;
        const auto at = int64_t(row) * columns + column;
        keys[i] = column < live ? source[at] : -1e30f;
        indices[i] = column < live ? (ids ? ids[at] : column) : pools;
    }
    Sort(*reinterpret_cast<typename Sort::TempStorage*>(temporary)).SortDescending(keys, indices);
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int rank = threadIdx.x * Items + i;
        if (rank < count) {
            output[out + rank] = keys[i];
            out_ids[out + rank] = indices[i];
        }
    }
}

__global__ void live_topk_blocks_kernel(const float* source, const int64_t* ids,
        float* output, int64_t* out_ids, const int64_t* positions, int columns,
        int pools, int pool, int count, int stage, bool compact) {
    constexpr int tile = 4096;
    extern __shared__ __align__(16) unsigned char temporary[];
    const int row = blockIdx.y, begin = int(blockIdx.x) * tile, tiles = (columns + tile - 1) / tile;
    int live = int(min(int64_t(pools), max(int64_t(0), (positions[row] + 1) / pool)));
    for (int s = 0; s < stage; ++s)
        live = live / tile * count + min(count, live % tile);
    const int available = min(tile, max(0, live - begin));
    const int64_t out = (int64_t(row) * tiles + blockIdx.x) * count;
    // Entirely unused tiles and already reduced candidate lists need no sort.
    if (available <= count) {
        for (int i = threadIdx.x; i < count; i += 256) {
            const auto at = int64_t(row) * columns + begin + i;
            output[out + i] = i < available ? source[at] : -1e30f;
            out_ids[out + i] = i < available ? (ids ? ids[at] : begin + i) : pools;
        }
        return;
    }
    // Sort only the power-of-two range containing live rows, retaining CUB's
    // stable input order and the graph's fixed reduction/output layout.
    if (compact && available <= 1024)
        live_topk_sort<4>(source, ids, output, out_ids, row, columns, begin, live, pools, count, out, temporary);
    else if (compact && available <= 2048)
        live_topk_sort<8>(source, ids, output, out_ids, row, columns, begin, live, pools, count, out, temporary);
    else
        live_topk_sort<16>(source, ids, output, out_ids, row, columns, begin, live, pools, count, out, temporary);
}
} // namespace

Tensor live_topk_blocks(const Tensor& scores, const Tensor& positions, int64_t pool, int64_t count) {
    const auto* option = std::getenv("MFQ_QSA_LIVE_TOPK");
    if ((option && option[0] == '0') || count > 1024)
        return std::get<1>(tb::topk(scores, count, -1, true, false));
    MFQ_RUNTIME_CHECK(scores.is_cuda() && scores.is_contiguous() && scores.dim() == 3 && scores.size(0) == 1 &&
        scores.scalar_type() == tb::kFloat32 && positions.is_cuda() && positions.is_contiguous() &&
        positions.scalar_type() == tb::kInt64 && positions.numel() == scores.size(1) && pool > 0 &&
        count > 0 && count <= scores.size(2), "invalid live selected-attention top-k geometry");
    MfqCudaGuard guard(scores.device());
    const int rows = int(scores.size(1)), pools = int(scores.size(2));
    const auto* compact_option = std::getenv("MFQ_QSA_TOPK_LIVE_TILE");
    const bool compact = !compact_option || compact_option[0] != '0';
    int columns = pools, stage = 0;
    auto keys = scores;
    Tensor ids;
    constexpr auto bytes = sizeof(cub::BlockRadixSort<float, 256, 16, int64_t>::TempStorage);
    if constexpr (bytes > 49152)
        MFQ_CUDA_CHECK(cudaFuncSetAttribute(live_topk_blocks_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    do {
        const int tiles = (columns + 4095) / 4096;
        auto next = tb::empty({1, rows, tiles * count}, scores.options());
        auto next_ids = tb::empty(next.sizes(), scores.options().dtype(tb::kInt64));
        live_topk_blocks_kernel<<<dim3(tiles, rows), 256, bytes, mfq_current_cuda_stream()>>>(
            keys.data_ptr<float>(), ids.defined() ? ids.data_ptr<int64_t>() : nullptr,
            next.data_ptr<float>(), next_ids.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
            columns, pools, int(pool), int(count), stage++, compact);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        keys = std::move(next);
        ids = std::move(next_ids);
        columns = tiles * int(count);
    } while (columns > count);
    return ids;
}


Tensor promoted_matmul(const Tensor& left, const Tensor& right) {
    // MLX matmul promotes mixed floating operands, unlike ATen matmul.
    const auto dtype = left.scalar_type() == right.scalar_type()
        ? left.scalar_type() : tb::kFloat32;
    return tb::matmul(left.to(dtype), right.to(dtype));
}

Tensor dots(const Tensor& query, const Tensor& pooled) {
    values({&query, &pooled});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && pooled.dim() == 3,
        "selected-attention scores require [B,T,H,D] query and [B,P,D] keys");
    MFQ_RUNTIME_CHECK(query.size(0) == pooled.size(0) &&
        query.size(3) == pooled.size(2) && query.size(2) > 0 && query.size(3) > 0,
        "selected-attention score dimensions disagree");
    // QSA rotates [B,H,T,D] then permutes back to [B,T,H,D]. Native
    // batched GEMM requires the leading batch axes to have compact strides.
    return tb::matmul(query.to(tb::kFloat32).contiguous(),
        pooled.to(tb::kFloat32).transpose(-1, -2).unsqueeze(1)).clamp_min(0.0);
}

void attention_shapes(const Tensor& query, const Tensor& key, const Tensor& value) {
    values({&query, &key, &value});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && key.dim() == 4 && value.sizes() == key.sizes(),
        "selected attention requires [B,H,T,D] query/key/value");
    MFQ_RUNTIME_CHECK(query.size(0) == key.size(0) && key.size(1) > 0 &&
        query.size(1) > 0 && query.size(1) % key.size(1) == 0 &&
        query.size(3) == key.size(3) && query.size(3) > 0,
        "selected-attention dimensions disagree");
}

__global__ void prepare_sparse_indices_kernel(
    const int32_t * __restrict__ source,
    int32_t * __restrict__ indices,
    half * __restrict__ mask,
    int64_t rows,
    int source_count,
    int padded_count,
    int cache_length, mfq::cuda::IndirectKvView indirect = {}) {
    const int64_t total = rows * padded_count;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int slot = static_cast<int>(linear % padded_count);
        const int64_t row = linear / padded_count;
        const int index = slot < source_count
            ? source[row * source_count + slot]
            : -1;
        const bool valid = index >= 0 && (indirect.states
            ? indirect.active(row) && index < indirect.capacity(row) : index < cache_length);
        indices[linear] = valid ? index : 0;
        mask[linear] = valid ? __float2half(0.0f) : __float2half(-CUDART_INF_F);
    }
}

// Each CTA retains the Metal kernel's eight warp partials and online-softmax
// recurrence. The generic wide-D path stores accumulators in its output only;
// it imposes no extra head-width limit. Published D=256/512 stay in registers.
struct AttentionLayout {
    int64_t batch, token, head, column;
};

struct AttentionEpilogue {
    const void* gate = nullptr;
    void* output = nullptr;
    bool float_gate = false, half_output = false, double_sigmoid = false;
    AttentionLayout layout{};
};

__device__ float apply_attention_gate(float value, float gate, bool double_sigmoid) {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    const float sigmoid = double_sigmoid
        ? float(__ddiv_rn(1.0, __dadd_rn(1.0, ::exp(-double(gate)))))
        : __fdiv_rn(1.f, __fadd_rn(1.f, expf(-gate)));
#else
    const float sigmoid = 1.f / (1.f + expf(-gate));
#endif
    return __fmul_rn(value, sigmoid);
}

__device__ void store_attention(float value, float* output, int64_t offset,
    int64_t batch, int64_t token, int64_t head, int64_t column,
    const AttentionEpilogue& epilogue) {
    if (!epilogue.gate) { output[offset] = value; return; }
    const auto at = batch * epilogue.layout.batch + token * epilogue.layout.token +
        head * epilogue.layout.head + column * epilogue.layout.column;
    const float gate = epilogue.float_gate ? static_cast<const float*>(epilogue.gate)[at]
        : __half2float(static_cast<const half*>(epilogue.gate)[at]);
    value = apply_attention_gate(value, gate, epilogue.double_sigmoid);
    if (epilogue.half_output) static_cast<half*>(epilogue.output)[offset] = __float2half_rn(value);
    else static_cast<float*>(epilogue.output)[offset] = value;
}

template<int Slices, bool use_hot = false>
__global__ void sparse_attention_kernel(
    const void* query, const half* key, const half* value, const int32_t* indices,
    float* output, int64_t heads, int64_t kv_heads, int64_t tokens,
    int64_t width, int64_t cache_length, int64_t topk, float scale,
    bool half_query, const void* causal_positions, bool wide_positions,
    AttentionEpilogue epilogue, mfq::cuda::HotKvView hot, mfq::cuda::IndirectKvView indirect) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t head = blockIdx.x % heads;
    const int64_t row = blockIdx.x / heads;
    const int64_t token = row % tokens;
    const int64_t batch = row / tokens;
    const int64_t kv_head = head / (heads / kv_heads);
    const bool active = !indirect.states || indirect.active(batch);
    if (indirect.states && active) {
        key = static_cast<const half*>(indirect.address(batch));
        value = static_cast<const half*>(indirect.address(batch, 1));
        cache_length = indirect.capacity(batch);
    }
    const int64_t cache_batch = indirect.states ? 0 : batch;
    const int64_t qb = ((batch * heads + head) * tokens + token) * width;
    const int64_t ob = ((batch * tokens + token) * heads + head) * width;
    __shared__ float partials[8];
    __shared__ float online[4];
    float accum[Slices > 0 ? Slices : 1] = {};
    if constexpr (Slices == 0) {
        for (int64_t d = tid; d < width; d += 256) output[ob + d] = 0.f;
    }
    if (tid == 0) {
        online[0] = -CUDART_INF_F;
        online[1] = online[2] = online[3] = 0.f;
    }
    __syncthreads();
    for (int64_t selected = 0; selected < topk; ++selected) {
        int64_t index;
        if (causal_positions) {
            const int64_t end = wide_positions ? static_cast<const int64_t*>(causal_positions)[token]
                : static_cast<const int32_t*>(causal_positions)[token];
            index = selected <= end ? selected : -1;
        } else index = indices[row * topk + selected];
        bool valid = active && index >= 0 && index < cache_length;
        const int64_t cb = ((cache_batch * kv_heads + kv_head) * cache_length +
            (valid ? index : 0)) * width;
        const half* source_key = indirect.states ? nullptr : key + cb;
        const half* source_value = indirect.states ? nullptr : value + cb;
        if (indirect.states) {
            source_key = indirect.resolve<const half>(batch, index, kv_head, width);
            source_value = indirect.resolve<const half>(batch, index, kv_head, width, 1);
            valid = valid && source_key && source_value;
        }
        if constexpr(use_hot) {
            if(valid) {const int slot=hot.mapping[index];if(slot>=0) {
                const auto at=((batch*kv_heads+kv_head)*hot.cells+slot)*width;
                source_key=static_cast<const half*>(hot.keys)+at;
                source_value=static_cast<const half*>(hot.values)+at;
            }else if(hot.cold_mapping) {
                const auto at=((batch*kv_heads+kv_head)*hot.cold_cells+hot.cold_mapping[index])*width;
                source_key=key+at;source_value=value+at;
            }}
        }
        float dot = 0.f;
        if (valid) {
            for (int64_t d = tid; d < width; d += 256) {
                const float q = half_query ? __half2float(static_cast<const half*>(query)[qb + d])
                    : static_cast<const float*>(query)[qb + d];
                dot += q * __half2float(source_key[d]);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        if (lane == 0) partials[warp] = dot;
        __syncthreads();
        if (tid == 0) {
            float score = 0.f;
            for (int group = 0; group < 8; ++group) score += partials[group];
            score = valid ? score * scale : -CUDART_INF_F;
            const float old_max = online[0];
            const float next_max = fmaxf(old_max, score);
            const float old_scale = isfinite(old_max) ? expf(old_max - next_max) : 0.f;
            const float next_scale = valid ? expf(score - next_max) : 0.f;
            online[0] = next_max;
            online[1] = online[1] * old_scale + next_scale;
            online[2] = old_scale;
            online[3] = next_scale;
        }
        __syncthreads();
        if constexpr (Slices > 0) {
            #pragma unroll
            for (int slice = 0; slice < Slices; ++slice) {
                const int64_t d = tid + slice * 256;
                const float v = valid && d < width ? __half2float(source_value[d]) : 0.f;
                accum[slice] = accum[slice] * online[2] + v * online[3];
            }
        } else {
            for (int64_t d = tid; d < width; d += 256) {
                const float v = valid ? __half2float(source_value[d]) : 0.f;
                output[ob + d] = output[ob + d] * online[2] + v * online[3];
            }
        }
        __syncthreads();
    }
    const float inverse = online[1] > 0.f ? 1.f / online[1] : 0.f;
    if constexpr (Slices > 0) {
        #pragma unroll
        for (int slice = 0; slice < Slices; ++slice) {
            const int64_t d = tid + slice * 256;
            if (d < width) store_attention(accum[slice] * inverse, output, ob + d,
                batch, token, head, d, epilogue);
        }
    } else {
        for (int64_t d = tid; d < width; d += 256)
            store_attention(output[ob + d] * inverse, output, ob + d,
                batch, token, head, d, epilogue);
    }
}

static Tensor sparse_impl(const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& selected, double scale, bool v_is_k_view,
    const Tensor& gate, bool half_output, bool fused_query = false, mfq::cuda::HotKvView hot = {},
    mfq::cuda::IndirectKvView indirect = {}) {
    if (indirect.states) {
        values({&q, &k, &v});
        MFQ_RUNTIME_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes() &&
            q.size(0) == indirect.rows && q.size(2) == 1 && k.size(0) == 1 &&
            q.size(1) % k.size(1) == 0 && q.size(3) == k.size(3), "invalid indirect attention geometry");
    } else {
        attention_shapes(q, k, v);
    }
    MFQ_RUNTIME_CHECK(selected.defined() && selected.is_cuda() &&
        selected.device() == q.device() && selected.dim() == 3 &&
        selected.size(0) == q.size(0) && selected.size(1) == q.size(2) &&
        selected.size(2) > 0, "sparse indices must have [B,T,topk] shape");
    MFQ_RUNTIME_CHECK(selected.scalar_type() == tb::kInt32 ||
        selected.scalar_type() == tb::kInt64, "sparse indices must be integer");
    MFQ_RUNTIME_CHECK(std::isfinite(scale), "attention scale must be finite");
    MfqCudaGuard guard(q.device());
    const bool use_qsa_mma = !v_is_k_view && q.size(3) == 256;
    const bool use_mla_mma = v_is_k_view && q.size(3) == 512;
    // The shared MMA tiles cast every query element to FP16 after applying the
    // softmax scale, so a strided F32/F16/BF16 query yields the same products in
    // place as a pre-converted contiguous FP32 copy. MLA (D = 512, V reuses the
    // K cache) always reads it directly; QSA keeps that behind its fusion
    // switch. MFQ_MLA_SPARSE_INLINE_SELECTION=0 restores the prepared index and
    // mask path, which stays the fallback for QSA tiles and every other width.
    const char* inline_selection = std::getenv("MFQ_MLA_SPARSE_INLINE_SELECTION");
    const bool inline_mla = use_mla_mma && (!inline_selection || inline_selection[0] != '0');
    const bool direct_query = inline_mla ||
        (fused_query && use_qsa_mma && (q.size(1) == 24 || selected.size(2) <= 512) &&
            (q.scalar_type() == tb::kFloat16 || q.scalar_type() == tb::kFloat32));
    auto query = direct_query ? q : q.to(tb::kFloat32).contiguous();
    auto key = k.to(tb::kFloat16).contiguous();
    auto value = v_is_k_view ? key : v.to(tb::kFloat16).contiguous();
    auto indices = selected.to(tb::kInt32).contiguous();
    if (q.size(0) == 0 || q.size(2) == 0) {
        return tb::empty({q.size(0), q.size(2), q.size(1), q.size(3)},
            q.options().dtype(tb::kFloat32));
    }
    if (key.size(2) == 0) {
        return tb::zeros({q.size(0), q.size(2), q.size(1), q.size(3)},
            q.options().dtype(tb::kFloat32));
    }

    if (inline_mla) {
        // The attention tile rejects negative, out-of-range and padded slots
        // from their sentinel value, so the selection reaches it unpadded and
        // without a materialized FP16 mask or FP32 query copy.
        MFQ_RUNTIME_CHECK(indices.size(2) <= std::numeric_limits<int>::max() &&
            key.size(2) <= std::numeric_limits<int>::max(),
            "sparse attention exceeds 32-bit index geometry");
        return mfq_sparse_attn_mma::launch<512, 512, 1, 16, true>(
            query, key, key, indices, Tensor{}, Tensor{}, Tensor{}, scale,
            "glm5_sparse_mla_attention", gate, half_output, Tensor{}, 0, hot,
            {}, true, key.size(2));
    }
    if (use_qsa_mma || use_mla_mma) {
        constexpr int index_tile = 32;
        const int64_t source_count = indices.size(2);
        MFQ_RUNTIME_CHECK(
            source_count <= std::numeric_limits<int>::max() - index_tile &&
            key.size(2) <= std::numeric_limits<int>::max(),
            "sparse attention exceeds 32-bit index geometry");
        const int64_t padded_count =
            ((source_count + index_tile - 1) / index_tile) * index_tile;
        auto shape = indices.sizes().vec();
        shape[2] = padded_count;
        auto safe_indices = tb::empty(shape, indices.options());
        auto mask = tb::empty(shape, query.options().dtype(tb::kFloat16));
        MFQ_RUNTIME_CHECK(
            indices.size(0) <= std::numeric_limits<int64_t>::max() /
                indices.size(1),
            "sparse attention row count is too large");
        const int64_t rows = indices.size(0) * indices.size(1);
        MFQ_RUNTIME_CHECK(
            rows <= std::numeric_limits<int64_t>::max() / padded_count,
            "sparse attention index workspace is too large");
        const int64_t total = rows * padded_count;
        const int blocks = static_cast<int>(std::min<int64_t>(
            65535, std::max<int64_t>(1, (total + 255) / 256)));
        prepare_sparse_indices_kernel<<<blocks, 256, 0, mfq_current_cuda_stream()>>>(
            indices.data_ptr<int32_t>(), safe_indices.data_ptr<int32_t>(),
            reinterpret_cast<half *>(mask.data_ptr<mfq_half>()), rows,
            static_cast<int>(source_count), static_cast<int>(padded_count),
            static_cast<int>(key.size(2)), indirect);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        if (use_qsa_mma) {
            if(hot.mapping || indirect.states)return mfq_sparse_attn_mma::launch<256,256,1,16,false,true>(
                query,key,value,safe_indices,mask,Tensor{},Tensor{},scale,
                "qwen4_sparse_gqa_attention_hot",gate,half_output,Tensor{},0,hot,indirect);
            return mfq_sparse_attn_mma::launch<256, 256, 1, 16, false>(
                query, key, value, safe_indices, mask, Tensor{}, Tensor{},
                scale, "qwen4_sparse_gqa_attention", gate, half_output, Tensor{}, 0, {}, indirect);
        }
        return mfq_sparse_attn_mma::launch<512, 512, 1, 16, true>(
            query, key, key, safe_indices, mask, Tensor{}, Tensor{},
            scale, "glm5_sparse_mla_attention");
    }

    auto output = tb::empty({q.size(0), q.size(2), q.size(1), q.size(3)},
        q.options().dtype(tb::kFloat32));
    if (output.numel() == 0) return output;
    const int64_t blocks = output.numel() / q.size(3);
    MFQ_RUNTIME_CHECK(blocks <= std::numeric_limits<int>::max(),
        "sparse attention exceeds CUDA grid size");
    const auto stream = mfq_current_cuda_stream();
    #define MFQ_LAUNCH_SPARSE(S,H) sparse_attention_kernel<S,H><<<unsigned(blocks), 256, 0, stream>>>( \
        query.data_ptr<float>(), reinterpret_cast<const half*>(key.data_ptr()), \
        reinterpret_cast<const half*>(value.data_ptr()), indices.data_ptr<int32_t>(), \
        output.data_ptr<float>(), q.size(1), k.size(1), q.size(2), q.size(3), \
        k.size(2), selected.size(2), static_cast<float>(scale), false, nullptr, false, AttentionEpilogue{},hot,indirect)
    if (q.size(3) <= 256) { if(hot.mapping){MFQ_LAUNCH_SPARSE(1,true);}else{MFQ_LAUNCH_SPARSE(1,false);} }
    else if (q.size(3) <= 512) { if(hot.mapping){MFQ_LAUNCH_SPARSE(2,true);}else{MFQ_LAUNCH_SPARSE(2,false);} }
    else if (q.size(3) <= 1024) { if(hot.mapping){MFQ_LAUNCH_SPARSE(4,true);}else{MFQ_LAUNCH_SPARSE(4,false);} }
    else { if(hot.mapping){MFQ_LAUNCH_SPARSE(0,true);}else{MFQ_LAUNCH_SPARSE(0,false);} }
    #undef MFQ_LAUNCH_SPARSE
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

Tensor sparse(const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& selected, double scale, bool v_is_k_view, mfq::cuda::HotKvView hot,
    mfq::cuda::IndirectKvView indirect) {
    return sparse_impl(q, k, v, selected, scale, v_is_k_view, Tensor{}, false, false, hot, indirect);
}

__global__ void attention_gated_kernel(const void* attended, bool half_input,
    AttentionLayout layout, AttentionEpilogue epilogue, int64_t count,
    int64_t heads, int64_t tokens, int64_t width) {
    for (int64_t linear = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < count; linear += int64_t(gridDim.x) * blockDim.x) {
        const int64_t column = linear % width, row = linear / width;
        const int64_t head = row % heads, token = (row / heads) % tokens,
            batch = row / (heads * tokens);
        const auto at = batch * layout.batch + token * layout.token + head * layout.head + column * layout.column;
        const float value = half_input ? __half2float(static_cast<const half*>(attended)[at])
            : static_cast<const float*>(attended)[at];
        store_attention(value, nullptr, linear, batch, token, head, column, epilogue);
    }
}

AttentionLayout attention_layout(const Tensor& x) {
    return {x.stride(0), x.stride(1), x.stride(2), x.stride(3)};
}

bool half_or_float(const Tensor& x) {
    return x.scalar_type() == tb::kFloat16 || x.scalar_type() == tb::kFloat32;
}

Tensor gated(const Tensor& attended, const Tensor& gate, bool half_output) {
    values({&attended, &gate});
    MFQ_RUNTIME_CHECK(attended.dim() == 4 && attended.sizes() == gate.sizes(),
        "attention gate requires matching [B,T,H,D] inputs");
    if (!half_or_float(attended) || !half_or_float(gate))
        return (attended.to(tb::kFloat32) * tb::sigmoid(gate.to(tb::kFloat32)))
            .to(half_output ? tb::kFloat16 : tb::kFloat32);
    MfqCudaGuard guard(attended.device());
    auto output = tb::empty(attended.sizes(), attended.options().dtype(half_output ? tb::kFloat16 : tb::kFloat32));
    if (!output.numel()) return output;
    const AttentionEpilogue epilogue{gate.data_ptr(), output.data_ptr(),
        gate.scalar_type() == tb::kFloat32, half_output,
        gate.scalar_type() == tb::kFloat32 && !gate.is_contiguous(), attention_layout(gate)};
    const int blocks = int(std::min<int64_t>(65535, (output.numel() + 255) / 256));
    attention_gated_kernel<<<blocks, 256, 0, mfq_current_cuda_stream()>>>(attended.data_ptr(),
        attended.scalar_type() == tb::kFloat16, attention_layout(attended), epilogue,
        output.numel(), output.size(2), output.size(1), output.size(3));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

Tensor sparse_gated(const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& indices, const Tensor& gate, double scale, bool half_output, mfq::cuda::HotKvView hot) {
    attention_shapes(q, k, v);
    values({&q, &gate});
    MFQ_RUNTIME_CHECK(gate.dim() == 4 && gate.size(0) == q.size(0) &&
        gate.size(1) == q.size(2) && gate.size(2) == q.size(1) &&
        gate.size(3) == q.size(3), "sparse attention gate geometry mismatch");
    // The QSA MMA epilogue combines stream-K normalization, gate and cast.
    // Keep the shared selected-attention arithmetic for all other widths.
    if (q.size(3) != 256 || !half_or_float(gate) || !q.numel() || !k.size(2))
        return gated(sparse(q, k, v, indices, scale, false,hot), gate, half_output);
    const char* fused_query = std::getenv("MFQ_QSA_SPARSE_QUERY_FUSED");
    return sparse_impl(q, k, v, indices, scale, false, gate, half_output,
        !fused_query || fused_query[0] != '0',hot);
}

template<class BlockIndex, class Position>
__global__ void block_indices_to_tokens_kernel(const BlockIndex* blocks,
    const Position* positions, int32_t* output, int64_t rows, int tokens,
    int block_count, int pool, int budget, int columns) {
    for (int64_t linear = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
         linear < rows*columns; linear += int64_t(gridDim.x)*blockDim.x) {
        const int slot = int(linear%columns);
        const int64_t row = linear/columns;
        const int64_t position = positions[row%tokens];
        int64_t selected = -1;
        if (position+1 <= budget) {
            if (slot <= position) selected = slot;
        } else if (int64_t(slot) < int64_t(block_count)*pool) {
            const int64_t block = blocks[row*block_count+slot/pool];
            if (block*pool+pool-1 <= position) selected = block*pool+slot%pool;
        } else if (slot >= budget) {
            const int offset = slot-budget;
            const int64_t tail = (position+1)%pool;
            if (offset < tail) selected = position+1-tail+offset;
        }
        output[linear] = int32_t(selected);
    }
}

Tensor block_indices_to_tokens(const Tensor& blocks, const Tensor& positions,
    int64_t pool, int64_t budget) {
    MFQ_RUNTIME_CHECK(blocks.defined() && positions.defined() && blocks.is_cuda() &&
        positions.is_cuda() && blocks.device() == positions.device() &&
        blocks.dim() == 3 && positions.dim() == 1 && positions.numel() == blocks.size(1) &&
        (blocks.scalar_type() == tb::kInt32 || blocks.scalar_type() == tb::kInt64) &&
        (positions.scalar_type() == tb::kInt32 || positions.scalar_type() == tb::kInt64) &&
        pool > 0 && budget > 0 && pool <= INT_MAX && budget <= INT_MAX-pool &&
        blocks.size(1) <= INT_MAX && blocks.size(2) <= budget/pool,
        "block-selected token geometry mismatch");
    MfqCudaGuard guard(blocks.device());
    const int64_t columns = budget+pool-1;
    auto output = tb::empty({blocks.size(0),blocks.size(1),columns},blocks.options().dtype(tb::kInt32));
    if (!output.numel()) return output;
    auto compact_blocks = blocks.contiguous(), compact_positions = positions.contiguous();
    const int grid = int(std::min<int64_t>(65535,(output.numel()+255)/256));
    const auto launch = [&](auto block_type,auto position_type) {
        using Block = decltype(block_type); using Position = decltype(position_type);
        block_indices_to_tokens_kernel<<<grid,256,0,mfq_current_cuda_stream()>>>(
            compact_blocks.data_ptr<Block>(),compact_positions.data_ptr<Position>(),output.data_ptr<int32_t>(),
            blocks.size(0)*blocks.size(1),int(blocks.size(1)),int(blocks.size(2)),int(pool),int(budget),int(columns));
    };
    if (blocks.scalar_type() == tb::kInt64) {
        if (positions.scalar_type() == tb::kInt64) launch(int64_t{},int64_t{});
        else launch(int64_t{},int32_t{});
    } else {
        if (positions.scalar_type() == tb::kInt64) launch(int32_t{},int64_t{});
        else launch(int32_t{},int32_t{});
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void prepare_causal_attention_kernel(const void* q, bool half_query,
    AttentionLayout layout, float* query, int64_t query_count, int64_t heads,
    int64_t tokens, int64_t width, const void* positions, bool wide_positions,
    int32_t* indices, half* mask, int64_t index_count, int64_t columns,
    int64_t padded_columns, int64_t capacity) {
    const int64_t first = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = int64_t(gridDim.x) * blockDim.x;
    for (int64_t linear = first; linear < query_count; linear += step) {
        const int64_t column = linear % width, row = linear / width;
        const int64_t token = row % tokens, head = (row / tokens) % heads,
            batch = row / (tokens * heads);
        const auto at = batch * layout.batch + token * layout.token +
            head * layout.head + column * layout.column;
        query[linear] = half_query ? __half2float(static_cast<const half*>(q)[at])
            : static_cast<const float*>(q)[at];
    }
    for (int64_t linear = first; linear < index_count; linear += step) {
        const int64_t index = linear % padded_columns,
            token = (linear / padded_columns) % tokens;
        const int64_t end = wide_positions ? static_cast<const int64_t*>(positions)[token]
            : static_cast<const int32_t*>(positions)[token];
        const bool valid = index < columns && index <= end && index < capacity;
        indices[linear] = valid ? int32_t(index) : 0;
        mask[linear] = __float2half_rn(valid ? 0.f : -CUDART_INF_F);
    }
}

__global__ void mma_reduce_gate_serial_kernel(const float* partial, const float2* metadata,
    const void* gate, void* output, AttentionLayout gate_layout, int tokens, int heads,
    int kv_heads, int width, int total_blocks, int parts, int query_columns, int head_columns,
    bool float_gate, bool half_output, bool double_sigmoid) {
    const int head=blockIdx.x%heads, row=blockIdx.x/heads;
    const int token=row%tokens, batch=row/tokens, lane=threadIdx.x;
    const int grouped_heads=heads/kv_heads;
    const int head_tiles=(grouped_heads+head_columns-1)/head_columns;
    const int token_tiles=(tokens+query_columns-1)/query_columns;
    const int tile=(((batch*kv_heads+head/grouped_heads)*head_tiles+
        (head%grouped_heads)/head_columns)*token_tiles+token/query_columns);
    const int tile_column=(token%query_columns)*head_columns+(head%grouped_heads)%head_columns;
    const int columns=query_columns*head_columns, last=(tile+1)*parts-1;
    const size_t output_base=size_t(row)*heads*width+head*width;
    const float* contributions=reinterpret_cast<const float*>(metadata)+size_t(total_blocks)*4*columns;
    extern __shared__ float2 factors[];
    __shared__ float normalizer;
    if(lane==0) {
        float denominator=1.f;
        if(parts>1) {
            auto statistics=metadata[size_t(last)*columns+tile_column];
            float peak=statistics.x;
            denominator=statistics.y;
            for(int step=1;step<parts;++step) {
                statistics=metadata[size_t(total_blocks+last-step)*columns+tile_column];
                const float maximum=fmaxf(peak,statistics.x);
                const float previous=peak-maximum, incoming=statistics.x-maximum;
                const float lhs=previous>=-20.f?expf(previous):0.f;
                const float rhs=incoming>=-20.f?expf(incoming):0.f;
                factors[step-1]=make_float2(lhs,rhs);
                denominator=denominator*lhs+statistics.y*rhs;
                peak=maximum;
            }
        }
        normalizer=denominator;
    }
    __syncthreads();
    for(int column=lane;column<width;column+=blockDim.x) {
        float result=partial[output_base+column];
        for(int step=1;step<parts;++step) {
            const float incoming=contributions[(size_t(last-step)*columns+tile_column)*width+column];
            const auto weights=factors[step-1];
            result=result*weights.x+incoming*weights.y;
        }
        if(parts>1)result=normalizer!=0.f?result/normalizer:0.f;
        const auto gate_index=int64_t(batch)*gate_layout.batch+int64_t(token)*gate_layout.token+
            int64_t(head)*gate_layout.head+int64_t(column)*gate_layout.column;
        const float g=float_gate?static_cast<const float*>(gate)[gate_index]
            :__half2float(static_cast<const half*>(gate)[gate_index]);
        result=apply_attention_gate(result,g,double_sigmoid);
        if(half_output)static_cast<half*>(output)[output_base+column]=__float2half_rn(result);
        else static_cast<float*>(output)[output_base+column]=result;
    }
}

__global__ void mma_reduce_gate_kernel(const float* partial,const float2* metadata,
    const void* gate,void* output,AttentionLayout gate_layout,int tokens,int heads,
    int kv_heads,int width,int total_blocks,int parts,int query_columns,int head_columns,
    bool float_gate,bool half_output,bool double_sigmoid) {
    const int head=blockIdx.x%heads,row=blockIdx.x/heads;
    const int token=row%tokens,batch=row/tokens;
    const int grouped_heads=heads/kv_heads;
    const int head_tiles=(grouped_heads+head_columns-1)/head_columns;
    const int token_tiles=(tokens+query_columns-1)/query_columns;
    const int tile=(((batch*kv_heads+head/grouped_heads)*head_tiles+
        (head%grouped_heads)/head_columns)*token_tiles+token/query_columns);
    const int tile_column=(token%query_columns)*head_columns+(head%grouped_heads)%head_columns;
    const int columns=query_columns*head_columns,last=(tile+1)*parts-1;
    const size_t output_base=size_t(row)*heads*width+head*width;
    const float* contributions=reinterpret_cast<const float*>(metadata)+size_t(total_blocks)*4*columns;
    // Like the llama.cpp stream-K fixup, each output lane combines statistics
    // and values together. Avoid a serial statistics pass and CTA barrier.
    for(int column=threadIdx.x;column<width;column+=blockDim.x) {
        float result=partial[output_base+column],denominator=1.f,peak=0.f;
        if(parts>1) {
            auto statistics=metadata[size_t(last)*columns+tile_column];
            peak=statistics.x;denominator=statistics.y;
            for(int step=1;step<parts;++step) {
                const float incoming=contributions[(size_t(last-step)*columns+tile_column)*width+column];
                statistics=metadata[size_t(total_blocks+last-step)*columns+tile_column];
                const float maximum=fmaxf(peak,statistics.x);
                const float previous=peak-maximum,next=statistics.x-maximum;
                const float lhs=previous>=-20.f?expf(previous):0.f;
                const float rhs=next>=-20.f?expf(next):0.f;
                result=result*lhs+incoming*rhs;
                denominator=denominator*lhs+statistics.y*rhs;
                peak=maximum;
            }
            result=denominator!=0.f?result/denominator:0.f;
        }
        const auto at=int64_t(batch)*gate_layout.batch+int64_t(token)*gate_layout.token+
            int64_t(head)*gate_layout.head+int64_t(column)*gate_layout.column;
        const float g=float_gate?static_cast<const float*>(gate)[at]:__half2float(static_cast<const half*>(gate)[at]);
        result=apply_attention_gate(result,g,double_sigmoid);
        if(half_output)static_cast<half*>(output)[output_base+column]=__float2half_rn(result);
        else static_cast<float*>(output)[output_base+column]=result;
    }
}

Tensor mma_reduce_gated(const Tensor& partial,const Tensor& metadata,const Tensor& gate,
    bool half_output,int kv_heads,int total_blocks,int parts,int query_columns,int head_columns) {
    values({&partial,&metadata,&gate});
    MFQ_RUNTIME_CHECK(partial.dim()==4 && partial.sizes()==gate.sizes() &&
        partial.scalar_type()==tb::kFloat32 && metadata.scalar_type()==tb::kFloat32 &&
        partial.is_contiguous() && metadata.is_contiguous() && half_or_float(gate) &&
        kv_heads>0 && partial.size(2)%kv_heads==0 && total_blocks>0 && parts>0 &&
        query_columns>0 && head_columns>0,"MMA gate reduction geometry mismatch");
    const int64_t heads=partial.size(2),tokens=partial.size(1),width=partial.size(3);
    const int64_t tiles=partial.size(0)*kv_heads*((heads/kv_heads+head_columns-1)/head_columns)*
        ((tokens+query_columns-1)/query_columns);
    MFQ_RUNTIME_CHECK(tokens>0 && heads>0 && width>0 && tiles*parts==total_blocks &&
        (parts==1 || metadata.numel()>=int64_t(total_blocks)*query_columns*head_columns*(4+width)),
        "MMA gate reduction metadata mismatch");
    MFQ_RUNTIME_CHECK(partial.numel()/width<=INT_MAX && width<=INT_MAX && tokens<=INT_MAX && heads<=INT_MAX,
        "MMA gate reduction exceeds CUDA geometry");
    const size_t shared=size_t(parts-1)*sizeof(float2);
    MFQ_RUNTIME_CHECK(shared+sizeof(float)<=48*1024,"MMA gate reduction shared memory exceeded");
    MfqCudaGuard guard(partial.device());
    auto output=tb::empty(partial.sizes(),partial.options().dtype(half_output?tb::kFloat16:tb::kFloat32));
    const char* parallel=std::getenv("MFQ_QSA_PARALLEL_REDUCE");
    const bool parallel_reduce=!parallel || parallel[0]!='0';
    const auto kernel=parallel_reduce?mma_reduce_gate_kernel:mma_reduce_gate_serial_kernel;
    if(output.numel())kernel<<<unsigned(output.numel()/width),256,parallel_reduce?0:shared,mfq_current_cuda_stream()>>>(
        partial.data_ptr<float>(),reinterpret_cast<const float2*>(metadata.data_ptr<float>()),
        gate.data_ptr(),output.data_ptr(),attention_layout(gate),int(tokens),int(heads),kv_heads,
        int(width),total_blocks,parts,query_columns,head_columns,gate.scalar_type()==tb::kFloat32,
        half_output,gate.scalar_type()==tb::kFloat32 && !gate.is_contiguous());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

Tensor causal_gated(const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& positions, const Tensor& gate, int64_t columns, double scale,
    bool half_output, bool fused_mma_gate, bool fused_mma_prepare, mfq::cuda::HotKvView hot) {
    attention_shapes(q, k, v);
    values({&q, &gate});
    MFQ_RUNTIME_CHECK(gate.dim() == 4 && gate.size(0) == q.size(0) && gate.size(1) == q.size(2) &&
        gate.size(2) == q.size(1) && gate.size(3) == q.size(3) && positions.defined() &&
        positions.is_cuda() && positions.device() == q.device() && positions.dim() == 1 &&
        positions.numel() == q.size(2) && (positions.scalar_type() == tb::kInt32 || positions.scalar_type() == tb::kInt64) &&
        columns > 0 && columns <= INT_MAX && std::isfinite(scale), "causal attention gate geometry mismatch");
    MfqCudaGuard guard(q.device());
    auto cache_positions = positions.contiguous();
    if (!half_or_float(q) || !half_or_float(gate) || (q.size(3) == 256 && !k.size(2))) {
        auto selected = tb::arange(columns, positions.options()).reshape({1, 1, columns})
            .expand({q.size(0), q.size(2), columns});
        selected = tb::where(selected <= cache_positions.reshape({1, q.size(2), 1}),
            selected, tb::full_like(selected, -1));
        return gated(sparse(q, k, v, selected, scale, false,hot), gate, half_output);
    }
    if (q.size(3) == 256) {
        auto key = k.to(tb::kFloat16).contiguous(), value = v.to(tb::kFloat16).contiguous();
        if(hot.mapping && q.numel()) {
            auto attended=mfq_sparse_attn_mma::launch<256,256,1,16,false,true>(q,key,value,
                Tensor{},Tensor{},Tensor{},Tensor{},scale,"causal_gqa_attention_hot",
                fused_mma_gate?gate:Tensor{},half_output,cache_positions,columns,hot);
            return fused_mma_gate?attended:gated(attended,gate,half_output);
        }
        if (fused_mma_prepare && q.numel()) {
            auto attended = mfq_sparse_attn_mma::launch<256, 256, 1, 16, false>(
                q, key, value, Tensor{}, Tensor{}, Tensor{}, Tensor{}, scale,
                "causal_gqa_attention_gate", fused_mma_gate ? gate : Tensor{}, half_output,
                cache_positions, columns);
            return fused_mma_gate ? attended : gated(attended, gate, half_output);
        }
        // Prepare the original MMA query, safe indices and validity mask together.
        MFQ_RUNTIME_CHECK(columns <= INT_MAX - 32, "causal MMA index geometry exceeds 32 bits");
        const int64_t padded_columns = ((columns + 31) / 32) * 32;
        auto query = tb::empty(q.sizes(), q.options().dtype(tb::kFloat32));
        auto indices = tb::empty({q.size(0), q.size(2), padded_columns}, q.options().dtype(tb::kInt32));
        auto mask = tb::empty(indices.sizes(), q.options().dtype(tb::kFloat16));
        const int blocks = int(std::min<int64_t>(65535, (std::max(query.numel(), indices.numel()) + 255) / 256));
        if (!blocks) return tb::empty({q.size(0), q.size(2), q.size(1), q.size(3)},
            q.options().dtype(half_output ? tb::kFloat16 : tb::kFloat32));
        const auto layout = AttentionLayout{q.stride(0), q.stride(2), q.stride(1), q.stride(3)};
        prepare_causal_attention_kernel<<<blocks, 256, 0, mfq_current_cuda_stream()>>>(
            q.data_ptr(), q.scalar_type() == tb::kFloat16, layout, query.data_ptr<float>(), query.numel(),
            q.size(1), q.size(2), q.size(3), cache_positions.data_ptr(), cache_positions.scalar_type() == tb::kInt64,
            indices.data_ptr<int32_t>(), reinterpret_cast<half*>(mask.data_ptr()), indices.numel(),
            columns, padded_columns, k.size(2));
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        auto attended = mfq_sparse_attn_mma::launch<256, 256, 1, 16, false>(
            query, key, value, indices, mask, Tensor{}, Tensor{}, scale, "qwen4_causal_gqa_attention_gate",
            fused_mma_gate?gate:Tensor{},half_output);
        if(fused_mma_gate)return attended;
        return gated(attended, gate, half_output);
    }
    auto query = q.contiguous();
    auto key = k.to(tb::kFloat16).contiguous(), value = v.to(tb::kFloat16).contiguous();
    auto output = tb::empty({q.size(0), q.size(2), q.size(1), q.size(3)},
        q.options().dtype(half_output ? tb::kFloat16 : tb::kFloat32));
    if (!output.numel()) return output;
    auto scratch = q.size(3) > 1024 ? tb::empty(output.sizes(), q.options().dtype(tb::kFloat32)) : Tensor{};
    const auto blocks = output.numel() / q.size(3);
    MFQ_RUNTIME_CHECK(blocks <= INT_MAX, "causal attention exceeds CUDA grid size");
    const AttentionEpilogue epilogue{gate.data_ptr(), output.data_ptr(),
        gate.scalar_type() == tb::kFloat32, half_output,
        gate.scalar_type() == tb::kFloat32 && !gate.is_contiguous(), attention_layout(gate)};
    #define MFQ_LAUNCH_CAUSAL(S,H) sparse_attention_kernel<S,H><<<unsigned(blocks), 256, 0, mfq_current_cuda_stream()>>>( \
        query.data_ptr(), reinterpret_cast<const half*>(key.data_ptr()), reinterpret_cast<const half*>(value.data_ptr()), \
        nullptr, scratch.defined() ? scratch.data_ptr<float>() : nullptr, q.size(1), k.size(1), q.size(2), q.size(3), \
        k.size(2), columns, float(scale), q.scalar_type() == tb::kFloat16, cache_positions.data_ptr(), \
        cache_positions.scalar_type() == tb::kInt64, epilogue,hot,{})
    if (q.size(3) <= 256) { if(hot.mapping){MFQ_LAUNCH_CAUSAL(1,true);}else{MFQ_LAUNCH_CAUSAL(1,false);} }
    else if (q.size(3) <= 512) { if(hot.mapping){MFQ_LAUNCH_CAUSAL(2,true);}else{MFQ_LAUNCH_CAUSAL(2,false);} }
    else if (q.size(3) <= 1024) { if(hot.mapping){MFQ_LAUNCH_CAUSAL(4,true);}else{MFQ_LAUNCH_CAUSAL(4,false);} }
    else { if(hot.mapping){MFQ_LAUNCH_CAUSAL(0,true);}else{MFQ_LAUNCH_CAUSAL(0,false);} }
    #undef MFQ_LAUNCH_CAUSAL
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

Tensor dense(const Tensor& query, const Tensor& key, const Tensor& value,
                       int64_t offset, double scale, mfq::cuda::HotKvView hot) {
    attention_shapes(query, key, value);
    const auto count = key.size(2);
    MFQ_RUNTIME_CHECK(count > 0 && offset >= 0 && offset <= count &&
        query.size(2) <= count - offset, "attention query range is outside the cache");
    MFQ_RUNTIME_CHECK(std::isfinite(scale), "attention scale must be finite");
    MfqCudaGuard guard(query.device());
    if(hot.mapping) {
        auto positions=tb::arange(offset,offset+query.size(2),query.options().dtype(tb::kInt64));
        if(query.size(3)==256)return mfq_sparse_attn_mma::launch<256,256,1,16,false,true>(
            query,key,value,Tensor{},Tensor{},Tensor{},Tensor{},scale,"dense_gqa_attention_hot",
            Tensor{},false,positions,offset+query.size(2),hot);
        auto selected=tb::arange(offset+query.size(2),positions.options()).reshape({1,1,-1}).expand({1,query.size(2),offset+query.size(2)});
        selected=tb::where(selected<=positions.reshape({1,-1,1}),selected,tb::full_like(selected,-1));
        return sparse(query,key,value,selected,scale,false,hot);
    }
    const auto options = query.options().dtype(tb::kInt64);
    const auto kp = tb::arange(count, options).unsqueeze(0);
    const auto qp = tb::arange(query.size(2), options).unsqueeze(1) + offset;
    auto mask = (kp <= qp).unsqueeze(0).unsqueeze(0);
    return tb::scaled_dot_product_attention(query, key.to(query.scalar_type()),
        value.to(query.scalar_type()), mask, 0.0, false, scale, true).permute({0, 2, 1, 3});
}

} // namespace mfq_selected_attention
