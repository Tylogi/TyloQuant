#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "deepseek_v41_index.h"
#include <climits>
#include <cstdint>
#include <cfloat>
#include <algorithm>
#include <tuple>

namespace {

__global__ void candidate_positions8_kernel(
    const int32_t *__restrict__ blocks, int64_t *__restrict__ output,
    int64_t pairs, int64_t pool_length) {
    const int64_t pair = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    const int64_t base = int64_t(blocks[pair >> 2]) * 8 + (pair & 3) * 2;
    reinterpret_cast<longlong2 *>(output)[pair] = make_longlong2(
        base >= 0 && base < pool_length ? base : -1,
        base + 1 >= 0 && base + 1 < pool_length ? base + 1 : -1);
}

template<bool Indexed>
__global__ void finalize_index_topk_kernel(
    const half *__restrict__ scores, const int32_t *__restrict__ relative,
    const int64_t *__restrict__ source, int32_t *__restrict__ output,
    int columns, int pool_length) {
    __shared__ int exchange[2][512];
    const int row = blockIdx.x, tid = threadIdx.x;
    int keys[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int index = relative[int64_t(row) * 512 + tid + i * 256];
        int key = pool_length;
        if (index >= 0 && index < columns) {
            const auto at = int64_t(row) * columns + index;
            const int64_t selected = Indexed ? source[at] : int64_t(index);
            const float value = __half2float(scores[at]);
            if (selected >= 0 && selected < pool_length && value != -INFINITY) {
                key = int(selected);
            }
        }
        keys[i] = key;
    }
    int buffer = 0;
#pragma unroll
    for (int size = 2; size <= 512; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride; stride >>= 1) {
            int other[2];
            if (stride < 64) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    other[i] = __shfl_xor(keys[i], stride, 64);
                }
            } else if (stride == 256) {
                other[0] = keys[1]; other[1] = keys[0];
            } else {
                exchange[buffer][tid] = keys[0];
                exchange[buffer][tid + 256] = keys[1];
                __syncthreads();
                other[0] = exchange[buffer][tid ^ stride];
                other[1] = exchange[buffer][(tid ^ stride) + 256];
                // Alternating buffers let the next stage reuse the barrier.
                buffer ^= 1;
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int logical = tid + i * 256;
                const bool minimum = ((logical & size) == 0) == ((logical & stride) == 0);
                keys[i] = minimum ? min(keys[i], other[i]) : max(keys[i], other[i]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        output[int64_t(row) * 512 + tid + i * 256] = keys[i] < pool_length ? keys[i] : -1;
    }
}

template<bool AlignedRows>
__global__ void block_scores8_kernel(
    const half *__restrict__ scores, half *__restrict__ output,
    int columns, int blocks, int tokens, int ratio, int64_t position) {
    const int block = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (block >= blocks) return;
    const int token = blockIdx.y;
    const auto row = int64_t(blockIdx.z) * tokens + token;
    const auto source = row * columns + int64_t(block) * 8;
    float maximum = -FLT_MAX;
    if constexpr (AlignedRows) {
        union Values { uint4 packed; half scalar[8]; } values;
        values.packed = *reinterpret_cast<const uint4 *>(scores + source);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float value = __half2float(values.scalar[i]);
            // Match generic amax: ignore NaN, retain the first signed zero.
            if (value > maximum) maximum = value;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float value = int64_t(block) * 8 + i < columns
                ? __half2float(scores[source + i]) : -INFINITY;
            if (value > maximum) maximum = value;
        }
    }
    const auto visible = (position + token + 1) / ratio;
    const auto last = visible > 0 ? (visible - 1) / 8 : int64_t(-1);
    output[row * blocks + block] = __float2half_rn(block == last ? INFINITY : maximum);
}

__global__ void mask_candidate_blocks_kernel(
    const half *__restrict__ values, const int64_t *__restrict__ selected,
    int32_t *__restrict__ output, int64_t total) {
    const int64_t at = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (at >= total) return;
    if (at + 1 < total) {
        const float2 value = __half22float2(*reinterpret_cast<const half2 *>(values + at));
        const longlong2 index = *reinterpret_cast<const longlong2 *>(selected + at);
        *reinterpret_cast<int2 *>(output + at) = make_int2(
            value.x != -INFINITY ? int(index.x) : -1,
            value.y != -INFINITY ? int(index.y) : -1);
    } else {
        output[at] = __half2float(values[at]) != -INFINITY ? int(selected[at]) : -1;
    }
}

__device__ __forceinline__ unsigned ordered_half_i16(unsigned bits) {
    // Float radix sort treats both signed zeros as equivalent, preserving IDs.
    return (bits & 0x7fff) == 0 ? 0 : bits ^ ((bits & 0x8000) ? 0x7fff : 0);
}

__global__ void block_scores_to_keys16_kernel(
    const half *__restrict__ scores, int16_t *__restrict__ output, int64_t total) {
    const int64_t at = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (at >= total) return;
    if (at + 1 < total) {
        const unsigned packed = *reinterpret_cast<const unsigned *>(scores + at);
        *reinterpret_cast<unsigned *>(output + at) =
            ordered_half_i16(packed & 0xffff) | (ordered_half_i16(packed >> 16) << 16);
    } else {
        output[at] = int16_t(ordered_half_i16(__half_as_ushort(scores[at])));
    }
}

__global__ void mask_ordered_blocks16_kernel(
    const int16_t *__restrict__ values, const int64_t *__restrict__ selected,
    int32_t *__restrict__ output, int64_t total) {
    const int64_t at = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (at >= total) return;
    constexpr unsigned negative_infinity = 0x83ff;
    if (at + 1 < total) {
        const unsigned packed = *reinterpret_cast<const unsigned *>(values + at);
        const longlong2 index = *reinterpret_cast<const longlong2 *>(selected + at);
        *reinterpret_cast<int2 *>(output + at) = make_int2(
            (packed & 0xffff) != negative_infinity ? int(index.x) : -1,
            (packed >> 16) != negative_infinity ? int(index.y) : -1);
    } else {
        output[at] = uint16_t(values[at]) != negative_infinity ? int(selected[at]) : -1;
    }
}

} // namespace

mfq_tensor_backend::Tensor deepseek_v41_candidate_positions8_cuda(
    mfq_tensor_backend::Tensor blocks, std::int64_t pool_length) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(blocks.is_cuda() && blocks.is_contiguous() && blocks.dim() == 3 &&
        blocks.scalar_type() == tb::kInt32 && pool_length >= 0 && pool_length <= INT_MAX &&
        blocks.numel() <= int64_t(INT_MAX) * 64,
        "DeepSeek candidate positions expects contiguous CUDA int32 [B,T,K] blocks");
    auto output = tb::empty({blocks.size(0), blocks.size(1), blocks.size(2) * 8},
                            blocks.options().dtype(tb::kInt64));
    const auto pairs = output.numel() / 2;
    if (pairs) {
        const auto grid = static_cast<unsigned>((pairs + 255) / 256);
        candidate_positions8_kernel<<<grid, 256, 0, mfq_current_cuda_stream()>>>(
            blocks.data_ptr<int32_t>(), output.data_ptr<int64_t>(), pairs, pool_length);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

mfq_tensor_backend::Tensor deepseek_v41_block_scores8_cuda(
    mfq_tensor_backend::Tensor scores, std::int64_t ratio, std::int64_t position) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(scores.is_cuda() && scores.is_contiguous() && scores.dim() == 3 &&
        scores.scalar_type() == tb::kFloat16 && scores.size(2) > 0 && scores.size(2) <= INT_MAX &&
        scores.size(0) > 0 && scores.size(0) <= 65535 &&
        scores.size(1) > 0 && scores.size(1) <= 65535 &&
        ratio > 0 && ratio <= INT_MAX && position >= 0 && position <= INT_MAX - scores.size(1),
        "DeepSeek block8 scores expects contiguous CUDA f16 [B,T,K]");
    const int columns = int(scores.size(2));
    const int blocks = int((int64_t(columns) + 7) / 8);
    auto output = tb::empty({scores.size(0), scores.size(1), blocks}, scores.options());
    const dim3 grid((blocks + 255) / 256, unsigned(scores.size(1)), unsigned(scores.size(0)));
#define MFQ_BLOCK_SCORES8(ALIGNED) \
    block_scores8_kernel<ALIGNED><<<grid, 256, 0, mfq_current_cuda_stream()>>>( \
        reinterpret_cast<const half *>(scores.data_ptr<mfq_half>()), \
        reinterpret_cast<half *>(output.data_ptr<mfq_half>()), columns, blocks, \
        int(scores.size(1)), int(ratio), position)
    if (columns % 8 == 0 && reinterpret_cast<std::uintptr_t>(scores.data_ptr()) % 16 == 0) {
        MFQ_BLOCK_SCORES8(true);
    } else {
        MFQ_BLOCK_SCORES8(false);
    }
#undef MFQ_BLOCK_SCORES8
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

mfq_tensor_backend::Tensor deepseek_v41_candidate_blocks8_f16_cuda(
    mfq_tensor_backend::Tensor scores, std::int64_t ratio,
    std::int64_t position, std::int64_t requested_blocks) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(requested_blocks >= 0, "DeepSeek candidate block count must be nonnegative");
    auto block_scores = deepseek_v41_block_scores8_cuda(scores, ratio, position);
    const auto count = std::min(requested_blocks, block_scores.size(2));
    auto result = tb::topk(block_scores, count, -1, true, false);
    const auto &values = std::get<0>(result);
    const auto &selected = std::get<1>(result);
    auto output = tb::empty(selected.sizes(), selected.options().dtype(tb::kInt32));
    const auto total = output.numel();
    if (total) {
        mask_candidate_blocks_kernel<<<unsigned((total + 511) / 512), 256, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const half *>(values.data_ptr<mfq_half>()),
            selected.data_ptr<int64_t>(), output.data_ptr<int32_t>(), total);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

mfq_tensor_backend::Tensor deepseek_v41_candidate_blocks8_ordered16_cuda(
    mfq_tensor_backend::Tensor scores, std::int64_t ratio,
    std::int64_t position, std::int64_t requested_blocks) {
    namespace tb = mfq_tensor_backend;
    const auto blocks = scores.dim() == 3 ? (scores.size(2) + 7) / 8 : 0;
    const auto count = std::min(requested_blocks, blocks);
    const bool full_sort = count > 256 && (count > 2048 || count * 4 > blocks);
    if (!full_sort) {
        return deepseek_v41_candidate_blocks8_f16_cuda(scores, ratio, position, requested_blocks);
    }
    auto block_scores = deepseek_v41_block_scores8_cuda(scores, ratio, position);
    auto keys = tb::empty(block_scores.sizes(), block_scores.options().dtype(tb::kInt16));
    const auto total_keys = block_scores.numel();
    block_scores_to_keys16_kernel<<<unsigned((total_keys + 511) / 512), 256, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half *>(block_scores.data_ptr<mfq_half>()),
        keys.data_ptr<int16_t>(), total_keys);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    // Use the existing native Int16 segmented sort: no FP32 score conversion.
    auto result = tb::topk(keys, count, -1, true, false);
    const auto &values = std::get<0>(result);
    const auto &selected = std::get<1>(result);
    auto output = tb::empty(selected.sizes(), selected.options().dtype(tb::kInt32));
    const auto total = output.numel();
    mask_ordered_blocks16_kernel<<<unsigned((total + 511) / 512), 256, 0, mfq_current_cuda_stream()>>>(
        values.data_ptr<int16_t>(), selected.data_ptr<int64_t>(), output.data_ptr<int32_t>(), total);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}



mfq_tensor_backend::Tensor deepseek_v41_finalize_index_topk_cuda(
    mfq_tensor_backend::Tensor scores, mfq_tensor_backend::Tensor relative,
    mfq_tensor_backend::Tensor source_indices, std::int64_t pool_length) {
    namespace tb = mfq_tensor_backend;
    const bool indexed = source_indices.defined();
    MFQ_RUNTIME_CHECK(scores.is_cuda() && scores.is_contiguous() && scores.dim() == 3 &&
        scores.scalar_type() == tb::kFloat16 && scores.size(2) > 512 &&
        scores.size(2) <= INT_MAX && scores.size(0) * scores.size(1) <= INT_MAX &&
        relative.device() == scores.device() && relative.is_contiguous() && relative.dim() == 3 &&
        relative.scalar_type() == tb::kInt32 && relative.size(0) == scores.size(0) &&
        relative.size(1) == scores.size(1) && relative.size(2) == 512 &&
        pool_length >= 0 && pool_length <= INT_MAX &&
        (!indexed || (source_indices.device() == scores.device() && source_indices.is_contiguous() &&
         source_indices.scalar_type() == tb::kInt64 && source_indices.sizes() == scores.sizes())),
        "DeepSeek index topk finalize expects f16 scores and int32 top512 indices");
    auto output = tb::empty(relative.sizes(), relative.options());
    const int rows = int(scores.size(0) * scores.size(1));
    if (rows) {
#define MFQ_FINALIZE_TOPK(INDEXED, SOURCE) \
        finalize_index_topk_kernel<INDEXED><<<rows, 256, 0, mfq_current_cuda_stream()>>>( \
            reinterpret_cast<const half *>(scores.data_ptr<mfq_half>()), relative.data_ptr<int32_t>(), \
            SOURCE, output.data_ptr<int32_t>(), int(scores.size(2)), int(pool_length))
        if (indexed) { MFQ_FINALIZE_TOPK(true, source_indices.data_ptr<int64_t>()); }
        else { MFQ_FINALIZE_TOPK(false, nullptr); }
#undef MFQ_FINALIZE_TOPK
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}
