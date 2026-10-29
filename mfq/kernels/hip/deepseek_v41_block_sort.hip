#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipcub/device/device_segmented_radix_sort.hpp>
#include "deepseek_v41_index.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstddef>

namespace {
__device__ __forceinline__ unsigned ordered_half_i16(unsigned bits) {
    // Float radix sort treats both signed zeros as equivalent, preserving IDs.
    return (bits & 0x7fff) == 0 ? 0 : bits ^ ((bits & 0x8000) ? 0x7fff : 0);
}

__global__ void pack_block_keys32_kernel(
    const half *__restrict__ scores, uint32_t *__restrict__ output,
    int32_t *__restrict__ offsets, int blocks, int tokens, int rows, int index_bits) {
    const int column = int(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row = int(blockIdx.z) * tokens + int(blockIdx.y);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        offsets[row] = row * blocks;
        if (row == rows - 1) offsets[rows] = rows * blocks;
    }
    if (column >= blocks) return;
    const auto at = int64_t(row) * blocks + column;
    const unsigned ordinal = ordered_half_i16(__half_as_ushort(scores[at])) ^ 0x8000u;
    const unsigned mask = (1u << index_bits) - 1;
    output[at] = (ordinal << index_bits) | (mask - unsigned(column));
}

__global__ void decode_block_keys32_kernel(
    const uint32_t *__restrict__ sorted, int32_t *__restrict__ output,
    int blocks, int count, int tokens, int index_bits) {
    const int column = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= count) return;
    const int row = int(blockIdx.z) * tokens + int(blockIdx.y);
    const unsigned key = sorted[int64_t(row) * blocks + column];
    const unsigned mask = (1u << index_bits) - 1;
    output[int64_t(row) * count + column] = (key >> index_bits) == 0x03ff
        ? -1 : int(mask - (key & mask));
}

} // namespace

mfq_tensor_backend::Tensor deepseek_v41_candidate_blocks8_cuda(
    mfq_tensor_backend::Tensor scores, std::int64_t ratio,
    std::int64_t position, std::int64_t requested_blocks) {
    namespace tb = mfq_tensor_backend;
    const auto blocks = scores.dim() == 3 ? (scores.size(2) + 7) / 8 : 0;
    const auto count = std::min(requested_blocks, blocks);
    const bool full_sort = count > 256 && (count > 2048 || count * 4 > blocks);
    const auto rows = scores.dim() == 3 ? scores.size(0) * scores.size(1) : 0;
    if (!full_sort || blocks > 4096 || rows * blocks > INT_MAX) {
        return deepseek_v41_candidate_blocks8_ordered16_cuda(scores, ratio, position, requested_blocks);
    }
    auto block_scores = deepseek_v41_block_scores8_cuda(scores, ratio, position);
    auto packed = tb::empty(block_scores.sizes(), block_scores.options().dtype(tb::kInt32));
    auto sorted = tb::empty(packed.sizes(), packed.options());
    auto offsets = tb::empty({rows + 1}, packed.options());
    const int index_bits = 32 - __builtin_clz(unsigned(blocks - 1));
    const auto stream = mfq_current_cuda_stream();
    const dim3 grid(unsigned((blocks + 255) / 256), unsigned(scores.size(1)), unsigned(scores.size(0)));
    pack_block_keys32_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const half *>(block_scores.data_ptr<mfq_half>()),
        reinterpret_cast<uint32_t *>(packed.data_ptr<int32_t>()), offsets.data_ptr<int32_t>(),
        int(blocks), int(scores.size(1)), int(rows), index_bits);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    std::size_t bytes = 0;
#define MFQ_SORT_PACKED_BLOCKS(WORKSPACE) \
    hipcub::DeviceSegmentedRadixSort::SortKeysDescending( \
        WORKSPACE, bytes, reinterpret_cast<const uint32_t *>(packed.data_ptr<int32_t>()), \
        reinterpret_cast<uint32_t *>(sorted.data_ptr<int32_t>()), int(rows * blocks), int(rows), \
        offsets.data_ptr<int32_t>(), offsets.data_ptr<int32_t>() + 1, 0, 16 + index_bits, stream)
    MFQ_RUNTIME_CHECK(MFQ_SORT_PACKED_BLOCKS(nullptr) == hipSuccess, "DeepSeek packed block sort size query failed");
    auto workspace = tb::empty({int64_t(std::max<std::size_t>(bytes, 1))}, packed.options().dtype(tb::kUInt8));
    MFQ_RUNTIME_CHECK(MFQ_SORT_PACKED_BLOCKS(workspace.data_ptr<uint8_t>()) == hipSuccess,
                      "DeepSeek packed block sort failed");
#undef MFQ_SORT_PACKED_BLOCKS
    auto output = tb::empty({scores.size(0), scores.size(1), count}, packed.options());
    const dim3 final_grid(unsigned((count + 255) / 256), unsigned(scores.size(1)), unsigned(scores.size(0)));
    decode_block_keys32_kernel<<<final_grid, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t *>(sorted.data_ptr<int32_t>()), output.data_ptr<int32_t>(),
        int(blocks), int(count), int(scores.size(1)), index_bits);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
