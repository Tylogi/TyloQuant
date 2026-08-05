#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace mfq {

struct alignas(8) MoeCacheScatterDescriptor {
    std::uint64_t destination = 0;
    std::uint64_t source_offset = 0;
    std::uint64_t nbytes = 0;
};

static_assert(
    sizeof(MoeCacheScatterDescriptor) == 24,
    "MoE cache scatter descriptor layout changed");

void moe_cache_scatter_cuda(
    const std::uint8_t * staging,
    std::int64_t descriptor_offset,
    int transfer_count,
    cudaStream_t stream);

}  // namespace mfq
