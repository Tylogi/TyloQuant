#pragma once

#include "mfq_cuda_context.h"

#include <algorithm>
#include <limits>

namespace mfq::cuda {

// Ordinary temporary tensors can reuse idle default-pool pages already charged
// to WDDM. Private graph/resident pools are not available to these allocations.
inline std::size_t available_workspace_memory(const Context& context) {
    const auto available=context.available_memory();
    if(!context.supports_async_allocations())return available;
    const auto memory=context.memory_stats();
    if(memory.limit && context.local_memory_usage()>memory.limit)return available;
    DeviceGuard guard(context.device());
    cudaMemPool_t pool=nullptr;
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,context.device()));
    std::uint64_t used=0,reserved=0;
    MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemCurrent,&used));
    MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&reserved));
    const auto idle=std::size_t(reserved>used?reserved-used:0);
    auto bytes=idle>std::numeric_limits<std::size_t>::max()-available
        ?std::numeric_limits<std::size_t>::max():available+idle;
    if(memory.limit)bytes=std::min(bytes,memory.limit-std::min(memory.limit,memory.allocated));
    return bytes;
}

} // namespace mfq::cuda
