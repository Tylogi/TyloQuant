#pragma once
#include "tq.h"
#include "storage/moe_expert_cache.h"

namespace mfq::cuda {
// Resource ownership follows the loaded bank implementation. The service uses
// these operations without interpreting a tensor's row precision or encoding.
inline std::vector<std::pair<std::string,double>> routed_expert_memory_metrics(
        const CudaExecutionContext& execution) {
    return execution.tq_expert_cache?tq_expert_memory_metrics(execution.tq_expert_cache):
        moe_expert_memory_metrics(execution.moe_expert_cache);
}
inline std::size_t routed_expert_cache_budget_bytes(const CudaExecutionContext& execution) {
    return execution.tq_expert_cache?tq_expert_cache_budget_bytes(execution.tq_expert_cache):
        moe_expert_cache_budget_bytes(execution.moe_expert_cache);
}
inline void resize_routed_expert_cache(CudaExecutionContext& execution,std::size_t bytes) {
    if(execution.tq_expert_cache)resize_tq_expert_cache(execution.tq_expert_cache,bytes);
    else resize_moe_expert_cache(execution.moe_expert_cache,bytes);
}
inline std::size_t reclaim_routed_optional_host(CudaExecutionContext& execution,std::size_t allocation_bytes) {
    return execution.tq_expert_cache?reserve_tq_expert_host(execution.tq_expert_cache,allocation_bytes):
        reclaim_moe_optional_host(execution.moe_expert_cache,allocation_bytes);
}
}
