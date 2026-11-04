#pragma once

#include "cuda_execution.h"
#include "mfq_cuda_context.h"
#include "storage/moe_expert_cache.h"
#include "prefill_retry.h"
#include "prefill_budget.h"

#include <iostream>
#include <set>

namespace mfq::cuda::internal {

template<class Model> std::size_t prefill_moe_row_bytes(const Model& model) {
    const auto routes=[&]()->std::size_t {
        if constexpr(requires { model.config.num_experts_per_tok; })return model.config.num_experts_per_tok;
        else if constexpr(requires { model.config.topk; })return model.config.topk;
        else if constexpr(requires { model.config.top_k; })return model.config.top_k;
        return 0;
    }();
    // Serial layers share one activation workspace; do not multiply by depth.
    return mfq::engine::prefill_budget_multiply(routes,model.execution->prefill_moe_route_bytes);
}

// Call only after failed forward frames and graph consumers have retired.
// The engine callback returns optional prefix/expert pages without touching KV.
template<class Model> void reclaim_prefill_memory(Model& model) {
    if(model.reclaim_prefill_workspace)model.reclaim_prefill_workspace();
    release_moe_decode_buffers(model.execution->moe_expert_cache);
    std::set<int> devices(model.execution->layer_placement.devices.begin(),
        model.execution->layer_placement.devices.end());
    devices.insert(model.execution->tensor_parallel.devices.begin(),model.execution->tensor_parallel.devices.end());
    devices.insert(model.execution->expert_parallel.devices.begin(),model.execution->expert_parallel.devices.end());
    devices.insert(model.execution->layer_placement.primary_device());
    for(const auto device:devices)default_context(device)->reclaim_idle_memory();
}

// Rebuild only this uncommitted row. Its original scheduling quantum completes
// in smaller tiles, so other rows and scheduler progress retain their positions.
template<class Ops,class Reclaim>
auto retry_batch_prefill(Ops& ops,const std::shared_ptr<typename Ops::Request>& request,
        mfq::engine::PrefillChunk chunk,Reclaim reclaim,std::exception_ptr failure={}) {
    using Sample=typename Ops::Sample;
    auto width=request->prefill_retry_width>0?std::min(chunk.count,request->prefill_retry_width):chunk.count;
    const auto run=[&](int64_t start) {
        Sample result;
        const auto end=chunk.offset+chunk.count;
        for(auto offset=start;offset<end;) {
            const auto count=std::min(width,end-offset);
            result=ops.prefill_once(request,{offset,count,offset+count==int64_t(request->prompt.size())});
            offset+=count;
        }
        return result;
    };
    if(!failure) {
        try { return run(chunk.offset); }
        catch(...) { failure=std::current_exception(); }
    }
    if(!mfq::engine::prefill_resource_failure(failure))std::rethrow_exception(failure);
    for(;;) {
        ops.discard_prefill(request);
        reclaim();
        width=mfq::engine::smaller_prefill_chunk(width,failure);
        request->prefill_retry_width=width;
        std::cerr<<"runtime_prefill_retry scope=row chunk="<<width<<" replay_tokens="<<chunk.offset+chunk.count<<'\n';
        try {
            return run(0);
        } catch(...) {
            failure=std::current_exception();
            if(!mfq::engine::prefill_resource_failure(failure))std::rethrow_exception(failure);
        }
    }
}

} // namespace mfq::cuda::internal
