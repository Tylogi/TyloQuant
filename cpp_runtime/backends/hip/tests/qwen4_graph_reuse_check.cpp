#include "engine/generation.h"
#include "models/qwen4_exp/mtp.h"
#include "storage/model_loader.h"
#include <nlohmann/json.hpp>

namespace mfq::cuda::tests {
std::vector<Tensor> qwen4_graph_reuse(const nlohmann::json& p) {
    CudaExecutionContext execution;
    auto model=load_causal_lm<Qwen4CausalLm>(execution,p.at("model"),"",32);
    model.reuse_decode_graphs=p.value("reuse",false);
    const bool speculative=p.value("mtp",false);
    auto predictor=speculative?qwen4_exp::Qwen4ExpMtp::load_if_present(execution,*model.source,model.config):
        std::optional<qwen4_exp::Qwen4ExpMtp>{};
    const auto prompts=p.at("prompts").get<std::vector<std::vector<int64_t>>>();
    const int outputs=p.value("outputs",12);
    const auto cpu=TensorOptions().device(kCPU).dtype(kInt64);
    const auto gpu=TensorOptions().device(kCUDA).dtype(kInt64);
    std::vector<int64_t> tokens,stats;Tensor batch_hidden;std::vector<Tensor> old_buffers;
    for(size_t index=0;index<prompts.size();++index) {
        std::fprintf(stderr,"graph_reuse_fixture request=%zu reuse=%d mtp=%d captures=%lld\n",index,
            int(model.reuse_decode_graphs),int(speculative),(long long)model.graph_captures);
        if(index==3) {
            model.reset(2);
            auto ids=mfq_tensor_backend::tensor(std::vector<int64_t>{1,2,3,7,6,5},gpu).reshape({2,3});
            batch_hidden=model.hidden_forward(ids).to(kCPU).clone();
        }
        model.reset(1);
        std::fprintf(stderr,"graph_reuse_fixture reset_states=%zu\n",model.blocks.front()->graph_warmup_state().size());
        if(speculative) {
            MFQ_RUNTIME_CHECK(predictor,"Qwen4 graph reuse fixture has no MTP head");predictor->reset(1);
            mfq::engine::InferenceRequest request;request.prompt=prompts[index];request.chat=false;request.benchmark_fixed_tokens=true;
            request.sampling.temperature=0;request.sampling.max_tokens=outputs;request.sampling.top_k=1;request.sampling.enable_mtp=true;
            mfq::engine::InferenceOutput output(request,nullptr,"qwen4-graph-reuse-check");
            auto generation=run_mtp_generation(model,*predictor,request,output,4);
            size_t count=0;
            while(auto next=generation.next())if(next.value)
                if(const auto* delta=std::get_if<mfq::engine::TokenOutput>(&*next.value)) {
                    count+=delta->token_ids.size();tokens.insert(tokens.end(),delta->token_ids.begin(),delta->token_ids.end());
                }
            MFQ_RUNTIME_CHECK(count==size_t(outputs),"graph reuse MTP output was incomplete");
        } else {
            auto current=mfq_tensor_backend::tensor(prompts[index],gpu).reshape({1,-1});
            for(int token=0;token<outputs;++token) {
                auto next=model.next_token(current);
                tokens.push_back(next.template item<int64_t>());current=next.reshape({1,1});
            }
        }
        stats.insert(stats.end(),{model.graph_captures,model.graph_retained_resets,
            model.graph_cross_request_replays,model.graph_binding_invalidations});
        if(index==1 && p.value("replace_state",false)) {
            const auto states=model.blocks.front()->graph_warmup_state();
            MFQ_RUNTIME_CHECK(!states.empty(),"state replacement fixture has no recurrent tensor");
            // Preserve old storage during this diagnostic. A faulty binding
            // check can produce a mismatch without accessing freed memory.
            old_buffers.push_back(*states.front());*states.front()=states.front()->clone();
        }
    }
    return {mfq_tensor_backend::tensor(tokens,cpu).reshape({int64_t(prompts.size()),outputs}),
        mfq_tensor_backend::tensor(stats,cpu).reshape({int64_t(prompts.size()),4}),batch_hidden};
}
} // namespace mfq::cuda::tests
