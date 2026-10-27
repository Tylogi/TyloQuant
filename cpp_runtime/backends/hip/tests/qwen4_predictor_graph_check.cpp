#include "qwen4_predictor_graph_check.h"
#include "models/qwen4_exp/mtp.h"
#include "storage/model_loader.h"
#include "mfq_hip_sampling_ops.h"

namespace mfq::cuda::tests {
std::vector<Tensor> qwen4_predictor_graph_check(const std::vector<Tensor>& a,const nlohmann::json& p) {
    namespace tb=mfq_tensor_backend;
    CudaExecutionContext execution;
    auto model=load_causal_lm<Qwen4CausalLm>(execution,p.at("model"),"",32);
    auto eager=qwen4_exp::Qwen4ExpMtp::load_if_present(execution,*model.source,model.config);
    auto graph=qwen4_exp::Qwen4ExpMtp::load_if_present(execution,*model.source,model.config);
    MFQ_RUNTIME_CHECK(eager && graph,"predictor graph fixture has no head");
    const MtpTarget target(model);std::vector<Tensor> results;
    eager->reset(1);graph->reset(1);
    eager->evaluate(target,a.at(0),a.at(1));graph->evaluate(target,a.at(0),a.at(1));
    std::vector<int64_t> counters;
    const auto check=[&](const Tensor& hidden,const Tensor& ids) {
        auto expected=eager->evaluate(target,hidden,ids);
        auto actual=graph->step(target,hidden,ids);
        const auto rows=expected.first.size(1);
        auto reference_logits=target.logits(expected.first.narrow(1,rows-1,1).contiguous());
        auto actual_logits=actual.sample_logits.defined()?actual.sample_logits:
            target.logits(actual.sample_hidden.narrow(1,rows-1,1).contiguous());
        results.insert(results.end(),{expected.first,actual.sample_hidden,expected.second,actual.chain_hidden,
            reference_logits,actual_logits});
        if(actual.greedy_token>=0) {
            auto expected_token=sample_greedy_cuda(reference_logits.reshape({1,-1}).contiguous()).item<int64_t>();
            MFQ_RUNTIME_CHECK(expected_token==actual.greedy_token,"captured predictor greedy token differs");
        }
        counters.insert(counters.end(),{int64_t(graph->graph_captures),int64_t(graph->graph_replays)});
    };
    check(a.at(2),a.at(3));check(a.at(4),a.at(5));
    const auto before=eager->cache_position();
    auto independent_eager=eager->evaluate(target,a.at(2),a.at(3),0,false);
    auto independent_graph=graph->evaluate(target,a.at(2),a.at(3),0,false);
    MFQ_RUNTIME_CHECK(before==eager->cache_position() && before==graph->cache_position(),"uncached predictor changed cache length");
    results.insert(results.end(),{independent_eager.first,independent_graph.first,
        independent_eager.second,independent_graph.second});
    check(a.at(2).narrow(1,0,1).contiguous(),a.at(3).narrow(1,0,1).contiguous());
    eager->reset(2);graph->reset(2);
    auto prefix=a.at(0).expand({2,a.at(0).size(1),a.at(0).size(2)}).contiguous();
    auto ids=a.at(1).expand({2,a.at(1).size(1)}).contiguous();
    auto expected=eager->evaluate(target,prefix,ids),actual=graph->evaluate(target,prefix,ids);
    results.insert(results.end(),{expected.first,actual.first,expected.second,actual.second});
    results.push_back(tb::tensor(counters).reshape({3,2}));return results;
}
}
