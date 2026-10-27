#include "qwen4_session_check.h"
#include "models/qwen4_exp/mtp.h"
#include "storage/model_loader.h"
#include "storage/session_state.h"
#include <numeric>
#include <cstring>

namespace mfq::cuda::tests {
std::vector<Tensor> qwen4_session_check(const nlohmann::json& p) {
    namespace tb=mfq_tensor_backend;
    CudaExecutionContext execution;
    auto model=load_causal_lm<Qwen4CausalLm>(execution,p.at("model"),"",32);
    model.reuse_decode_graphs=true;
    const auto gpu=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
    const auto ids=[&](const std::vector<int64_t>& value){return tb::tensor(value,gpu).reshape({1,-1});};
    const auto equal=[](const Tensor& left,const Tensor& right) {
        if(left.sizes()!=right.sizes() || left.scalar_type()!=right.scalar_type())return false;
        auto a=left.to(tb::kCPU).contiguous(),b=right.to(tb::kCPU).contiguous();
        return std::memcmp(a.data_ptr(),b.data_ptr(),size_t(a.numel())*a.element_size())==0;
    };
    std::vector<int64_t> prefix(p.value("prefix",7));std::iota(prefix.begin(),prefix.end(),1);
    model.reset(1);Tensor raw;
    Tensor supplied;
    if(p.value("axes",1)==3) {
        supplied=tb::arange(prefix.size(),gpu).reshape({1,-1})+tb::arange(3,gpu).reshape({3,1});
    }
    auto hidden=model.hidden_forward(ids(prefix),supplied.defined()?std::optional<Tensor>(supplied):std::nullopt,
        {},nullptr,{},&raw);
    if(supplied.defined())model.decode_position_delta=4;
    if(p.value("graph",false))for(int64_t token:{8,9}) {
        auto graph=qwen4_decode_graph_hidden(model,ids({token}),&raw);
        MFQ_RUNTIME_CHECK(graph.has_value(),"session test did not capture a target graph");prefix.push_back(token);
    }
    auto saved=model.capture_text_session_state(prefix);
    MFQ_RUNTIME_CHECK(saved.kind()==TextSessionStateKind::Qwen4 && saved.bytes>0,"missing Qwen4 snapshot");
    const bool paged=p.value("paged",false);const size_t block_size=p.value("block_size",4);
    MFQ_RUNTIME_CHECK(block_size>0,"invalid paged session test block size");
    int codec_rejected=0;
    const auto roundtrip=[&](const TextSessionState& state) {
        std::vector<CudaPagedPayload> blocks;
        for(size_t i=0;i<(state.tokens.size()+block_size-1)/block_size;++i)
            blocks.push_back(encode_cuda_qwen4_paged_block(state,block_size,i));
        MFQ_RUNTIME_CHECK(cuda_paged_decodable_blocks(blocks)==blocks.size(),"missing Qwen4 exact tail");
        for(int failure=0;failure<4;++failure) {
            auto broken=blocks;auto contents=std::make_shared<std::vector<uint8_t>>(*blocks.front());
            if(failure==0)contents->resize(7);
            else if(failure==1)(*contents)[0]^=0x40;
            else if(failure==2)(*contents)[28]=2;
            else contents->push_back(0);
            broken.front()=contents;bool caught=false;
            try {decode_cuda_qwen4_paged_session(broken,state.tokens,block_size);}
            catch(const CudaSessionStateError&){caught=true;}
            MFQ_RUNTIME_CHECK(caught,"corrupt Qwen4 paged payload was accepted");++codec_rejected;
        }
        return decode_cuda_qwen4_paged_session(blocks,state.tokens,block_size);
    };
    if(paged)saved=roundtrip(saved);
    const std::vector<int64_t> suffix{10,11,12};
    auto expected=model.hidden_forward(ids(suffix)).to(tb::kCPU).clone();
    model.reset(2);
    model.hidden_forward(tb::tensor(std::vector<int64_t>{3,4,5,5,4,3},gpu).reshape({2,3}));
    model.restore_text_session_state(saved);
    auto actual=model.hidden_forward(ids(suffix)).to(tb::kCPU).clone();
    MFQ_RUNTIME_CHECK(equal(expected,actual),"restored Qwen4 target output differs");
    int rejected=0;
    const auto reject=[&](TextSessionState invalid) {
        model.restore_text_session_state(saved);const auto before=model.cache_pos;
        bool caught=false;
        try {model.restore_text_session_state(invalid);}catch(const CudaSessionStateError&){caught=true;}
        MFQ_RUNTIME_CHECK(caught && model.cache_pos==before,"invalid Qwen4 state changed the active cache");
        auto after=model.hidden_forward(ids(suffix)).to(tb::kCPU);
        MFQ_RUNTIME_CHECK(equal(after,expected),"failed Qwen4 restore changed recurrent state");++rejected;
    };
    {auto bad=saved;bad.tokens.pop_back();reject(std::move(bad));}
    {auto bad=saved;bad.qwen4_positions=bad.qwen4_positions.to(tb::kFloat32);reject(std::move(bad));}
    {auto bad=saved;auto& layers=std::get<std::vector<Qwen4BlockSessionState>>(bad.payload);
     auto& last=layers.back();last.keys=last.keys.narrow(2,0,last.keys.size(2)-1);reject(std::move(bad));}
    {auto bad=saved;auto& layers=std::get<std::vector<Qwen4BlockSessionState>>(bad.payload);
     layers.front().recurrent=layers.front().recurrent.to(tb::kFloat16);reject(std::move(bad));}
    if(std::get<std::vector<Qwen4BlockSessionState>>(saved.payload).front().ple_batch) {
        auto bad=saved;std::get<std::vector<Qwen4BlockSessionState>>(bad.payload).front().ple_tokens.pop_back();
        reject(std::move(bad));
    }
    auto predictor=qwen4_exp::Qwen4ExpMtp::load_if_present(execution,*model.source,model.config);
    MFQ_RUNTIME_CHECK(predictor && predictor->supports_session_state(),"missing Qwen4 predictor session support");
    predictor->reset(1);const MtpTarget target(model);const auto width=predictor->hidden_norm.numel();
    auto previous=(tb::arange(prefix.size()*width,gpu).remainder(31).to(tb::kFloat32)*.01-.15)
        .reshape({1,int64_t(prefix.size()),width});
    for(int64_t depth=0;depth<int64_t(predictor->layers.size());++depth)
        predictor->evaluate(target,previous,ids(prefix),depth);
    const auto last=previous.narrow(1,previous.size(1)-1,1).clone();
    auto predictor_saved=predictor->capture_session_state(prefix.size()+1,last);
    if(paged) {
        model.restore_text_session_state(saved);model.hidden_forward(ids({17}));
        auto tokens=prefix;tokens.push_back(17);auto holder=model.capture_text_session_state(tokens);
        const auto tail=(tokens.size()-1)/block_size;
        auto without_mtp=encode_cuda_qwen4_paged_block(holder,block_size,tail);
        holder.mtp=predictor_saved;
        MFQ_RUNTIME_CHECK(cuda_paged_checkpoint_needs_update(without_mtp,holder),"Qwen4 tail did not request predictor upgrade");
        auto with_mtp=encode_cuda_qwen4_paged_block(holder,block_size,tail);
        MFQ_RUNTIME_CHECK(!cuda_paged_checkpoint_needs_update(with_mtp,holder),"Qwen4 predictor tail requests redundant rewrite");
        auto decoded=roundtrip(holder);MFQ_RUNTIME_CHECK(decoded.mtp.has_value(),"Qwen4 predictor was not persisted");
        predictor_saved=std::move(*decoded.mtp);
    }
    auto future=(tb::arange(2*width,gpu).remainder(29).to(tb::kFloat32)*.02-.2).reshape({1,2,width});
    auto predictor_expected=predictor->evaluate(target,future,ids({13,14})).first.to(tb::kCPU).clone();
    predictor->reset(2);predictor->restore_session_state(predictor_saved);
    auto predictor_actual=predictor->evaluate(target,future,ids({13,14})).first.to(tb::kCPU).clone();
    MFQ_RUNTIME_CHECK(equal(predictor_expected,predictor_actual),"restored Qwen4 predictor output differs");
    auto invalid=predictor_saved;invalid.qwen4_lengths[0]++;
    bool caught=false;try{predictor->restore_session_state(invalid);}catch(const CudaSessionStateError&){caught=true;}
    MFQ_RUNTIME_CHECK(caught,"invalid Qwen4 predictor snapshot was accepted");
    return {expected,actual,predictor_expected,predictor_actual,
        tb::tensor(std::vector<int64_t>{rejected+1+codec_rejected,int64_t(saved.bytes),int64_t(predictor_saved.bytes)})};
}
}
