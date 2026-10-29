#pragma once
#include "mfq_hip_linear_attention_ops.h"
#include <nlohmann/json.hpp>

namespace mfq::cuda::tests {
inline std::unique_ptr<mfq::cuda::qwen4_exp::Gdn> gdn_window_model(
    const std::vector<Tensor>& a,const nlohmann::json& p) {
    using namespace mfq::cuda::qwen4_exp;
    const int nk=p.at("key_heads"),nv=p.at("value_heads"),channels=(2*nk+nv)*128;
    std::vector<Tensor> projected={a.at(0).narrow(2,0,channels),a.at(1).narrow(2,0,nv*128),
        a.at(2).slice(2,0,2*nv,2),a.at(3).slice(2,0,2*nv,2)};
    const auto projection=[&](int index)->Linear {
        const auto value=projected.at(index),base=a.at(0);
        return [value,base](CudaExecutionContext&,const Tensor& x) {
            const auto bytes=base.scalar_type()==mfq_tensor_backend::kFloat16?2:4;
            const auto offset=reinterpret_cast<uintptr_t>(x.data_ptr())-reinterpret_cast<uintptr_t>(base.data_ptr());
            return value.narrow(1,int64_t(offset)/(base.stride(1)*bytes),x.size(1));
        };
    };
    GdnWeights weights{projection(0),projection(1),projection(2),projection(3),
        [](CudaExecutionContext&,const Tensor& x){return x;},a.at(6),a.at(7),a.at(8),a.at(9)};
    auto block=std::make_unique<Gdn>(std::move(weights),nk,nv,128,4,1e-6,p.value("silu_gate",true),
        true,true,true,true,true,p.value("fused_core",true));
    auto states=block->graph_state();*states.at(0)=a.at(4);*states.at(1)=a.at(5);
    block->prepare_graph_checkpoint();return block;
}

inline std::vector<Tensor> gdn_window_model_run(mfq::cuda::qwen4_exp::Gdn& block,CudaExecutionContext& execution,
    const std::vector<Tensor>& a,const nlohmann::json& p) {
    auto output=block.forward_graph(execution,a.at(0),p.value("confirmed",1));
    const auto states=block.graph_state();auto final_conv=states.at(0)->clone(),final_state=states.at(1)->clone();
    block.arm_graph_checkpoint();block.rollback();
    return {output,final_conv,final_state,states.at(0)->clone(),states.at(1)->clone()};
}

inline std::vector<Tensor> gdn_window(const std::vector<Tensor>& a,const nlohmann::json& p) {
    const int nk=p.at("key_heads"),nv=p.at("value_heads"),tokens=a.at(0).size(1);
    const int confirmed=p.value("confirmed",1),channels=(2*nk+nv)*128;
    auto qkv=a.at(0).narrow(2,0,channels),gate=a.at(1).narrow(2,0,nv*128);
    auto alpha=a.at(2),beta=a.at(3);
    if(p.value("strided_gates",false)) {alpha=alpha.slice(2,0,2*nv,2);beta=beta.slice(2,0,2*nv,2);}
    std::vector<Tensor> conv,state;
    for(int i=0;i<tokens-confirmed;++i) {
        const bool omitted=p.value("sparse",false) && i>0;
        conv.push_back(omitted?Tensor{}:a.at(11+2*i));state.push_back(omitted?Tensor{}:a.at(12+2*i));
    }
    if(p.value("bad_alias",false) && !conv.empty())conv[0]=a.at(4);
    if(p.value("bad_overlap",false) && conv.size()>1)conv[1]=conv[0];
    std::vector<Tensor> result;
    if(p.value("legacy",false)) {
        auto history=a.at(4);std::vector<Tensor> outputs;
        for(int t=0;t<tokens;++t) {
            auto decoded=gdn_decode_core_cuda(qkv.narrow(1,t,1),gate.narrow(1,t,1),alpha.narrow(1,t,1),
                beta.narrow(1,t,1),history,a.at(5),a.at(6),a.at(7),a.at(8),a.at(9),nk,nv,128,1e-6,1e-6,
                p.value("silu_gate",true),p.value("output_half",false),p.value("transposed",true),a.at(10),true);
            outputs.push_back(decoded[0]);history=decoded[1];
            const int i=t+1-confirmed;
            if(i>=0 && t+1<tokens && conv[i].defined()) {conv[i].copy_(history);state[i].copy_(decoded[2]);}
        }
        result={mfq_tensor_backend::cat(outputs,1),history,a.at(5)};
    } else result=gdn_decode_window_cuda(qkv,gate,alpha,beta,a.at(4),a.at(5),a.at(6),a.at(7),a.at(8),a.at(9),
        nk,nv,confirmed,1e-6,1e-6,p.value("silu_gate",true),p.value("output_half",false),p.value("transposed",true),conv,state);
    result.push_back(a.at(4));
    for(int i=0;i<tokens-confirmed;++i) {result.push_back(a.at(11+2*i));result.push_back(a.at(12+2*i));}
    return result;
}

inline std::vector<Tensor> gdn_stateful(const std::vector<Tensor>& a,const nlohmann::json& p,
                                       const Tensor& addresses={}) {
    const int64_t nk=p.at("key_heads"),nv=p.at("value_heads"),width=p.at("width");
    const auto channels=(2*nk+nv)*width;
    auto qkv=a.at(0).narrow(2,0,channels),gate=a.at(1).narrow(2,0,nv*width);
    auto alpha=a.at(2),beta=a.at(3);
    if(p.value("strided_gates",false)) {
        alpha=alpha.slice(2,0,2*nv,2);beta=beta.slice(2,0,2*nv,2);
    }
    auto conv=a.at(4),state=a.at(5),workspace=a.at(10);
    if(addresses.defined()) {
        conv=conv.narrow(0,0,1);state=state.narrow(0,0,1);workspace=workspace.narrow(0,0,1);
        auto result=gdn_decode_core_cuda(qkv,gate,alpha,beta,conv,state,a.at(6),a.at(7),a.at(8),a.at(9),
            nk,nv,width,1e-6,p.value("eps",1e-6),p.value("silu_gate",false),p.value("output_half",false),
            p.value("transposed",true),workspace,p.value("inplace",true),addresses);
        return {result.at(0),result.at(1),a.at(5),a.at(4),a.at(10)};
    }
    auto snapshot_conv=p.value("bad_alias",false)?conv:a.at(11);
    std::vector<Tensor> result;
    if(p.value("legacy",false)) {
        result=gdn_decode_core_cuda(qkv,gate,alpha,beta,conv,state,a.at(6),a.at(7),a.at(8),a.at(9),
            nk,nv,width,1e-6,p.value("eps",1e-6),p.value("silu_gate",false),p.value("output_half",false),
            p.value("transposed",true),workspace,false,{});
        snapshot_conv.copy_(result.at(1));auto snapshot_state=a.at(12);snapshot_state.copy_(result.at(2));
    } else result=gdn_decode_core_checkpoint_cuda(qkv,gate,alpha,beta,conv,state,a.at(6),a.at(7),a.at(8),a.at(9),
            nk,nv,width,1e-6,p.value("eps",1e-6),p.value("silu_gate",false),p.value("output_half",false),
            p.value("transposed",true),workspace,p.value("inplace",false),snapshot_conv,a.at(12));
    return {result.at(0),result.at(1),result.at(2),snapshot_conv,a.at(12),a.at(4),a.at(5),a.at(10)};
}

inline Tensor gdn_state_addresses(const std::vector<Tensor>& a,const nlohmann::json& p) {
    const int64_t batch=a.at(0).size(0);
    auto permutation=p.at("permutation").get<std::vector<int64_t>>();
    MFQ_RUNTIME_CHECK(int64_t(permutation.size())==batch,"state permutation size differs");
    auto sorted=permutation;std::sort(sorted.begin(),sorted.end());
    for(int64_t row=0;row<batch;++row)MFQ_RUNTIME_CHECK(sorted[row]==row,"state permutation is invalid");
    std::vector<int64_t> addresses;
    for(const auto row:permutation)for(const int index:{4,5,10})
        addresses.push_back(reinterpret_cast<int64_t>(a.at(index).data_ptr<float>()+row*a.at(index).stride(0)));
    return mfq_tensor_backend::tensor(addresses,TensorOptions().device(a.at(0).device()).dtype(kInt64)).reshape({batch,3});
}
} // namespace mfq::cuda::tests
