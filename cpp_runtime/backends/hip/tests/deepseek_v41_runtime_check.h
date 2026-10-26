#pragma once
#include "models/deepseek_v41/vision.h"
#include "storage/session_state.h"
#include "engine/generation.h"
#include <nlohmann/json.hpp>

namespace mfq::cuda::tests {
inline std::vector<Tensor> dsv41_model(const nlohmann::json& p) {
    CudaExecutionContext execution;
    auto model=load_causal_lm<DeepseekV41CausalLm>(execution,p.at("model"),"",128);
    model.reset(1);
    const auto options=TensorOptions().device(kCUDA).dtype(kInt64);
    auto tokens=p.at("ids").get<std::vector<int64_t>>();
    auto ids=mfq_tensor_backend::tensor(tokens,options).reshape({1,-1});
    const auto mode=p.value("mode",std::string("full"));
    if(mode=="mtp_tokens") {
        auto predictor=deepseek_v41_runtime::load_dspark_if_present(execution,*model.source,model.config);
        MFQ_RUNTIME_CHECK(predictor,"DSpark fixture has no predictor");predictor->reset(1);
        mfq::engine::InferenceRequest request;request.prompt=tokens;request.chat=false;request.benchmark_fixed_tokens=true;
        request.sampling.temperature=0;request.sampling.max_tokens=p.value("outputs",16);request.sampling.enable_mtp=true;
        mfq::engine::InferenceOutput output(request,nullptr,"dspark-native-check");
        std::vector<int64_t> speculative;
        auto generation=run_mtp_generation(model,*predictor,request,output,p.value("chunk",4));
        while(auto step=generation.next())if(step.value)
            if(const auto* delta=std::get_if<mfq::engine::TokenOutput>(&*step.value))
                speculative.insert(speculative.end(),delta->token_ids.begin(),delta->token_ids.end());
        MFQ_RUNTIME_CHECK(speculative.size()==std::size_t(request.sampling.max_tokens),"DSpark generation returned incomplete tokens");
        model.reset(1);std::vector<int64_t> expected;
        auto hidden=deepseek_v41_prefill_hidden(model,ids,p.value("chunk",4));
        for(int index=0;index<request.sampling.max_tokens;++index) {
            auto logits=model.logits_from_hidden(hidden.narrow(1,hidden.size(1)-1,1)).reshape({1,-1}).contiguous();
            auto next=sample_greedy_cuda(logits);expected.push_back(next.item<int64_t>());
            if(index+1<request.sampling.max_tokens)hidden=model.hidden_forward(next.reshape({1,1}));
        }
        return {mfq_tensor_backend::tensor(speculative,options),mfq_tensor_backend::tensor(expected,options),
            mfq_tensor_backend::tensor(std::vector<float>{float(predictor->last_stats.cycles),
                float(predictor->last_stats.drafted_tokens),float(predictor->last_stats.accepted_tokens)},options.dtype(kFloat32))};
    }
    if(mode=="full")return {model.hidden_forward(ids)};
    if(mode=="chunk") {
        const int64_t chunk=p.at("chunk");std::vector<Tensor> parts;
        for(int64_t offset=0;offset<ids.size(1);offset+=chunk)
            parts.push_back(model.hidden_forward(ids.narrow(1,offset,std::min(chunk,ids.size(1)-offset)).contiguous()));
        return {mfq_tensor_backend::cat(parts,1)};
    }
    if(mode=="encoder") {
        const int64_t chunk=p.at("chunk"),layers=p.value("layers",2);
        std::vector<Tensor> hidden_parts,pre_parts;
        for(int64_t offset=0;offset<ids.size(1);offset+=chunk) {
            const auto count=std::min(chunk,ids.size(1)-offset);
            typename DeepseekV41CausalLm::Inputs inputs;
            inputs.ids=ids.narrow(1,offset,count).contiguous();inputs.input_embeddings=model.embed_forward(inputs.ids);
            auto ops=model.forward_ops(model,std::move(inputs));ops.prepare_inputs();
            ops.cache_positions=ops.position_range(model.cache_pos,count);ops.pos=ops.full_positions=ops.cache_positions;
            auto hidden=ops.prepare_hidden(1,count);model.adapter_begin_forward(false);
            for(int64_t layer=0;layer<layers;++layer)hidden=ops.layer(model.blocks.at(layer),std::move(hidden));
            hidden_parts.push_back(hidden.clone());pre_parts.push_back(model.shared->previous_pre.clone());
            model.cache_pos+=count;
        }
        std::vector<Tensor> result{mfq_tensor_backend::cat(hidden_parts,1),mfq_tensor_backend::cat(pre_parts,1)};
        for(int64_t layer=0;layer<layers;++layer) {
            const auto& block=dynamic_cast<const deepseek_v41_runtime::Block&>(*model.blocks.at(layer));
            result.push_back(block.state.local_kv.clone());
            if(block.kv_source()) {
                result.push_back(block.state.compressed_kv.narrow(1,0,block.state.compressed_length).clone());
                result.push_back(block.state.index_k.narrow(1,0,block.state.compressed_length).clone());
            }
        }
        return result;
    }
    if(mode=="ced") {
        auto hidden=deepseek_v41_prefill_hidden(model,ids,p.at("chunk"));
        const auto& block=dynamic_cast<const deepseek_v41_runtime::Block&>(*model.blocks.at(2));
        return {hidden,block.state.compressed_kv.narrow(1,0,block.state.compressed_length).clone(),
            block.state.index_k.narrow(1,0,block.state.compressed_length).clone(),model.shared->topk.clone(),
            model.shared->candidate_blocks.clone()};
    }
    const int64_t prefix=p.value("prefix",5);
    model.hidden_forward(ids.narrow(1,0,prefix).contiguous());
    const std::vector<int64_t> prompt(tokens.begin(),tokens.begin()+prefix);
    auto saved=model.capture_text_session_state(prompt);
    if(p.value("paged",false)) {
        const size_t block=p.value("block_size",4);std::vector<CudaPagedPayload> payloads;
        MFQ_RUNTIME_CHECK(block>0,"invalid DSV4.1 page size");
        for(size_t i=0;i<(prompt.size()+block-1)/block;++i)
            payloads.push_back(encode_cuda_deepseek_v41_paged_block(saved,block,i));
        MFQ_RUNTIME_CHECK(cuda_paged_decodable_blocks(payloads)==payloads.size(),"missing DSV4.1 exact tail");
        for(int failure=0;failure<3;++failure) {
            auto broken=payloads;auto contents=std::make_shared<std::vector<uint8_t>>(*payloads.front());
            if(failure==0)contents->resize(7);
            else if(failure==1)(*contents)[0]^=0x40;
            else (*contents)[16]=(*contents)[17]=(*contents)[18]=(*contents)[19]=0;
            broken.front()=contents;bool caught=false;
            try {decode_cuda_deepseek_v41_paged_session(broken,prompt,block);}
            catch(const CudaSessionStateError&){caught=true;}
            MFQ_RUNTIME_CHECK(caught,"corrupt DSV4.1 paged payload was accepted");
        }
        saved=decode_cuda_deepseek_v41_paged_session(payloads,prompt,block);
    }
    if(mode=="session") {
        auto suffix=ids.narrow(1,prefix,ids.size(1)-prefix).contiguous();
        auto expected=model.hidden_forward(suffix).clone();
        model.restore_text_session_state(saved);
        auto malformed=saved;
        auto& layer=std::get<std::vector<DeepseekV41BlockSessionState>>(malformed.payload).at(1);
        layer.partial_length=99;
        bool rejected=false;
        try {model.restore_text_session_state(malformed);}catch(const CudaSessionStateError&) {rejected=true;}
        MFQ_RUNTIME_CHECK(rejected && model.cache_pos==prefix,"invalid session mutated active state");
        auto actual=model.hidden_forward(suffix);
        return {expected,actual};
    }
    MFQ_RUNTIME_CHECK(mode=="speculate" || mode=="joint","unknown DSV41 test mode");
    auto drafts=ids.narrow(1,prefix,3).contiguous();
    auto next=ids.narrow(1,prefix+3,2).contiguous();
    std::vector<Tensor> result;
    for(int64_t kept=0;kept<=3;++kept) {
        if(mode=="joint" && kept==0)continue;
        model.restore_text_session_state(saved);
        if(mode=="speculate") {
            model.begin_speculative_suffix(3);
            model.hidden_forward_speculative_suffix(drafts);
            model.rollback_speculative(kept);
        } else {
            model.hidden_forward(drafts,mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,nullptr,1);
            model.rollback_speculative(kept-1);
        }
        MFQ_RUNTIME_CHECK(model.cache_pos==prefix+kept,"retained prefix position differs");
        result.push_back(model.hidden_forward(next).clone());
        model.restore_text_session_state(saved);
        if(kept)model.hidden_forward(drafts.narrow(1,0,kept).contiguous());
        result.push_back(model.hidden_forward(next).clone());
    }
    return result;
}

inline std::vector<Tensor> dsv41_prompt(const std::vector<Tensor>& input,const nlohmann::json& p) {
    CudaExecutionContext execution;
    auto model=load_causal_lm<DeepseekV41CausalLm>(execution,p.at("model"),"",128);
    model.reset(1);
    auto tokens=p.at("ids").get<std::vector<int64_t>>();
    MfqMultimodalInput media;
    media.processor=MfqMultimodalProcessor::deepseek_v41;media.processor_name="deepseek_v41";
    media.image_bounds=p.at("bounds").get<std::vector<int64_t>>();
    media.vision_grid=p.at("grid").get<std::vector<int32_t>>();
    media.vision_grid_shape={static_cast<int64_t>(media.image_bounds.size()/4),4};
    media.pixel_shape=input.at(0).sizes().vec();
    auto host=input.at(0).to(kFloat32).contiguous().cpu();
    media.pixel_values.assign(host.data_ptr<float>(),host.data_ptr<float>()+host.numel());
    auto vision=deepseek_v41_runtime::Vision::load(model.source,model.config);
    CudaPreparedPrompt prepared;
    for(int reuse=0;reuse<2;++reuse) {
        auto steps=vision.prepare(model,tokens,media);
        while(auto step=steps.next())if(step.value)prepared=std::move(*step.value);
    }
    auto ids=mfq_tensor_backend::tensor(tokens,TensorOptions().device(kCUDA).dtype(kInt64)).reshape({1,-1});
    return {prepared.embeddings,prepared.image_mask,model.embed_forward(ids)};
}
} // namespace mfq::cuda::tests
