#include "ops.h"
#include "storage/session_state.h"

namespace mfq::cuda {
namespace {
using NativeBlock=deepseek_v41_runtime::Block;
using Tensor=mfq_tensor_backend::Tensor;

bool session_tensor(const Tensor& value,int64_t tokens,int64_t width,
                    mfq_tensor_backend::ScalarType dtype) {
    return value.defined() && value.is_cuda() && value.dim()==3 && value.size(0)==1 &&
        value.size(1)==tokens && value.size(2)==width && value.scalar_type()==dtype;
}

void validate_layer(const NativeBlock& block,const DeepseekV41BlockSessionState& saved,
                    int64_t position) {
    const auto& config=block.config;
    const bool local=session_tensor(saved.local_kv,config.sliding_window,config.head_dim,
                                     mfq_tensor_backend::kFloat16);
    bool global=saved.compressed_length==0 && saved.partial_length==0 &&
        !saved.compressed_kv.defined() && !saved.index_k.defined() &&
        !saved.partial_kv.defined() && !saved.partial_score.defined();
    if(block.kv_source()) {
        global=block.ratio>0 && saved.compressed_length==position/block.ratio &&
            saved.partial_length==(block.ratio>1?position%block.ratio:0) &&
            session_tensor(saved.compressed_kv,saved.compressed_length,config.head_dim,
                           mfq_tensor_backend::kFloat16) &&
            session_tensor(saved.index_k,saved.compressed_length,config.index_head_dim,
                           mfq_tensor_backend::kFloat16);
        if(block.ratio>1)
            global=global && session_tensor(saved.partial_kv,block.ratio,config.head_dim,
                                            mfq_tensor_backend::kFloat32) &&
                session_tensor(saved.partial_score,block.ratio,config.head_dim,
                               mfq_tensor_backend::kFloat32);
        else global=global && !saved.partial_kv.defined() && !saved.partial_score.defined();
    }
    if(!local || !global)throw CudaSessionStateError("DeepSeek-V4.1 session layer geometry is incompatible");
}
}

TextSessionStateKind CudaSessionCodec<DeepseekV41Model>::kind(const Model& model) {
    return model.shared && model.shared->engram_hash && !model.blocks.empty() &&
        std::all_of(model.blocks.begin(),model.blocks.end(),[](const auto& block) {
            return dynamic_cast<const NativeBlock*>(block.get())!=nullptr;
        })?TextSessionStateKind::DeepseekV41:TextSessionStateKind::Unsupported;
}

bool CudaSessionCodec<DeepseekV41Model>::supports_paged(const Model& model) {
    return kind(model)==TextSessionStateKind::DeepseekV41;
}

TextSessionState CudaSessionCodec<DeepseekV41Model>::capture(const Model& model,
        const std::vector<int64_t>& tokens) {
    if(kind(model)!=TextSessionStateKind::DeepseekV41 || model.cache_pos<=0 ||
            std::size_t(model.cache_pos)!=tokens.size())
        throw std::runtime_error("DeepSeek-V4.1 text session token count disagrees with cache");
    auto hash=model.shared->engram_hash->snapshot();
    if(hash.batch!=1 || hash.position!=model.cache_pos)
        throw std::runtime_error("DeepSeek-V4.1 session Engram position disagrees with cache");
    TextSessionState state;
    state.tokens=tokens;state.cache_pos=model.cache_pos;
    state.deepseek_v41_engram=std::make_shared<const mfq::models::deepseek_v41::EngramHashSnapshot>(std::move(hash));
    state.bytes=state.deepseek_v41_engram->context.size()*sizeof(int64_t);
    state.payload=std::vector<DeepseekV41BlockSessionState>{};
    auto& layers=std::get<std::vector<DeepseekV41BlockSessionState>>(state.payload);
    layers.reserve(model.blocks.size());
    for(const auto& item:model.blocks) {
        const auto& block=dynamic_cast<const NativeBlock&>(*item);
        MfqCudaGuard guard(block.cuda_device);
        const auto& source=block.state;
        if(block.speculative_pending || source.position!=model.cache_pos)
            throw std::runtime_error("DeepSeek-V4.1 session attention state is not committed");
        DeepseekV41BlockSessionState saved;
        const auto copy=[&](const Tensor& value) {
            if(!value.defined())return Tensor{};
            auto result=value.clone();state.bytes+=session_tensor_bytes(result);return result;
        };
        saved.local_kv=copy(source.local_kv);
        saved.compressed_length=source.compressed_length;saved.partial_length=source.partial_length;
        if(source.compressed_kv.defined())
            saved.compressed_kv=copy(source.compressed_kv.narrow(1,0,source.compressed_length));
        if(source.index_k.defined())saved.index_k=copy(source.index_k.narrow(1,0,source.compressed_length));
        saved.partial_kv=copy(source.partial_kv);saved.partial_score=copy(source.partial_score);
        validate_layer(block,saved,state.cache_pos);
        layers.push_back(std::move(saved));
    }
    return state;
}

void CudaSessionCodec<DeepseekV41Model>::restore(Model& model,const TextSessionState& state) {
    const auto* layers=std::get_if<std::vector<DeepseekV41BlockSessionState>>(&state.payload);
    if(kind(model)!=TextSessionStateKind::DeepseekV41 || !layers || layers->size()!=model.blocks.size() ||
            state.cache_pos<=0 || state.cache_pos>model.max_position_embeddings() ||
            std::size_t(state.cache_pos)!=state.tokens.size() || state.decode_position_delta!=0 ||
            !state.deepseek_v41_engram || state.deepseek_v41_engram->batch!=1 ||
            state.deepseek_v41_engram->position!=state.cache_pos)
        throw CudaSessionStateError("DeepSeek-V4.1 text session state is incompatible");
    // Validate every layer and rolling n-gram before changing the active cache.
    auto hash=*model.shared->engram_hash;
    hash.restore(*state.deepseek_v41_engram);
    for(std::size_t i=0;i<layers->size();++i)
        validate_layer(dynamic_cast<const NativeBlock&>(*model.blocks[i]),(*layers)[i],state.cache_pos);
    model.reset(1);
    for(std::size_t i=0;i<layers->size();++i) {
        auto& block=dynamic_cast<NativeBlock&>(*model.blocks[i]);
        MfqCudaGuard guard(block.cuda_device);
        const auto& saved=(*layers)[i];auto& target=block.state;
        restore_session_tensor(target.local_kv,saved.local_kv);
        if(block.kv_source()) {
            restore_session_prefix_tensor(target.compressed_kv,saved.compressed_kv,1,
                                          std::max<int64_t>(1,block.max_context/block.ratio));
            restore_session_prefix_tensor(target.index_k,saved.index_k,1,
                                          std::max<int64_t>(1,block.max_context/block.ratio));
            if(block.ratio>1) {
                restore_session_tensor(target.partial_kv,saved.partial_kv);
                restore_session_tensor(target.partial_score,saved.partial_score);
            }
        }
        target.position=state.cache_pos;target.compressed_length=saved.compressed_length;
        target.partial_length=saved.partial_length;
    }
    *model.shared->engram_hash=std::move(hash);
    model.shared->set_image_mask({});
    model.cache_pos=state.cache_pos;
}
} // namespace mfq::cuda
