#include "session_state.h"
#include "cpp_runtime/models/deepseek_v41/engram.h"

#include <algorithm>
#include <array>

namespace {
namespace tb=mfq_tensor_backend;
constexpr std::array<uint8_t,8> magic{'M','F','Q','C','U','D','3',0};
struct Header { uint32_t start,count,layers,flags; };
Header header(CudaPagedReader& reader) {
    std::array<uint8_t,8> value{};reader.raw(value.data(),value.size(),"magic");
    if(value!=magic || reader.scalar<uint32_t>("version")!=1)
        throw CudaSessionStateError("unsupported DeepSeek-V4.1 prefix payload");
    Header result{reader.scalar<uint32_t>("start"),reader.scalar<uint32_t>("count"),
        reader.scalar<uint32_t>("layers"),reader.scalar<uint32_t>("flags")};
    if(!result.count || !result.layers || result.layers>4096 || result.flags>3 ||
            ((result.flags&2) && !(result.flags&1)) ||
            uint64_t(result.start)+result.count>INT64_MAX)
        throw CudaSessionStateError("invalid DeepSeek-V4.1 prefix header");
    return result;
}
void optional(CudaPagedWriter& writer,const tb::Tensor& value) {
    writer.scalar<uint32_t>(value.defined()?1:0);
    if(value.defined())writer.tensor(value);
}
tb::Tensor optional(CudaPagedReader& reader,int expected_device) {
    const auto present=reader.scalar<uint32_t>("tensor presence");
    if(present>1)throw CudaSessionStateError("invalid DeepSeek-V4.1 tensor presence");
    if(!present)return {};
    auto [value,device]=reader.tensor();
    if(device!=expected_device)throw CudaSessionStateError("inconsistent DeepSeek-V4.1 prefix device");
    return value;
}
bool shape(const tb::Tensor& value,int64_t rows,int64_t width,tb::ScalarType dtype) {
    return value.defined() && value.dim()==3 && value.size(0)==1 &&
        value.size(1)==rows && value.size(2)==width && value.scalar_type()==dtype;
}
struct Layout {
    int64_t ratio,kv_width,index_width;
    int device;
    bool operator==(const Layout&) const=default;
};
}

CudaPagedPayload encode_cuda_deepseek_v41_paged_block(
        const TextSessionState& state,size_t block_size,size_t index) {
    const auto* layers=std::get_if<std::vector<DeepseekV41BlockSessionState>>(&state.payload);
    if(!block_size || !layers || layers->empty() || state.cache_pos<=0 ||
            state.tokens.size()!=size_t(state.cache_pos) || state.cache_pos>UINT32_MAX ||
            index>(state.tokens.size()-1)/block_size || !state.deepseek_v41_engram ||
            state.deepseek_v41_engram->batch!=1 ||
            state.deepseek_v41_engram->position!=state.cache_pos || state.decode_position_delta!=0)
        throw std::runtime_error("invalid DeepSeek-V4.1 paged session");
    const auto start=index*block_size,count=std::min(block_size,state.tokens.size()-start),end=start+count;
    const bool exact=end==state.tokens.size();
    CudaPagedWriter writer;writer.raw(magic.data(),magic.size());writer.scalar<uint32_t>(1);
    for(const auto value:{start,count,layers->size(),size_t(exact?(state.mtp?3:1):0)})
        writer.scalar<uint32_t>(uint32_t(value));
    for(const auto& layer:*layers) {
        const int64_t ratio=layer.compressed_kv.defined()?(layer.partial_kv.defined()?layer.partial_kv.size(1):1):0;
        const int64_t kv_width=ratio?layer.compressed_kv.size(2):0;
        const int64_t index_width=ratio && layer.index_k.defined()?layer.index_k.size(2):0;
        if(!layer.local_kv.is_cuda() || (ratio && (!index_width ||
                layer.compressed_length!=state.cache_pos/ratio ||
                layer.partial_length!=state.cache_pos%ratio ||
                !shape(layer.compressed_kv,layer.compressed_length,kv_width,tb::kFloat16) ||
                !shape(layer.index_k,layer.compressed_length,index_width,tb::kFloat16))))
            throw std::runtime_error("inconsistent DeepSeek-V4.1 compression checkpoint");
        writer.scalar<int64_t>(ratio);writer.scalar<int64_t>(kv_width);writer.scalar<int64_t>(index_width);
        writer.scalar<int32_t>(layer.local_kv.get_device());
        const auto range=[&](const tb::Tensor& value) {
            return ratio && end/ratio>start/ratio
                ?value.narrow(1,int64_t(start)/ratio,int64_t(end/ratio-start/ratio)):tb::Tensor{};
        };
        optional(writer,range(layer.compressed_kv));optional(writer,range(layer.index_k));
        optional(writer,exact?layer.local_kv:tb::Tensor{});
        optional(writer,exact?layer.partial_kv:tb::Tensor{});
        optional(writer,exact?layer.partial_score:tb::Tensor{});
    }
    if(exact) {
        const auto& context=state.deepseek_v41_engram->context;
        writer.scalar<uint32_t>(uint32_t(context.size()));
        for(auto token:context)writer.scalar<int64_t>(token);
        if(state.mtp) {
            const auto& mtp=*state.mtp;
            if(mtp.cache_pos!=state.cache_pos || mtp.deepseek_v41_rings.empty() ||
                    !mtp.blocks.empty() || !mtp.qwen4_blocks.empty() || !mtp.last_target_hidden.defined())
                throw std::runtime_error("incompatible DeepSeek-V4.1 DSpark checkpoint");
            writer.tensor(mtp.last_target_hidden);
            writer.scalar<uint32_t>(uint32_t(mtp.deepseek_v41_rings.size()));
            for(const auto& ring:mtp.deepseek_v41_rings)writer.tensor(ring);
        }
    }
    return std::move(writer).finish();
}

bool cuda_deepseek_v41_paged_exact(const CudaPagedPayload& payload) {
    if(!payload)throw CudaSessionStateError("null DeepSeek-V4.1 prefix payload");
    CudaPagedReader reader(*payload);return (header(reader).flags&1)!=0;
}
bool cuda_paged_checkpoint_needs_update(const CudaPagedPayload& payload,const TextSessionState& state) {
    if(state.kind()==TextSessionStateKind::Qwen4)return cuda_qwen4_paged_checkpoint_needs_update(payload,state);
    if(state.kind()!=TextSessionStateKind::DeepseekV41)return false;
    if(!payload)return true;
    CudaPagedReader reader(*payload);const auto flags=header(reader).flags;
    return !(flags&1) || (state.mtp && !(flags&2));
}

TextSessionState decode_cuda_deepseek_v41_paged_session(
        const std::vector<CudaPagedPayload>& payloads,const std::vector<int64_t>& tokens,size_t block_size) {
    if(!block_size || tokens.empty() || tokens.size()>UINT32_MAX ||
            payloads.size()!=(tokens.size()-1)/block_size+1)
        throw CudaSessionStateError("invalid DeepSeek-V4.1 prefix chain length");
    TextSessionState state;state.tokens=tokens;state.cache_pos=int64_t(tokens.size());
    std::vector<Layout> layouts;
    std::vector<DeepseekV41BlockSessionState> layers;
    std::vector<std::array<std::vector<tb::Tensor>,2>> fragments;
    auto hash=std::make_shared<mfq::models::deepseek_v41::EngramHashSnapshot>();
    hash->batch=1;hash->position=state.cache_pos;
    std::vector<std::pair<tb::Tensor,int>> predictor;
    for(size_t i=0;i<payloads.size();++i) {
        if(!payloads[i])throw CudaSessionStateError("null DeepSeek-V4.1 prefix block");
        CudaPagedReader reader(*payloads[i]);const auto h=header(reader);
        const auto end=uint64_t(h.start)+h.count;
        const bool exact=(h.flags&1)!=0;
        if(h.start!=i*block_size || h.count!=std::min(block_size,tokens.size()-h.start) ||
                (i && h.layers!=layers.size()) || (i+1==payloads.size() && !exact))
            throw CudaSessionStateError("incompatible DeepSeek-V4.1 prefix chain");
        if(!i){layers.resize(h.layers);fragments.resize(h.layers);}
        for(size_t j=0;j<layers.size();++j) {
            Layout layout{reader.scalar<int64_t>("ratio"),reader.scalar<int64_t>("KV width"),
                reader.scalar<int64_t>("Indexer width"),reader.scalar<int32_t>("device")};
            if(layout.device<0 || layout.ratio<0 || layout.ratio>65536 ||
                    (layout.ratio?layout.kv_width<=0 || layout.index_width<=0:
                     layout.kv_width!=0 || layout.index_width!=0) || (i && layouts[j]!=layout))
                throw CudaSessionStateError("inconsistent DeepSeek-V4.1 prefix layout");
            if(!i)layouts.push_back(layout);
            const auto rows=layout.ratio?int64_t(end/layout.ratio-h.start/layout.ratio):0;
            for(size_t field=0;field<2;++field) {
                auto value=optional(reader,layout.device);
                if(value.defined()!=(rows>0) || (rows &&
                        !shape(value,rows,field?layout.index_width:layout.kv_width,tb::kFloat16)))
                    throw CudaSessionStateError("invalid DeepSeek-V4.1 compressed prefix rows");
                if(rows)fragments[j][field].push_back(std::move(value));
            }
            auto local=optional(reader,layout.device),partial=optional(reader,layout.device),score=optional(reader,layout.device);
            if(local.defined()!=exact || (exact && (local.dim()!=3 || local.size(0)!=1 ||
                    local.size(1)<=0 || local.size(2)<=0 || local.scalar_type()!=tb::kFloat16)) ||
                    partial.defined()!=(exact && layout.ratio>1) || score.defined()!=partial.defined() ||
                    (partial.defined() && (!shape(partial,layout.ratio,layout.kv_width,tb::kFloat32) ||
                                          !shape(score,layout.ratio,layout.kv_width,tb::kFloat32))))
                throw CudaSessionStateError("invalid DeepSeek-V4.1 exact checkpoint tensors");
            if(i+1==payloads.size()) {
                layers[j].local_kv=std::move(local);layers[j].partial_kv=std::move(partial);layers[j].partial_score=std::move(score);
            }
        }
        if(exact) {
            const auto size=reader.scalar<uint32_t>("Engram history size");
            if(size>65536)throw CudaSessionStateError("invalid DeepSeek-V4.1 Engram history size");
            hash->context.clear();
            for(uint32_t j=0;j<size;++j)hash->context.push_back(reader.scalar<int64_t>("Engram token"));
            predictor.clear();
            if(h.flags&2) {
                predictor.push_back(reader.tensor());
                const auto count=reader.scalar<uint32_t>("DSpark stages");
                if(!count || count>64)throw CudaSessionStateError("invalid DeepSeek-V4.1 DSpark stage count");
                for(uint32_t j=0;j<count;++j)predictor.push_back(reader.tensor());
            }
        }
        reader.expect_end();
    }
    // Validate the complete chain before allocating device state.
    for(size_t i=0;i<predictor.size();++i) {
        const auto& value=predictor[i].first;
        if(value.dim()!=3 || value.size(0)!=1 || (!i && value.size(1)!=1) || value.scalar_type()!=tb::kFloat16)
            throw CudaSessionStateError("invalid DeepSeek-V4.1 DSpark tensor");
    }
    for(size_t j=0;j<layers.size();++j) {
        auto& layer=layers[j];const auto& layout=layouts[j];
        const auto device=tb::Device{tb::DeviceType::cuda,layout.device};
        layer.local_kv=layer.local_kv.to(device);
        if(layout.ratio) {
            layer.compressed_length=state.cache_pos/layout.ratio;layer.partial_length=state.cache_pos%layout.ratio;
            std::array<tb::Tensor*,2> destination{&layer.compressed_kv,&layer.index_k};
            for(size_t field=0;field<2;++field) {
                const auto width=field?layout.index_width:layout.kv_width;
                auto& parts=fragments[j][field];
                auto value=parts.empty()?tb::empty({1,0,width},tb::TensorOptions().dtype(tb::kFloat16).device(tb::kCPU)):
                    parts.size()==1?parts.front():tb::cat(parts,1);
                if(!shape(value,layer.compressed_length,width,tb::kFloat16))
                    throw CudaSessionStateError("incomplete DeepSeek-V4.1 compressed prefix");
                *destination[field]=value.to(device);
            }
            if(layout.ratio>1) {
                layer.partial_kv=layer.partial_kv.to(device);layer.partial_score=layer.partial_score.to(device);
            }
        }
        for(const auto* tensor:{&layer.local_kv,&layer.compressed_kv,&layer.index_k,&layer.partial_kv,&layer.partial_score})
            state.bytes+=session_tensor_bytes(*tensor);
    }
    if(!predictor.empty()) {
        auto& mtp=state.mtp.emplace();mtp.cache_pos=state.cache_pos;
        for(size_t i=0;i<predictor.size();++i) {
            auto& [value,device]=predictor[i];
            auto target=value.to(tb::Device{tb::DeviceType::cuda,device});mtp.bytes+=session_tensor_bytes(target);
            if(!i)mtp.last_target_hidden=std::move(target);else mtp.deepseek_v41_rings.push_back(std::move(target));
        }
        state.bytes+=mtp.bytes;
    }
    state.bytes+=hash->context.size()*sizeof(int64_t);state.deepseek_v41_engram=std::move(hash);
    state.payload=std::move(layers);return state;
}
