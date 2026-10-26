#include "session_state.h"
#include <algorithm>
#include <array>

namespace {
constexpr std::array<uint8_t,8> qwen4_magic{'M','F','Q','C','U','D','2',0};
struct Qwen4PagedHeader { uint32_t start,count,layers;bool exact; };
Qwen4PagedHeader read_qwen4_header(CudaPagedReader& reader) {
    std::array<uint8_t,8> magic{};reader.raw(magic.data(),magic.size(),"magic");
    if(magic!=qwen4_magic || reader.scalar<uint32_t>("version")!=1)
        throw CudaSessionStateError("unsupported Qwen4 prefix cache payload");
    Qwen4PagedHeader header{reader.scalar<uint32_t>("start"),reader.scalar<uint32_t>("count"),
        reader.scalar<uint32_t>("layers"),false};
    const auto exact=reader.scalar<uint32_t>("exact boundary");
    if(exact>1)throw CudaSessionStateError("invalid Qwen4 checkpoint presence");
    header.exact=exact!=0;
    if(!header.count || !header.layers || header.layers>4096)throw CudaSessionStateError("invalid Qwen4 prefix geometry");
    return header;
}
void write_optional(CudaPagedWriter& writer,const mfq_tensor_backend::Tensor& tensor) {
    writer.scalar<uint32_t>(tensor.defined()?1:0);if(tensor.defined())writer.tensor(tensor);
}
mfq_tensor_backend::Tensor read_optional(CudaPagedReader& reader) {
    const auto present=reader.scalar<uint32_t>("tensor presence");
    if(present>1)throw CudaSessionStateError("invalid Qwen4 tensor presence");
    if(!present)return {};
    auto [tensor,device]=reader.tensor();
    return tensor.to(mfq_tensor_backend::Device{mfq_tensor_backend::DeviceType::cuda,device});
}
void write_recurrent(CudaPagedWriter& writer,const Qwen4BlockSessionState& layer) {
    write_optional(writer,layer.convolution);write_optional(writer,layer.recurrent);write_optional(writer,layer.ple_convolution);
    writer.scalar<int64_t>(layer.ple_batch);writer.scalar<uint32_t>(uint32_t(layer.ple_tokens.size()));
    for(auto token:layer.ple_tokens)writer.scalar<int64_t>(token);
}
void read_recurrent(CudaPagedReader& reader,Qwen4BlockSessionState& layer) {
    layer.convolution=read_optional(reader);layer.recurrent=read_optional(reader);layer.ple_convolution=read_optional(reader);
    layer.ple_batch=reader.scalar<int64_t>("PLE batch");
    const auto count=reader.scalar<uint32_t>("PLE history length");
    if(count>65536)throw CudaSessionStateError("invalid Qwen4 PLE history");
    for(uint32_t i=0;i<count;++i)layer.ple_tokens.push_back(reader.scalar<int64_t>("PLE token"));
}
void write_qwen4_mtp(CudaPagedWriter& writer,const MtpSessionState& state) {
    if(state.qwen4_blocks.empty() || state.qwen4_blocks.size()!=state.qwen4_positions.size() ||
        state.qwen4_blocks.size()!=state.qwen4_lengths.size() || !state.blocks.empty() || !state.deepseek_v41_rings.empty())
        throw std::runtime_error("incompatible Qwen4 paged predictor state");
    writer.scalar<int64_t>(state.cache_pos);write_optional(writer,state.last_target_hidden);
    writer.scalar<uint32_t>(uint32_t(state.qwen4_blocks.size()));
    for(std::size_t i=0;i<state.qwen4_blocks.size();++i) {
        const auto& layer=state.qwen4_blocks[i];
        write_optional(writer,layer.keys);write_optional(writer,layer.values);write_optional(writer,layer.index);
        write_recurrent(writer,layer);write_optional(writer,state.qwen4_positions[i]);
        writer.scalar<int64_t>(state.qwen4_lengths[i]);
    }
}
std::size_t qwen4_layer_bytes(const Qwen4BlockSessionState& layer) {
    std::size_t bytes=layer.ple_tokens.size()*sizeof(int64_t);
    for(const auto* tensor:{&layer.keys,&layer.values,&layer.index,&layer.convolution,&layer.recurrent,&layer.ple_convolution})
        bytes+=session_tensor_bytes(*tensor);
    return bytes;
}
MtpSessionState read_qwen4_mtp(CudaPagedReader& reader) {
    MtpSessionState state;state.cache_pos=reader.scalar<int64_t>("MTP position");state.last_target_hidden=read_optional(reader);
    const auto count=reader.scalar<uint32_t>("MTP layers");
    if(!count || count>64)throw CudaSessionStateError("invalid Qwen4 MTP layers");
    state.bytes=session_tensor_bytes(state.last_target_hidden);
    for(uint32_t i=0;i<count;++i) {
        Qwen4BlockSessionState layer;layer.keys=read_optional(reader);layer.values=read_optional(reader);layer.index=read_optional(reader);
        read_recurrent(reader,layer);state.bytes+=qwen4_layer_bytes(layer);state.qwen4_blocks.push_back(std::move(layer));
        auto positions=read_optional(reader);state.bytes+=session_tensor_bytes(positions);state.qwen4_positions.push_back(std::move(positions));
        state.qwen4_lengths.push_back(reader.scalar<int64_t>("MTP layer position"));
    }
    return state;
}
}

CudaPagedPayload encode_cuda_qwen4_paged_block(const TextSessionState& state,size_t block_size,size_t index) {
    const auto* layers=std::get_if<std::vector<Qwen4BlockSessionState>>(&state.payload);
    if(!block_size || !layers || layers->empty() || state.cache_pos<=0 || state.tokens.size()!=std::size_t(state.cache_pos) ||
        index>state.tokens.size()/block_size || index*block_size>=state.tokens.size() || state.cache_pos>UINT32_MAX)
        throw std::runtime_error("invalid Qwen4 prefix block boundary");
    if(state.decode_position_delta!=0)throw std::runtime_error("Qwen4 paged text sessions require zero position offset");
    const auto start=index*block_size,count=std::min(block_size,state.tokens.size()-start);
    const bool exact=start+count==state.tokens.size();
    CudaPagedWriter writer;writer.raw(qwen4_magic.data(),qwen4_magic.size());writer.scalar<uint32_t>(1);
    writer.scalar<uint32_t>(uint32_t(start));writer.scalar<uint32_t>(uint32_t(count));
    writer.scalar<uint32_t>(uint32_t(layers->size()));writer.scalar<uint32_t>(exact?1:0);
    for(const auto& layer:*layers) {
        for(const auto* tensor:{&layer.keys,&layer.values,&layer.index})
            write_optional(writer,tensor->defined()?tensor->narrow(1,int64_t(start),int64_t(count)):mfq_tensor_backend::Tensor{});
        if(exact)write_recurrent(writer,layer);
    }
    if(exact) {
        write_optional(writer,state.qwen4_positions);writer.scalar<uint32_t>(state.mtp?1:0);
        if(state.mtp)write_qwen4_mtp(writer,*state.mtp);
    }
    return std::move(writer).finish();
}

size_t cuda_paged_payload_tokens(const CudaPagedPayload& payload) {
    if(!payload)throw CudaSessionStateError("null CUDA prefix block");
    CudaPagedReader reader(*payload);std::array<uint8_t,8> magic{};reader.raw(magic.data(),magic.size(),"magic");
    constexpr std::array<uint8_t,8> full_magic{'M','F','Q','C','U','D','1',0};
    constexpr std::array<uint8_t,8> deepseek_magic{'M','F','Q','C','U','D','3',0};
    if((magic!=qwen4_magic && magic!=full_magic && magic!=deepseek_magic) || reader.scalar<uint32_t>("version")!=1)
        throw CudaSessionStateError("invalid CUDA prefix block header");
    (void)reader.scalar<uint32_t>("start");return reader.scalar<uint32_t>("count");
}
size_t cuda_paged_decodable_blocks(const std::vector<CudaPagedPayload>& payloads) {
    for(size_t count=payloads.size();count>0;--count) {
        const auto& payload=payloads[count-1];
        (void)cuda_paged_payload_tokens(payload);
        if((*payload)[6]=='1')return count;
        if((*payload)[6]=='3') {
            if(cuda_deepseek_v41_paged_exact(payload))return count;
            continue;
        }
        CudaPagedReader reader(*payload);if(read_qwen4_header(reader).exact)return count;
    }
    return 0;
}

TextSessionState decode_cuda_qwen4_paged_session(const std::vector<CudaPagedPayload>& payloads,
    const std::vector<int64_t>& tokens,size_t block_size) {
    namespace tb=mfq_tensor_backend;
    TextSessionState state;state.tokens=tokens;state.cache_pos=int64_t(tokens.size());
    std::vector<Qwen4BlockSessionState> layers;
    std::vector<std::array<std::vector<tb::Tensor>,3>> fragments;
    size_t position=0;
    for(size_t i=0;i<payloads.size();++i) {
        if(!payloads[i])throw CudaSessionStateError("null Qwen4 prefix payload");
        CudaPagedReader reader(*payloads[i]);const auto header=read_qwen4_header(reader);
        if(header.start!=position || header.count>block_size || (i+1<payloads.size() && header.count!=block_size) ||
            (i && header.layers!=layers.size()))throw CudaSessionStateError("incompatible Qwen4 prefix chain");
        if(!i){layers.resize(header.layers);fragments.resize(header.layers);}
        position+=header.count;
        for(size_t layer=0;layer<layers.size();++layer) {
            for(size_t field=0;field<3;++field) {
                auto tensor=read_optional(reader);
                if(tensor.defined()) {
                    if(tensor.dim()!=3 || tensor.size(0)!=1 || tensor.size(1)!=header.count)
                        throw CudaSessionStateError("invalid Qwen4 QSA prefix geometry");
                    fragments[layer][field].push_back(std::move(tensor));
                }
            }
            if(header.exact) {
                layers[layer].convolution={};layers[layer].recurrent={};layers[layer].ple_convolution={};layers[layer].ple_tokens.clear();
                read_recurrent(reader,layers[layer]);
            }
        }
        if(header.exact) {
            state.qwen4_positions=read_optional(reader);
            state.mtp.reset();if(reader.scalar<uint32_t>("MTP presence"))state.mtp=read_qwen4_mtp(reader);
        }
        if(i+1==payloads.size() && !header.exact)throw CudaSessionStateError("Qwen4 prefix has no exact recurrent checkpoint");
        reader.expect_end();
    }
    if(payloads.empty() || position!=tokens.size() || !state.qwen4_positions.defined() || state.qwen4_positions.size(-1)!=int64_t(position))
        throw CudaSessionStateError("Qwen4 prefix boundary mismatch");
    for(size_t i=0;i<layers.size();++i) {
        auto& layer=layers[i];std::array<tb::Tensor*,3> fields{&layer.keys,&layer.values,&layer.index};
        for(size_t j=0;j<3;++j)if(!fragments[i][j].empty()) {
            if(fragments[i][j].size()!=payloads.size())throw CudaSessionStateError("incomplete Qwen4 QSA prefix chain");
            *fields[j]=fragments[i][j].size()==1?fragments[i][j].front():tb::cat(fragments[i][j],1);
        }
        state.bytes+=qwen4_layer_bytes(layer);
    }
    state.bytes+=session_tensor_bytes(state.qwen4_positions)+(state.mtp?state.mtp->bytes:0);
    state.payload=std::move(layers);return state;
}


namespace {
// Inspect an existing tail without copying its KV tensors to host or device.
void skip_qwen4_optional(CudaPagedReader& reader) {
    const auto present=reader.scalar<uint32_t>("tensor presence");
    if(present>1)throw CudaSessionStateError("invalid Qwen4 tensor presence");
    if(!present)return;
    namespace tb=mfq_tensor_backend;
    const auto type=static_cast<tb::ScalarType>(reader.scalar<int32_t>("dtype"));
    const auto device=reader.scalar<int32_t>("device");
    const auto rank=reader.scalar<uint32_t>("rank");(void)reader.scalar<uint32_t>("tensor flags");
    const uint64_t width=type==tb::kInt64?8:(type==tb::kInt32 || type==tb::kFloat32)?4:
        (type==tb::kFloat16 || type==tb::kBFloat16)?2:0;
    if(!width || device<0 || !rank || rank>8)throw CudaSessionStateError("invalid Qwen4 tensor metadata");
    uint64_t elements=1;
    for(uint32_t i=0;i<rank;++i) {
        const auto size=reader.scalar<int64_t>("shape");
        if(size<=0 || elements>UINT64_MAX/uint64_t(size))throw CudaSessionStateError("invalid Qwen4 tensor shape");
        elements*=uint64_t(size);
    }
    const auto bytes=reader.scalar<uint64_t>("tensor size");
    if(elements>UINT64_MAX/width || bytes!=elements*width || bytes>SIZE_MAX)
        throw CudaSessionStateError("invalid Qwen4 tensor size");
    reader.skip(size_t(bytes),"Qwen4 tensor contents");
}
}
bool cuda_qwen4_paged_checkpoint_needs_update(const CudaPagedPayload& payload,const TextSessionState& state) {
    if(!payload)return true;
    CudaPagedReader reader(*payload);const auto header=read_qwen4_header(reader);
    if(!header.exact)return true;
    if(!state.mtp)return false;
    for(uint32_t layer=0;layer<header.layers;++layer) {
        for(int field=0;field<6;++field)skip_qwen4_optional(reader);
        (void)reader.scalar<int64_t>("PLE batch");
        const auto count=reader.scalar<uint32_t>("PLE history size");
        if(count>65536)throw CudaSessionStateError("invalid Qwen4 PLE history size");
        reader.skip(size_t(count)*sizeof(int64_t),"PLE history");
    }
    skip_qwen4_optional(reader);
    const auto present=reader.scalar<uint32_t>("MTP presence");
    if(present>1)throw CudaSessionStateError("invalid Qwen4 predictor presence");
    return !present;
}

