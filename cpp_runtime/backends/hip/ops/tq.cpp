#include "tq.h"
#include "tq_moe.h"
#include "quant_linear.h"
#include "moe_types.h"
#include "moe.h"
#include "qwen4_exp.h"
extern "C" int mfq_tq_gr(ProjectionView,const void*,const void*,void*,int,int,int,int,hipStream_t);
extern "C" uint64_t mfq_tq_prefill_f32_workspace(ProjectionView,int);
#include "mfq_hip_moe_ops.h"
#include "tq_input_gate.h"
#include "tq_residual.h"
#include "mfq/tq_rows.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <future>
#include <climits>

namespace tb=mfq_tensor_backend;
namespace {
std::vector<std::int64_t> logical_shape(const mfq::ModelSource&,const std::string&);
int dtype(tb::ScalarType type) {
    if(type==tb::kFloat16)return 0;
    if(type==tb::kBFloat16)return 1;
    if(type==tb::kFloat32)return 2;
    throw std::invalid_argument("TQ HIP activation dtype must be FP16, BF16 or FP32");
}
void status(int code) {MFQ_CUDA_CHECK(static_cast<hipError_t>(code));}
void trace(int role,const char* name,const tb::Tensor& x) {
    static const bool enabled=[] {
        const char* option=std::getenv("MFQ_HIP_TQ_TRACE");
        return option && std::strcmp(option,"1")==0;
    }();
    if(!enabled || x.numel()!=x.size(-1))return;
    static std::atomic<unsigned> seen{0};
    const int type=dtype(x.scalar_type());const unsigned bit=1u<<(role*3+type);
    if(!(seen.fetch_or(bit)&bit))std::fprintf(stderr,"tq_hip_path role=%s activation=%s tokens=1 input_columns=%lld\n",
        name,type==0?"FP16":type==1?"BF16":"FP32",static_cast<long long>(x.size(-1)));
}
}

bool tq_q8_enabled() {
    static const bool enabled=[] {
        const char* option=std::getenv("MFQ_HIP_TQ_Q8");
        return option && std::strcmp(option,"1")==0;
    }();
    return enabled;
}

tb::Tensor tq_gate_input(tb::Tensor input,tb::Tensor gate,int mode) {
    if(!input.is_cuda() || !gate.is_cuda() || input.device()!=gate.device() ||
       input.sizes()!=gate.sizes() || (mode!=1 && mode!=2))
        throw std::invalid_argument("TQ input gate requires matching GPU tensors and sigmoid/SiLU");
    const MfqCudaGuard guard(input.get_device());
    const bool double_math=gate.scalar_type()==input.scalar_type() && !gate.is_contiguous();
    const auto x=input.contiguous(),g=gate.contiguous();
    auto result=tb::empty(x.sizes(),x.options());
    status(mfq_tq_input_gate(x.data_ptr(),g.data_ptr(),result.data_ptr(),x.numel(),
        dtype(x.scalar_type()),dtype(g.scalar_type()),mode,double_math,mfq_current_cuda_stream()));
    return result;
}

std::vector<tb::Tensor> tq_forward_group(
        const std::vector<const TqWeight*>& weights,tb::Tensor input) {
    if(weights.empty() || weights.size()>4 || !input.is_cuda())
        throw std::invalid_argument("TQ projection group requires 1..4 GPU weights");
    const auto x=input.contiguous();
    if(x.numel()!=weights[0]->neuron_len || x.size(-1)!=weights[0]->neuron_len)
        throw std::invalid_argument("TQ fused group requires one input token");
    const MfqCudaGuard guard(x.get_device());
    trace(1,"group",x);
    std::vector<tb::Tensor> result;
    for(const auto* w:weights) {
        if(w->neuron_len!=x.size(-1) || w->image.get_device()!=x.get_device())
            throw std::invalid_argument("TQ grouped weight geometry or device differs");
        auto shape=x.sizes().vec();shape.back()=w->out;
        result.push_back(tb::empty(shape,x.options()));
    }
    // Schedule tiny gate projections early so their few blocks overlap the
    // larger projections. Output tensors retain the caller's original order.
    std::vector<int> order;for(int i=0;i<int(weights.size());++i)order.push_back(i);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return (weights[a]->out<=64)>(weights[b]->out<=64);});
    TqProjectionPlan plan{};plan.count=weights.size();
    for(int i=0;i<plan.count;++i) {
        const int j=order[i];plan.jobs[i]={weights[j]->view,result[j].data_ptr(),0};
    }
    status(mfq_tq_group(plan,x.data_ptr(),dtype(x.scalar_type()),mfq_current_cuda_stream()));
    return result;
}

tb::Tensor tq_swiglu(const TqWeight& gate,const TqWeight& up,tb::Tensor input) {
    if(!input.is_cuda() || input.numel()!=gate.neuron_len || input.size(-1)!=gate.neuron_len ||
       gate.neuron_len!=up.neuron_len || gate.out!=up.out ||
       input.get_device()!=gate.image.get_device() || input.get_device()!=up.image.get_device())
        throw std::invalid_argument("TQ fused SwiGLU geometry or device differs");
    const MfqCudaGuard guard(input.get_device());
    const auto x=input.contiguous();auto shape=x.sizes().vec();shape.back()=gate.out;
    trace(2,"swiglu",x);
    auto result=tb::empty(shape,x.options());
    status(mfq_tq_swiglu(gate.view,up.view,x.data_ptr(),result.data_ptr(),
        dtype(x.scalar_type()),mfq_current_cuda_stream()));
    return result;
}

tb::Tensor tq_forward_residual(const TqWeight& weight,tb::Tensor input,tb::Tensor residual) {
    if(!input.is_cuda() || !residual.is_cuda() || input.scalar_type()!=residual.scalar_type() ||
       input.numel()!=weight.neuron_len ||
       residual.numel()!=weight.out || input.size(-1)!=weight.neuron_len ||
       residual.size(-1)!=weight.out || input.get_device()!=weight.image.get_device() ||
       residual.get_device()!=input.get_device())
        throw std::invalid_argument("TQ fused residual requires compatible single-token tensors");
    const MfqCudaGuard guard(input.get_device());
    const auto x=input.contiguous(),skip=residual.contiguous();
    trace(3,"residual",x);
    auto shape=x.sizes().vec();shape.back()=weight.out;
    auto result=tb::empty(shape,x.options());
    status(mfq_tq_residual(weight.view,x.data_ptr(),skip.data_ptr(),result.data_ptr(),
        dtype(x.scalar_type()),mfq_current_cuda_stream()));
    return result;
}

TqWeight to_device_tq(const std::vector<std::uint8_t>& blob,std::int64_t columns,int device) {
    const mfq::TqLayout m(blob.data(),blob.size());
    if(columns<1 || columns>m.width || m.rows>INT_MAX || m.width>INT_MAX)
        throw std::invalid_argument("invalid TQ HIP logical shape");
    // Validate descriptors, complete packed coverage, and finite positive scales
    // before uploading. The retained tensor keeps its ordinary RTRELL05 layout.
    // The complete image already has contiguous descriptors. Validate them
    // directly instead of rebuilding every selector and copying all weights.
    std::uint64_t expected=0;
    for(std::size_t row=0;row<m.rows;++row) {
        const auto descriptor=mfq::TqLayout::integer(blob.data()+64+row*8,8);
        const auto bits=((descriptor>>32)&7)+1,anchor=(descriptor>>40)&0xffff;
        if((descriptor & ~0x00ffff07ffffffffULL) || anchor>0x7bff ||
           std::uint32_t(descriptor)!=expected)
            throw std::invalid_argument("invalid TQ row descriptor");
        expected+=std::uint64_t(m.width)*bits/32;
    }
    if(expected*4!=m.packed_bytes)throw std::invalid_argument("TQ row payload range differs");
    if(m.columns)for(std::size_t i=0;i<m.width;++i) {
        const auto bits=mfq::TqLayout::integer(blob.data()+m.scales+i*2,2);
        if(!bits || bits>0x7bff)throw std::invalid_argument("invalid TQ column scale");
    }
    const auto words=(m.packed+3)&~std::size_t(3),relative=words+m.packed_bytes;
    std::vector<std::uint8_t> image(relative+32);
    std::memcpy(image.data(),blob.data(),m.packed);
    std::memcpy(image.data()+words,blob.data()+m.packed,m.packed_bytes);
    constexpr std::uint16_t gains[]{0x3400,0x34d0,0x35ca,0x36f7,0x3830,0x390a,0x3a10,0x3b4b,
        0x3c63,0x3d47,0x3e59,0x3fa4,0x4098,0x4187,0x42a6,0x4400};
    std::memcpy(image.data()+relative,gains,sizeof(gains));
    const MfqCudaGuard guard(device);
    TqWeight result;
    result.image=tb::from_blob(image.data(),{static_cast<std::int64_t>(image.size())},
        tb::TensorOptions().dtype(tb::kUInt8)).to(tb::Device(tb::kCUDA,device),false,false).contiguous();
    status(int(hipStreamSynchronize(mfq_current_cuda_stream())));
    const auto* p=result.image.data_ptr<std::uint8_t>();
    int gslog=0;for(auto gs=m.gs;gs>1;gs>>=1)++gslog;
    result.view={reinterpret_cast<const std::uint32_t*>(p+words),
        reinterpret_cast<const std::uint64_t*>(p+64),p+m.scales,p+relative,nullptr,
        int(m.rows),int(m.width),int(columns),gslog,int(m.columns)};
    result.out=m.rows;result.neuron_len=columns;
    return result;
}

TqWeight load_tq_weight(const mfq::ModelSource& source,const std::string& name,int device) {
    const auto* record=source.find_tensor(name);
    if(!record || record->dtype!="TQ")throw std::invalid_argument("expected TQ tensor: "+name);
    const auto shape=logical_shape(source,name);
    if(shape.size()==3 && shape[0]>INT_MAX/shape[1])throw std::overflow_error("TQ expert row count exceeds limits");
    std::vector<std::uint8_t> blob(record->nbytes);
    source.tensor_reader(name)(0,reinterpret_cast<std::byte*>(blob.data()),blob.size());
    auto result=to_device_tq(blob,shape.back(),device);
    const auto rows=shape.size()==3?shape[0]*shape[1]:shape[0];
    if(result.out!=rows)throw std::invalid_argument("TQ output rows differ: "+name);
    return result;
}

tb::Tensor TqWeight::materialize(tb::ScalarType type) const {
    if(type!=tb::kFloat16 && type!=tb::kFloat32)
        throw std::invalid_argument("TQ exact materialization requires FP16 or FP32");
    const MfqCudaGuard guard(image.get_device());
    auto result=tb::empty({out,neuron_len},image.options().dtype(type));
    status(mfq_tq_materialize(view,result.data_ptr(),dtype(type),mfq_current_cuda_stream()));
    return result;
}

tb::Tensor TqWeight::forward(tb::Tensor input) const {
    if(input.dim()<1 || input.size(-1)!=neuron_len || !input.is_cuda() ||
       input.get_device()!=image.get_device())
        throw std::invalid_argument("TQ HIP projection input shape or device differs");
    const int type=dtype(input.scalar_type());
    const MfqCudaGuard guard(image.get_device());
    const auto x=input.contiguous();
    const auto tokens=x.numel()/neuron_len;
    trace(0,"projection",x);
    auto shape=x.sizes().vec();shape.back()=out;
    if(!tokens)return tb::empty(shape,x.options());
    // Same geometry policy as the Python HIP operator. Dense storage is
    // temporary per invocation; compressed weights remain the model storage.
    const auto area=out*neuron_len;
    const bool small=tokens<=8 || (tokens<=32 && out<=64);
    // Match CUDA's optional Q8 activation algorithm. Exact FP16 weight
    // reconstruction remains the default and is used for calibration.
    const bool q8=tq_q8_enabled();
    if(q8 && tokens<=4 && out>64 && type!=1 && (view.columns || view.gslog>=4)) {
        auto result=tb::empty(shape,x.options());
        const auto bytes=tokens*((neuron_len+511)/512*512/16)*24;
        auto scratch=tb::empty({bytes},x.options().dtype(tb::kUInt8));
        status(mfq_tq_q8_tokens(view,x.data_ptr(),result.data_ptr(),int(tokens),type,
            scratch.data_ptr(),bytes,mfq_current_cuda_stream()));
        return result;
    }
    const bool staged=!small && neuron_len>64 && area>=16384 &&
        (neuron_len>=512 || tokens*area>=(1LL<<27));
    // Only enable the direct FP32 geometry that beats the existing native
    // caller. Weights are reconstructed in LDS; workspace holds activations.
    if(type==2 && tokens==2048 && neuron_len==2560 && (out==6144 || out==12288) && !view.columns &&
            mfq_tq_prefill_f32_supported(image.get_device())) {
        auto result=tb::empty(shape,x.options());
        const auto bytes=mfq_tq_prefill_f32_workspace(view,int(tokens));
        auto workspace=tb::empty({static_cast<int64_t>(bytes)},x.options().dtype(tb::kUInt8));
        status(mfq_tq_prefill_f32(view,x.data_ptr<float>(),result.data_ptr<float>(),int(tokens),
            workspace.data_ptr(),bytes,mfq_current_cuda_stream()));
        return result;
    }
    if(staged) {
        const auto compute=type==0?tb::kFloat16:tb::kFloat32;
        auto w=materialize(compute);
        return tb::linear(x.reshape({tokens,neuron_len}).to(compute),w)
            .to(x.scalar_type()).reshape(shape);
    }
    auto result=tb::empty(shape,x.options());
    const auto bytes=x.element_size();
    for(std::int64_t start=0;start<tokens;start+=65535*16) {
        status(mfq_tq_forward(view,static_cast<const std::uint8_t*>(x.data_ptr())+start*neuron_len*bytes,
            static_cast<std::uint8_t*>(result.data_ptr())+start*out*bytes,
            int(std::min<std::int64_t>(tokens-start,65535*16)),type,mfq_current_cuda_stream()));
    }
    return result;
}

tb::Tensor tq_gr_activation(const TqWeight& weight,tb::Tensor input,int64_t streams,bool injection) {
    auto x=(input.scalar_type()==tb::kFloat16?input:input.to(tb::kFloat32)).contiguous();
    const auto tokens=x.numel()/weight.neuron_len;
    if(x.is_cuda() && x.size(-1)==weight.neuron_len && tokens>0 && tokens<=8 && streams>0 && streams<=INT_MAX) {
        MfqCudaGuard guard(x.device());auto shape=x.sizes().vec();shape.back()=weight.out;
        auto output=tb::empty(shape,x.options());
        status(mfq_tq_gr(weight.view,x.data_ptr(),nullptr,output.data_ptr(),int(tokens),int(streams),
            injection?2:1,dtype(x.scalar_type()),mfq_current_cuda_stream()));
        return output;
    }
    auto projected=weight.forward(x);
    return injection?mfq_qwen4_exp::gated_residual_injection(projected,streams):
        mfq_qwen4_exp::gated_residual_bottleneck(projected,streams);
}

tb::Tensor tq_gr_mix(const TqWeight& weight,tb::Tensor input,tb::Tensor normalized,int64_t streams) {
    auto x=(input.scalar_type()==tb::kFloat16?input:input.to(tb::kFloat32)).contiguous();
    const auto tokens=x.numel()/weight.neuron_len;
    if(x.is_cuda() && x.size(-1)==weight.neuron_len && tokens>0 && tokens<=8 && streams>0 && streams<=4 &&
        weight.out%streams==0 && normalized.scalar_type()==x.scalar_type() && normalized.is_contiguous() &&
        normalized.device()==x.device() && normalized.size(-1)==weight.out && normalized.numel()==tokens*weight.out) {
        MfqCudaGuard guard(x.device());auto shape=x.sizes().vec();shape.back()=weight.out/streams;
        auto output=tb::empty(shape,x.options());
        status(mfq_tq_gr(weight.view,x.data_ptr(),normalized.data_ptr(),output.data_ptr(),int(tokens),int(streams),
            3,dtype(x.scalar_type()),mfq_current_cuda_stream()));
        return output;
    }
    return mfq_qwen4_exp::gated_residual_mix(weight.forward(x),normalized,streams);
}

namespace {
struct TqHipRoutes {
    tb::Tensor ids;
    std::vector<tb::Tensor> map;
    TqMoeRoutePlan plan{};
    TqHipRoutes(tb::Tensor route_ids,int experts,const tb::Tensor& source) {
        if(route_ids.dim()!=2 || route_ids.size(0)!=source.size(0) || route_ids.size(1)<1 ||
            route_ids.numel()>INT_MAX || route_ids.device()!=source.device())
            throw std::invalid_argument("invalid TQ expert routes");
        ids=route_ids.to(tb::kInt32).contiguous();
        plan.ids=ids.data_ptr<int32_t>();plan.tokens=ids.size(0);plan.routes=ids.size(1);plan.expert_count=experts;
        hipDeviceProp_t properties{};status(hipGetDeviceProperties(&properties,source.get_device()));
        if(properties.warpSize!=64)throw std::invalid_argument("TQ MoE currently requires a CDNA wave64 device");
        if((plan.tokens>6 && int64_t(plan.tokens)*plan.routes>=4LL*experts) || ids.numel()>65535) {
            // Reuse decoded weights while limiting padding for sparsely populated experts.
            plan.tile_rows=plan.tokens>=1024 && int64_t(plan.tokens)*plan.routes>=32LL*experts?32:16;
            if(source.scalar_type()==tb::kFloat16 && plan.tile_rows==32 &&
                (int64_t(plan.tokens)*plan.routes<64LL*experts || plan.tokens>=2048))plan.tile_rows=64;
            map=moe_build_expert_map_cuda(ids,experts,plan.tile_rows);
            plan.pairs=map[0].data_ptr<int32_t>();plan.bounds=map[1].data_ptr<int32_t>();
            plan.tiles=map[2].data_ptr<int32_t>();plan.experts=map[3].data_ptr<int32_t>();
        }
    }
};

}
static TqWeight load_tq_expert_pool(const mfq::ModelSource& source,const std::string& name,
        int64_t count,int64_t outputs,int64_t columns,int device,int64_t begin,int64_t finish) {
    const auto* record=source.find_tensor(name);
    if(!record || record->dtype!="TQ" || record->nbytes<64)throw std::invalid_argument("invalid TQ expert record");
    auto read=source.tensor_reader(name);
    std::vector<uint8_t> blob(64);
    read(0,reinterpret_cast<std::byte*>(blob.data()),blob.size());
    const auto integer=[&](size_t offset,int bytes){return mfq::TqLayout::integer(blob.data()+offset,bytes);};
    if(std::memcmp(blob.data(),"RTQEXP01",8) || integer(8,4)!=1 || integer(12,4)!=uint64_t(count) ||
       integer(16,4)!=uint64_t(outputs) || integer(20,4)!=uint64_t(columns) || integer(24,4)!=16 ||
       integer(28,4) || count<1 || outputs<1 || count>INT_MAX/outputs || columns<1 || columns>INT_MAX ||
       uint64_t(count)>(record->nbytes-64)/16 ||
       std::any_of(blob.begin()+32,blob.begin()+64,[](uint8_t v){return v!=0;}))
        throw std::invalid_argument("invalid TQ expert container header");
    const size_t table_bytes=64+size_t(count)*16;
    blob.resize(table_bytes);read(64,reinterpret_cast<std::byte*>(blob.data()+64),table_bytes-64);
    size_t end=table_bytes;
    for(int64_t expert=0;expert<count;++expert) {
        const auto offset=integer(64+expert*16,8),bytes=integer(72+expert*16,8);
        if(offset!=end || offset>record->nbytes || bytes<64 || bytes>record->nbytes-offset)
            throw std::invalid_argument("invalid TQ expert container span");
        end=offset+bytes;
    }
    if(end!=record->nbytes)throw std::invalid_argument("TQ expert container length differs");
    const size_t payload_begin=integer(64+begin*16,8);
    const size_t payload_end=finish==count?record->nbytes:integer(64+finish*16,8);
    blob.resize(table_bytes+payload_end-payload_begin);
    read(payload_begin,reinterpret_cast<std::byte*>(blob.data()+table_bytes),payload_end-payload_begin);
    std::vector<mfq::TqLayout> layouts;layouts.reserve(finish-begin);
    size_t total_packed=0,stored_width=0,group_size=1u<<20;
    bool column_scales=false;
    for(int64_t expert=begin;expert<finish;++expert) {
        const auto offset=integer(64+expert*16,8),bytes=integer(72+expert*16,8);
        const mfq::TqLayout layout(blob.data()+table_bytes+offset-payload_begin,bytes);
        if(layout.rows!=uint64_t(outputs) || layout.width<uint64_t(columns) || layout.width>INT_MAX ||
           (expert!=begin && (layout.width!=stored_width || layout.columns!=column_scales)) || layout.packed_bytes%4)
            throw std::invalid_argument("TQ expert container dimensions differ");
        stored_width=layout.width;column_scales=layout.columns;
        if(!column_scales)group_size=std::min(group_size,size_t(layout.gs));
        total_packed+=layout.packed_bytes;
        layouts.push_back(layout);
    }
    if(total_packed/4>UINT32_MAX)throw std::invalid_argument("TQ expert packed offsets exceed limits");
    const size_t rows=size_t(finish-begin)*outputs,groups=column_scales?0:(stored_width+group_size-1)/group_size;
    const size_t scales=64+rows*8,scale_bytes=column_scales?size_t(finish-begin)*stored_width*2:(rows*groups+1)/2;
    const size_t words=(scales+scale_bytes+3)&~size_t(3),relative=words+total_packed;
    std::vector<uint8_t> image(relative+32);std::memcpy(image.data(),blob.data(),64);
    size_t base_words=0;
    for(int64_t expert=0;expert<finish-begin;++expert) {
        const auto index=expert+begin;
        const auto offset=integer(64+index*16,8);const auto* child=blob.data()+table_bytes+offset-payload_begin;const auto& layout=layouts[expert];
        uint64_t expected=0;
        for(int64_t row=0;row<outputs;++row) {
            const auto descriptor=mfq::TqLayout::integer(child+64+row*8,8);
            if((descriptor & ~0x00ffff07ffffffffULL) || ((descriptor>>40)&0xffff)>0x7bff || uint32_t(descriptor)!=expected)
                throw std::invalid_argument("invalid TQ expert row descriptor");
            const auto normalized=(descriptor & 0xffffffff00000000ULL)|(uint64_t(base_words)+expected);
            mfq::TqLayout::integer(image.data()+64+(size_t(expert)*outputs+row)*8,normalized,8);
            expected+=uint64_t(stored_width)*(((descriptor>>32)&7)+1)/32;
        }
        if(expected*4!=layout.packed_bytes)throw std::invalid_argument("TQ expert packed offsets differ");
        if(column_scales) {
            for(size_t column=0;column<stored_width;++column) {
                const auto value=mfq::TqLayout::integer(child+layout.scales+column*2,2);
                if(!value || value>0x7bff)throw std::invalid_argument("invalid TQ expert column scale");
            }
            std::memcpy(image.data()+scales+size_t(expert)*stored_width*2,child+layout.scales,stored_width*2);
        } else if(layout.gs==group_size && (size_t(outputs)*groups)%2==0) {
            std::memcpy(image.data()+scales+size_t(expert)*outputs*groups/2,child+layout.scales,layout.scale_bytes);
        } else {
            for(size_t row=0;row<size_t(outputs);++row)for(size_t group=0;group<groups;++group) {
                const auto index=row*layout.groups+group*group_size/layout.gs;
                const auto code=(child[layout.scales+index/2]>>((index&1)*4))&15;
                const auto target=(size_t(expert)*outputs+row)*groups+group;
                image[scales+target/2]|=code<<((target&1)*4);
            }
        }
        std::memcpy(image.data()+words+base_words*4,child+layout.packed,layout.packed_bytes);
        base_words+=layout.packed_bytes/4;
    }
    constexpr uint16_t gains[]{0x3400,0x34d0,0x35ca,0x36f7,0x3830,0x390a,0x3a10,0x3b4b,
        0x3c63,0x3d47,0x3e59,0x3fa4,0x4098,0x4187,0x42a6,0x4400};
    std::memcpy(image.data()+relative,gains,sizeof(gains));
    MfqCudaGuard guard(device);const auto stream=mfq_get_current_cuda_stream(device);
    MfqCudaStreamGuard stream_guard(stream);TqWeight result;
    result.image=tb::from_blob(image.data(),{int64_t(image.size())},tb::TensorOptions().dtype(tb::kUInt8))
        .to(tb::Device(tb::kCUDA,device),false,false).contiguous();
    status(int(hipStreamSynchronize(mfq_current_cuda_stream())));
    const auto* address=result.image.data_ptr<uint8_t>();int gslog=0;
    if(!column_scales)for(auto gs=group_size;gs>1;gs>>=1)++gslog;
    result.view={reinterpret_cast<const uint32_t*>(address+words),reinterpret_cast<const uint64_t*>(address+64),
        address+scales,address+relative,nullptr,int(rows),int(stored_width),int(columns),gslog,column_scales?2:0};
    result.out=rows;result.neuron_len=columns;return result;
}

TqRoutedWeight::TqRoutedWeight(const mfq::ModelSource& source,const std::string& name,int device,
        int64_t count,int64_t outputs,int64_t columns,int64_t begin,int64_t end):out(outputs),input(columns) {
    if(end<0)end=count;
    if(begin<0 || end<=begin || end>count)throw std::invalid_argument("invalid TQ expert shard");
    experts=end-begin;
    std::array<uint8_t,8> magic{};source.tensor_reader(name)(0,reinterpret_cast<std::byte*>(magic.data()),magic.size());
    if(std::memcmp(magic.data(),"RTQEXP01",8)==0)
        weight=load_tq_expert_pool(source,name,count,outputs,columns,device,begin,end);
    else if(begin==0 && end==count)weight=load_tq_weight(source,name,device);
    else {
        const auto& record=*source.find_tensor(name);
        mfq::TqRows rows(record.nbytes,source.tensor_reader(name));
        weight=to_device_tq(rows.slice_rows(begin*outputs,end*outputs),columns,device);
    }
    if(experts<1 || experts>4096 || out<1 || out>INT_MAX || input<1 || input>INT_MAX ||
        experts>INT_MAX/out || weight.out!=experts*out || weight.neuron_len!=input)
        throw std::invalid_argument("TQ expert bank shape differs: "+name);
}
tb::Tensor TqRoutedWeight::forward(tb::Tensor input_value,tb::Tensor ids)const {
    if(!shards.empty()) {
        std::vector<tb::Tensor> partials;partials.reserve(shards.size());
        for(const auto& shard:shards) {
            MfqCudaGuard guard(shard.device);
            const auto stream=mfq_get_current_cuda_stream(shard.device);MfqCudaStreamGuard stream_guard(stream);
            const char* stage="copy input";
            try {
                auto x=tensor_to_cuda_device(execution->model_parallel_collectives,input_value,shard.device);
                stage="copy routes";auto local=tensor_to_cuda_device(execution->model_parallel_collectives,ids,shard.device);
                stage="shift routes";local=local-shard.begin;
                stage="projection";partials.push_back(shard.weight->forward(x,local));
            }catch(const std::exception& e) {
                throw std::runtime_error("TQ shard "+std::to_string(shard.device)+" "+stage+": "+e.what());
            }
        }
        try { return reduce_model_parallel_outputs(*execution,std::move(partials)); }
        catch(const std::exception& e) {throw std::runtime_error(std::string("TQ shard reduction: ")+e.what());}
    }
    if(!input_value.is_cuda() || input_value.get_device()!=weight.image.get_device() ||
        (input_value.dim()!=2 && input_value.dim()!=3) || input_value.size(-1)!=input ||
        ids.dim()!=2 || input_value.size(0)!=ids.size(0) ||
        (input_value.dim()==3 && input_value.size(1)!=ids.size(1)))
        throw std::invalid_argument("invalid TQ routed projection input");
    MfqCudaGuard guard(input_value.device());auto x=input_value.contiguous();
    auto output=tb::zeros({ids.size(0),ids.size(1),out},x.options());
    if(!ids.numel())return output;
    TqHipRoutes route(ids,int(experts),x);
    status(mfq_tq_moe_projection(weight.view,x.data_ptr(),output.data_ptr(),int(out),route.plan,
        dtype(x.scalar_type()),x.dim()==3,mfq_current_cuda_stream()));
    return output;
}
namespace {
bool tq_peer_access(int a,int b) {
    if(a<0 || a>=16 || b<0 || b>=16 || a==b)return false;
    static std::atomic<uint64_t> enabled[4]{};
    const unsigned key=unsigned(a*16+b),word=key/64;const uint64_t bit=uint64_t(1)<<(key%64);
    if(enabled[word].load(std::memory_order_relaxed)&bit)return true;
    for(int direction=0;direction<2;++direction) {
        const int source=direction?b:a,destination=direction?a:b;
        hipMemPool_t pool=nullptr;hipMemAccessFlags flags=hipMemAccessFlagsProtNone;
        hipMemLocation location{};location.type=hipMemLocationTypeDevice;location.id=destination;
        if(hipDeviceGetDefaultMemPool(&pool,source)!=hipSuccess || hipMemPoolGetAccess(&flags,pool,&location)!=hipSuccess ||
            flags!=hipMemAccessFlagsProtReadWrite)return false;
    }
    enabled[word].fetch_or(bit,std::memory_order_relaxed);return true;
}
}
struct TqMoePeer {
    struct Buffers { tb::Tensor hidden,pairs,partial,input,route_ids; };
    std::array<std::array<Buffers,2>,6> buffers;
    std::array<Buffers,2> prefill_buffers;
    int primary,secondary;
    hipEvent_t ready=nullptr,done=nullptr,joined=nullptr;
    hipStream_t last_primary=nullptr;
    bool pending=false;
    TqMoePeer(int a,int b):primary(a),secondary(b) {
        { MfqCudaGuard guard(primary);status(hipEventCreateWithFlags(&ready,hipEventDisableTiming));
          status(hipEventCreateWithFlags(&joined,hipEventDisableTiming)); }
        { MfqCudaGuard guard(secondary);status(hipEventCreateWithFlags(&done,hipEventDisableTiming)); }
    }
    ~TqMoePeer() {
        { MfqCudaGuard guard(primary);if(pending)(void)hipEventSynchronize(joined);
          if(ready)(void)hipEventDestroy(ready);if(joined)(void)hipEventDestroy(joined); }
        { MfqCudaGuard guard(secondary);if(done)(void)hipEventDestroy(done); }
    }
    tb::Tensor forward(const TqMoeBanks& banks,tb::Tensor x,tb::Tensor ids,tb::Tensor weights,float limit) {
        MfqCudaGuard primary_guard(primary);const auto parent=mfq_get_current_cuda_stream(primary);
        MfqCudaStreamGuard parent_guard(parent);
        if(pending && parent.stream()!=last_primary)status(hipStreamWaitEvent(parent.stream(),joined,0));
        x=x.contiguous();auto route_ids=ids.to(tb::kInt32).contiguous();auto coefficients=weights.to(tb::kFloat32).contiguous();
        auto output=tb::empty({x.size(0),banks.down->out},x.options());
        const bool prefill=x.size(0)>6;
        if(prefill && pending && prefill_buffers[0].hidden.defined() &&
                (prefill_buffers[0].hidden.size(0)!=x.size(0) || prefill_buffers[0].hidden.size(1)!=ids.size(1)))
            status(hipEventSynchronize(joined));
        status(hipEventRecord(ready,parent.stream()));
        for(size_t launch_index=0;launch_index<2;++launch_index) {
            const size_t i=1-launch_index;
            const auto& g=banks.gate->shards[i];const auto& u=banks.up->shards[i];const auto& d=banks.down->shards[i];
            MfqCudaGuard guard(g.device);const auto stream=mfq_get_current_cuda_stream(g.device);MfqCudaStreamGuard stream_guard(stream);
            if(g.device!=primary)status(hipStreamWaitEvent(stream.stream(),ready,0));
            auto& b=prefill?prefill_buffers[i]:buffers[x.size(0)-1][i];auto options=x.options().device(tb::Device(tb::kCUDA,g.device));
            if(!b.hidden.defined() || b.hidden.size(0)!=x.size(0) || b.hidden.size(1)!=ids.size(1)) {
                b.hidden=tb::empty({x.size(0),ids.size(1),banks.gate->out},options);
                b.pairs=tb::empty({x.size(0),ids.size(1),banks.down->out},options);
                b.partial=tb::empty({x.size(0),banks.down->out},options);
            }
            std::unique_ptr<TqHipRoutes> grouped_route;
            auto local_input=x;
            TqMoeRoutePlan plan{};
            if(prefill) {
                auto local_ids=route_ids;
                if(g.device!=primary) {
                    if(!b.input.defined() || b.input.sizes()!=x.sizes())b.input=tb::empty(x.sizes(),options);
                    if(!b.route_ids.defined() || b.route_ids.sizes()!=ids.sizes())
                        b.route_ids=tb::empty(ids.sizes(),options.dtype(tb::kInt32));
                    status(hipMemcpyPeerAsync(b.input.data_ptr(),g.device,x.data_ptr(),primary,x.numel()*2,stream.stream()));
                    status(hipMemcpyPeerAsync(b.route_ids.data_ptr(),g.device,route_ids.data_ptr(),primary,route_ids.numel()*4,stream.stream()));
                    local_input=b.input;local_ids=b.route_ids;
                }
                if(g.begin)local_ids=local_ids-g.begin;
                grouped_route=std::make_unique<TqHipRoutes>(local_ids,int(g.end-g.begin),local_input);
                plan=grouped_route->plan;
                status(hipMemsetAsync(b.pairs.data_ptr(),0,b.pairs.numel()*2,stream.stream()));
            } else {
                plan.ids=route_ids.data_ptr<int32_t>();plan.tokens=x.size(0);plan.routes=ids.size(1);
                plan.expert_count=g.end-g.begin;plan.expert_offset=g.begin;
            }
            status(mfq_tq_moe_swiglu(g.weight->weight.view,u.weight->weight.view,local_input.data_ptr(),b.hidden.data_ptr(),
                int(banks.gate->out),plan,0,limit,stream.stream()));
            status(mfq_tq_moe_projection(d.weight->weight.view,b.hidden.data_ptr(),b.pairs.data_ptr(),
                int(banks.down->out),plan,0,true,stream.stream()));
            status(mfq_tq_moe_reduce(b.pairs.data_ptr(),coefficients.data_ptr<float>(),b.partial.data_ptr(),
                plan.tokens,plan.routes,int(banks.down->out),0,stream.stream()));
            if(g.device!=primary)status(hipEventRecord(done,stream.stream()));
        }
        status(hipStreamWaitEvent(parent.stream(),done,0));
        const auto& b=prefill?prefill_buffers:buffers[x.size(0)-1];
        status(mfq_tq_moe_peer_join(b[0].partial.data_ptr(),b[1].partial.data_ptr(),output.data_ptr(),output.numel(),parent.stream()));
        status(hipEventRecord(joined,parent.stream()));pending=true;last_primary=parent.stream();
        static const bool log=std::getenv("MFQ_HIP_TQ_EP_PREFILL_LOG")!=nullptr;
        static std::atomic_flag logged=ATOMIC_FLAG_INIT;
        if(prefill && log && !logged.test_and_set(std::memory_order_relaxed))
            std::fprintf(stderr,"native_tq_ep_prefill streams=2 local_input_copy=1 grouped=1\n");
        return output;
    }
};
tb::Tensor TqMoeBanks::forward(tb::Tensor input_value,tb::Tensor ids,tb::Tensor weights,float limit)const {
    if(gate && !gate->shards.empty()) {
        if(!up || !down || gate->shards.size()!=up->shards.size() || gate->shards.size()!=down->shards.size())
            throw std::invalid_argument("TQ expert shard banks differ");
        const auto* enabled=std::getenv("MFQ_HIP_TQ_EP_PEER");
        const auto* pp=std::getenv("MFQ_HIP_TQ_EP_PREFILL");
        const bool prefill_enabled=!pp || pp[0]!='0';
        if((!enabled || enabled[0]!='0') && gate->shards.size()==2 && input_value.scalar_type()==tb::kFloat16 &&
            input_value.dim()==2 && input_value.size(0)>=1 && (input_value.size(0)<=6 || prefill_enabled) &&
            input_value.size(1)==gate->input && ids.dim()==2 && ids.size(0)==input_value.size(0) &&
            ids.device()==input_value.device() && weights.device()==input_value.device() && weights.sizes()==ids.sizes() &&
            gate->shards[0].device==input_value.get_device() && tq_peer_access(gate->shards[0].device,gate->shards[1].device)) {
            if(!peer)peer=std::make_shared<TqMoePeer>(gate->shards[0].device,gate->shards[1].device);
            return peer->forward(*this,input_value,ids,weights,limit);
        }
        auto& execution=*gate->execution;std::vector<tb::Tensor> partials;
        for(size_t i=0;i<gate->shards.size();++i) {
            const auto& shard=gate->shards[i];const auto& u=up->shards[i];const auto& d=down->shards[i];
            if(shard.device!=u.device || shard.device!=d.device || shard.begin!=u.begin || shard.begin!=d.begin ||
               shard.end!=u.end || shard.end!=d.end)throw std::invalid_argument("TQ expert shard boundaries differ");
            MfqCudaGuard guard(shard.device);
            const auto stream=mfq_get_current_cuda_stream(shard.device);MfqCudaStreamGuard stream_guard(stream);
            auto x=tensor_to_cuda_device(execution.model_parallel_collectives,input_value,shard.device);
            auto local=tensor_to_cuda_device(execution.model_parallel_collectives,ids,shard.device)-shard.begin;
            auto coefficients=tensor_to_cuda_device(execution.model_parallel_collectives,weights,shard.device);
            partials.push_back(TqMoeBanks{shard.weight,u.weight,d.weight}.forward(x,local,coefficients,limit));
        }
        return reduce_model_parallel_outputs(execution,std::move(partials));
    }
    if(!gate || !up || !down || gate->experts!=up->experts || gate->experts!=down->experts ||
        gate->out!=up->out || gate->input!=up->input || gate->out!=down->input || gate->input!=down->out ||
        !input_value.is_cuda() || input_value.dim()!=2 || input_value.size(1)!=gate->input ||
        ids.dim()!=2 || ids.size(0)!=input_value.size(0) || weights.sizes()!=ids.sizes() ||
        weights.device()!=input_value.device() || gate->weight.image.device()!=input_value.device() ||
        up->weight.image.device()!=input_value.device() || down->weight.image.device()!=input_value.device())
        throw std::invalid_argument("invalid TQ three-bank MoE geometry");
    MfqCudaGuard guard(input_value.device());const auto x=input_value.contiguous();
    if(!ids.numel())return tb::zeros({x.size(0),down->out},x.options());
    auto output=tb::empty({x.size(0),down->out},x.options());
    auto coefficients=weights.to(tb::kFloat32).contiguous();TqHipRoutes route(ids,int(gate->experts),x);
    auto hidden=tb::empty({ids.size(0),ids.size(1),gate->out},x.options());
    // SIMT writes every pair, including invalid routes. Grouped tiles visit valid pairs only.
    const std::vector<int64_t> pair_shape{ids.size(0),ids.size(1),down->out};
    auto pairs=route.plan.pairs?tb::zeros(pair_shape,x.options()):tb::empty(pair_shape,x.options());
    const auto stream=mfq_current_cuda_stream();const int type=dtype(x.scalar_type());
    status(mfq_tq_moe_swiglu(gate->weight.view,up->weight.view,x.data_ptr(),hidden.data_ptr(),int(gate->out),route.plan,type,limit,stream));
    status(mfq_tq_moe_projection(down->weight.view,hidden.data_ptr(),pairs.data_ptr(),int(down->out),route.plan,type,true,stream));
    status(mfq_tq_moe_reduce(pairs.data_ptr(),coefficients.data_ptr<float>(),output.data_ptr(),int(ids.size(0)),int(ids.size(1)),int(down->out),type,stream));
    return output;
}

namespace {
std::vector<std::int64_t> logical_shape(const mfq::ModelSource& source,const std::string& name) {
    const auto* record=source.find_tensor(name);
    if(record && record->dtype!="TQ")throw std::invalid_argument("expected TQ tensor: "+name);
    std::vector<std::int64_t> shape;
    bool found=false;
    for(const auto* key:{"tq_shapes","tensor_shapes"}) {
        const auto metadata=source.metadata().find(key);
        if(metadata==source.metadata().end())continue;
        const auto shapes=nlohmann::json::parse(metadata->second);
        if(!shapes.is_object())throw std::invalid_argument("invalid TQ shape metadata");
        if(shapes.contains(name)) {
            shape=shapes.at(name).get<std::vector<std::int64_t>>();found=true;break;
        }
    }
    if(!found && record && record->shape)shape=*record->shape;
    if(shape.empty() && record && record->nbytes>=64) {
        std::array<uint8_t,64> header{};source.tensor_reader(name)(0,reinterpret_cast<std::byte*>(header.data()),header.size());
        if(!std::memcmp(header.data(),"RTQEXP01",8))shape={int64_t(mfq::TqLayout::integer(header.data()+12,4)),
            int64_t(mfq::TqLayout::integer(header.data()+16,4)),int64_t(mfq::TqLayout::integer(header.data()+20,4))};
        else if(!std::memcmp(header.data(),"RTRELL05",8))shape={int64_t(mfq::TqLayout::integer(header.data()+12,4)),
            int64_t(mfq::TqLayout::integer(header.data()+16,4))};
    }
    if(shape.size()!=2 && shape.size()!=3)
        throw std::invalid_argument("TQ logical tensor shape must have rank 2 or 3: "+name);
    std::uint64_t count=1;
    for(const auto dimension:shape) {
        if(dimension<1 || count>UINT64_MAX/std::uint64_t(dimension))
            throw std::invalid_argument("invalid TQ logical tensor shape: "+name);
        count*=dimension;
    }
    return shape;
}
std::vector<std::string> row_parts(const mfq::ModelSource& source,const std::string& name) {
    for(const auto* key:{"tq_row_shards","tq.segments"}) {
        const auto entry=source.metadata().find(key);
        if(entry==source.metadata().end())continue;
        const auto tables=nlohmann::json::parse(entry->second);
        if(!tables.is_object())throw std::invalid_argument("invalid TQ row-shard manifest");
        if(!tables.contains(name))continue;
        if(std::string_view(key)=="tq_row_shards" && source.find_tensor(name))
            throw std::invalid_argument("ambiguous TQ row table: "+name);
        const auto names=tables.at(name).get<std::vector<std::string>>();
        if(names.empty())throw std::invalid_argument("empty TQ row table: "+name);
        return names;
    }
    return {name};
}
std::vector<std::uint8_t> join_rows(const std::vector<std::vector<std::uint8_t>>& blobs) {
    if(blobs.empty())throw std::invalid_argument("empty TQ row selection");
    const mfq::TqLayout first(blobs[0].data(),blobs[0].size());
    std::uint64_t rows=0,packed_bytes=0;
    for(const auto& blob:blobs) {
        const mfq::TqLayout layout(blob.data(),blob.size());
        if(layout.width!=first.width || layout.columns!=first.columns || layout.gs!=first.gs)
            throw std::invalid_argument("incompatible TQ row segments");
        rows+=layout.rows;packed_bytes+=layout.packed_bytes;
    }
    if(rows>UINT32_MAX || packed_bytes>UINT32_MAX)throw std::overflow_error("selected TQ rows exceed RTRELL05 limits");
    const auto scale_bytes=first.columns?first.scale_bytes:(rows*first.groups+1)/2;
    const auto scale_offset=64+rows*8,packed_offset=scale_offset+scale_bytes;
    std::vector<std::uint8_t> result(packed_offset+packed_bytes);
    std::memcpy(result.data(),blobs[0].data(),64);
    mfq::TqLayout::integer(result.data()+12,rows,4);
    mfq::TqLayout::integer(result.data()+28,packed_bytes,4);
    std::uint64_t row_offset=0,byte_offset=0;
    if(first.columns)std::memcpy(result.data()+scale_offset,blobs[0].data()+first.scales,scale_bytes);
    for(const auto& blob:blobs) {
        const mfq::TqLayout m(blob.data(),blob.size());
        for(std::size_t row=0;row<m.rows;++row) {
            auto descriptor=mfq::TqLayout::integer(blob.data()+64+row*8,8);
            descriptor+=byte_offset/4;
            mfq::TqLayout::integer(result.data()+64+(row_offset+row)*8,descriptor,8);
        }
        if(!m.columns)for(std::size_t i=0;i<std::size_t(m.rows)*m.groups;++i) {
            const auto selector=(blob[m.scales+i/2]>>((i&1)*4))&15;
            const auto index=row_offset*m.groups+i;
            result[scale_offset+index/2]|=selector<<((index&1)*4);
        }
        std::memcpy(result.data()+packed_offset+byte_offset,blob.data()+m.packed,m.packed_bytes);
        row_offset+=m.rows;byte_offset+=m.packed_bytes;
    }
    return result;
}
}

namespace {
std::vector<std::uint8_t> tq_image(const std::vector<std::uint8_t>& blob,const mfq::TqLayout& m) {
    // The serialized scale stream can end at an odd byte. Align the packed
    // word stream in one compressed image instead of uploading fields apart.
    const auto words_offset=(m.packed+3)&~std::size_t(3);
    const auto relative_offset=words_offset+m.packed_bytes;
    std::vector<std::uint8_t> image(relative_offset+32);
    std::memcpy(image.data(),blob.data(),m.packed);
    std::memcpy(image.data()+words_offset,blob.data()+m.packed,m.packed_bytes);
    // Exact FP16 RELATIVE table from the TQ format, independent of device.
    constexpr std::uint16_t relative[]{0x3400,0x34d0,0x35ca,0x36f7,0x3830,0x390a,0x3a10,0x3b4b,
        0x3c63,0x3d47,0x3e59,0x3fa4,0x4098,0x4187,0x42a6,0x4400};
    std::memcpy(image.data()+relative_offset,relative,sizeof(relative));
    return image;
}

}
TqRowTable::TqRowTable(const mfq::ModelSource& source,const std::string& name) {
    const auto shape=logical_shape(source,name);
    columns_=shape.back();
    const auto names=row_parts(source,name);
    std::unordered_set<std::string> unique;
    std::uint64_t end=0;
    for(const auto& part:names) {
        const auto* record=source.find_tensor(part);
        if(!unique.insert(part).second || !record || record->dtype!="TQ")
            throw std::invalid_argument("invalid TQ row segment: "+part);
        // Keep the compressed table on the CPU. Sparse row selection should
        // copy only the requested ranges, without a seek/read syscall for
        // every descriptor, selector and payload in a long prompt.
        auto reader=source.mapped_tensor_reader(part);
        if(!reader){parallel_reads_=false;reader=source.tensor_reader(part);}
        auto rows=std::make_shared<mfq::TqRows>(record->nbytes,std::move(reader));
        if(columns_<1 || columns_>rows->layout().width)throw std::invalid_argument("TQ table logical width differs");
        if(end>UINT64_MAX-rows->layout().rows || storage_bytes_>UINT64_MAX-record->nbytes)
            throw std::overflow_error("TQ row table size exceeds bounds");
        storage_bytes_+=record->nbytes;
        end+=rows->layout().rows;ends_.push_back(end);segments_.push_back(std::move(rows));
    }
    std::uint64_t expected=1;
    for(std::size_t i=0;i+1<shape.size();++i)expected*=shape[i];
    if(end!=expected)throw std::invalid_argument("TQ table row count differs");
}
std::vector<std::uint8_t> TqRowTable::slice_rows(std::uint64_t begin,std::uint64_t end)const {
    if(begin>=end || end>rows())throw std::out_of_range("invalid TQ table row range");
    std::uint64_t start=0;
    std::vector<std::vector<std::uint8_t>> pieces;
    for(std::size_t i=0;i<segments_.size();++i) {
        const auto a=std::max(begin,start),b=std::min(end,ends_[i]);
        if(a<b)pieces.push_back(segments_[i]->slice_rows(a-start,b-start));
        start=ends_[i];if(start>=end)break;
    }
    if(pieces.size()==1)return std::move(pieces[0]);
    return join_rows(pieces);
}

std::size_t TqRowTable::row_image_bytes_bound()const {
    std::size_t bytes=0;
    for(const auto& segment:segments_) {
        const auto& m=segment->layout();
        if(m.width>INT_MAX)throw std::overflow_error("TQ row stage width exceeds CUDA limits");
        const auto scales=m.columns?m.scale_bytes:(m.groups+1)/2;
        // One descriptor, aligned packed words, up to8 bits/weight, relative table.
        bytes=std::max(bytes,((64+8+scales+3)&~std::size_t(3))+m.width+32);
    }
    return (bytes+7)&~std::size_t(7);
}
ProjectionView TqRowTable::stage_row(std::uint64_t row,std::uint8_t* host,
        std::uint8_t* device,std::size_t capacity)const {
    if(row>=rows())throw std::out_of_range("TQ staged row is out of bounds");
    const auto segment=std::upper_bound(ends_.begin(),ends_.end(),row)-ends_.begin();
    const auto start=segment?ends_[segment-1]:0;
    const auto& m=segments_[segment]->layout();
    const auto scale_bytes=m.columns?m.scale_bytes:(m.groups+1)/2;
    const auto packed=72+scale_bytes,words=(packed+3)&~std::size_t(3);
    if(capacity<words+std::size_t(m.width)+32)
        throw std::overflow_error("TQ staged row image exceeds capacity");
    const auto wire=segments_[segment]->slice_row_into(row-start,host,capacity);
    const auto packed_bytes=wire-packed,relative=words+packed_bytes;
    std::memmove(host+words,host+packed,packed_bytes);
    constexpr std::uint16_t gains[]{0x3400,0x34d0,0x35ca,0x36f7,0x3830,0x390a,0x3a10,0x3b4b,
        0x3c63,0x3d47,0x3e59,0x3fa4,0x4098,0x4187,0x42a6,0x4400};
    std::memcpy(host+relative,gains,sizeof(gains));
    int gslog=0;for(auto gs=m.gs;gs>1;gs>>=1)++gslog;
    return {reinterpret_cast<const std::uint32_t*>(device+words),
        reinterpret_cast<const std::uint64_t*>(device+64),device+72,device+relative,nullptr,
        1,int(m.width),int(columns_),gslog,int(m.columns)};
}
tb::Tensor TqRowTable::lookup(const std::int64_t* ids,std::size_t count,const tb::Device& device)const {
    if(!device.is_cuda() || !count || !ids)throw std::invalid_argument("TQ table lookup requires CUDA and row IDs");
    std::vector<std::int64_t> selected(ids,ids+count);
    std::sort(selected.begin(),selected.end());selected.erase(std::unique(selected.begin(),selected.end()),selected.end());
    if(selected.front()<0 || std::uint64_t(selected.back())>=rows())throw std::out_of_range("TQ table lookup row is out of bounds");
    // Shared table IDs occur across tokens and heads. Read each compressed
    // row once, coalesce adjacent rows, then restore the caller's order on GPU.
    // Sorting also groups metadata reads without fetching intervening weights.
    const MfqCudaGuard guard(device);
    const auto gather=[&](std::size_t segment,std::size_t begin,std::size_t end) {
        const auto start=segment?ends_[segment-1]:0;
        std::vector<std::vector<std::uint8_t>> pieces;
        for(auto run=begin;run<end;) {
            auto next=run+1;
            while(next<end && selected[next]==selected[next-1]+1)++next;
            pieces.push_back(segments_[segment]->slice_rows(selected[run]-start,selected[next-1]+1-start));
            run=next;
        }
        return to_device_tq(join_rows(pieces),columns_,device.index);
    };
    const auto first=std::upper_bound(ends_.begin(),ends_.end(),selected.front())-ends_.begin();
    const auto last=std::upper_bound(ends_.begin(),ends_.end(),selected.back())-ends_.begin();
    if(first==last) {
        // Preserve the single-segment path: decode duplicates directly in
        // caller order, without an intermediate table or gather kernel.
        const auto weight=gather(first,0,selected.size());
        if(selected.size()!=count || !std::equal(selected.begin(),selected.end(),ids)) {
            std::vector<std::int64_t> inverse;inverse.reserve(count);
            for(std::size_t i=0;i<count;++i)
                inverse.push_back(std::lower_bound(selected.begin(),selected.end(),ids[i])-selected.begin());
            const auto index=tb::from_blob(inverse.data(),{static_cast<std::int64_t>(count)},tb::TensorOptions().dtype(tb::kInt64))
                .clone().to(device,false,false);
            return weight.embedding(index);
        }
        if(count>65535)return weight.embedding(tb::arange(count,weight.image.options().dtype(tb::kInt64)));
        auto result=tb::empty({static_cast<std::int64_t>(count),columns_},weight.image.options().dtype(tb::kFloat16));
        status(mfq_tq_embedding_hip(weight.view,nullptr,static_cast<int>(count),result.data_ptr(),0,mfq_current_cuda_stream()));
        return result;
    }
    auto result=tb::empty({static_cast<std::int64_t>(selected.size()),columns_},
        tb::TensorOptions().device(device).dtype(tb::kFloat16));
    std::size_t begin=0;
    // Decode each physical segment separately: its padding, group size and
    // column scales need not match the other segments of this logical table.
    for(std::size_t segment=0;segment<segments_.size() && begin<selected.size();++segment) {
        const auto end=std::lower_bound(selected.begin()+begin,selected.end(),ends_[segment])-selected.begin();
        if(std::size_t(end)>begin) {
            const auto weight=gather(segment,begin,end);
            const auto count=end-begin;
            if(count>65535) {
                const auto index=tb::arange(count,weight.image.options().dtype(tb::kInt64));
                result.slice(0,begin,end).copy_(weight.embedding(index));
            } else status(mfq_tq_embedding_hip(weight.view,nullptr,static_cast<int>(count),
                static_cast<std::uint8_t*>(result.data_ptr())+begin*columns_*sizeof(std::uint16_t),
                0,mfq_current_cuda_stream()));
        }
        begin=end;
    }
    if(selected.size()!=count || !std::equal(selected.begin(),selected.end(),ids)) {
        std::vector<std::int64_t> inverse;inverse.reserve(count);
        for(std::size_t i=0;i<count;++i)inverse.push_back(std::lower_bound(selected.begin(),selected.end(),ids[i])-selected.begin());
        const auto index=tb::from_blob(inverse.data(),{static_cast<std::int64_t>(count)},tb::TensorOptions().dtype(tb::kInt64))
            .clone().to(device,false,false);
        return result.index_select(0,index);
    }
    return result;
}
tb::Tensor TqRowTable::lookup(tb::Tensor ids)const {
    auto host=ids.contiguous().cpu().to(tb::kInt64);
    auto shape=ids.sizes().vec();shape.push_back(columns_);
    if(!ids.numel())return tb::empty(shape,ids.options().dtype(tb::kFloat16));
    return lookup(host.data_ptr<std::int64_t>(),host.numel(),ids.device()).reshape(shape);
}

tb::Tensor lookup_tq_shards(const std::vector<std::shared_ptr<TqRowTable>>& shards,
        std::int64_t rows_per_shard,const std::vector<std::int64_t>& ids,const tb::Device& device) {
    if(shards.empty() || !shards.front() || rows_per_shard<1 || !device.is_cuda())
        throw std::invalid_argument("invalid TQ table shard geometry");
    const auto columns=shards.front()->width();
    for(const auto& table:shards)if(!table || table->width()!=columns)
        throw std::invalid_argument("TQ table shard widths differ");
    static std::atomic<bool> reported{false};
    if(std::getenv("MFQ_TQ_PROFILE_CACHE") && !reported.exchange(true))
        std::fprintf(stderr,"tq_table_shards_dispatch enabled=1 shards=%zu lookups=%zu width=%lld CPU_ids=1\n",
            shards.size(),ids.size(),static_cast<long long>(columns));
    const MfqCudaGuard guard(device);
    auto result=tb::empty({static_cast<std::int64_t>(ids.size()),columns},
        tb::TensorOptions().device(device).dtype(tb::kFloat16));
    std::vector<std::vector<std::int64_t>> local(shards.size()),positions(shards.size());
    for(std::size_t i=0;i<ids.size();++i) {
        if(ids[i]<0 || std::uint64_t(ids[i]/rows_per_shard)>=shards.size())
            throw std::out_of_range("TQ embedding shard ID is out of bounds");
        const auto shard=ids[i]/rows_per_shard;
        local[shard].push_back(ids[i]%rows_per_shard);positions[shard].push_back(static_cast<std::int64_t>(i));
    }
    // Hashing already produced CPU IDs. Keep shard selection on CPU, read
    // only that shard's requested rows, then scatter once on GPU. There is no
    // masked whole-batch lookup or GPU->CPU round trip for each shard.
    for(std::size_t shard=0;shard<shards.size();++shard)if(!local[shard].empty()) {
        const auto values=shards[shard]->lookup(local[shard].data(),local[shard].size(),device);
        const auto index=tb::from_blob(positions[shard].data(),{static_cast<std::int64_t>(positions[shard].size())},
            tb::TensorOptions().dtype(tb::kInt64)).clone().to(device,false,false);
        result.index_copy_(0,index,values);
    }
    return result;
}

namespace {
class TqRowStage final:public RowStage {
    std::vector<std::shared_ptr<TqRowTable>> tables_;
    std::vector<std::int64_t> shape_;
    std::int64_t rows_per_shard_,count_=1,columns_=0;
    std::size_t image_stride_=0,images_offset_=0,bytes_=0;
    tb::Device device_;
    std::unique_ptr<mfq::cuda::HostBuffer> host_;
    tb::Tensor packet_,output_;
    std::uint8_t* packet_address_=nullptr;
    bool mapped_=false,charged_=false;
    std::vector<MfqCudaStream> streams_;
    void retain_stream() {
        const auto stream=mfq_get_current_cuda_stream(device_.index);
        if(std::none_of(streams_.begin(),streams_.end(),[&](const auto& s){return s.stream()==stream.stream();}))
            streams_.push_back(stream);
    }
    ProjectionView pack_row(const std::vector<std::uint8_t>& wire,std::size_t slot) {
        const mfq::TqLayout layout(wire.data(),wire.size());
        const auto image=tq_image(wire,layout);
        if(layout.rows!=1 || columns_>layout.width || image.size()>image_stride_)
            throw std::invalid_argument("TQ stage row exceeds its prepared layout");
        const auto offset=images_offset_+slot*image_stride_;
        std::memcpy(static_cast<std::uint8_t*>(host_->data())+offset,image.data(),image.size());
        auto* address=packet_address_+offset;
        const auto words=(layout.packed+3)&~std::size_t(3);
        int gslog=0;for(auto gs=layout.gs;gs>1;gs>>=1)++gslog;
        return {reinterpret_cast<const std::uint32_t*>(address+words),
            reinterpret_cast<const std::uint64_t*>(address+64),address+layout.scales,
            address+words+layout.packed_bytes,nullptr,1,int(layout.width),int(columns_),gslog,int(layout.columns)};
    }
public:
    TqRowStage(std::vector<std::shared_ptr<TqRowTable>> tables,std::int64_t rows_per_shard,
            std::vector<std::int64_t> shape,const tb::Device& device)
        :tables_(std::move(tables)),shape_(std::move(shape)),rows_per_shard_(rows_per_shard),device_(device) {
        if(tables_.empty() || !tables_[0] || rows_per_shard_<1 || !device.is_cuda())
            throw std::invalid_argument("invalid TQ row stage geometry");
        for(const auto size:shape_) {
            if(size<1 || count_>INT_MAX/size)throw std::overflow_error("TQ row stage shape exceeds bounds");
            count_*=size;
        }
        columns_=tables_[0]->width();
        for(const auto& table:tables_) {
            if(!table || table->width()!=columns_)throw std::invalid_argument("TQ row stage table widths differ");
            image_stride_=std::max(image_stride_,table->row_image_bytes_bound());
        }
        if(columns_>INT_MAX || std::uint64_t(count_)>(SIZE_MAX-7)/sizeof(ProjectionView))
            throw std::overflow_error("TQ row stage metadata exceeds bounds");
        images_offset_=(std::size_t(count_)*sizeof(ProjectionView)+7)&~std::size_t(7);
        if(std::uint64_t(count_)>(SIZE_MAX-images_offset_)/image_stride_)
            throw std::overflow_error("TQ row stage packet exceeds bounds");
        bytes_=images_offset_+std::size_t(count_)*image_stride_;
        if(bytes_>INT64_MAX)throw std::overflow_error("TQ row stage packet exceeds tensor limits");
        const MfqCudaGuard guard(device);
        retain_stream();
        int can_map=0;MFQ_CUDA_CHECK(hipDeviceGetAttribute(&can_map,hipDeviceAttributeCanMapHostMemory,device.index));
        const auto* setting=std::getenv("MFQ_PLE_MAPPED_ROWS");
        mapped_=can_map && (setting?setting[0]=='1':bytes_<=16u*1024);
        host_=std::make_unique<mfq::cuda::HostBuffer>(bytes_,mapped_);
        if(mapped_) {
            void* address=nullptr;
            const auto error=hipHostGetDevicePointer(&address,host_->data(),0);
            if(error==hipErrorNotSupported || error==hipErrorInvalidValue) {
                (void)hipGetLastError();mapped_=false;
            }else {MFQ_CUDA_CHECK(error);packet_address_=static_cast<std::uint8_t*>(address);}
        }
        const auto options=tb::TensorOptions().device(device).dtype(tb::kFloat16);
        if(!mapped_) {
            packet_=tb::empty({static_cast<std::int64_t>(bytes_)},options.dtype(tb::kUInt8));
            packet_address_=packet_.data_ptr<std::uint8_t>();
        }
        auto output_shape=shape_;output_shape.push_back(columns_);
        output_=tb::zeros(output_shape,options);
        std::memset(host_->data(),0,bytes_);
        // Valid zero-anchor rows allow graph preparation without source reads
        // or model-state changes. Actual upload publishes the requested rows.
        const auto zero_width=(std::uint64_t(columns_)+255)&~std::uint64_t(255);
        if(zero_width>INT_MAX)throw std::overflow_error("TQ row stage padding exceeds CUDA limits");
        std::vector<std::uint8_t> zero(72+zero_width/32+zero_width/8);
        std::memcpy(zero.data(),"RTRELL05",8);
        mfq::TqLayout::integer(zero.data()+8,5,4);
        mfq::TqLayout::integer(zero.data()+12,1,4);
        mfq::TqLayout::integer(zero.data()+16,zero_width,4);
        mfq::TqLayout::integer(zero.data()+24,16,4);
        mfq::TqLayout::integer(zero.data()+28,zero_width/8,4);
        const auto view=pack_row(zero,0);
        auto* views=static_cast<ProjectionView*>(host_->data());
        std::fill_n(views,count_,view);
        if(!mapped_)MFQ_CUDA_CHECK(hipMemcpyAsync(packet_address_,host_->data(),bytes_,hipMemcpyHostToDevice,mfq_current_cuda_stream()));
        MFQ_CUDA_CHECK(hipStreamSynchronize(mfq_current_cuda_stream()));
        mfq::cuda::charge_tensor_host_bytes(bytes_);charged_=true;
    }
    ~TqRowStage()override {
        // The pinned packet is not tensor storage. Fence its I/O and target
        // consumers explicitly before freeing mapped memory or DMA sources.
        int previous=0;const bool restore=hipGetDevice(&previous)==hipSuccess;
        (void)hipSetDevice(device_.index);
        for(const auto& stream:streams_)(void)hipStreamSynchronize(stream.stream());
        output_={};packet_={};
        if(charged_)mfq::cuda::tensor_host_bytes.fetch_sub(bytes_);
        if(restore)(void)hipSetDevice(previous);
    }
    tb::Tensor output()const override{return output_;}
    bool mapped_rows()const override{return mapped_;}
    void upload(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape)override {
        if(shape!=shape_ || ids.size()!=std::size_t(count_))throw std::invalid_argument("TQ stage upload shape changed");
        const MfqCudaGuard guard(device_);
        retain_stream();
        std::unordered_map<std::int64_t,std::size_t> unique;
        unique.reserve(ids.size());
        std::vector<std::int64_t> selected;selected.reserve(ids.size());
        std::vector<std::size_t> inverse;inverse.reserve(ids.size());
        auto* views=static_cast<ProjectionView*>(host_->data());
        for(const auto id:ids) {
            if(id<0 || std::uint64_t(id/rows_per_shard_)>=tables_.size())
                throw std::out_of_range("TQ stage shard ID is out of bounds");
            const auto [position,inserted]=unique.try_emplace(id,selected.size());
            if(inserted)selected.push_back(id);
            inverse.push_back(position->second);
        }
        std::vector<ProjectionView> prepared(selected.size());
        const auto gather=[&](std::size_t begin,std::size_t end) {
            for(auto index=begin;index<end;++index) {
                const auto id=selected[index],shard=id/rows_per_shard_,row=id%rows_per_shard_;
                const auto offset=images_offset_+index*image_stride_;
                prepared[index]=tables_[shard]->stage_row(row,
                    static_cast<std::uint8_t*>(host_->data())+offset,packet_address_+offset,image_stride_);
            }
        };
        if(selected.size()>=4096 && std::all_of(tables_.begin(),tables_.end(),[](const auto& table){return table->parallel_reads();})) {
            // Mapped readers copy independent ranges. Keep all HIP calls on
            // the caller thread and parallelize only CPU row preparation.
            std::vector<std::future<void>> workers;
            const auto grain=(selected.size()+3)/4;
            for(std::size_t worker=1;worker<4;++worker) {
                const auto begin=worker*grain,end=std::min(selected.size(),begin+grain);
                workers.push_back(std::async(std::launch::async,gather,begin,end));
            }
            gather(0,std::min(grain,selected.size()));
            for(auto& worker:workers)worker.get();
        } else gather(0,selected.size());
        for(std::size_t index=0;index<ids.size();++index) {
            views[index]=prepared[inverse[index]];
        }
        if(!mapped_) {
            mfq_hip_record_stream(packet_,mfq_get_current_cuda_stream(device_.index));
            const auto used=images_offset_+unique.size()*image_stride_;
            MFQ_CUDA_CHECK(hipMemcpyAsync(packet_address_,host_->data(),used,hipMemcpyHostToDevice,mfq_current_cuda_stream()));
        }
    }
    void decode()override {
        const MfqCudaGuard guard(device_);retain_stream();
        mfq_hip_record_stream(output_,mfq_get_current_cuda_stream(device_.index));
        if(packet_.defined())mfq_hip_record_stream(packet_,mfq_get_current_cuda_stream(device_.index));
        status(mfq_tq_embedding_rows_hip(reinterpret_cast<const ProjectionView*>(packet_address_),
            int(count_),int(columns_),output_.data_ptr(),0,mfq_current_cuda_stream()));
    }
};
}
std::shared_ptr<RowStage> make_tq_row_stage(const std::vector<std::shared_ptr<TqRowTable>>& tables,
        std::int64_t rows_per_shard,const std::vector<std::int64_t>& shape,const tb::Device& device) {
    return std::make_shared<TqRowStage>(tables,rows_per_shard,shape,device);
}

bool is_tq_row_table(const mfq::ModelSource& source,const std::string& name) {
    if(const auto* record=source.find_tensor(name);record && record->dtype=="TQ")return true;
    for(const char* key:{"tq_row_shards","tq.segments"}) {
        const auto value=source.metadata().find(key);
        if(value!=source.metadata().end()) {const auto table=nlohmann::json::parse(value->second);if(table.is_object() && table.contains(name))return true;}
    }
    return false;
}
tb::Tensor TqWeight::embedding(tb::Tensor ids)const {
    MfqCudaGuard guard(image.device());auto index=ids.to(image.device()).to(tb::kInt64).contiguous();
    auto shape=ids.sizes().vec();shape.push_back(neuron_len);
    auto result=tb::empty(shape,image.options().dtype(tb::kFloat16));
    if(index.numel()>INT_MAX)throw std::overflow_error("TQ embedding row count exceeds limits");
    if(index.numel())status(mfq_tq_embedding_hip(view,index.data_ptr<int64_t>(),int(index.numel()),result.data_ptr(),0,mfq_current_cuda_stream()));
    return result;
}

std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(const mfq::ModelSource& source,
        const std::string& name,int device,CudaExecutionContext* execution) {
    const auto shape=logical_shape(source,name);
    if(shape.size()!=3)throw std::invalid_argument("TQ expert bank must be [experts,out,in]: "+name);
    if(execution && moe_parallel_config(*execution).enabled()) {
        auto result=std::make_shared<TqRoutedWeight>();result->experts=shape[0];result->out=shape[1];result->input=shape[2];
        result->execution=execution;
        for(const auto& slice:plan_moe_expert_parallel_slices(moe_parallel_config(*execution),shape[0],name))
            result->shards.push_back({slice.device,int(slice.begin),int(slice.end),
                std::make_shared<TqRoutedWeight>(source,name,slice.device,shape[0],shape[1],shape[2],slice.begin,slice.end)});
        return result;
    }
    return std::make_shared<TqRoutedWeight>(source,name,device,shape[0],shape[1],shape[2]);
}
