#include "packed_bench_utils.h"
#include "mfq_cuda_graph_capture.h"
#include "mfq_cuda_quant_ops.h"

#include <cuda_runtime_api.h>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace {
using namespace mfq::cuda;
void select(bool enabled) {
    const char* key="MFQ_NINT_SINGLE_ROW";
#ifdef _WIN32
    _putenv_s(key,enabled ? "1" : "0");
#else
    setenv(key,enabled ? "1" : "0",1);
#endif
}
void exact(const Tensor& output,const Tensor& reference) {
    auto actual=output.to(kCPU).contiguous();
    if(actual.sizes()!=reference.sizes() || actual.scalar_type()!=kFloat16 ||
            std::memcmp(actual.data_ptr(),reference.data_ptr(),std::size_t(actual.numel())*2))
        throw std::runtime_error("CUDA NINT benchmark original Half bits mismatch");
}
struct Weight {
    Tensor q,bits,offsets,scale,minimum,outer,outer_min,x,qx,xscale,reference;
    int gs;
    std::size_t bytes;
    Weight(int rows,int width,int qbits,int group_size,std::uint32_t seed):gs(group_size) {
        const Device gpu{DeviceType::cuda,0};
        const int groups=(width+gs-1)/gs,k=groups*gs;
        const auto packed=(std::uint64_t(rows)*k*qbits+7)/8;
        std::vector<std::uint8_t> payload(packed+8),s(rows*groups),mn(rows*groups);
        mfq::bench::fill(payload,seed);mfq::bench::fill(s,seed);mfq::bench::fill(mn,seed);
        for(auto& value:s)value&=std::uint8_t(qbits==4 ? 63 : 127);
        for(auto& value:mn)value&=std::uint8_t(qbits==4 ? 63 : 127);
        std::vector<std::int64_t> off(rows);
        std::vector<float> ns(rows),nm(rows),input(width);
        for(int row=0;row<rows;++row) {
            off[row]=std::int64_t(row)*k*qbits;
            ns[row]=.000123f*float(row%13-6);nm[row]=.004731f*float(row%5-2);
        }
        for(int i=0;i<width;++i)input[i]=std::sin(float(i+7)*.193f)*.31f;
        q=tensor(payload).to(gpu);bits=tensor(std::vector<std::uint8_t>(rows,qbits)).to(gpu);
        offsets=tensor(off).to(gpu);scale=tensor(s).reshape({rows,groups}).to(gpu);
        minimum=tensor(mn).reshape({rows,groups}).to(gpu);
        outer=tensor(ns).to(gpu);outer_min=tensor(nm).to(gpu);
        x=tensor(input).reshape({1,width}).to(gpu,kFloat16);
        const auto options=TensorOptions().device(gpu);
        qx=empty({1,k},options.dtype(kInt8));xscale=empty({1,groups},options.dtype(kFloat32));
        bytes=payload.size()+off.size()*8+std::size_t(rows)+s.size()+mn.size()+(ns.size()+nm.size())*4;
    }
    Tensor run() {
        return nint_matmul_ws_cuda(q,bits,offsets,scale,minimum,outer,outer_min,x,gs,qx,xscale);
    }
};
class Replay {
    StreamHandle stream_=mfq_get_stream_from_pool(false);
    MfqCudaGraph operator_graph_;
    std::vector<Tensor> outputs_;
    cudaGraph_t kernel_graph_=nullptr;
    cudaGraphExec_t kernel_exec_=nullptr;
public:
    std::size_t calls=0,total_kernel_nodes=0;
    int registers=0;
    Replay(std::vector<std::unique_ptr<Weight>>& weights,std::size_t count,bool enabled) {
        MfqCudaStreamGuard guard(stream_);select(enabled);
        calls=count*16;
        mfq_prepare_cuda_graph_memory(operator_graph_);
        outputs_.reserve(calls);
        for(std::size_t i=0;i<calls;++i)outputs_.push_back(weights[i%count]->run());
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));outputs_.clear();
        operator_graph_.capture_begin();
        for(std::size_t i=0;i<calls;++i)outputs_.push_back(weights[i%count]->run());
        cudaGraph_t captured=nullptr;cudaStreamCaptureStatus status;
        MFQ_NATIVE_CUDA_CHECK(mfq_stream_capture_info(stream_.stream(),&status,nullptr,&captured));
        if(status!=cudaStreamCaptureStatusActive || !captured)throw std::runtime_error("benchmark capture unavailable");
        operator_graph_.capture_end();operator_graph_.replay();
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));
        for(std::size_t i=0;i<count;++i)exact(outputs_[i],weights[i]->reference);
        std::size_t node_count=0;
        MFQ_NATIVE_CUDA_CHECK(cudaGraphGetNodes(captured,nullptr,&node_count));
        std::vector<cudaGraphNode_t> nodes(node_count);
        MFQ_NATIVE_CUDA_CHECK(cudaGraphGetNodes(captured,nodes.data(),&node_count));
        std::vector<std::pair<std::size_t,cudaKernelNodeParams>> math;
        for(auto node:nodes) {
            cudaGraphNodeType type;MFQ_NATIVE_CUDA_CHECK(cudaGraphNodeGetType(node,&type));
            if(type!=cudaGraphNodeTypeKernel)continue;
            ++total_kernel_nodes;cudaKernelNodeParams params{};
            MFQ_NATIVE_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&params));
            const char* name=nullptr;MFQ_NATIVE_CUDA_CHECK(cudaFuncGetName(&name,params.func));
            if(!name || !std::strstr(name,"nint_matmul_kernel"))continue;
            if(!std::strstr(name,enabled ? "ILb1E" : "ILb0E"))
                throw std::runtime_error("benchmark selected wrong NINT template");
            cudaFuncAttributes attributes{};
            MFQ_NATIVE_CUDA_CHECK(cudaFuncGetAttributes(&attributes,params.func));registers=attributes.numRegs;
            void* output=*static_cast<void**>(params.kernelParams[9]);
            std::size_t index=0;
            while(index<calls && outputs_[index].data_ptr()!=output)++index;
            if(index==calls)throw std::runtime_error("benchmark NINT output ownership mismatch");
            math.emplace_back(index,params);
        }
        if(math.size()!=calls)throw std::runtime_error("benchmark NINT kernel coverage incomplete");
        std::sort(math.begin(),math.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        MFQ_NATIVE_CUDA_CHECK(cudaGraphCreate(&kernel_graph_,0));
        cudaGraphNode_t previous=nullptr;
        for(std::size_t i=0;i<calls;++i) {
            if(math[i].first!=i)throw std::runtime_error("benchmark NINT node order invalid");
            cudaGraphNode_t node;
            MFQ_NATIVE_CUDA_CHECK(cudaGraphAddKernelNode(&node,kernel_graph_,previous ? &previous : nullptr,previous ? 1 : 0,&math[i].second));
            previous=node;
        }
        MFQ_NATIVE_CUDA_CHECK(cudaGraphInstantiate(&kernel_exec_,kernel_graph_,0));
        MFQ_NATIVE_CUDA_CHECK(cudaGraphUpload(kernel_exec_,stream_.stream()));
        MFQ_NATIVE_CUDA_CHECK(cudaGraphLaunch(kernel_exec_,stream_.stream()));
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));
        for(std::size_t i=0;i<count;++i)exact(outputs_[i],weights[i]->reference);
    }
    ~Replay() {
        if(kernel_exec_)cudaGraphExecDestroy(kernel_exec_);
        if(kernel_graph_)cudaGraphDestroy(kernel_graph_);
    }
    Replay(const Replay&)=delete;
    Replay& operator=(const Replay&)=delete;
    double measure(bool kernel_only,std::size_t repeats) {
        cudaEvent_t start=nullptr,end=nullptr;
        MFQ_NATIVE_CUDA_CHECK(cudaEventCreate(&start));MFQ_NATIVE_CUDA_CHECK(cudaEventCreate(&end));
        MFQ_NATIVE_CUDA_CHECK(cudaEventRecord(start,stream_.stream()));
        for(std::size_t i=0;i<repeats;++i) {
            if(kernel_only)MFQ_NATIVE_CUDA_CHECK(cudaGraphLaunch(kernel_exec_,stream_.stream()));
            else operator_graph_.replay();
        }
        MFQ_NATIVE_CUDA_CHECK(cudaEventRecord(end,stream_.stream()));
        MFQ_NATIVE_CUDA_CHECK(cudaEventSynchronize(end));float milliseconds=0;
        MFQ_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&milliseconds,start,end));
        MFQ_NATIVE_CUDA_CHECK(cudaEventDestroy(start));MFQ_NATIVE_CUDA_CHECK(cudaEventDestroy(end));
        return milliseconds;
    }
};
void run_case(int rows,int width,int bits,int gs,const cudaDeviceProp& properties,const mfq::bench::Settings& settings) {
    std::vector<std::unique_ptr<Weight>> weights;
    weights.emplace_back(std::make_unique<Weight>(rows,width,bits,gs,20261006u));
    const auto target=std::max<std::size_t>(1,std::size_t(properties.l2CacheSize)*4);
    const auto copies=std::max<std::size_t>(2,(target+weights[0]->bytes-1)/weights[0]->bytes);
    for(std::size_t i=1;i<copies;++i)weights.emplace_back(std::make_unique<Weight>(rows,width,bits,gs,20261006u+std::uint32_t(i)*97));
    for(auto& weight:weights) {
        select(false);weight->reference=weight->run().to(kCPU).contiguous();
        select(true);exact(weight->run(),weight->reference);
    }
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceSynchronize());
    for(const bool rotating:{false,true}) {
        const auto count=rotating ? weights.size() : std::size_t(1);
        Replay off(weights,count,false),on(weights,count,true);
        for(const bool kernel_only:{true,false}) {
            off.measure(kernel_only,1);on.measure(kernel_only,1);
            std::size_t repeats=1;
            while(std::min(off.measure(kernel_only,repeats),on.measure(kernel_only,repeats))<settings.target_ms) {
                if(repeats>std::numeric_limits<std::size_t>::max()/2)throw std::overflow_error("CUDA benchmark repeats overflow");
                repeats*=2;
            }
            std::vector<double> a,b;
            for(int sample=0;sample<settings.samples;++sample) {
                double x,y;
                if(sample%2) {y=on.measure(kernel_only,repeats);x=off.measure(kernel_only,repeats);}
                else {x=off.measure(kernel_only,repeats);y=on.measure(kernel_only,repeats);}
                a.push_back(x*1000/(repeats*off.calls));b.push_back(y*1000/(repeats*on.calls));
            }
            std::cout<<"{\"backend\":\"cuda\",\"operator\":\"nint_gemv\",\"rows\":"<<rows<<",\"width\":"<<width
                     <<",\"M\":1,\"bits\":"<<bits<<",\"group_size\":"<<gs<<",\"timing_scope\":\""
                     <<(kernel_only ? "kernel_device" : "quantization_and_gemv_device")<<"\",\"cache\":\""
                     <<(rotating ? "rotating" : "hot")<<"\",\"working_sets\":"<<count<<",\"weight_bytes\":"<<weights[0]->bytes
                     <<",\"working_bytes\":"<<weights[0]->bytes*count<<",\"device_l2_bytes\":"<<properties.l2CacheSize
                     <<",\"calls_per_graph\":"<<off.calls<<",\"operator_kernel_nodes\":"<<off.total_kernel_nodes
                     <<",\"old_registers\":"<<off.registers<<",\"new_registers\":"<<on.registers
                     <<",\"repeats\":"<<repeats<<",\"samples\":"<<settings.samples<<",\"exact_original_bits\":true";
            mfq::bench::timing(a,b);std::cout<<"}\n"<<std::flush;
        }
    }
    select(false);
}
} // namespace
int main(int argc,char** argv) try {
    const auto settings=mfq::bench::settings(argc,argv);
    auto context=mfq::cuda::default_context(0);
    cudaDeviceProp properties{};MFQ_NATIVE_CUDA_CHECK(cudaGetDeviceProperties(&properties,0));
    std::cout<<std::setprecision(9);
    for(const auto profile:std::array<std::pair<int,int>,2>{{{4,24},{5,28}}})
        for(const auto shape:std::array<std::pair<int,int>,4>{{{640,2560},{2560,640},{6144,2560},{2560,6144}}})
            run_case(shape.first,shape.second,profile.first,profile.second,properties,settings);
    return 0;
}catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
