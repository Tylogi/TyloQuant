#include "timing.h"
#include "mfq_cuda_graph_capture.h"
#include "cuda_execution.h"
#include "gated_residual_fused.h"
#include "storage/weight_loader.h"
#include "mfq/model_source.h"
#include <array>
#include <climits>
#include <cstring>
#include <memory>

namespace tb=mfq_tensor_backend;
namespace fb=format_bench;
namespace {
bool dense_vector_selected=false;
void fixed_groups(bool enabled) {
#ifdef _WIN32
    _putenv_s("MFQ_GR_FIXED_GROUPS",enabled?"1":"0");
#else
    setenv("MFQ_GR_FIXED_GROUPS",enabled?"1":"0",1);
#endif
}
void up_threads(int threads) {
    const auto value=std::to_string(threads);
#ifdef _WIN32
    _putenv_s("MFQ_GR_UP_THREADS",value.c_str());
#else
    setenv("MFQ_GR_UP_THREADS",value.c_str(),1);
#endif
}
void projection_threads(int threads) {
    const auto value=std::to_string(threads);
#ifdef _WIN32
    _putenv_s("MFQ_GR_PROJECTION_THREADS",value.c_str());
#else
    setenv("MFQ_GR_PROJECTION_THREADS",value.c_str(),1);
#endif
}
void fixed_formats(bool enabled) {
#ifdef _WIN32
    _putenv_s("MFQ_GR_FIXED_FORMATS",enabled?"1":"0");
#else
    setenv("MFQ_GR_FIXED_FORMATS",enabled?"1":"0",1);
#endif
}
void async_normalized(bool enabled) {
#ifdef _WIN32
    _putenv_s("MFQ_GR_ASYNC_NORMALIZED",enabled?"1":"0");
#else
    setenv("MFQ_GR_ASYNC_NORMALIZED",enabled?"1":"0",1);
#endif
}
std::size_t storage(const NintWeight& w) {
    std::size_t bytes=0;
    for(const auto* field:{&w.q_packed,&w.row_q_bits,&w.row_q_bit_offsets,&w.sub_scale,
            &w.sub_min,&w.neuron_scale,&w.neuron_min,&w.q8_zero_scale})
        if(field->defined())bytes+=field->numel()*field->element_size();
    return bytes;
}
struct Bank {
    int layer;
    std::string role;
    QuantLinear down,up,inject;
    tb::Tensor norm,right,vector_right;
    std::size_t bytes;
    std::array<std::size_t,3> stage_bytes{};
    Bank(CudaExecutionContext& execution,const mfq::ModelSource& model,int index,const char* branch)
        :layer(index),role(branch) {
        const auto prefix="model.block."+std::to_string(index)+"."+branch+".mhc.";
        norm=load_dense_gpu(execution,model,prefix+"pre.norm.weight").to(tb::kFloat32).contiguous();
        down=load_quant_linear(execution,model,prefix+"pre.down.weight");
        up=load_quant_linear(execution,model,prefix+"pre.up.weight");
        inject=load_quant_linear(execution,model,prefix+"post.inject.weight");
        if(!down.is_nint() || !up.is_nint())throw std::runtime_error("GR benchmark requires raw NINT Down/Up");
        bytes=storage(down.nint)+storage(up.nint);
        if(inject.is_dense()) {
            if(inject.dense.scalar_type()!=tb::kBFloat16)throw std::runtime_error("GR dense injection must be BF16");
            right=inject.dense.transpose(-1,-2).to(tb::kFloat32);
            if(std::getenv("MFQ_GR_BENCH_DENSE_VECTOR"))vector_right=prepare_gr_dense_vector_right(right);
            bytes+=right.numel()*right.element_size();
        } else if(inject.is_nint())bytes+=storage(inject.nint);
        else throw std::runtime_error("unsupported GR injection format");
        stage_bytes={std::size_t(norm.numel())*norm.element_size(),
            bytes-storage(up.nint),storage(up.nint)};
        std::cout<<"BANK layer="<<layer<<" role="<<role<<" bytes="<<bytes
            <<" down_group="<<down.nint.gs<<" up_group="<<up.nint.gs
            <<" down_aligned="<<down.nint.aligned_q8<<" up_aligned="<<up.nint.aligned_q8<<'\n';
    }
    std::vector<tb::Tensor> run(const tb::Tensor& branch,const tb::Tensor& residual,
            const tb::Tensor& prior,bool after) const {
        return gated_residual_two_stage_cuda(branch,residual,prior,norm,down.nint,up.nint,
            inject.is_nint()?&inject.nint:nullptr,4,1e-6,after,
            inject.is_dense()?inject.dense:tb::Tensor{},dense_vector_selected && vector_right.defined()?vector_right:right);
    }
};
void exact(const tb::Tensor& actual,const tb::Tensor& expected,const std::string& label) {
    auto a=actual.cpu().contiguous();
    if(a.sizes()!=expected.sizes() || a.scalar_type()!=expected.scalar_type())
        throw std::runtime_error(label+" shape/dtype differs");
    if(std::memcmp(a.data_ptr(),expected.data_ptr(),a.numel()*a.element_size()))
        throw std::runtime_error(label+" output bytes differ");
    auto values=a.to(tb::kFloat32);
    for(int64_t i=0;i<values.numel();++i)if(!std::isfinite(values.data_ptr<float>()[i]))
        throw std::runtime_error(label+" nonfinite output");
}
void profile_stages(cudaGraph_t captured,cudaStream_t stream,std::size_t calls,
        const std::array<std::size_t,3>& bytes,int l2,int phase) {
    // This GR graph is a stream-ordered chain of norm/projection/Up kernels.
    // Its ordinary replay has already populated all intermediate inputs.
    std::size_t roots=0;fb::check(cudaGraphGetRootNodes(captured,nullptr,&roots));
    if(roots!=1)throw std::runtime_error("GR stage profiling requires one graph root");
    cudaGraphNode_t node=nullptr;fb::check(cudaGraphGetRootNodes(captured,&node,&roots));
    std::array<std::vector<cudaKernelNodeParams>,3> stages;
    std::size_t index=0;
    while(node) {
        cudaGraphNodeType type;fb::check(cudaGraphNodeGetType(node,&type));
        if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("GR stage graph contains a non-kernel node");
        cudaKernelNodeParams p{};fb::check(cudaGraphKernelNodeGetParams(node,&p));
        stages[index++%3].push_back(p);
        std::size_t children=0;fb::check(mfq_graph_dependent_nodes(node,nullptr,&children));
        if(children>1)throw std::runtime_error("GR stage graph contains a branch");
        if(children)fb::check(mfq_graph_dependent_nodes(node,&node,&children));
        else node=nullptr;
    }
    if(index!=3*calls)throw std::runtime_error("GR stage profiling requires three kernels per bank");
    const char* names[]={"norm","down_projection","up_projection_and_mix"};
    for(int stage=0;stage<3;++stage) {
        cudaGraph_t graph=nullptr;cudaGraphExec_t replay=nullptr;cudaGraphNode_t previous=nullptr;
        fb::check(cudaGraphCreate(&graph,0));
        int min_regs=INT_MAX,max_regs=0;std::size_t max_local=0;
        for(const auto& p:stages[stage]) {
            cudaGraphNode_t current=nullptr;
            fb::check(cudaGraphAddKernelNode(&current,graph,previous?&previous:nullptr,previous?1:0,&p));
            previous=current;
            cudaFuncAttributes attr{};fb::check(cudaFuncGetAttributes(&attr,p.func));
            min_regs=std::min(min_regs,attr.numRegs);max_regs=std::max(max_regs,attr.numRegs);
            max_local=std::max(max_local,attr.localSizeBytes);
        }
        fb::check(cudaGraphInstantiate(&replay,graph,0));fb::check(cudaGraphUpload(replay,stream));
        const auto samples=fb::measure([&]{fb::check(cudaGraphLaunch(replay,stream));},stream,calls);
        std::cout<<"{\"operator\":\"gated_residual_stage\",\"stage\":\""<<names[stage]
            <<"\",\"phase\":"<<phase<<",\"calls\":"<<calls<<",\"isolated\":true,\"us\":"<<fb::median(samples)
            <<",\"working_stored_weight_bytes\":"<<bytes[stage]<<",\"l2_bytes\":"<<l2
            <<",\"min_registers\":"<<min_regs<<",\"max_registers\":"<<max_regs
            <<",\"max_local_bytes\":"<<max_local
            <<",\"grid_x\":"<<stages[stage].front().gridDim.x
            <<",\"block_x\":"<<stages[stage].front().blockDim.x
            <<",\"dynamic_shared_bytes\":"<<stages[stage].front().sharedMemBytes<<"}\n"<<std::flush;
        fb::check(cudaGraphExecDestroy(replay));fb::check(cudaGraphDestroy(graph));
    }
}
}
int main(int argc,char** argv) try {
    if(argc!=2)throw std::runtime_error("usage: mfq-cuda-real-gr-bench MODEL.mfq");
    const auto model=mfq::open_model_source(argv[1]);
    CudaExecutionContext execution;
    std::vector<std::unique_ptr<Bank>> banks;
    std::size_t bytes=0;
    std::array<std::size_t,3> stage_bytes{};
    for(int layer:{0,8,16,24,32,40})for(const auto* role:{"attention","mlp"}) {
        const auto prefix="model.block."+std::to_string(layer)+"."+role+".mhc.pre.";
        const auto* down=model->find_tensor(prefix+"down.weight");
        const auto* up=model->find_tensor(prefix+"up.weight");
        if(!down || !up)throw std::runtime_error("missing GR benchmark tensor");
        if(down->dtype!="NINT" || up->dtype!="NINT") {
            std::cout<<"SKIP layer="<<layer<<" role="<<role<<" down="<<down->dtype
                <<" up="<<up->dtype<<" fused_raw_nint_unsupported\n";
            continue;
        }
        banks.push_back(std::make_unique<Bank>(execution,*model,layer,role));
        bytes+=banks.back()->bytes;
        for(int stage=0;stage<3;++stage)stage_bytes[stage]+=banks.back()->stage_bytes[stage];
    }
    int l2=0;
    fb::check(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,mfq_current_cuda_device()));
    if(bytes<4*std::size_t(l2))throw std::runtime_error("GR working weights must exceed four times L2");
    constexpr int width=10240,hidden=2560;
    const char* thread_mode=std::getenv("MFQ_GR_BENCH_UP_THREADS");
    const int thread_target=thread_mode?std::atoi(thread_mode):0;
    if(thread_target!=0 && thread_target!=128 && thread_target!=512)
        throw std::runtime_error("MFQ_GR_BENCH_UP_THREADS must be 0, 128 or 512");
    const char* combined_mode=std::getenv("MFQ_GR_BENCH_FIXED");
    const bool combined_fixed=combined_mode && combined_mode[0]=='1';
    const bool stage_mode=std::getenv("MFQ_GR_BENCH_STAGES")!=nullptr;
    const char* projection_mode=std::getenv("MFQ_GR_BENCH_PROJECTION_THREADS");
    const int projection_target=projection_mode?std::atoi(projection_mode):0;
    if(projection_target!=0 && projection_target!=128 && projection_target!=512)
        throw std::runtime_error("MFQ_GR_BENCH_PROJECTION_THREADS must be 0, 128 or 512");
    const bool format_mode=std::getenv("MFQ_GR_BENCH_FIXED_FORMATS")!=nullptr;
    const bool async_mode=std::getenv("MFQ_GR_BENCH_ASYNC_NORMALIZED")!=nullptr;
    const bool vector_mode=std::getenv("MFQ_GR_BENCH_DENSE_VECTOR")!=nullptr;
    const auto source=tb::tensor(fb::input(1,width)).reshape({1,1,width}).to(tb::kCUDA);
    const auto branch_source=tb::tensor(fb::input(1,hidden)).reshape({1,1,hidden}).to(tb::kCUDA,tb::kFloat16);
    auto prior=tb::tensor(std::vector<float>{.13f,.37f,.61f,.83f}).reshape({1,1,4}).to(tb::kCUDA);
    std::size_t exact_checks=0;
    for(bool half:{true,false})for(bool after:{false,true}) {
        auto residual=source.to(half?tb::kFloat16:tb::kFloat32).clone();
        auto branch=branch_source.clone();
        auto gates=half?prior.to(tb::kFloat16):prior;
        std::array<std::vector<std::vector<tb::Tensor>>,3> expected;
        for(int phase=0;phase<3;++phase) {
            const bool fixed=stage_mode || vector_mode || projection_target || format_mode || async_mode || (phase==1 && (!thread_target || combined_fixed));
            const int threads=stage_mode || vector_mode || projection_target || format_mode || async_mode?512:phase==1 && thread_target?thread_target:256;
            const int projection=projection_target && phase==1?projection_target:256;
            fixed_groups(fixed);up_threads(threads);
            dense_vector_selected=vector_mode && phase==1;
            if(projection_target)projection_threads(projection);
            if(format_mode)fixed_formats(phase==1);
            if(async_mode)async_normalized(phase==1);
            const char* current_formats=std::getenv("MFQ_GR_FIXED_FORMATS");
            const char* current_async=std::getenv("MFQ_GR_ASYNC_NORMALIZED");
            residual.copy_(source.to(residual.scalar_type()));branch.copy_(branch_source);
            MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
            std::vector<std::vector<tb::Tensor>> outputs;
            const auto body=[&] {
                outputs.clear();
                for(const auto& bank:banks)outputs.push_back(bank->run(branch,residual,gates,after));
            };
            body();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));outputs.clear();
            graph.capture_begin();body();
            cudaGraph_t captured=nullptr;
            if(stage_mode) {
                cudaStreamCaptureStatus capture_status;
                fb::check(mfq_stream_capture_info(mfq_current_cuda_stream(),&capture_status,nullptr,&captured));
            }
            graph.capture_end();
            for(int step=0;step<3;++step) {
                residual.copy_((source*(.25+.625*step)).to(residual.scalar_type()));
                branch.copy_((branch_source*(1.+.25*step)).to(tb::kFloat16));
                graph.replay();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
                if(phase==0) {
                    for(const auto& values:outputs) {
                        std::vector<tb::Tensor> saved;
                        for(const auto& value:values)saved.push_back(value.cpu().contiguous());
                        expected[step].push_back(std::move(saved));
                    }
                } else for(std::size_t i=0;i<outputs.size();++i)for(std::size_t j=0;j<outputs[i].size();++j) {
                    exact(outputs[i][j],expected[step][i][j],"GR fixed group graph");++exact_checks;
                }
            }
            residual.copy_(source.to(residual.scalar_type()));branch.copy_(branch_source);
            const auto samples=fb::measure([&]{graph.replay();},mfq_current_cuda_stream(),banks.size());
            const auto us=fb::median(samples);
            std::cout<<std::setprecision(10)<<"{\"operator\":\"gated_residual\",\"phase\":"<<phase
                <<",\"fixed_groups\":"<<fixed<<",\"up_threads\":"<<threads
                <<",\"dense_vector\":"<<dense_vector_selected
                <<",\"projection_threads\":"<<projection
                <<",\"fixed_formats\":"<<(!current_formats || current_formats[0]=='1')
                <<",\"async_normalized_requested\":"<<(!current_async || current_async[0]=='1')
                <<",\"async_normalized\":"<<(!half && (!current_async || current_async[0]=='1'))
                <<",\"half\":"<<half<<",\"after\":"<<after
                <<",\"calls\":"<<banks.size()<<",\"us\":"<<us<<",\"effective_GBs\":"
                <<double(bytes)/banks.size()/(us*1000.)<<",\"working_bytes\":"<<bytes<<",\"l2_bytes\":"<<l2
                <<",\"graph_nodes\":"<<graph.nodes()<<",\"expanded_weights\":false,\"inputs\":\"synthetic\",\"samples_us\":[";
            for(std::size_t i=0;i<samples.size();++i)std::cout<<(i?",":"")<<samples[i];
            std::cout<<"]}\n"<<std::flush;
            if(stage_mode && (phase==0 || ((vector_mode || projection_target || format_mode || async_mode) && phase==1)) && half && !after)
                profile_stages(captured,mfq_current_cuda_stream(),banks.size(),stage_bytes,l2,phase);
        }
    }
    std::cout<<"fixed_group_graph_output_checks="<<exact_checks<<" exact PASS\n";
    return 0;
} catch(const std::exception& error) {
    std::cerr<<"GR benchmark: "<<error.what()<<'\n';return 1;
}
