#include "weights.h"
#include "mfq_cuda_graph_capture.h"
#include "nint_row_metadata.h"
#include "mfq_cuda_moe_ops.h"
#include <set>

constexpr int routes=10;
static bool format_matches(const MixedMoePool& a,const MixedMoePool& b) {
    if(a.family!=b.family)return false;
    if(a.family==MixedMoeFamily::Nint)return a.nint.gs==b.nint.gs;
    return a.nvq.format==b.nvq.format && a.nvq.gs==b.nvq.gs &&
        a.nvq.sub_bits==b.nvq.sub_bits && a.nvq.sign_mode==b.nvq.sign_mode;
}
static std::vector<MixedMoePool> collect(const fb::Case& c,const std::vector<std::string>& paths) {
    std::vector<MixedMoePool> result;
    std::set<std::string> visited;
    MixedMoePool reference;uint64_t reference_bits=0;bool have_reference=false;
    auto scan=[&](const mfq::ModelSource& model,const std::string& path,const std::string& tensor) {
        if(!visited.insert(path+"|"+tensor).second || result.size()==routes)return;
        const auto* record=model.find_tensor(tensor);if(!record)return;
        auto reader=model.tensor_reader(tensor);
        auto store=std::make_shared<mfq::MfeQuantExpertStore>(size_t(record->nbytes),
            [reader](size_t offset,uint8_t* data,size_t size){reader(offset,reinterpret_cast<std::byte*>(data),size);});
        MoeQuantRangeSource source(store);
        if(!have_reference) {
            reference=source.read_expert(c.expert);reference_bits=store->expert_values_bits(c.expert);have_reference=true;
        }
        for(size_t pool=0;pool<store->pool_count() && result.size()<routes;++pool) {
            if(!format_matches(reference,source.metadata()->pools[pool]))continue;
            for(int id:store->pool_expert_ids(pool)) {
                if(reference.family==MixedMoeFamily::Nint && store->expert_values_bits(id)!=reference_bits)continue;
                auto expert=source.read_expert(id);
                if(!format_matches(reference,expert))throw std::runtime_error("cohort metadata mismatch");
                result.push_back(std::move(expert));
                std::cout<<"EXPERT "<<c.name()<<'\t'<<path<<'\t'<<tensor<<'\t'<<id<<'\n';
                if(result.size()==routes)break;
            }
        }
    };
    std::vector<std::string> ordered{c.model};
    for(const auto& path:paths)if(path!=c.model)ordered.push_back(path);
    for(const auto& path:ordered) {
        auto model=mfq::open_model_source(path);
        if(path==c.model)scan(*model,path,c.tensor);
        for(int layer=0;layer<48 && result.size()<routes;++layer) {
            for(const char* role:c.k==640 ? std::vector<const char*>{"down"} : std::vector<const char*>{"gate","up"}) {
                scan(*model,path,"model.block."+std::to_string(layer)+".mlp.experts."+role+".weight");
                if(result.size()==routes)break;
            }
        }
        if(result.size()==routes)break;
    }
    if(result.size()!=routes)throw std::runtime_error("fewer than ten distinct real experts for "+c.name());
    return result;
}
// A homogeneous cohort uses the same contiguous expert layout as the runtime loader.
// Keep different codebooks/strides in separate pools if weights span model layers.
static bool mergeable(MixedMoePool& a,MixedMoePool& b) {
    if(!format_matches(a,b))return false;
    const auto af=fields(a),bf=fields(b);
    for(size_t i=0;i<af.size();++i) {
        if(af[i]->sizes()!=bf[i]->sizes())return false;
        if(af[i]==&a.nvq.codebook && std::memcmp(af[i]->data_ptr(),bf[i]->data_ptr(),af[i]->numel()*af[i]->element_size()))return false;
    }
    return true;
}
static std::vector<MixedMoePool> coalesce(std::vector<MixedMoePool>& leaves,int n) {
    std::vector<MixedMoePool> result;std::set<int> used;
    for(int first=0;first<routes;++first)if(!used.count(first)) {
        std::vector<int> members;
        for(int i=first;i<routes;++i)if(!used.count(i) && mergeable(leaves[first],leaves[i])) {
            used.insert(i);members.push_back(i);
        }
        auto pool=leaves[first];auto destination=fields(pool);
        for(size_t field=0;field<destination.size();++field) {
            if(destination[field]==&pool.nvq.codebook)continue;
            std::vector<tb::Tensor> parts;
            for(int member:members)parts.push_back(*fields(leaves[member])[field]);
            *destination[field]=tb::cat(parts,0);
        }
        pool.local_experts=int(members.size());
        if(pool.family==MixedMoeFamily::Nint) {pool.nint.out=int64_t(n)*members.size();pool.nint.shape={pool.nint.out,pool.nint.neuron_len};}
        else {pool.nvq.out=int64_t(n)*members.size();pool.nvq.shape={pool.nvq.out,pool.nvq.neuron_len};}
        std::vector<int32_t> local(routes,-1);
        for(size_t i=0;i<members.size();++i)local[members[i]]=int32_t(i);
        pool.expert_local=tb::tensor(local);
        prepare_bench_execution_layout(pool);
        result.push_back(std::move(pool));
    }
    return result;
}
struct Bank {
    MixedMoeRuntime runtime;
    Bank(const std::vector<MixedMoePool>& cpu,const fb::Case& c,std::vector<tb::Tensor>& books) {
        CudaExecutionContext execution;runtime.n_experts=routes;runtime.out_per_expert=c.n;runtime.neuron_len=c.k;
        for(size_t i=0;i<cpu.size();++i) {
            auto pool=upload(cpu[i],books[i],false);pool.expert_local=cpu[i].expert_local.to(tb::kCUDA);
            const char* metadata=std::getenv("MFQ_BENCH_PREPARE_NINT_METADATA");
            if(pool.family==MixedMoeFamily::Nint && metadata && metadata[0]!='0') {
                const auto& w=cpu[i].nint;
                pool.nint.route_metadata=make_nint_row_metadata(w.row_q_bit_offsets,
                    w.neuron_scale,w.neuron_min,pool.nint.q_packed.device());
            }
            if(pool.family==MixedMoeFamily::Nvq)books[i]=pool.nvq.codebook;
            runtime.pools.push_back(std::move(pool));
        }
        initialize_mixed_nvq_dispatch(runtime,execution.config);
    }
};
struct Replay {
    MfqCudaGraph graph;std::vector<tb::Tensor> output;size_t calls;
    cudaGraph_t captured=nullptr;
    Replay(std::vector<std::unique_ptr<Bank>>& banks,const tb::Tensor& x,const MoeRoutePlan& route,CudaExecutionContext& execution)
        :calls(banks.size()*8) {
        auto run=[&](size_t i){return banks[i%banks.size()]->runtime.forward(execution.config,execution.kl_mmq,true,false,x,route);};
        mfq_prepare_cuda_graph_memory(graph);output.reserve(calls);
        for(size_t i=0;i<calls;++i)output.push_back(run(i));
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));output.clear();
        graph.capture_begin();for(size_t i=0;i<calls;++i)output.push_back(run(i));
        cudaStreamCaptureStatus status;
        fb::check(mfq_stream_capture_info(mfq_current_cuda_stream(),&status,nullptr,&captured));
        graph.capture_end();
        graph.replay();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
    }
};
static void grouped_paths(bool enabled) {
    for(const char* name:{"MFQ_NINT_GROUP_DOT","MFQ_NVQ_MOE_GROUP_DOT"}) {
#ifdef _WIN32
        _putenv_s(name,enabled?"1":"0");
#else
        setenv(name,enabled?"1":"0",1);
#endif
    }
}
// Exercise captured launches against the original routed reductions, including
// descriptor maps and route IDs changed after capture. Single-pool, per-pool,
// heterogeneous and active-plan dispatches use the same real weight cohort.
static void nvq_option(const char* name,const char* value) {
#ifdef _WIN32
    _putenv_s(name,value?value:"");
#else
    if(value)setenv(name,value,1);else unsetenv(name);
#endif
}
static void verify_routes(const fb::Case& c,std::vector<MixedMoePool>& leaves,bool compare_option=false,
        const char* option="MFQ_NVQ_BYTE_SIGNS",bool bitexact=true,
        const char* off_value="0",const char* on_value="1") {
    size_t checks=0,different_values=0;double worst=0,worst_absolute=0;
    for(bool split:{false,true}) {
        auto cpu=split?leaves:coalesce(leaves,c.n);
        if(split)for(int i=0;i<routes;++i) {
            std::vector<int32_t> map(routes,-1);map[i]=0;
            cpu[i].expert_local=tb::tensor(map);
        }
        std::vector<tb::Tensor> books(cpu.size());Bank bank(cpu,c,books);
        bank.runtime.partial_experts=true;
        for(int tokens:{1,3,8})for(bool routed:{false,true})for(int dispatch=0;dispatch<3;++dispatch) {
            CudaExecutionContext execution;
            execution.config.moe_nvq_active_decode=dispatch==2;
            auto x=tb::zeros(routed?std::vector<int64_t>{tokens,routes,c.k}:std::vector<int64_t>{tokens,c.k},
                tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
            auto ids=tb::zeros({tokens,routes},x.options().dtype(tb::kInt32));
            MoeRoutePlan route;route.ids=ids;route.n_experts=routes;
            auto run=[&](){return bank.runtime.forward(execution.config,execution.kl_mmq,true,dispatch==0,x,route);};
            grouped_paths(true);if(compare_option)nvq_option(option,on_value);
            MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
            auto result=run();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            result={};
            graph.capture_begin();result=run();graph.capture_end();
            for(int step=0;step<4;++step) {
                std::vector<float> values(x.numel());
                for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+step*31)*.071f)*(step==0?.13f:.37f);
                x.copy_(tb::tensor(values).reshape(x.sizes().vec()).to(tb::kCUDA,tb::kFloat16));
                std::vector<int32_t> selected(tokens*routes);
                for(int i=0;i<tokens*routes;++i) {
                    selected[i]=((step==1?i/2:i)*3+step)%routes;
                    if(step==1 && i%4==0)selected[i]=-1;
                    if(step==1 && i%4==1)selected[i]=routes;
                    if(step==3)selected[i]=routes;
                }
                ids.copy_(tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA));
                for(size_t i=0;i<cpu.size();++i) {
                    auto map=cpu[i].expert_local.clone();
                    if(step==2)for(int e=0;e<routes;e+=2)map.data_ptr<int32_t>()[e]=-1;
                    bank.runtime.pools[i].expert_local.copy_(map.to(tb::kCUDA));
                }
                grouped_paths(compare_option);if(compare_option)nvq_option(option,off_value);
                auto expected=run().to(tb::kCPU,tb::kFloat32);
                grouped_paths(true);if(compare_option)nvq_option(option,on_value);
                graph.replay();auto actual=result.to(tb::kCPU,tb::kFloat32);
                if(actual.sizes()!=expected.sizes())throw std::runtime_error("captured routed shape changed");
                if(compare_option && bitexact &&
                    std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                    throw std::runtime_error(std::string(option)+" changed captured routed output bits");
                double error=0,norm=0;
                for(int64_t i=0;i<actual.numel();++i) {
                    const double a=expected.data_ptr<float>()[i],b=actual.data_ptr<float>()[i];
                    if(!std::isfinite(b))throw std::runtime_error("nonfinite captured routed result");
                    error+=(a-b)*(a-b);norm+=a*a;
                    different_values+=a!=b;worst_absolute=std::max(worst_absolute,std::abs(a-b));
                    const int expert=selected[size_t(i)/c.n];
                    if((expert<0 || expert>=routes || (step==2 && expert%2==0)) && b!=0)
                        throw std::runtime_error("masked routed output is nonzero");
                }
                const double relative=std::sqrt(error/std::max(norm,1e-30));worst=std::max(worst,relative);
                if(relative>2e-4)throw std::runtime_error("captured routed/legacy relative error: "+std::to_string(relative));
                ++checks;
            }
        }
    }
    if(compare_option)nvq_option(option,nullptr);
    const char* label=!compare_option?"VERIFY_ROUTES ":
        std::strcmp(option,"MFQ_NINT_ROUTE_ROW_WORKSPACE")==0?"VERIFY_NINT_ROW_WORKSPACE_ROUTES ":
        std::strcmp(option,"MFQ_NINT_ROUTE_METADATA")==0?"VERIFY_NINT_METADATA_ROUTES ":
        std::strcmp(option,"MFQ_NVQ_E8_NARROW")==0?"VERIFY_E8_NARROW_ROUTES ":
        std::strcmp(option,"MFQ_NVQ_E8_STATIC_SIGNS")==0?"VERIFY_E8_STATIC_SIGNS_ROUTES ":
        std::strcmp(option,"MFQ_NVQ_E8_WORD_WINDOW")==0?"VERIFY_E8_WORD_WINDOW_ROUTES ":
        std::strcmp(option,"MFQ_NVQ_ROUTE_ROWS")==0?"VERIFY_ROWS2_ROUTES ":
        std::strcmp(option,"MFQ_NVQ_ROUTE_LANES8")==0?"VERIFY_LANES8_ROUTES ":
        std::strcmp(option,"MFQ_NVQ_ROUTE_WARPS8")==0?"VERIFY_WARPS8_ROUTES ":"VERIFY_BYTE_SIGNS_ROUTES ";
    std::cout<<label<<c.name()
        <<" checks="<<checks<<" max_relative_l2="<<worst;
    if(!bitexact)std::cout<<" different_values="<<different_values<<" max_absolute="<<worst_absolute;
    std::cout<<" PASS\n"<<std::flush;
}
int main(int argc,char** argv)try {
    if(argc<3 || argc>4)throw std::runtime_error("usage: mfq-cuda-format-moe-bench cases.tsv output-dir [case-filter]");
    const char* cooperative=std::getenv("MFQ_NVQ_COOPERATIVE_READS");
    if(std::getenv("MFQ_BENCH_VERIFY_NVQ_COOPERATIVE_READS") || (cooperative && cooperative[0]=='1'))
        throw std::runtime_error("Cooperative read prototype was rejected; archived under runs/windows-ram-20261009");
    const char* state_lut=std::getenv("MFQ_NVQ_STATE_LUT");
    if(std::getenv("MFQ_BENCH_VERIFY_NVQ_STATE_LUT") || (state_lut && state_lut[0]=='1'))
        throw std::runtime_error("Register table prototype was rejected; archived under runs/windows-ram-20261009");
    const char* expansion=std::getenv("MFQ_NVQ_ROUTE_EXPANSION");
    if(std::getenv("MFQ_BENCH_VERIFY_NVQ_EXPANSION") || (expansion && expansion[0]=='1'))
        throw std::runtime_error("8-value expansion prototype was rejected; archived under runs/windows-ram-20261009");
    const char* cursor=std::getenv("MFQ_NVQ_ROUTE_CURSOR");
    if(std::getenv("MFQ_BENCH_VERIFY_NVQ_CURSOR") || (cursor && cursor[0]=='1'))
        throw std::runtime_error("Packed cursor prototype was rejected; archived under runs/windows-ram-20261009");
    const fs::path output=argv[2];fs::create_directories(output);
    auto context=mfq::cuda::default_context(0);auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    CudaExecutionContext execution;cudaDeviceProp prop{};fb::check(cudaGetDeviceProperties(&prop,0));
    const auto cases=fb::read_cases(argv[1]);std::vector<std::string> paths;
    for(const auto& c:cases)if(std::find(paths.begin(),paths.end(),c.model)==paths.end())paths.push_back(c.model);
    for(const auto& c:cases) {
        if(argc==4 && c.name().find(argv[3])==std::string::npos)continue;
        auto leaves=collect(c,paths);
        if(std::getenv("MFQ_BENCH_VERIFY_NINT_ROW_WORKSPACE")) {
            if(leaves.front().family!=MixedMoeFamily::Nint)
                throw std::runtime_error("NINT row workspace verification received a different format");
            verify_routes(c,leaves,true,"MFQ_NINT_ROUTE_ROW_WORKSPACE");continue;
        }
        if(std::getenv("MFQ_BENCH_VERIFY_NINT_METADATA")) {
            const char* prepare=std::getenv("MFQ_BENCH_PREPARE_NINT_METADATA");
            if(!prepare || prepare[0]=='0')throw std::runtime_error("NINT metadata verification requires prepared row records");
            if(leaves.front().family!=MixedMoeFamily::Nint)
                throw std::runtime_error("NINT metadata verification received a different format");
            verify_routes(c,leaves,true,"MFQ_NINT_ROUTE_METADATA");continue;
        }
        if(std::getenv("MFQ_BENCH_VERIFY_E8_NARROW")){
            if(c.label.find("NVQ-E8-")!=0)throw std::runtime_error("E8 narrow verification received a different format");
            nvq_option("MFQ_NVQ_E8_STATIC_SIGNS","1");
            verify_routes(c,leaves,true,"MFQ_NVQ_E8_NARROW");continue;
        }
        if(std::getenv("MFQ_BENCH_VERIFY_E8_STATIC_SIGNS")){
            if(c.label.find("NVQ-E8-")!=0)throw std::runtime_error("E8 static signs verification received a different format");
            verify_routes(c,leaves,true,"MFQ_NVQ_E8_STATIC_SIGNS");continue;
        }
        if(std::getenv("MFQ_BENCH_VERIFY_E8_WORD_WINDOW")){
            if(c.label.find("NVQ-E8-")!=0)throw std::runtime_error("E8 word window verification received a different format");
            verify_routes(c,leaves,true,"MFQ_NVQ_E8_WORD_WINDOW");continue;
        }
        if(std::getenv("MFQ_BENCH_VERIFY_NVQ_ROWS2")){
            if(c.label.find("NVQ-E8-")!=0)throw std::runtime_error("E8 rows verification received a different format");
            nvq_option("MFQ_NVQ_ROUTE_LANES","16");
            verify_routes(c,leaves,true,"MFQ_NVQ_ROUTE_ROWS",true,"1","2");continue;
        }
        if(std::getenv("MFQ_BENCH_VERIFY_NVQ_LANES8")){
            verify_routes(c,leaves,true,"MFQ_NVQ_ROUTE_LANES8",false);continue;
        }
        if(std::getenv("MFQ_BENCH_VERIFY_NVQ_WARPS8")){
            verify_routes(c,leaves,true,"MFQ_NVQ_ROUTE_WARPS8");continue;
        }
        if(std::getenv("MFQ_BENCH_VERIFY_NVQ_BYTE_SIGNS")){verify_routes(c,leaves,true);continue;}
        if(std::getenv("MFQ_BENCH_VERIFY_ROUTES")){verify_routes(c,leaves);continue;}
        const bool export_tensors=std::getenv("MFQ_BENCH_NO_EXPORT")==nullptr;
        if(export_tensors) {
            std::vector<tb::Tensor> dense;
            for(const auto& leaf:leaves)dense.push_back(decoded(upload(leaf)).to(tb::kCPU));
            write_tensor(output/(c.name()+".moe.weight.f32"),tb::cat(dense,0));
        }
        auto cpu=coalesce(leaves,c.n);size_t bytes=0;for(auto& pool:cpu)bytes+=storage(pool);
        const size_t count=std::max<size_t>(2,(4*size_t(prop.l2CacheSize)+bytes-1)/bytes);
        std::vector<std::unique_ptr<Bank>> banks;std::vector<tb::Tensor> books(cpu.size());
        for(size_t i=0;i<count;++i)banks.push_back(std::make_unique<Bank>(cpu,c,books));
        bytes=0;
        for(auto& pool:banks.front()->runtime.pools)bytes+=storage(pool);
        for(size_t i=0;i<cpu.size();++i)if(cpu[i].family==MixedMoeFamily::Nvq)
            std::cout<<"POOL "<<c.name()<<" pool="<<i<<" source_format="<<cpu[i].nvq.format
                <<" kernel_format="<<banks.front()->runtime.pools[i].nvq.kernel_format<<'\n';
        auto x=tb::tensor(fb::input(c.k==640 ? routes : 1,c.k)).to(tb::kCUDA,tb::kFloat16);
        x=c.k==640 ? x.reshape({1,routes,c.k}) : x.reshape({1,c.k});
        if(export_tensors)write_tensor(output/(c.name()+".moe.x.f32"),x);
        std::vector<int32_t> selected;for(int i=0;i<routes;++i)selected.push_back(i);
        auto route=build_moe_route_plan(tb::tensor(selected).reshape({1,routes}).to(tb::kCUDA),routes);
        Replay replay(banks,x,route,execution);
        size_t metadata_workspace=0;
        for(const auto& entry:banks.front()->runtime.nint_input_plans)
            if(entry.second.row_metadata_workspace.defined())metadata_workspace+=entry.second.row_metadata_workspace.nbytes();
        if(metadata_workspace)std::cout<<"WORKSPACE "<<c.name()<<" row_metadata_bytes="<<metadata_workspace
            <<" all_banks_bytes="<<metadata_workspace*banks.size()<<'\n';
        if(export_tensors)write_tensor(output/(c.name()+".moe.y.f32"),replay.output.front());
        std::vector<tb::Tensor> reference;
        for(int i=0;i<routes;++i) {
            auto w=upload(leaves[i]);
            auto input=c.k==640 ? x.reshape({routes,c.k}).narrow(0,i,1).contiguous() : x;
            auto value=w.family==MixedMoeFamily::Nint ? nint_matmul(execution.profiler,w.nint,input) :
                nvq_matmul(execution.profiler,w.nvq,input);
            reference.push_back(value.to(tb::kCPU,tb::kFloat32));
        }
        auto expected=tb::cat(reference,0),actual=replay.output.front().to(tb::kCPU,tb::kFloat32);
        double error=0,norm=0;
        for(int64_t i=0;i<actual.numel();++i) {
            const double a=expected.data_ptr<float>()[i],b=actual.data_ptr<float>()[i];
            if(!std::isfinite(b))throw std::runtime_error("nonfinite routed output");
            error+=(a-b)*(a-b);norm+=a*a;
        }
        const double relative=std::sqrt(error/std::max(norm,1e-30));
        if(relative>2e-4)throw std::runtime_error("routed/cohort output differs from individual experts");
        std::cout<<"VERIFY "<<c.name()<<" routes=10 pools="<<cpu.size()<<" relative_l2="<<relative<<" PASS\n";
        auto samples=fb::measure([&](){replay.graph.replay();},stream.stream(),replay.calls);
        fb::report(c,"MFQ",c.label,1,c.k,bytes,count,prop.l2CacheSize,samples,routes);
        if(std::getenv("MFQ_BENCH_PROFILE_KERNELS")) {
            fb::check(cudaGraphDebugDotPrint(replay.captured,(output/(c.name()+".dot")).string().c_str(),cudaGraphDebugDotFlagsVerbose));
            fb::profile_kernels(replay.captured,stream.stream(),replay.calls,c.name());
        }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
