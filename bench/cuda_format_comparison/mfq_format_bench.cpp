#include "weights.h"
#include "mfq_cuda_graph_capture.h"

static void select_reference(bool enabled) {
    for(const char* key:{"MFQ_NINT_DENSE_REFERENCE","MFQ_NVQ_DENSE_REFERENCE"}) {
#ifdef _WIN32
        _putenv_s(key,enabled ? "1" : "");
#else
        if(enabled)setenv(key,"1",1);else unsetenv(key);
#endif
    }
}
static void verify(const fb::Case& c,MixedMoePool& pool,CudaExecutionContext& execution,
        const MixedMoePool& source) {
    const auto run=[&](const tb::Tensor& x) {
        return pool.family==MixedMoeFamily::Nint ? nint_matmul(execution.profiler,pool.nint,x) :
            nvq_matmul(execution.profiler,pool.nvq,x);
    };
    double worst=0;int cases=0;
    for(int m:{1,2,3,4,5,7,8})for(int seed=0;seed<4;++seed) {
        auto values=fb::input(m,c.k);
        for(std::size_t i=0;i<values.size();++i)
            values[i]=seed==0 ? 0.f : values[i]*float(seed*seed)*std::cos(float(i+seed*29)*.017f);
        auto x=tb::tensor(values).reshape({m,c.k}).to(tb::kCUDA,tb::kFloat16);
        select_reference(true);auto a=run(x).to(tb::kCPU,tb::kFloat32);
        select_reference(false);auto b=run(x).to(tb::kCPU,tb::kFloat32);
        double sum=0,error=0,largest=0,max_error=0;
        for(int64_t i=0;i<a.numel();++i) {
            const double av=a.data_ptr<float>()[i],bv=b.data_ptr<float>()[i];
            if(!std::isfinite(bv))throw std::runtime_error("nonfinite verification output");
            sum+=av*av;error+=(av-bv)*(av-bv);
            largest=std::max(largest,std::abs(av));max_error=std::max(max_error,std::abs(av-bv));
        }
        const double relative=std::sqrt(error/std::max(sum,1e-30));
        if(relative>2e-4 || max_error>largest*2e-3+6e-8)
            throw std::runtime_error("reference mismatch: "+c.name()+" M="+std::to_string(m));
        worst=std::max(worst,relative);++cases;
    }
    std::cout<<"VERIFY "<<c.name()<<" cases="<<cases<<" max_relative_l2="<<worst<<" PASS\n";
    if(pool.family==MixedMoeFamily::Nvq && pool.nvq.decode_records.defined()) {
        const char* setting=std::getenv("MFQ_NVQ1_GROUP_RECORDS");
        const std::string saved=setting?setting:"";
#ifdef _WIN32
        _putenv_s("MFQ_NVQ1_GROUP_RECORDS","0");
#else
        setenv("MFQ_NVQ1_GROUP_RECORDS","0",1);
#endif
        auto raw=upload(source,pool.nvq.codebook);
#ifdef _WIN32
        _putenv_s("MFQ_NVQ1_GROUP_RECORDS",saved.c_str());
#else
        if(saved.empty())unsetenv("MFQ_NVQ1_GROUP_RECORDS");
        else setenv("MFQ_NVQ1_GROUP_RECORDS",saved.c_str(),1);
#endif
        int checks=0;
        for(int m:{1,3,8,16,64})for(int seed=0;seed<4;++seed) {
            auto values=fb::input(m,c.k);
            for(std::size_t i=0;i<values.size();++i)
                values[i]=seed==0?0.f:values[i]*float(seed*seed)*std::cos(float(i+seed*29)*.017f);
            auto x=tb::tensor(values).reshape({m,c.k}).to(tb::kCUDA,tb::kFloat16);
            auto a=nvq_matmul(execution.profiler,raw.nvq,x).to(tb::kCPU);
            auto b=run(x).to(tb::kCPU);
            if(a.sizes()!=b.sizes() || std::memcmp(a.data_ptr(),b.data_ptr(),a.numel()*a.element_size()))
                throw std::runtime_error("NVQ1 records changed dense output bits: "+c.name()+" M="+std::to_string(m));
            ++checks;
        }
        std::cout<<"VERIFY_RECORD_DENSE "<<c.name()<<" checks="<<checks
            <<" raw/record bitexact, M=1/3/8/16/64 PASS\n";
    }
}
struct Replay {
    MfqCudaGraph graph;
    std::vector<tb::Tensor> outputs;
    std::size_t calls;
    Replay(std::vector<MixedMoePool>& banks,CudaExecutionContext& execution,const tb::Tensor& x):calls(banks.size()*8) {
        auto run=[&](MixedMoePool& pool) {
            return pool.family==MixedMoeFamily::Nint ? nint_matmul(execution.profiler,pool.nint,x) :
                nvq_matmul(execution.profiler,pool.nvq,x);
        };
        mfq_prepare_cuda_graph_memory(graph);
        outputs.reserve(calls);
        for(std::size_t i=0;i<calls;++i)outputs.push_back(run(banks[i%banks.size()]));
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));outputs.clear();
        graph.capture_begin();
        for(std::size_t i=0;i<calls;++i)outputs.push_back(run(banks[i%banks.size()]));
        if(std::getenv("MFQ_BENCH_GRAPH_DOT")) {
            cudaStreamCaptureStatus status;unsigned long long id;cudaGraph_t captured=nullptr;
            fb::check(mfq_stream_capture_info(mfq_current_cuda_stream(),&status,&id,&captured));
            if(std::getenv("MFQ_BENCH_GRAPH_DOT"))
                fb::check(cudaGraphDebugDotPrint(captured,std::getenv("MFQ_BENCH_GRAPH_DOT"),cudaGraphDebugDotFlagsKernelNodeParams));
        }
        graph.capture_end();graph.replay();
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
    }

};
int main(int argc,char** argv)try {
    if(argc<3 || argc>4)throw std::runtime_error("usage: mfq-cuda-format-bench cases.tsv output-dir [case-filter]");
    const fs::path output=argv[2];fs::create_directories(output);
    auto context=mfq::cuda::default_context(0);
    auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard stream_guard(stream);
    CudaExecutionContext execution;
    cudaDeviceProp properties{};fb::check(cudaGetDeviceProperties(&properties,0));
    std::map<std::string,std::shared_ptr<const mfq::ModelSource>> models;
    for(const auto& c:fb::read_cases(argv[1])) {
        if(argc==4 && c.name().find(argv[3])==std::string::npos)continue;
        auto& model=models[c.model];if(!model)model=mfq::open_model_source(c.model);
        const auto* record=model->find_tensor(c.tensor);
        if(!record)throw std::runtime_error("missing tensor: "+c.tensor);
        auto reader=model->tensor_reader(c.tensor);
        auto store=std::make_shared<mfq::MfeQuantExpertStore>(std::size_t(record->nbytes),
            [reader](std::size_t offset,uint8_t* data,std::size_t size){reader(offset,reinterpret_cast<std::byte*>(data),size);});
        MoeQuantRangeSource source(store);
        auto cpu=source.read_expert(c.expert);
        const auto source_bytes=storage(cpu);
        // Independent addresses exceed L2 by four times; all banks preserve the real packed weight.
        const auto count=std::getenv("MFQ_BENCH_VERIFY") ? 1 :
            std::max<std::size_t>(2,(4*std::size_t(properties.l2CacheSize)+source_bytes-1)/source_bytes);
        std::vector<MixedMoePool> banks;
        for(std::size_t i=0;i<count;++i)
            banks.push_back(upload(cpu,i ? banks.front().nvq.codebook : tb::Tensor{}));
        const auto bytes=storage(banks.front());
        if(std::getenv("MFQ_BENCH_VERIFY")) {
            verify(c,banks.front(),execution,cpu);
            continue;
        }
        if(!std::getenv("MFQ_BENCH_NO_EXPORT"))
            write_tensor(output/(c.name()+".weight.f32"),decoded(banks.front()));
        std::vector<int> batches{1,4,16,64};
        if(std::getenv("MFQ_BENCH_M"))batches={std::stoi(std::getenv("MFQ_BENCH_M"))};
        for(int m:batches) {
            auto x=tb::tensor(fb::input(m,c.k)).reshape({m,c.k}).to(tb::kCUDA,tb::kFloat16);
            if(!std::getenv("MFQ_BENCH_NO_EXPORT"))
                write_tensor(output/(c.name()+".x"+std::to_string(m)+".f32"),x);
            Replay replay(banks,execution,x);
            auto samples=fb::measure([&](){replay.graph.replay();},stream.stream(),replay.calls);
            const auto actual=replay.outputs.front().to(tb::kCPU).to(tb::kFloat32).contiguous();
            const auto last=replay.outputs.back().to(tb::kCPU).to(tb::kFloat32).contiguous();
            if(actual.numel()!=last.numel() || std::memcmp(actual.data_ptr(),last.data_ptr(),actual.numel()*sizeof(float)))
                throw std::runtime_error("independent weight banks produced different outputs");
            if(!std::getenv("MFQ_BENCH_NO_EXPORT"))
                write_tensor(output/(c.name()+".y"+std::to_string(m)+".f32"),actual);
            fb::report(c,"mfq",c.label,m,c.k,bytes,count,properties.l2CacheSize,samples);
        }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
