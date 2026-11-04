#include "storage/moe_ffn_pipeline.h"
#include "runtime/decode_window.h"
#include "runtime/execution_options.h"
#include "runtime/moe_pipeline.h"
#include "runtime/router_lookahead_audit.h"
#include "mfq_cuda_moe_ops.h"
#include "moe_cache_transfer.h"
#include "runtime/moe_residency.h"
#include "runtime/kv_expert_loan.h"
#include "storage/moe_quant_range_source.h"
#include "storage/moe_expert_cache.h"
#include "storage/moe_expert_cache_internal.h"
#include "storage/weight_loader.h"
#include "moe.h"
#include "mfe_ffn_runtime.h"
#include "core/rope.h"
#include "mfq/moe_dispatch_plan.h"
#include "mfq/runtime.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <regex>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>
#include <iomanip>
namespace tb=mfq_tensor_backend;
namespace {
std::size_t bytes(const MixedMoePool& p) {
    std::vector<tb::Tensor> fields=p.family==MixedMoeFamily::Nint ?
        std::vector<tb::Tensor>{p.nint.q_packed,p.nint.row_q_bits,p.nint.row_q_bit_offsets,p.nint.sub_scale,p.nint.sub_min,p.nint.neuron_scale,p.nint.neuron_min} :
        std::vector<tb::Tensor>{p.nvq.indices_packed,p.nvq.aux_packed,p.nvq.sub_scale_packed,p.nvq.neuron_scale};
    std::size_t n=0;for(const auto& f:fields)n+=f.numel()*f.element_size();return n;
}
std::shared_ptr<MoeQuantRangeSource> source(const std::filesystem::path& path) {
    auto model=mfq::open_model_source(path.string());auto read=model->tensor_reader("linear.weight");
    auto store=std::make_shared<mfq::MfeQuantExpertStore>(require_tensor(*model,"linear.weight").nbytes,
        [read](std::size_t off,uint8_t* dst,std::size_t n){read(off,reinterpret_cast<std::byte*>(dst),n);});
    return std::make_shared<MoeQuantRangeSource>(store);
}
QuantLinear linear(const MixedMoePool& pool,int output,int width) {
    QuantLinear l;l.logical_out=output;l.logical_neuron_len=width;
    if(pool.family==MixedMoeFamily::Nint){l.kind=QuantLinearKind::Nint;l.nint=pool.nint;}
    else {l.kind=QuantLinearKind::Nvq;l.nvq=pool.nvq;}return l;
}

void verify_nvq_bank_map(const std::filesystem::path& root) {
    int cases=0;
    for(const char* format:{"nvq2j","nvq2j-l","nvq2j-xl","nvq3j","nvq3j-512","nvq3j-l"})
    for(const char* shape:{"-640-2560.mfq","-2560-640.mfq"}) {
        auto value=source(root/(std::string(format)+shape));
        for(const auto& pool:value->metadata()->pools) {
            const auto& weight=pool.nvq;
            const auto metadata=weight.codebook.cpu().contiguous();
            const auto* bytes=metadata.data_ptr<int8_t>();
            for(unsigned state=0;state<16;++state)
                if(((weight.bank_map>>(2*state))&3u)!=uint8_t(bytes[36+state]))
                    throw std::runtime_error("cached JSC map differs from immutable codebook");
            ++cases;
        }
        // Expert slices reuse the shared immutable codebook and its map.
        const auto expert=value->read_expert(0);
        if(expert.nvq.bank_map!=value->metadata()->pools[0].nvq.bank_map)
            throw std::runtime_error("expert slice lost cached JSC map");
    }
    std::cout<<"nvq_bank_map_cases="<<cases<<" states=16 expert_slices=12 PASS\n";
}

void verify_nvq_dense_default(const std::filesystem::path& root) {
    struct Environment {
        const char* names[3]={"MFQ_MOE_NVQ_DENSE","MFQ_MOE_PRELOAD_ALL","MFQ_MOE_ASSERT_RESIDENT"};
        std::string previous[3];
        Environment(){for(int i=0;i<3;++i)if(const auto* value=std::getenv(names[i]))previous[i]=value;}
        void set(int i,const char* value)const {
#ifdef _WIN32
            _putenv_s(names[i],value);
#else
            if(*value)setenv(names[i],value,1);else unsetenv(names[i]);
#endif
        }
        ~Environment(){for(int i=0;i<3;++i)set(i,previous[i].c_str());}
    } env;
    int cases=0,guards=0,converted=0;
    for(const auto* format:{"nvq2j-xl","nvq3j-l"}) {
        env.set(0,"0");env.set(1,"0");env.set(2,"0");
        const auto path=root/(std::string(format)+"-640-2560.mfq");
        const auto canonical=source(path);
        const auto canonical_bytes=bytes(canonical->metadata()->pools[0]);
        for(const auto* option:{"","0","1"})for(int preload:{0,1})for(int sealed:{0,1}) {
            env.set(0,option);env.set(1,preload?"1":"0");env.set(2,sealed?"1":"0");
            const bool complete=preload && sealed;
            const bool reject=*option=='1' && !complete;
            std::shared_ptr<MoeQuantRangeSource> value;
            try{value=source(path);}catch(const std::invalid_argument&){if(!reject)throw;++guards;}
            if(reject) {
                if(value)throw std::runtime_error("compact default omitted sealed-preload guard");
            }else {
                const bool expected=complete && *option!='0';
                const auto& weight=value->metadata()->pools[0];
                if(weight.nvq.dense_groups!=expected || bytes(weight)!=canonical_bytes ||
                   bool(value->dense_materializations())!=expected)
                    throw std::runtime_error("compact default selected wrong layout or changed bytes");
                converted+=expected;
            }
            ++cases;
        }
    }
    if(cases!=24 || guards!=6 || converted!=4)throw std::runtime_error("compact default coverage incomplete");
    std::cout<<"nvq_dense_default_cases="<<cases<<" sealed_preload_guards="<<guards
        <<" compact_sources="<<converted<<" payload_bytes_unchanged=1 PASS\n";
}

int verify_shared_input(const std::filesystem::path& root) {
    CudaExecutionContext execution;
    auto load=[&](const std::array<std::string,4>& formats) {
        MixedMoeRuntime result;result.n_experts=12;result.out_per_expert=640;result.neuron_len=2560;
        result.partial_experts=true;
        for(int i=0;i<4;++i) {
            auto model=mfq::open_model_source((root/(formats[i]+"-640-2560.mfq")).string());
            auto single=make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),true,execution.config);
            for(auto& pool:single->pools) {
                std::vector<int32_t> local(12,-1);
                for(int expert=0;expert<3;++expert)local[i*3+expert]=expert;
                pool.expert_local=tb::tensor(local).to(tb::kCUDA);result.pools.push_back(std::move(pool));
            }
        }
        initialize_mixed_nvq_dispatch(result,execution.config);return result;
    };
    auto reference_gate=load({"nint4","nint5","nvq2j","nvq3j-512"});
    auto reference_up=load({"nint6","nint8","nvq1-l","nvq2j-l"});
    auto gate=reference_gate,up=reference_up;
    auto geometries=gate.activation_geometry();
    const auto gate_geometries=geometries;
    for(const auto& geometry:up.activation_geometry())
        if(std::find(geometries.begin(),geometries.end(),geometry)==geometries.end())geometries.push_back(geometry);
    if(geometries.size()<=gate_geometries.size())throw std::runtime_error("shared-input fixture lacks Up-only geometry");
    const auto same=[](const tb::Tensor& a,const tb::Tensor& b,const char* label) {
        auto x=a.to(tb::kCPU).contiguous(),y=b.to(tb::kCPU).contiguous();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type())
            throw std::runtime_error(std::string("shared-input ")+label+" shape or dtype mismatch");
        if(std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size())) {
            auto actual=x.to(tb::kFloat32),expected=y.to(tb::kFloat32);
            const auto* aa=static_cast<const uint8_t*>(x.data_ptr());
            const auto* bb=static_cast<const uint8_t*>(y.data_ptr());
            std::size_t byte=0;while(aa[byte]==bb[byte])++byte;
            const auto index=byte/x.element_size();
            std::ostringstream message;message<<"shared-input "<<label<<" is not byte exact"
                <<" index="<<index<<" actual="<<actual.data_ptr<float>()[index]
                <<" expected="<<expected.data_ptr<float>()[index];
            throw std::runtime_error(message.str());
        }
    };
    const auto forward=[&](MixedMoeRuntime& runtime,const tb::Tensor& x,const MoeRoutePlan& route,bool prepared) {
        return runtime.forward(execution.config,execution.kl_mmq,false,false,x,route,prepared);
    };
    const auto stream=mfq_current_cuda_stream();auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    context->begin_graph_pool(stream);int cases=0;
    for(int tokens:{1,3,8}) {
        auto x=tb::zeros({tokens,2560},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
        auto ids=tb::zeros({tokens,10},x.options().dtype(tb::kInt32));
        std::vector<int64_t> pointers;int groups=0;
        for(const auto& geometry:geometries) {
            auto workspace=gate.activation_workspace(x,tokens,geometry.groups,geometry.gs,{});
            up.activation_workspaces.insert_or_assign({tokens,geometry.groups,geometry.gs,x.get_device(),{}},workspace);
            pointers.insert(pointers.end(),{reinterpret_cast<int64_t>(workspace.qx.data_ptr()),
                reinterpret_cast<int64_t>(workspace.xscale.data_ptr()),geometry.groups,geometry.gs});
            groups+=geometry.groups;
        }
        auto descriptors=tb::tensor(pointers).reshape({int64_t(geometries.size()),4}).to(tb::kCUDA);
        tb::Tensor actual_gate,actual_up;mfq::cuda::DecodeWindow window(stream);
        window.capture([&] {
            auto route=build_moe_route_plan(ids,12);
            moe_quantize_shared_input_cuda(x,descriptors,groups);
            actual_gate=forward(gate,x,route,true);actual_up=forward(up,x,route,true);
        },[]{},[&]{actual_gate={};actual_up={};});
        for(int step=0;step<3;++step) {
            std::vector<float> input(tokens*2560);
            for(int t=0;t<tokens;++t)for(int k=0;k<2560;++k)
                input[t*2560+k]=k<64?0.0f:std::sin(float(k+t*17+step*13)*.173f)*.13f;
            std::vector<int32_t> selected(tokens*10);
            for(int i=0;i<tokens*10;++i)selected[i]=(i*5+step*3)%12;
            x.copy_(tb::tensor(input).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA));
            ids.copy_(tb::tensor(selected).reshape({tokens,10}).to(tb::kCUDA));
            auto route=build_moe_route_plan(ids,12);
            // Hold arithmetic dispatch fixed when checking byte-exact input
            // reuse. Heterogeneous-vs-pool arithmetic has its own tests.
            auto expected_gate=forward(reference_gate,x,route,false),expected_up=forward(reference_up,x,route,false);
            window.run();
            for(const auto& geometry:geometries) {
                const bool in_gate=std::find(gate_geometries.begin(),gate_geometries.end(),geometry)!=gate_geometries.end();
                auto expected=(in_gate?reference_gate:reference_up).activation_workspace(x,tokens,geometry.groups,geometry.gs,{});
                auto actual=gate.activation_workspace(x,tokens,geometry.groups,geometry.gs,{});
                same(actual.qx,expected.qx,"quantized bytes/tail padding");same(actual.xscale,expected.xscale,"group scales");
            }
            same(actual_gate,expected_gate,"mixed Gate output");same(actual_up,expected_up,"mixed Up output");
            ++cases;
        }
    }
    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));context->end_graph_pool(stream);return cases;
}

struct NintDispatchError {
    double value=0.0,grouped_mass=0.0,pool_mass=0.0;
};

// Decode the original CPU payload, independently of the device's expert-stride repacking.
NintDispatchError nint_dispatch_reference(const NintCpu& weight,const Workspace& input,int source,int row) {
    mfq_half scale_half,minimum_half;
    std::memcpy(&scale_half,&weight.neuron_scale_h[row],sizeof(scale_half));
    std::memcpy(&minimum_half,&weight.neuron_min_h[row],sizeof(minimum_half));
    const double outer_scale=static_cast<float>(scale_half),outer_minimum=static_cast<float>(minimum_half);
    const auto* codes=input.qx.data_ptr<int8_t>()+int64_t(source)*weight.ng*weight.gs;
    const auto* scales=input.xscale.data_ptr<float>()+int64_t(source)*weight.ng;
    NintDispatchError result;
    for(int group=0;group<weight.ng;++group) {
        const auto metadata=int64_t(row)*weight.ng+group;
        const double scale=outer_scale*weight.sub_scale[metadata];
        const double minimum=outer_minimum*weight.sub_min[metadata];
        int dot=0,sum=0;
        for(int chunk=0;chunk<weight.gs;chunk+=4) {
            int chunk_dot=0,chunk_sum=0;
            for(int element=chunk;element<std::min(chunk+4,weight.gs);++element) {
                const int column=group*weight.gs+element;
                const auto bit=weight.row_q_bit_offsets[row]+int64_t(column)*weight.row_q_bits[row];
                const int code=cpu_load_packed_bits(weight.q_packed.data(),weight.q_packed.size(),
                    bit,weight.row_q_bits[row]);
                chunk_dot+=code*codes[column];
                chunk_sum+=codes[column];
            }
            dot+=chunk_dot;
            sum+=chunk_sum;
            result.pool_mass+=scales[group]*(std::abs(scale*chunk_dot)+std::abs(minimum*chunk_sum));
        }
        result.value+=scales[group]*(scale*dot-minimum*sum);
        result.grouped_mass+=scales[group]*(std::abs(scale*dot)+std::abs(minimum*sum));
    }
    return result;
}

double nint_dispatch_error_bound(double value,double mass,int iterations,int groups) {
    // Integer dots/sums fit int32 and are exact in FP32 (gs<=64, q<=255, |x|<=127).
    // Each affine term has at most four rounded operations before accumulation:
    // row*sub, *dot (or *sum), subtraction, *xscale. FMA can only reduce this count.
    // A lane adds ceil(G/32) grouped terms, or ceil(G/floor(32/ceil(gs/4))) pool chunks,
    // followed by five warp tree levels.
    // Thus |FP32-R| <= gamma_(4+iterations+5) * sum(|positive term|+|negative term|),
    // gamma_n=n*u/(1-n*u), u=2^-24. The scalar FP64 oracle has gamma_(G+4), u=2^-53.
    // FP16 storage adds u16*|R|+(1+u16)*FP32_bound plus half a subnormal ULP.
    const double u32=std::ldexp(1.0,-24),u64=std::ldexp(1.0,-53),u16=std::ldexp(1.0,-11);
    const double n32=iterations+9,n64=groups+4;
    const double arithmetic=(n32*u32/(1-n32*u32)+n64*u64/(1-n64*u64))*mass;
    return u16*std::abs(value)+(1+u16)*arithmetic+std::ldexp(1.0,-25);
}

struct NintDispatchStats {
    int64_t values=0,different=0,grouped_closer=0,pool_closer=0;
    double grouped_squared=0.0,pool_squared=0.0,grouped_max=0.0,pool_max=0.0,max_bound_ratio=0.0;
};

void verify_nint_dispatch_output(const tb::Tensor& actual,const tb::Tensor& expected,
        const std::vector<NintCpu>& weights,const std::vector<Workspace>& inputs,
        const std::vector<int32_t>& selected,bool down,int tokens,int step,std::vector<NintDispatchStats>& stats) {
    if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=tb::kFloat16 ||
            expected.scalar_type()!=tb::kFloat16)
        throw std::runtime_error("grouped NINT output shape or dtype differs from original pool kernels");
    auto x=actual.cpu().to(tb::kFloat32).contiguous(),y=expected.cpu().to(tb::kFloat32).contiguous();
    const auto* a=x.data_ptr<float>();
    const auto* b=y.data_ptr<float>();
    const int output=int(actual.size(2));
    for(int pair=0;pair<int(selected.size());++pair) {
        const int expert=selected[pair],pool=expert/3,local=expert%3;
        const bool owned=pool<int(weights.size()) && local!=step;
        for(int row=0;row<output;++row) {
            const auto index=int64_t(pair)*output+row;
            if(!owned) {
                if(a[index]!=0.0f || b[index]!=0.0f)
                    throw std::runtime_error("unowned NINT route did not remain exactly zero");
                continue;
            }
            const auto& weight=weights[pool];
            const auto reference=nint_dispatch_reference(weight,inputs[pool],down?pair:pair/10,local*output+row);
            const int chunks=(weight.gs+3)/4,groups_per_warp=32/chunks;
            const int pool_iterations=(weight.ng+groups_per_warp-1)/groups_per_warp;
            const bool grouped=weight.gs==24 || weight.gs==28;
            const double grouped_bound=nint_dispatch_error_bound(reference.value,
                grouped?reference.grouped_mass:reference.pool_mass,
                grouped?(weight.ng+31)/32:pool_iterations,weight.ng);
            const double pool_bound=nint_dispatch_error_bound(
                reference.value,reference.pool_mass,pool_iterations,weight.ng);
            const double ae=std::abs(a[index]-reference.value),be=std::abs(b[index]-reference.value);
            if(!std::isfinite(a[index]) || !std::isfinite(b[index]) || !std::isfinite(reference.value) ||
                    !std::isfinite(grouped_bound) || !std::isfinite(pool_bound) || ae>grouped_bound || be>pool_bound) {
                std::ostringstream message;
                message<<std::setprecision(12)<<"NINT output exceeds FP32 reduction + FP16 storage error bound"
                    <<" down="<<down<<" tokens="<<tokens<<" step="<<step<<" index="<<index
                    <<" bits="<<int(weight.row_q_bits[local*output+row])<<" reference="<<reference.value
                    <<" grouped="<<a[index]<<" pool="<<b[index]
                    <<" grouped_bound="<<grouped_bound<<" pool_bound="<<pool_bound;
                throw std::runtime_error(message.str());
            }
            auto& s=stats[pool];
            if(a[index]!=b[index] && !s.different)
                std::cout<<std::setprecision(12)<<"nint_dispatch_difference bits="<<weight.bits
                    <<" down="<<down<<" tokens="<<tokens<<" step="<<step<<" index="<<index
                    <<" grouped="<<a[index]<<" pool="<<b[index]<<" reference="<<reference.value
                    <<" grouped_error="<<ae<<" pool_error="<<be
                    <<" grouped_bound="<<grouped_bound<<" pool_bound="<<pool_bound<<'\n';
            ++s.values;
            s.different+=a[index]!=b[index];
            s.grouped_closer+=ae<be;
            s.pool_closer+=be<ae;
            s.grouped_squared+=ae*ae;
            s.pool_squared+=be*be;
            s.grouped_max=std::max(s.grouped_max,ae);
            s.pool_max=std::max(s.pool_max,be);
            s.max_bound_ratio=std::max({s.max_bound_ratio,ae/grouped_bound,be/pool_bound});
        }
    }
}

int verify_nint_dispatch(const std::filesystem::path& root) {
    CudaExecutionContext execution;int cases=0;
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();
    context->begin_graph_pool(stream);
    std::vector<NintDispatchStats> stats(4);
    for(bool down:{false,true}) {
        const int width=down?640:2560,output=down?2560:640;
        MixedMoeRuntime runtime;runtime.n_experts=512;runtime.neuron_len=width;
        runtime.out_per_expert=output;runtime.partial_experts=true;
        int index=0;
        std::vector<NintCpu> weights;
        for(const std::string format:{"nint4","nint5","nint6","nint8"}) {
            auto model=mfq::open_model_source((root/(format+(down?"-2560-640.mfq":"-640-2560.mfq"))).string());
            auto cpu=load_mfe_cpu(*model,"linear.weight");
            for(const auto& pool:cpu.pools)weights.push_back(pool.weight);
            auto single=make_mixed_moe_runtime(cpu,true,execution.config);
            for(auto& pool:single->pools) {
                std::vector<int32_t> local(512,-1);for(int e=0;e<3;++e)local[index*3+e]=e;
                pool.expert_local=tb::tensor(local).to(tb::kCUDA);runtime.pools.push_back(std::move(pool));
            }
            ++index;
        }
        initialize_mixed_nvq_dispatch(runtime,execution.config);
        if(!runtime.nint_dispatch || runtime.nint_dispatch->pools.size()!=4)
            throw std::runtime_error("mixed NINT fixture did not initialize four format pools");
        for(int tokens:{1,3,8}) {
            auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
            auto x=tb::zeros(down?std::vector<int64_t>{tokens,10,width}:std::vector<int64_t>{tokens,width},options);
            auto ids=tb::zeros({tokens,10},options.dtype(tb::kInt32));tb::Tensor actual;
            const auto forward=[&] {
                return runtime.forward(execution.config,execution.kl_mmq,false,false,x,build_moe_route_plan(ids,512));
            };
            execution.config.moe_nint_heterogeneous_decode=true;
            mfq::cuda::DecodeWindow window(stream);
            window.capture([&]{actual=forward();},[]{},[&]{actual={};});
            for(int step=0;step<3;++step) {
                std::vector<float> values(x.numel());std::vector<int32_t> selected(tokens*10);
                for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+step*31)*.017f)*.3f;
                for(int i=0;i<tokens*10;++i)selected[i]=i%10==9?511:(i*5+step*3)%12;
                for(std::size_t p=0;p<runtime.pools.size();++p) {
                    std::vector<int32_t> map(512,-1);
                    for(int e=0;e<3;++e)if(e!=step)map[p*3+e]=e;
                    runtime.pools[p].expert_local.copy_(tb::tensor(map).to(tb::kCUDA));
                }
                x.copy_(tb::tensor(values).reshape(x.sizes().vec()).to(tb::kFloat16).to(tb::kCUDA));
                ids.copy_(tb::tensor(selected).reshape({tokens,10}).to(tb::kCUDA));
                execution.config.moe_nint_heterogeneous_decode=false;
                auto expected=forward();
                std::vector<Workspace> inputs;
                for(const auto& weight:weights) {
                    const auto workspace=runtime.activation_workspace(x,int(x.numel()/width),weight.ng,weight.gs,{});
                    Workspace input;
                    input.qx=workspace.qx.cpu().contiguous();
                    input.xscale=workspace.xscale.cpu().contiguous();
                    inputs.push_back(std::move(input));
                }
                execution.config.moe_nint_heterogeneous_decode=true;
                window.run();
                for(std::size_t p=0;p<weights.size();++p) {
                    const auto workspace=runtime.activation_workspace(
                        x,int(x.numel()/width),weights[p].ng,weights[p].gs,{});
                    const auto qx=workspace.qx.cpu().contiguous(),scale=workspace.xscale.cpu().contiguous();
                    if(std::memcmp(qx.data_ptr(),inputs[p].qx.data_ptr(),qx.numel()*qx.element_size()) ||
                            std::memcmp(scale.data_ptr(),inputs[p].xscale.data_ptr(),
                                scale.numel()*scale.element_size())) {
                        throw std::runtime_error(
                            "grouped NINT quantized input or scales differ from original pool kernels");
                    }
                }
                verify_nint_dispatch_output(actual,expected,weights,inputs,selected,down,tokens,step,stats);
                ++cases;
            }
        }
    }
    for(std::size_t p=0;p<stats.size();++p) {
        const auto& s=stats[p];
        std::cout<<"nint_dispatch_accuracy bits="<<(p==3?8:p+4)<<" values="<<s.values
            <<" different="<<s.different<<" grouped_closer="<<s.grouped_closer<<" pool_closer="<<s.pool_closer
            <<" grouped_rmse="<<std::sqrt(s.grouped_squared/s.values)<<" pool_rmse="<<std::sqrt(s.pool_squared/s.values)
            <<" grouped_max="<<s.grouped_max<<" pool_max="<<s.pool_max<<" max_bound_ratio="<<s.max_bound_ratio<<'\n';
    }
    std::cout<<"nint_dispatch_cases="<<cases
        <<" real640/2560 four-format FP64 oracle, FP32/FP16 error bounds,"
        <<" exact input bytes/scales and unowned zeros PASS\n";
    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));context->end_graph_pool(stream);
    return cases;
}

int verify_phased_wire_bytes() {
    const std::vector<int64_t> sizes={3,1280,2564,640*107*16+7};
    const std::vector<int64_t> prefixes={1,16,4,0};
    const std::size_t header=(32+24*sizes.size()+15)&~std::size_t(15);
    std::vector<uint8_t> packed(header,0);
    std::vector<tb::Tensor> outputs;
    std::vector<std::vector<uint8_t>> initial,expected;
    std::vector<uint64_t> descriptors;
    std::size_t boundary=0;
    for(std::size_t field=0;field<sizes.size();++field) {
        initial.emplace_back(sizes[field]+prefixes[field]+32,0xa7);expected.push_back(initial.back());
        outputs.push_back(tb::tensor(initial.back()).to(tb::kCUDA));
        packed.resize((packed.size()+15)&~std::size_t(15),0);
        if(field==2)boundary=packed.size();
        descriptors.push_back(reinterpret_cast<uint64_t>(outputs.back().data_ptr<uint8_t>()+prefixes[field]));
        descriptors.push_back(packed.size());descriptors.push_back(sizes[field]);
        for(int64_t i=0;i<sizes[field];++i) {
            const auto value=static_cast<uint8_t>(i*29+field*71);
            packed.push_back(value);expected.back()[prefixes[field]+i]=value;
        }
    }
    const uint64_t words[4]={sizes.size(),header,packed.size(),boundary};
    std::memcpy(packed.data(),words,sizeof(words));std::memcpy(packed.data()+32,descriptors.data(),descriptors.size()*8);
    auto wire=tb::tensor(packed).to(tb::kCUDA);int blocks=0,threads=0;
    MFQ_CUDA_CHECK(mfq::cuda::moe_wire_launch_geometry(&blocks,&threads));
    auto cancelled=tb::zeros({1},wire.options().dtype(tb::kInt32));int cases=0;
    for(bool abort:{false,true}) {
        for(std::size_t f=0;f<outputs.size();++f)outputs[f].copy_(tb::tensor(initial[f]).to(tb::kCUDA));
        cancelled.fill_(abort?1:0);
        for(int phase=1;phase<=2;++phase) {
            mfq::cuda::moe_scatter_wire_phase(wire.data_ptr(),reinterpret_cast<const uint32_t*>(cancelled.data_ptr<int32_t>()),
                blocks,threads,phase,mfq_current_cuda_stream());
            for(std::size_t f=0;f<outputs.size();++f) {
                const auto& oracle=abort || (phase==1 && f>=2)?initial[f]:expected[f];
                auto actual=outputs[f].cpu().contiguous();
                if(std::memcmp(actual.data_ptr(),oracle.data(),oracle.size()))
                    throw std::runtime_error("phased scatter crossed the Gate/Up/Down boundary or neighboring byte guards");
            }
            ++cases;
        }
    }
    // Empty Gate/Up and empty Down each preserve the complete wire contract.
    for(int empty:{1,2}) {
        auto modified=packed;const uint64_t split=empty==1?header:(packed.size()+15)&~std::size_t(15);
        std::memcpy(modified.data()+24,&split,8);wire.copy_(tb::tensor(modified).to(tb::kCUDA));cancelled.zero_();
        for(std::size_t f=0;f<outputs.size();++f)outputs[f].copy_(tb::tensor(initial[f]).to(tb::kCUDA));
        for(int phase=1;phase<=2;++phase)mfq::cuda::moe_scatter_wire_phase(wire.data_ptr(),
            reinterpret_cast<const uint32_t*>(cancelled.data_ptr<int32_t>()),blocks,threads,phase,mfq_current_cuda_stream());
        for(std::size_t f=0;f<outputs.size();++f) {
            auto actual=outputs[f].cpu().contiguous();
            if(std::memcmp(actual.data_ptr(),expected[f].data(),expected[f].size()))
                throw std::runtime_error("empty phased packet changed original payload bytes");
        }
        ++cases;
    }
    std::cout<<"phased_wire_cases="<<cases<<" boundary and tail bytes, cancelled/empty packets exact PASS\n";
    return cases;
}

int verify_wire_bytes() {
    const std::vector<int64_t> sizes={3,1280,2564,640*107*16+7};
    const std::vector<int64_t> prefixes={1,16,4,0};
    const std::size_t header=(32+24*sizes.size()+15)&~std::size_t(15);
    std::vector<uint8_t> packed(header,0);
    std::vector<tb::Tensor> outputs;
    std::vector<std::vector<uint8_t>> expected;
    std::vector<uint64_t> descriptors;
    for(std::size_t field=0;field<sizes.size();++field) {
        std::vector<uint8_t> initial(sizes[field]+prefixes[field]+32,0xa7);
        outputs.push_back(tb::tensor(initial).to(tb::kCUDA));expected.push_back(initial);
        packed.resize((packed.size()+15)&~std::size_t(15),0);
        descriptors.push_back(reinterpret_cast<uint64_t>(outputs.back().data_ptr<uint8_t>()+prefixes[field]));
        descriptors.push_back(packed.size());descriptors.push_back(sizes[field]);
        for(int64_t i=0;i<sizes[field];++i) {
            const auto value=static_cast<uint8_t>(i*29+field*71);
            packed.push_back(value);expected.back()[prefixes[field]+i]=value;
        }
    }
    const uint64_t words[4]={sizes.size(),header,packed.size(),0};
    std::memcpy(packed.data(),words,sizeof(words));
    std::memcpy(packed.data()+32,descriptors.data(),descriptors.size()*sizeof(uint64_t));
    auto wire=tb::tensor(packed).to(tb::kCUDA);int blocks=0,threads=0;
    auto cancelled=tb::zeros({1},wire.options().dtype(tb::kInt32));
    mfq::cuda::HostBuffer flags(128,true);std::memset(flags.data(),0,flags.size());
    auto* host_flag=static_cast<uint32_t*>(flags.data());uint32_t* mapped_flag=nullptr;
    MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped_flag),host_flag,0));
    auto slot=tb::ones({1},cancelled.options());
    auto kind=tb::empty({1},cancelled.options());host_flag[2]=1;
    mfq::cuda::publish_mapped_flag(host_flag);
    std::vector<tb::Tensor> maps;
    for(int i=0;i<3;++i)maps.push_back(tb::empty({1},cancelled.options()));
    std::vector<int64_t> pointers(3,reinterpret_cast<int64_t>(slot.data_ptr()));
    for(const auto& map:maps)pointers.push_back(reinterpret_cast<int64_t>(map.data_ptr()));
    auto dispatch=tb::tensor(pointers).to(tb::kCUDA);
    MFQ_CUDA_CHECK(mfq::cuda::moe_wire_launch_geometry(&blocks,&threads));
    for(int replay=0;replay<3;++replay) {
        if(replay==2)mfq::cuda::publish_mapped_flag(host_flag+1);
        mfq::cuda::wait_mapped_plan(mapped_flag,reinterpret_cast<const int32_t*>(mapped_flag+2),kind.data_ptr<int32_t>(),
            1,mapped_flag+1,reinterpret_cast<uint32_t*>(cancelled.data_ptr<int32_t>()),mfq_current_cuda_stream());
        mfq::cuda::moe_update_dispatch_maps(reinterpret_cast<const uint64_t*>(dispatch.data_ptr<int64_t>()),
            1,1,kind.data_ptr<int32_t>(),reinterpret_cast<const uint32_t*>(cancelled.data_ptr<int32_t>()),
            mfq_current_cuda_stream());
        mfq::cuda::moe_scatter_wire(wire.data_ptr(),reinterpret_cast<const uint32_t*>(cancelled.data_ptr<int32_t>()),
            blocks,threads,mfq_current_cuda_stream());
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        if(cancelled.to(tb::kCPU).data_ptr<int32_t>()[0]!=(replay==2))
            throw std::runtime_error("mapped cancellation snapshot differs from host publication");
        if(kind.to(tb::kCPU).data_ptr<int32_t>()[0]!=1)
            throw std::runtime_error("mapped dispatch snapshot differs from host publication");
        for(int i=0;i<3;++i)if(maps[i].to(tb::kCPU).data_ptr<int32_t>()[0]!=(i==0 && replay!=2 ? 1 : -1))
            throw std::runtime_error("dispatch maps differ from cancellation snapshot");
        for(std::size_t field=0;field<sizes.size();++field) {
            auto actual=outputs[field].to(tb::kCPU).contiguous();
            if(std::memcmp(actual.data_ptr(),expected[field].data(),expected[field].size()))
                throw std::runtime_error("packed DMA scatter changed field bytes or neighboring guards");
        }
        if(!replay) {
            std::fill_n(packed.data(),8,uint8_t(0));
            for(std::size_t i=header;i<packed.size();++i)packed[i]^=0x5a;
            wire.copy_(tb::tensor(packed).to(tb::kCUDA));
        }
        if(replay==1) {
            std::memcpy(packed.data(),words,sizeof(words));wire.copy_(tb::tensor(packed).to(tb::kCUDA));
        }
    }
    return 3;
}
}
void verify_copy_kernel_overlap() {
    for(bool use_kernel:{true,false})for(bool allocate_during_tail:{false,true}) {
        const auto compute=mfq_current_cuda_stream();
        const auto copy=mfq_get_stream_from_pool();
        auto source=tb::tensor(std::vector<uint8_t>(64,0x79)).to(tb::kCUDA);
        auto destination=tb::zeros({64},source.options());
        mfq::MoeCacheMappedCopyDescriptor descriptor{reinterpret_cast<uint64_t>(destination.data_ptr()),
            reinterpret_cast<uint64_t>(source.data_ptr()),64};
        auto descriptors=tb::empty({int64_t(sizeof(descriptor))},source.options());
        MFQ_CUDA_CHECK(cudaMemcpy(descriptors.data_ptr(),&descriptor,sizeof(descriptor),cudaMemcpyHostToDevice));
        mfq::cuda::HostBuffer flags(128,true),result(64);
        uint32_t* alias=nullptr;std::memset(flags.data(),0,flags.size());
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&alias),flags.data(),0));
        const auto gather=[&] {mfq::moe_cache_mapped_gather_cuda(
            reinterpret_cast<const mfq::MoeCacheMappedCopyDescriptor*>(descriptors.data_ptr()),1,1,copy.stream());};
        gather();mfq_cuda_synchronize(); // load the kernel before blocking the model stream
        cudaEvent_t done;MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        MFQ_CUDA_CHECK(cudaStreamBeginCapture(compute,cudaStreamCaptureModeThreadLocal));
        mfq::cuda::wait_mapped_flag(alias,compute);
        MFQ_CUDA_CHECK(cudaStreamEndCapture(compute,&graph));
        MFQ_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
        MFQ_CUDA_CHECK(cudaGraphLaunch(executable,compute));mfq::cuda::check_stream_progress(compute);
        std::atomic_bool finished=false,rescued=false;
        std::jthread watchdog([&] {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(750);
            while(!finished && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if(!finished){rescued=true;mfq::cuda::publish_mapped_flag(static_cast<uint32_t*>(flags.data()));}
        });
        tb::Tensor new_allocation;
        {MfqCudaStreamGuard selected(copy);
            if(allocate_during_tail)new_allocation=tb::empty({32*1024*1024},source.options());
            if(use_kernel)gather();
            else MFQ_CUDA_CHECK(cudaMemcpyAsync(destination.data_ptr(),source.data_ptr(),64,cudaMemcpyDeviceToDevice,copy.stream()));
            MFQ_CUDA_CHECK(cudaMemcpyAsync(result.data(),destination.data_ptr(),64,cudaMemcpyDeviceToHost,copy.stream()));
            MFQ_CUDA_CHECK(cudaEventRecord(done,copy.stream()));
            MFQ_CUDA_CHECK(cudaEventSynchronize(done));
        }
        finished=true;mfq::cuda::publish_mapped_flag(static_cast<uint32_t*>(flags.data()));
        watchdog.join();mfq_cuda_synchronize();
        for(int i=0;i<64;++i)if(static_cast<uint8_t*>(result.data())[i]!=0x79)
            throw std::runtime_error("overlap probe copy bytes differ");
        std::cout<<"copy_kernel_overlap use_kernel="<<use_kernel<<" allocate_during_tail="<<allocate_during_tail
                 <<" completed_before_tail="<<!rescued.load()<<" bytes_exact=1"<<std::endl;
        cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);cudaEventDestroy(done);
    }
}
void verify_mapped_refill_descriptors() {
    constexpr int fields=8;
    constexpr int64_t stride=65536;
    std::vector<uint8_t> source_bytes(fields*stride),expected(fields*stride),actual(fields*stride);
    auto input=tb::empty({fields*stride},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
    auto output=tb::empty({fields*stride},input.options());
    mfq::cuda::HostBuffer mapped(fields*sizeof(mfq::MoeCacheMappedCopyDescriptor),true);
    auto* host=static_cast<mfq::MoeCacheMappedCopyDescriptor*>(mapped.data());
    mfq::MoeCacheMappedCopyDescriptor* device=nullptr;
    MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device),mapped.data(),0));
    const auto stream=mfq_current_cuda_stream();int cases=0;
    for(int round=0;round<96;++round) {
        const int count=1+round%fields;
        for(std::size_t i=0;i<source_bytes.size();++i)source_bytes[i]=uint8_t(i*29+round*17);
        std::fill(expected.begin(),expected.end(),uint8_t(0xa7));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(input.data_ptr(),source_bytes.data(),source_bytes.size(),cudaMemcpyHostToDevice,stream));
        MFQ_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(),0xa7,expected.size(),stream));
        for(int f=0;f<count;++f) {
            const int64_t prefixes[]={16,8,4,2,3,1};
            const auto prefix=prefixes[round%6];
            const int64_t bytes=round%8?4096*(1+f)+round%16:1+f;
            const auto from=(fields-1-f)*stride+prefix+(round%12>=6?1:0),to=f*stride+prefix;
            host[f]={reinterpret_cast<uint64_t>(output.data_ptr<uint8_t>()+to),
                reinterpret_cast<uint64_t>(input.data_ptr<uint8_t>()+from),uint64_t(bytes)};
            std::memcpy(expected.data()+to,source_bytes.data()+from,std::size_t(bytes));
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        mfq::moe_cache_mapped_gather_cuda(device,count,round%2?1:8,stream);
        MFQ_CUDA_CHECK(cudaMemcpyAsync(actual.data(),output.data_ptr(),actual.size(),cudaMemcpyDeviceToHost,stream));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
        if(actual!=expected)throw std::runtime_error("mapped refill descriptor changed payload or guard bytes");
        ++cases;
    }
    std::cout<<"mapped_refill_descriptor_cases="<<cases<<" changing pointers/counts/unaligned tails/guards exact PASS\n";
}
void verify_speculative_route_prefix(const std::filesystem::path& root) {
    CudaExecutionContext execution;auto& config=execution.config;
    config.moe_residency_warm=true;config.moe_residency_adapt=true;
    config.moe_pipeline=true;config.moe_ram_pcie=true;
    config.moe_preload_all=true;config.moe_assert_resident=true;config.moe_residency_projection_heat=false;
    std::array<std::shared_ptr<MoeQuantRangeSource>,3> sources={source(root/"nint5-640-2560.mfq"),
        source(root/"nint5-640-2560.mfq"),source(root/"nint5-2560-640.mfq")};
    std::size_t bundle=0;for(const auto& item:sources)bundle+=bytes(item->metadata()->pools[0]);
    config.moe_host_cache_bytes=3*bundle;auto cache=make_moe_expert_cache(bundle,config);
    std::array<std::shared_ptr<MfeWeight>,3> weights;
    for(int p=0;p<3;++p)weights[p]=std::make_shared<MfeWeight>(
        cache_quant_moe_weight(cache,std::to_string(p),sources[p],1,0,std::to_string(p)));
    auto forward=make_moe_ffn_pipeline({weights[0],weights[1]},{weights[2]});
    finalize_moe_expert_cache(cache);MoeResidencyManager manager(cache.get());
    const auto* heat_option=std::getenv("MFQ_MOE_MTP_WINDOW_HEAT");
    const int heat_mode=heat_option?std::clamp(std::atoi(heat_option),0,2):2;
    const bool window_heat=heat_mode!=0;
    for(int i=0;i<3;++i) {
        manager.begin_speculative_routes(4);
        manager.after_layer(0,{1,2,2,2},4);
        if(manager.stats().observed_routes!=std::uint64_t(i))
            throw std::runtime_error("unverified routes entered expert heat");
        manager.commit_speculative_routes(1);
    }
    manager.begin_speculative_routes(4);manager.after_layer(0,{2,2,2,2},4);
    manager.discard_speculative_routes();manager.discard_speculative_routes();
    manager.begin_speculative_routes(4);
    manager.after_layer(0,{2,2},2);manager.after_layer(0,{1,2,2,2},4);
    manager.commit_speculative_routes(1);
    manager.begin_speculative_routes(4);manager.after_layer(0,{1,1,2,2},4);
    manager.commit_speculative_routes(2);
    manager.begin_speculative_routes(3);manager.after_layer(0,{1,2,2},3);
    bool rejected=false;
    try{manager.commit_speculative_routes(4);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("invalid accepted route prefix was committed");
    manager.discard_speculative_routes();
    manager.begin_speculative_routes(4);rejected=false;
    try{manager.after_layer(0,{1,2,2,2},4,false,true);}catch(const std::logic_error&){rejected=true;}
    if(!rejected)throw std::runtime_error("independent sequences entered speculative journal");
    manager.discard_speculative_routes();
    manager.begin_speculative_routes(4);rejected=false;
    try{manager.after_layer(0,{1,INT32_MAX,2,2},4);}catch(const std::out_of_range&){rejected=true;}
    if(!rejected)throw std::runtime_error("invalid expert entered speculative journal");
    manager.discard_speculative_routes();
    manager.after_layer(0,{0},1);
    const auto stats=manager.stats();
    if(stats.observed_routes!=7 || stats.speculative_route_windows!=5 ||
            stats.speculative_routes_committed!=6 || stats.speculative_routes_discarded!=21 ||
            stats.speculative_route_cancels!=4 || stats.speculative_heat_routes!=(window_heat?5:6))
        throw std::runtime_error("accepted route prefix accounting changed");
    // Rejected expert 2 appeared more often than accepted expert 1. Actual
    // residency must follow the committed prefix rather than those drafts.
    manager.apply_pending();
    manager.after_layer(0,{},32);manager.apply_pending();
    for(const auto& source:sources)
        if(!source->gpu_resident(1) || source->gpu_resident(2))
            throw std::runtime_error("rejected draft routes drove expert promotion");
    manager.begin_speculative_routes(4);manager.after_layer(0,{0,1,2,1},4);
    manager.commit_speculative_routes(4);manager.apply_pending();
    const auto full=manager.stats();
    if(full.observed_routes!=11 || full.speculative_route_windows!=6 ||
            full.speculative_routes_committed!=10 || full.speculative_routes_discarded!=21 ||
            full.speculative_heat_routes!=(window_heat?8:10))
        throw std::runtime_error("fully accepted route window lost observations");
    // A single verification's repeated use does not pay for four transfers.
    // Check actual residency, including the legacy policy used for the A/B.
    for(int p=0;p<3;++p)sources[p]=source(root/(p==2?"nint5-2560-640.mfq":"nint5-640-2560.mfq"));
    auto repeated_cache=make_moe_expert_cache(bundle,config);
    std::array<std::shared_ptr<MfeWeight>,3> repeated_weights;
    for(int p=0;p<3;++p)repeated_weights[p]=std::make_shared<MfeWeight>(
        cache_quant_moe_weight(repeated_cache,std::to_string(p),sources[p],1,0,std::to_string(p)));
    auto repeated_forward=make_moe_ffn_pipeline({repeated_weights[0],repeated_weights[1]},{repeated_weights[2]});
    finalize_moe_expert_cache(repeated_cache);MoeResidencyManager repeated(repeated_cache.get());
    repeated.begin_speculative_routes(4);repeated.after_layer(0,{1,1,1,1},4);
    repeated.commit_speculative_routes(4);repeated.apply_pending();
    repeated.after_layer(0,{},32);repeated.apply_pending();
    for(const auto& source:sources)if(source->gpu_resident(1)==(heat_mode==1))
        throw std::runtime_error("one repeated draft window used token frequency for transfer heat");
    for(int step=0;step<2;++step) {
        repeated.begin_speculative_routes(4);repeated.after_layer(0,{2,2,2,2},4);
        repeated.commit_speculative_routes(4);repeated.apply_pending();
    }
    repeated.after_layer(0,{},32);repeated.apply_pending();
    for(const auto& source:sources)if(!source->gpu_resident(2))
        throw std::runtime_error("repeated committed windows did not promote their hot expert");
    for(int p=0;p<3;++p)sources[p]=source(root/(p==2?"nint5-2560-640.mfq":"nint5-640-2560.mfq"));
    auto mixed_cache=make_moe_expert_cache(bundle,config);
    std::array<std::shared_ptr<MfeWeight>,3> mixed_weights;
    for(int p=0;p<3;++p)mixed_weights[p]=std::make_shared<MfeWeight>(
        cache_quant_moe_weight(mixed_cache,std::to_string(p),sources[p],1,0,std::to_string(p)));
    auto mixed_forward=make_moe_ffn_pipeline({mixed_weights[0],mixed_weights[1]},{mixed_weights[2]});
    finalize_moe_expert_cache(mixed_cache);MoeResidencyManager mixed(mixed_cache.get());
    mixed.begin_speculative_routes(4);mixed.after_layer(0,{1,1,1,2},4);
    mixed.commit_speculative_routes(4);mixed.apply_pending();
    mixed.after_layer(0,{},32);mixed.apply_pending();
    // Both experts save one transfer. Normalized heat ranks their equal gain
    // by the stable expert tie-break; legacy token heat picks repeated expert 1.
    for(const auto& source:sources)
        if(source->gpu_resident(1)!=(heat_mode==0) || source->gpu_resident(2)!=(heat_mode==2))
            throw std::runtime_error("mixed route multiplicity changed transfer ranking or hysteresis scale");
    std::cout<<"speculative_route_prefix pending/partial/full-prefix, cancel/reset, repeated graph setup, ordinary routes and actual promoted expert PASS\n";
}
void verify_short_decode_residency_interval(const std::filesystem::path& root) {
    struct Interval {
        std::string saved;
        Interval(){if(const auto* value=std::getenv("MFQ_MOE_RESIDENCY_INTERVAL"))saved=value;set("4");}
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("MFQ_MOE_RESIDENCY_INTERVAL",value);
#else
            if(*value)setenv("MFQ_MOE_RESIDENCY_INTERVAL",value,1);else unsetenv("MFQ_MOE_RESIDENCY_INTERVAL");
#endif
        }
        ~Interval(){set(saved.c_str());}
    } interval;
    int checks=0;
    for(bool independent:{false,true})for(int tokens:{1,2,3,4,5,6,8,31}) {
        CudaExecutionContext execution;auto& config=execution.config;config.moe_residency_warm=true;
        config.moe_pipeline=true;config.moe_ram_pcie=true;
        config.moe_preload_all=true;config.moe_assert_resident=true;config.moe_residency_projection_heat=false;
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> sources={source(root/"nint5-640-2560.mfq"),
            source(root/"nint5-640-2560.mfq"),source(root/"nint5-2560-640.mfq")};
        std::size_t bundle=0;for(const auto& item:sources)bundle+=bytes(item->metadata()->pools[0]);
        config.moe_host_cache_bytes=3*bundle;
        auto cache=make_moe_expert_cache(bundle,config);
        std::array<std::shared_ptr<MfeWeight>,3> weights;
        for(int projection=0;projection<3;++projection)weights[projection]=std::make_shared<MfeWeight>(
            cache_quant_moe_weight(cache,std::to_string(projection),sources[projection],1,0,std::to_string(projection)));
        auto forward=make_moe_ffn_pipeline({weights[0],weights[1]},{weights[2]});
        finalize_moe_expert_cache(cache);
        MoeResidencyManager manager(cache.get());
        for(int step=1;step<=8;++step) {
            manager.after_layer(0,std::vector<int32_t>(tokens,step%2),tokens,false,independent);
            manager.apply_pending();
            const auto stats=manager.stats();
            if(stats.scheduled_rounds!=std::uint64_t(tokens<=8?step/4:step) ||
                stats.observed_routes!=std::uint64_t(tokens*step))
                throw std::runtime_error("short decode interval lost route observations or scheduled too often");
            ++checks;
        }
    }
    std::cout<<"short_decode_residency_interval checks="<<checks<<" independent and speculative rows1-6/8, prefill unchanged, all route observations retained PASS\n";
}
int verify_warm_residency(const std::filesystem::path& root, bool shared_cases=true) {
    int cases=0,phase_cases=0,async_cases=0,window_cases=0,fence_cases=0;
    struct TailGate {
        MoeResidencyManager& manager;
        mfq::cuda::HostBuffer flags{128,true};
        uint32_t* alias=nullptr;
        cudaStream_t stream=mfq_current_cuda_stream();
        cudaGraph_t graph=nullptr;
        cudaGraphExec_t executable=nullptr;
        std::atomic_bool waiting=false,rescued=false;
        std::atomic<int64_t> started_ms=0;
        std::jthread watchdog;
        static int64_t now_ms() {return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();}
        TailGate(MoeResidencyManager& owner,int last):manager(owner) {
            std::memset(flags.data(),0,flags.size());
            MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&alias),flags.data(),0));
            manager.record_window_expert_fence(last,stream);
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
            MFQ_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            mfq::cuda::wait_mapped_flag(alias,stream);
            manager.record_window_expert_fence(last,stream);
            mfq::cuda::wait_mapped_flag(alias+16,stream);
            MFQ_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
            MFQ_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
            watchdog=std::jthread([this](std::stop_token stop) {
                while(!stop.stop_requested()) {
                    if(waiting.load() && now_ms()-started_ms.load()>750) {
                        rescued=true;before_done();tail_done();
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
        }
        void before_done(){mfq::cuda::publish_mapped_flag(static_cast<uint32_t*>(flags.data()));}
        void tail_done(){mfq::cuda::publish_mapped_flag(static_cast<uint32_t*>(flags.data())+16);waiting=false;}
        void start(bool hold_before=false) {
            manager.before_layer(0);
            std::memset(flags.data(),0,flags.size());if(!hold_before)before_done();
            rescued=false;started_ms=now_ms();waiting=true;
            MFQ_CUDA_CHECK(cudaGraphLaunch(executable,stream));mfq::cuda::check_stream_progress(stream);
        }
        ~TailGate() {
            watchdog.request_stop();if(watchdog.joinable())watchdog.join();
            before_done();tail_done();(void)cudaStreamSynchronize(stream);
            if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        }
    };
    const auto wait_copies=[](const std::atomic_bool& complete) {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(750);
        while(!complete.load(std::memory_order_acquire)) {
            if(std::chrono::steady_clock::now()>deadline)
                throw std::runtime_error("expert copies still wait for the blocked output tail");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };
    const auto wait_preparation=[](const std::atomic_bool& prepared) {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while(!prepared.load(std::memory_order_acquire)) {
            if(std::chrono::steady_clock::now()>deadline)
                throw std::runtime_error("window residency preparation did not progress before graph finish");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };
    for(const std::string format:{"nint5","nvq3j-512"})for(int mode:{0,1,2}) {
        CudaExecutionContext execution;execution.config.moe_residency_warm=true;
        const bool limited=mode==1,shared=mode==2;
        if(shared && !shared_cases)continue;
        if(std::getenv("MFQ_TEST_RESIDENCY_TRACE"))
            std::cerr<<"fence_fixture format="<<format<<" mode="<<mode<<std::endl;
        if(shared && !execution.config.moe_direct_ram)continue;
        execution.config.moe_host_physical_bytes=limited?1024:0;
        const int count=mode?2:33;
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> original={source(root/(format+"-640-2560.mfq")),
            source(root/(format+"-640-2560.mfq")),source(root/(format+"-2560-640.mfq"))};
        std::size_t bundle_bytes=0;for(const auto& s:original)bundle_bytes+=bytes(s->metadata()->pools[0]);
        execution.config.moe_host_cache_bytes=3*count*bundle_bytes;
        auto cache=make_moe_expert_cache(count*bundle_bytes,execution.config);
        std::vector<std::array<std::shared_ptr<MoeQuantRangeSource>,3>> sources(count);
        std::vector<std::array<std::shared_ptr<MfeWeight>,3>> weights(count);
        std::vector<MoeFfnForward> forwards;
        for(int layer=0;layer<count;++layer) {
            for(int i=0;i<3;++i) {
                sources[layer][i]=source(root/(format+(i==2?"-2560-640.mfq":"-640-2560.mfq")));
                weights[layer][i]=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,
                    std::to_string(layer)+"/"+std::to_string(i),sources[layer][i],1,layer,std::to_string(i)));
            }
            forwards.push_back(make_moe_ffn_pipeline({weights[layer][0],weights[layer][1]},{weights[layer][2]}));
        }
        finalize_moe_expert_cache(cache);
        if(shared) {
            if(std::getenv("MFQ_TEST_RESIDENCY_TRACE"))std::cerr<<"fence_shared before_forward"<<std::endl;
            std::vector<float> values(2560);for(int j=0;j<2560;++j)values[j]=std::sin(float(j)*.013f);
            auto x=tb::tensor(values).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA);
            // The shared-backup case exchanges `count` expert bundles. Size
            // the decode staging packet for that many routes; a one-route
            // packet now intentionally reserves only one bundle's storage.
            auto ids=tb::zeros({1,count},x.options().dtype(tb::kInt32));
            auto route_weights=tb::ones({1,count},x.options())/count;
            auto reference=forwards[0](execution,x,ids,route_weights).cpu().contiguous();
            if(std::getenv("MFQ_TEST_RESIDENCY_TRACE"))std::cerr<<"fence_shared after_forward"<<std::endl;
            MoeResidencyManager failed(cache.get(),[](const char* phase,std::size_t) {
                if(std::string(phase)=="publish")throw std::runtime_error("shared backup publication failure");
            });
            for(int layer=0;layer<count;++layer)failed.after_layer(layer,std::vector<int32_t>(35,1),35);
            bool rejected=false;
            try {failed.apply_pending();}catch(const std::runtime_error& e) {
                rejected=std::string(e.what())=="shared backup publication failure";
            }
            if(!rejected || !failed.stats().shared_backup_bytes)
                throw std::runtime_error("shared residency backup rollback was not exercised");
            if(failed.prepare_during_window()) {
                std::atomic_bool prepared=false;
                {
                    MoeResidencyManager cancelled(cache.get(),[&](const char* phase,std::size_t index) {
                        if(std::string(phase)=="upload" && index==std::size_t(count*3-1))
                            prepared.store(true,std::memory_order_release);
                    });
                    for(int layer=0;layer<count;++layer)
                        cancelled.after_layer(layer,std::vector<int32_t>(35,1),35,true);
                    wait_preparation(prepared);
                    if(cancelled.stats().committed_projections)
                        throw std::runtime_error("unfinished window published its residency transaction");
                    // Destruction must wake a deferred publisher and restore
                    // every touched GPU field without exchanging RAM leases.
                }
                ++window_cases;
            }
            auto actual=forwards[0](execution,x,ids,route_weights).cpu().contiguous();
            if(std::memcmp(actual.data_ptr(),reference.data_ptr(),actual.numel()*actual.element_size()))
                throw std::runtime_error("shared residency backup corrupted GPU weights");
            for(int layer=0;layer<count;++layer)for(int i=0;i<3;++i) {
                if(!sources[layer][i]->gpu_resident(0) || sources[layer][i]->gpu_resident(1))
                    throw std::runtime_error("shared backup failure changed tier ownership");
                const int width=i==2?640:2560,output=i==2?2560:640;
                auto input=tb::tensor(std::vector<float>(values.begin(),values.begin()+width)).reshape({1,width});
                auto lease=sources[layer][i]->acquire_expert(1);
                auto a=linear(lease->weights,output,width).forward(execution,input).to(tb::kFloat32).contiguous();
                auto b=linear(original[i]->read_expert(1),output,width).forward(execution,input).to(tb::kFloat32).contiguous();
                if(std::memcmp(a.data_ptr(),b.data_ptr(),a.numel()*sizeof(float)))
                    throw std::runtime_error("shared backup failure changed RAM weights");
            }
        }
        auto memory=limited?MoeResidencyManager::MemoryProbe([]{return MoeResidencyManager::MemorySample{1000,1ull<<30};})
            :MoeResidencyManager::MemoryProbe{};
        std::atomic_bool prepared=false,copies_complete=false;
        const auto observation_start=std::chrono::steady_clock::now();
        MoeResidencyManager manager(cache.get(),[&](const char* phase,std::size_t index) {
            if(std::getenv("MFQ_TEST_RESIDENCY_TRACE") && (index==0 || std::string(phase)=="copies_complete"))
                std::cerr<<"fence_probe phase="<<phase<<" ms="<<std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-observation_start).count()<<std::endl;
            if(std::string(phase)=="copies_complete")copies_complete.store(true,std::memory_order_release);
            if(std::string(phase)=="upload" && index==std::size_t(count*3-1))
                prepared.store(true,std::memory_order_release);
        },std::move(memory));
        const bool in_window=manager.prepare_during_window() && !limited;
        const char* fence_option=std::getenv("MFQ_MOE_RESIDENCY_EXPERT_FENCE");
        std::unique_ptr<TailGate> tail;
        if(in_window && fence_option && fence_option[0]=='1') {
            tail=std::make_unique<TailGate>(manager,count-1);tail->start();
        }
        for(int layer=0;layer<count;++layer)manager.after_layer(layer,std::vector<int32_t>(35,1),35,in_window);
        if(in_window) {
            wait_preparation(prepared);
            if(tail) {
                wait_copies(copies_complete);
                if(tail->rescued || manager.stats().expert_fence_waits!=1 || cudaStreamQuery(mfq_current_cuda_stream())!=cudaErrorNotReady)
                    throw std::runtime_error("expert-fence test did not copy beside the blocked output tail; rescued="+
                        std::to_string(tail->rescued.load())+" fence_waits="+std::to_string(manager.stats().expert_fence_waits));
                ++fence_cases;
            }
            if(manager.stats().committed_projections || manager.stats().window_prepares!=1)
                throw std::runtime_error("window preparation crossed its publication barrier");
            for(int layer=0;layer<count;++layer)for(int p=0;p<3;++p)
                if(!sources[layer][p]->gpu_resident(0) || sources[layer][p]->gpu_resident(1))
                    throw std::runtime_error("window preparation changed expert ownership before graph finish");
            if(tail){tail->tail_done();MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));}
            manager.finish_window(count-1);++window_cases;
        }
        if(manager.stats().async_publish && !limited) {
            // Publication must progress without a call on the model thread.
            // The remaining checks verify both GPU ownership and RAM bytes.
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            while(manager.stats().committed_bundles!=std::uint64_t(count)) {
                if(std::chrono::steady_clock::now()>deadline)
                    throw std::runtime_error("residency publication still requires the model thread");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ++async_cases;
        }
        // A predictor may run before the next backbone layer zero. Its entry
        // must finish the pending ownership transaction and release staging.
        manager.before_layer(-1);const auto stats=manager.stats();
        if(stats.batched && !limited && !tail && (stats.backup_copies!=1 || stats.batch_descriptor_copies!=1 ||
                !stats.batch_device_bytes || !stats.upload_copies))
            throw std::runtime_error("warm residency did not batch the complete backup and field descriptors");
        if(stats.observed_routes!=std::uint64_t(count*35) || stats.scheduled_rounds!=1)
            throw std::runtime_error("warm residency did not observe the complete prefill");
        if(limited) {
            if(stats.committed_bundles || stats.planned_bundles || stats.memory_rejections!=std::uint64_t(count))
                throw std::runtime_error("warm residency exceeded injected physical memory bound");
            if(!stats.staging_budget_skips || stats.staging_budget_skips!=stats.projection_memory_rejections)
                throw std::runtime_error("insufficient staging still prepared RAM leases and GPU replacements");
        } else if(stats.committed_bundles!=std::uint64_t(count) || stats.planned_bundles!=std::uint64_t(count))
            throw std::runtime_error("warm residency retained an arbitrary expert count cap");
        if(shared && (!stats.shared_backup_bytes || stats.backup_peak_bytes!=stats.shared_backup_bytes))
            throw std::runtime_error("warm residency did not reuse the pinned transfer workspace");
        for(int layer=0;layer<count;++layer)for(int i=0;i<3;++i) {
            if(sources[layer][i]->gpu_resident(1)==limited || sources[layer][i]->gpu_resident(0)!=limited ||
                sources[layer][i]->expert_disk_reads_after_preload())
                throw std::runtime_error("warm residency publication or disk sealing changed");
            if(!limited) {
                const int width=i==2?640:2560,output=i==2?2560:640;
                std::vector<float> input_values(width);for(int j=0;j<width;++j)input_values[j]=std::sin(float(j)*.013f);
                auto input=tb::tensor(input_values).reshape({1,width});
                auto lease=sources[layer][i]->acquire_expert(0);
                auto actual=linear(lease->weights,output,width).forward(execution,input).to(tb::kFloat32).contiguous();
                auto expected=linear(original[i]->read_expert(0),output,width).forward(execution,input).to(tb::kFloat32).contiguous();
                if(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*sizeof(float)))
                    throw std::runtime_error("warm residency changed demoted RAM expert arithmetic");
            }
        }
        if(!limited && execution.config.moe_direct_ram && (stats.host_pack_bytes || !stats.direct_upload_bytes))
            throw std::runtime_error("warm residency did not use registered RAM directly");
        if(stats.batched && !limited) {
            if(tail) {prepared=false;copies_complete=false;tail->start(true);}
            for(int layer=0;layer<count;++layer)
                manager.after_layer(layer,layer?std::vector<int32_t>{}:std::vector<int32_t>(35,2),1,bool(tail));
            if(tail) {
                // The mapped completion was published by the previous replay.
                // The new transaction must wait for this replay's boundary.
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                if(copies_complete.load(std::memory_order_acquire))
                    throw std::runtime_error("expert fence reused the previous graph replay's completion");
                tail->before_done();wait_copies(copies_complete);
                if(tail->rescued)throw std::runtime_error("expert fence needed the output tail watchdog");
                if(manager.stats().committed_bundles!=stats.committed_bundles)
                    throw std::runtime_error("expert fence published RAM ownership before window finish");
                tail->tail_done();MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
                manager.finish_window(count-1);++fence_cases;
            }
            manager.apply_pending();const auto next=manager.stats();
            if(!tail && (!next.batch_device_bytes || next.batch_device_bytes>=stats.batch_device_bytes))
                throw std::runtime_error("decode retained the larger prefill device copy workspace");
            for(int p=0;p<3;++p) {
                if(!sources[0][p]->gpu_resident(2) || sources[0][p]->gpu_resident(1))
                    throw std::runtime_error("workspace resize changed incoming expert ownership");
                const int width=p==2?640:2560,output=p==2?2560:640;
                auto input=tb::ones({1,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
                auto lease=sources[0][p]->acquire_expert(1);
                auto actual=linear(lease->weights,output,width).forward(execution,input).contiguous();
                auto expected=linear(original[p]->read_expert(1),output,width).forward(execution,input).contiguous();
                if(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                    throw std::runtime_error("workspace resize changed the demoted RAM expert bytes");
            }
            if(!tail)++phase_cases;
        }
        ++cases;
    }
    std::cout<<"residency_workspace_phase_cases="<<phase_cases<<" changing copy workspace/ownership/demoted RAM exact PASS\n";
    std::cout<<"residency_async_publish_cases="<<async_cases<<" worker publishes GPU/RAM transaction before model-thread join PASS\n";
    std::cout<<"residency_window_prepare_cases="<<window_cases<<" prepare before graph finish/publication barrier/cancellation rollback PASS\n";
    std::cout<<"residency_expert_fence_cases="<<fence_cases<<" copies before blocked tail/replay publication reset/deferred RAM publication PASS\n";
    std::cout<<"warm_residency_cases="<<cases<<" 33bundles/singleprefill/physicalbudget/shared-backup-rollback/RAMexact PASS\n";
    return cases;
}
int verify_projection_heat(const std::filesystem::path& root,bool transfer_cached=false) {
    int cases=0;
    const std::array<std::array<std::string,3>,2> formats={
        std::array<std::string,3>{"nint5","nvq3j-512","nint5"},
        std::array<std::string,3>{"nint5","nint6","nvq3j-512"}};
    for(const std::string phase:{"disabled","none","upload","map","host","publish","late-reader"}) {
        CudaExecutionContext execution;execution.config.moe_residency_warm=true;
        execution.config.moe_residency_projection_heat=phase!="disabled";
        execution.config.moe_ram_pcie_fraction=1.0;
        if(transfer_cached) {
            execution.config.moe_two_stage_ffn=true;
            execution.config.moe_direct_ram=true;
        }
        std::array<std::array<std::shared_ptr<MoeQuantRangeSource>,3>,2> sources;
        std::array<std::array<std::shared_ptr<MfeWeight>,3>,2> weights;
        std::array<MoeFfnForward,2> forwards;
        std::array<MfeWeight,3> baseline;
        std::size_t gpu_bytes=0;
        for(int layer=0;layer<2;++layer)for(int p=0;p<3;++p) {
            const auto path=root/(formats[layer][p]+(p==2?"-2560-640.mfq":"-640-2560.mfq"));
            sources[layer][p]=source(path);gpu_bytes+=bytes(sources[layer][p]->metadata()->pools[0]);
            if(!layer) {
                auto model=mfq::open_model_source(path.string());
                baseline[p]=stage_cpu_mixed_moe(make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),false),execution.config);
            }
        }
        execution.config.moe_host_cache_bytes=3*gpu_bytes;
        auto cache=make_moe_expert_cache(gpu_bytes,execution.config);
        for(int layer=0;layer<2;++layer) {
            for(int p=0;p<3;++p)weights[layer][p]=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,
                std::to_string(layer)+"/"+std::to_string(p),sources[layer][p],1,layer,std::to_string(p)));
            forwards[layer]=make_moe_ffn_pipeline({weights[layer][0],weights[layer][1]},{weights[layer][2]});
        }
        finalize_moe_expert_cache(cache);
        for(const auto& layer:sources)for(const auto& s:layer)
            if(!s->gpu_resident(0) || s->gpu_resident(1))throw std::runtime_error("projection-heat initial tier differs");
        const auto host_before=sources[0][0]->host_cache()->stats();
        std::vector<float> values(2560);for(int i=0;i<2560;++i)values[i]=std::sin(float(i)*.023f)*.3f;
        auto x=tb::tensor(values).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA);
        auto ids=tb::ones({1,1},x.options().dtype(tb::kInt32)),route_weights=tb::ones({1,1},x.options().dtype(tb::kFloat32));
        tb::Tensor actual;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{actual=forwards[0](execution,x,ids,route_weights);},[]{},[&]{actual={};});
        const auto verify_gpu=[&] {
            auto route=build_moe_route_plan(ids,3);
            auto g=baseline[0].forward(execution,x,route),u=baseline[1].forward(execution,x,route);
            auto expected=baseline[2].forward(execution,(g*tb::sigmoid(g))*u,route).reshape({1,2560}).cpu().to(tb::kFloat32).contiguous();
            window.run();auto output=actual.cpu().to(tb::kFloat32).contiguous();
            if(std::memcmp(output.data_ptr(),expected.data_ptr(),output.numel()*sizeof(float)))
                throw std::runtime_error("mixed partial projection GPU graph changed canonical output bytes");
        };
        verify_gpu();
        if(transfer_cached) {
            verify_gpu();
            std::ostringstream counters;print_moe_expert_cache_stats(cache,counters);
            std::smatch match;const auto text=counters.str();
            if(!std::regex_search(text,match,std::regex("pipeline_transfer_cache_hits=([0-9]+)")) ||
                    std::stoull(match[1])<3)
                throw std::runtime_error("promotion fixture did not retain all three cold projections on GPU");
            // Release the previous upload's RAM leases while retaining the
            // transfer-cache slots. The exchange must use those exact bytes.
            ids.copy_(tb::zeros({1,1},ids.options()));verify_gpu();
            ids.copy_(tb::ones({1,1},ids.options()));
        }
        MoeHostExpertCache::Lease reader;
        MoeResidencyManager manager(cache.get(),[&](const char* event,std::size_t index) {
            if(index)return;
            if(phase==event)throw std::runtime_error("injected projection exchange failure");
            if(phase=="late-reader" && std::string(event)=="upload")reader=sources[0][0]->acquire_expert(1);
        });
        std::vector<int32_t> heat(120,0);std::fill(heat.begin()+80,heat.end(),1);
        manager.after_layer(0,heat,120);manager.after_layer(1,{},120);
        bool rejected=false;
        try {manager.apply_pending();}catch(const std::runtime_error& e) {
            if(std::string(e.what())!="injected projection exchange failure")throw;rejected=true;
        }
        const auto stats=manager.stats();const bool changed=phase=="none";
        if(transfer_cached && changed) {
            const auto* mode=std::getenv("MFQ_MOE_RESIDENCY_PROMOTE_TRANSFER");
            const bool reuse=mode && mode[0]=='2';
            if((stats.cached_copy_bytes>0)!=reuse || (reuse && stats.direct_upload_bytes))
                throw std::runtime_error("cached promotion did not use the requested RAM/GPU copy source");
            const auto* refill=std::getenv("MFQ_MOE_RESIDENCY_CACHED_REFILL");
            if(reuse && refill && refill[0]=='1' && !stats.cached_copy_batches)
                throw std::runtime_error("cached promotion did not exercise the batched GPU refill");
        }
        if((stats.committed_projections!=0)!=changed || stats.committed_bundles ||
            (phase=="disabled" ? stats.candidate_projections!=0 : stats.planned_projections!=1))
            throw std::runtime_error("projection heat did not select exactly one compatible cold projection");
        if((phase=="upload" || phase=="map" || phase=="host" || phase=="publish")!=rejected)
            throw std::runtime_error("projection heat failure was not propagated");
        if(phase=="late-reader" && !reader)throw std::runtime_error("projection heat late lease missing");
        for(int layer=0;layer<2;++layer)for(int p=0;p<3;++p) {
            if(sources[layer][p]->gpu_resident(0)!=(changed && layer==1 && p==0?false:true) ||
                sources[layer][p]->gpu_resident(1)!=(changed && layer==0 && p==0) ||
                sources[layer][p]->expert_disk_reads_after_preload())
                throw std::runtime_error("projection heat changed an unrelated projection or read SSD");
        }
        const int layer=changed?1:0,expert=changed?0:1;
        auto lease=sources[layer][0]->acquire_expert(expert);
        auto original=source(root/(formats[layer][0]+"-640-2560.mfq"));
        auto cpu=x.cpu().to(tb::kFloat32);
        auto a=linear(lease->weights,640,2560).forward(execution,cpu).to(tb::kFloat32).contiguous();
        auto b=linear(original->read_expert(expert),640,2560).forward(execution,cpu).to(tb::kFloat32).contiguous();
        if(std::memcmp(a.data_ptr(),b.data_ptr(),a.numel()*sizeof(float)))throw std::runtime_error("partial projection RAM transaction changed bytes");
        lease.reset();reader.reset();
        const auto host_after=sources[0][0]->host_cache()->stats();
        if(host_before.managed_bytes!=host_after.managed_bytes || host_before.managed_peak_bytes!=host_after.managed_peak_bytes)
            throw std::runtime_error("partial projection transaction duplicated RAM residency");
        for(int step=0;step<3;++step) {
            x.copy_(tb::tensor(values).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA)*(1.0+step*.25));
            ids.copy_(tb::tensor(std::vector<int32_t>{step%2}).reshape({1,1}).to(tb::kCUDA));verify_gpu();
        }
        ++cases;
    }
    std::cout<<(transfer_cached?"transfer_promotion_cases=":"projection_heat_cases=")<<cases<<" incompatible GU/Down triples, one-projection cross-layer swap, RAM bytes, original GPU graph bytes and upload/map/host/publish/late-lease rollback exact PASS\n";
    return cases;
}
int verify_rounded_swiglu() {
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();
    auto table=moe_swiglu_sigmoid_table_cuda();
    context->begin_graph_pool(stream);
    int cases=0;
    std::vector<std::vector<int64_t>> shapes;
    for(int tokens:{1,3,8,35})for(int routes:{1,10})for(int width:{640,641})
        shapes.push_back({tokens,routes,width});
    shapes.push_back({1,1,65536});shapes.push_back({1,2,32768});
    for(const auto& shape:shapes) {
        auto gate=tb::zeros(shape,table.options()),up=tb::zeros(shape,table.options());
        tb::Tensor result;mfq::cuda::DecodeWindow window(stream);
        window.capture([&]{result=moe_swiglu_rounded_cuda(gate,up,table);},[]{},[&]{result={};});
        for(int step=0;step<2;++step) {
            const bool exhaustive=gate.numel()==65536;
            auto cpu_gate=tb::empty(shape,tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat16));
            auto cpu_up=tb::empty(shape,cpu_gate.options());
            auto* g=reinterpret_cast<uint16_t*>(cpu_gate.data_ptr());
            auto* u=reinterpret_cast<uint16_t*>(cpu_up.data_ptr());
            const uint16_t special[8]={0,0x8000,0x3c00,0xbc00,0x3555,0x0400,0x7bff,0x7c00};
            for(int64_t i=0;i<gate.numel();++i) {
                if(exhaustive)g[i]=static_cast<uint16_t>(i);
                else {
                    const mfq_half value(std::sin(float(i+step*17)*.071f)*12.0f);
                    std::memcpy(g+i,&value,sizeof(value));
                    if(i%127==0)g[i]=special[(i/127+step)%8];
                }
                u[i]=step==0?0x3c00:special[(i+step)%8];
            }
            gate.copy_(cpu_gate.to(tb::kCUDA));up.copy_(cpu_up.to(tb::kCUDA));
            auto joined=tb::cat({gate,up},-1);
            auto old_gate=joined.narrow(-1,0,shape[2]),old_up=joined.narrow(-1,shape[2],shape[2]);
            auto expected=((old_gate*tb::sigmoid(old_gate))*old_up).cpu().contiguous();
            for(bool graph:{false,true}) {
                tb::Tensor output;
                if(graph){window.run();output=result;}
                else output=moe_swiglu_rounded_cuda(gate,up,table);
                auto actual=output.cpu().contiguous();
                if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type() ||
                    std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                    throw std::runtime_error("rounded SwiGLU changed concatenated sigmoid/multiply Half bits");
                ++cases;
            }
        }
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    context->end_graph_pool(stream);
    std::cout<<"rounded_swiglu_cases="<<cases<<" all65536Half gate patterns, contiguous/strided sigmoid,1/3/8/35tokens,1/10routes,640/641width, signed-zero/inf/nan and changing ordinary/graph outputs exact PASS\n";
    return cases;
}
int verify_resident_classification() {
    CudaExecutionContext execution;
    constexpr int experts=512;
    const std::array<int,3> counts={3,2,4};
    const int maps=counts[0]+counts[1]+counts[2];
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32);
    auto kind=tb::empty({experts},options),cancelled=tb::empty({1},options);
    std::vector<std::array<tb::Tensor,6>> fields(maps);
    std::vector<int64_t> pointers;
    for(auto& map:fields)for(auto& field:map) {
        field=tb::full({experts},-1,options);
        pointers.push_back(reinterpret_cast<int64_t>(field.data_ptr()));
    }
    auto dispatch=tb::tensor(pointers).to(tb::kCUDA);
    const auto stream=mfq_current_cuda_stream();
    const auto run=[&] {
        mfq::cuda::moe_classify_resident(reinterpret_cast<const uint64_t*>(dispatch.data_ptr<int64_t>()),
            counts[0],counts[1],counts[2],experts,kind.data_ptr<int32_t>(),
            reinterpret_cast<uint32_t*>(cancelled.data_ptr<int32_t>()),stream);
        mfq::cuda::moe_update_dispatch_maps(reinterpret_cast<const uint64_t*>(dispatch.data_ptr<int64_t>()),
            maps,experts,kind.data_ptr<int32_t>(),reinterpret_cast<const uint32_t*>(cancelled.data_ptr<int32_t>()),stream);
        MFQ_CUDA_CHECK(cudaGetLastError());
    };
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    context->begin_graph_pool(stream);
    mfq::cuda::DecodeWindow window(stream);window.capture(run,[]{},[]{});
    int cases=0;
    for(int step=0;step<8;++step) {
        std::vector<std::vector<int32_t>> slots(maps,std::vector<int32_t>(experts,-1));
        int offset=0;
        for(int projection=0;projection<3;++projection) {
            for(int e=0;e<experts;++e) {
                const bool resident=step==0 || (step!=1 && (e+step+projection)%5!=0);
                if(resident)slots[offset+(e+step)%counts[projection]][e]=(e+step)%17;
            }
            offset+=counts[projection];
        }
        for(int map=0;map<maps;++map)fields[map][0].copy_(tb::tensor(slots[map]).to(tb::kCUDA));
        std::vector<int32_t> expected(experts,1);offset=0;
        for(int projection=0;projection<3;++projection) {
            for(int e=0;e<experts;++e) {
                bool present=false;
                for(int map=offset;map<offset+counts[projection];++map)present|=slots[map][e]>=0;
                expected[e]&=present;
            }
            offset+=counts[projection];
        }
        for(bool graph:{false,true}) {
            cancelled.fill_(1);if(graph)window.run();else run();
            auto actual=kind.cpu().contiguous();
            if(std::memcmp(actual.data_ptr(),expected.data(),experts*sizeof(int32_t)) ||
                cancelled.cpu().data_ptr<int32_t>()[0]!=0)
                throw std::runtime_error("GPU resident classification differs from independent projection intersection");
            for(int map=0;map<maps;++map)for(int field=3;field<6;++field) {
                auto output=fields[map][field].cpu().contiguous();
                for(int e=0;e<experts;++e)if(output.data_ptr<int32_t>()[e]!=
                    (field==3 && expected[e]?slots[map][e]:-1))
                    throw std::runtime_error("GPU resident dispatch read or enabled a cold projection");
            }
            ++cases;
        }
    }
    context->end_graph_pool(stream);
    std::cout<<"resident_classification_cases="<<cases<<" 512experts,3/2/4format cohorts, all/empty/partial/changing projection ownership, ordinary and graph CPU oracle exact PASS\n";
    return cases;
}
int verify_active_nvq(const std::filesystem::path& root) {
    CudaExecutionContext execution;
    const auto same=[](const tb::Tensor& a,const tb::Tensor& b) {
        auto x=a.to(tb::kCPU).contiguous(),y=b.to(tb::kCPU).contiguous();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
            std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size()))
            throw std::runtime_error("active NVQ differs from original heterogeneous kernel bits");
    };
    const auto stream=mfq_current_cuda_stream();
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    context->begin_graph_pool(stream);int cases=0;
    const std::array<std::string,8> formats={"nvq1-s","nvq1-l","nvq2j","nvq2j-l",
        "nvq2j-xl","nvq3j","nvq3j-512","nvq3j-l"};
    for(const auto shape:{std::pair<int,int>{640,2560},{2560,640}})for(bool routed:{false,true}) {
        MixedMoeRuntime actual;actual.n_experts=512;actual.out_per_expert=shape.first;
        actual.neuron_len=shape.second;actual.partial_experts=true;
        int group=0;
        for(const auto& format:formats) {
            auto model=mfq::open_model_source((root/(format+"-"+std::to_string(shape.first)+"-"+
                std::to_string(shape.second)+".mfq")).string());
            auto single=make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),true,execution.config);
            for(auto& pool:single->pools) {
                std::vector<int32_t> slots(512,-1);
                for(int e=0;e<3;++e)slots[group*3+e]=e;
                pool.expert_local=tb::tensor(slots).to(tb::kCUDA);actual.pools.push_back(std::move(pool));
            }
            ++group;
        }
        initialize_mixed_nvq_dispatch(actual,execution.config);auto original=actual;
        for(int tokens:{1,3,8}) {
            auto input=tb::zeros(routed?std::vector<int64_t>{tokens,10,shape.second}:
                std::vector<int64_t>{tokens,shape.second},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
            auto ids=tb::zeros({tokens,10},input.options().dtype(tb::kInt32));
            const auto forward=[&](MixedMoeRuntime& runtime) {
                MoeRoutePlan route;route.ids=ids;route.n_experts=512;
                return runtime.forward(execution.config,execution.kl_mmq,true,false,input,route);
            };
            tb::Tensor result;mfq::cuda::DecodeWindow window(stream);
            execution.config.moe_nvq_active_decode=true;
            window.capture([&]{result=forward(actual);},[]{},[&]{result={};});
            for(int step=0;step<4;++step) {
                std::vector<float> values(input.numel());
                for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+step*19)*.071f)*.13f;
                input.copy_(tb::tensor(values).reshape(input.sizes().vec()).to(tb::kFloat16).to(tb::kCUDA));
                std::vector<int32_t> selected(tokens*10);
                const int pattern[10]={0,3,6,9,12,15,18,21,0,511};
                for(int i=0;i<tokens*10;++i) {
                    int expert=pattern[(i+step)%10];
                    if(step==1)expert=511;
                    if(step==2 && i%10==7)expert=-1;
                    if(step==2 && i%10==8)expert=512;
                    if(step==3 && expert<24)++expert;
                    selected[i]=expert;
                }
                ids.copy_(tb::tensor(selected).reshape({tokens,10}).to(tb::kCUDA));
                for(size_t p=0;p<actual.pools.size();++p) {
                    std::vector<int32_t> slots(512,-1);
                    for(int e=0;e<3;++e)if(step!=3 || e!=0)slots[int(p)*3+e]=e;
                    actual.pools[p].expert_local.copy_(tb::tensor(slots).to(tb::kCUDA));
                }
                execution.config.moe_nvq_active_decode=false;auto expected=forward(original);
                execution.config.moe_nvq_active_decode=true;window.run();same(result,expected);
                auto plan=actual.nvq_active_plans.at(tokens*10).cpu().contiguous();
                const auto* data=plan.data_ptr<int32_t>();int count=0;
                for(int pair=0;pair<tokens*10;++pair) {
                    const int expert=selected[pair];
                    const bool valid=expert>=0 && expert<int(formats.size())*3 && (step!=3 || expert%3!=0);
                    if(valid) {
                        if(data[1+2*tokens*10+count]!=pair || data[1+pair]!=expert/3 ||
                                data[1+tokens*10+pair]!=expert%3)
                            throw std::runtime_error("active NVQ plan differs from CPU route/slot oracle");
                        ++count;
                    }
                }
                if(data[0]!=count)throw std::runtime_error("active NVQ plan count differs from CPU oracle");
                ++cases;
            }
            MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    context->end_graph_pool(stream);
    std::cout<<"active_nvq_cases="<<cases<<" active_nvq_formats="<<formats.size()<<" actual640/2560, shared/routed input,1/3/8tokens, repeated/invalid/empty routes, changing ownership and original heterogeneous kernel bytes exact PASS\n";
    return cases;
}
#include "moe_ffn_reference_test.h"
int verify_two_stage_shared(const std::filesystem::path& root,bool transfer_cache=false,bool comparison=false,
        bool compare_input_views=false,bool compare_default_math=false,bool compare_e8_narrow=false,
        bool compare_resident_overlap=false,bool compare_dense=false,bool compare_nint_whole=false,
        bool compare_token_groups=false,bool compare_shared_stream=false,bool compare_group_quant=false,bool compare_shared_short=false,
        bool compare_cached_reduce=false,bool compare_shared_early=false) {
    const bool compare_math=compare_default_math || compare_e8_narrow || compare_resident_overlap || compare_dense || compare_nint_whole || compare_token_groups || compare_shared_stream || compare_group_quant || compare_shared_short || compare_cached_reduce || compare_shared_early;
    struct MathEnvironment {
        const char* option;
        bool enabled;
        std::string previous;
        void set(const char* value)const {
#ifdef _WIN32
            _putenv_s(option,value);
#else
            if(*value)setenv(option,value,1);else unsetenv(option);
#endif
        }
        explicit MathEnvironment(bool active,const char* name):option(name),enabled(active) {
            if(active)if(const auto* value=std::getenv(option))previous=value;
        }
        void select(bool candidate)const {if(enabled)set(candidate?"1":"0");}
        ~MathEnvironment(){if(enabled)set(previous.c_str());}
    } math_environment(compare_math,compare_shared_early?"MFQ_MFE_SHARED_EARLY":compare_cached_reduce?"MFQ_MFE_CACHED_REDUCE":compare_shared_short?"MFQ_MFE_SHARED_SHORT":compare_group_quant?"MFQ_MFE_GU_GROUP_QUANT":compare_shared_stream?"MFQ_MOE_SHARED_STREAM":compare_token_groups?"MFQ_MFE_TOKEN_GROUPS":compare_nint_whole?"MFQ_MFE_NINT_WHOLE":compare_dense?"MFQ_MOE_NVQ_DENSE":compare_resident_overlap?"MFQ_MFE_RESIDENT_PLAN_OVERLAP":
        compare_e8_narrow?"MFQ_MFE_E8_NARROW":"MFQ_MFE_DEFAULT_MATH");
    MathEnvironment early_overlap_environment(compare_shared_early,"MFQ_MFE_RESIDENT_PLAN_OVERLAP");
    if(compare_shared_early)early_overlap_environment.set("1");
    struct RestoreEnvironment {
        std::string previous;
        bool present=false;
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("MFQ_MOE_TRANSFER_CACHE",value);
#else
            if(*value)setenv("MFQ_MOE_TRANSFER_CACHE",value,1);else unsetenv("MFQ_MOE_TRANSFER_CACHE");
#endif
        }
        explicit RestoreEnvironment(bool enabled) {
            if(const auto* value=std::getenv("MFQ_MOE_TRANSFER_CACHE")){previous=value;present=true;}
            set(enabled?"1":"0");
        }
        ~RestoreEnvironment(){set(present?previous.c_str():"");}
    } environment(transfer_cache);
    const std::array<std::string,3> formats={"nint5","nvq3j-l","nvq2j-xl"};
    std::array<std::shared_ptr<const QuantLinear>,3> shared;
    std::array<NintCpu,3> shared_cpu;
    std::unique_ptr<MoeSharedOutputReference> reference;
    for(int p=0;p<3;++p) {
        const int output=p==2?2560:640,width=p==2?640:2560;
        const std::array<const char*,3> roles={"gate","up","down"};
        std::ifstream stream(root/(std::string("shared-s2-")+roles[p]+".nint"),std::ios::binary);
        if(!stream)throw std::runtime_error("missing released shared FFN tensor fixture");
        const std::vector<uint8_t> blob((std::istreambuf_iterator<char>(stream)),std::istreambuf_iterator<char>());
        const auto cpu=unpack_nint(blob);
        shared_cpu[p]=cpu;
        if(cpu.gs!=24 || cpu.bits!=6 || cpu.neuron_len!=width || cpu.out!=output)
            throw std::runtime_error("released shared FFN fixture geometry mismatch");
        QuantLinear weight;weight.nint=to_gpu_nint(cpu);weight.logical_out=output;weight.logical_neuron_len=width;
        shared[p]=std::make_shared<QuantLinear>(std::move(weight));
    }
    const auto counter=[](const std::shared_ptr<MoeExpertCache>& cache,const char* name) {
        std::ostringstream stream;print_moe_expert_cache_stats(cache,stream);
        std::smatch match;const auto text=stream.str();
        if(!std::regex_search(text,match,std::regex(std::string(name)+"=([0-9]+)")))
            throw std::runtime_error("missing two-stage integration counter");
        return std::stoll(match[1]);
    };
    MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    const std::array<MfqCudaStream,2> streams={mfq_get_stream_from_pool(),mfq_get_current_cuda_stream()};
    int cases=0;std::int64_t cpu=0,transfers=0,resident=0,mapped_serves=0,mapped_bytes=0,mapped_overlaps=0,phased_serves=0,dense_conversions=0;
    for(bool graph:{false,true})for(auto gate_type:{tb::kFloat16,tb::kBFloat16,tb::kFloat32})
    for(int tokens:compare_shared_short?std::vector<int>{2,3,4,5,6,8}:(compare_group_quant || compare_cached_reduce || compare_shared_early)?std::vector<int>{1}:compare_token_groups?std::vector<int>{2,3,4,5,6,8,16,32}:
        compare_shared_stream?std::vector<int>{1,2,3,4,5,6,8}:std::vector<int>{1,3,6})for(int routes:{3,10}) {
        std::array<CudaExecutionContext,2> executions;
        std::array<std::shared_ptr<MoeExpertCache>,2> caches;
        std::array<std::array<std::shared_ptr<MoeQuantRangeSource>,3>,2> sources;
        std::array<MoeFfnForward,2> forwards;
        std::array<mfq::MoeDispatchPlan,2> plans;
        std::vector<std::shared_ptr<MfeWeight>> comparison_gu,comparison_down;
        auto shared_gate=[gate_type](CudaExecutionContext&,const tb::Tensor& x) {
            return tb::sigmoid(x.mean(1).unsqueeze(-1).to(gate_type));
        };
        auto original_shared=[shared,shared_gate](CudaExecutionContext& execution,const tb::Tensor& x) {
            auto g=shared[0]->forward(execution,x),u=shared[1]->forward(execution,x);
            return shared_gate(execution,x)*shared[2]->forward(execution,(g*tb::sigmoid(g))*u);
        };
        for(int mode=0;mode<2;++mode) {
            if(compare_dense || compare_nint_whole)math_environment.select(mode==1);
            MfqCudaStreamGuard stream_guard(streams[mode]);
            auto& config=executions[mode].config;
            executions[mode].continuous_batch_cache_serial=compare_token_groups && tokens>8;
            config.moe_two_stage_ffn=mode==1 || compare_input_views || compare_math;
            config.moe_ram_pcie_fraction=transfer_cache && !comparison?1.0:.5;
            if(comparison || compare_cached_reduce || compare_shared_early)config.moe_residency_adapt=false;
            if(transfer_cache)config.moe_direct_ram=true;
            config.moe_preload_all=true;config.moe_assert_resident=true;
            config.moe_pipeline=true;config.moe_ram_pcie=true;
            config.moe_residency_projection_heat=false;
            std::size_t bundle=0,extra=0;
            for(int p=0;p<3;++p) {
                sources[mode][p]=source(root/(formats[p]+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
                const auto size=bytes(sources[mode][p]->metadata()->pools[0]);
                bundle+=size;if(!p)extra=size;
            }
            config.moe_host_cache_bytes=3*bundle;
            caches[mode]=make_moe_expert_cache(bundle+extra,config);
            std::vector<std::shared_ptr<MfeWeight>> gu,down;
            for(int p=0;p<3;++p) {
                auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(caches[mode],
                    std::to_string(p),sources[mode][p],1,0,std::to_string(p)));
                (p==2?down:gu).push_back(weight);
            }
            forwards[mode]=make_moe_ffn_pipeline(gu,down,original_shared,
                [&,mode](const mfq::MoeDispatchPlan& plan){plans[mode]=plan;},
                MoeFfnSharedWeights{shared,shared_gate});
            if(comparison && mode==1){comparison_gu=gu;comparison_down=down;}
            finalize_moe_expert_cache(caches[mode]);
            if(compare_dense)for(int p=0;p<3;++p) {
                const auto count=sources[mode][p]->dense_materializations();
                if((mode==0 && count) || (mode==1 && p>0 && !count))
                    throw std::runtime_error("compact NVQ pipeline omitted expected layout conversion");
                dense_conversions+=count;
            }
            bool partial=false;
            for(int e=0;e<3;++e) {
                int hits=0;for(const auto& s:sources[mode])hits+=s->gpu_resident(e);
                partial=partial || (hits>0 && hits<3);
            }
            if(!partial)throw std::runtime_error("weighted shared fixture lacks partially resident projection");
        }
        auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
        auto input=tb::zeros({tokens,2560},options);
        auto ids=tb::zeros({tokens,routes},options.dtype(tb::kInt32));
        auto weights=tb::zeros({tokens,routes},options.dtype(tb::kFloat32));
        std::int64_t comparison_dma=-1;
        for(int pass=0;pass<(comparison?3:1);++pass) {
        if(comparison) {
            executions[1].config.moe_ffn_transfer_phases=pass==1;
            if(pass) {
                forwards[1]={};
                forwards[1]=make_moe_ffn_pipeline(comparison_gu,comparison_down,original_shared,
                    [&](const mfq::MoeDispatchPlan& plan){plans[1]=plan;},
                    MoeFfnSharedWeights{shared,shared_gate});
            }
            if(!pass) {
                bool rejected=false;
                try {prepare_moe_pipeline_comparison(caches[1],true);}
                catch(const std::logic_error&){rejected=true;}
                if(!rejected)throw std::runtime_error("empty comparison replay was accepted");
            }
            prepare_moe_pipeline_comparison(caches[1],pass>0);
        }
        const auto dma_before=comparison?counter(caches[1],"ram_pcie_bytes"):0;
        tb::Tensor actual;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        math_environment.select(true);
        if(graph)window.capture([&] {
            if(compare_input_views) {
                const auto temporary_input=input.clone(),temporary_ids=ids.clone(),temporary_weights=weights.clone();
                actual=forwards[1](executions[1],temporary_input,temporary_ids,temporary_weights);
            } else actual=forwards[1](executions[1],input,ids,weights);
        },[]{},[&]{actual={};});
        for(int step=0;step<4;++step) {
            for(const auto& cache:caches)finish_moe_expert_exchanges(cache);
            std::vector<float> values(tokens*2560),route_weights(tokens*routes);
            std::vector<int32_t> selected(tokens*routes);
            for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+step*31)*.017f)*.13f;
            for(int i=0;i<tokens*routes;++i) {
                selected[i]=compare_cached_reduce && step==3?0:(i+step)%3;
                route_weights[i]=float((i*7+step*3)%17+1)/37;
            }
            input.copy_(tb::tensor(values).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA));
            ids.copy_(tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA));
            weights.copy_(tb::tensor(route_weights).reshape({tokens,routes}).to(tb::kCUDA));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(streams[1]));
            tb::Tensor expected;
            {MfqCudaStreamGuard stream_guard(streams[0]);
                math_environment.select(false);
                expected=forwards[0](executions[0],input,ids,weights).cpu().contiguous();}
            math_environment.select(true);
            if(graph)window.run();else actual=forwards[1](executions[1],input,ids,weights);
            auto result=actual.cpu().contiguous();
            if(plans[0].kinds!=plans[1].kinds)throw std::runtime_error("two-stage fusion changed CPU/GPU dispatch");
            for(const auto kind:plans[1].kinds) {
                cpu+=kind==mfq::MoeDispatchKind::Cpu;
                transfers+=kind==mfq::MoeDispatchKind::GpuTransfer;
                resident+=kind==mfq::MoeDispatchKind::GpuResident;
            }
            if(result.sizes()!=expected.sizes() || result.scalar_type()!=expected.scalar_type())
                throw std::runtime_error("weighted shared two-stage output shape or dtype mismatch");
            for(const auto& value:{result,expected}) {
                const auto numeric=value.to(tb::kFloat32).contiguous();
                for(int64_t i=0;i<numeric.numel();++i)if(!std::isfinite(numeric.data_ptr<float>()[i]))
                    throw std::runtime_error("weighted shared output is not finite");
            }
            if(std::memcmp(result.data_ptr(),expected.data_ptr(),result.numel()*result.element_size())) {
                if(!reference)reference=std::make_unique<MoeSharedOutputReference>(root,formats,executions[0]);
                reference->verify(executions[0],input,ids,weights,selected,plans[0],shared,shared_cpu,
                    shared_gate(executions[0],input),result,expected);
            }
            for(const auto& mode:sources)for(const auto& s:mode)
                if(s->expert_disk_reads_after_preload())throw std::runtime_error("weighted shared fusion read expert SSD after preload");
            ++cases;
        }
        if(comparison) {
            if(finish_moe_pipeline_comparison(caches[1])!=4)
                throw std::runtime_error("comparison did not retain all original CPU assignments");
            const auto dma=counter(caches[1],"ram_pcie_bytes")-dma_before;
            if(comparison_dma<0)comparison_dma=dma;
            if(dma!=comparison_dma || !dma || counter(caches[1],"residency_committed_projections"))
                throw std::runtime_error("comparison changed weight transfers or primary ownership");
        }
        }
        if(counter(caches[0],"pipeline_two_stage_serves")!=((compare_input_views || compare_math)?4:0) ||
            counter(caches[1],"pipeline_two_stage_serves")!=(comparison?12:4))
            throw std::runtime_error("weighted shared test did not compare old and two-stage serving paths");
        if(compare_resident_overlap && (counter(caches[0],"pipeline_early_gate_up_enabled")!=0 ||
                counter(caches[1],"pipeline_early_gate_up_enabled")!=1))
            throw std::runtime_error("resident overlap comparison did not exercise both dispatch settings");
        if(compare_input_views && (counter(caches[0],"pipeline_window_input_view_serves") ||
                counter(caches[1],"pipeline_window_input_view_serves")!=(graph?4:0)))
            throw std::runtime_error("shared input-view check did not exercise its graph-only addresses");
        if(transfer_cache && (!counter(caches[1],"pipeline_transfer_cache_hits") ||
            !counter(caches[1],"pipeline_transfer_cache_misses") || !counter(caches[1],"pipeline_transfer_cache_saved_bytes")))
            throw std::runtime_error("retained transfer fields were not reused by changing shared FFN graphs");
        if(transfer_cache) {
            const auto* phases=std::getenv("MFQ_MOE_FFN_TRANSFER_PHASES");
            const auto* mapped=std::getenv("MFQ_MOE_MAPPED_COPY");
            const bool enabled_phase=(!phases || std::string(phases)=="1") && !(mapped && std::string(mapped)=="1");
            const auto actual_phases=counter(caches[1],"pipeline_phased_transfer_serves");
            if(actual_phases!=(comparison?4:enabled_phase?4:0) ||
                counter(caches[0],"pipeline_phased_transfer_serves")!=(compare_math?actual_phases:0))
                throw std::runtime_error("phased-transfer native arm was not exercised");
            if((enabled_phase || comparison) && (!counter(caches[1],"pipeline_gate_up_dma_bytes") || !counter(caches[1],"pipeline_down_dma_bytes")))
                throw std::runtime_error("phased-transfer test omitted an original projection payload");
            phased_serves+=actual_phases;
            const auto* enabled=std::getenv("MFQ_MOE_MAPPED_COPY");
            if(enabled && std::string(enabled)=="1") {
                const auto serves=counter(caches[1],"pipeline_mapped_copy_serves");
                const auto bytes=counter(caches[1],"pipeline_mapped_copy_bytes");
                if(serves!=4 || !bytes || counter(caches[0],"pipeline_mapped_copy_serves")!=(compare_math?4:0))
                    throw std::runtime_error("shared graph did not compare DMA with mapped RAM transfer");
                const auto* overlap=std::getenv("MFQ_MOE_MAPPED_COPY_OVERLAP");
                const auto overlap_serves=counter(caches[1],"pipeline_mapped_overlap_serves");
                if(overlap_serves!=(overlap && std::string(overlap)=="1"?4:0))
                    throw std::runtime_error("mapped-copy overlap arm was not exercised");
                mapped_overlaps+=overlap_serves;
                mapped_serves+=serves;mapped_bytes+=bytes;
            }
        }
    }
    if(((!transfer_cache || comparison) && !cpu) || !transfers || !resident)throw std::runtime_error("weighted shared fusion omitted a dispatch tier");
    std::cout<<(compare_nint_whole?"nint_whole_pipeline_cases=":compare_dense?"nvq_dense_pipeline_cases=":compare_resident_overlap?"resident_overlap_pipeline_cases=":compare_e8_narrow?"e8_narrow_pipeline_cases=":compare_default_math?"default_math_pipeline_cases=":compare_input_views?"shared_window_input_view_cases=":transfer_cache?"transfer_cache_shared_cases=":"two_stage_shared_cases=")<<cases<<" cpu_positions="<<cpu<<" transfer_positions="<<transfers
        <<" resident_positions="<<resident<<" mixed G/U/Down, released shared adaptive NINT6 gs24/sub7,"
        <<" Float/Half/BFloat gate, changing weighted graphs and partial VRAM/RAM FP64 error bounds PASS\n";
    if(reference)std::cout<<"weighted_shared_accuracy checks="<<reference->checks
        <<" max_bound_ratio="<<reference->max_bound_ratio<<" quantized_rows="<<reference->quantized_rows
        <<" hidden_checks="<<reference->hidden_checks<<'\n';
    if(comparison)std::cout<<"transfer_comparison_shared_cases="<<cases<<" passes=108 replay_guards=36 equal_dma=1 fixed_primary=1 PASS\n";
    if(phased_serves)std::cout<<"phased_transfer_shared_cases="<<phased_serves<<" PASS\n";
    if(mapped_serves)std::cout<<"mapped_copy_shared_cases="<<mapped_serves<<" mapped_bytes="<<mapped_bytes<<" PASS\n";
    if(mapped_overlaps)std::cout<<"mapped_overlap_shared_cases="<<mapped_overlaps<<" PASS\n";
    if(compare_dense)std::cout<<"nvq_dense_pipeline_conversions="<<dense_conversions<<" PASS\n";
    return cases;
}
#include "../../../../bench/moe_residency_copy_bench.h"
int verify_transfer_hit_reservation(const std::filesystem::path& root) {
    const std::array<MfqCudaStream,2> streams={mfq_get_stream_from_pool(),mfq_get_current_cuda_stream()};
    const auto counter=[](const std::shared_ptr<MoeExpertCache>& cache,const char* name) {
        std::ostringstream stream;print_moe_expert_cache_stats(cache,stream);const auto text=stream.str();
        std::smatch match;
        if(!std::regex_search(text,match,std::regex(std::string(name)+"=([0-9]+)")))
            throw std::runtime_error("missing transfer reservation counter");
        return std::stoll(match[1]);
    };
    struct Environment {
        std::string old;
        Environment(){const auto* p=std::getenv("MFQ_MOE_TRANSFER_CACHE");if(p)old=p;}
        static void set(const char* p) {
#if defined(_WIN32)
            _putenv_s("MFQ_MOE_TRANSFER_CACHE",p);
#else
            if(*p)setenv("MFQ_MOE_TRANSFER_CACHE",p,1);else unsetenv("MFQ_MOE_TRANSFER_CACHE");
#endif
        }
        ~Environment(){set(old.c_str());}
    } environment;
    int cases=0;
    std::int64_t dma_graph_launches=0,dma_graph_fallbacks=0;
    for(const std::string format:{"nint5","nvq3j-512"}) {
        std::array<std::shared_ptr<MoeExpertCache>,2> caches;
        std::array<std::array<MoeFfnForward,2>,2> forwards;
        std::array<CudaExecutionContext,2> executions;
        for(int mode=0;mode<2;++mode) {
            MfqCudaStreamGuard guard(streams[mode]);
            Environment::set(mode?"1":"0");auto& execution=executions[mode];
            execution.config.moe_hybrid_cpu=false;execution.config.moe_direct_ram=true;execution.config.moe_two_stage_ffn=true;
            execution.config.moe_ram_pcie_fraction=1.0;
            auto proto=source(root/(format+"-640-2560.mfq"));
            auto down=source(root/(format+"-2560-640.mfq"));
            const auto bundle_bytes=2*bytes(proto->metadata()->pools[0])+bytes(down->metadata()->pools[0]);
            execution.config.moe_host_cache_bytes=6*bundle_bytes;
            caches[mode]=make_moe_expert_cache(2*bundle_bytes,execution.config);
            for(int layer=0;layer<2;++layer) {
                std::vector<std::shared_ptr<MfeWeight>> gates,downs;
                for(int p=0;p<3;++p) {
                    auto s=source(root/(format+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
                    auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(caches[mode],
                        std::to_string(layer)+"/"+std::to_string(p),s,1,layer,std::to_string(p)));
                    (p==2?downs:gates).push_back(weight);
                }
                forwards[mode][layer]=make_moe_ffn_pipeline(gates,downs);
            }
            finalize_moe_expert_cache(caches[mode]);
        }
        std::vector<float> values(2560);for(int i=0;i<2560;++i)values[i]=std::sin(float(i)*.019f)*.1f;
        auto x=tb::tensor(values).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA);
        const std::array<std::vector<int32_t>,3> routes={std::vector<int32_t>{2},{1,2},{1,2}};
        for(int step=0;step<3;++step) {
            const auto before=counter(caches[1],"pipeline_transfer_cache_hits");
            auto ids=tb::tensor(routes[step]).reshape({1,int64_t(routes[step].size())}).to(tb::kCUDA);
            auto weights=tb::ones(ids.sizes().vec(),x.options().dtype(tb::kFloat32));
            const int layer=step==1?1:0;
            const auto run=[&](int mode){MfqCudaStreamGuard guard(streams[mode]);
                return forwards[mode][layer](executions[mode],x,ids,weights).cpu().contiguous();};
            auto expected=run(0),actual=run(1);
            if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type() ||
                    std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*actual.element_size()))
                throw std::runtime_error("reserved transfer hits changed GPU output bytes");
            if(step==2 && counter(caches[1],"pipeline_transfer_cache_hits")-before!=3)
                throw std::runtime_error("an earlier transfer miss evicted a later route hit, actual_hits="+
                    std::to_string(counter(caches[1],"pipeline_transfer_cache_hits")-before)+
                    " two_stage="+std::to_string(counter(caches[1],"pipeline_two_stage_serves")));
            ++cases;
        }
        dma_graph_launches+=counter(caches[1],"pipeline_dma_graph_launches");
        dma_graph_fallbacks+=counter(caches[1],"pipeline_dma_graph_fallbacks");
        const auto skipped=counter(caches[1],"pipeline_gpu_only_cpu_wait_skips");
        if(bool(skipped)!=mfq::cuda::runtime_options::skip_gpu_only_cpu_wait())
            throw std::runtime_error("GPU-only transfer check did not exercise its CPU-wait option");
        const bool skip_timing=mfq::cuda::runtime_options::skip_fixed_gpu_timing() &&
            !mfq::cuda::runtime_options::moe_diagnostics().dma;
        if(bool(counter(caches[1],"pipeline_fixed_gpu_timing_skips"))!=skip_timing)
            throw std::runtime_error("GPU-only transfer check did not exercise its rate-timing option");
        if(skip_timing && counter(caches[1],"pipeline_gpu_ns"))
            throw std::runtime_error("skipped GPU rate events still contributed timing samples");
    }
    if(mfq::cuda::runtime_options::dma_graph_batch() && !dma_graph_launches)
        throw std::runtime_error("transfer reservation check did not execute DMA copy graphs");
    std::cout<<"transfer_hit_reservation_cases="<<cases<<" two sources sharing full staging arenas, miss-before-hit and original output bytes exact PASS\n";
    if(dma_graph_launches)std::cout<<"transfer_dma_graph_launches="<<dma_graph_launches<<" ordinary_fallbacks="<<dma_graph_fallbacks<<" PASS\n";
    return cases;
}
void verify_early_no_cpu_ready(const std::filesystem::path& root,bool input_views=false,
        bool noncontiguous=false,bool gpu_cold=false,int tokens=1,bool serial_layers=false) {
    const auto counter=[](const std::shared_ptr<MoeExpertCache>& cache,const char* name) {
        std::ostringstream out;print_moe_expert_cache_stats(cache,out);
        std::smatch match;const auto text=out.str();
        if(!std::regex_search(text,match,std::regex(std::string(name)+"=([0-9]+)")))
            throw std::runtime_error("missing CPU-ready counter");
        return std::stoull(match[1]);
    };
    struct Environment {
        const char* name;
        std::string old;
        bool existed=false;
        explicit Environment(const char* key):name(key){if(const auto* value=std::getenv(name)){old=value;existed=true;}}
        void set(const char* value) {
#ifdef _WIN32
            _putenv_s(name,value);
#else
            if(*value)setenv(name,value,1);
            else unsetenv(name);
#endif
        }
        ~Environment(){set(existed?old.c_str():"");}
    } environment(input_views?"MFQ_MFE_WINDOW_INPUT_VIEWS":"MFQ_MOE_EARLY_NO_CPU_READY");
    const bool check_transfer_poll=std::getenv("MFQ_TEST_TRANSFER_POLL")!=nullptr;
    const bool check_kv_direct=std::getenv("MFQ_TEST_KV_DIRECT_READ")!=nullptr;
    const bool check_wire_fallback=std::getenv("MFQ_TEST_WIRE_FALLBACK")!=nullptr;
    const bool adaptive_kv_direct=check_kv_direct && std::getenv("MFQ_TEST_KV_DIRECT_ADAPT");
    const bool trace_kv_direct=std::getenv("MFQ_TRACE_KV_DIRECT_READ")!=nullptr;
    const bool force_direct_off=std::getenv("MFQ_TEST_KV_DIRECT_FORCE_OFF")!=nullptr;
    const int route_count=std::getenv("MFQ_TEST_KV_WIDE_ROUTES")?10:2;
    const bool partial_gu=std::getenv("MFQ_TEST_KV_PARTIAL_GU")!=nullptr;
    const bool speculative_lru=std::getenv("MFQ_TEST_MTP_KV_LRU")!=nullptr;
    Environment notification_environment("MFQ_MOE_POLL_TRANSFER_READY");
    Environment direct_environment("MFQ_MOE_KV_DIRECT_READ");
    Environment lru_environment("MFQ_MOE_MTP_KV_LRU");
    int cases=0,cpu_groups=0,gpu_only=0,cancelled=0;
    const std::array<MfqCudaStream,2> streams={mfq_get_stream_from_pool(),mfq_get_current_cuda_stream()};
    for(const std::string format:{"nint5","nvq3j-512"}) {
        std::array<CudaExecutionContext,2> executions;
        std::array<std::shared_ptr<MoeExpertCache>,2> caches;
        std::array<std::shared_ptr<mfq::cuda::KvExpertLoan>,2> loans;
        std::array<std::size_t,2> loan_bytes{};
        std::array<std::array<std::shared_ptr<MoeQuantRangeSource>,3>,2> sources;
        std::array<MoeFfnForward,2> forwards,tail_forwards;
        std::array<std::array<std::shared_ptr<MoeQuantRangeSource>,3>,2> tail_sources;
        std::array<tb::Tensor,2> outputs;
        std::array<std::unique_ptr<mfq::cuda::DecodeWindow>,2> windows;
        std::array<int,2> observed_cpu{};
        int hot=-1,cold=-1;
        for(int mode=0;mode<2;++mode) {
            MfqCudaStreamGuard guard(streams[mode]);
            auto& execution=executions[mode];auto& cfg=execution.config;
            cfg.moe_preload_all=true;cfg.moe_assert_resident=true;cfg.moe_pipeline=true;
            cfg.moe_hybrid_cpu=true;cfg.moe_direct_ram=true;cfg.moe_two_stage_ffn=true;
            cfg.moe_ram_pcie=true;cfg.moe_ram_pcie_fraction=gpu_cold?1.:0.;cfg.moe_residency_adapt=adaptive_kv_direct;
            if(adaptive_kv_direct)cfg.moe_residency_warm=true;
            std::size_t bundle=0;
            for(int p=0;p<3;++p) {
                sources[mode][p]=source(root/(format+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
                bundle+=bytes(sources[mode][p]->metadata()->pools[0]);
            }
            cfg.moe_host_cache_bytes=(serial_layers?6:3)*bundle;caches[mode]=make_moe_expert_cache(2*bundle,cfg);
            if(check_kv_direct) {
                loan_bytes[mode]=std::max<std::size_t>(8<<20,bundle);
                loans[mode]=std::make_shared<mfq::cuda::KvExpertLoan>();
                loans[mode]->add(tb::zeros({int64_t(loan_bytes[mode])},
                    tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8)),loan_bytes[mode],1,false);
                set_moe_kv_expert_loan(caches[mode],loans[mode]);
            }
            std::vector<std::shared_ptr<MfeWeight>> gu,down;
            for(int p=0;p<3;++p) {
                auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(caches[mode],
                    "early-cpu/"+std::to_string(p),sources[mode][p],1,0,std::to_string(p)));
                (p==2?down:gu).push_back(weight);
            }
            forwards[mode]=make_moe_ffn_pipeline(gu,down,{},[&,mode](const mfq::MoeDispatchPlan& plan) {

                observed_cpu[mode]=int(std::count_if(plan.groups.begin(),plan.groups.end(),[](const auto& group) {
                    return group.kind==mfq::MoeDispatchKind::Cpu;
                }));
            });
            if(serial_layers) {
                std::vector<std::shared_ptr<MfeWeight>> tail_gu,tail_down;
                for(int p=0;p<3;++p) {
                    tail_sources[mode][p]=source(root/(format+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
                    auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(caches[mode],
                        "direct-tail/"+std::to_string(p),tail_sources[mode][p],1,1,std::to_string(p)));
                    (p==2?tail_down:tail_gu).push_back(weight);
                }
                tail_forwards[mode]=make_moe_ffn_pipeline(tail_gu,tail_down);
            }
            finalize_moe_expert_cache(caches[mode]);
            for(int expert=0;expert<3;++expert) {
                const bool all_hot=std::all_of(sources[mode].begin(),sources[mode].end(),[&](const auto& s){return s->gpu_resident(expert);});
                const bool all_cold=std::none_of(sources[mode].begin(),sources[mode].end(),[&](const auto& s){return s->gpu_resident(expert);});
                if(!mode && all_hot)hot=expert;
                if(!mode && all_cold)cold=expert;
            }
        }
        if(hot<0 || cold<0)throw std::runtime_error("CPU-ready test lacks a full GPU and RAM bundle");
        const auto route_tensor=[&](int32_t a,int32_t b) {
            std::vector<int32_t> rows(tokens*route_count);
            for(int i=0;i<tokens;++i)for(int r=0;r<route_count;++r)rows[i*route_count+r]=r&1?b:a;
            return tb::tensor(rows).reshape({tokens,route_count}).to(tb::kCUDA);
        };
        const auto weight_tensor=[&](float a,float b) {
            std::vector<float> rows(tokens*route_count);
            for(int i=0;i<tokens;++i)for(int r=0;r<route_count;++r)rows[i*route_count+r]=(r&1?b:a)/(route_count/2);
            return tb::tensor(rows).reshape({tokens,route_count}).to(tb::kCUDA);
        };
        std::vector<float> values(tokens*2560);for(int i=0;i<tokens*2560;++i)values[i]=std::sin(float(i)*.023f)*.15f;
        auto x=tb::tensor(values).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA);
        auto ids=route_tensor(hot,hot);
        auto weights=weight_tensor(.375f,.625f);
        mfq_cuda_synchronize();
        for(int mode=0;mode<2;++mode) {
            MfqCudaStreamGuard guard(streams[mode]);
            environment.set(mode?"1":"0");
            if(check_transfer_poll)notification_environment.set(mode?"1":"0");
            if(check_kv_direct)direct_environment.set(mode && !force_direct_off?"1":"0");
            windows[mode]=std::make_unique<mfq::cuda::DecodeWindow>(streams[mode].stream());
            windows[mode]->capture([&,mode] {
                if(!input_views) {
                    outputs[mode]=forwards[mode](executions[mode],x,ids,weights);
                    if(serial_layers)outputs[mode]=tail_forwards[mode](executions[mode],moe_weighted_reduce_cuda(outputs[mode],weights).to(tb::kFloat16),ids,weights);
                }
                else {
                    {
                        auto temporary_x=noncontiguous?tb::empty({tokens,2560,2},x.options()).select(2,0):x.clone();
                        auto temporary_ids=noncontiguous?tb::empty({tokens,2,2},ids.options()).select(2,0):ids.clone();
                        auto temporary_weights=noncontiguous?tb::empty({tokens,2,2},weights.options()).select(2,0):weights.clone();
                        if(noncontiguous) {
                            temporary_x.copy_(x);temporary_ids.copy_(ids);temporary_weights.copy_(weights);
                            if(temporary_x.is_contiguous() || temporary_ids.is_contiguous() || temporary_weights.is_contiguous())
                                throw std::runtime_error("window view fallback test lacks strided producers");
                        }
                        outputs[mode]=forwards[mode](executions[mode],temporary_x,temporary_ids,temporary_weights);
                    }
                    // Reuse temporary allocation sizes after the FFN consumers,
                    // exercising graph-pool lifetimes rather than persistent inputs.
                    auto overwrite_x=tb::full({tokens,2560},-.75f,x.options());
                    auto overwrite_ids=tb::full({tokens,2},-123,ids.options());
                    auto overwrite_weights=tb::full({tokens,2},-4.f,weights.options());
                }
            },[]{},[&,mode]{outputs[mode]={};});
            if(adaptive_kv_direct)std::fprintf(stderr,"kv_direct_adapt mode=%d phase=captured\n",mode);
        }
        const std::array<std::array<int32_t,2>,7> routes={{{hot,hot},{cold,cold},{hot,cold},
            {cold,hot},{hot,hot},{cold,cold},{hot,hot}}};
        const char* checked_counter=input_views?"pipeline_window_input_view_serves":"pipeline_early_no_cpu_ready_serves";

        const std::array<uint64_t,2> early_before={counter(caches[0],checked_counter),counter(caches[1],checked_counter)};

        uint64_t expected_early=0;
        for(int repeat=0;repeat<2;++repeat)for(const auto& route:routes) {

            if(input_views) {
                for(int i=0;i<tokens*2560;++i)values[i]=std::sin(float(i+cases*19)*.023f)*(.11f+.01f*repeat);
                x.copy_(tb::tensor(values).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA));
                weights.copy_(weight_tensor(.125f+.025f*cases,.875f-.013f*cases));
            }
            ids.copy_(route_tensor(route[0],route[1]));
            if(speculative_lru) {
                std::vector<int32_t> per_row(tokens*route_count);
                for(int row=0;row<tokens;++row)for(int r=0;r<route_count;++r)
                    per_row[row*route_count+r]=row?route[1]:route[0];
                ids.copy_(tb::tensor(per_row).reshape({tokens,route_count}).to(tb::kCUDA));
            }

            mfq_cuda_synchronize();

            std::array<tb::Tensor,2> actual;
            for(int mode=0;mode<2;++mode) {
                MfqCudaStreamGuard guard(streams[mode]);
                if(check_kv_direct) {
                    direct_environment.set(mode && !force_direct_off?"1":"0");
                    // Force staging misses while retaining the loan cache,
                    // then reuse the filled staging slots in the second pass.
                    if(!repeat && !adaptive_kv_direct)prepare_moe_pipeline_comparison(caches[mode],false);
                    else if(repeat && route==routes.front())loans[mode]->ensure(
                        check_wire_fallback?loan_bytes[mode]:loan_bytes[mode]/2);
                    if(partial_gu && mode && gpu_cold) {
                        // This fixture registers GU/Down as source 0/1/2,
                        // followed by the second layer's 3/4/5. Prior output
                        // copies above have completed the previous readers.
                        const int fields=format=="nint5"?7:4;
                        loans[mode]->retire_expert(2,cold,fields);
                        if(serial_layers)loans[mode]->retire_expert(5,cold,fields);
                        loans[mode]->next_packet();
                    }
                }
                if(adaptive_kv_direct && trace_kv_direct)std::fprintf(stderr,"kv_direct_adapt repeat=%d case=%d mode=%d route=%d,%d phase=replay\n",
                    repeat,cases,mode,route[0],route[1]);

                environment.set(mode?"1":"0");
                if(speculative_lru) {
                    lru_environment.set(mode?"1":"0");begin_moe_speculative_routes(caches[mode],tokens);
                }
                try {
                    windows[mode]->run();
                    if(speculative_lru) {
                        if(cases%5==4)discard_moe_speculative_routes(caches[mode]);
                        else commit_moe_speculative_routes(caches[mode],1+cases%tokens);
                    }
                }catch(...) {
                    if(speculative_lru)discard_moe_speculative_routes(caches[mode]);throw;
                }

                actual[mode]=outputs[mode].cpu().contiguous();
                // Finish this independent cache before the other comparison
                // graph shares its CUDA context with a background allocation.
                if(adaptive_kv_direct)finish_moe_expert_exchanges(caches[mode]);

                if(adaptive_kv_direct && trace_kv_direct)std::fprintf(stderr,"kv_direct_adapt repeat=%d case=%d mode=%d phase=done\n",repeat,cases,mode);
            }
            if(observed_cpu[0]!=observed_cpu[1] || std::memcmp(actual[0].data_ptr(),actual[1].data_ptr(),actual[0].numel()*actual[0].element_size()))
                throw std::runtime_error("early CPU completion changed replayed GPU/CPU output bytes");
            cpu_groups+=observed_cpu[1];gpu_only+=observed_cpu[1]==0;++cases;
            expected_early+=input_views?!noncontiguous:(observed_cpu[1]==0)*(serial_layers?2:1);
        }
        if(counter(caches[0],checked_counter)!=early_before[0] ||
            counter(caches[1],checked_counter)-early_before[1]!=expected_early)
            throw std::runtime_error("window completion/input-view option did not exercise its intended routes");
        if(!gpu_cold && (counter(caches[0],"pipeline_gpu_only_cpu_wait_skips") ||
            counter(caches[1],"pipeline_gpu_only_cpu_wait_skips")))
            throw std::runtime_error("a graph with a CPU quota omitted its CPU completion wait");
        if(!gpu_cold && (counter(caches[0],"pipeline_fixed_gpu_timing_skips") ||
            counter(caches[1],"pipeline_fixed_gpu_timing_skips")))
            throw std::runtime_error("a graph with a CPU quota omitted its rate timing");
        if(check_transfer_poll) {
            const auto baseline=counter(caches[0],"pipeline_transfer_event_polls");
            const auto candidate=counter(caches[1],"pipeline_transfer_event_polls");
            if(baseline || (gpu_cold?!candidate:bool(candidate)))
                throw std::runtime_error("transfer event notifications did not respect GPU-only dispatch");
            std::cout<<"transfer_poll format="<<format<<" gpu_cold="<<gpu_cold
                <<" baseline="<<baseline<<" candidate="<<candidate<<" exact PASS\n";
        }
        if(check_wire_fallback) {
            const auto metric=[](const std::shared_ptr<MoeExpertCache>& cache,const char* key) {
                for(const auto& item:moe_expert_memory_metrics(cache))if(item.first==key)return item.second;
                throw std::runtime_error("wire fallback metric missing");
            };
            if(metric(caches[0],"expert_wire_fallback_cells") ||
                    !metric(caches[1],"expert_wire_fallback_cells") ||
                    !metric(caches[1],"expert_wire_fallback_bytes"))
                throw std::runtime_error("forced KV exhaustion did not execute the direct wire fallback");
            std::cout<<"wire_fallback format="<<format<<" tokens="<<tokens
                <<" bytes="<<metric(caches[1],"expert_wire_fallback_bytes")<<" exact PASS\n";
        }
        if(check_kv_direct) {
            const auto baseline=counter(caches[0],"pipeline_kv_direct_read_bytes");
            const auto candidate=counter(caches[1],"pipeline_kv_direct_read_bytes");
            if(baseline || ((gpu_cold && !force_direct_off)?!candidate:bool(candidate)))
                throw std::runtime_error("direct KV fields did not respect GPU-only window dispatch");
            std::cout<<"kv_direct_read format="<<format<<" tokens="<<tokens<<" gpu_cold="<<gpu_cold
                <<" baseline_bytes="<<baseline<<" candidate_bytes="<<candidate
                <<" loan_hits="<<loans[1]->hits()<<" staged_reuse_and_growth exact PASS\n";
        }
        if(speculative_lru) {
            const auto control=loans[0]->speculative_stats(),candidate=loans[1]->speculative_stats();
            if(control.windows || !candidate.windows || !candidate.kept ||
               !(candidate.reverted_hits+candidate.cold_insertions) || candidate.disabled)
                throw std::runtime_error("MTP KV LRU comparison did not execute accepted/rejected cache routes");
            std::cout<<"mtp_kv_lru format="<<format<<" tokens="<<tokens<<" windows="<<candidate.windows
                <<" kept="<<candidate.kept<<" reverted="<<candidate.reverted_hits<<" cold="<<candidate.cold_insertions
                <<" cancels="<<candidate.cancels<<" exact PASS\n";
        }
        if(const auto* native=std::getenv("MFQ_MOE_RESIDENCY_NATIVE_DMA");native && native[0]=='1' && adaptive_kv_direct) {
            const auto batches=counter(caches[1],"residency_native_dma_batches");
            if(!batches)throw std::runtime_error("adaptive native DMA comparison did not execute a batch");
            std::cout<<"adaptive_native_dma format="<<format<<" batches="<<batches
                <<" copies="<<counter(caches[1],"residency_native_dma_copies")<<" exact PASS\n";
        }
        if(const auto* native=std::getenv("MFQ_MOE_NATIVE_DMA_BATCH");native && native[0]!='0' && gpu_cold) {
            const auto batches=counter(caches[1],"pipeline_native_dma_batches");
            if(!batches)throw std::runtime_error("native DMA comparison did not execute a batch");
            std::cout<<"native_dma format="<<format<<" tokens="<<tokens<<" batches="<<batches
                <<" copies="<<counter(caches[1],"pipeline_native_dma_copies")
                <<" fallbacks="<<counter(caches[1],"pipeline_native_dma_fallbacks")<<" exact PASS\n";
        }
        if(const auto* projection=std::getenv("MFQ_MOE_KV_PROJECTION_CACHE");projection && projection[0]=='1' && check_kv_direct) {
            const auto baseline=counter(caches[0],"pipeline_kv_projection_hits");
            const auto hits=counter(caches[1],"pipeline_kv_projection_hits");
            const auto misses=counter(caches[1],"pipeline_kv_projection_misses");
            const auto small=counter(caches[1],"pipeline_kv_projection_small_hit_bytes");
            if(baseline || ((gpu_cold && !force_direct_off)?(!hits || !misses || !small):bool(hits || misses || small)))
                throw std::runtime_error("projection cache did not exercise whole-field reuse or dispatch isolation");
            std::cout<<"kv_projection_cache format="<<format<<" tokens="<<tokens<<" gpu_cold="<<gpu_cold
                <<" hits="<<hits<<" misses="<<misses<<" small_hit_bytes="<<small
                <<" fallbacks="<<counter(caches[1],"pipeline_kv_projection_fallbacks")<<" exact PASS\n";
        }
        if(std::getenv("MFQ_TEST_GU_BUILD_OVERLAP") && check_kv_direct) {
            const auto baseline=counter(caches[0],"pipeline_gu_build_overlap_serves");
            const auto candidate=counter(caches[1],"pipeline_gu_build_overlap_serves");
            if(baseline || ((gpu_cold && !force_direct_off)?!candidate:bool(candidate)))
                throw std::runtime_error("GU upload did not overlap Down page preparation");
            std::cout<<"gu_build_overlap format="<<format<<" tokens="<<tokens<<" gpu_cold="<<gpu_cold
                <<" candidate="<<candidate<<" exact PASS\n";
        }
        if(const auto* early=std::getenv("MFQ_MOE_KV_EARLY_GU");early && early[0]=='1' && check_kv_direct) {
            const auto baseline=counter(caches[0],"pipeline_kv_early_gu_serves");
            const auto candidate=counter(caches[1],"pipeline_kv_early_gu_serves");
            if(baseline || ((gpu_cold && !force_direct_off)?!candidate:bool(candidate)))
                throw std::runtime_error("early GU submission did not respect direct-read dispatch");
            std::cout<<"early_gu_dma format="<<format<<" tokens="<<tokens<<" gpu_cold="<<gpu_cold
                <<" candidate="<<candidate<<" exact PASS\n";
        }
        if(const auto* packet=std::getenv("MFQ_MOE_KV_HOST_PACKET");packet && packet[0]=='1' && check_kv_direct) {
            const auto baseline=counter(caches[0],"pipeline_host_packet_serves");
            const auto served=counter(caches[1],"pipeline_host_packet_serves");
            const auto bytes=counter(caches[1],"pipeline_host_packet_descriptor_bytes");
            const auto cached=counter(caches[1],"pipeline_cached_phase_publishes");
            const auto resident=counter(caches[1],"pipeline_kv_gpu_resident_experts");
            const bool descriptors_dma=counter(caches[1],"pipeline_native_dma_batches")>0;
            if(baseline || ((gpu_cold && !force_direct_off)?(!served || (!bytes && !resident && !descriptors_dma)):bool(served || bytes || cached || resident)))
                throw std::runtime_error("host cache packet did not respect direct-read dispatch: baseline="+
                    std::to_string(baseline)+" serves="+std::to_string(served)+" bytes="+std::to_string(bytes)+
                    " cached="+std::to_string(cached));
            const auto* projection=std::getenv("MFQ_MOE_KV_PROJECTION_CACHE");
            if(projection && projection[0]=='1' && gpu_cold && !adaptive_kv_direct && !force_direct_off && !cached && !resident)
                throw std::runtime_error("host cache packet never published a cached phase without DMA");
            std::cout<<"kv_host_packet format="<<format<<" tokens="<<tokens<<" gpu_cold="<<gpu_cold
                <<" serves="<<served<<" descriptor_bytes="<<bytes<<" cached_phase_publishes="<<cached<<" exact PASS\n";
        }
        if(std::getenv("MFQ_TEST_DIRECT_PACKET_BATCH") && check_kv_direct) {
            const auto* option=std::getenv("MFQ_MOE_NATIVE_DIRECT_PACKET");
            const bool expected=gpu_cold && !force_direct_off && tokens<=8 && (!option || option[0]!='0');
            const auto served=counter(caches[1],"pipeline_direct_packet_serves");
            const auto batched=counter(caches[1],"pipeline_direct_packet_batch_serves");
            if(counter(caches[0],"pipeline_direct_packet_serves") || bool(served)!=expected ||
                    bool(batched)!=(expected && tokens>1))
                throw std::runtime_error("short-window direct packet dispatch was not exercised or ignored its opt-out");
            std::cout<<"direct_packet_batch format="<<format<<" tokens="<<tokens
                <<" serves="<<served<<" batch_serves="<<batched<<" exact PASS\n";
        }
        if(const auto* first=std::getenv("MFQ_MOE_KV_LOOKUP_FIRST");first && first[0]=='1' && check_kv_direct) {
            const auto baseline=counter(caches[0],"pipeline_kv_projection_early_hits");
            const auto hits=counter(caches[1],"pipeline_kv_projection_early_hits");
            if(baseline || ((gpu_cold && !force_direct_off)?!hits:bool(hits)))
                throw std::runtime_error("complete projection cache did not bypass RAM lookup");
            std::cout<<"kv_projection_early_hits format="<<format<<" tokens="<<tokens<<" hits="<<hits<<" exact PASS\n";
        }
        if(const auto* option=std::getenv("MFQ_MOE_KV_GPU_RESIDENT");option && option[0]=='1' && check_kv_direct) {
            const auto baseline=counter(caches[0],"pipeline_kv_gpu_resident_experts");
            const auto hits=counter(caches[1],"pipeline_kv_gpu_resident_experts");
            const auto early=counter(caches[1],"pipeline_kv_gpu_early_gu_experts");
            if(baseline || ((gpu_cold && !force_direct_off && !adaptive_kv_direct)?(!hits && !(partial_gu && early)):(!gpu_cold && hits)))
                throw std::runtime_error("cached resident dispatch was not exercised or escaped GPU-only paths");
            std::cout<<"kv_gpu_resident format="<<format<<" tokens="<<tokens<<" experts="<<hits<<" exact PASS\n";
            if(partial_gu && gpu_cold && !early)throw std::runtime_error("partial cache fixture did not execute early GU");
            if(partial_gu)std::cout<<"kv_gpu_early_gu format="<<format<<" tokens="<<tokens<<" experts="<<early<<" exact PASS\n";
            if(route_count>2 && gpu_cold && !force_direct_off) {
                const auto fences=counter(caches[1],"pipeline_reader_fence_reuses");
                if(counter(caches[0],"pipeline_reader_fence_reuses") || !fences)
                    throw std::runtime_error("wide-route fixture did not exercise reader-fenced lease release");
                std::cout<<"kv_reader_fence format="<<format<<" tokens="<<tokens<<" skips="<<fences<<" exact PASS\n";
            }
        }
        // The public dispatcher rejects invalid expert IDs. Both notification
        // schedules must cancel an in-flight graph and reject its next replay.
        ids.copy_(route_tensor(cold,-1));
        mfq_cuda_synchronize();
        for(int mode=0;mode<2;++mode) {
            MfqCudaStreamGuard guard(streams[mode]);environment.set(mode?"1":"0");
            bool rejected=false,unavailable=false;
            try{windows[mode]->run();}catch(const std::exception& error) {
                rejected=std::string(error.what()).find("expert ID is out of range")!=std::string::npos;
            }
            try{windows[mode]->run();}catch(const std::exception& error) {
                unavailable=std::string(error.what()).find("unavailable")!=std::string::npos;
            }
            if(!rejected || !unavailable)throw std::runtime_error("CPU-ready failure did not cancel and invalidate its graph");
            ++cancelled;
        }
    }
    if((!gpu_cold && !cpu_groups) || !gpu_only)throw std::runtime_error("CPU-ready/view test did not exercise its configured dispatch tiers");
    std::cout<<(input_views?"window_input_view_cases=":"early_no_cpu_ready_cases=")<<cases
        <<" tokens="<<tokens<<" strided="<<noncontiguous<<" gpu_cold="<<gpu_cold<<" cpu_groups="<<cpu_groups<<" gpu_only="<<gpu_only
        <<" cancellations="<<cancelled<<" repeated GPU/CPU transitions and duplicate routes exact; invalid-route cancellation PASS\n";
}
void verify_router_prefetch(const std::filesystem::path& root) {
    struct Environment {
        std::string old;
        Environment() {if(auto* p=std::getenv("MFQ_MOE_TRANSFER_CACHE"))old=p;set("1");}
        static void set(const char* p) {
#ifdef _WIN32
            _putenv_s("MFQ_MOE_TRANSFER_CACHE",p);
#else
            if(*p)setenv("MFQ_MOE_TRANSFER_CACHE",p,1);else unsetenv("MFQ_MOE_TRANSFER_CACHE");
#endif
        }
        ~Environment(){set(old.c_str());}
    } environment;
    const auto counter=[](const std::shared_ptr<MoeExpertCache>& cache,const char* name) {
        std::ostringstream out;print_moe_expert_cache_stats(cache,out);
        const auto text=out.str();std::smatch match;
        if(!std::regex_search(text,match,std::regex(std::string(name)+"=([0-9]+)")))
            throw std::runtime_error("missing router prefetch counter");
        return std::stoll(match[1]);
    };
    int cases=0,guards=0,recovered=0;int64_t transferred=0,hits=0,busy=0;
    for(const auto& formats:std::array<std::array<std::string,3>,3>{
        std::array<std::string,3>{"nint5","nint5","nint5"},
        {"nvq3j-512","nvq3j-512","nvq3j-512"},
        {"nint5","nvq3j-l","nvq2j-xl"}}) {
        std::array<CudaExecutionContext,2> executions;
        std::array<std::shared_ptr<MoeExpertCache>,2> caches;
        std::array<std::array<MoeFfnForward,2>,2> forward;
        std::array<std::array<std::array<std::shared_ptr<MoeQuantRangeSource>,3>,2>,2> sources;
        std::array<MfqCudaStream,2> streams={mfq_get_stream_from_pool(),mfq_get_current_cuda_stream()};
        MoeFfnPrefetch prefetch;
        for(int mode=0;mode<2;++mode) {
            MfqCudaStreamGuard guard(streams[mode]);auto& cfg=executions[mode].config;
            cfg.moe_preload_all=true;cfg.moe_assert_resident=true;cfg.moe_pipeline=true;
            cfg.moe_hybrid_cpu=false;cfg.moe_direct_ram=true;cfg.moe_two_stage_ffn=true;
            cfg.moe_ram_pcie=true;cfg.moe_ram_pcie_fraction=1.;cfg.moe_residency_adapt=false;
            int64_t bundle=0;
            for(int p=0;p<3;++p) {
                auto proto=source(root/(formats[p]+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
                bundle+=bytes(proto->metadata()->pools[0]);
            }
            cfg.moe_host_cache_bytes=6*bundle;caches[mode]=make_moe_expert_cache(2*bundle,cfg);
            for(int layer=0;layer<2;++layer) {
                std::vector<std::shared_ptr<MfeWeight>> gu,down;
                for(int p=0;p<3;++p) {
                    auto& src=sources[mode][layer][p];
                    src=source(root/(formats[p]+(p==2?"-2560-640.mfq":"-640-2560.mfq")));
                    auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(caches[mode],
                        "prefetch/"+std::to_string(layer)+"/"+std::to_string(p),src,1,layer,std::to_string(p)));
                    (p==2?down:gu).push_back(weight);
                }
                forward[mode][layer]=make_moe_ffn_pipeline(gu,down,{},{},{},mode==1 && layer==1?&prefetch:nullptr);
            }
            finalize_moe_expert_cache(caches[mode]);
        }
        auto opt=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
        auto input=tb::zeros({1,2560},opt),first_ids=tb::zeros({1,3},opt.dtype(tb::kInt32)),
            next_ids=tb::zeros({1,2},opt.dtype(tb::kInt32)),predicted=tb::zeros({1,2},opt.dtype(tb::kInt32));
        auto first_weights=tb::ones({1,3},opt.dtype(tb::kFloat32))*.17,
            next_weights=tb::ones({1,2},opt.dtype(tb::kFloat32))*.23;
        std::array<tb::Tensor,2> output;
        std::array<std::unique_ptr<mfq::cuda::DecodeWindow>,2> windows;
        for(int mode=0;mode<2;++mode) {
            MfqCudaStreamGuard guard(streams[mode]);windows[mode]=std::make_unique<mfq::cuda::DecodeWindow>(streams[mode].stream());
            windows[mode]->capture([&,mode] {
                mfq::cuda::DecodeWindow::Task prediction{};
                if(mode)prediction=prefetch(predicted);
                auto first=forward[mode][0](executions[mode],input,first_ids,first_weights);
                if(mode)windows[mode]->enroll(std::move(prediction));
                auto hidden=(tb::tanh(first.mean(1))*.0001).to(tb::kFloat16);
                output[mode]=forward[mode][1](executions[mode],hidden,next_ids,next_weights);
            },[]{},[&,mode]{output[mode]={};});
        }
        const std::array<std::array<int32_t,3>,6> first={{{0,0,0},{1,1,1},{2,2,2},{1,2,1},{0,0,0},{0,1,2}}};
        const std::array<std::array<int32_t,2>,6> next={{{2,2},{2,2},{1,1},{1,2},{2,2},{1,2}}};
        const std::array<std::array<int32_t,2>,6> guesses={{{2,2},{1,1},{0,2},{1,2},{2,2},{1,2}}};
        for(int step=0;step<18;++step) {
            std::vector<float> values(2560);for(int i=0;i<2560;++i)values[i]=std::sin(float(i+step*23)*.017)*.13;
            input.copy_(tb::tensor(values).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA));
            const int s=step%6;
            first_ids.copy_(tb::tensor(std::vector<int32_t>(first[s].begin(),first[s].end())).reshape({1,3}).to(tb::kCUDA));
            next_ids.copy_(tb::tensor(std::vector<int32_t>(next[s].begin(),next[s].end())).reshape({1,2}).to(tb::kCUDA));
            predicted.copy_(tb::tensor(std::vector<int32_t>(guesses[s].begin(),guesses[s].end())).reshape({1,2}).to(tb::kCUDA));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
            std::array<tb::Tensor,2> host;
            for(int mode=0;mode<2;++mode) {MfqCudaStreamGuard guard(streams[mode]);windows[mode]->run();host[mode]=output[mode].cpu().contiguous();}
            if(host[0].sizes()!=host[1].sizes() || host[0].scalar_type()!=host[1].scalar_type() ||
                std::memcmp(host[0].data_ptr(),host[1].data_ptr(),host[0].numel()*host[0].element_size()))
                throw std::runtime_error("router prefetch changed original output bytes at step "+std::to_string(step));
            for(const auto& mode:sources)for(const auto& layer:mode)for(const auto& src:layer)
                if(src->expert_disk_reads_after_preload())throw std::runtime_error("router prefetch read expert SSD");
            ++cases;
        }
        transferred+=counter(caches[1],"pipeline_prefetch_bytes");
        hits+=counter(caches[1],"pipeline_prefetch_hit_bytes");busy+=counter(caches[1],"pipeline_prefetch_busy_skips");
        if(counter(caches[0],"pipeline_prefetch_bytes") || counter(caches[1],"hybrid_cpu_experts"))
            throw std::runtime_error("router prefetch native mode changed its serving policy");
        bool rejected=false;
        try{prefetch(predicted);}catch(const std::invalid_argument&){rejected=true;++guards;}
        if(!rejected)throw std::runtime_error("router prefetch accepted an unrecorded call");
        // Fail after reserving one cold expert, then reuse the same cache in a
        // fresh window. The incomplete speculative entry must be discarded.
        prepare_moe_pipeline_comparison(caches[1],false);
        first_ids.copy_(tb::zeros({1,3},first_ids.options()));
        next_ids.copy_(tb::tensor(std::vector<int32_t>{2,2}).reshape({1,2}).to(tb::kCUDA));
        predicted.copy_(tb::tensor(std::vector<int32_t>{2,3}).reshape({1,2}).to(tb::kCUDA));
        rejected=false;
        try{windows[1]->run();}catch(const std::runtime_error& e) {
            rejected=std::string(e.what())=="MFQ prefetch expert outside router geometry";if(rejected)++guards;
        }
        if(!rejected)throw std::runtime_error("router prefetch failed to reject an invalid expert");
        rejected=false;
        try{windows[1]->run();}catch(const std::runtime_error&){rejected=true;++guards;}
        if(!rejected)throw std::runtime_error("router prefetch reused its failed window");
        predicted.copy_(next_ids);MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        {MfqCudaStreamGuard guard(streams[0]);windows[0]->run();}
        mfq::cuda::DecodeWindow fresh(streams[1].stream());tb::Tensor recovery;
        fresh.capture([&] {
            auto prediction=prefetch(predicted);
            auto first=forward[1][0](executions[1],input,first_ids,first_weights);
            fresh.enroll(std::move(prediction));
            recovery=forward[1][1](executions[1],(tb::tanh(first.mean(1))*.0001).to(tb::kFloat16),next_ids,next_weights);
        },[]{},[&]{recovery={};});
        fresh.run();auto expected=output[0].cpu().contiguous(),actual=recovery.cpu().contiguous();
        if(expected.sizes()!=actual.sizes() || std::memcmp(expected.data_ptr(),actual.data_ptr(),actual.numel()*actual.element_size()))
            throw std::runtime_error("router prefetch retained incomplete fields after failure");
        ++recovered;
    }
    if(!transferred || !hits || !busy)throw std::runtime_error("router prefetch omitted speculative DMA, actual hits or busy-slot protection");
    std::cout<<"router_prefetch_exact_cases="<<cases<<" bytes="<<transferred<<" hit_bytes="<<hits
        <<" busy_skips="<<busy<<" expert_SSD=0 PASS\n";
    if(guards!=9 || recovered!=3)throw std::runtime_error("router prefetch failure checks incomplete");
    std::cout<<"router_prefetch_failure_guards="<<guards<<" recovered_exact_cases="<<recovered<<" PASS\n";
}
void verify_router_lookahead_inputs() {
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    struct Pool {
        std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
        ~Pool(){context->end_graph_pool(stream);}
    } pool{context,stream};
    mfq::cuda::RouterLookaheadInputs audit(3,7,3);
    auto input=tb::zeros({3,7},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
    mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
    window.capture([&] {
        auto snapshot=input.clone();
        for(int layer=0;layer<3;++layer)audit.input(layer,snapshot.narrow(0,layer,1));
        window.enroll({&audit,[]{},[]{},[&]{audit.finish_sample();},{}});
    },[]{},[]{});
    if(audit.samples())throw std::runtime_error("lookahead audit recorded capture warmup as decode");
    std::vector<tb::Tensor> originals;
    for(int step=0;step<3;++step) {
        std::vector<float> values(21);for(int i=0;i<21;++i)values[i]=float(i+step*32)/16;
        values[step]=-0.0f;
        auto original=tb::tensor(values).to(tb::kFloat16).reshape({3,7}).contiguous();
        input.copy_(original.to(tb::kCUDA));window.run();originals.push_back(original);
    }
    for(int layer=0;layer<3;++layer) {
        std::vector<tb::Tensor> rows;for(const auto& original:originals)rows.push_back(original.narrow(0,layer,1));
        auto expected=tb::cat(rows,0).contiguous(),actual=audit.rows(layer);
        if(actual.sizes()!=expected.sizes() || std::memcmp(actual.data_ptr(),expected.data_ptr(),expected.numel()*expected.element_size()))
            throw std::runtime_error("lookahead audit changed input bytes or sample/layer order");
    }
    int guards=0;
    try {audit.finish_sample();}catch(const std::runtime_error&){++guards;}
    try {audit.input(3,input.narrow(0,0,1));}catch(const std::invalid_argument&){++guards;}
    try {audit.input(0,input);}catch(const std::invalid_argument&){++guards;}
    try {audit.rows(-1);}catch(const std::invalid_argument&){++guards;}
    if(guards!=4)throw std::runtime_error("lookahead audit shape/sample guards incomplete");
    std::cout<<"router_lookahead_input_samples=3 layer_rows=9 warmup_samples=0 guards=4 original_bytes_exact=1 PASS\n";
}
#include "residency_direct_exchange_test.h"
void verify_predictor_residency(const std::filesystem::path& root,bool prefer_resident=false) {
    struct Preference {
        std::string previous=std::getenv("MFQ_MTP_RESIDENT_EXPERTS")?std::getenv("MFQ_MTP_RESIDENT_EXPERTS"):"";
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("MFQ_MTP_RESIDENT_EXPERTS",value);
#else
            if(*value)setenv("MFQ_MTP_RESIDENT_EXPERTS",value,1);else unsetenv("MFQ_MTP_RESIDENT_EXPERTS");
#endif
        }
        ~Preference(){set(previous.c_str());}
    } preference;
    Preference::set(prefer_resident?"1":"0");
    CudaExecutionContext execution;auto& config=execution.config;
    config.moe_pipeline=true;config.moe_preload_all=true;config.moe_assert_resident=true;
    config.moe_residency_warm=true;config.moe_residency_projection_heat=true;
    config.moe_ram_pcie=true;config.moe_ram_pcie_fraction=1;config.moe_direct_ram=true;
    config.moe_two_stage_ffn=prefer_resident;
    const int main_layers=prefer_resident?10:1,predictor_source=main_layers*3;
    std::vector<std::shared_ptr<MoeQuantRangeSource>> sources(predictor_source+2);
    std::size_t gpu_bytes=0;
    for(int p=0;p<int(sources.size());++p) {
        const bool down=p<predictor_source?p%3==2:p==predictor_source+1;
        sources[p]=source(root/(std::string("nint5")+(down?"-2560-640.mfq":"-640-2560.mfq")));
        gpu_bytes+=bytes(sources[p]->metadata()->pools[0]);
    }
    config.moe_host_cache_bytes=3*gpu_bytes;
    auto cache=make_moe_expert_cache(gpu_bytes,config);
    std::vector<MfeWeight> weights(sources.size());
    for(int p=0;p<int(sources.size());++p)weights[p]=cache_quant_moe_weight(cache,std::to_string(p),sources[p],1,p<predictor_source?p/3:-1,std::to_string(p));
    // Exercise the physical cache topology independently of FFN dimensions:
    // predictor Gate and Up alias one source, while Down owns another.
    std::vector<std::array<int,3>> bundles;
    for(int layer=0;layer<main_layers;++layer)bundles.push_back({layer*3,layer*3+1,layer*3+2});
    bundles.push_back({predictor_source,predictor_source,predictor_source+1});
    MoeResidencyTestAccess::bundles(*cache,std::move(bundles));
    finalize_moe_expert_cache(cache);
    if(prefer_resident) {
        for(int p=predictor_source;p<int(sources.size());++p)for(int e=0;e<3;++e)
            if(!sources[p]->gpu_resident(e))throw std::runtime_error("predictor preference left an expert in RAM");
        std::cout<<"predictor_residency preferred=1 all_predictor_experts_on_GPU=1 shared_budget=1 PASS\n";
        return;
    }
    const auto before=sources[predictor_source]->host_cache()->stats();
    MoeResidencyManager manager(cache.get());
    manager.after_layer(-1,std::vector<int32_t>(100,2),100);
    manager.after_layer(0,{},100);
    manager.apply_pending();
    const auto stats=manager.stats();
    if(!sources[3]->gpu_resident(2) || !sources[4]->gpu_resident(2) ||
            stats.committed_projections!=2 || stats.committed_bundles!=1 || stats.observed_routes!=100)
        throw std::runtime_error("combined predictor did not promote exactly its two physical projections");
    const auto after=sources[3]->host_cache()->stats();
    if(before.managed_bytes!=after.managed_bytes || before.managed_peak_bytes!=after.managed_peak_bytes)
        throw std::runtime_error("predictor promotion duplicated RAM weights");
    for(const auto& source:sources)if(source->expert_disk_reads_after_preload())
        throw std::runtime_error("predictor promotion reread SSD weights");
    std::cout<<"predictor_residency combined_gate_up=1 physical_promotions=2 routes=100 RAM_bytes_unchanged=1 SSD_reads=0 PASS\n";
}
#include "prefill_pipeline_test.h"
#include "compact_prefill_mma_test.h"
#include "nvq_prefill_k_tile_test.h"
#include "mfe_token_group_test.h"
#include "d4_bank_map_test.h"
#include "mfe_wide_compact_test.h"
#include "mfe_shape_dispatch_test.h"
#include "d4_analytic_scale_test.h"
void verify_yarn() {
    // Independent values for dim=64, theta=10000, original context=4096,
    // factor=4. The ramp spans pairs 10..23 (32 and 1 full rotations).
    mfq::models::YarnScaling scaling;
    scaling.factor=4;scaling.original_context=4096;
    const int64_t positions[]={0,63,4095,8191};
    auto opts=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    auto x=tb::ones({1,1,4,68},opts);
    auto pos=tb::tensor(std::vector<int64_t>(positions,positions+4)).to(tb::kCUDA);
    mfq::cuda::RotaryEmbedding rotary(64,16384,10000,{},false,std::nullopt,scaling);
    auto output=rotary.forward(x,pos).cpu().contiguous();
    RopeCache cache(16384,64,10000,0,-1,tb::Device(tb::kCUDA),false,scaling);
    const auto cosine=cache.cos.cpu().contiguous(),sine=cache.sin.cpu().contiguous();
    const double amplitude=1+.1*std::log(4.0);
    for(int t=0;t<4;++t)for(int p=0;p<32;++p) {
        const double ramp=std::clamp((p-10.0)/13.0,0.0,1.0);
        const float frequency=float(std::pow(10000.,-2.*p/64)*(1-.75*ramp));
        const float angle=float(positions[t])*frequency;
        const auto c=std::cos(angle)*amplitude,s=std::sin(angle)*amplitude;
        if(std::abs(output.data_ptr<float>()[t*68+p]-(c-s))>2e-5 ||
            std::abs(output.data_ptr<float>()[t*68+32+p]-(s+c))>2e-5 ||
            std::abs(cosine.data_ptr<float>()[positions[t]*32+p]-c)>2e-5 ||
            std::abs(sine.data_ptr<float>()[positions[t]*32+p]-s)>2e-5)
            throw std::runtime_error("CUDA YaRN differs from interpolated frequencies and attention scaling");
    }
    for(int t=0;t<4;++t)for(int p=64;p<68;++p)
        if(output.data_ptr<float>()[t*68+p]!=1)throw std::runtime_error("YaRN changed non-rotary dimensions");
    std::cout<<"cuda_yarn_tables_dynamic_partial_rotary PASS\n";
}

void verify_online_expert_budget(const std::filesystem::path& root) {
    for(const std::string format:{"nint5","nvq3j-512"}) {
        CudaExecutionContext execution;
        execution.config.moe_pipeline=true;execution.config.moe_preload_all=true;
        execution.config.moe_ram_pcie=true;execution.config.moe_ram_pcie_fraction=1;
        execution.config.moe_residency_adapt=false;execution.config.moe_assert_resident=true;
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> sources={
            source(root/(format+"-640-2560.mfq")),source(root/(format+"-640-2560.mfq")),
            source(root/(format+"-2560-640.mfq"))};
        std::size_t bundle=0;for(const auto& s:sources)bundle+=bytes(s->metadata()->pools[0]);
        execution.config.moe_host_cache_bytes=16*bundle;
        auto cache=make_moe_expert_cache(4*bundle,execution.config);
        std::vector<std::shared_ptr<MfeWeight>> gu,down;
        const std::array<std::string,3> roles={"gate","up","down"};
        for(int p=0;p<3;++p) {
            auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,roles[p],sources[p],1,0,roles[p]));
            (p==2?down:gu).push_back(std::move(weight));
        }
        auto ffn=make_moe_ffn_pipeline(gu,down,{});
        finalize_moe_expert_cache(cache);
        auto x=tb::full({2,2560},.001,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
        auto ids=tb::arange(3,x.options().dtype(tb::kInt32)).unsqueeze(0).expand({2,3}).contiguous();
        auto weights=tb::ones({2,3},x.options().dtype(tb::kFloat32));
        auto expected=ffn(execution,x,ids,weights).to(tb::kFloat32).cpu().contiguous();
        auto prompt=tb::full({32,2560},.001,x.options());
        auto prompt_ids=ids.narrow(0,0,1).expand({32,3}).contiguous();
        auto prompt_weights=tb::ones({32,3},weights.options());
        auto expected_prompt=ffn(execution,prompt,prompt_ids,prompt_weights).to(tb::kFloat32).cpu().contiguous();
        const auto compare=[&](tb::Tensor result,const tb::Tensor& reference) {
            auto actual=result.to(tb::kFloat32).cpu().contiguous();
            if(actual.sizes()!=reference.sizes())throw std::runtime_error("online expert resize changed output shape");
            for(int64_t i=0;i<actual.numel();++i)
                if(!std::isfinite(actual.data_ptr<float>()[i]) ||
                    std::abs(actual.data_ptr<float>()[i]-reference.data_ptr<float>()[i])>
                        0.002f+0.001f*std::abs(reference.data_ptr<float>()[i]))
                    throw std::runtime_error("online expert resize changed FFN output at "+std::to_string(i)+
                        " rows="+std::to_string(actual.size(0))+" actual="+std::to_string(actual.data_ptr<float>()[i])+
                        " expected="+std::to_string(reference.data_ptr<float>()[i]));
        };
        for(const auto target:{2*bundle,4*bundle}) {
            resize_moe_expert_cache(cache,target);
            compare(ffn(execution,x,ids,weights),expected);
            compare(ffn(execution,prompt,prompt_ids,prompt_weights),expected_prompt);
            release_moe_decode_buffers(cache);
            tb::Tensor output;
            {
                mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
                window.capture([&]{output=ffn(execution,x,ids,weights);},[]{},[&]{output={};});
                window.run();compare(output,expected);
            }
            if(moe_expert_cache_budget_bytes(cache)!=target)throw std::runtime_error("expert budget was not applied");
            for(const auto& source:sources)if(source->expert_disk_reads_after_preload())
                throw std::runtime_error("online expert budget reread sealed SSD weights");
        }
        bool rejected=false;
        try {resize_moe_expert_cache(cache,1);}catch(const std::exception&){rejected=true;}
        if(!rejected || moe_expert_cache_budget_bytes(cache)!=4*bundle)
            throw std::runtime_error("invalid expert budget changed active cache");
        (void)ffn(execution,x,ids,weights).cpu();
        std::cout<<"online_expert_budget format="<<format<<" shrink_grow_reject prefill graph sealed_RAM PASS\n";
    }
}

void verify_prefill_quota_recovery(const std::filesystem::path& root) {
    struct Setting {
        std::string previous;
        Setting() {
            if(const auto* value=std::getenv("MFQ_MOE_PREFILL_LAYER"))previous=value;
            set("1");
        }
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("MFQ_MOE_PREFILL_LAYER",value);
#else
            if(*value)setenv("MFQ_MOE_PREFILL_LAYER",value,1);else unsetenv("MFQ_MOE_PREFILL_LAYER");
#endif
        }
        ~Setting(){set(previous.c_str());}
    } setting;
    for(const std::string format:{"nint5","nvq3j-512"})
    for(const std::string failure:{"CUDA VRAM limit exceeded:","CUDA pinned allocation failed:"}) {
        CudaExecutionContext execution;
        execution.config.moe_pipeline=true;execution.config.moe_preload_all=true;
        execution.config.moe_ram_pcie=true;execution.config.moe_ram_pcie_fraction=1;
        execution.config.moe_direct_ram=true;execution.config.moe_residency_adapt=false;
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> sources={
            source(root/(format+"-640-2560.mfq")),source(root/(format+"-640-2560.mfq")),
            source(root/(format+"-2560-640.mfq"))};
        std::size_t bundle=0;for(const auto& s:sources)bundle+=bytes(s->metadata()->pools[0]);
        execution.config.moe_host_cache_bytes=12*bundle;
        auto cache=make_moe_expert_cache(2*bundle,execution.config);
        std::vector<std::shared_ptr<MfeWeight>> gu,down;
        const std::array<std::string,3> roles={"gate","up","down"};
        for(int p=0;p<3;++p) {
            auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,roles[p],sources[p],1,0,roles[p]));
            (p==2?down:gu).push_back(std::move(weight));
        }
        bool inject=true,fatal=false;int shared_calls=0;
        auto ffn=make_moe_ffn_pipeline(gu,down,[&](CudaExecutionContext&,const tb::Tensor&) {
            ++shared_calls;
            if(inject) {
                if(failure=="CUDA VRAM limit exceeded:")throw MfqResourceExhausted(failure+" injected after prefill transfer submission");
                throw mfq::cuda::Error(failure+" injected after prefill transfer submission");
            }
            if(fatal)throw std::runtime_error("injected non-quota prefill failure");
            return tb::Tensor{};
        });
        finalize_moe_expert_cache(cache);
        auto x=tb::full({16,2560},.0001,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
        auto ids=tb::arange(3,x.options().dtype(tb::kInt32)).unsqueeze(0).expand({16,3}).contiguous();
        auto weights=tb::ones({16,3},x.options().dtype(tb::kFloat32));
        bool rejected=false;
        try {(void)ffn(execution,x,ids,weights);}
        catch(const std::exception& error) {rejected=std::string(error.what()).starts_with(failure);}
        if(!rejected || shared_calls!=1)throw std::runtime_error("prefill quota fixture did not reach the eager transfer path");
        inject=false;
        auto first=ffn(execution,x,ids,weights).to(tb::kCPU).contiguous();
        auto second=ffn(execution,x,ids,weights).to(tb::kCPU).contiguous();
        if(shared_calls!=3 || first.sizes()!=second.sizes() ||
                std::memcmp(first.data_ptr(),second.data_ptr(),tensor_nbytes(first)))
            throw std::runtime_error("prefill quota recovery changed subsequent FFN execution");
        fatal=true;bool fatal_rejected=false;
        try {(void)ffn(execution,x,ids,weights);}
        catch(const std::runtime_error& error) {fatal_rejected=std::string(error.what())=="injected non-quota prefill failure";}
        fatal=false;bool poisoned=false;
        try {(void)ffn(execution,x,ids,weights);}
        catch(const std::runtime_error& error) {poisoned=std::string(error.what()).starts_with("MFQ FFN session requires recovery");}
        if(!fatal_rejected || !poisoned)throw std::runtime_error("prefill recovery suppressed a non-quota failure");
        release_moe_decode_buffers(cache);
        std::cout<<"prefill_quota_recovery format="<<format<<" failure="<<failure
            <<" subsequent_requests=2 output=PASS fatal_error_preserved=PASS\n";
    }
}

void verify_memory_phase_reuse(const std::filesystem::path& root) {
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    int cases=0;
    for(const std::string format:{"nint5","nvq3j-512"}) {
        CudaExecutionContext execution;
        execution.config.moe_pipeline=true;execution.config.moe_preload_all=true;
        execution.config.moe_ram_pcie=true;execution.config.moe_ram_pcie_fraction=1;
        execution.config.moe_residency_adapt=false;
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> sources={
            source(root/(format+"-640-2560.mfq")),source(root/(format+"-640-2560.mfq")),
            source(root/(format+"-2560-640.mfq"))};
        std::size_t gpu_bytes=0;for(const auto& s:sources)gpu_bytes+=bytes(s->metadata()->pools[0]);
        execution.config.moe_host_cache_bytes=3*gpu_bytes;
        auto cache=make_moe_expert_cache(gpu_bytes,execution.config);
        std::vector<std::shared_ptr<MfeWeight>> gu,down;
        const std::array<std::string,3> roles={"gate","up","down"};
        for(int p=0;p<3;++p) {
            auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,roles[p],sources[p],1,0,roles[p]));
            (p==2?down:gu).push_back(std::move(weight));
        }
        auto ffn=make_moe_ffn_pipeline(gu,down);finalize_moe_expert_cache(cache);
        if(!ffn)throw std::runtime_error("phase reuse fixture has no FFN pipeline");
        for(int tokens:{1,3,6}) {
            auto x=tb::full({tokens,2560},.0001,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
            auto ids=tb::arange(3,x.options().dtype(tb::kInt32)).unsqueeze(0).expand({tokens,3}).contiguous();
            auto weights=tb::ones({tokens,3},x.options().dtype(tb::kFloat32));
            auto expected=ffn(execution,x,ids,weights).to(tb::kCPU).to(tb::kFloat32).contiguous();
            const auto before=context->memory_stats().allocated;
            release_moe_decode_buffers(cache);context->reclaim_idle_memory();
            const auto after=context->memory_stats().allocated;
            if(after>=before)throw std::runtime_error("phase transition retained all decode buffers");
            tb::Tensor result;
            {
                mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
                window.capture([&]{result=ffn(execution,x,ids,weights);},[]{},[&]{result={};});
                window.run();
                auto actual=result.to(tb::kCPU).to(tb::kFloat32).contiguous();
                if(actual.sizes()!=expected.sizes())throw std::runtime_error("phase reuse changed FFN geometry");
                for(int64_t i=0;i<actual.numel();++i)
                    if(!std::isfinite(actual.data_ptr<float>()[i]) ||
                       std::abs(actual.data_ptr<float>()[i]-expected.data_ptr<float>()[i])>2e-5+1e-3*std::abs(expected.data_ptr<float>()[i]))
                        throw std::runtime_error("phase reuse changed FFN output");
            }
            result={};release_moe_decode_buffers(cache);++cases;
            std::cout<<"phase_reuse format="<<format<<" tokens="<<tokens
                <<" released_bytes="<<before-after<<" output=PASS\n";
        }
    }
    std::cout<<"memory_phase_reuse_cases="<<cases<<" PASS\n";
}
void verify_activation_workspace_reuse() {
    for(bool mixed:{false,true}) {
        MfeWeight weight;MixedMoeRuntime runtime;
        for(int rows:{1,17,33,129,513,1024,2049,4096,8193,22}) {
            auto x=tb::empty({rows,256},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
            auto workspace=mixed?runtime.activation_workspace(x,rows,8,32,{},false):
                weight.activation_workspace(x,rows,8,32,false);
            if(workspace.qx.size(0)!=rows || workspace.xscale.size(0)!=rows)
                throw std::runtime_error("activation workspace view does not match the row count");
        }
        const auto entries=mixed?runtime.activation_workspaces.size():weight.activation_workspaces->size();
        if(entries!=1)throw std::runtime_error("eager activation scratch retained one buffer per prompt length");
    }
    std::cout<<"activation_workspace_reuse eager_rows=10 runtimes=2 PASS\n";
}
void verify_activation_workspace_binding() {
    for(int rows:{1,3,8,1024,2048,4096})for(bool stable:{false,true}) {
        auto options=tb::TensorOptions().device(tb::kCUDA);
        auto x=tb::empty({rows,256},options.dtype(tb::kFloat16));
        const MoeActivationWorkspace shared{tb::empty({rows,256},options.dtype(tb::kInt8)),
            tb::empty({rows,8},options.dtype(tb::kFloat32))};
        MfeWeight weight;MixedMoeRuntime runtime;
        weight.activation_workspaces->insert_or_assign({rows,8,32,x.get_device()},shared);
        runtime.activation_workspaces.insert_or_assign({rows,8,32,x.get_device(),{}},shared);
        for(auto workspace:{weight.activation_workspace(x,rows,8,32,stable),
                runtime.activation_workspace(x,rows,8,32,{},stable)})
            if(workspace.qx.data_ptr()!=shared.qx.data_ptr() || workspace.xscale.data_ptr()!=shared.xscale.data_ptr())
                throw std::runtime_error("activation scratch ignored a producer's prequantized binding");
        if(weight.activation_workspaces->size()!=1 || runtime.activation_workspaces.size()!=1)
            throw std::runtime_error("prequantized binding allocated redundant activation scratch");
    }
    std::cout<<"activation_workspace_binding rows=6 policies=2 runtimes=2 PASS\n";
}
void verify_activation_workspace_stable_allocation() {
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();
    for(bool mixed:{false,true}) {
        MfeWeight weight;MixedMoeRuntime runtime;
        const auto scratch=[&](const tb::Tensor& x,int rows,bool stable) {
            return mixed?runtime.activation_workspace(x,rows,8,32,{},stable):
                weight.activation_workspace(x,rows,8,32,stable);
        };
        auto small=tb::empty({1,256},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
        auto captured=scratch(small,1,true);
        const void* pointer=captured.qx.data_ptr();
        const void* scale_pointer=captured.xscale.data_ptr();
        const auto initial=mixed?runtime.activation_workspaces.begin()->second.qx.size(0):
            weight.activation_workspaces->begin()->second.qx.size(0);
        if(initial!=1)throw std::runtime_error("one-token scratch reserved a full short-prefill buffer");
        context->begin_graph_pool(stream);
        {
            mfq::cuda::DecodeWindow window(stream);
            window.capture([&] {
                MFQ_CUDA_CHECK(cudaMemsetAsync(captured.qx.data_ptr(),0x35,256,stream));
                MFQ_CUDA_CHECK(cudaMemsetAsync(captured.xscale.data_ptr(),0,8*sizeof(float),stream));
            },[]{},[]{});
            for(int rows:{2,3,8,9,33,129,513,1024,2048,2049,8193}) {
                auto x=tb::empty({rows,256},small.options());
                auto workspace=scratch(x,rows,rows<=2048);
                auto matching=mixed?runtime.activation_workspace(x,rows,8,32,{}):
                    weight.activation_workspace(x,rows,8,32);
                if(workspace.qx.data_ptr()!=matching.qx.data_ptr())
                    throw std::runtime_error("forward and producer default activation policies differ");
            }
            auto inputs=tb::empty({128,256},small.options());
            for(int rows=1;rows<=128;++rows)(void)scratch(inputs.narrow(0,0,rows),rows,true);
            const auto retained=scratch(small,1,true);
            if(retained.qx.data_ptr()!=pointer || retained.xscale.data_ptr()!=scale_pointer)
                throw std::runtime_error("larger prefill moved graph-captured activation scratch");
            window.run();
            auto result=captured.qx.cpu();
            for(int i=0;i<256;++i)if(result.data_ptr<int8_t>()[i]!=0x35)
                throw std::runtime_error("captured activation scratch was not valid on replay");
            auto scales=captured.xscale.cpu();
            for(int i=0;i<8;++i)if(scales.data_ptr<float>()[i]!=0.0f)
                throw std::runtime_error("captured activation scales were not valid on replay");
            std::size_t rows=0,entries=0;
            const auto count=[&](const auto& values) {
                for(const auto& item:values)if(item.first.input_rows<0) {
                    rows+=std::size_t(item.second.qx.size(0));++entries;
                }
            };
            if(mixed)count(runtime.activation_workspaces);else count(*weight.activation_workspaces);
            if(rows>4095 || entries>12)
                throw std::runtime_error("stable activation capacity buckets exceeded their memory bound");
        }
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));context->end_graph_pool(stream);
    }
    std::cout<<"activation_workspace_stable_allocation graph_replay=2 bounded_buckets=1 PASS\n";
}

int main(int argc,char** argv)try {
    std::cout.setf(std::ios::unitbuf);
    if(argc==2 && std::string(argv[1])=="--activation-workspace-reuse-check") {
        verify_activation_workspace_reuse();return 0;
    }
    if(argc==2 && std::string(argv[1])=="--activation-workspace-binding-check") {
        verify_activation_workspace_binding();return 0;
    }
    if(argc==2 && std::string(argv[1])=="--activation-workspace-stable-check") {
        verify_activation_workspace_stable_allocation();return 0;
    }
    if(argc!=2 && argc!=3)throw std::runtime_error("expected real-shape range fixture directory");
    const std::filesystem::path root(argv[1]);
    auto cuda_context=mfq::cuda::default_context(mfq_current_cuda_device());
    if(argc==3) {
        if(std::string(argv[2])=="--online-budget-check") {verify_online_expert_budget(root);return 0;}
        if(std::string(argv[2])=="--mfe-shared-early-check") {
            for(bool transfer_cache:{false,true})verify_two_stage_shared(root,transfer_cache,false,false,false,false,false,false,false,false,false,false,false,false,true);
            return 0;
        }
        if(std::string(argv[2])=="--mfe-cached-reduce-check") {
            for(bool transfer_cache:{false,true})verify_two_stage_shared(root,transfer_cache,false,false,false,false,false,false,false,false,false,false,false,true);
            return 0;
        }
        if(std::string(argv[2])=="--mfe-shared-short-check") {
            for(bool transfer_cache:{false,true})verify_two_stage_shared(root,transfer_cache,false,false,false,false,false,false,false,false,false,false,true);
            return 0;
        }
        if(std::string(argv[2])=="--mfe-gu-group-pipeline-check") {
            for(bool transfer_cache:{false,true})verify_two_stage_shared(root,transfer_cache,false,false,false,false,false,false,false,false,false,true);
            return 0;
        }
        if(std::string(argv[2])=="--mfe-shared-stream-check") {
            for(bool transfer_cache:{false,true})verify_two_stage_shared(root,transfer_cache,false,false,false,false,false,false,false,false,true);
            return 0;
        }
        if(std::string(argv[2])=="--yarn-check") {verify_yarn();return 0;}
        if(std::string(argv[2])=="--nvq-bank-map-check") {verify_nvq_bank_map(root);return 0;}
        if(std::string(argv[2])=="--d4-bank-map-check") {verify_d4_bank_map(root);return 0;}
        if(std::string(argv[2])=="--d4-bank-map-memory-check") {verify_d4_bank_map(root,false);return 0;}
        if(std::string(argv[2])=="--d4-bank-map-stages") {verify_d4_bank_map(root,2);return 0;}
        if(std::string(argv[2])=="--mfe-wide-compact-check") {verify_mfe_wide_compact(root);return 0;}
        if(std::string(argv[2])=="--mfe-shape-dispatch-check") {verify_mfe_shape_dispatch(root);return 0;}
        if(std::string(argv[2])=="--mfe-wide-compact-memory-check") {verify_mfe_wide_compact(root,false);return 0;}
        if(std::string(argv[2])=="--mfe-aligned-compact-check") {verify_mfe_wide_compact(root,true,true);return 0;}
        if(std::string(argv[2])=="--mfe-aligned-compact-memory-check") {verify_mfe_wide_compact(root,false,true);return 0;}
        if(std::string(argv[2])=="--mfe-wide-legacy-memory-check") {verify_mfe_wide_compact(root,false,false,true);return 0;}
        if(std::string(argv[2])=="--mfe-shared-quad-check") {verify_mfe_wide_compact(root,true,true,false,true);return 0;}
        if(std::string(argv[2])=="--mfe-shared-quad-memory-check") {verify_mfe_wide_compact(root,false,true,false,true);return 0;}
        if(std::string(argv[2])=="--mfe-shared-split-check") {verify_mfe_wide_compact(root,true,true,false,true,true);return 0;}
        if(std::string(argv[2])=="--mfe-shared-split-memory-check") {verify_mfe_wide_compact(root,false,true,false,true,true);return 0;}
        if(std::string(argv[2])=="--mfe-aligned-warps-check") {verify_mfe_wide_compact(root,true,true,false,false,false,true);return 0;}
        if(std::string(argv[2])=="--mfe-aligned-warps-memory-check") {verify_mfe_wide_compact(root,false,true,false,false,false,true);return 0;}
        if(std::string(argv[2])=="--mfe-nonaligned-quad-check") {verify_mfe_wide_compact(root,true,true,false,true,false,false,true);return 0;}
        if(std::string(argv[2])=="--mfe-nonaligned-quad-memory-check") {verify_mfe_wide_compact(root,false,true,false,true,false,false,true);return 0;}
        if(std::string(argv[2])=="--d4-analytic-scale-check") {verify_d4_analytic_scales(root);return 0;}
        if(std::string(argv[2])=="--d4-analytic-scale-memory-check") {verify_d4_analytic_scales(root,false);return 0;}
        if(std::string(argv[2])=="--memory-phase-check") {verify_memory_phase_reuse(root);return 0;}
        if(std::string(argv[2])=="--prefill-quota-recovery-check") {verify_prefill_quota_recovery(root);return 0;}
        if(std::string(argv[2])=="--mfe-expert-plan-pipeline-check") {
            verify_mfe_expert_plan_pipeline(root);return 0;
        }
        if(std::string(argv[2])=="--mfe-expert-plan-check") {verify_mfe_expert_plan(root);return 0;}
        if(std::string(argv[2])=="--mfe-token-group-check") {verify_mfe_token_groups(root);return 0;}
        if(std::string(argv[2])=="--speculative-route-prefix-check") {verify_speculative_route_prefix(root);return 0;}
        if(std::string(argv[2])=="--mfe-token-group-pipeline-check") {
            for(bool transfer_cache:{false,true})verify_two_stage_shared(root,transfer_cache,false,false,false,false,false,false,false,true);
            return 0;
        }
        if(std::string(argv[2])=="--nvq-prefill-k-tile-check") {verify_nvq_prefill_k_tiles(root);return 0;}
        if(std::string(argv[2])=="--nvq-prefill-cohort-default-check") {verify_nvq_prefill_k_tiles(root,true,true);return 0;}
        if(std::string(argv[2])=="--nvq-prefill-cohort-check") {verify_nvq_prefill_k_tiles(root,true);return 0;}
        if(std::string(argv[2])=="--compact-prefill-mma-check") {verify_compact_prefill_mma(root);return 0;}
        if(std::string(argv[2])=="--prefill-stream-oracle-check") {
            verify_prefill_graph_execution(root,true,1,true,true);
            verify_prefill_graph_execution(root,true,16,true,true);return 0;
        }
        if(std::string(argv[2])=="--prefill-graph-check") {verify_prefill_graph_execution(root);return 0;}
        if(std::string(argv[2])=="--prefill-layer-oracle-check") {verify_prefill_graph_execution(root,false,16,true,true,true);return 0;}
        if(std::string(argv[2])=="--prefill-layer-residency-check") {verify_prefill_graph_execution(root,false,16,true,true,true,false,false,true);return 0;}
        if(std::string(argv[2])=="--prefill-layer-partial-pin-check") {verify_prefill_graph_execution(root,false,16,true,true,true,false,false,true,true);return 0;}
        if(std::string(argv[2])=="--prefill-layer-exact-check") {verify_prefill_graph_execution(root,false,16,true,false,true);return 0;}
        if(std::string(argv[2])=="--prefill-layer-batch-check") {verify_prefill_graph_execution(root,false,16,true,true,true,false,true);return 0;}
        if(std::string(argv[2])=="--prefill-layer-layout-check") {verify_prefill_graph_execution(root,false,16,false,false,true,true);return 0;}
        if(std::string(argv[2])=="--prefill-group-check") {verify_prefill_graph_execution(root,true);return 0;}
        if(std::string(argv[2])=="--prefill-group-sparse-check") {verify_prefill_graph_execution(root,true,1,true);return 0;}
        if(std::string(argv[2])=="--prefill-group16-check") {verify_prefill_graph_execution(root,true,16);return 0;}
        if(std::string(argv[2])=="--copy-overlap-check") {verify_copy_kernel_overlap();return 0;}
        if(std::string(argv[2])=="--mapped-refill-check") {verify_mapped_refill_descriptors();return 0;}
        if(std::string(argv[2])=="--direct-exchange-check") {verify_direct_exchange(root);return 0;}
        if(std::string(argv[2])=="--indexed-shadow-check") {verify_indexed_shadow_exchange(root);return 0;}
        if(std::string(argv[2])=="--prefill-cpu-prefetch-check") {verify_prefill_cpu_prefetch(root);return 0;}
        if(std::string(argv[2])=="--transfer-promotion-check") {
            const auto* mode=std::getenv("MFQ_MOE_RESIDENCY_PROMOTE_TRANSFER");
            if(!mode || (mode[0]!='1' && mode[0]!='2'))throw std::runtime_error("promotion check requires explicit candidate mode");
            verify_projection_heat(root,true);return 0;
        }
        if(std::string(argv[2])=="--nint-whole-check") {
            for(bool transfer_cache:{false,true})verify_two_stage_shared(root,transfer_cache,false,false,false,false,false,false,true);
            return 0;
        }
        if(std::string(argv[2])=="--nvq-dense-default-check") {
            verify_nvq_dense_default(root);return 0;
        }
        if(std::string(argv[2])=="--nvq-dense-check") {
            for(bool transfer_cache:{false,true})verify_two_stage_shared(root,transfer_cache,false,false,false,false,false,true);
            return 0;
        }
        if(std::string(argv[2])=="--resident-overlap-check") {
            for(bool transfer_cache:{false,true})verify_two_stage_shared(root,transfer_cache,false,false,false,false,true);
            return 0;
        }
        if(std::string(argv[2])=="--e8-narrow-check") {
            for(bool transfer_cache:{false,true})verify_two_stage_shared(root,transfer_cache,false,false,false,true);
            return 0;
        }
        if(std::string(argv[2])=="--default-math-check") {
            for(bool transfer_cache:{false,true})verify_two_stage_shared(root,transfer_cache,false,false,true);
            return 0;
        }
        if(std::string(argv[2])=="--shared-input-check") {
            const auto cases=verify_shared_input(root);
            std::cout<<"shared_input_cases="<<cases<<" mixed Gate/Up outputs, bytes and scales exact PASS\n";
            return 0;
        }
        if(std::string(argv[2])=="--nint-dispatch-check") {
            verify_nint_dispatch(root);return 0;
        }
        if(std::string(argv[2])=="--warm-residency-check") {
            verify_warm_residency(root);return 0;
        }
        if(std::string(argv[2])=="--short-decode-residency-interval-check") {
            verify_short_decode_residency_interval(root);return 0;
        }
        if(std::string(argv[2])=="--predictor-residency-check") {
            verify_predictor_residency(root);verify_predictor_residency(root,true);return 0;
        }
        if(std::string(argv[2])=="--warm-residency-dedicated-check") {
            verify_warm_residency(root,false);return 0;
        }
        if(std::string(argv[2])=="--router-prefetch-check") {
            verify_router_prefetch(root);return 0;
        }
        if(std::string(argv[2])=="--router-lookahead-input-check") {
            verify_router_lookahead_inputs();return 0;
        }
        if(std::string(argv[2])=="--predictor-exchange-overlap-check") {
            verify_predictor_exchange_overlap(root);return 0;
        }
        if(std::string(argv[2])=="--transfer-comparison-check") {
            verify_two_stage_shared(root,true,true);return 0;
        }
        if(std::string(argv[2])=="--phased-transfer-check") {
            verify_phased_wire_bytes();verify_two_stage_shared(root,true);return 0;
        }
        if(std::string(argv[2])=="--weighted-shared-check") {
            verify_two_stage_shared(root);return 0;
        }
        if(std::string(argv[2])=="--transfer-reservation-check") {
            verify_transfer_hit_reservation(root);return 0;
        }
        if(std::string(argv[2])=="--early-no-cpu-ready-check") {
            verify_early_no_cpu_ready(root);return 0;
        }
        if(std::string(argv[2])=="--window-input-views-check") {
            for(bool strided:{false,true})for(bool gpu_cold:{false,true})
                verify_early_no_cpu_ready(root,true,strided,gpu_cold);
            return 0;
        }
        if(std::string(argv[2])=="--transfer-poll-check") {
            if(!std::getenv("MFQ_TEST_TRANSFER_POLL"))throw std::runtime_error("missing transfer poll comparison option");
            for(int tokens:{1,2,6})for(bool gpu_cold:{false,true})
                verify_early_no_cpu_ready(root,true,false,gpu_cold,tokens);
            return 0;
        }
        if(std::string(argv[2])=="--kv-direct-read-check") {
            if(!std::getenv("MFQ_TEST_KV_DIRECT_READ"))throw std::runtime_error("missing direct KV comparison option");
            const std::vector<int> token_groups=std::getenv("MFQ_TEST_DIRECT_PACKET_BATCH")?
                std::vector<int>{1,2,3,4,6,8}:std::vector<int>{1,2,6};
            for(int tokens:token_groups)for(bool gpu_cold:{true,false}) {
                if(!gpu_cold && std::getenv("MFQ_TEST_KV_DIRECT_GPU_ONLY"))continue;
                verify_early_no_cpu_ready(root,true,false,gpu_cold,tokens);
            }
            return 0;
        }
        if(std::string(argv[2])=="--mtp-kv-lru-check") {
            if(!std::getenv("MFQ_TEST_KV_DIRECT_READ") || !std::getenv("MFQ_TEST_MTP_KV_LRU"))
                throw std::runtime_error("missing speculative KV LRU comparison options");
            for(int tokens:{2,4,6})verify_early_no_cpu_ready(root,true,false,true,tokens);return 0;
        }
        if(std::string(argv[2])=="--kv-direct-read-serial-check") {
            if(!std::getenv("MFQ_TEST_KV_DIRECT_READ"))throw std::runtime_error("missing serial direct KV comparison option");
            for(int tokens:{1,4,6})verify_early_no_cpu_ready(root,false,false,true,tokens,true);
            return 0;
        }
        if(std::string(argv[2])=="--kv-direct-read-adapt-check") {
            if(!std::getenv("MFQ_TEST_KV_DIRECT_ADAPT"))throw std::runtime_error("missing adaptive direct KV comparison option");
            verify_early_no_cpu_ready(root,false,false,true,4);return 0;
        }
        if(std::string(argv[2])=="--shared-window-input-views-check") {
            if(!mfq::cuda::runtime_options::window_input_views())
                throw std::runtime_error("shared input-view check requires MFQ_MFE_WINDOW_INPUT_VIEWS=1");
            verify_two_stage_shared(root,false,false,true);return 0;
        }
        if(std::string(argv[2])!="--residency-copy-bench")throw std::runtime_error("unknown residency benchmark mode");
        residency_copy_benchmark(root);return 0;
    }
    verify_transfer_hit_reservation(root);
    verify_rounded_swiglu();
    verify_resident_classification();
    verify_active_nvq(root);
    const int shared_input_cases=verify_shared_input(root);
    const int warm_residency_cases=verify_warm_residency(root);
    const int nint_dispatch_cases=verify_nint_dispatch(root);
    const int projection_heat_cases=verify_projection_heat(root);
    const int two_stage_shared_cases=verify_two_stage_shared(root);
    const int transfer_cache_shared_cases=verify_two_stage_shared(root,true);
    const std::vector<std::string> formats={"nint4","nint5","nint6","nint8","nvq1-s","nvq1-l","nvq2j","nvq2j-l","nvq2j-xl","nvq3j","nvq3j-512","nvq3j-l"};
    int cases=0,transactions=0,adaptive_cases=0,registered_ram_cases=0,cpu_profile_cases=0;
    int dma_profile_cases=0;
    std::int64_t dma_profile_samples=0,dma_profile_window_samples=0;
    std::int64_t dma_profile_window_copies=0,dma_profile_window_bytes=0;
    int shared_cost_cases=0,shared_cost_warm_cases=0,two_stage_format_cases=0;std::int64_t shared_cost_ready_queries=0;
    const int wire_checks=verify_wire_bytes();
    for(const auto& format:formats) {
        CudaExecutionContext execution;
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> cold={source(root/(format+"-640-2560.mfq")),
            source(root/(format+"-640-2560.mfq")),source(root/(format+"-2560-640.mfq"))};
        std::array<std::shared_ptr<MoeQuantRangeSource>,3> oracle={source(root/(format+"-640-2560.mfq")),
            source(root/(format+"-640-2560.mfq")),source(root/(format+"-2560-640.mfq"))};
        std::size_t gpu_bytes=0;for(const auto& s:cold)gpu_bytes+=bytes(s->metadata()->pools[0]);
        execution.config.moe_host_cache_bytes=3*gpu_bytes;
        auto cache=make_moe_expert_cache(gpu_bytes,execution.config);
        std::vector<std::shared_ptr<MfeWeight>> gu,down;
        std::array<MfeWeight,3> baseline;
        for(int i=0;i<3;++i) {
            auto model=mfq::open_model_source((root/(format+(i==2 ? "-2560-640.mfq" : "-640-2560.mfq"))).string());
            baseline[i]=stage_cpu_mixed_moe(make_mixed_moe_runtime(load_mfe_cpu(*model,"linear.weight"),false),execution.config);
            auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(cache,std::to_string(i),cold[i],1,0,std::to_string(i)));
            (i==2 ? down : gu).push_back(weight);
        }
        mfq::MoeDispatchPlan observed_dispatch;
        int observed_dispatch_cases=0;
        auto ffn=make_moe_ffn_pipeline(gu,down,{},
            [&](const mfq::MoeDispatchPlan& plan){observed_dispatch=plan;});
        if(!ffn)throw std::runtime_error("MFQ FFN was not enabled");
        finalize_moe_expert_cache(cache);
        const auto initial_host=cold[0]->host_cache()->stats();
        std::ifstream f(root/(format+"-640-2560.input.f32"),std::ios::binary);
        std::vector<float> inputs(3*2560);f.read(reinterpret_cast<char*>(inputs.data()),inputs.size()*sizeof(float));
        if(!f)throw std::runtime_error("missing real input fixture");
        auto verify_original=[&] {
            for(int i=0;i<3;++i) {
                if(!cold[i]->gpu_resident(0) || cold[i]->gpu_resident(2) ||
                    !cold[i]->host_cache()->contains(cold[i]->host_key(2)) ||
                    cold[i]->host_cache()->contains(cold[i]->host_key(0)))
                    throw std::runtime_error("failed bundle exchange changed tier ownership");
                const int width=i==2 ? 640 : 2560,output=i==2 ? 2560 : 640;
                auto x=tb::tensor(std::vector<float>(inputs.begin(),inputs.begin()+width)).reshape({1,width}).to(tb::kFloat16).to(tb::kCUDA);
                auto ids=tb::zeros({1,1},x.options().dtype(tb::kInt32));auto route=build_moe_route_plan(ids,3);
                auto actual=(i==2 ? down[0] : gu[i])->forward(execution,x,route).to(tb::kCPU).to(tb::kFloat32).contiguous();
                auto expected=baseline[i].forward(execution,x,route).to(tb::kCPU).to(tb::kFloat32).contiguous();
                for(int64_t j=0;j<actual.numel();++j)if(!std::isfinite(actual.data_ptr<float>()[j]) ||
                    std::abs(actual.data_ptr<float>()[j]-expected.data_ptr<float>()[j])>2e-5+1e-3*std::abs(expected.data_ptr<float>()[j]))
                    throw std::runtime_error("failed bundle exchange corrupted an outgoing GPU expert");
                auto lease=cold[i]->acquire_expert(2);auto cpu=x.to(tb::kCPU).to(tb::kFloat32);
                actual=linear(lease->weights,output,width).forward(execution,cpu).to(tb::kFloat32).contiguous();
                expected=linear(oracle[i]->read_expert(2),output,width).forward(execution,cpu).to(tb::kFloat32).contiguous();
                if(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*sizeof(float)))
                    throw std::runtime_error("failed bundle exchange corrupted an incoming RAM expert");
                if(cold[i]->expert_disk_reads_after_preload())throw std::runtime_error("bundle exchange rollback reread SSD");
            }
            const auto host=cold[0]->host_cache()->stats();
            if(host.managed_bytes!=initial_host.managed_bytes || host.managed_peak_bytes!=initial_host.managed_peak_bytes)
                throw std::runtime_error("bundle exchange rollback duplicated RAM experts");
        };
        for(const std::string phase:{"upload","map","host","publish"})for(int fail_at=0;fail_at<(phase=="publish" ? 1 : 3);++fail_at) {
            MoeResidencyManager manager(cache.get(),[&](const char* event,std::size_t index) {
                if(phase==event && index==std::size_t(fail_at))throw std::runtime_error("injected bundle exchange failure");
            });
            for(int step=0;step<4;++step)manager.after_layer(0,{2,2,2},1);
            bool rejected=false;try{manager.apply_pending();}catch(const std::runtime_error& e){rejected=std::string(e.what())=="injected bundle exchange failure";}
            if(!rejected)throw std::runtime_error("bundle exchange swallowed an injected failure");
            verify_original();++transactions;
        }
        {
            MoeHostExpertCache::Lease reader;
            MoeResidencyManager manager(cache.get(),[&](const char* event,std::size_t index) {
                if(std::string(event)=="upload" && index==2)reader=cold[1]->acquire_expert(2);
            });
            for(int step=0;step<4;++step)manager.after_layer(0,{2,2,2},1);
            manager.apply_pending();if(!reader)throw std::runtime_error("late CPU reader was not exercised");
            verify_original();reader.reset();++transactions;
        }
        {
            CudaExecutionContext cross;cross.config.moe_host_cache_bytes=6*gpu_bytes;
            auto tier=make_moe_expert_cache(2*gpu_bytes,cross.config);
            std::array<std::array<std::shared_ptr<MoeQuantRangeSource>,3>,2> layers;
            std::array<MoeFfnForward,2> forwards;
            for(int layer=0;layer<2;++layer) {
                std::vector<std::shared_ptr<MfeWeight>> gates,downs;
                for(int i=0;i<3;++i) {
                    layers[layer][i]=source(root/(format+(i==2 ? "-2560-640.mfq" : "-640-2560.mfq")));
                    auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(tier,
                        std::to_string(layer)+"/"+std::to_string(i),layers[layer][i],1,layer,std::to_string(i)));
                    (i==2 ? downs : gates).push_back(weight);
                }
                forwards[layer]=make_moe_ffn_pipeline(gates,downs);
            }
            finalize_moe_expert_cache(tier);const auto host_before=layers[0][0]->host_cache()->stats();
            MoeResidencyManager manager(tier.get());
            for(int step=0;step<4;++step) {
                manager.after_layer(0,{0},1);
                manager.after_layer(1,{1,1,1,1,1,1,2,2,2,2,2,2},1);
            }
            manager.apply_pending();
            for(int i=0;i<3;++i)if(layers[0][i]->gpu_resident(0) || layers[1][i]->gpu_resident(0) ||
                    !layers[1][i]->gpu_resident(1) || !layers[1][i]->gpu_resident(2))
                throw std::runtime_error("cross-layer heat did not move whole expert bundles to the busy layer");
            auto x=tb::tensor(std::vector<float>(inputs.begin(),inputs.begin()+2560)).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA);
            auto ids=tb::tensor(std::vector<int32_t>{1,2}).reshape({1,2}).to(tb::kCUDA);
            auto weights=tb::ones({1,2},x.options().dtype(tb::kFloat32));auto route=build_moe_route_plan(ids,3);
            auto actual=forwards[1](cross,x,ids,weights).to(tb::kCPU).to(tb::kFloat32).contiguous();
            auto g=baseline[0].forward(cross,x,route),u=baseline[1].forward(cross,x,route);
            auto expected=baseline[2].forward(cross,(g*tb::sigmoid(g))*u,route).to(tb::kCPU).to(tb::kFloat32).contiguous();
            for(int64_t j=0;j<actual.numel();++j)if(!std::isfinite(actual.data_ptr<float>()[j]) ||
                    std::abs(actual.data_ptr<float>()[j]-expected.data_ptr<float>()[j])>2e-5+1e-3*std::abs(expected.data_ptr<float>()[j]))
                throw std::runtime_error("cross-layer bundle exchange changed GPU arithmetic");
            for(const auto& layer:layers)for(int i=0;i<3;++i) {
                const int width=i==2 ? 640 : 2560,output=i==2 ? 2560 : 640;
                auto cpu=tb::tensor(std::vector<float>(inputs.begin(),inputs.begin()+width)).reshape({1,width});
                auto held=layer[i]->acquire_expert(0);
                actual=linear(held->weights,output,width).forward(cross,cpu).to(tb::kFloat32).contiguous();
                expected=linear(oracle[i]->read_expert(0),output,width).forward(cross,cpu).to(tb::kFloat32).contiguous();
                if(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.numel()*sizeof(float)))
                    throw std::runtime_error("cross-layer demotion changed CPU arithmetic");
                if(layer[i]->expert_disk_reads_after_preload())throw std::runtime_error("cross-layer exchange reread expert SSD");
            }
            const auto host_after=layers[0][0]->host_cache()->stats();
            if(host_after.managed_bytes!=host_before.managed_bytes || host_after.managed_peak_bytes!=host_before.managed_peak_bytes)
                throw std::runtime_error("cross-layer exchange duplicated RAM weights");
            ++transactions;
        }
        {
            CudaExecutionContext partial;partial.config.moe_host_cache_bytes=3*gpu_bytes;
            auto tier=make_moe_expert_cache(gpu_bytes+bytes(oracle[0]->metadata()->pools[0]),partial.config);
            std::array<std::shared_ptr<MoeQuantRangeSource>,3> parts;
            std::vector<std::shared_ptr<MfeWeight>> gates,downs;
            for(int i=0;i<3;++i) {
                parts[i]=source(root/(format+(i==2 ? "-2560-640.mfq" : "-640-2560.mfq")));
                auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(tier,std::to_string(i),parts[i],1,0,std::to_string(i)));
                (i==2 ? downs : gates).push_back(weight);
            }
            auto forward=make_moe_ffn_pipeline(gates,downs);finalize_moe_expert_cache(tier);
            int mixed=-1;
            for(int expert=0;expert<3;++expert) {
                int hits=0;for(const auto& part:parts)hits+=part->gpu_resident(expert);
                if(hits && hits!=3){mixed=expert;break;}
            }
            if(mixed<0)throw std::runtime_error("partial-residency fixture did not create a split expert");
            auto x=tb::tensor(std::vector<float>(inputs.begin(),inputs.begin()+2560)).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA);
            auto ids=tb::tensor(std::vector<int32_t>{mixed,mixed,mixed}).reshape({1,3}).to(tb::kCUDA);
            auto weights=tb::ones({1,3},x.options().dtype(tb::kFloat32));auto route=build_moe_route_plan(ids,3);
            auto actual=forward(partial,x,ids,weights).to(tb::kCPU).to(tb::kFloat32).contiguous();
            auto g=baseline[0].forward(partial,x,route),u=baseline[1].forward(partial,x,route);
            auto expected=baseline[2].forward(partial,(g*tb::sigmoid(g))*u,route).to(tb::kCPU).to(tb::kFloat32).contiguous();
            for(int64_t j=0;j<actual.numel();++j)if(!std::isfinite(actual.data_ptr<float>()[j]) ||
                    std::abs(actual.data_ptr<float>()[j]-expected.data_ptr<float>()[j])>2e-5+1e-3*std::abs(expected.data_ptr<float>()[j]))
                throw std::runtime_error("partial expert GPU/RAM assembly changed arithmetic");
            for(const auto& part:parts)if(part->expert_disk_reads_after_preload())
                throw std::runtime_error("partial expert reread SSD instead of using its GPU/RAM fields");
            ++transactions;
        }
        for(int tokens:{1,3,12})for(int iteration=0;iteration<8;++iteration) {
            finish_moe_expert_exchanges(cache);
            if(tokens==1 && iteration==4) {
                for(const auto& s:cold)if(!s->gpu_resident(2) || s->gpu_resident(0) ||
                    !s->host_cache()->contains(s->host_key(0)) || s->host_cache()->contains(s->host_key(2)))
                    throw std::runtime_error(format+" routing heat did not exchange VRAM and RAM ownership: gpu0="+
                        std::to_string(s->gpu_resident(0))+" gpu2="+std::to_string(s->gpu_resident(2))+
                        " ram0="+std::to_string(s->host_cache()->contains(s->host_key(0)))+
                        " ram2="+std::to_string(s->host_cache()->contains(s->host_key(2))));
            }
            std::vector<float> input(tokens*2560);for(int t=0;t<tokens;++t)std::copy_n(inputs.data()+(t%3)*2560,2560,input.data()+t*2560);
            auto x=tb::tensor(input).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA);
            std::vector<int32_t> selected(tokens*3);
            for(int p=0;p<tokens*3;++p)selected[p]=iteration<2 ? (p+iteration)%3 : iteration<6 ? 2 : 0;
            auto ids=tb::tensor(selected).reshape({tokens,3}).to(tb::kCUDA);
            auto weights=tb::ones({tokens,3},x.options().dtype(tb::kFloat32));
            auto actual=ffn(execution,x,ids,weights).to(tb::kCPU).to(tb::kFloat32).contiguous();
            if(observed_dispatch.kinds.size()!=selected.size())
                throw std::runtime_error("FFN dispatch observer omitted route assignments");
            ++observed_dispatch_cases;
            auto route=build_moe_route_plan(ids,3);
            auto gu_gpu=tb::cat({baseline[0].forward(execution,x,route),baseline[1].forward(execution,x,route)},-1);
            auto g=gu_gpu.narrow(-1,0,640),u=gu_gpu.narrow(-1,640,640);
            auto expected_gpu=baseline[2].forward(execution,(g*tb::sigmoid(g))*u,route).to(tb::kCPU).to(tb::kFloat32).contiguous();
            for(int t=0;t<tokens;++t)for(int r=0;r<3;++r) {
                const int e=selected[t*3+r];
                const auto kind=observed_dispatch.kinds[t*3+r];
                if(kind!=mfq::MoeDispatchKind::Cpu && kind!=mfq::MoeDispatchKind::GpuResident &&
                        kind!=mfq::MoeDispatchKind::GpuTransfer)
                    throw std::runtime_error("FFN dispatch observer assigned an invalid route kind");
                const bool gpu=kind!=mfq::MoeDispatchKind::Cpu;
                tb::Tensor expected;
                if(!gpu) {
                    auto input_cpu=x.narrow(0,t,1).to(tb::kCPU).to(tb::kFloat32).contiguous();
                    auto g=linear(oracle[0]->read_expert(e),640,2560).forward(execution,input_cpu);
                    auto u=linear(oracle[1]->read_expert(e),640,2560).forward(execution,input_cpu);
                    std::vector<float> activated(640);
                    for(int i=0;i<640;++i) {
                        const float gate=float(g.data_ptr<mfq_half>()[i]),up=float(u.data_ptr<mfq_half>()[i]);
                        const float sigmoid=float(mfq_half(1.0/(1.0+std::exp(-double(gate)))));
                        activated[i]=float(mfq_half(float(mfq_half(gate*sigmoid))*up));
                    }
                    auto hidden=tb::tensor(activated).reshape({1,640});
                    expected=linear(oracle[2]->read_expert(e),2560,640).forward(execution,hidden.to(tb::kFloat32)).to(tb::kFloat32);
                }
                for(int column=0;column<2560;++column) {
                    const float reference=gpu ? expected_gpu.data_ptr<float>()[(t*3+r)*2560+column] : expected.data_ptr<float>()[column];
                    const float value=actual.data_ptr<float>()[(t*3+r)*2560+column];
                    if(std::isfinite(value) && std::isfinite(reference) &&
                        std::abs(value-reference)<=2e-5+1e-3*std::abs(reference))continue;
                    throw std::runtime_error(format+" FFN graph/tier mismatch tokens="+std::to_string(tokens)+
                        " iteration="+std::to_string(iteration)+" at "+std::to_string(t*3+r)+","+
                        std::to_string(column)+" value="+std::to_string(value)+" expected="+std::to_string(reference));
                }
            }
            for(const auto& s:cold)if(s->expert_disk_reads_after_preload()!=0)throw std::runtime_error("FFN replay reread expert SSD");
            const auto host=cold[0]->host_cache()->stats();
            if(host.managed_bytes!=initial_host.managed_bytes || host.managed_peak_bytes!=initial_host.managed_peak_bytes)
                throw std::runtime_error("adaptive exchange allocated another retained RAM expert");
            ++cases;
        }
        std::cout<<"ffn_dispatch_oracle format="<<format<<" observed_cases="<<observed_dispatch_cases<<'\n';
        {
            auto adaptive_stream=mfq_get_stream_from_pool();MfqCudaStreamGuard adaptive_guard(adaptive_stream);
            CudaExecutionContext adaptive;adaptive.config.moe_ram_pcie_fraction.reset();
            const auto* shared_cost=std::getenv("MFQ_MOE_SHARED_CPU_COST");
            const bool shared_cost_enabled=shared_cost && std::strcmp(shared_cost,"1")==0;
            // Keep whole RAM experts eligible while obtaining repeated cost
            // observations; projection-wise exchanges are covered separately.
            if(shared_cost_enabled)adaptive.config.moe_residency_projection_heat=false;
            adaptive.config.moe_host_cache_bytes=3*gpu_bytes;
            auto tier=make_moe_expert_cache(gpu_bytes,adaptive.config);
            std::array<std::shared_ptr<MoeQuantRangeSource>,3> parts;
            std::vector<std::shared_ptr<MfeWeight>> gates,downs;
            for(int i=0;i<3;++i) {
                parts[i]=source(root/(format+(i==2 ? "-2560-640.mfq" : "-640-2560.mfq")));
                auto weight=std::make_shared<MfeWeight>(cache_quant_moe_weight(tier,
                    "adaptive/"+std::to_string(i),parts[i],1,0,std::to_string(i)));
                (i==2 ? downs : gates).push_back(weight);
            }
            mfq::MoeDispatchPlan observed;
            auto forward=make_moe_ffn_pipeline(gates,downs,{},[&](const mfq::MoeDispatchPlan& plan){observed=plan;});
            finalize_moe_expert_cache(tier);
            auto x=tb::tensor(std::vector<float>(inputs.begin(),inputs.begin()+2560)).reshape({1,2560}).to(tb::kFloat16).to(tb::kCUDA);
            auto ids=tb::zeros({1,5},x.options().dtype(tb::kInt32));
            auto weights=tb::ones({1,5},x.options().dtype(tb::kFloat32));
            mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());tb::Tensor output;
            window.capture([&]{output=forward(adaptive,x,ids,weights);},[]{},[&]{output={};});
            for(int step=shared_cost_enabled?-4:0;step<8;++step) {
                finish_moe_expert_exchanges(tier);
                std::vector<int32_t> selected={step<0?0:step%3,1,2,1,2};
                ids.copy_(tb::tensor(selected).reshape({1,5}).to(tb::kCUDA));
                window.run();
                auto actual=output.to(tb::kCPU).to(tb::kFloat32).contiguous();
                const auto route=build_moe_route_plan(ids,3);
                auto g=baseline[0].forward(adaptive,x,route),u=baseline[1].forward(adaptive,x,route);
                auto expected=baseline[2].forward(adaptive,(g*tb::sigmoid(g))*u,route).to(tb::kCPU).to(tb::kFloat32).contiguous();
                if(observed.kinds.size()!=selected.size())throw std::runtime_error("adaptive dispatch was not observed");
                for(std::size_t r=0;r<selected.size();++r) {
                    tb::Tensor expected_cpu;
                    const bool cpu=observed.kinds[r]==mfq::MoeDispatchKind::Cpu;
                    if(cpu) {
                        const int e=selected[r];
                        auto input_cpu=x.to(tb::kCPU).to(tb::kFloat32).contiguous();
                        auto gate=linear(oracle[0]->read_expert(e),640,2560).forward(adaptive,input_cpu);
                        auto up=linear(oracle[1]->read_expert(e),640,2560).forward(adaptive,input_cpu);
                        std::vector<float> activated(640);
                        for(int i=0;i<640;++i) {
                            const float gv=float(gate.data_ptr<mfq_half>()[i]),uv=float(up.data_ptr<mfq_half>()[i]);
                            const float sigmoid=float(mfq_half(1.0/(1.0+std::exp(-double(gv)))));
                            activated[i]=float(mfq_half(float(mfq_half(gv*sigmoid))*uv));
                        }
                        auto hidden=tb::tensor(activated).reshape({1,640});
                        expected_cpu=linear(oracle[2]->read_expert(e),2560,640).forward(adaptive,hidden).to(tb::kFloat32).contiguous();
                    }
                    for(int column=0;column<2560;++column) {
                        const float reference=cpu ? expected_cpu.data_ptr<float>()[column] : expected.data_ptr<float>()[r*2560+column];
                        const float value=actual.data_ptr<float>()[r*2560+column];
                        if(!std::isfinite(value) || std::abs(value-reference)>2e-5+1e-3*std::abs(reference))
                            throw std::runtime_error(format+" measured adaptive window changed canonical output step="+
                                std::to_string(step)+" route="+std::to_string(r)+" expert="+std::to_string(selected[r])+
                                " cpu="+std::to_string(cpu)+" column="+std::to_string(column)+
                                " value="+std::to_string(value)+" expected="+std::to_string(reference));
                    }
                }
                for(const auto& part:parts)if(part->expert_disk_reads_after_preload())
                    throw std::runtime_error("measured adaptive window reread expert SSD");
                if(step<0)++shared_cost_warm_cases;else ++adaptive_cases;
            }
            std::ostringstream cost_stats;print_moe_expert_cache_stats(tier,cost_stats);
            const auto cost_counter=[&](const char* name) {
                const auto text=cost_stats.str(),field=std::string(name)+"=";
                const auto position=text.find(field);
                if(position==std::string::npos)throw std::runtime_error("missing shared CPU cost evidence");
                return std::stoll(text.substr(position+field.size()));
            };
            const auto* enabled=std::getenv("MFQ_MOE_SHARED_CPU_COST");
            if(enabled && std::strcmp(enabled,"1")==0) {
                if(cost_counter("pipeline_shared_cpu_cost_enabled")!=1 ||
                   cost_counter("pipeline_cpu_cost_queries")<=0 || cost_counter("pipeline_cpu_cost_samples")<=0)
                    throw std::runtime_error("shared CPU cost mode did not observe real adaptive work");
                const auto* background=std::getenv("MFQ_MOE_CPU_BACKGROUND_CALIBRATION");
                const auto* budget=std::getenv("MFQ_MOE_CPU_TRANSFER_BUDGET");
                const bool expected_background=(!background || std::strcmp(background,"1")==0) &&
                    (!budget || budget[0]!='0');
                const auto jobs=cost_counter("pipeline_cpu_calibration_jobs");
                if((expected_background && jobs<=0) || (!expected_background && jobs) ||
                   jobs!=cost_counter("pipeline_cpu_calibration_observations"))
                    throw std::runtime_error("CPU calibration execution/completion differs from selected mode");
                shared_cost_ready_queries+=cost_counter("pipeline_cpu_cost_ready_queries");
                ++shared_cost_cases;
            } else if(cost_counter("pipeline_shared_cpu_cost_enabled") || cost_counter("pipeline_cpu_cost_samples"))
                throw std::runtime_error("disabled shared CPU cost mode collected samples");
        }
        // Two successive expert layers share transfer arenas. Changing input
        // and routes must reach both layers in one launch without a wait on an
        // event queued behind the same, still-in-flight graph.
        mfq::MoeDispatchPlan second_dispatch;
        auto second=make_moe_ffn_pipeline(gu,down,{},
            [&](const mfq::MoeDispatchPlan& plan){second_dispatch=plan;});
        for(int tokens:{1,3,12}) {
            auto x=tb::zeros({tokens,2560},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
            auto ids=tb::zeros({tokens,3},x.options().dtype(tb::kInt32));
            auto weights=tb::ones({tokens,3},x.options().dtype(tb::kFloat32));
            mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
            tb::Tensor output,graph_first;
            for(int step=0;step<3;++step) {
                finish_moe_expert_exchanges(cache);
                std::vector<float> input(tokens*2560);for(int i=0;i<tokens*2560;++i)input[i]=inputs[(i+step*17)%inputs.size()];
                std::vector<int32_t> selected(tokens*3);for(int i=0;i<tokens*3;++i)selected[i]=(i+step)%3;
                x.copy_(tb::tensor(input).reshape({tokens,2560}).to(tb::kFloat16).to(tb::kCUDA));
                ids.copy_(tb::tensor(selected).reshape({tokens,3}).to(tb::kCUDA));
                auto first=ffn(execution,x,ids,weights);
                // Synthetic fixture weights amplify coherent inputs strongly.
                // Bound the inter-layer input while keeping its graph dependency.
                const auto next_input=tb::tanh(first.mean(1))*.0001;
                const auto eager=second(execution,next_input,ids,weights).cpu().to(tb::kFloat32).contiguous();
                const auto expected_eager=moe_ffn_reference_parts(execution,next_input,ids,
                    baseline,oracle,second_dispatch);
                for(int64_t i=0;i<eager.numel();++i)if(!std::isfinite(eager.data_ptr<float>()[i]) ||
                    std::abs(eager.data_ptr<float>()[i]-expected_eager.data_ptr<float>()[i])>
                        2e-5+1e-3*std::abs(expected_eager.data_ptr<float>()[i]))
                    throw std::runtime_error(format+
                        " two-layer eager output differs from its observed dispatch oracle");
                window.capture([&] {
                    graph_first=ffn(execution,x,ids,weights);
                    output=second(execution,tb::tanh(graph_first.mean(1))*.0001,ids,weights);
                },[]{},[&]{output={};graph_first={};});
                window.run();auto actual=output.to(tb::kCPU).to(tb::kFloat32).contiguous();
                const auto first_actual=graph_first.cpu().to(tb::kFloat32).contiguous();
                const auto first_expected=moe_ffn_reference_parts(execution,x,ids,baseline,oracle,observed_dispatch);
                for(int64_t i=0;i<first_actual.numel();++i)if(!std::isfinite(first_actual.data_ptr<float>()[i]) ||
                    std::abs(first_actual.data_ptr<float>()[i]-first_expected.data_ptr<float>()[i])>
                        2e-5+1e-3*std::abs(first_expected.data_ptr<float>()[i]))
                    throw std::runtime_error(format+
                        " two-layer graph first layer differs from its observed dispatch oracle");
                const auto expected=moe_ffn_reference_parts(execution,tb::tanh(graph_first.mean(1))*.0001,
                    ids,baseline,oracle,second_dispatch);
                for(int64_t i=0;i<actual.numel();++i)if(!std::isfinite(actual.data_ptr<float>()[i]) ||
                    std::abs(actual.data_ptr<float>()[i]-expected.data_ptr<float>()[i])>2e-5+1e-3*std::abs(expected.data_ptr<float>()[i]))
                    throw std::runtime_error(format+" two-layer window mismatch tokens="+std::to_string(tokens)+" step="+std::to_string(step)+" at "+std::to_string(i)+" actual="+std::to_string(actual.data_ptr<float>()[i])+" expected="+std::to_string(expected.data_ptr<float>()[i]));
                for(const auto& s:cold)if(s->expert_disk_reads_after_preload())throw std::runtime_error("window reread resident expert SSD");
                ++cases;
            }
        }
        {
            mfq::cuda::DecodeWindow failed(mfq_current_cuda_stream());
            auto x=tb::zeros({1,2560},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
            auto ids=tb::tensor(std::vector<int32_t>{0,1,2}).reshape({1,3}).to(tb::kCUDA);
            auto weights=tb::ones({1,3},x.options().dtype(tb::kFloat32));
            failed.capture([&]{failed.enroll({&failed,[]{},[]{throw std::runtime_error("injected host read failure");},{},{}});ffn(execution,x,ids,weights);},[]{});
            bool rejected=false;try{failed.run();}catch(const std::runtime_error& e){rejected=std::string(e.what())=="injected host read failure";}
            if(!rejected)throw std::runtime_error("window swallowed its host failure");
            MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
            rejected=false;try{failed.run();}catch(const std::runtime_error&){rejected=true;}
            if(!rejected)throw std::runtime_error("failed window accepted another replay");
        }
        std::ostringstream statistics;print_moe_expert_cache_stats(cache,statistics);
        auto counter=[&](const std::string& name) {
            std::smatch match;const auto text=statistics.str();
            if(!std::regex_search(text,match,std::regex(name+"=([0-9]+)")))
                throw std::runtime_error("missing RAM transfer counter "+name);
            return std::stoll(match[1]);
        };
        if(const auto* value=std::getenv("MFQ_TRACE_CPU_ROWS");value && std::strcmp(value,"1")==0) {
            const auto text=statistics.str();
            const auto* fail=std::getenv("MFQ_TRACE_CPU_ROWS_FAIL_ALLOC");
            if(fail && std::strcmp(fail,"1")==0) {
                if(text.find("moe_cpu_profile enabled=0 dropped=1 samples=0")==std::string::npos)
                    throw std::runtime_error("optional CPU trace allocation failure did not disable tracing");
            } else {
                if(text.find("moe_cpu_profile enabled=1 dropped=0")==std::string::npos)
                    throw std::runtime_error("CPU trace lost its observations");
                std::istringstream lines(text);std::string line;std::int64_t projections=0;
                while(std::getline(lines,line))if(line.rfind("moe_cpu_projection ",0)==0) {
                    const auto field=[&](const char* name) {
                        std::smatch found;
                        if(!std::regex_search(line,found,std::regex(std::string(name)+"=([0-9]+)")))
                            throw std::runtime_error("CPU projection observation missing field");
                        return std::stoll(found[1]);
                    };
                    if(field("rows")!=field("outputs") || field("positions")<=0 || field("callbacks")<=0)
                        throw std::runtime_error("CPU trace omitted or duplicated projection rows");
                    ++projections;
                }
                if(!projections || projections!=3*counter("hybrid_cpu_experts"))
                    throw std::runtime_error("CPU trace projection count differs from real dispatch");
            }
            ++cpu_profile_cases;
        }
        if(const auto* value=std::getenv("MFQ_TRACE_MOE_DMA");value && std::strcmp(value,"1")==0) {
            const auto text=statistics.str();const auto* fail=std::getenv("MFQ_TRACE_MOE_DMA_FAIL_ALLOC");
            if(fail && std::strcmp(fail,"1")==0) {
                if(text.find("moe_dma_profile enabled=0 dropped=1 samples=0")==std::string::npos)
                    throw std::runtime_error("optional DMA trace allocation failure did not disable tracing");
            } else {
                std::smatch head;
                if(!std::regex_search(text,head,std::regex("moe_dma_profile enabled=1 dropped=0 samples=([0-9]+) skipped=([0-9]+)")))
                    throw std::runtime_error("DMA trace lost its observations");
                std::istringstream lines(text);std::string line;
                std::int64_t samples=0,windows=0,copies=0,bytes=0;
                std::vector<std::int64_t> serves;
                while(std::getline(lines,line))if(line.rfind("moe_dma_call ",0)==0) {
                    const auto field=[&](const char* name) {
                        std::smatch found;
                        if(!std::regex_search(line,found,std::regex(std::string("(?:^| )")+name+"=(-?[0-9]+)")))
                            throw std::runtime_error("DMA observation missing field");
                        return std::stoll(found[1]);
                    };
                    const auto serve=field("serve");
                    if(std::find(serves.begin(),serves.end(),serve)!=serves.end())
                        throw std::runtime_error("DMA trace duplicated a served call");
                    serves.push_back(serve);
                    // CUDA event timestamps have approximately 0.5 us resolution.
                    // Independently measured intervals must agree within 2 us.
                    if(field("copies")<=0 || field("copy_bytes")<field("payload_bytes") ||
                        field("first_copy_bytes")<=0 || field("first_copy_bytes")>field("copy_bytes") ||
                        field("copy_ns")<0 || field("notice_ns")<0 || field("hot_ns")<=0 ||
                        std::abs(field("copy_ns")-(field("copy_end_after_hot_ns")-field("copy_begin_after_hot_ns")))>2000 ||
                        std::abs(field("notice_ns")-(field("notice_end_after_hot_ns")-field("copy_end_after_hot_ns")))>2000 ||
                        field("prepare_ns")+field("enqueue_ns")>field("fetch_ns"))
                        throw std::runtime_error("DMA trace has inconsistent transfer timestamps or bytes");
                    ++samples;
                    if(field("window")) {++windows;copies+=field("copies");bytes+=field("copy_bytes");}
                }
                if(samples!=std::stoll(head[1]) || (!samples && std::stoll(head[2])==0))
                    throw std::runtime_error("DMA trace sample count differs from recorded calls");
                if(std::stoll(head[2])==0 &&
                    (copies!=counter("pipeline_window_dma_copies") || bytes!=counter("pipeline_window_dma_bytes")))
                    throw std::runtime_error("DMA trace window byte/copy counts differ from real submissions");
                dma_profile_samples+=samples;dma_profile_window_samples+=windows;
                dma_profile_window_copies+=copies;dma_profile_window_bytes+=bytes;
            }
            ++dma_profile_cases;
        } else if(statistics.str().find("moe_dma_")!=std::string::npos)
            throw std::runtime_error("disabled DMA profiling still produced observations");
        if(execution.config.moe_two_stage_ffn) {
            if(counter("pipeline_two_stage_serves")<=0)throw std::runtime_error(format+" did not serve the two-stage candidate");
            ++two_stage_format_cases;
        }
        if(execution.config.moe_direct_ram) {
            if(counter("pipeline_ram_registered_bytes")<=0 || counter("pipeline_direct_ram_bytes")<=0 ||
                counter("pipeline_staged_ram_bytes")!=0) {
                throw std::runtime_error(format+
                    " did not retain registered RAM transfers across cache exchanges: registered="+
                    std::to_string(counter("pipeline_ram_registered_bytes"))+" direct="+
                    std::to_string(counter("pipeline_direct_ram_bytes"))+" staged="+
                    std::to_string(counter("pipeline_staged_ram_bytes")));
            }
            ++registered_ram_cases;
        }else if(counter("pipeline_ram_registered_bytes") || counter("pipeline_direct_ram_bytes"))
            throw std::runtime_error("disabled direct RAM transfer still registered memory");
    }
    std::cout<<"MFQ FFN canonical graph/tier/heat exchange checks PASS cases="<<cases
        <<" transactions="<<transactions<<" adaptive="<<adaptive_cases<<" wire="<<wire_checks<<'\n';
    std::cout<<"registered_ram_cases="<<registered_ram_cases<<'\n';
    std::cout<<"two_stage_format_cases="<<two_stage_format_cases<<" two_stage_shared_cases="<<two_stage_shared_cases<<'\n';
    std::cout<<"transfer_cache_shared_cases="<<transfer_cache_shared_cases<<'\n';
    std::cout<<"cpu_profile_cases="<<cpu_profile_cases<<" optional observations/failure continuity checked\n";
    std::cout<<"dma_profile_cases="<<dma_profile_cases<<" dma_profile_samples="<<dma_profile_samples
        <<" dma_profile_window_samples="<<dma_profile_window_samples
        <<" dma_profile_window_copies="<<dma_profile_window_copies
        <<" dma_profile_window_bytes="<<dma_profile_window_bytes<<" optional observations/failure continuity checked\n";
    if(shared_cost_cases && !shared_cost_ready_queries)throw std::runtime_error("adaptive dispatch never consumed a calibrated shared CPU rate");
    std::cout<<"shared_cpu_cost_cases="<<shared_cost_cases<<" calibrated_queries="<<shared_cost_ready_queries
        <<" numeric_warm_cases="<<shared_cost_warm_cases<<'\n';
    std::cout<<"shared_input_cases="<<shared_input_cases<<" mixed Gate/Up outputs, bytes and scales exact PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
