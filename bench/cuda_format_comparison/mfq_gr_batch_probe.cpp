// Actual GR weights; the reference uses the service's unfused composition.
#define main single_row_gr_main
#include "mfq_gr_bench.cpp"
#include "mfq_cuda_graph_capture.h"
#undef main
#include <unordered_set>
#include "mfq/kernels/cuda/qwen4_exp.h"

namespace wl=mfq::cuda::weight_loader;

static void mixed_two_plane_check(const tb::Tensor& actual,const tb::Tensor& expected) {
    auto a=actual.to(tb::kFloat32).cpu().contiguous(),b=expected.to(tb::kFloat32).cpu().contiguous();
    if(a.sizes()!=b.sizes())throw std::runtime_error("two-plane GR geometry changed");
    const bool half=actual.scalar_type()==tb::kFloat16;
    double error=0,signal=0,maximum=0;
    for(int64_t i=0;i<a.numel();++i) {
        const double x=a.data_ptr<float>()[i],y=b.data_ptr<float>()[i],diff=std::abs(x-y);
        if(!std::isfinite(x) || !std::isfinite(y) || diff>(half?.003:.0002)+.0001*std::abs(y))
            throw std::runtime_error("two-plane GR absolute error exceeds probe tolerance: index="+
                std::to_string(i)+" actual="+std::to_string(x)+" expected="+std::to_string(y)+
                " difference="+std::to_string(diff));
        error+=diff*diff;signal+=y*y;maximum=std::max(maximum,diff);
    }
    const double relative=std::sqrt(error/std::max(signal,1e-30));
    if(relative>(half?.0004:.0001))throw std::runtime_error("two-plane GR relative error exceeds probe tolerance");
    std::cout<<"GR_MIXED_ERROR half="<<half<<" relative="<<relative<<" absolute="<<maximum<<'\n';
}

struct BatchGr {
    Bank weights;
    wl::ResidualLinear down,up,inject;
    tb::Tensor floating_coefficients;
    BatchGr(CudaExecutionContext& execution,const mfq::ModelSource& model,int layer,const char* role)
        :weights(execution,model,layer,role) {
        const auto prefix="model.block."+std::to_string(layer)+"."+role+".mhc.";
        down=wl::residual_linear(execution,model,prefix+"pre.down.weight");
        up=wl::residual_linear(execution,model,prefix+"pre.up.weight");
        inject=wl::residual_linear(execution,model,prefix+"post.inject.weight");
        const auto* metadata_option=std::getenv("MFQ_GR_UP_GROUP_COEFFICIENTS");
        if(metadata_option && metadata_option[0]=='2' && up.weight->nint.aligned_q8 && !up.weight->nint.q8_zero &&
                up.weight->nint.gs==48 && up.weight->nint.out==10240 && up.weight->nint.neuron_len==320)
            floating_coefficients=prepare_gr_up_group_coefficients(up.weight->nint);
    }
    std::vector<tb::Tensor> run(CudaExecutionContext& execution,const tb::Tensor& branch,
            const tb::Tensor& residual,const tb::Tensor& prior,bool after,bool fused) const {
        if(fused) {
            const auto* coefficient_option=std::getenv("MFQ_GR_UP_GROUP_COEFFICIENTS");
            tb::Tensor coefficients;
            if(coefficient_option && coefficient_option[0]!='0')
                coefficients=coefficient_option[0]=='1' && floating_coefficients.defined()?floating_coefficients:up.up_group_coefficients;
            return gated_residual_two_stage_cuda(branch,residual,prior,weights.norm,
                down.weight->nint,up.weight->nint,inject.weight->is_nint()?&inject.weight->nint:nullptr,
                4,1e-6,after,inject.weight->is_dense()?inject.weight->dense:tb::Tensor{},inject.prepared_right,{},coefficients);
        }
        if(after)return gated_residual_pre_after_projected(execution,branch,residual,prior,weights.norm,
            down.forward,up.forward,inject.forward,2560,4,1e-6,up.mixed,{},down.activated,inject.activated);
        return gated_residual_pre_projected(execution,residual,weights.norm,down.forward,up.forward,
            inject.forward,2560,4,1e-6,up.mixed,{},down.activated,inject.activated);
    }
};

int main(int argc,char** argv)try {
    std::cout<<std::unitbuf;
    if(argc!=3 && !(argc==5 && std::string(argv[3])=="--rows"))
        throw std::runtime_error("expected model-first-shard and --check/--bench/--up-capacity-sweep/--down-capacity-sweep [--rows N]");
    const bool down_sweep=std::string(argv[2])=="--down-capacity-sweep";
    const bool group_sweep=std::string(argv[2])=="--group-read-sweep";
    const bool metadata_sweep=std::string(argv[2])=="--up-metadata-sweep";
    const bool unroll_sweep=std::string(argv[2])=="--up-unroll-sweep";
    const bool norm_sweep=std::string(argv[2])=="--norm-cache-sweep";
    const bool two_plane_sweep=std::string(argv[2])=="--two-plane-sweep";
    const bool three_plane_sweep=std::string(argv[2])=="--three-plane-sweep";
    const bool integer16_sweep=std::string(argv[2])=="--integer16-sweep";
    const bool up_integer_sweep=std::string(argv[2])=="--up-integer-sweep";
    const bool integer_best_sweep=std::string(argv[2])=="--integer-best-sweep";
    const bool up_mix_sweep=std::string(argv[2])=="--up-mix-sweep";
    const bool up_prepare_sweep=std::string(argv[2])=="--up-prepare-sweep";
    const bool up_groups_sweep=std::string(argv[2])=="--up-groups-sweep";
    const bool up_mma_sweep=std::string(argv[2])=="--up-mma-sweep";
    const bool up_mma_uniform_sweep=std::string(argv[2])=="--up-mma-uniform16-sweep";
    const bool up_mma_half_sweep=std::string(argv[2])=="--up-mma-f16-sweep";
    const bool up_mma_m1_half_sweep=std::string(argv[2])=="--up-mma-f16-m1-sweep";
    const bool up_coefficient_sweep=std::string(argv[2])=="--up-coefficient-sweep";
    const bool up_direct_s16_sweep=std::string(argv[2])=="--up-direct-s16-sweep";
    const bool up_half_register_sweep=std::string(argv[2])=="--up-half-register-sweep";
    const bool up_half_unroll_sweep=std::string(argv[2])=="--up-half-unroll-sweep";
    const bool up_fused_direct_sweep=std::string(argv[2])=="--up-fused-direct-sweep";
    const bool up_parts_vector_sweep=std::string(argv[2])=="--up-parts-vector-sweep";
    const bool up_split_major_sweep=std::string(argv[2])=="--up-split-major-sweep";
    const bool up_fused_split_sweep=std::string(argv[2])=="--up-fused-split-sweep";
    const bool up_direct_stats_sweep=std::string(argv[2])=="--up-direct-stats-sweep";
    const bool fused_unsigned_sweep=std::string(argv[2])=="--fused-unsigned-sweep";
    const bool norm_threads_sweep=std::string(argv[2])=="--norm-threads-sweep";
    const bool norm_stats_sweep=std::string(argv[2])=="--norm-stats-sweep";
    const bool down_split_sweep=std::string(argv[2])=="--down-split-sweep";
    const bool up_fast_sigmoid_sweep=std::string(argv[2])=="--up-fast-sigmoid-sweep";
    const bool norm_fast_sweep=std::string(argv[2])=="--norm-fast-rsqrt-sweep";
    const bool up_int_mma_sweep=std::string(argv[2])=="--up-int16-mma-sweep";
    const bool down_fused_mma_sweep=std::string(argv[2])=="--down-mma-fused-up-sweep";
    const bool up_four_sweep=std::string(argv[2])=="--up-four-outputs-sweep";
    const bool up_half_fused_sweep=std::string(argv[2])=="--up-half-fused-prepare-sweep";
    const bool up_vector_sweep=std::string(argv[2])=="--up-vector-weights-sweep";
    const bool norm_register_sweep=std::string(argv[2])=="--norm-register-quant-sweep";
    const bool norm_inline_sweep=std::string(argv[2])=="--norm-inline-down-sweep";
    const bool down_s16_sweep=std::string(argv[2])=="--down-direct-s16-sweep";
    const bool norm_replicated_sweep=std::string(argv[2])=="--norm-replicated-sweep";
    const bool norm_vector_sweep=std::string(argv[2])=="--norm-vector-stats-sweep";
    const bool norm_fixed_sweep=std::string(argv[2])=="--norm-fixed-types-sweep" || norm_register_sweep || norm_inline_sweep || down_s16_sweep || norm_vector_sweep || norm_replicated_sweep;
    const bool norm_many_sweep=std::string(argv[2])=="--norm-many-slices-sweep";
    const bool norm_slices_sweep=std::string(argv[2])=="--norm-slices-sweep";
    const bool norm_quant_sweep=std::string(argv[2])=="--norm-quant-sweep";
    const bool down_mma_sweep=std::string(argv[2])=="--down-mma-sweep";
    const bool combined_sweep=std::string(argv[2])=="--combined-affine-sweep";
    const bool norm_wide_sweep=std::string(argv[2])=="--norm-wide-sweep";
    const bool norm_group_sweep=std::string(argv[2])=="--norm-group-quant-sweep";
    const bool unsigned_dot_sweep=std::string(argv[2])=="--unsigned-dot-sweep";
    const bool uniform_prepare_sweep=std::string(argv[2])=="--up-uniform-prepare-sweep";
    const bool packed24_sweep=std::string(argv[2])=="--packed24-sweep";
    const bool packed24_pair_sweep=std::string(argv[2])=="--packed24-pair-sweep";
    const bool chain_post_sweep=std::string(argv[2])=="--chain-post-sweep";
    const bool int_pair_sweep=std::string(argv[2])=="--int-pair-sweep";
    const bool thread_sweep=std::string(argv[2])=="--up-thread-sweep";
    const bool capacity_sweep=std::string(argv[2])=="--up-capacity-sweep";
    const bool sweep=norm_fixed_sweep || norm_many_sweep || down_fused_mma_sweep || up_four_sweep || up_half_fused_sweep || up_vector_sweep || norm_slices_sweep || up_int_mma_sweep || norm_fast_sweep || up_fast_sigmoid_sweep || down_split_sweep || norm_stats_sweep || norm_threads_sweep || up_direct_stats_sweep || fused_unsigned_sweep || up_fused_split_sweep || up_split_major_sweep || up_parts_vector_sweep || up_fused_direct_sweep || up_half_unroll_sweep || up_half_register_sweep || up_direct_s16_sweep || capacity_sweep || down_sweep || thread_sweep || group_sweep || metadata_sweep || unroll_sweep || norm_sweep || two_plane_sweep || three_plane_sweep || integer16_sweep || up_integer_sweep || integer_best_sweep || up_mix_sweep || up_prepare_sweep || up_groups_sweep || up_mma_sweep || up_mma_uniform_sweep || up_mma_half_sweep || up_mma_m1_half_sweep || up_coefficient_sweep || down_mma_sweep || combined_sweep || norm_quant_sweep || norm_wide_sweep || norm_group_sweep || unsigned_dot_sweep || uniform_prepare_sweep || packed24_sweep || packed24_pair_sweep || int_pair_sweep || chain_post_sweep;
    const int variants=norm_replicated_sweep || down_s16_sweep || norm_inline_sweep || norm_register_sweep || norm_many_sweep || norm_slices_sweep || up_int_mma_sweep || down_split_sweep || norm_stats_sweep || norm_threads_sweep || up_direct_stats_sweep || up_fused_direct_sweep || up_half_unroll_sweep || up_direct_s16_sweep || integer16_sweep || integer_best_sweep || packed24_sweep || uniform_prepare_sweep || up_mma_m1_half_sweep || up_coefficient_sweep?3:2;
    const bool verify=std::string(argv[2])=="--check";
    const bool timing=std::getenv("MFQ_GR_PROBE_NO_TIMING")==nullptr;
    if(!verify && !sweep && !uniform_prepare_sweep && std::string(argv[2])!="--bench")throw std::runtime_error("unknown batch GR mode");
    // At 32 rows the old dispatcher selects prefill arithmetic. Correctness
    // compares its ordered FP32 alternative; timings retain old dispatch.
    if(verify) {
#ifdef _WIN32
        if(_putenv_s("MFQ_GR_PREFILL_MATMUL","0"))throw std::runtime_error("GR reference option");
#else
        if(setenv("MFQ_GR_PREFILL_MATMUL","0",1))throw std::runtime_error("GR reference option");
#endif
    }
    if(up_coefficient_sweep) {
#ifdef _WIN32
        if(_putenv_s("MFQ_GR_UP_GROUP_COEFFICIENTS","2"))throw std::runtime_error("GR coefficient prepare option");
#else
        if(setenv("MFQ_GR_UP_GROUP_COEFFICIENTS","2",1))throw std::runtime_error("GR coefficient prepare option");
#endif
    }
    auto context=mfq::cuda::default_context(0);
    auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    auto model=mfq::open_model_source(argv[1]);CudaExecutionContext execution;
    std::vector<std::unique_ptr<BatchGr>> banks;
    const auto* selected_role=std::getenv("MFQ_GR_PROBE_ROLE");
    for(int layer:{0,8,16,24,32,40})for(const char* role:{"attention","mlp"})
        if(!selected_role || !std::strcmp(selected_role,role))
        banks.push_back(std::make_unique<BatchGr>(execution,*model,layer,role));
    context->begin_graph_pool(stream.stream());
    int cases=0,checks=0;
    const auto batches=argc==5?std::vector<int>{std::stoi(argv[4])}:
        verify?std::vector<int>{9,16,17,32}:norm_replicated_sweep?std::vector<int>{1,2,3,4,5,6}:std::vector<int>{16,32};
    for(int batch:batches)
    for(bool half:{false,true})for(bool after:{false,true}) {
        if(norm_replicated_sweep && !half && argc!=5)continue;
        if(chain_post_sweep && !after)continue;
        auto residual=tb::tensor(fb::input(batch,10240)).reshape({batch,1,10240})
            .to(tb::kCUDA,half?tb::kFloat16:tb::kFloat32);
        const bool chain_branch_float=(chain_post_sweep || norm_fixed_sweep) && std::getenv("MFQ_GR_CHAIN_BRANCH_FLOAT");
        auto branch=tb::tensor(fb::input(batch,2560)).reshape({batch,1,2560}).to(tb::kCUDA,
            chain_branch_float?tb::kFloat32:tb::kFloat16);
        auto prior=tb::tensor(fb::input(batch,4)).reshape({batch,1,4}).to(tb::kCUDA,
            chain_post_sweep?tb::kFloat32:residual.scalar_type());
        const auto source=residual.clone(),branch_source=branch.clone();
        std::array<std::vector<std::vector<tb::Tensor>>,3> outputs;
        std::array<MfqCudaGraph,3> graphs;context->begin_graph_warmup(stream.stream());
        const auto body=[&](int fused) {
            // Legacy normalization sweeps compare the original two-stage
            // implementations; the replicated candidate has its own sweep.
            if(norm_fixed_sweep && !norm_replicated_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_REPLICATED","0"))throw std::runtime_error("GR norm baseline option");
#else
                if(setenv("MFQ_GR_NORM_REPLICATED","0",1))throw std::runtime_error("GR norm baseline option");
#endif
            }
            if(down_fused_mma_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_DOWN_MMA_FUSED_UP",fused?"1":"0"))throw std::runtime_error("GR Down MMA fused Up option");
#else
                if(setenv("MFQ_GR_DOWN_MMA_FUSED_UP",fused?"1":"0",1))throw std::runtime_error("GR Down MMA fused Up option");
#endif
            }
            if(up_four_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_DIRECT_GROUPS",fused?"1":"0"))throw std::runtime_error("GR four output option");
#else
                if(setenv("MFQ_GR_UP_DIRECT_GROUPS",fused?"1":"0",1))throw std::runtime_error("GR four output option");
#endif
            }
            if(up_half_fused_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_MMA_F16_M1",fused?"3":"0"))throw std::runtime_error("GR fused half prepare option");
#else
                if(setenv("MFQ_GR_UP_MMA_F16_M1",fused?"3":"0",1))throw std::runtime_error("GR fused half prepare option");
#endif
            }
            if(up_vector_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_VECTOR_WEIGHTS",fused?"1":"0"))throw std::runtime_error("GR Up vector weight option");
#else
                if(setenv("MFQ_GR_UP_VECTOR_WEIGHTS",fused?"1":"0",1))throw std::runtime_error("GR Up vector weight option");
#endif
            }
            if(norm_replicated_sweep) {
                const char* mode=fused==2?"8":fused==1?"4":"0";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_REPLICATED",mode))throw std::runtime_error("GR replicated norm option");
#else
                if(setenv("MFQ_GR_NORM_REPLICATED",mode,1))throw std::runtime_error("GR replicated norm option");
#endif
            }
            if(norm_vector_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_VECTOR_STATS",fused?"1":"0"))throw std::runtime_error("GR vector stats option");
#else
                if(setenv("MFQ_GR_NORM_VECTOR_STATS",fused?"1":"0",1))throw std::runtime_error("GR vector stats option");
#endif
            }
            if(norm_fixed_sweep && !norm_register_sweep && !norm_inline_sweep && !down_s16_sweep && !norm_vector_sweep && !norm_replicated_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_FIXED_TYPES",fused?"1":"0"))throw std::runtime_error("GR fixed norm type option");
#else
                if(setenv("MFQ_GR_NORM_FIXED_TYPES",fused?"1":"0",1))throw std::runtime_error("GR fixed norm type option");
#endif
            }
            if(norm_register_sweep) {
                const char* mode=fused==2?"2":fused==1?"1":"0";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_REGISTER_QUANT",mode))throw std::runtime_error("GR register norm quant option");
#else
                if(setenv("MFQ_GR_NORM_REGISTER_QUANT",mode,1))throw std::runtime_error("GR register norm quant option");
#endif
            }
            if(norm_inline_sweep) {
                const char* mode=fused==2?"2":fused==1?"1":"0";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_INLINE_NORM_DOWN",mode))throw std::runtime_error("GR inline norm Down option");
#else
                if(setenv("MFQ_GR_INLINE_NORM_DOWN",mode,1))throw std::runtime_error("GR inline norm Down option");
#endif
            }
            if(down_s16_sweep) {
                const char* mode=fused==2?"2":fused==1?"1":"0";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_DOWN_DIRECT_S16",mode))throw std::runtime_error("GR direct S16 Down option");
#else
                if(setenv("MFQ_GR_DOWN_DIRECT_S16",mode,1))throw std::runtime_error("GR direct S16 Down option");
#endif
            }
            if(norm_many_sweep) {
                const char* group=fused?"2":"0";
                const char* slices=fused==2?"32":"16";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_GROUP_QUANT",group) || _putenv_s("MFQ_GR_NORM_GROUP_SLICES",slices) || _putenv_s("MFQ_GR_NORM_GROUP_THREADS","1024"))throw std::runtime_error("GR many norm slices option");
#else
                if(setenv("MFQ_GR_NORM_GROUP_QUANT",group,1) || setenv("MFQ_GR_NORM_GROUP_SLICES",slices,1) || setenv("MFQ_GR_NORM_GROUP_THREADS","1024",1))throw std::runtime_error("GR many norm slices option");
#endif
            }
            if(norm_slices_sweep) {
                const char* group=fused?"2":"0";
                const char* slices=fused==2?"8":"4";
                const char* threads=fused==2?"256":"512";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_GROUP_QUANT",group) || _putenv_s("MFQ_GR_NORM_GROUP_SLICES",slices) || _putenv_s("MFQ_GR_NORM_GROUP_THREADS",threads))throw std::runtime_error("GR norm slices option");
#else
                if(setenv("MFQ_GR_NORM_GROUP_QUANT",group,1) || setenv("MFQ_GR_NORM_GROUP_SLICES",slices,1) || setenv("MFQ_GR_NORM_GROUP_THREADS",threads,1))throw std::runtime_error("GR norm slices option");
#endif
            }
            if(up_int_mma_sweep) {
                const char* mode=fused==2?"3":fused==1?"2":"0";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_INT16_MMA",mode))throw std::runtime_error("GR Up int16 MMA option");
#else
                if(setenv("MFQ_GR_UP_INT16_MMA",mode,1))throw std::runtime_error("GR Up int16 MMA option");
#endif
            }
            if(norm_fast_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_FAST_RSQRT",fused?"1":"0"))throw std::runtime_error("GR fast norm option");
#else
                if(setenv("MFQ_GR_NORM_FAST_RSQRT",fused?"1":"0",1))throw std::runtime_error("GR fast norm option");
#endif
            }
            if(up_fast_sigmoid_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_FAST_SIGMOID",fused?"1":"0"))throw std::runtime_error("GR Up fast sigmoid option");
#else
                if(setenv("MFQ_GR_UP_FAST_SIGMOID",fused?"1":"0",1))throw std::runtime_error("GR Up fast sigmoid option");
#endif
            }
            if(down_split_sweep) {
                const char* splits=fused==2?"7":fused==1?"8":"13";
                const auto* probe_tile=std::getenv("MFQ_GR_DOWN_SPLIT_PROBE_TILE");
                const char* tile=fused && probe_tile?probe_tile:"16";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_DOWN_SPLITS",splits) || _putenv_s("MFQ_GR_UP_FUSED_DIRECT_TILE",tile))throw std::runtime_error("GR Down split option");
#else
                if(setenv("MFQ_GR_DOWN_SPLITS",splits,1) || setenv("MFQ_GR_UP_FUSED_DIRECT_TILE",tile,1))throw std::runtime_error("GR Down split option");
#endif
            }
            if(norm_stats_sweep) {
                const char* mode=fused==2?"3":fused==1?"2":"0";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_QUANT_FUSED",mode))throw std::runtime_error("GR norm stats option");
#else
                if(setenv("MFQ_GR_NORM_QUANT_FUSED",mode,1))throw std::runtime_error("GR norm stats option");
#endif
            }
            if(norm_threads_sweep) {
                const char* threads=fused==2?"512":fused==1?"640":"1024";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_THREADS",threads))throw std::runtime_error("GR norm threads option");
#else
                if(setenv("MFQ_GR_NORM_THREADS",threads,1))throw std::runtime_error("GR norm threads option");
#endif
            }
            if(up_direct_stats_sweep || fused_unsigned_sweep) {
                const char* stats=up_direct_stats_sweep && fused?"1":"0";
                const char* unsigned_dot=(up_direct_stats_sweep?fused==2:bool(fused))?"1":"0";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_UNIFORM_PREPARE","3") || _putenv_s("MFQ_GR_UP_DIRECT_S16","1") ||
                        _putenv_s("MFQ_GR_UP_SPLIT_MAJOR","1") || _putenv_s("MFQ_GR_UP_FUSED_DIRECT_TILE","16") ||
                        _putenv_s("MFQ_GR_UP_DIRECT_STATS",stats) || _putenv_s("MFQ_GR_DOWN_UNSIGNED_DOT",unsigned_dot))
                    throw std::runtime_error("GR direct stats/unsigned option");
#else
                if(setenv("MFQ_GR_UP_UNIFORM_PREPARE","3",1) || setenv("MFQ_GR_UP_DIRECT_S16","1",1) ||
                        setenv("MFQ_GR_UP_SPLIT_MAJOR","1",1) || setenv("MFQ_GR_UP_FUSED_DIRECT_TILE","16",1) ||
                        setenv("MFQ_GR_UP_DIRECT_STATS",stats,1) || setenv("MFQ_GR_DOWN_UNSIGNED_DOT",unsigned_dot,1))
                    throw std::runtime_error("GR direct stats/unsigned option");
#endif
            }
            if(up_fused_split_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_UNIFORM_PREPARE",fused?"3":"1") || _putenv_s("MFQ_GR_UP_DIRECT_S16","1") ||
                        _putenv_s("MFQ_GR_UP_PARTS_VECTOR","0") || _putenv_s("MFQ_GR_UP_SPLIT_MAJOR",fused?"1":"0") ||
                        _putenv_s("MFQ_GR_UP_FUSED_DIRECT_TILE","16"))
                    throw std::runtime_error("GR fused split major option");
#else
                if(setenv("MFQ_GR_UP_UNIFORM_PREPARE",fused?"3":"1",1) || setenv("MFQ_GR_UP_DIRECT_S16","1",1) ||
                        setenv("MFQ_GR_UP_PARTS_VECTOR","0",1) || setenv("MFQ_GR_UP_SPLIT_MAJOR",fused?"1":"0",1) ||
                        setenv("MFQ_GR_UP_FUSED_DIRECT_TILE","16",1))
                    throw std::runtime_error("GR fused split major option");
#endif
            }
            if(up_split_major_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_UNIFORM_PREPARE","1") || _putenv_s("MFQ_GR_UP_DIRECT_S16","1") ||
                        _putenv_s("MFQ_GR_UP_PARTS_VECTOR","0") || _putenv_s("MFQ_GR_UP_SPLIT_MAJOR",fused?"1":"0"))
                    throw std::runtime_error("GR split major option");
#else
                if(setenv("MFQ_GR_UP_UNIFORM_PREPARE","1",1) || setenv("MFQ_GR_UP_DIRECT_S16","1",1) ||
                        setenv("MFQ_GR_UP_PARTS_VECTOR","0",1) || setenv("MFQ_GR_UP_SPLIT_MAJOR",fused?"1":"0",1))
                    throw std::runtime_error("GR split major option");
#endif
            }
            if(up_parts_vector_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_UNIFORM_PREPARE","1") || _putenv_s("MFQ_GR_UP_DIRECT_S16","1") ||
                        _putenv_s("MFQ_GR_UP_MMA_F16_REGISTER_PREPARE","0") || _putenv_s("MFQ_GR_UP_PARTS_VECTOR",fused?"1":"0"))
                    throw std::runtime_error("GR vector split parts option");
#else
                if(setenv("MFQ_GR_UP_UNIFORM_PREPARE","1",1) || setenv("MFQ_GR_UP_DIRECT_S16","1",1) ||
                        setenv("MFQ_GR_UP_MMA_F16_REGISTER_PREPARE","0",1) || setenv("MFQ_GR_UP_PARTS_VECTOR",fused?"1":"0",1))
                    throw std::runtime_error("GR vector split parts option");
#endif
            }
            if(up_fused_direct_sweep) {
                const char* mode=fused?"3":"1";
                const char* tile=fused==2?"16":"24";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_MMA_F16_M1","0") || _putenv_s("MFQ_GR_UP_GROUP_COEFFICIENTS","0") ||
                        _putenv_s("MFQ_GR_UP_DIRECT_S16","1") || _putenv_s("MFQ_GR_UP_UNIFORM_PREPARE",mode) ||
                        _putenv_s("MFQ_GR_UP_FUSED_DIRECT_TILE",tile))
                    throw std::runtime_error("GR fused direct preparation option");
#else
                if(setenv("MFQ_GR_UP_MMA_F16_M1","0",1) || setenv("MFQ_GR_UP_GROUP_COEFFICIENTS","0",1) ||
                        setenv("MFQ_GR_UP_DIRECT_S16","1",1) || setenv("MFQ_GR_UP_UNIFORM_PREPARE",mode,1) ||
                        setenv("MFQ_GR_UP_FUSED_DIRECT_TILE",tile,1))
                    throw std::runtime_error("GR fused direct preparation option");
#endif
            }
            if(up_half_unroll_sweep) {
                const char* value=fused==0?"1":fused==1?"2":"7";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_MMA_F16_SPLIT","1") || _putenv_s("MFQ_GR_UP_MMA_F16_REGISTER_PREPARE","0") ||
                        _putenv_s("MFQ_GR_UP_MMA_F16_GROUP_UNROLL",value))
                    throw std::runtime_error("GR FP16 group unroll option");
#else
                if(setenv("MFQ_GR_UP_MMA_F16_SPLIT","1",1) || setenv("MFQ_GR_UP_MMA_F16_REGISTER_PREPARE","0",1) ||
                        setenv("MFQ_GR_UP_MMA_F16_GROUP_UNROLL",value,1))
                    throw std::runtime_error("GR FP16 group unroll option");
#endif
            }
            if(up_half_register_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_MMA_F16_SPLIT","1") || _putenv_s("MFQ_GR_UP_MMA_F16_REGISTER_PREPARE",fused?"1":"0"))
                    throw std::runtime_error("GR FP16 register preparation option");
#else
                if(setenv("MFQ_GR_UP_MMA_F16_SPLIT","1",1) || setenv("MFQ_GR_UP_MMA_F16_REGISTER_PREPARE",fused?"1":"0",1))
                    throw std::runtime_error("GR FP16 register preparation option");
#endif
            }
            if(up_direct_s16_sweep) {
                const char* value=fused==0?"0":fused==1?"1":"2";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_MMA_F16_M1","0") || _putenv_s("MFQ_GR_UP_UNIFORM_PREPARE","1") ||
                        _putenv_s("MFQ_GR_UP_GROUP_COEFFICIENTS","0") || _putenv_s("MFQ_GR_UP_DIRECT_S16",value))
                    throw std::runtime_error("GR direct int16 option");
#else
                if(setenv("MFQ_GR_UP_MMA_F16_M1","0",1) || setenv("MFQ_GR_UP_UNIFORM_PREPARE","1",1) ||
                        setenv("MFQ_GR_UP_GROUP_COEFFICIENTS","0",1) || setenv("MFQ_GR_UP_DIRECT_S16",value,1))
                    throw std::runtime_error("GR direct int16 option");
#endif
            }
            if(up_coefficient_sweep) {
                const char* value=fused==0?"0":fused==1?"1":"2";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_GROUP_COEFFICIENTS",value))throw std::runtime_error("GR coefficient option");
#else
                if(setenv("MFQ_GR_UP_GROUP_COEFFICIENTS",value,1))throw std::runtime_error("GR coefficient option");
#endif
            }
            if(up_mma_m1_half_sweep) {
                const char* value=fused==0?"0":fused==1?"1":"2";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_MMA_F16_M1",value))throw std::runtime_error("GR M1 FP16 MMA option");
#else
                if(setenv("MFQ_GR_UP_MMA_F16_M1",value,1))throw std::runtime_error("GR M1 FP16 MMA option");
#endif
            }
            if(up_mma_half_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_MMA_F16_SPLIT",fused?"1":"0"))throw std::runtime_error("GR Up FP16 MMA option");
#else
                if(setenv("MFQ_GR_UP_MMA_F16_SPLIT",fused?"1":"0",1))throw std::runtime_error("GR Up FP16 MMA option");
#endif
            }
            if(up_mma_uniform_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_MMA_F16_SPLIT","0"))throw std::runtime_error("GR disable FP16 MMA for Uniform16 probe");
                if(_putenv_s("MFQ_GR_UP_MMA_UNIFORM16",fused?"1":"0"))throw std::runtime_error("GR Up MMA Uniform16 option");
#else
                if(setenv("MFQ_GR_UP_MMA_F16_SPLIT","0",1))throw std::runtime_error("GR disable FP16 MMA for Uniform16 probe");
                if(setenv("MFQ_GR_UP_MMA_UNIFORM16",fused?"1":"0",1))throw std::runtime_error("GR Up MMA Uniform16 option");
#endif
            }
            if(uniform_prepare_sweep) {
                const char* value=fused==0?"0":fused==1?"1":"2";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_UNIFORM_PREPARE",value))throw std::runtime_error("GR Uniform16 preparation option");
#else
                if(setenv("MFQ_GR_UP_UNIFORM_PREPARE",value,1))throw std::runtime_error("GR Uniform16 preparation option");
#endif
            }
            if(unsigned_dot_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_DOWN_UNSIGNED_DOT",fused?"1":"0"))throw std::runtime_error("GR unsigned weight dot option");
#else
                if(setenv("MFQ_GR_DOWN_UNSIGNED_DOT",fused?"1":"0",1))throw std::runtime_error("GR unsigned weight dot option");
#endif
            }
            if(norm_group_sweep) {
                const auto* probe_mode=std::getenv("MFQ_GR_NORM_GROUP_PROBE_MODE");
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_GROUP_QUANT",fused?(probe_mode?probe_mode:"1"):"0"))throw std::runtime_error("GR norm group option");
#else
                if(setenv("MFQ_GR_NORM_GROUP_QUANT",fused?(probe_mode?probe_mode:"1"):"0",1))throw std::runtime_error("GR norm group option");
#endif
            }
            if(norm_wide_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_WIDE",fused?"1":"0"))throw std::runtime_error("GR wide norm option");
#else
                if(setenv("MFQ_GR_NORM_WIDE",fused?"1":"0",1))throw std::runtime_error("GR wide norm option");
#endif
            }
            if(combined_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_COMBINED_AFFINE",fused?"1":"0"))throw std::runtime_error("GR combined affine option");
#else
                if(setenv("MFQ_GR_COMBINED_AFFINE",fused?"1":"0",1))throw std::runtime_error("GR combined affine option");
#endif
            }
            if(down_mma_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_DOWN_MMA",fused?"1":"0"))throw std::runtime_error("GR Down MMA option");
#else
                if(setenv("MFQ_GR_DOWN_MMA",fused?"1":"0",1))throw std::runtime_error("GR Down MMA option");
#endif
            }
            if(norm_quant_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_QUANT_FUSED",fused?"1":"0"))throw std::runtime_error("GR norm quant option");
#else
                if(setenv("MFQ_GR_NORM_QUANT_FUSED",fused?"1":"0",1))throw std::runtime_error("GR norm quant option");
#endif
            }
            if(up_mma_sweep) {
                const auto* mode=std::getenv("MFQ_GR_UP_MMA_PROBE_MODE");
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_MMA",fused?(mode?mode:"1"):"0"))throw std::runtime_error("GR Up MMA option");
#else
                if(setenv("MFQ_GR_UP_MMA",fused?(mode?mode:"1"):"0",1))throw std::runtime_error("GR Up MMA option");
#endif
            }
            if(packed24_pair_sweep) {
                const auto* mode=std::getenv("MFQ_GR_PACKED24_PAIR_PLANES");
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_PACKED24_INT_DOT",mode?mode:"1") ||
                   _putenv_s("MFQ_GR_PACKED24_PAIR_DOWN",fused?"1":"0"))throw std::runtime_error("GR packed24 pair option");
#else
                if(setenv("MFQ_GR_PACKED24_INT_DOT",mode?mode:"1",1) ||
                   setenv("MFQ_GR_PACKED24_PAIR_DOWN",fused?"1":"0",1))throw std::runtime_error("GR packed24 pair option");
#endif
            }
            if(packed24_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_PACKED24_INT_DOT",fused==2?"2":fused?"1":"0"))throw std::runtime_error("GR packed24 option");
#else
                if(setenv("MFQ_GR_PACKED24_INT_DOT",fused==2?"2":fused?"1":"0",1))throw std::runtime_error("GR packed24 option");
#endif
            }
            if(up_groups_sweep) {
                const auto* reference=std::getenv("MFQ_GR_UP_PROBE_REFERENCE");
                const auto* probe_threads=std::getenv("MFQ_GR_UP_PROBE_THREADS");
                const char* old_mode=reference?(!std::strcmp(reference,"default")?"":reference):batch==1?"5":"7";
#ifdef _WIN32
                const auto* mode=std::getenv("MFQ_GR_UP_PROBE_MODE");
                if(_putenv_s("MFQ_GR_UP_INT_DOT",fused?(mode?mode:"9"):old_mode))throw std::runtime_error("GR Up group option");
                if(probe_threads && _putenv_s("MFQ_GR_UP_GROUP_THREADS",fused?probe_threads:""))throw std::runtime_error("GR Up group threads option");
#else
                const auto* mode=std::getenv("MFQ_GR_UP_PROBE_MODE");
                const char* selected=fused?(mode?mode:"9"):old_mode;
                if(selected[0]?setenv("MFQ_GR_UP_INT_DOT",selected,1):unsetenv("MFQ_GR_UP_INT_DOT"))throw std::runtime_error("GR Up group option");
                if(probe_threads && (fused?setenv("MFQ_GR_UP_GROUP_THREADS",probe_threads,1):unsetenv("MFQ_GR_UP_GROUP_THREADS")))throw std::runtime_error("GR Up group threads option");
#endif
            }
            if(int_pair_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_INT_PAIR_DOWN",fused?"1":"0"))throw std::runtime_error("GR pair option");
#else
                if(setenv("MFQ_GR_INT_PAIR_DOWN",fused?"1":"0",1))throw std::runtime_error("GR pair option");
#endif
            }
            if(up_integer_sweep || integer_best_sweep || up_mix_sweep || up_prepare_sweep) {
                const auto* mode=std::getenv("MFQ_GR_UP_PROBE_MODE");
                const char* selected=up_prepare_sweep?(fused?(mode?mode:"7"):"5"):up_mix_sweep?(fused?"6":"5"):integer_best_sweep?(fused==2?"":"0"):fused?(mode?mode:"5"):"0";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_INT_DOT",selected))throw std::runtime_error("GR Up integer dot option");
#else
                if(selected[0]?setenv("MFQ_GR_UP_INT_DOT",selected,1):unsetenv("MFQ_GR_UP_INT_DOT"))throw std::runtime_error("GR Up integer dot option");
#endif
            }
            if(two_plane_sweep || three_plane_sweep || integer16_sweep || integer_best_sweep) {
                const char* planes=integer_best_sweep?(fused==2?"":fused==1?"3":"0"):integer16_sweep?(fused==2?"4":fused==1?"3":"0"):fused?(three_plane_sweep?"3":"2"):"0";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_INT_DOT",planes))throw std::runtime_error("GR integer dot option");
#else
                if(planes[0]?setenv("MFQ_GR_INT_DOT",planes,1):unsetenv("MFQ_GR_INT_DOT"))throw std::runtime_error("GR integer dot option");
#endif
            }
            if(norm_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_NORM_CACHE",fused?"1":"0"))throw std::runtime_error("GR norm option");
#else
                if(setenv("MFQ_GR_NORM_CACHE",fused?"1":"0",1))throw std::runtime_error("GR norm option");
#endif
            }
            if(metadata_sweep || unroll_sweep) {
                const char* mode=fused?(unroll_sweep?"2":"1"):"0";
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_METADATA",mode))throw std::runtime_error("GR metadata option");
#else
                if(setenv("MFQ_GR_UP_METADATA",mode,1))throw std::runtime_error("GR metadata option");
#endif
            }
            if(group_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_GROUP_READ",fused?"2":"0"))throw std::runtime_error("GR group read option");
#else
                if(setenv("MFQ_GR_GROUP_READ",fused?"2":"0",1))throw std::runtime_error("GR group read option");
#endif
            }
            if(capacity_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_BATCH_SHARED_UP",fused?"4":"2"))throw std::runtime_error("GR Up capacity option");
#else
                if(setenv("MFQ_GR_BATCH_SHARED_UP",fused?"4":"2",1))throw std::runtime_error("GR Up capacity option");
#endif
            }
            if(down_sweep) {
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_BATCH_SHARED_DOWN",fused?"1":"0"))throw std::runtime_error("GR Down capacity option");
#else
                if(setenv("MFQ_GR_BATCH_SHARED_DOWN",fused?"1":"0",1))throw std::runtime_error("GR Down capacity option");
#endif
            }
            if(thread_sweep) {
                const char* threads=batch==1?(fused?"256":"512"):(fused?"128":"256");
#ifdef _WIN32
                if(_putenv_s("MFQ_GR_UP_THREADS",threads))throw std::runtime_error("GR Up threads option");
#else
                if(setenv("MFQ_GR_UP_THREADS",threads,1))throw std::runtime_error("GR Up threads option");
#endif
            }
            outputs[fused].clear();
            for(const auto& bank:banks) {
                if(chain_post_sweep) {
                    auto input=fused?residual:mfq_qwen4_exp::gated_residual_post(branch,residual,prior,4);
                    outputs[fused].push_back(bank->run(execution,branch,input,prior,bool(fused),true));
                }else outputs[fused].push_back(bank->run(execution,branch,residual,prior,after,bool(fused)||sweep));
            }
        };
        for(int fused=0;fused<variants;++fused) {
            body(fused);fb::check(cudaStreamSynchronize(stream.stream()));
        }
        for(auto& output:outputs)output.clear();
        for(int fused=0;fused<variants;++fused) {
            graphs[fused].capture_begin_shared();
            body(fused);
            if((norm_fixed_sweep || norm_many_sweep || down_fused_mma_sweep || up_four_sweep || up_half_fused_sweep || up_vector_sweep || norm_slices_sweep || up_int_mma_sweep || norm_fast_sweep || up_fast_sigmoid_sweep || down_split_sweep || norm_stats_sweep || norm_threads_sweep || up_direct_stats_sweep || fused_unsigned_sweep || up_fused_split_sweep || up_split_major_sweep || up_parts_vector_sweep || up_fused_direct_sweep || up_half_unroll_sweep || up_half_register_sweep || up_direct_s16_sweep || metadata_sweep || unroll_sweep || norm_sweep || two_plane_sweep || three_plane_sweep || integer16_sweep || up_integer_sweep || integer_best_sweep || up_mix_sweep || up_prepare_sweep || up_mma_sweep || up_mma_uniform_sweep || up_mma_half_sweep || up_mma_m1_half_sweep || up_coefficient_sweep || down_mma_sweep || norm_quant_sweep || capacity_sweep || int_pair_sweep || uniform_prepare_sweep) && std::getenv("MFQ_GR_PROBE_RESOURCES")) {
                cudaStreamCaptureStatus status;unsigned long long capture_id;cudaGraph_t captured;
                fb::check(mfq_stream_capture_info(stream.stream(),&status,&capture_id,&captured));
                size_t count=0;fb::check(cudaGraphGetNodes(captured,nullptr,&count));
                std::vector<cudaGraphNode_t> nodes(count);fb::check(cudaGraphGetNodes(captured,nodes.data(),&count));
                std::unordered_set<void*> seen;
                for(auto node:nodes) {
                    cudaGraphNodeType kind;fb::check(cudaGraphNodeGetType(node,&kind));
                    if(kind!=cudaGraphNodeTypeKernel)continue;
                    cudaKernelNodeParams parameters{};fb::check(cudaGraphKernelNodeGetParams(node,&parameters));
                    if(!seen.insert(parameters.func).second)continue;
                    cudaFuncAttributes attributes{};fb::check(cudaFuncGetAttributes(&attributes,parameters.func));
                    std::cout<<"GR_RESOURCE batch="<<batch<<" candidate="<<fused<<" threads="<<parameters.blockDim.x
                        <<" registers="<<attributes.numRegs<<" static_shared="<<attributes.sharedSizeBytes
                        <<" dynamic_shared="<<parameters.sharedMemBytes<<" local_bytes="<<attributes.localSizeBytes<<'\n';
                }
            }
            graphs[fused].capture_end();
        }
        for(int step=0;step<(verify||sweep?3:1);++step) {
            residual.copy_((source*(step==0?0.:.25+.5*step)).to(residual.scalar_type()));
            branch.copy_((branch_source*(1.+.25*step)).to(branch.scalar_type()));
            for(int fused=0;fused<variants;++fused) {
                graphs[fused].replay();
                fb::check(cudaStreamSynchronize(stream.stream()));
            }
            fb::check(cudaStreamSynchronize(stream.stream()));
            if(verify||sweep)for(int candidate=1;candidate<variants;++candidate)
            for(size_t bank=0;bank<banks.size();++bank)for(size_t out=0;out<3;++out) {
                if(((norm_fixed_sweep && !norm_register_sweep && !norm_vector_sweep && !norm_replicated_sweep) || norm_many_sweep || down_fused_mma_sweep || up_four_sweep || up_half_fused_sweep || norm_fast_sweep || up_fast_sigmoid_sweep || down_split_sweep || two_plane_sweep || three_plane_sweep || integer16_sweep || up_integer_sweep || integer_best_sweep || up_groups_sweep || up_mma_sweep || up_mma_uniform_sweep || up_mma_half_sweep || up_mma_m1_half_sweep || down_mma_sweep || combined_sweep || packed24_sweep) && out==0)mixed_two_plane_check(outputs[candidate][bank][out],outputs[0][bank][out]);
                else if(norm_fast_sweep && out==2)mixed_two_plane_check(outputs[candidate][bank][out],outputs[0][bank][out]);
                else exact(outputs[candidate][bank][out],outputs[0][bank][out].cpu(),"continuous GR batch="+
                    std::to_string(batch)+" bank="+std::to_string(bank)+" output="+std::to_string(out));++checks;
            }
            ++cases;
        }
        if(!verify && timing) {
            residual.copy_(source);branch.copy_(branch_source);
            std::array<std::vector<double>,3> times;
            for(int repeat=0;repeat<3;++repeat)for(int offset=0;offset<variants;++offset) {
                const int candidate=(repeat+offset)%variants;
                times[candidate].push_back(fb::median(fb::measure([&]{graphs[candidate].replay();},
                    stream.stream(),banks.size())));
            }
            std::cout<<"actual_gr_batch role="<<(selected_role?selected_role:"both")<<" batch="<<batch<<" half="<<half<<" after="<<after
                <<(integer16_sweep || integer_best_sweep?" int24_us=":metadata_sweep || unroll_sweep?" register_metadata_us=":group_sweep?" scalar_group_us=":thread_sweep?" default_threads_us=":down_sweep?" down1_us=":capacity_sweep?" capacity2_us=":" old_us=")<<fb::median(times[integer16_sweep || integer_best_sweep?1:0])
                <<(integer_best_sweep?" best_integer_us=":integer16_sweep?" uniform16_us=":thread_sweep?" half_threads_us=":down_sweep?" down2_us=":capacity_sweep?" capacity4_us=":" fused_us=")<<fb::median(times[integer16_sweep || integer_best_sweep?2:1]);
            if(packed24_sweep)std::cout<<" packed_uniform16_us="<<fb::median(times[2]);
            if(uniform_prepare_sweep)std::cout<<" down_prepared_us="<<fb::median(times[2]);
            if(up_mma_m1_half_sweep)std::cout<<" local_half_us="<<fb::median(times[2]);
            if(up_coefficient_sweep)std::cout<<" packed_metadata_us="<<fb::median(times[2]);
            if(up_direct_s16_sweep)std::cout<<" preload4_us="<<fb::median(times[2]);
            if(up_half_unroll_sweep)std::cout<<" full_unroll_us="<<fb::median(times[2]);
            if(up_fused_direct_sweep)std::cout<<" tile16_us="<<fb::median(times[2]);
            if(up_direct_stats_sweep)std::cout<<" paired_unsigned_us="<<fb::median(times[2]);
            if(norm_threads_sweep)std::cout<<" threads512_us="<<fb::median(times[2]);
            if(norm_stats_sweep)std::cout<<" stats512_us="<<fb::median(times[2]);
            if(up_int_mma_sweep)std::cout<<" mma_shared_unroll_us="<<fb::median(times[2]);
            if(norm_replicated_sweep)std::cout<<" groups8_us="<<fb::median(times[2]);
            if(norm_many_sweep)std::cout<<" slices32_threads1024_us="<<fb::median(times[2]);
            if(norm_register_sweep)std::cout<<" register128_us="<<fb::median(times[2]);
            if(norm_inline_sweep)std::cout<<" tile16_us="<<fb::median(times[2]);
            if(down_s16_sweep)std::cout<<" half_groups_s16_us="<<fb::median(times[2]);
            if(norm_slices_sweep)std::cout<<" slices8_threads256_us="<<fb::median(times[2]);
            if(down_split_sweep)std::cout<<" splits7_us="<<fb::median(times[2]);
            std::cout<<"\n"<<std::flush;
        }
    }
    context->end_graph_pool(stream.stream());
    std::cout<<"actual_gr_batch cases="<<cases<<" output_checks="<<checks
        <<(!timing?" NUMERICAL_ONLY_PASS\n":norm_fast_sweep?" MIXED_OUTPUT_INJECTION_UPDATED_EXACT TIMING PASS\n":down_s16_sweep || down_fused_mma_sweep || up_half_fused_sweep || up_fast_sigmoid_sweep || down_split_sweep || two_plane_sweep || three_plane_sweep || integer16_sweep || up_integer_sweep || integer_best_sweep || up_groups_sweep || up_mma_sweep || up_mma_uniform_sweep || up_mma_half_sweep || up_mma_m1_half_sweep || down_mma_sweep || combined_sweep || packed24_sweep?" MIXED_TOLERANCE_UPDATED_INJECTION_EXACT TIMING PASS\n":verify?" BYTE_EXACT PASS\n":sweep?" BYTE_EXACT TIMING PASS\n":" TIMING PASS\n");
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
