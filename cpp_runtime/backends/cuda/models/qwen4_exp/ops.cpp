#include "../../runtime/execution_options.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq_cuda_shared_gate.h"
#include "mfq_cuda_sampling_ops.h"
#include "mfq/runtime.h"
#include "ops.h"
#include "model.h"
#include "batch_attention.h"
#include "runtime/batch_decode_graph.h"
#include "mtp.h"
#include "storage/weight_loader.h"
#include "storage/moe_ffn_pipeline.h"
#include "storage/moe_expert_cache.h"
#include "prefill_budget.h"
#include "runtime/nint_row_pipeline.h"
#include "runtime/decode_window.h"
#include "runtime/moe_pipeline.h"
#include "runtime/router_lookahead_audit.h"
#include <fstream>
#include <iostream>
#include <atomic>
#include "models/common/gated_mlp.h"
#include "models/common/moe.h"
#include "models/common/transformer_layer.h"
#include "storage/session_codec.h"
#include "gated_residual.h"
#include "gated_residual_fused.h"
#include "float_projection.h"
#include "nint_prefill_cache.h"
#ifdef MFQ_NATIVE_CUDA_RUNTIME
#include "mfq_native_matmul_rows.h"
#endif
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>

namespace mfq::cuda {
static bool streamed_decode_graphs(const CudaExecutionContext& execution) {
    if(!execution.qsa_kv_store)return true;
    const auto* option=std::getenv("MFQ_KV_STREAM_GRAPHS");
    return execution.kv_memory && execution.kv_memory->offload() && (!option || option[0]!='0');
}
static mfq_tensor_backend::Tensor join_request_outputs(
        const std::vector<mfq_tensor_backend::Tensor>& outputs,int64_t axis) {
    const auto option=mfq::cuda::runtime_options::single_output_view();
    if(outputs.size()==1 && outputs.front().is_contiguous() && (!option || *option))
        return outputs.front();
    return mfq_tensor_backend::cat(outputs,axis);
}
static mfq_tensor_backend::Tensor append_decode_positions(
        const mfq_tensor_backend::Tensor& previous,const mfq_tensor_backend::Tensor& current,
        mfq_tensor_backend::Tensor& storage,int64_t maximum,
        const char* option_name="MFQ_QWEN4_POSITION_BUFFER") {
    namespace tb=mfq_tensor_backend;
    if(const auto option=mfq::cuda::runtime_options::position_buffer(option_name))if(!*option)
        return previous.defined()?tb::cat({previous,current},-1):current.clone();
    MFQ_RUNTIME_CHECK(current.dim()>=1 && (!previous.defined() || (previous.dim()==current.dim() &&
        previous.scalar_type()==current.scalar_type() && previous.device()==current.device())),
        "invalid decode position history");
    const int64_t count=previous.defined()?previous.size(-1):0,required=count+current.size(-1);
    MFQ_RUNTIME_CHECK(required<=maximum,"decode position history exceeds context");
    for(int axis=0;previous.defined() && axis+1<current.dim();++axis)
        MFQ_RUNTIME_CHECK(previous.size(axis)==current.size(axis),"decode position history axes changed");
    // Rollback retains a prefix view; restored histories receive a new arena.
    // The same last-axis append serves 1D target and 3D MTP rotary positions.
    const bool owns_prefix=storage.defined() && previous.defined() &&
        previous.data_ptr()==storage.data_ptr();
    if(!owns_prefix || storage.size(-1)<required) {
        int64_t capacity=owns_prefix?storage.size(-1):std::min<int64_t>(256,maximum);
        while(capacity<required)capacity=capacity>maximum/2?maximum:capacity*2;
        auto shape=current.sizes().vec();shape.back()=capacity;
        auto expanded=tb::empty(shape,current.options());
        if(count)expanded.narrow(-1,0,count).copy_(previous);
        storage=std::move(expanded);
    }
    storage.narrow(-1,count,current.size(-1)).copy_(current);
    return storage.narrow(-1,0,required);
}
}

namespace mfq::cuda::qwen4_exp {
using Config = mfq::models::qwen4_exp::Config;

struct Gr {
    Tensor norm;
    weight_loader::ResidualLinear down, up, injection;
    int64_t hidden, streams;
    double eps;
    int64_t prefill_work_chunk(const Tensor& input)const {
        if(input.dim()!=3 || input.size(0)!=1)return 0;
        const auto option=mfq::cuda::runtime_options::gr_prefill_work_chunk();
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const int64_t chunk=option?*option:8192;
        // Chat framing just beyond 8k needs no large temporary-memory cut.
        // Avoid extra concatenation there; longer inputs use 8k work tiles.
        if(!option && input.size(1)<=9216)return 0;
#else
        const int64_t chunk=option?*option:0;
#endif
        return chunk>=256 && input.size(1)>chunk?chunk:0;
    }
    template<class Project>
    std::vector<Tensor> bounded_pre(const Tensor& input,int64_t chunk,bool reuse_input,Project project)const {
        const auto tokens=input.size(1);
        std::vector<Tensor> output(3);
        if(reuse_input)output[1]=input;
        for(int64_t begin=0;begin<tokens;) {
            auto count=std::min(chunk,tokens-begin);
            const auto tail=tokens-begin-count;
            // Both pieces must retain the main GR prefill kernels; a tiny
            // remainder selects different projection arithmetic.
            if(tail>0 && tail<128)count-=128-tail;
            auto part=project(begin,count);
            for(size_t index=0;index<part.size();++index) {
                if(!part[index].defined() || (reuse_input && index==1))continue;
                if(!output[index].defined()) {
                    auto shape=part[index].sizes().vec();shape[1]=tokens;
                    output[index]=tb::empty(shape,part[index].options());
                }
                output[index].narrow(1,begin,count).copy_(part[index]);
            }
            begin+=count;
        }
        return output;
    }
    bool can_fuse(CudaExecutionContext& execution,const Tensor& x)const {
        if(!execution.config.gr_two_stage || !execution.config.gr_fused_projections ||
            !execution.config.gr_native_projection_input || !down.weight || !up.weight || !injection.weight ||
            !down.weight->is_nint() || !up.weight->is_nint() || streams<1 || streams>4 ||
            x.dim()<1 || x.size(-1)!=hidden*streams || x.numel()<hidden*streams ||
            x.numel()/(hidden*streams)>32 || !x.is_cuda() || !x.is_contiguous() ||
            (x.scalar_type()!=tb::kFloat16 && x.scalar_type()!=tb::kFloat32) ||
            !norm.is_contiguous() || norm.scalar_type()!=tb::kFloat32)return false;
        if(!injection.weight->is_nint() && !(execution.config.gr_two_stage_dense_injection &&
            injection.weight->is_dense() && injection.weight->dense.is_contiguous() &&
            injection.weight->dense.scalar_type()==tb::kBFloat16))return false;
        const auto& w=up.weight->nint;
        const auto rows=x.numel()/(hidden*streams);
        const auto continuous=mfq::cuda::runtime_options::gr_continuous_two_stage();
        if(rows>8 && (!execution.continuous_batch_cache_serial || (continuous && !*continuous)))
            return false;
        const auto batch=mfq::cuda::runtime_options::gr_batch_two_stage();
        if(rows>1 && batch && !*batch)return false;
        const bool selected=w.ng==1 || ((w.out+3)/4)*rows>=nint_float_projection_resident_blocks(x.get_device());
        const auto trace=mfq::cuda::runtime_options::trace_gr_dispatch();
        if(selected && trace && *trace) {
            static thread_local uint64_t seen[2]{};
            const int kind=x.scalar_type()==tb::kFloat16?0:1;
            if(!(seen[kind]&(uint64_t(1)<<rows))) {
                seen[kind]|=uint64_t(1)<<rows;
                std::fprintf(stderr,"gr_two_stage rows=%lld half=%d continuous=%d\n",
                    static_cast<long long>(rows),kind==0,int(execution.continuous_batch_cache_serial));
            }
        }
        return selected;
    }
    std::vector<Tensor> fused(const Tensor& branch,const Tensor& residual,const Tensor& prior,bool after,
        const GatedResidualMarker& marker={})const {
        const auto& projection=*injection.weight;
        return gated_residual_two_stage_cuda(branch,residual,prior,norm,down.weight->nint,up.weight->nint,
            projection.is_nint()?&projection.nint:nullptr,streams,eps,after,
            projection.is_dense()?projection.dense:Tensor{},injection.prepared_right,marker,up.up_group_coefficients);
    }
    std::vector<Tensor> pre(CudaExecutionContext &execution, const Tensor &x,
        const GatedResidualMarker& marker={}) const {
        if(const auto chunk=prefill_work_chunk(x))
            return bounded_pre(x,chunk,true,[&](int64_t begin,int64_t count) {
                return pre(execution,x.narrow(1,begin,count),marker);
            });
        if(can_fuse(execution,x))return fused({},x,{},false,marker);
        return gated_residual_pre_projected(execution, x, norm, down.forward, up.forward, injection.forward,
            hidden, streams, eps,up.mixed,marker,down.activated,injection.activated);
    }
    Tensor post(const Tensor &branch, const std::vector<Tensor> &inputs) const {
        return mfq_qwen4_exp::gated_residual_post(branch, inputs[1], inputs[2], streams);
    }
    std::vector<Tensor> pre_after(CudaExecutionContext& execution,const Tensor& branch,
        const std::vector<Tensor>& inputs,const GatedResidualMarker& marker={}) const {
        if(const auto chunk=prefill_work_chunk(inputs[1]))
            return bounded_pre(inputs[1],chunk,false,[&](int64_t begin,int64_t count) {
                const std::vector<Tensor> part{{},inputs[1].narrow(1,begin,count),inputs[2].narrow(1,begin,count)};
                return pre_after(execution,branch.narrow(1,begin,count),part,marker);
            });
        if(can_fuse(execution,inputs[1]) && branch.is_contiguous() && inputs[2].is_contiguous() &&
            (branch.scalar_type()==tb::kFloat16 || branch.scalar_type()==tb::kFloat32) &&
            (inputs[2].scalar_type()==tb::kFloat16 || inputs[2].scalar_type()==tb::kFloat32))
            return fused(branch,inputs[1],inputs[2],true,marker);
        return gated_residual_pre_after_projected(execution,branch,inputs[1],inputs[2],norm,
            down.forward,up.forward,injection.forward,hidden,streams,eps,
            up.mixed,marker,down.activated,injection.activated);
    }
};

// Preserve router and shared-expert arithmetic while regrouping routed experts.
static thread_local int64_t qwen_prefill_reference_rows=0;
struct QwenPrefillReferenceRowsScope {
    int64_t previous;
    explicit QwenPrefillReferenceRowsScope(int64_t rows):previous(qwen_prefill_reference_rows) {
        qwen_prefill_reference_rows=rows;
    }
    ~QwenPrefillReferenceRowsScope(){qwen_prefill_reference_rows=previous;}
};
template<class Projection>
static Tensor qwen_prefill_preserve_rows(const Tensor& source,Projection project) {
    const auto rows=qwen_prefill_reference_rows;
    if(rows<=0 || source.size(0)<=rows)return project(source);
    Tensor output;
    for(int64_t begin=0;begin<source.size(0);begin+=rows) {
        const auto count=std::min<int64_t>(rows,source.size(0)-begin);
        auto part=project(source.narrow(0,begin,count));
        if(!output.defined()) {
            auto shape=part.sizes().vec();shape[0]=source.size(0);
            output=tb::empty(shape,part.options());
        }
        output.narrow(0,begin,count).copy_(part);
    }
    return output;
}

// This scope is active only while an entire GDN layer is evaluated. Retain
// each projection's original token count and therefore its GEMM arithmetic.
static thread_local int64_t qwen_gdn_prefill_projection_rows=0;
struct QwenGdnPrefillProjectionRowsScope {
    int64_t previous;
    explicit QwenGdnPrefillProjectionRowsScope(int64_t rows):previous(qwen_gdn_prefill_projection_rows) {
        qwen_gdn_prefill_projection_rows=rows;
    }
    ~QwenGdnPrefillProjectionRowsScope(){qwen_gdn_prefill_projection_rows=previous;}
};
static Linear qwen_gdn_preserve_projection_rows(Linear project) {
    return [project](CudaExecutionContext& execution,const Tensor& source) {
        const auto rows=qwen_gdn_prefill_projection_rows;
        if(rows<=0 || source.dim()!=3 || source.size(0)!=1 || source.size(1)<=rows)
            return project(execution,source);
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const auto parallel_option=mfq::cuda::runtime_options::gdn_prefill_projection_streams();
        const int streams=parallel_option?*parallel_option:2;
        const auto tail=source.size(1)%rows;
        if(streams>0 && (tail==0 || tail>8)) {
            const mfq::cuda::MatmulRowsScope gemm_rows(rows,streams);
            return project(execution,source);
        }
#endif
        Tensor output;
        for(int64_t begin=0;begin<source.size(1);begin+=rows) {
            const auto count=std::min<int64_t>(rows,source.size(1)-begin);
            auto part=project(execution,source.narrow(1,begin,count));
            if(!output.defined()) {
                auto shape=part.sizes().vec();shape[1]=source.size(1);
                output=tb::empty(shape,part.options());
            }
            output.narrow(1,begin,count).copy_(part);
        }
        return output;
    };
}
static LinearGroup qwen_gdn_preserve_projection_rows(LinearGroup project) {
    if(!project)return {};
    return [project](CudaExecutionContext& execution,const Tensor& source) {
        const auto rows=qwen_gdn_prefill_projection_rows;
        if(rows<=0 || source.dim()!=3 || source.size(0)!=1 || source.size(1)<=rows)
            return project(execution,source);
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const auto parallel_option=mfq::cuda::runtime_options::gdn_prefill_projection_streams();
        const int streams=parallel_option?*parallel_option:2;
        const auto tail=source.size(1)%rows;
        if(streams>0 && (tail==0 || tail>8)) {
            const mfq::cuda::MatmulRowsScope gemm_rows(rows,streams);
            return project(execution,source);
        }
#endif
        std::vector<Tensor> output;
        for(int64_t begin=0;begin<source.size(1);begin+=rows) {
            const auto count=std::min<int64_t>(rows,source.size(1)-begin);
            auto part=project(execution,source.narrow(1,begin,count));
            if(output.empty())for(const auto& value:part) {
                auto shape=value.sizes().vec();shape[1]=source.size(1);
                output.push_back(tb::empty(shape,value.options()));
            }
            MFQ_RUNTIME_CHECK(part.size()==output.size(),"GDN projection count changed across chunks");
            for(size_t i=0;i<part.size();++i)output[i].narrow(1,begin,count).copy_(part[i]);
        }
        return output;
    };
}

template<class Projection>
static Projection qwen_gdn_profile_projection(Projection project,const char* begin,const char* end) {
    if(!project)return {};
    return [project,begin,end](CudaExecutionContext& execution,const Tensor& input) {
        auto* window=mfq::cuda::DecodeWindow::recording();
        if(window)window->mark_current(begin);
        auto output=project(execution,input);
        if(window)window->mark_current(end);
        return output;
    };
}

static Linear qwen_moe(weight_loader::Routed gate_up, weight_loader::Routed down,
                       Linear router, Linear shared_gate, Linear sg, Linear su, Linear sd,
                       const Config &c, MoeFfnPrefetch* prefetch=nullptr) {
    std::shared_ptr<TqProjectionGroup> shared_tq;
    const auto shared_g=weight_loader::linear_weight(sg),shared_u=weight_loader::linear_weight(su);
    if(shared_g && shared_u && shared_g->is_tq() && shared_u->is_tq() &&
       shared_g->tq.image.is_cuda() && shared_u->tq.image.is_cuda())
        shared_tq=std::make_shared<TqProjectionGroup>(std::vector<TqWeight>{shared_g->tq,shared_u->tq});
    auto shared_raw=[sg,su,sd,shared_gate,shared_tq](CudaExecutionContext& execution,const Tensor& source) {
        if(shared_tq) {
            auto output=sd(execution,shared_tq->swiglu(source));
            return tb::sigmoid(shared_gate(execution,source))*output;
        }
        auto unfused=[](const auto&...){return std::optional<Tensor>{};};
        auto output=mfq::models::gated_mlp(source,false,0.0,unfused,unfused,
            [&](Tensor input){return std::array<Tensor,2>{sg(execution,input),su(execution,input)};},
            [](Tensor g,Tensor u,mfq::models::GatedActivation,double){return (g*tb::sigmoid(g))*u;},
            [&](Tensor hidden){return sd(execution,hidden);},unfused);
        return tb::sigmoid(shared_gate(execution,source))*output;
    };
    const std::array<Linear,3> shared_projections{sg,su,sd};
    const bool shared_row_parallel=std::all_of(shared_projections.begin(),shared_projections.end(),
        [](const auto& projection) {
            const auto weight=weight_loader::linear_weight(projection);
            return weight && weight->is_nint() && !weight->tensor_parallel() && !weight->nint.q8_zero;
        });
    auto shared=[shared_raw,shared_projections,shared_gate,shared_row_parallel](CudaExecutionContext& execution,const Tensor& source) {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const auto rows=qwen_prefill_reference_rows;
        const auto parallel_option=mfq::cuda::runtime_options::moe_prefill_shared_streams();
        const int streams=parallel_option?*parallel_option:0;
        const auto chain_option=mfq::cuda::runtime_options::moe_prefill_shared_chain_streams();
        const int chain_streams=chain_option?*chain_option:4;
        const auto tail=rows>0?source.size(0)%rows:0;
        if(chain_streams>0 && shared_row_parallel && rows>8 && source.size(0)>=8192 &&
                source.size(0)>rows && (tail==0 || tail>8)) {
            // The first original chunk prepares all three cached weight
            // matrices. Worker pipelines then read them after the ready fence.
            auto first=shared_raw(execution,source.narrow(0,0,rows));
            auto shape=first.sizes().vec();shape[0]=source.size(0);
            auto output=tb::empty(shape,first.options());
            output.narrow(0,0,rows).copy_(first);
            mfq::cuda::submit_parallel_rows(source.get_device(),rows,source.size(0),rows,chain_streams,
                [&](int64_t begin,int64_t count) {
                    auto part=shared_raw(execution,source.narrow(0,begin,count));
                    output.narrow(0,begin,count).copy_(part);
                });
            return output;
        }
        if(streams>0 && shared_row_parallel && rows>8 && source.size(0)>rows && (tail==0 || tail>8)) {
            // Each GEMM retains its original row count. Only its submission
            // stream changes; activations remain in their canonical FP16 form.
            const auto project=[&](int index,const Tensor& input) {
                const mfq::cuda::MatmulRowsScope gemm_rows(rows,streams);
                return shared_projections[index](execution,input);
            };
            auto gate=project(0,source),up=project(1,source);
            auto output=project(2,(gate*tb::sigmoid(gate))*up);
            auto scale=qwen_prefill_preserve_rows(source,[&](const Tensor& part) {
                return tb::sigmoid(shared_gate(execution,part));
            });
            return scale*output;
        }
#endif
        return qwen_prefill_preserve_rows(source,[&](const Tensor& part){return shared_raw(execution,part);});
    };
    MoeFfnSharedWeights shared_weights;
    shared_weights.projections={weight_loader::linear_weight(sg),weight_loader::linear_weight(su),weight_loader::linear_weight(sd)};
    const auto shared_gate_weight=weight_loader::linear_weight(shared_gate);
    shared_weights.scalar_gate=shared_gate_weight;
    shared_weights.gate=[shared_gate,shared_gate_weight](CudaExecutionContext& execution,const Tensor& input) {
        const auto option=mfq::cuda::runtime_options::shared_gate_fused();
        const auto* nint_option=std::getenv("MFQ_SHARED_GATE_NINT");
        if((!option || *option) && (!nint_option || nint_option[0]!='0') && shared_gate_weight &&
                shared_gate_weight->is_nint() && !shared_gate_weight->tensor_parallel())
            if(auto output=try_nint_shared_gate_sigmoid_cuda(input,shared_gate_weight->nint))return *output;
        if((!option || *option) && shared_gate_weight && shared_gate_weight->is_dense() &&
            !shared_gate_weight->tensor_parallel())
            if(auto output=try_shared_gate_sigmoid_cuda(input,shared_gate_weight->dense))return *output;
        return tb::sigmoid(shared_gate(execution,input));
    };
    std::shared_ptr<TqMoeBanks> tq;
    if(!gate_up.tq_projections.empty() || !down.tq_projections.empty()) {
        MFQ_RUNTIME_CHECK(gate_up.tq_projections.size()==2 && down.tq_projections.size()==1 &&
            gate_up.projections.empty() && down.projections.empty(),
            "TQ MoE requires separate Gate, Up and Down banks");
        tq=std::make_shared<TqMoeBanks>();
        tq->gate=gate_up.tq_projections[0];tq->up=gate_up.tq_projections[1];tq->down=down.tq_projections[0];
    }
    auto pipeline=tq?MoeFfnForward{}:make_moe_ffn_pipeline(gate_up.projections,down.projections,shared,{},std::move(shared_weights),prefetch);
    return [gate_up, down, router, shared, pipeline, tq, c](CudaExecutionContext &execution,
                                                               const Tensor &x) {
        auto source = x.reshape({-1, c.hidden}).to(tb::kFloat16);
        const auto layer_prefill = mfq::cuda::runtime_options::moe_prefill_layer();
        const bool combined_prefill = source.size(0)>8 &&
            execution.kl_mmq.mode==KlMmqMode::Default &&
            ((layer_prefill && *layer_prefill!=0 && moe_expert_cache_layer_prefill_ready(execution.moe_expert_cache)) ||
             moe_expert_cache_ssd_experts(execution.moe_expert_cache));
        const bool use_pipeline=pipeline && (gate_up.projections.size()!=1 ||
            combined_prefill ||
            ((source.size(0)<=8 || (source.size(0)<=32 && execution.continuous_batch_cache_serial)) &&
             execution.config.moe_two_stage_ffn &&
             execution.kl_mmq.mode==KlMmqMode::Default));
        const auto routing = c.routing();
        return mfq::models::mixture_of_experts(
            [&] { return qwen_prefill_preserve_rows(source,[&](const Tensor& part){return router(execution,part);}).contiguous(); },
            [&](Tensor logits) {
                auto selected = moe_topk_cuda(
                    logits, c.topk, routing.activation == mfq::models::RouterActivation::sigmoid,
                    routing.activation == mfq::models::RouterActivation::sqrt_softplus,
                    routing.normalize, routing.delayed_softmax, mfq_nullopt, 1e-20, routing.scale);
                mask_batch_routes(execution, selected);
                return selected;
            },
            [&](const auto &) {
                return use_pipeline ? Tensor{} : shared(execution,source);
            },
            [&](const auto &selected) {
                if(use_pipeline)return pipeline(execution,source,selected[0].to(tb::kInt32).contiguous(),selected[1]);
                if(tq)return execution.profiler.measure("tq.moe",[&] {
                    return tq->forward(source,selected[0],selected[1],0,&execution.profiler);
                });
                return mfq::models::routed_experts(
                    [&] { return gate_up(execution, source, selected[0]); },
                    [&](Tensor gu) {
                        auto gate = gu.narrow(-1, 0, c.moe_width),
                             up = gu.narrow(-1, c.moe_width, c.moe_width);
                        return (gate * tb::sigmoid(gate)) * up;
                    },
                    [&](Tensor hidden) { return down(execution, hidden, selected[0]); },
                    [&](Tensor pairs) {
                        auto reduced = tb::zeros({source.size(0), c.hidden},
                                                 source.options().dtype(tb::kFloat32));
                        for (int64_t r = 0; r < c.topk; ++r)
                            reduced = reduced + pairs.select(1, r).to(tb::kFloat32) *
                                                    selected[1].select(1, r).unsqueeze(-1);
                        return reduced.to(pairs.scalar_type());
                    });
            },
            [&](Tensor routed, Tensor shared_output, const auto &) {
                if(use_pipeline)return routed.reshape(x.sizes());
                return (routed + shared_output).reshape(x.sizes());
            });
    };
}

struct BlockLoader : weight_loader::Loader {
    using GdnWeights = qwen4_exp::GdnWeights;
    using QsaWeights = qwen4_exp::QsaWeights;
    using PleWeights = qwen4_exp::PleWeights;
    Linear* router_capture=nullptr;
    MoeFfnPrefetch* prefetch_capture=nullptr;
    bool* tq_graph_capture=nullptr;
    struct Embedding {
        attention_ops::Embedding function;
        std::shared_ptr<mfq::NintRows> range;
        std::shared_ptr<TqRowTable> tq;
        Tensor operator()(const Tensor& ids) const { return function(ids); }
    };

    auto residual_linear(const std::string &name) const {
        return weight_loader::residual_linear(execution, source, name);
    }
    static Gr residual(Tensor norm, weight_loader::ResidualLinear down,
        weight_loader::ResidualLinear up, weight_loader::ResidualLinear injection, const Config &c) {
        return {std::move(norm), std::move(down), std::move(up), std::move(injection),
                c.hidden, c.streams, c.eps};
    }
    static auto final_mixer(Gr weights) { return std::make_unique<Gr>(std::move(weights)); }
    auto gdn_weights(const std::string &p) const {
        const bool independent = has(p + ".query.weight") || has(p + ".key.weight") || has(p + ".value.weight");
        MFQ_RUNTIME_CHECK(!independent || !has(p + ".qkv.weight"), "ambiguous Qwen4 GDN input projections");
        GdnWeights w{independent ? Linear{} : linear(p + ".qkv.weight"), linear(p + ".gate.weight"),
            linear(p + ".alpha.weight"), linear(p + ".beta.weight"), linear(p + ".output.weight"),
            dense(p + ".conv.weight"), dense(p + ".dt_bias"), dense(p + ".a"), dense(p + ".norm.weight")};
        if (independent)
            for (const auto* name : {"query", "key", "value"})
                w.split_qkv.push_back(linear(p + "." + name + ".weight"));
        return w;
    }
    auto qsa_weights(const std::string &p) const {
        QsaWeights w{linear(p + ".query.weight"), linear(p + ".key.weight"), linear(p + ".value.weight"),
            linear(p + ".output.weight"), linear(p + ".indexer.query_key.weight"), dense(p + ".query_norm.weight"),
            dense(p + ".key_norm.weight"), dense(p + ".indexer.query_norm.weight"), dense(p + ".indexer.key_norm.weight")};
        if (has(p + ".gate.weight")) w.gate = linear(p + ".gate.weight");
        return w;
    }
    auto gdn(GdnWeights weights, const Config &c) const {
        std::vector<Linear> projections = weights.split_qkv.empty()
            ? std::vector<Linear>{weights.qkv} : weights.split_qkv;
        const bool independent = !weights.split_qkv.empty();
        projections.insert(projections.end(), {weights.gate,weights.alpha,weights.beta});
        const auto packed_qkv_option=mfq::cuda::runtime_options::gdn_packed_qkv();
        const bool packed_qkv=independent && (!packed_qkv_option || *packed_qkv_option);
        auto group = weight_loader::grouped_linear(execution,projections,packed_qkv?3:1);
        if (independent) {
            if(group && packed_qkv)weights.input_projection=std::move(group);
            else if (group)
                weights.input_projection = [group](CudaExecutionContext &e, const Tensor &x) {
                    auto p = group(e,x);
                    return std::vector<Tensor>{tb::cat({p[0],p[1],p[2]},-1),p[3],p[4],p[5]};
                };
            weights.qkv = [q=projections[0],k=projections[1],v=projections[2]](CudaExecutionContext &e, const Tensor &x) {
                return tb::cat({q(e,x),k(e,x),v(e,x)},-1);
            };
        } else {
            weights.input_projection = std::move(group);
            weights.qkv = std::move(projections[0]);
        }
        const auto start = independent ? 3 : 1;
        weights.gate=std::move(projections[start]);weights.alpha=std::move(projections[start+1]);
        weights.beta=std::move(projections[start+2]);
        weights.input_projection=qwen_gdn_preserve_projection_rows(std::move(weights.input_projection));
        weights.qkv=qwen_gdn_preserve_projection_rows(std::move(weights.qkv));
        weights.gate=qwen_gdn_preserve_projection_rows(std::move(weights.gate));
        weights.alpha=qwen_gdn_preserve_projection_rows(std::move(weights.alpha));
        weights.beta=qwen_gdn_preserve_projection_rows(std::move(weights.beta));
        weights.output=qwen_gdn_preserve_projection_rows(std::move(weights.output));
        weights.input_projection=qwen_gdn_profile_projection(std::move(weights.input_projection),
            "gdn_input_begin","gdn_input_end");
        weights.output=qwen_gdn_profile_projection(std::move(weights.output),
            "gdn_output_begin","gdn_output_end");
        return std::make_unique<Gdn>(std::move(weights), c.key_heads, c.value_heads, c.linear_width,
                                     c.kernel, c.eps, c.silu_gate,true,execution.config.gdn_transposed_state,
                                     execution.config.gdn_fused_output,true,execution.config.gdn_fused_preparation,
                                     execution.config.gdn_fused_core);
    }
    auto qsa(QsaWeights weights, const Config &c) const {
        std::vector<Linear> projections{weights.query,weights.key,weights.value,weights.index_query_key};
        if (weights.gate) projections.push_back(weights.gate);
        auto group = weight_loader::grouped_linear(execution,projections);
        if (weights.gate) {
            const auto heads = c.heads, width = c.width;
            const auto combine = [heads,width](const Tensor &q,const Tensor &g) {
                const auto b=q.size(0),t=q.size(1);
                return tb::cat({q.reshape({b,t,heads,width}),g.reshape({b,t,heads,width})},-1)
                    .reshape({b,t,2*heads*width});
            };
            if (group)
                weights.input_projection = [group,combine](CudaExecutionContext &e,const Tensor &x) {
                    auto p=group(e,x);p[0]=combine(p[0],p[4]);p.resize(4);return p;
                };
            projections[0] = [q=projections[0],g=projections[4],combine](CudaExecutionContext &e,const Tensor &x) {
                return combine(q(e,x),g(e,x));
            };
        } else weights.input_projection=std::move(group);
        weights.query=std::move(projections[0]);weights.key=std::move(projections[1]);
        weights.value=std::move(projections[2]);weights.index_query_key=std::move(projections[3]);
        QsaConfig geometry{c.heads, c.kv_heads, c.width, c.index_heads, c.index_width,
                           c.pool, c.budget, c.maximum, c.eps};
        auto rotary = std::make_shared<RotaryEmbedding>(c.rotary, c.maximum, c.rope_base, c.sections, c.interleaved,std::nullopt,execution.yarn);
        return std::make_unique<Qsa>(std::move(weights), geometry, std::move(rotary), true, true, execution.qsa_kv_store, execution.kv_memory);
    }
    auto moe(weight_loader::Routed gate_up, weight_loader::Routed down, Linear router,
                    Linear shared_gate, Linear sg, Linear su, Linear sd, const Config &c) const {
        if(router_capture)*router_capture=router;
        if(tq_graph_capture)*tq_graph_capture=gate_up.tq_projections.size()==2 && down.tq_projections.size()==1;
        return qwen_moe(std::move(gate_up), std::move(down), std::move(router), std::move(shared_gate),
                        std::move(sg), std::move(su), std::move(sd), c, prefetch_capture);
    }
    auto embedding(const std::string &name) const {
        if(is_tq_row_table(source,name)) {
            auto table=std::make_shared<TqRowTable>(source,name);
            Embedding lookup{[table](const Tensor& ids){return table->lookup(ids);},{},table};
            return std::pair{std::move(lookup),std::array<int64_t,2>{static_cast<int64_t>(table->rows()),table->width()}};
        }
        if (require_tensor(source, name).dtype == "NINT") {
            auto table = load_nint_row_table(source, name);
            Embedding lookup{[table](const Tensor &ids) {
                return nint_row_embedding_lookup(*table, ids);
            },table};
            return std::pair{std::move(lookup), std::array<int64_t, 2>{table->rows(), table->width()}};
        }
        auto weight = std::make_shared<QuantLinear>(load_quant_linear(execution, source, name));
        Embedding lookup{[weight](const Tensor &ids) { return quant_embedding_lookup(*weight, ids); },{}};
        return std::pair{std::move(lookup), std::array<int64_t, 2>{weight->out(), weight->neuron_len()}};
    }
    std::vector<int64_t> integers(const std::string &name) const {
        const auto &type = require_tensor(source, name).dtype;
        MFQ_RUNTIME_CHECK(type == "I64" || type == "I32",
                          "Qwen4 PLE metadata must be a dense integer array: ", name);
        auto host = load_dense_gpu(execution, source, name).to(tb::kInt64).contiguous().cpu();
        MFQ_RUNTIME_CHECK(host.dim() == 1, "Qwen4 PLE metadata must be a vector");
        return {host.data_ptr<int64_t>(), host.data_ptr<int64_t>() + host.numel()};
    }
    static auto ple(std::vector<Embedding> shards, int64_t rows, int64_t width, const Config &c,
                    std::vector<int64_t> multipliers, std::vector<int64_t> offsets,
                    std::vector<int64_t> sizes, PleWeights weights) {
        std::vector<attention_ops::Embedding> functions;std::vector<std::shared_ptr<mfq::NintRows>> ranges;
        std::vector<std::shared_ptr<TqRowTable>> tq;
        for(auto& shard:shards){functions.push_back(std::move(shard.function));ranges.push_back(std::move(shard.range));tq.push_back(std::move(shard.tq));}
        std::shared_ptr<NintRowPipeline> pipeline;
        if(std::all_of(ranges.begin(),ranges.end(),[](const auto& p){return bool(p);})) {
            pipeline=std::make_shared<NintRowPipeline>(std::move(ranges));
        }
        NgramEmbedding::RowLookup row_lookup;
        NgramEmbedding::StageFactory stage_factory;
        if(std::all_of(tq.begin(),tq.end(),[](const auto& p){return bool(p);})) {
            row_lookup=[tq,rows](const auto& ids,const auto& device) {
                return lookup_tq_shards(tq,rows,ids,device);
            };
            stage_factory=[tq=std::move(tq),rows](const auto& shape,const auto& device) {
                return make_tq_row_stage(tq,rows,shape,device);
            };
        }
        NgramEmbedding embedding(std::move(functions), rows, width, c.ngram, c.ngram_heads, c.eos,
                                 std::move(multipliers), std::move(offsets), std::move(sizes),std::move(pipeline),std::move(row_lookup),std::move(stage_factory));
        return std::make_unique<Ple>(std::move(embedding), std::move(weights), c.hidden, c.streams,
                                     c.ngram, c.eps);
    }
    std::unique_ptr<Qwen4Block> block(const Config &c, int layer, bool predictor);
};

struct RecurrentGraphCheckpoint {
    std::vector<Tensor> gdn_conv,gdn_state,ple_conv;
};

struct Qwen4Block final : Block {
    using Tensor = mfq_tensor_backend::Tensor;
    mfq::models::qwen4_exp::Config config;
    Gr attention_gr, ffn_gr;
    Linear ffn,router;
    Linear next_router;
    MoeFfnPrefetch prefetch,next_prefetch;
    bool tq_graph_ffn=false;
    std::shared_ptr<RouterLookaheadInputs> router_audit;
    std::unique_ptr<Gdn> gdn;
    std::unique_ptr<Qsa> qsa;
    std::unique_ptr<Ple> ple;
    Tensor batch_ffn(CudaExecutionContext& execution,const Tensor& input) {
        if(input.size(0)<=8)return ffn(execution,input);
        const auto option=mfq::cuda::runtime_options::continuous_ffn_tile();
        const int64_t rows=option?*option:8;
        MFQ_RUNTIME_CHECK(rows==8 || rows==16,"continuous FFN tile must be 8 or 16 rows");
        auto* window=DecodeWindow::recording();
        const int previous_profile_tile=window ? window->set_profile_tile(-1) : -1;
        struct TileScope {
            CudaExecutionContext& execution;
            int tile,tiles;
            DecodeWindow* window;
            int profile_tile;
            ~TileScope() {execution.continuous_batch_ffn_tile=tile;execution.continuous_batch_ffn_tiles=tiles;
                if(window)window->set_profile_tile(profile_tile);}
        } scope{execution,execution.continuous_batch_ffn_tile,execution.continuous_batch_ffn_tiles,window,previous_profile_tile};
        execution.continuous_batch_ffn_tiles=int((input.size(0)+rows-1)/rows);
        std::vector<Tensor> outputs;
        for(int64_t start=0;start<input.size(0);start+=rows) {
            BatchDecodeMaskScope row_mask(execution, execution.decode_row_mask,
                execution.decode_row_offset + start);
            execution.continuous_batch_ffn_tile=int(start/rows);
            if(window)window->set_profile_tile(int(start/rows));
            outputs.push_back(ffn(execution,input.narrow(0,start,std::min(rows,input.size(0)-start))));
        }
        return tb::cat(outputs,0);
    }
    Qwen4Block(CudaExecutionContext &execution, const mfq::ModelSource &file,
               const Config &c, int layer, bool predictor = false) {
        BlockLoader loader{{execution, file, "qwen4_exp"},&router,&prefetch,&tq_graph_ffn};
        mfq::models::qwen4_exp::load_block(*this, loader, c, layer, predictor);
    }
    bool retain_graph_storage=false,retained_prefix=false;
    void reset(int64_t) override {
        if (gdn)
            gdn->reset(retain_graph_storage);
        if (qsa)
            qsa->reset();
        if (ple)
            ple->reset(retain_graph_storage);
    }
    bool supports_speculation() const noexcept override { return true; }
    void prefetch_token_ids(const Tensor& ids) override { if(ple)ple->prefetch_tokens(ids); }
    void commit_speculative() override {
        if (gdn)
            gdn->commit();
        if (ple)
            ple->commit();
    }
    void rollback_speculative(int64_t keep) override {
        if (gdn)
            gdn->rollback(keep);
        if (qsa)
            qsa->truncate(keep);
        if (ple)
            ple->rollback(keep);
    }
    Tensor execute(CudaExecutionContext &execution, Tensor x, const Tensor &ids,
                   const Tensor &positions, const Tensor &full_positions, int64_t confirmed = 0,const Tensor& host_ids = {}) {
        retained_prefix=confirmed>0 && ids.size(1)<=8;
        return mfq::models::qwen4_exp::decoder_layer_chained(
            std::move(x), bool(ple), bool(gdn),
            [&](const Tensor &hidden) {
                return execution.profiler.measure("prefill.ple",[&] {
                    return ple->forward(execution, hidden, ids, true, confirmed,host_ids,
                        full_positions.size(-1)-ids.size(1));
                });
            },
            [](Tensor hidden, Tensor positional) { return hidden + positional; },
            [&](const Tensor &hidden) { return execution.profiler.measure("prefill.gr_attention",[&] {return attention_gr.pre(execution, hidden);}); },
            [&](Tensor branch) { return execution.profiler.measure("prefill.gdn",[&] {return gdn->forward(execution, branch, true, confirmed,
                full_positions.size(-1)-ids.size(1));}); },
            [&](Tensor branch) {
                return execution.profiler.measure("prefill.qsa",[&] {return qsa->forward(execution, branch, positions, full_positions, true);});
            },
            [&](Tensor branch, const auto &mix) { return execution.profiler.measure("prefill.gr_ffn_pre",[&] {return ffn_gr.pre_after(execution, branch, mix);}); },
            [&](Tensor branch) { return execution.profiler.measure("prefill.ffn",[&] {return ffn(execution, branch);}); },
            [&](Tensor branch, const auto &mix) { return execution.profiler.measure("prefill.gr_ffn_post",[&] {return ffn_gr.post(branch, mix);}); });
    }
    Tensor forward(CudaExecutionContext &, Tensor, Tensor, int64_t, const MfqOptional<Tensor> &,
                   const RopeCache &, const MfqOptional<Tensor> & = mfq_nullopt,
                   const MfqOptional<Tensor> & = mfq_nullopt) override {
        throw std::runtime_error(
            "Qwen4 block requires the unified model position/PLE input lifecycle");
    }
    Tensor forward_context(CudaExecutionContext &execution, Tensor x, const Context &context,
                           const RopeCache &) override {
        return execute(execution, std::move(x), context.token_ids, context.positions,
                       context.full_positions, context.confirmed_prefix,context.host_token_ids);
    }
    Tensor prefill_ple(CudaExecutionContext& execution,Tensor x,const Context& context) {
        if(ple)x=x+execution.profiler.measure("prefill.ple",[&] {
            return ple->forward(execution,x,context.token_ids,true,0,context.host_token_ids);
        });
        return x;
    }
    Tensor prefill_attention_core(CudaExecutionContext& execution,const Tensor& input,const Context& context) {
        return gdn?
            execution.profiler.measure("prefill.gdn",[&] {return gdn->forward(execution,input,true,0);}):
            execution.profiler.measure("prefill.qsa",[&] {return qsa->forward(execution,input,context.positions,context.full_positions,true);});
    }
    std::vector<Tensor> prefill_attention(CudaExecutionContext& execution,Tensor x,const Context& context) {
        x=prefill_ple(execution,std::move(x),context);
        auto mix=execution.profiler.measure("prefill.gr_attention",[&] {return attention_gr.pre(execution,x);});
        auto branch=prefill_attention_core(execution,mix[0],context);
        return execution.profiler.measure("prefill.gr_ffn_pre",[&] {return ffn_gr.pre_after(execution,branch,mix);});
    }
    std::vector<Tensor*> graph_warmup_state() override {
        std::vector<Tensor*> result;
        if(gdn)for(auto* value:gdn->graph_state())if(value->defined())result.push_back(value);
        if(ple)for(auto* value:ple->graph_state())if(value->defined())result.push_back(value);
        return result;
    }
    Tensor execute_graph(CudaExecutionContext& execution,Tensor x,const Tensor& ids,
                         const Tensor& positions,const Tensor& cache_positions,const Tensor& ple_rows,int layer,
                         int64_t confirmed=0,RecurrentGraphCheckpoint* checkpoint=nullptr,
                         const std::vector<Tensor>& prior_mix={},std::vector<Tensor>* deferred_mix=nullptr) {
        if(auto* window=DecodeWindow::recording())window->set_profile_layer_kind(layer,gdn?"gdn":"qsa");
        const auto mark=[&](const char* phase){if(auto* window=DecodeWindow::recording())window->mark(layer,phase);};
        const bool detailed=mfq::cuda::runtime_options::mhc_timings();
        const auto marker=[&](const char* role)->GatedResidualMarker {
            if(!detailed || !DecodeWindow::recording())return {};
            return [&,role](const char* phase) {
                const auto name=std::string(role)+phase;mark(name.c_str());
            };
        };
        return mfq::models::qwen4_exp::decoder_layer_chained(std::move(x),bool(ple),bool(gdn),
            [&](const Tensor& hidden){return confirmed
                ? ple->forward_graph(execution,hidden,ids,ple_rows,confirmed,checkpoint->ple_conv)
                : ple->forward_chunk(execution,hidden,ids,true,{},ple_rows);},
            [](Tensor hidden,Tensor positional){return hidden+positional;},
            [&](const Tensor& hidden){auto result=prior_mix.empty()?
                attention_gr.pre(execution,hidden,marker("attention_mhc_")):
                attention_gr.pre_after(execution,hidden,prior_mix,marker("attention_mhc_"));
                mark("attention_core_begin");return result;},
            [&](Tensor branch){auto result=confirmed
                ? gdn->forward_graph(execution,branch,confirmed,checkpoint->gdn_conv,checkpoint->gdn_state)
                : gdn->forward_chunk(execution,branch,true);mark("attention_core_end");return result;},
            [&](Tensor branch){auto result=qsa->forward_graph(execution,branch,positions,cache_positions);mark("attention_core_end");return result;},
            [&](Tensor branch,const auto& mix){auto result=ffn_gr.pre_after(execution,branch,mix,marker("ffn_mhc_"));mark("ffn_pre_end");return result;},
            [&](Tensor branch) {
                DecodeWindow::Task prediction{};
                if(next_prefetch && branch.numel()==config.hidden && DecodeWindow::recording()->prediction_enabled()) {
                    const auto routing=config.routing();
                    auto logits=next_router(execution,branch.reshape({1,config.hidden}).to(tb::kFloat16)).to(tb::kFloat32).contiguous();
                    const int prediction_count=std::min<int64_t>(logits.size(-1),
                        moe_expert_cache_ssd_experts(execution.moe_expert_cache)?4:2);
                    auto selected=moe_topk_cuda(logits,prediction_count,
                        routing.activation==mfq::models::RouterActivation::sigmoid,
                        routing.activation==mfq::models::RouterActivation::sqrt_softplus,
                        routing.normalize,routing.delayed_softmax,mfq_nullopt,1e-20,routing.scale);
                    prediction=next_prefetch(selected[0].to(tb::kInt32).contiguous());
                }
                if(router_audit)router_audit->input(layer,branch);
                const bool early_prediction=prediction.identity && moe_expert_cache_ssd_experts(execution.moe_expert_cache);
                if(early_prediction)DecodeWindow::recording()->enroll(std::move(prediction));
                auto result=ffn(execution,branch);
                // Reserve current-layer readers before selecting speculative
                // slots. The route export itself precedes current FFN math.
                if(prediction.identity && !early_prediction)DecodeWindow::recording()->enroll(std::move(prediction));
                return result;
            },
            [&](Tensor branch,const auto& mix){
                if(deferred_mix){*deferred_mix=mix;return branch;}
                return ffn_gr.post(branch,mix);});
    }
};

std::unique_ptr<Qwen4Block> BlockLoader::block(const Config &c, int layer, bool predictor) {
    return std::make_unique<Qwen4Block>(execution, source, c, layer, predictor);
}

// One word changes the captured readout's work without retaining a second
// graph or its scratch. DecodeWindow waits for the previous replay before the
// host changes this flag. Greedy requests do not consume compact top-k.
struct MtpCompactControl {
    mfq::cuda::HostBuffer host{128};
    Tensor storage=tb::zeros({1},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
    const volatile int32_t* device=nullptr;
    int previous=-1;
    MtpCompactControl() {
        device=storage.data_ptr<int32_t>();
        mfq::cuda::charge_tensor_host_bytes(host.size());
    }
    ~MtpCompactControl(){mfq::cuda::tensor_host_bytes.fetch_sub(host.size());}
    void set(bool enabled,const char* role,int rows) {
        if(previous!=int(enabled)) {
            *static_cast<int32_t*>(host.data())=enabled?1:0;
            MFQ_CUDA_CHECK(cudaMemcpyAsync(storage.data_ptr(),host.data(),sizeof(int32_t),
                cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            std::cerr<<"runtime_mtp_compact_control role="<<role<<" rows="<<rows<<" enabled="<<int(enabled)<<'\n';
            previous=int(enabled);
        }
    }
};
struct Qwen4MtpGraph {
    struct Cell {
        bool used_in_request=false;
        int profile_replays=0;
        Tensor ids,hidden,positions,cache_positions,normalized,multi,logits,greedy_host,compact_host;
        MtpCompactControl compact_control;
        DecodeWindow window;
        Cell(const Tensor& source,const Tensor& token_ids,cudaStream_t stream)
            :ids(tb::empty_like(token_ids)),hidden(tb::empty_like(source)),
             positions(tb::empty({3,1,source.size(1)},token_ids.options().dtype(tb::kInt32))),
             cache_positions(tb::empty({source.size(1)},token_ids.options().dtype(tb::kInt64))),
             greedy_host(tb::empty({1},token_ids.options().device(tb::kCPU).dtype(tb::kInt64).pinned_memory(true))),
             compact_host(tb::empty({1,64},token_ids.options().device(tb::kCPU).dtype(tb::kInt64).pinned_memory(true))),
             window(stream,mfq::cuda::runtime_options::layer_timings()) {
            window.set_profile_geometry("mtp_predictor",static_cast<int>(source.size(1)));
        }
    };
    MfqCudaStream stream=mfq_get_current_cuda_stream();
    std::map<std::tuple<int,int,int>,std::unique_ptr<Cell>> cells;
    Tensor position_storage;
    int64_t capacity=0;
    bool failed=false;
    Tensor last_compact;
    bool reusable(const Qwen4ExpMtp& predictor) const {
        return predictor.reuse_decode_graphs && !failed && streamed_decode_graphs(*predictor.execution) &&
            !predictor.layers.empty() && predictor.layers[0]->qsa && predictor.layers[0]->qsa->graph_capacity()>0;
    }
    void restore_legacy(Qwen4ExpMtp& predictor) {
        last_compact={};cells.clear();predictor.layers[0]->qsa->leave_graph();
    }
    MtpStep run(Qwen4ExpMtp& predictor,const MtpTarget& target,const Tensor& hidden,const Tensor& ids,
               const Tensor& positions = {},bool history_only=false,bool scalar_history=false) {
        using ProfileTime=std::chrono::steady_clock;
        const auto profile_started=ProfileTime::now();
        const auto profile_transfer_before=moe_expert_streamed_bytes(predictor.execution->moe_expert_cache);
        MFQ_RUNTIME_CHECK(!failed && stream.stream()==mfq_current_cuda_stream(),"MTP predictor graph is unavailable");
        auto& execution=*predictor.execution;auto& block=*predictor.layers[0];
        last_compact={};
        // Predictor prefill may retain a multi-GiB expert image. Retire it
        // before graph replay so the first draft does not page graph buffers.
        release_moe_prefill_buffers(execution.moe_expert_cache);
        const auto profile_released=ProfileTime::now();
        const int64_t tokens=ids.size(1),start=predictor.lengths[0],required=start+tokens;
        MFQ_RUNTIME_CHECK(required<=predictor.config.maximum,"MTP predictor exceeds context capacity");
        if(block.qsa->graph_capacity()) {
            execution.kv_memory->ensure(required);
            capacity=block.qsa->graph_capacity();
        }
        if(required>capacity) {
            finish_moe_expert_exchanges(execution.moe_expert_cache);restore_legacy(predictor);
            capacity=std::min<int64_t>(predictor.config.maximum,std::max<int64_t>(2048,capacity));
            while(capacity<required)capacity=std::min(predictor.config.maximum,capacity*2);
        }
        const bool scalar_chain=history_only && scalar_history && tokens>1;
        const auto key=std::make_tuple(int(tokens),int(hidden.scalar_type()),history_only?(scalar_chain?2:1):0);
        auto& owner=cells[key];if(!owner)owner=std::make_unique<Cell>(hidden,ids,stream.stream());
        auto& cell=*owner;cell.hidden.copy_(hidden);cell.ids.copy_(ids);
        if(!history_only)cell.compact_control.set(predictor.compact_graph_sampling,"predictor",int(tokens));
        cell.window.set_profile_geometry(scalar_chain?"mtp_history_scalar_chain":history_only?"mtp_history":"mtp_predictor",int(tokens));
        predictor.graph_initialized=predictor.graph_initialized || !cell.used_in_request;
        mfq_qwen4_exp::fill_decode_positions(cell.cache_positions,cell.positions,start);
        if (positions.defined()) cell.positions.copy_(positions);
        try {
            const auto profile_exchange_begin=ProfileTime::now();
            // A warmed history graph touches attention KV and dense weights
            // only. Let the expert exchange proceed until the full draft.
            if(!cell.window.valid())
                finish_moe_expert_exchanges(execution.moe_expert_cache);
            else if(!history_only)prepare_moe_predictor_exchange(execution.moe_expert_cache);
            const auto profile_exchange_end=ProfileTime::now();
            block.qsa->prepare_graph(predictor.positions[0],capacity);
            if(!cell.window.valid()) {
                DecodeGraphBranchScope branch_scope(execution.decode_graph_serial_branches);
                predictor.graph_initialized=true;
                const auto capture_started=std::chrono::steady_clock::now();
                cell.window.capture([&] {
                    cell.window.mark(-1,"window_begin");
                    const auto make_input=[&](const Tensor& hidden_rows,const Tensor& token_rows,int64_t rows) {
                        auto embeddings=target.embed(token_rows);
                        auto e=predictor.embedding_fusion(execution,rms_norm(embeddings,predictor.embedding_norm+1,predictor.config.eps));
                        auto h=predictor.hidden_fusion(execution,rms_norm(hidden_rows,predictor.hidden_norm+1,predictor.config.eps)
                            .reshape({1,rows,predictor.config.streams,predictor.config.hidden}));
                        return (h+e.unsqueeze(-2)).reshape({1,rows,predictor.config.streams*predictor.config.hidden});
                    };
                    if(scalar_chain) {
                        // Keep scalar projection/reduction order while replaying
                        // all accepted history rows with one set of submissions.
                        for(int64_t row=0;row<tokens;++row) {
                            auto input=make_input(cell.hidden.narrow(1,row,1),cell.ids.narrow(1,row,1),1);
                            auto mix=block.attention_gr.pre(execution,input);
                            block.qsa->forward_graph(execution,mix[0],cell.positions.narrow(-1,row,1).contiguous(),
                                cell.cache_positions.narrow(0,row,1),{},true,true);
                        }
                    } else if(history_only) {
                        auto input=make_input(cell.hidden,cell.ids,tokens);
                        auto mix=block.attention_gr.pre(execution,input);
                        block.qsa->forward_graph(execution,mix[0],cell.positions,cell.cache_positions,{},true,true);
                    } else {
                        auto input=make_input(cell.hidden,cell.ids,tokens);
                        cell.multi=block.execute_graph(execution,std::move(input),cell.ids,cell.positions,cell.cache_positions,{},0);
                        cell.window.mark(-1,"head_begin");
                        cell.normalized=predictor.final_mixer->pre(execution,cell.multi)[0];
                        cell.logits=target.logits(cell.normalized.narrow(1,tokens-1,1).contiguous());
                        if(cell.logits.size(-1)>=64) {
                            auto compact=sample_top_k_compact_if_cuda(cell.logits.reshape({1,-1}).contiguous(),64,
                                cell.compact_control.device);
                            MFQ_CUDA_CHECK(cudaMemcpyAsync(cell.compact_host.data_ptr(),compact.data_ptr(),64*sizeof(int64_t),
                                cudaMemcpyDeviceToHost,stream.stream()));
                        }
                        auto greedy=sample_greedy_cuda(cell.logits.reshape({1,-1}).contiguous());
                        MFQ_CUDA_CHECK(cudaMemcpyAsync(cell.greedy_host.data_ptr(),greedy.data_ptr(),sizeof(int64_t),
                            cudaMemcpyDeviceToHost,stream.stream()));
                    }
                    cell.window.mark(-1,"window_end");
                },[]{},[&]{cell.multi={};cell.normalized={};cell.logits={};});
                ++predictor.graph_captures;
                std::cerr<<"runtime_qwen4_graph_capture role=predictor rows="<<tokens
                    <<" history_only="<<history_only<<" scalar_chain="<<scalar_chain<<" segments="<<cell.window.segments()
                    <<" count="<<predictor.graph_captures<<" ms="
                    <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-capture_started).count()<<'\n';
            }
            const auto profile_run_begin=ProfileTime::now();
            cell.window.run();
            const auto profile_run_end=ProfileTime::now();
            cell.used_in_request=true;++predictor.graph_replays;block.qsa->advance_graph(tokens);
            predictor.positions[0]=append_decode_positions(predictor.positions[0],cell.positions,
                position_storage,predictor.config.maximum,"MFQ_MTP_POSITION_BUFFER");
            predictor.lengths[0]=required;predictor.batch=1;
            const auto* profile_option=std::getenv("MFQ_PROFILE_MTP");
            if(profile_option && profile_option[0]=='1' && ++cell.profile_replays<=(history_only?2:16)) {
                const auto ms=[](auto begin,auto end){return std::chrono::duration<double,std::milli>(end-begin).count();};
                std::fprintf(stderr,"runtime_mtp_predictor_profile rows=%lld history=%d replay=%d release_ms=%.4f setup_ms=%.4f exchange_ms=%.4f prepare_ms=%.4f run_ms=%.4f tail_ms=%.4f transfer_mb=%.4f\n",
                    static_cast<long long>(tokens),int(history_only),cell.profile_replays,
                    ms(profile_started,profile_released),ms(profile_released,profile_exchange_begin),
                    ms(profile_exchange_begin,profile_exchange_end),ms(profile_exchange_end,profile_run_begin),
                    ms(profile_run_begin,profile_run_end),ms(profile_run_end,ProfileTime::now()),
                    (moe_expert_streamed_bytes(execution.moe_expert_cache)-profile_transfer_before)/1048576.0);
            }
            if(history_only)return {};
            if(predictor.compact_graph_sampling && cell.logits.size(-1)>=64)last_compact=cell.compact_host;
            return {cell.normalized.clone(),cell.multi.clone(),cell.logits.clone(),
                static_cast<int32_t>(*cell.greedy_host.data_ptr<int64_t>())};
        } catch(...) {failed=true;throw;}
    }
};
Tensor Qwen4ExpMtp::graph_compact_topk() const {
    return graph_registry?graph_registry->last_compact:Tensor{};
}

Qwen4ExpMtp::Qwen4ExpMtp() = default;
Qwen4ExpMtp::~Qwen4ExpMtp() = default;
Qwen4ExpMtp::Qwen4ExpMtp(Qwen4ExpMtp&&) noexcept = default;
Qwen4ExpMtp& Qwen4ExpMtp::operator=(Qwen4ExpMtp&&) noexcept = default;

std::optional<Qwen4ExpMtp> Qwen4ExpMtp::load_if_present(CudaExecutionContext &execution,
                                                      const mfq::ModelSource &file,
                                                      const mfq::models::qwen4_exp::Config &main) {
    Qwen4ExpMtp result;
    result.execution = &execution;
    BlockLoader loader{{execution, file, "qwen4_exp"}};
    if (!mfq::models::qwen4_exp::load_predictor(result, loader, main))
        return std::nullopt;
    return result;
}

void Qwen4ExpMtp::reset(int64_t next_batch) {
    MFQ_RUNTIME_CHECK(next_batch > 0, "Qwen4-Exp MTP batch must be positive");
    const bool retain=next_batch==1 && graph_registry && graph_registry->reusable(*this);
    if(!retain)graph_registry.reset();
    else for(auto& [key,cell]:graph_registry->cells) {cell->used_in_request=false;cell->profile_replays=0;}
    graph_initialized=false;
    for (auto &layer : layers) {
        layer->retain_graph_storage=retain;
        layer->reset(next_batch);
    }
    for (auto &pos : positions)
        pos = Tensor();
    std::fill(lengths.begin(), lengths.end(), 0);
    batch = next_batch;
}

std::pair<Tensor, Tensor> Qwen4ExpMtp::evaluate(const MtpTarget &target, const Tensor &hidden,
                                       const Tensor &ids, int64_t depth, bool cache,
                                       const Tensor &supplied_positions,
                                       const Tensor &supplied_embeddings, bool history_only) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(!history_only || (cache && layers.size()==1),
        "cache-only MTP priming requires a single predictor layer");
    if(graph_registry) {
        if(graph_registry->reusable(*this) && ids.size(0)==1 && cache)layers[0]->qsa->leave_graph();
        else {graph_registry->restore_legacy(*this);graph_registry.reset();}
    }
    MFQ_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) &&
                          hidden.size(2) == hidden_norm.numel(),
                      "Qwen4-Exp MTP input geometry mismatch");
    const auto b = ids.size(0), t = ids.size(1);
    return mfq::models::projected_predictor(
        *this, b, t, depth, cache, [&](int64_t next) { reset(next); },
        [&] {
            auto embeds =
                supplied_embeddings.defined() ? supplied_embeddings : target.embed(ids);
            MFQ_RUNTIME_CHECK(embeds.sizes().vec() ==
                                  std::vector<int64_t>({b, t, config.hidden}),
                              "Qwen4-Exp MTP embedding shape mismatch");
            return embeds;
        },
        [&](mfq::models::PredictorCursor cursor) {
            // Full positions persist across predictor priming tiles and must
            // not occupy expert storage borrowed by this prefill phase.
            mfq::cuda::ResidentAllocationScope persistent_positions;
            auto current = supplied_positions.defined()
                               ? supplied_positions
                               : tb::arange(cursor.start, cursor.start + t,
                                            ids.options().dtype(tb::kInt32));
            if (current.dim() == 1)
                current = current.reshape({1, 1, t}).expand({3, 1, t}).contiguous();
            else if (current.dim() == 2)
                current = current.unsqueeze(1);
            if (current.dim() == 3 && current.size(0) == 4)
                current = current.narrow(0, 1, 3);
            MFQ_RUNTIME_CHECK(current.dim() == 3 && current.size(0) == 3 &&
                                  current.size(-1) == t &&
                                  (current.size(1) == 1 || current.size(1) == b),
                              "Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
            current = current.to(tb::kInt32).expand({3, b, t}).contiguous();
            auto full = cache && positions[cursor.layer].defined()
                            ? tb::cat({positions[cursor.layer], current}, -1)
                            : current;
            return std::array<Tensor, 2>{current, full};
        },
        [&](Tensor embeds, const auto &pos) {
            return embedding_fusion(*execution,
                                    rms_norm(embeds, embedding_norm + 1, config.eps));
        },
        [&] {
            return hidden_fusion(*execution,
                                 rms_norm(hidden, hidden_norm + 1, config.eps)
                                     .reshape({b, t, config.streams, config.hidden}));
        },
        [&](Tensor e, Tensor streams) {
            return (streams + e.unsqueeze(-2)).reshape({b, t, config.streams * config.hidden});
        },
        [&](Tensor x, int64_t layer, const auto &pos) {
            auto &block = *layers[layer];
            // Teacher-forced rows each receive the target hidden state. Only
            // the attention input contributes persistent predictor history;
            // attention readout, FFN and final mixer have no consumer here.
            if(history_only) {
                auto mix=block.attention_gr.pre(*execution,x);
                block.qsa->forward(*execution,mix[0],pos[0],pos[1],true,nullptr,{},true,true);
                return Tensor{};
            }
            return mfq::models::qwen4_exp::decoder_layer(
                std::move(x), false, false, [](const Tensor &) { return Tensor{}; },
                [](Tensor value, Tensor) { return value; },
                [&](const Tensor &value) { return block.attention_gr.pre(*execution, value); },
                [](Tensor) -> Tensor { throw std::logic_error("Qwen4 MTP requires QSA"); },
                [&](Tensor branch) {
                    return block.qsa->forward(*execution, branch, pos[0], pos[1], cache);
                },
                [&](Tensor branch, const auto &mix) {
                    return block.attention_gr.post(branch, mix);
                },
                [&](const Tensor &value) { return block.ffn_gr.pre(*execution, value); },
                [&](Tensor branch) { return block.ffn(*execution, branch); },
                [&](Tensor branch, const auto &mix) { return block.ffn_gr.post(branch, mix); });
        },
        [&](const Tensor &multi) { return history_only?Tensor{}:final_mixer->pre(*execution, multi)[0]; },
        [&](int64_t layer, const auto &pos) { positions[layer] = pos[1]; });
}

Tensor Qwen4ExpMtp::forward(const MtpTarget &target, Tensor hidden, Tensor ids) {
    return evaluate(target, hidden, ids).first;
}

MtpStep Qwen4ExpMtp::step(const MtpTarget &target, Tensor hidden, Tensor ids) {
    return step_positioned(target,std::move(hidden),std::move(ids),{});
}

MtpStep Qwen4ExpMtp::step_positioned(const MtpTarget &target, Tensor hidden, Tensor ids, Tensor positions) {
    if (positions.defined()) {
        if (positions.dim()==1) positions=positions.reshape({1,1,-1}).expand({3,ids.size(0),ids.size(1)});
        else if (positions.dim()==2) positions=positions.unsqueeze(1);
        if (positions.dim()!=3 || positions.size(0)!=3 || positions.size(1)!=ids.size(0) || positions.size(2)!=ids.size(1))
            throw std::invalid_argument("Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
        positions=positions.to(tb::kInt32).contiguous();
    }
    const auto graph=mfq::cuda::runtime_options::mtp_predictor_graph();
    if((!graph || *graph) && execution->config.moe_pipeline && execution->config.moe_two_stage_ffn &&
        execution->kl_mmq.mode==KlMmqMode::Default && execution->moe_expert_cache &&
        streamed_decode_graphs(*execution) && batch==1 && ids.dim()==2 && ids.size(0)==1 && ids.size(1)>0 && ids.size(1)<=8 &&
        hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==ids.size(1) && hidden.size(2)==hidden_norm.numel() &&
        ids.is_cuda() && hidden.is_cuda() && !lengths.empty() && lengths[0]>0 &&
        layers[0]->qsa && layers[0]->prefetch && this->positions[0].dim()==3 && this->positions[0].size(1)==1) {
        if(!graph_registry)graph_registry=std::make_unique<Qwen4MtpGraph>();
        return graph_registry->run(*this,target,hidden,ids,positions);
    }
    auto result = evaluate(target, hidden, ids, 0, true, positions);
    return {std::move(result.first), std::move(result.second)};
}

void Qwen4ExpMtp::history_step(const MtpTarget& target,Tensor hidden,Tensor ids,Tensor positions,bool scalar_chain) {
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(positions.defined()) {
        if(positions.dim()==1)positions=positions.reshape({1,1,-1}).expand({3,ids.size(0),ids.size(1)});
        else if(positions.dim()==2)positions=positions.unsqueeze(1);
        MFQ_RUNTIME_CHECK(positions.dim()==3 && positions.size(0)==3 &&
            positions.size(1)==ids.size(0) && positions.size(2)==ids.size(1),"invalid MTP history positions");
        positions=positions.to(tb::kInt32).contiguous();
    }
    const auto graph=mfq::cuda::runtime_options::mtp_predictor_graph();
    if(reuse_decode_graphs && (!graph || *graph) && execution->config.moe_pipeline &&
            execution->config.moe_two_stage_ffn && execution->kl_mmq.mode==KlMmqMode::Default &&
            execution->moe_expert_cache && streamed_decode_graphs(*execution) && layers.size()==1 && batch==1 &&
            ids.dim()==2 && ids.size(0)==1 && ids.size(1)>0 && ids.size(1)<=8 &&
            hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==ids.size(1) &&
            hidden.size(2)==hidden_norm.numel() && ids.is_cuda() && hidden.is_cuda() &&
            !lengths.empty() && lengths[0]>0 && layers[0]->qsa && layers[0]->prefetch &&
            this->positions[0].dim()==3 && this->positions[0].size(1)==1) {
        if(!graph_registry)graph_registry=std::make_unique<Qwen4MtpGraph>();
        (void)graph_registry->run(*this,target,hidden,ids,positions,true,scalar_chain);
    } else if(scalar_chain && ids.size(1)>1) {
        for(int64_t row=0;row<ids.size(1);++row)
            (void)step_positioned(target,hidden.narrow(1,row,1),ids.narrow(1,row,1),
                positions.defined()?positions.narrow(-1,row,1):Tensor{});
    } else (void)step_positioned(target,std::move(hidden),std::move(ids),std::move(positions));
}

int64_t Qwen4ExpMtp::cache_position() const noexcept {
    return lengths.empty() ? 0 : lengths.front();
}

void Qwen4ExpMtp::trim_cache_to(int64_t position) {
    MFQ_RUNTIME_CHECK(position >= 0 && position <= cache_position(),
                      "Qwen4-Exp MTP cache trim position is invalid");
    constexpr size_t index = 0;
    if (index < layers.size() && layers[index]->qsa)
        layers[index]->qsa->truncate(position);
    if (positions[index].defined())
        positions[index] = positions[index].narrow(-1, 0, position);
    lengths[index] = position;
}

bool Qwen4ExpMtp::teacher_forced_prompt_prime() const noexcept { return true; }

bool Qwen4ExpMtp::target_bootstrap_decode() const noexcept { return true; }

bool Qwen4ExpMtp::preserve_output_dtype() const noexcept { return true; }

namespace {
struct IndependentPredictorState final : MtpBatchState {
    std::unique_ptr<Qsa> attention;
    Tensor positions;
    void trim(int64_t value) override {
        MFQ_RUNTIME_CHECK(value >= 0 && value <= position, "invalid independent predictor trim");
        attention->truncate(value);
        if (positions.defined()) positions = positions.narrow(-1, 0, value);
        position = value;
    }
};
}

std::unique_ptr<MtpBatchState> Qwen4ExpMtp::make_batch_state() const {
    MFQ_RUNTIME_CHECK(!layers.empty() && layers.front()->qsa, "MTP predictor requires QSA");
    auto state = std::make_unique<IndependentPredictorState>();
    state->attention = layers.front()->qsa->fork_empty();
    return state;
}

MtpStep Qwen4ExpMtp::batch_step(const MtpTarget& target, const std::vector<MtpBatchState*>& states,
        Tensor hidden, Tensor ids) {
    const int64_t b = states.size(), t = ids.size(1);
    MFQ_RUNTIME_CHECK(b > 0 && ids.size(0) == b && hidden.size(0) == b && hidden.size(1) == t,
                      "invalid independent predictor geometry");
    auto embeds = target.embed(ids);
    auto e = embedding_fusion(*execution, rms_norm(embeds, embedding_norm + 1, config.eps));
    auto streams = hidden_fusion(*execution, rms_norm(hidden, hidden_norm + 1, config.eps)
        .reshape({b, t, config.streams, config.hidden}));
    auto multi = (streams + e.unsqueeze(-2)).reshape({b, t, config.streams * config.hidden});
    auto& block = *layers.front();
    auto mix = block.attention_gr.pre(*execution, multi);
    auto projections = block.qsa->project_inputs(*execution, mix[0]);
    std::vector<Tensor> attended;
    for (int64_t row = 0; row < b; ++row) {
        auto& state = static_cast<IndependentPredictorState&>(*states[row]);
        MFQ_RUNTIME_CHECK(t <= config.maximum - state.position, "independent predictor exceeds context");
        auto current = (tb::arange(t, ids.options().dtype(tb::kInt32)) + state.position)
            .reshape({1, 1, t}).expand({3, 1, t}).contiguous();
        auto full = state.positions.defined() ? tb::cat({state.positions, current}, -1) : current;
        std::vector<Tensor> projected;
        for (const auto& value : projections) projected.push_back(value.narrow(0, row, 1));
        attended.push_back(state.attention->forward(*execution, mix[0].narrow(0, row, 1), current,
            full, true, nullptr, std::move(projected), false));
        state.positions = std::move(full);
        state.position += t;
    }
    auto attention = block.qsa->project_output(*execution, join_request_outputs(attended, 0));
    multi = block.attention_gr.post(attention, mix);
    auto ffn_mix = block.ffn_gr.pre(*execution, multi);
    multi = block.ffn_gr.post(block.ffn(*execution, ffn_mix[0]), ffn_mix);
    auto output = final_mixer->pre(*execution, multi)[0];
    return {std::move(output), std::move(multi)};
}

Qwen4BlockSessionState capture_block_session(const Qwen4Block& block,int64_t position,std::size_t& bytes) {
    Qwen4BlockSessionState state;
    const auto save=[&](const Tensor& tensor) {
        MFQ_RUNTIME_CHECK(tensor.defined(),"Qwen4 recurrent session state is unavailable");
        auto saved=tensor.clone();bytes+=session_tensor_bytes(saved);return saved;
    };
    if(block.qsa)block.qsa->capture_session(state,position,bytes);
    if(block.gdn) {
        const auto tensors=block.gdn->graph_state();
        state.convolution=save(*tensors[0]);state.recurrent=save(*tensors[1]);
    }
    if(block.ple) {
        state.ple_convolution=save(block.ple->conv_state());
        const auto& context=block.ple->session_context();
        state.ple_batch=context.batch;state.ple_tokens=context.tokens;
        bytes+=state.ple_tokens.size()*sizeof(int64_t);
    }
    return state;
}

void restore_block_session(Qwen4Block& block,const Qwen4BlockSessionState& state,int64_t position) {
    block.reset(1);
    if(block.qsa)block.qsa->restore_session(state,position);
    if(block.gdn)block.gdn->restore_session(state.convolution,state.recurrent,block.retain_graph_storage);
    if(block.ple)block.ple->restore_session(state.ple_convolution,{state.ple_batch,state.ple_tokens},block.retain_graph_storage);
}

bool Qwen4ExpMtp::supports_session_state() const noexcept {
    return execution && !execution->qsa_kv_store && !layers.empty();
}

MtpSessionState Qwen4ExpMtp::capture_session_state(int64_t target_position,const Tensor& hidden) const {
    const auto position=target_position-1;
    MFQ_RUNTIME_CHECK(supports_session_state() && position>0 && position<=cache_position() &&
        hidden.defined() && hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==1 && hidden.size(2)==hidden_norm.numel(),
        "Qwen4 MTP session boundary is unavailable");
    MtpSessionState state;state.cache_pos=position;
    for(std::size_t i=0;i<layers.size();++i) {
        const auto count=std::min(position,lengths[i]);
        state.qwen4_lengths.push_back(count);
        state.qwen4_blocks.push_back(count?capture_block_session(*layers[i],count,state.bytes):Qwen4BlockSessionState{});
        auto pos=count?positions[i].narrow(-1,0,count).clone():Tensor{};
        state.bytes+=session_tensor_bytes(pos);state.qwen4_positions.push_back(std::move(pos));
    }
    state.last_target_hidden=hidden.clone();state.bytes+=session_tensor_bytes(state.last_target_hidden);
    return state;
}

void Qwen4ExpMtp::restore_session_state(const MtpSessionState& state) {
    MFQ_RUNTIME_CHECK(supports_session_state() && state.cache_pos>0 &&
        state.qwen4_blocks.size()==layers.size() && state.qwen4_positions.size()==layers.size() &&
        state.qwen4_lengths.size()==layers.size() && state.qwen4_lengths.front()==state.cache_pos &&
        state.last_target_hidden.defined() && state.last_target_hidden.dim()==3 &&
        state.last_target_hidden.size(0)==1 && state.last_target_hidden.size(1)==1 &&
        state.last_target_hidden.size(2)==hidden_norm.numel(),"incompatible Qwen4 MTP session state");
    reset(1);
    for(std::size_t i=0;i<layers.size();++i)if(state.qwen4_lengths[i]) {
        restore_block_session(*layers[i],state.qwen4_blocks[i],state.qwen4_lengths[i]);
        positions[i]=state.qwen4_positions[i].clone();lengths[i]=state.qwen4_lengths[i];
    }
}

} // namespace mfq::cuda::qwen4_exp

namespace mfq::cuda {
bool Qwen4Model::adapter_prefetch_token_rows(Qwen4CausalLm& model,const mfq_tensor_backend::Tensor& ids) {
    return reuse_decode_graphs && graph_registry && qwen4_prefetch_decode_ple(model,ids);
}
bool qwen4_prefetch_decode_ple(Qwen4CausalLm& model,const mfq_tensor_backend::Tensor& host_ids) {
    bool issued=false;
    for(const auto& block:model.blocks) {
        auto* qwen4=dynamic_cast<qwen4_exp::Qwen4Block*>(block.get());
        if(qwen4 && qwen4->ple && qwen4->ple->graph_supported()) {
            qwen4->ple->prefetch_tokens(host_ids,true);
            issued=true;
        }
    }
    return issued;
}
struct Qwen4DecodeGraph {
    static inline thread_local bool startup_warmup=false;
    struct PleStage {
        qwen4_exp::Ple* ple;
        mfq_tensor_backend::Tensor rows;
        std::shared_ptr<RowStage> transfer;
        std::vector<qwen4_exp::NgramEmbedding::Context> prefix_context;
        HostBuffer flags{128,true};
        uint32_t* device=nullptr;
        bool profile=false;
        uint64_t calls=0,route_wait_ns=0,upload_ns=0,notify_ns=0;
        PleStage(qwen4_exp::Ple* p,int tokens,int hidden):ple(p) {
            const auto timing=mfq::cuda::runtime_options::trace_ple_timings();profile=timing && *timing;
            charge_tensor_host_bytes(128);
            MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device),flags.data(),0));
            std::memset(flags.data(),0,128);publish_mapped_flag(host()+16);
            transfer=ple->graph_stage(tokens,mfq_tensor_backend::Device{mfq_tensor_backend::kCUDA,mfq_current_cuda_device()});
            rows=transfer->output().reshape({1,tokens,hidden});
        }
        ~PleStage(){
            if(profile)std::cerr<<"ple_stage_host mapped_rows="<<transfer->mapped_rows()<<" calls="<<calls
                <<" route_wait_ns="<<route_wait_ns<<" upload_ns="<<upload_ns<<" notify_ns="<<notify_ns<<'\n';
            tensor_host_bytes.fetch_sub(128);
        }
        uint32_t* host(){return static_cast<uint32_t*>(flags.data());}
    };
    struct Cell {
        mfq_tensor_backend::Tensor ids,positions,cache_positions,host_ids,output,raw_hidden,logits,greedy_host,compact_host;
        qwen4_exp::MtpCompactControl compact_control;
        unsigned used_kinds=0;
        std::vector<std::unique_ptr<PleStage>> ple;
        std::vector<qwen4_exp::RecurrentGraphCheckpoint> checkpoints;
        DecodeWindow window;
        Cell(int tokens,int hidden,const std::vector<std::unique_ptr<Block>>& blocks,cudaStream_t stream,int confirmed,
                std::vector<qwen4_exp::RecurrentGraphCheckpoint>& shared_checkpoints,bool prediction)
            :window(stream,mfq::cuda::runtime_options::layer_timings()) {
            window.set_prediction_enabled(prediction);
            namespace tb=mfq_tensor_backend;
            const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
            ids=tb::zeros({1,tokens},options);positions=tb::zeros({tokens},options);cache_positions=tb::zeros({tokens},options);
            host_ids=tb::zeros({1,tokens},options.device(tb::kCPU));
            greedy_host=tb::empty({tokens},options.device(tb::kCPU).pinned_memory(true));
            compact_host=tb::empty({tokens,64},options.device(tb::kCPU).pinned_memory(true));
            if(confirmed && shared_checkpoints.empty())shared_checkpoints.resize(blocks.size());
            std::size_t layer=0;
            for(const auto& block:blocks) {
                auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                if(b.ple)ple.push_back(std::make_unique<PleStage>(b.ple.get(),tokens,hidden));
                if(confirmed) {
                    auto& shared=shared_checkpoints[layer];
                    qwen4_exp::RecurrentGraphCheckpoint checkpoint;
                    if(b.gdn) {
                        const auto state=b.gdn->graph_state();
                        while(shared.gdn_conv.size()<std::size_t(tokens-confirmed)) {
                            shared.gdn_conv.push_back(tb::empty_like(*state[0]));
                            shared.gdn_state.push_back(tb::empty_like(*state[1]));
                        }
                        checkpoint.gdn_conv.assign(shared.gdn_conv.begin(),shared.gdn_conv.begin()+tokens-confirmed);
                        checkpoint.gdn_state.assign(shared.gdn_state.begin(),shared.gdn_state.begin()+tokens-confirmed);
                    }
                    if(b.ple) {
                        while(shared.ple_conv.size()<std::size_t(tokens-confirmed))
                            shared.ple_conv.push_back(tb::empty_like(b.ple->conv_state()));
                        checkpoint.ple_conv.assign(shared.ple_conv.begin(),shared.ple_conv.begin()+tokens-confirmed);
                    }
                    checkpoints.push_back(std::move(checkpoint));
                }
                ++layer;
            }
        }
    };
    MfqCudaStream stream,ple_stream;
    std::map<std::tuple<int,int,int,bool>,std::unique_ptr<Cell>> cells;
    // Verification widths execute serially and commit/rollback before the next
    // window. Their snapshots can share stable allocations across graph shapes.
    std::vector<qwen4_exp::RecurrentGraphCheckpoint> shared_checkpoints;
    std::vector<qwen4_exp::Qwen4Block*> blocks;
    std::vector<qwen4_exp::Qsa*> qsa;
    mfq_tensor_backend::Tensor position_storage;
    int64_t kv_capacity=0;
    bool failed=false;
    mfq_tensor_backend::Tensor last_compact;
    explicit Qwen4DecodeGraph(Qwen4CausalLm& model):stream(mfq_get_current_cuda_stream()),ple_stream(mfq_get_stream_from_pool()) {
        const bool enabled=model.router_lookahead.value_or(mfq::cuda::runtime_options::router_lookahead_override().value_or(
            moe_expert_cache_ssd_experts(model.execution->moe_expert_cache)));
        for(std::size_t i=0;i<model.blocks.size();++i) {
            auto& current=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
            blocks.push_back(&current);
            current.next_router={};current.next_prefetch={};
            if(enabled && i+1<model.blocks.size()) {
                auto& next=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i+1]);
                MFQ_RUNTIME_CHECK(next.router && next.prefetch,"router lookahead requires the tiered FFN pipeline");
                current.next_router=next.router;current.next_prefetch=next.prefetch;
            }
        }
        for(const auto& block:model.blocks)if(auto* attention=static_cast<qwen4_exp::Qwen4Block&>(*block).qsa.get())qsa.push_back(attention);
    }
    ~Qwen4DecodeGraph() {(void)cudaStreamSynchronize(ple_stream.stream());cells.clear();}
    void restore_legacy() {
        last_compact={};
        cells.clear();shared_checkpoints.clear();for(auto* attention:qsa)attention->leave_graph();
        for(auto* block:blocks)block->retain_graph_storage=false;
    }
    bool reusable() const {
        return !failed && !qsa.empty() && std::all_of(qsa.begin(),qsa.end(),
            [](const auto* attention){return attention->graph_capacity()>0;});
    }

    mfq_tensor_backend::Tensor run(Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int kind,
                                  mfq_tensor_backend::Tensor* raw_hidden=nullptr,int confirmed=0,bool* initialized=nullptr,
                                  const mfq_tensor_backend::Tensor& host_ids={},mfq_tensor_backend::Tensor* logits=nullptr,
                                  std::vector<int32_t>* greedy_tokens=nullptr,bool compact_sampling=true) {
        namespace tb=mfq_tensor_backend;
        if(failed)throw std::runtime_error("MFQ model session requires recovery after its previous error");
        last_compact={};
        if(stream.stream()!=mfq_current_cuda_stream())throw std::runtime_error("MFQ model window changed execution stream");
        const int tokens=static_cast<int>(ids.size(1));
        // These consumers use the same single-row backbone. Keep its hidden
        // state, logits and selected token in one graph instead of reserving
        // a complete set of layer buffers for each output representation.
        const int result_kind=kind;
        if(tokens==1 && kind>=1 && kind<=4)kind=4;
        release_moe_prefill_buffers(model.execution->moe_expert_cache);
        const auto required=model.cache_pos+tokens;
        MFQ_RUNTIME_CHECK(required<=model.max_position_embeddings(),"Qwen4 graph exceeds context capacity");
        if(!qsa.empty() && qsa.front()->graph_capacity()) {
            model.execution->kv_memory->ensure(required);
            kv_capacity=qsa.front()->graph_capacity();
        }
        if(required>kv_capacity) {
            finish_moe_expert_exchanges(model.execution->moe_expert_cache);
            restore_legacy();
            // Small growing chunks preserve dynamic KV allocation without
            // recapturing all MTP batch shapes at every few decode tokens.
            const int64_t chunk=raw_hidden?2048:256;
            kv_capacity=std::min<int64_t>(model.max_position_embeddings(),std::max<int64_t>(chunk,kv_capacity));
            const auto planned=std::max(required,model.decode_reserve_until);
            while(kv_capacity<planned)
                kv_capacity=std::min(model.max_position_embeddings(),kv_capacity>model.max_position_embeddings()/2
                    ? model.max_position_embeddings() : kv_capacity*2);
            if(const auto* trace=std::getenv("MFQ_RUNTIME_TRACE_CUDA_GRAPH");trace && trace[0]=='1')
                std::cerr<<"runtime_qwen4_graph_capacity position="<<model.cache_pos
                    <<" required="<<required<<" planned="<<planned<<" capacity="<<kv_capacity<<'\n';
        }
        const bool prediction=tokens==1 && !model.execution->tq_expert_cache && model.router_lookahead.value_or(
            mfq::cuda::runtime_options::router_lookahead_override().value_or(
                moe_expert_cache_ssd_router_prefetch(model.execution->moe_expert_cache)));
        auto& owner=cells[{tokens,kind,confirmed,prediction}];
        if(!owner)owner=std::make_unique<Cell>(tokens,model.config.hidden,model.blocks,stream.stream(),confirmed,shared_checkpoints,prediction);
        owner->window.set_profile_geometry(raw_hidden?"mtp_target":"target_decode",tokens);
        auto& cell=*owner;
        if(kind==4 && tokens>1)cell.compact_control.set(compact_sampling,"target",tokens);
        // A retained graph's first replay can still pay cold transfer/cache
        // costs after prefill. Exclude that replay from steady MTP policy data.
        if(initialized)*initialized=!cell.window.valid() || !(cell.used_kinds&(1u<<result_kind));
        if(host_ids.defined()) {
            MFQ_RUNTIME_CHECK(!host_ids.is_cuda() && host_ids.scalar_type()==tb::kInt64 &&
                host_ids.sizes()==ids.sizes(),"Qwen4 graph host token geometry changed");
            cell.host_ids.copy_(host_ids);
        } else cell.host_ids.copy_(ids.to(tb::kCPU,tb::kInt64));
        cell.ids.copy_(ids);
        mfq_qwen4_exp::fill_decode_positions(cell.cache_positions,cell.positions,model.cache_pos,model.decode_position_delta);
        try {
            // PLE depends only on token IDs and the committed ngram context.
            // Start its read before waiting for adaptive expert exchanges,
            // rather than leaving that CPU interval outside the I/O overlap.
            // Task reset repeats issue() idempotently; it never advances the
            // context, which is still committed by upload_graph().
            const auto* early_ple=std::getenv("MFQ_PLE_EARLY_ISSUE");
            if(!early_ple || early_ple[0]!='0')
                for(const auto& stage:cell.ple)stage->ple->prefetch_tokens(cell.host_ids);
            finish_moe_expert_exchanges(model.execution->moe_expert_cache);
            // Prefix restoration and eager prefill update the same KV/index
            // storage; refresh pooled keys even when this graph is already valid.
            for(auto* attention:qsa)attention->prepare_graph(model.positions,kv_capacity);
            if(!cell.window.valid()) {
                if(!prepare_moe_graph_pool(model.execution->moe_expert_cache))
                    prepare_tq_graph_pool(model.execution->tq_expert_cache);
                DecodeGraphBranchScope branch_scope(
                    model.execution->decode_graph_serial_branches);
                const auto capture_started=std::chrono::steady_clock::now();
                struct State {tb::Tensor* target;tb::Tensor saved;const void* address;};
                std::vector<State> states;
                if(!startup_warmup)
                    for(const auto& block:model.blocks)for(auto* state:block->graph_warmup_state())
                        states.push_back({state,state->clone(),state->data_ptr()});
                auto restore=[&] {
                    for(const auto& state:states) {
                        MFQ_RUNTIME_CHECK(state.target->data_ptr()==state.address,"MFQ capture changed recurrent storage");
                        state.target->copy_(state.saved);
                    }
                };
                cell.window.capture([&] {
                    cell.window.mark(-1,"window_begin");
                    auto hidden=model.adapter_prepare_hidden(model.embed_tokens(cell.ids),1,tokens);
                    std::size_t ple_index=0;int layer=0;
                    std::vector<tb::Tensor> prior_mix;
                    const auto chain_setting=mfq::cuda::runtime_options::gr_chain_ffn_post();
                    const bool chain_post=(!chain_setting || *chain_setting) && tokens<=8;
                    for(const auto& block:model.blocks) {
                        cell.window.mark(layer,"block_begin");
                        auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);tb::Tensor rows;
                        if(b.ple) {
                            cell.window.mark(layer,"ple_begin");
                            auto* stage=cell.ple[ple_index++].get();rows=stage->rows;
                            cell.window.enroll({stage,
                                [&cell,stage] {std::memset(stage->flags.data(),0,128);stage->ple->prefetch_tokens(cell.host_ids);},
                                [this,&cell,stage,confirmed] {
                                    using Clock=std::chrono::steady_clock;
                                    auto point=stage->profile?Clock::now():Clock::time_point{};
                                    const auto sample=[&](uint64_t& total) {
                                        if(!stage->profile)return;
                                        const auto now=Clock::now();
                                        total+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-point).count();point=now;
                                    };
                                    wait_route_publication(stage->host(),stream.stream());
                                    sample(stage->route_wait_ns);
                                    MfqCudaStreamGuard guard(ple_stream);
                                    if(confirmed) {
                                        stage->prefix_context.clear();
                                        for(int i=confirmed;i<cell.ids.size(1);++i)
                                            stage->prefix_context.push_back(stage->ple->graph_prefix_context(cell.host_ids,i));
                                    }
                                    stage->ple->upload_graph(cell.host_ids,*stage->transfer);
                                    sample(stage->upload_ns);
                                    if(stage->transfer->mapped_rows())publish_mapped_flag(stage->host()+16);
                                    else {
                                        MFQ_CUDA_CHECK(cudaLaunchHostFunc(ple_stream.stream(),[](void* p){publish_mapped_flag(static_cast<uint32_t*>(p));},stage->host()+16));
                                        check_stream_progress(ple_stream.stream());
                                    }
                                    sample(stage->notify_ns);if(stage->profile)++stage->calls;
                                },{},[stage]{publish_mapped_flag(stage->host()+16);}});
                            signal_mapped_flag(stage->device,stream.stream());wait_mapped_flag(stage->device+16,stream.stream());
                            stage->transfer->decode();
                            cell.window.mark(layer,"ple_end");
                        }
                        cell.window.mark(layer,"attention_begin");
                        std::vector<tb::Tensor> next_mix;
                        bool defer=false;
                        if(chain_post && size_t(layer+1)<model.blocks.size()) {
                            auto& next=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[layer+1]);
                            defer=!next.ple && next.attention_gr.can_fuse(*model.execution,
                                prior_mix.empty()?hidden:prior_mix[1]);
                        }
                        hidden=b.execute_graph(*model.execution,std::move(hidden),cell.ids,cell.positions,
                            cell.cache_positions,rows,layer,confirmed,confirmed?&cell.checkpoints[layer]:nullptr,
                            prior_mix,defer?&next_mix:nullptr);
                        prior_mix=std::move(next_mix);
                        cell.window.mark(layer++,"block_end");
                    }
                    if(kind>=3)cell.raw_hidden=hidden.clone();
                    hidden=model.adapter_finalize_hidden(std::move(hidden),model.output_norm,1,tokens);
                    if(kind>=3) {
                        cell.output=std::move(hidden);
                        if(kind==4) {
                            cell.logits=model.logits_from_hidden(cell.output);
                            if(tokens>1 && cell.logits.size(-1)>=64) {
                                auto compact=sample_top_k_compact_if_cuda(cell.logits.reshape({tokens,-1}).contiguous(),64,
                                    cell.compact_control.device);
                                MFQ_CUDA_CHECK(cudaMemcpyAsync(cell.compact_host.data_ptr(),compact.data_ptr(),tokens*64*sizeof(int64_t),
                                    cudaMemcpyDeviceToHost,stream.stream()));
                            }
                            auto greedy=sample_greedy_cuda(cell.logits.reshape({tokens,-1}).contiguous());
                            MFQ_CUDA_CHECK(cudaMemcpyAsync(cell.greedy_host.data_ptr(),greedy.data_ptr(),tokens*sizeof(int64_t),
                                cudaMemcpyDeviceToHost,stream.stream()));
                        }
                    }
                    else if(kind==2)cell.output=model.adapter_next_token(model.lm_head,model.last_hidden(std::move(hidden)));
                    else if(kind==1)cell.output=model.adapter_last_logits(model.lm_head,model.last_hidden(std::move(hidden)));
                    else cell.output=model.logits_from_hidden(std::move(hidden));
                    cell.window.mark(-1,"window_end");
                    auto audit=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks.front()).router_audit;
                    if(audit)cell.window.enroll({audit.get(),[]{},[]{},[audit]{audit->finish_sample();},{}});
                },restore,[&]{cell.output={};cell.raw_hidden={};cell.logits={};});
                ++model.decode_graph_captures;
                std::cerr<<"runtime_qwen4_graph_capture role=target rows="<<tokens<<" kind="<<kind
                    <<" confirmed="<<confirmed<<" segments="<<cell.window.segments()<<" count="<<model.decode_graph_captures
                    <<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-capture_started).count()<<'\n';
            }
            cell.window.run();cell.used_kinds|=1u<<result_kind;
            if(confirmed) {
                // Captured copies save the confirmed prefix. Bind their owned
                // storage only after replay, so generic commit/rollback keeps
                // exactly the same transaction lifecycle as eager execution.
                std::size_t ple_index=0,layer=0;
                for(const auto& block:model.blocks) {
                    auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                    const auto& checkpoint=cell.checkpoints[layer++];
                    const auto position=model.cache_pos+confirmed;
                    if(b.gdn)b.gdn->bind_graph_checkpoint(checkpoint.gdn_conv,checkpoint.gdn_state,position);
                    if(b.ple)b.ple->bind_graph_checkpoint(checkpoint.ple_conv,cell.ple[ple_index++]->prefix_context,position);
                }
                model.speculative_start=model.cache_pos;model.speculative_confirmed=confirmed;
                model.speculative_prefix_retention=true;
            }
            for(const auto& block:model.blocks) {
                auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);if(b.qsa)b.qsa->advance_graph(tokens);
            }
            model.cache_pos+=tokens;
            model.positions=append_decode_positions(model.positions,cell.positions,position_storage,
                model.max_position_embeddings());model.batch=1;
            // MTP can retain history until a later probe. Each step must own its
            // raw residual rather than alias the next replay's capture buffer.
            if(raw_hidden)*raw_hidden=cell.raw_hidden.clone();
            if(kind==4 && tokens>1 && compact_sampling && cell.logits.size(-1)>=64)last_compact=cell.compact_host;
            if(logits)*logits=cell.logits.clone();
            if(greedy_tokens) {
                const auto* first=cell.greedy_host.data_ptr<int64_t>();
                greedy_tokens->assign(first,first+tokens);
            }
            if(tokens==1 && result_kind==1)return cell.logits.reshape({1,-1});
            if(tokens==1 && result_kind==2)return cell.greedy_host;
            return cell.output;
        } catch(...) {failed=true;throw;}
    }
};
mfq_tensor_backend::Tensor qwen4_decode_graph_compact_topk(const Qwen4CausalLm& model) {
    return model.graph_registry?model.graph_registry->last_compact:mfq_tensor_backend::Tensor{};
}

std::optional<mfq_tensor_backend::Tensor> qwen4_decode_graph_logits(Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int kind,
        const mfq_tensor_backend::Tensor& host_ids) {
    if (!streamed_decode_graphs(*model.execution)) return {};
    const bool tq=bool(model.execution->tq_expert_cache);
    if((!tq && (!model.execution->config.moe_pipeline || !model.execution->moe_expert_cache)) ||
            model.cache_pos==0 || model.speculative_start>=0)return {};
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 || model.positions.dim()!=1)return {};
    if(tq) {
        if(!tq_graph_supports_tokens(ids.size(1)))return {};
        for(const auto& block:model.blocks) {
            const auto& b=static_cast<const qwen4_exp::Qwen4Block&>(*block);
            if(!b.tq_graph_ffn || (b.ple && !b.ple->graph_supported()))return {};
        }
    }
    MFQ_RUNTIME_CHECK(model.cache_pos+ids.size(1)<=model.max_position_embeddings(),"MFQ window exceeds context");
    if(!model.graph_registry)model.graph_registry=std::make_unique<Qwen4DecodeGraph>(model);
    return model.graph_registry->run(model,model.device_ids(std::move(ids)),kind,nullptr,0,nullptr,host_ids);
}

void qwen4_report_decode_graph(const Qwen4CausalLm& model,std::ostream& output) {
    if(!model.graph_registry)return;
    for(const auto& [key,cell]:model.graph_registry->cells)
        for(const auto& phase:cell->window.gpu_phases())if(phase.samples)
            output<<"mtp_graph_phase tokens="<<std::get<0>(key)<<" confirmed="<<std::get<2>(key)
                <<" prediction="<<std::get<3>(key)
                <<" layer="<<phase.layer<<" from="<<phase.from<<" to="<<phase.to
                <<" samples="<<phase.samples<<" total_ns="<<phase.total_ns<<'\n';
}

std::vector<std::pair<std::string,double>> qwen4_decode_graph_profile_metrics(
        const Qwen4CausalLm& model,const MtpModule* mtp) {
    std::vector<std::pair<std::string,double>> result;
    if(!mfq::cuda::runtime_options::layer_timings())return result;
    const auto append=[&](const auto& cells,const char* role,const char* attribute,const char* state) {
        for(const auto& [key,cell]:cells) {
            auto prefix=std::string("cuda_graph_profile_")+role+"_rows_"+
                std::to_string(std::get<0>(key))+"_"+attribute+"_"+std::to_string(std::get<1>(key))+
                "_"+state+"_"+std::to_string(std::get<2>(key))+"_";
            if constexpr(std::tuple_size_v<std::remove_cvref_t<decltype(key)>> == 4)
                prefix+="prediction_"+std::to_string(std::get<3>(key))+"_";
            struct Sample {int64_t total_ns=0,max_ns=0;uint64_t samples=0;};
            std::map<std::string,Sample> phases;
            uint64_t windows=0;
            for(const auto& phase:cell->window.gpu_phases())if(phase.samples) {
                auto& sample=phases[phase.from+"_to_"+phase.to];
                sample.total_ns+=phase.total_ns;sample.samples+=phase.samples;
                sample.max_ns=std::max(sample.max_ns,phase.max_ns);
                windows=std::max(windows,phase.samples);
            }
            if(!windows)continue;
            result.emplace_back(prefix+"windows",double(windows));
            for(const auto& [name,sample]:phases) {
                result.emplace_back(prefix+name+"_ms",sample.total_ns/1e6);
                result.emplace_back(prefix+name+"_samples",double(sample.samples));
                result.emplace_back(prefix+name+"_max_ms",sample.max_ns/1e6);
            }
        }
    };
    if(model.graph_registry)append(model.graph_registry->cells,"target","kind","confirmed");
    const auto* predictor=dynamic_cast<const qwen4_exp::Qwen4ExpMtp*>(mtp);
    if(predictor && predictor->graph_registry)append(predictor->graph_registry->cells,"predictor","dtype","mode");
    return result;
}

std::optional<mfq_tensor_backend::Tensor> qwen4_decode_graph_hidden(
        Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor* raw_hidden,int64_t confirmed,
        bool* initialized,const mfq_tensor_backend::Tensor& host_ids,mfq_tensor_backend::Tensor* logits,
        std::vector<int32_t>* greedy_tokens,bool compact_sampling) {
    if(initialized)*initialized=false;
    const bool tq=bool(model.execution->tq_expert_cache);
    if(!streamed_decode_graphs(*model.execution) ||
            (!tq && (!model.execution->config.moe_pipeline || !model.execution->moe_expert_cache)) ||
            model.speculative_start>=0)return {};
    // Retained windows own initialized, reset recurrent/KV storage. The
    // first row of a new prompt can replay them without an eager FFN copy.
    if(model.cache_pos==0 && (!model.graph_registry || !model.graph_registry->reusable()))return {};
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 ||
        (model.cache_pos!=0 && model.positions.dim()!=1) ||
        confirmed<0 || (confirmed && confirmed>=ids.size(1)))return {};
    if(tq && !tq_graph_supports_tokens(ids.size(1)))return {};
    for(const auto& block:model.blocks) {
        const auto& b=static_cast<const qwen4_exp::Qwen4Block&>(*block);
        if((!b.prefetch && !b.tq_graph_ffn) || (b.ple && !b.ple->graph_supported()))return {};
    }
    MFQ_RUNTIME_CHECK(model.cache_pos+ids.size(1)<=model.max_position_embeddings(),"MFQ window exceeds context");
    if(!model.graph_registry)model.graph_registry=std::make_unique<Qwen4DecodeGraph>(model);
    return model.graph_registry->run(model,model.device_ids(std::move(ids)),(logits || greedy_tokens)?4:3,raw_hidden,
        static_cast<int>(confirmed),initialized,host_ids,logits,greedy_tokens,compact_sampling);
}

void qwen4_release_decode_graphs(Qwen4CausalLm& model,MtpModule* mtp) {
    finish_moe_expert_exchanges(model.execution->moe_expert_cache);
    if(model.graph_registry)model.graph_registry->restore_legacy();
    model.graph_registry.reset();
    for(auto& block:model.blocks)static_cast<qwen4_exp::Qwen4Block&>(*block).retain_graph_storage=false;
    if(auto* predictor=dynamic_cast<qwen4_exp::Qwen4ExpMtp*>(mtp)) {
        if(predictor->graph_registry)predictor->graph_registry->restore_legacy(*predictor);
        predictor->graph_registry.reset();
        for(auto& layer:predictor->layers)layer->retain_graph_storage=false;
    }
}

std::size_t qwen4_mtp_graph_reserve(const Qwen4CausalLm& model,const MtpModule* mtp,int maximum_draft_depth) {
    const auto* predictor=dynamic_cast<const qwen4_exp::Qwen4ExpMtp*>(mtp);
    const auto* reuse=std::getenv("MFQ_QWEN4_REUSE_GRAPHS");
    const auto* warmup=std::getenv("MFQ_QWEN4_WARMUP_GRAPHS");
    if(!predictor || (reuse && reuse[0]=='0') || (warmup && warmup[0]=='0') ||
            !model.execution->config.moe_pipeline || !streamed_decode_graphs(*model.execution))return 0;
    // The common decode allowance covers verification through four rows.
    // Wider windows retain extra shared recurrent checkpoints and per-shape
    // graph/staging allocations. Price those before admitting expert arenas.
    const int extra_rows=std::max(0,std::min({8,predictor->maximum_draft_depth()+1,maximum_draft_depth+1})-4);
    const auto& c=model.config;
    std::size_t recurrent_bytes=0;
    for(int64_t layer=0;layer<c.layers;++layer)if(c.linear_layer(layer))
        recurrent_bytes+=(std::size_t(c.kernel-1)*(2*c.key_heads+c.value_heads)*c.linear_width+
            std::size_t(c.value_heads)*c.linear_width*c.linear_width)*sizeof(float);
    return std::size_t(extra_rows)*(recurrent_bytes+(std::size_t(256)<<20));
}

void qwen4_warmup_decode_graphs(Qwen4CausalLm& model,MtpModule* mtp,int maximum_draft_depth) {
    const auto* reuse=std::getenv("MFQ_QWEN4_REUSE_GRAPHS");
    model.reuse_decode_graphs=!reuse || reuse[0]!='0';
    auto* predictor=dynamic_cast<qwen4_exp::Qwen4ExpMtp*>(mtp);
    if(predictor)predictor->reuse_decode_graphs=model.reuse_decode_graphs;
    const bool tq=bool(model.execution->tq_expert_cache);
    // Keep TQ startup to the target decode window. Wider verification shapes
    // are captured on demand; predictor graph preparation remains separate.
    if(tq)predictor=nullptr;
    const auto* warmup=std::getenv("MFQ_QWEN4_WARMUP_GRAPHS");
    if(!model.reuse_decode_graphs || (warmup && warmup[0]=='0') ||
        model.max_position_embeddings()<32 || !streamed_decode_graphs(*model.execution) ||
        (!tq && (!model.execution->config.moe_pipeline || !model.execution->moe_expert_cache)))return;
    for(const auto& block:model.blocks) {
        const auto& b=static_cast<const qwen4_exp::Qwen4Block&>(*block);
        if((!b.prefetch && !b.tq_graph_ffn) || (b.qsa && !b.qsa->graph_capacity()) || (b.ple && !b.ple->graph_supported()))return;
    }
    // Startup state contains synthetic tokens and is discarded before serving.
    // Saving every recurrent state during each capture needlessly doubles its
    // transient footprint. Request-time captures still restore their real state.
    struct WarmupScope {
        std::shared_ptr<MoeExpertCache> cache;
        bool previous,previous_learning;
        explicit WarmupScope(std::shared_ptr<MoeExpertCache> value):cache(std::move(value)),
            previous(suspend_moe_residency(cache,true)),previous_learning(set_moe_reuse_learning(cache,false))
            {Qwen4DecodeGraph::startup_warmup=true;}
        ~WarmupScope(){Qwen4DecodeGraph::startup_warmup=false;
            set_moe_reuse_learning(cache,previous_learning);suspend_moe_residency(cache,previous);}
    } startup_scope(model.execution->moe_expert_cache);
    namespace tb=mfq_tensor_backend;
    const auto start=std::chrono::steady_clock::now();
    auto scope=model.execution_scope();
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
    // Diverse synthetic IDs exercise more routed experts than sixteen copies
    // of one token. Only loading uses these rows; all request state is reset.
    std::vector<int64_t> warm_ids(16);
    for(std::size_t i=0;i<warm_ids.size();++i)warm_ids[i]=int64_t((i*7919+17)%model.vocab_size());
    auto ids=tb::tensor(warm_ids,options.device(tb::kCPU)).to(tb::kCUDA).reshape({1,16});
    model.reset(1);
    if(std::getenv("MFQ_TRACE_RESIDENCY_MEMORY"))std::cerr<<"warmup_memory phase=bootstrap\n";
    // Bootstrap with single-token rows. A sixteen-row eager forward otherwise
    // materializes a whole expert layer just to initialize graph state, before
    // the prompt's workspace loan exists, and defeats tight expert residency.
    std::vector<tb::Tensor> raw_rows;
    // Eager FFN cells retain their transfer wire per layer until this pass
    // completes; eight-row bootstrap cells multiply that transient footprint.
    for(int row=0;row<(tq?1:16);++row) {
        auto token=ids.narrow(1,row,1);
        tb::Tensor hidden;
        (void)model.hidden_forward(token,mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,&hidden);
        raw_rows.push_back(hidden.clone());
    }
    auto raw=tb::cat(raw_rows,1);
    raw_rows.clear();
    // Bootstrap cells are not the retained whole-model windows. Retire their
    // staging and captured scratch before preparing predictor windows.
    release_moe_decode_buffers(model.execution->moe_expert_cache);
    default_context(model.execution->layer_placement.primary_device())->reclaim_idle_memory();
    if(std::getenv("MFQ_TRACE_RESIDENCY_MEMORY"))std::cerr<<"warmup_memory phase=predictor\n";
    // Capture the normal greedy path and the common MTP verification widths.
    // Synthetic state is cleared afterwards; no prompt or sampling history is saved.
    if(predictor) {
        MtpTarget target(model);
        predictor->reset(1);
        // Full predictor drafts consume one row. Bootstrap the same shape so
        // startup does not retain multi-row expert and pinned transfer arenas.
        for(int row=0;row<16;++row)
            (void)predictor->evaluate(target,raw.narrow(1,row,1),ids.narrow(1,row,1));
        release_moe_decode_buffers(model.execution->moe_expert_cache);
        default_context(model.execution->layer_placement.primary_device())->reclaim_idle_memory();
        // Production drafts run one full predictor row. Committed history
        // appends KV without an FFN/head; multi-row full-predictor graphs
        // retain scratch and driver metadata with no production consumer.
        predictor->trim_cache_to(16);
        (void)predictor->step(target,raw.narrow(1,0,1),ids.narrow(1,0,1));
        const int verification_rows=std::min({8,predictor->maximum_draft_depth()+1,maximum_draft_depth+1});
        for(int rows=4;rows>=1;--rows) {
            predictor->trim_cache_to(16);
            predictor->history_step(target,raw.narrow(1,0,rows),ids.narrow(1,0,rows));
        }
        const auto* chain_option=std::getenv("MFQ_MTP_HISTORY_CHAIN");
        const bool history_chain=!chain_option || chain_option[0]!='0';
        const auto* chain_check=std::getenv("MFQ_VERIFY_MTP_HISTORY_CHAIN");
        const bool verify_chain=chain_check && chain_check[0]=='1' && predictor->layers[0]->qsa->supports_session();
        if(history_chain)for(int rows=2;rows<=std::min(4,verification_rows-1);++rows) {
            Qwen4BlockSessionState expected;
            if(verify_chain) {
                predictor->trim_cache_to(16);
                for(int row=0;row<rows;++row)
                    predictor->history_step(target,raw.narrow(1,row,1),ids.narrow(1,row,1));
                std::size_t bytes=0;
                predictor->layers[0]->qsa->capture_session(expected,16+rows,bytes);
            }
            predictor->trim_cache_to(16);
            predictor->history_step(target,raw.narrow(1,0,rows),ids.narrow(1,0,rows),{},true);
            if(verify_chain) {
                Qwen4BlockSessionState actual;std::size_t bytes=0;
                predictor->layers[0]->qsa->capture_session(actual,16+rows,bytes);
                for(const auto& pair:{std::pair{actual.keys,expected.keys},std::pair{actual.values,expected.values},
                                     std::pair{actual.index,expected.index}}) {
                    const auto a=pair.first.cpu().contiguous(),b=pair.second.cpu().contiguous();
                    MFQ_RUNTIME_CHECK(a.sizes()==b.sizes() && a.scalar_type()==b.scalar_type() &&
                        std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0,"MTP scalar history chain changed KV/index bytes");
                }
                std::cerr<<"runtime_mtp_history_chain_check rows="<<rows<<" KV/index exact PASS\n";
            }
        }
        // A depth-D draft verifies D+1 target rows. Prepare every supported
        // width before requests lend spare VRAM to resident experts; capturing
        // widths five/six afterwards also needs live recurrent-state clones.
        for(int rows=verification_rows;rows>=1;--rows) {
            if(model.cache_pos+rows+2>model.max_position_embeddings())model.reset(1);
            tb::Tensor hidden,logits;
            (void)qwen4_decode_graph_hidden(model,ids.narrow(1,0,rows),&hidden,
                rows>1?1:0,nullptr,{},&logits);
            if(rows>1)model.commit_speculative();
        }
        // Capturing and replaying once leaves cold per-width staging paths.
        // Exercise both single-row drafting and batched history/verification
        // again with different IDs before accepting the first request.
        for(int pass=0;pass<2;++pass)for(int rows=1;rows<=verification_rows;++rows) {
            if(model.cache_pos+rows+2>model.max_position_embeddings())break;
            const auto offset=4+pass*4;
            auto tokens=ids.narrow(1,offset,rows);
            predictor->trim_cache_to(16);
            if(history_chain && rows>2) {
                for(int first=0;first<rows-1;first+=4) {
                    const auto count=std::min(4,rows-1-first);
                    predictor->history_step(target,raw.narrow(1,offset+first,count),tokens.narrow(1,first,count),{},true);
                }
            }
            else for(int row=0;row+1<rows;++row)
                predictor->history_step(target,raw.narrow(1,offset+row,1),tokens.narrow(1,row,1));
            (void)predictor->step(target,raw.narrow(1,offset+rows-1,1),tokens.narrow(1,rows-1,1));
            tb::Tensor hidden,logits;
            (void)qwen4_decode_graph_hidden(model,tokens,&hidden,rows>1?1:0,nullptr,{},&logits);
            if(rows>1)model.commit_speculative();
        }
        predictor->reset(1);
    }
    (void)qwen4_decode_graph_logits(model,ids.narrow(1,0,1),2);
    (void)qwen4_decode_graph_logits(model,ids.narrow(1,0,1),1);
    if(moe_expert_cache_ssd_experts(model.execution->moe_expert_cache) &&
            !model.router_lookahead.has_value() && !runtime_options::router_lookahead_override().has_value()) {
        struct Restore {
            Qwen4CausalLm& model;std::optional<bool> previous;
            ~Restore(){model.router_lookahead=previous;}
        } restore{model,model.router_lookahead};
        model.router_lookahead=true;
        if(model.cache_pos+4>=model.max_position_embeddings())model.reset(1);
        (void)qwen4_decode_graph_logits(model,ids.narrow(1,0,1),1);
    }
    model.reset(1);
    MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    std::cerr<<"runtime_qwen4_graph_warmup target_captures="<<model.decode_graph_captures
        <<" maximum_draft_depth="<<(predictor?maximum_draft_depth:0)
        <<" predictor_captures="<<(predictor?predictor->graph_captures:0)<<" ms="
        <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
    release_moe_prefill_buffers(model.execution->moe_expert_cache);
    default_context(model.execution->layer_placement.primary_device())->reclaim_idle_memory();
}

void qwen4_set_rotary_fusion(Qwen4CausalLm& model, bool enabled) {
    MFQ_RUNTIME_CHECK(!model.graph_registry && model.cache_pos == 0,
                      "reset the model before switching rotary execution");
    mfq_cuda_synchronize();
    for (const auto& block : model.blocks) {
        auto* attention = static_cast<qwen4_exp::Qwen4Block&>(*block).qsa.get();
        if (attention) attention->set_rotary_fused(enabled);
    }
}

void qwen4_set_attention_grouping(Qwen4CausalLm& model, bool enabled) {
    MFQ_RUNTIME_CHECK(!model.graph_registry && model.cache_pos == 0,
                      "reset the model before switching attention projection groups");
    mfq_cuda_synchronize();
    for (const auto& block : model.blocks) {
        auto& attention = static_cast<qwen4_exp::Qwen4Block&>(*block);
        if (attention.gdn) attention.gdn->set_grouped_projection(enabled);
        if (attention.qsa) attention.qsa->set_grouped_projection(enabled);
    }
}

void qwen4_begin_router_lookahead_audit(Qwen4CausalLm& model,int samples) {
    if(model.blocks.size()<2 || model.graph_registry || samples<=0)
        throw std::invalid_argument("router lookahead audit requires a fresh multilayer decode session");
    auto audit=std::make_shared<RouterLookaheadInputs>(static_cast<int>(model.blocks.size()),
        static_cast<int>(model.config.hidden),samples);
    for(const auto& block:model.blocks) {
        auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
        if(!b.router)throw std::runtime_error("router lookahead audit has no retained router");
        b.router_audit=audit;
    }
}
void qwen4_finish_router_lookahead_audit(Qwen4CausalLm& model,const std::string& prefix) {
    auto audit=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks.front()).router_audit;
    if(!audit || !audit->samples())throw std::runtime_error("router lookahead audit captured no decode inputs");
    std::ofstream file(prefix+"-logits.f32",std::ios::binary);
    if(!file)throw std::runtime_error("cannot write router lookahead audit logits");
    for(std::size_t layer=1;layer<model.blocks.size();++layer) {
        auto& b=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[layer]);
        for(int input_layer:{static_cast<int>(layer)-1,static_cast<int>(layer)}) {
            auto input=audit->rows(input_layer).to(mfq_tensor_backend::kCUDA);
            auto logits=b.router(*model.execution,input).to(mfq_tensor_backend::kFloat32).cpu().contiguous();
            if(logits.dim()!=2 || logits.size(0)!=audit->samples() || logits.size(1)!=model.config.experts)
                throw std::runtime_error("router lookahead audit router shape mismatch");
            file.write(reinterpret_cast<const char*>(logits.data_ptr()),logits.numel()*logits.element_size());
            if(!file)throw std::runtime_error("router lookahead audit logits write failed");
        }
    }
    std::cout<<"router_lookahead_audit_samples="<<audit->samples()<<" layers="<<model.blocks.size()
        <<" hidden="<<model.config.hidden<<" experts="<<model.config.experts<<" topk="<<model.config.topk
        <<" scores=predicted_then_same_layer_oracle row_dtype=FP32\n";
}

std::size_t qwen4_prefix_snapshot_bytes(const Qwen4CausalLm& model,const MtpModule* mtp) {
    std::size_t bytes=session_tensor_bytes(model.positions);
    const auto count_block=[&](const qwen4_exp::Qwen4Block& block,int64_t count) {
        if(count<=0)return;
        if(block.qsa)bytes+=block.qsa->session_bytes(count);
        if(block.gdn)for(const auto* tensor:block.gdn->graph_state())bytes+=session_tensor_bytes(*tensor);
        if(block.ple)bytes+=session_tensor_bytes(block.ple->conv_state());
    };
    for(const auto& block:model.blocks)count_block(static_cast<const qwen4_exp::Qwen4Block&>(*block),model.cache_pos);
    if(const auto* predictor=dynamic_cast<const qwen4_exp::Qwen4ExpMtp*>(mtp)) {
        for(std::size_t i=0;i<predictor->layers.size();++i) {
            const auto count=std::min(model.cache_pos-1,predictor->lengths[i]);
            count_block(*predictor->layers[i],count);
            if(count>0)bytes+=std::size_t(count)*3*predictor->positions[i].element_size();
        }
        bytes+=std::size_t(model.config.hidden)*model.config.streams*sizeof(float);
    }
    return bytes;
}

std::vector<std::pair<std::string,double>> qwen4_cache_memory_metrics(
        const Qwen4CausalLm& model,const MtpModule* mtp) {
    // Shared checkpoint views and recurrent state aliases count once. These
    // are existing cache tensors; collecting telemetry never copies GPU data.
    std::map<std::pair<int,std::uintptr_t>,std::size_t> allocations;
    const auto add=[&](const mfq_tensor_backend::Tensor& tensor) {
        if(!tensor.defined() || !tensor.is_cuda())return;
        auto& bytes=allocations[{tensor.get_device(),reinterpret_cast<std::uintptr_t>(tensor.data_ptr())}];
        bytes=std::max(bytes,std::size_t(tensor.numel())*tensor.element_size());
    };
    const auto block_cache=[&](const qwen4_exp::Qwen4Block& block) {
        if(block.qsa)for(const auto& tensor:block.qsa->resident_cache_tensors())add(tensor);
        if(block.gdn)for(const auto* tensor:block.gdn->graph_state())add(*tensor);
        if(block.ple)for(const auto* tensor:block.ple->graph_state())add(*tensor);
    };
    for(const auto& block:model.blocks)block_cache(static_cast<const qwen4_exp::Qwen4Block&>(*block));
    add(model.positions);
    if(model.graph_registry)add(model.graph_registry->position_storage);
    if(model.graph_registry)for(const auto& [key,cell]:model.graph_registry->cells)
        for(const auto& checkpoint:cell->checkpoints) {
            for(const auto& tensor:checkpoint.gdn_conv)add(tensor);
            for(const auto& tensor:checkpoint.gdn_state)add(tensor);
            for(const auto& tensor:checkpoint.ple_conv)add(tensor);
        }
    if(const auto* predictor=dynamic_cast<const qwen4_exp::Qwen4ExpMtp*>(mtp)) {
        for(const auto& block:predictor->layers)block_cache(*block);
        for(const auto& positions:predictor->positions)add(positions);
        if(predictor->graph_registry)add(predictor->graph_registry->position_storage);
    }
    std::size_t bytes=0;
    std::map<int,std::size_t> devices;
    for(const auto& [address,size]:allocations) { bytes+=size; devices[address.first]+=size; }
    if(model.execution->kv_memory)
        for(const auto& metric:model.execution->kv_memory->metrics())
            if(metric.first=="kv_memory_gpu_bytes") {
                bytes+=std::size_t(metric.second);
                devices[model.execution->layer_placement.primary_device()]+=std::size_t(metric.second);
            }
    if(model.execution->qsa_kv_store)
        for(const auto& metric:model.execution->qsa_kv_store->metrics())
            if(metric.first=="qsa_kv_resident_bytes") {
                bytes+=std::size_t(metric.second);
                devices[model.execution->layer_placement.primary_device()]+=std::size_t(metric.second);
            }
    std::vector<std::pair<std::string,double>> result{{"kv_cache_bytes",double(bytes)},
        {"kv_cache_contexts",model.cache_pos>0?double(model.batch):0.0}};
    for(const auto& [device,size]:devices)result.emplace_back("gpu_"+std::to_string(device)+"_live_kv_bytes",double(size));
    return result;
}

Qwen4Model::Qwen4Model() = default;
Qwen4Model::~Qwen4Model() = default;
Qwen4Model::Qwen4Model(Qwen4Model&&) noexcept = default;
Qwen4Model& Qwen4Model::operator=(Qwen4Model&&) noexcept = default;

struct Qwen4BatchRequestState::Impl {
    struct Layer {
        std::unique_ptr<qwen4_exp::Gdn> gdn;
        std::unique_ptr<qwen4_exp::Qsa> qsa;
        std::unique_ptr<qwen4_exp::Ple> ple;
    };
    std::vector<Layer> layers;
    mfq_tensor_backend::Tensor positions;
    mfq_tensor_backend::Tensor position_storage;
    int64_t cache_position = 0, batch = 1;
    int64_t verify_start = -1, verify_width = 0;
    // Slots of the shared pool that hold this window's recurrent checkpoints.
    std::shared_ptr<Qwen4BatchCheckpointPool> checkpoint_pool;
    std::vector<int64_t> checkpoint_slots;
    void release_checkpoints() noexcept {
        if (checkpoint_pool) checkpoint_pool->release(checkpoint_slots);
        checkpoint_pool.reset();
        checkpoint_slots.clear();
    }
    ~Impl() { release_checkpoints(); }
    void swap(Qwen4CausalLm& model) {
        for (std::size_t i = 0; i < layers.size(); ++i) {
            auto& block = static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
            layers[i].gdn.swap(block.gdn);
            layers[i].qsa.swap(block.qsa);
            layers[i].ple.swap(block.ple);
        }
        std::swap(positions, model.positions);
        std::swap(cache_position, model.cache_pos);
        std::swap(batch, model.batch);
    }
    std::size_t growth_bytes(int64_t tokens) const {
        std::size_t bytes = 0;
        for (const auto& layer : layers) {
            if (layer.qsa) bytes += layer.qsa->growth_bytes(tokens);
            if (layer.gdn) bytes += layer.gdn->growth_bytes();
        }
        return bytes;
    }
};

Qwen4BatchRequestState::Qwen4BatchRequestState(Qwen4CausalLm& model)
    : state_(std::make_unique<Impl>()) {
    const auto primary = model.execution->layer_placement.primary_device();
    for (const auto& base : model.blocks) {
        MFQ_RUNTIME_CHECK(!base->cpu_offloaded && base->cuda_device == primary,
                          "Flash-Next continuous batching requires one CUDA device");
        const auto& block = static_cast<const qwen4_exp::Qwen4Block&>(*base);
        Impl::Layer layer;
        if (block.gdn) {
            layer.gdn = std::make_unique<qwen4_exp::Gdn>(*block.gdn);
            layer.gdn->reset();
        }
        if (block.qsa) layer.qsa = block.qsa->fork_empty();
        if (block.ple) { layer.ple = std::make_unique<qwen4_exp::Ple>(*block.ple); layer.ple->reset(); }
        state_->layers.push_back(std::move(layer));
    }
    std::size_t fixed=0;
    for(const auto& layer:state_->layers)if(layer.gdn)fixed+=layer.gdn->request_cache_growth_bytes();
    if(fixed && model.reserve_kv_workspace)model.reserve_kv_workspace(fixed+(std::size_t(64)<<20));
    for(auto& layer:state_->layers)if(layer.gdn)layer.gdn->initialize_request_cache(false);
}
Qwen4BatchRequestState::~Qwen4BatchRequestState() = default;
std::vector<mfq_tensor_backend::Tensor> Qwen4BatchRequestState::recurrent_state_tensors() const {
    std::vector<mfq_tensor_backend::Tensor> result;
    for (const auto& layer : state_->layers) {
        if (layer.gdn) {
            for (const auto& tensor : layer.gdn->indirect_state_tensors())
                if (tensor.defined()) result.push_back(tensor);
        }
        if (layer.ple && layer.ple->conv_state().defined()) result.push_back(layer.ple->conv_state());
    }
    return result;
}
int64_t Qwen4BatchRequestState::position() const { return state_->cache_position; }
void Qwen4BatchRequestState::resolve_verified(int64_t retained) {
    auto& state = *state_;
    MFQ_RUNTIME_CHECK(state.verify_start >= 0 && retained > 0 && retained <= state.verify_width,
                      "invalid independent target commit");
    const auto boundary = state.verify_start + retained;
    for (auto& layer : state.layers) {
        if (retained == state.verify_width) {
            if (layer.gdn) layer.gdn->commit();
            if (layer.ple) layer.ple->commit();
        } else {
            if (layer.gdn) layer.gdn->rollback(boundary);
            if (layer.ple) layer.ple->rollback(boundary);
            if (layer.qsa) layer.qsa->truncate(boundary);
        }
    }
    state.positions = state.positions.narrow(-1, 0, boundary);
    state.cache_position = boundary;
    state.verify_start = -1;
    state.verify_width = 0;
    state.release_checkpoints();
}

Qwen4BatchCheckpointPool::Qwen4BatchCheckpointPool(int64_t tokens) : busy_(std::max<int64_t>(0, tokens), false) {}

int64_t Qwen4BatchCheckpointPool::available() const noexcept {
    return std::count(busy_.begin(), busy_.end(), false);
}

std::vector<int64_t> Qwen4BatchCheckpointPool::acquire(int64_t count) {
    MFQ_RUNTIME_CHECK(count >= 0 && count <= available(), "batched verification exceeds checkpoint slots");
    std::vector<int64_t> slots;
    for (int64_t slot = 0; slot < tokens() && int64_t(slots.size()) < count; ++slot)
        if (!busy_[slot]) {
            busy_[slot] = true;
            slots.push_back(slot);
        }
    return slots;
}

void Qwen4BatchCheckpointPool::release(const std::vector<int64_t>& slots) noexcept {
    for (const auto slot : slots) busy_[slot] = false;
}

void Qwen4BatchCheckpointPool::allocate(Qwen4BatchRequestState& request) {
    namespace tb = mfq_tensor_backend;
    if (!gdn_conv_.empty() || !tokens()) return;
    // Rows share one geometry; slot `i` of a layer is rows [i, i + 1).
    const auto stack = [&](const Tensor& state) {
        if (!state.defined()) return Tensor{};
        MFQ_RUNTIME_CHECK(state.size(0) == 1, "batched checkpoints require one-row request state");
        auto shape = state.sizes().vec();
        shape[0] = tokens();
        return tb::empty(shape, state.options());
    };
    mfq::cuda::ResidentAllocationScope resident;
    // Keep the pool empty until every layer succeeds. An allocation failure
    // must not make a later verification reuse a partially initialized pool.
    std::vector<Tensor> convolution, recurrent, positional;
    convolution.reserve(request.state_->layers.size());
    recurrent.reserve(request.state_->layers.size());
    positional.reserve(request.state_->layers.size());
    for (auto& layer : request.state_->layers) {
        convolution.push_back(layer.gdn ? stack(*layer.gdn->graph_state()[0]) : Tensor{});
        recurrent.push_back(layer.gdn ? stack(*layer.gdn->graph_state()[1]) : Tensor{});
        positional.push_back(layer.ple ? stack(layer.ple->conv_state()) : Tensor{});
    }
    gdn_conv_.swap(convolution);
    gdn_state_.swap(recurrent);
    ple_conv_.swap(positional);
}
std::size_t Qwen4BatchRequestState::prefill_growth_bytes(int64_t tokens) const {
    return state_->growth_bytes(tokens);
}
std::size_t Qwen4BatchRequestState::resident_bytes() const {
    std::map<std::pair<int,std::uintptr_t>,std::size_t> allocations;
    const auto add = [&](const mfq_tensor_backend::Tensor& tensor) {
        if (!tensor.defined() || !tensor.is_cuda()) return;
        auto& bytes = allocations[{tensor.get_device(),reinterpret_cast<std::uintptr_t>(tensor.data_ptr())}];
        bytes = std::max(bytes, std::size_t(tensor.numel()) * tensor.element_size());
    };
    add(state_->positions);
    add(state_->position_storage);
    for (const auto& layer : state_->layers) {
        if (layer.qsa) for (const auto& tensor : layer.qsa->resident_cache_tensors()) add(tensor);
        if (layer.gdn) for (const auto* tensor : layer.gdn->graph_state()) add(*tensor);
        if (layer.ple) for (const auto* tensor : layer.ple->graph_state()) add(*tensor);
    }
    std::size_t bytes = 0;
    for (const auto& [address,size] : allocations) bytes += size;
    return bytes;
}
mfq_tensor_backend::Tensor Qwen4BatchRequestState::prefill(
        Qwen4CausalLm& model, mfq_tensor_backend::Tensor ids, mfq_tensor_backend::Tensor* raw) {
    MFQ_RUNTIME_CHECK(ids.size(0) == 1 && ids.size(1) > 0 &&
        ids.size(1) <= model.max_position_embeddings() - position(), "invalid Flash-Next request prefill");
    if(ids.size(1)>1024)reserve_moe_workspace(model.execution->moe_expert_cache,runtime_options::workspace_reserve_bytes());
    if (model.reserve_kv_workspace)
        model.reserve_kv_workspace(state_->growth_bytes(ids.size(1)) + (std::size_t(256) << 20));
    state_->swap(model);
    try {
        auto hidden = model.hidden_forward(std::move(ids), mfq_nullopt, mfq_nullopt, nullptr, mfq_nullopt, raw);
        state_->swap(model);
        return hidden;
    } catch (...) { state_->swap(model); throw; }
}

#include "batch_graph.h"

std::vector<mfq_tensor_backend::Tensor> qwen4_batch_prefill(Qwen4CausalLm& model,
        const std::vector<Qwen4BatchRequestState*>& requests,
        const std::vector<mfq_tensor_backend::Tensor>& chunks) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(!requests.empty() && requests.size()==chunks.size(),"invalid packed prefill rows");
    auto& execution = *model.execution;
    model.adapter_begin_forward(false);
    std::vector<int64_t> offsets{0};
    std::vector<tb::Tensor> current,full;
    for (std::size_t row=0;row<requests.size();++row) {
        const auto& state=*requests[row]->state_;
        const auto& ids=chunks[row];
        MFQ_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 && ids.size(1)>0 &&
            ids.size(1)<=model.max_position_embeddings()-state.cache_position,"invalid packed prefill chunk");
        auto positions=tb::arange(ids.size(1),ids.options())+state.cache_position;
        current.push_back(positions);
        full.push_back(state.positions.defined()?tb::cat({state.positions,positions},-1):positions);
        offsets.push_back(offsets.back()+ids.size(1));
    }
    auto ids=tb::cat(chunks,1);
    auto host_ids=ids.to(tb::kCPU);
    const auto tokens=offsets.back();
    if(tokens>1024)reserve_moe_workspace(model.execution->moe_expert_cache,runtime_options::workspace_reserve_bytes());
    auto hidden=model.adapter_prepare_hidden(model.embed_tokens(ids),1,tokens);
    for (std::size_t i=0;i<model.blocks.size();++i) {
        NintPrefillDecodeCacheScope decode_reuse(true);
        auto& block=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
        if (block.ple) {
            std::vector<tb::Tensor> positional;
            for (std::size_t row=0;row<requests.size();++row) {
                auto x=hidden.narrow(1,offsets[row],chunks[row].size(1));
                positional.push_back(x+requests[row]->state_->layers[i].ple->forward(
                    execution,x,chunks[row],true,0,host_ids.narrow(1,offsets[row],chunks[row].size(1))));
            }
            hidden=join_request_outputs(positional,1);
        }
        auto mix=block.attention_gr.pre(execution,hidden);
        const auto projections=block.gdn?block.gdn->project_inputs(execution,mix[0])
                                         :block.qsa->project_inputs(execution,mix[0]);
        std::vector<tb::Tensor> attended;
        for (std::size_t row=0;row<requests.size();++row) {
            auto& layer=requests[row]->state_->layers[i];
            auto branch=mix[0].narrow(1,offsets[row],chunks[row].size(1));
            std::vector<tb::Tensor> projected;
            for (const auto& value:projections)
                projected.push_back(value.narrow(1,offsets[row],chunks[row].size(1)));
            attended.push_back(layer.gdn?layer.gdn->forward_chunk(execution,branch,true,std::move(projected),false)
                :layer.qsa->forward(execution,branch,current[row],full[row],true,nullptr,std::move(projected),false));
        }
        auto attention=join_request_outputs(attended,1);
        attention=block.gdn?block.gdn->project_output(execution,attention):block.qsa->project_output(execution,attention);
        auto ffn_mix=block.ffn_gr.pre_after(execution,std::move(attention),mix);
        // All prompt tokens share each expert transfer and compressed decode.
        hidden=block.ffn_gr.post(block.ffn(execution,ffn_mix[0]),ffn_mix);
    }
    hidden=model.finalize_hidden(std::move(hidden),1,tokens);
    std::vector<tb::Tensor> result;
    for (std::size_t row=0;row<requests.size();++row) {
        result.push_back(hidden.narrow(1,offsets[row+1]-1,1));
        requests[row]->state_->positions=std::move(full[row]);
        requests[row]->state_->cache_position+=chunks[row].size(1);
    }
    return result;
}

mfq_tensor_backend::Tensor qwen4_batch_decode(Qwen4CausalLm& model,
        const std::vector<Qwen4BatchRequestState*>& requests, mfq_tensor_backend::Tensor ids,
        const mfq_tensor_backend::Tensor& host_ids, mfq_tensor_backend::Tensor* raw,
        const std::vector<int>& widths, bool speculative,
        const std::shared_ptr<Qwen4BatchCheckpointPool>& checkpoints) {
    namespace tb = mfq_tensor_backend;
    const auto batch = static_cast<int64_t>(requests.size());
    MFQ_RUNTIME_CHECK(batch > 0 && ids.dim() == 2 && ids.size(0) == batch && ids.size(1) > 0 &&
        (widths.empty() || widths.size() == requests.size()),
                      "invalid Flash-Next batch input");
    const auto tokens = ids.size(1);
    const auto row_width = [&](int64_t row) { return widths.empty() ? tokens : int64_t(widths[row]); };
    std::size_t growth = 0;
    std::vector<tb::Tensor> current, full;
    for (int64_t row = 0; row < batch; ++row) {
        auto& state = *requests[row]->state_;
        const auto width = row_width(row);
        MFQ_RUNTIME_CHECK(width > 0 && width <= tokens &&
            width <= model.max_position_embeddings() - state.cache_position, "batch request exceeds max_context");
        MFQ_RUNTIME_CHECK(state.verify_start < 0, "independent target has an unresolved window");
        if (speculative) {
            state.verify_start = state.cache_position;
            state.verify_width = width;
        }
        growth += state.growth_bytes(width);
        auto position = tb::arange(width, ids.options()) + state.cache_position;
        current.push_back(position);
        full.push_back(state.positions.defined() ? tb::cat({state.positions,position},-1) : position);
    }
    if (growth && model.reserve_kv_workspace)
        model.reserve_kv_workspace(growth + (std::size_t(64) << 20));
    const bool pooled = speculative && checkpoints;
    if (pooled) {
        checkpoints->allocate(*requests.front());
        for (int64_t row = 0; row < batch; ++row) {
            auto& state = *requests[row]->state_;
            state.checkpoint_slots = checkpoints->acquire(row_width(row) - 1);
            state.checkpoint_pool = checkpoints;
        }
    }
    // Slot `index` of a row's window, or fresh storage without a pool.
    using Slots = std::vector<Tensor> Qwen4BatchCheckpointPool::*;
    const auto checkpoint = [&](Slots pool, std::size_t layer, int64_t row, int64_t index, const Tensor& like) {
        if (!pooled) return tb::empty_like(like);
        return ((*checkpoints).*pool)[layer].narrow(0, requests[row]->state_->checkpoint_slots[index], 1);
    };
    auto& execution = *model.execution;
    auto hidden = model.adapter_prepare_hidden(model.embed_tokens(ids),batch,tokens);
    const auto padded = [&](Tensor value, int64_t width) {
        if (width == tokens) return value;
        auto shape = value.sizes().vec();
        shape[1] = tokens - width;
        return tb::cat({value, tb::zeros(shape, value.options())}, 1);
    };
    for (std::size_t i = 0; i < model.blocks.size(); ++i) {
        auto& block = static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
        if (block.ple) {
            std::vector<tb::Tensor> positional;
            for (int64_t row = 0; row < batch; ++row) {
                auto& layer = requests[row]->state_->layers[i];
                const auto width = row_width(row);
                auto x = hidden.narrow(0,row,1).narrow(1,0,width);
                auto row_ids = ids.narrow(0,row,1).narrow(1,0,width);
                auto row_host = host_ids.narrow(0,row,1).narrow(1,0,width);
                Tensor positional_value;
                if (speculative && width > 1) {
                    std::vector<Tensor> outputs, saved_conv;
                    std::vector<qwen4_exp::NgramEmbedding::Context> contexts;
                    for (int64_t token = 0; token < width; ++token) {
                        outputs.push_back(layer.ple->forward_chunk(execution, x.narrow(1,token,1),
                            row_ids.narrow(1,token,1), true, row_host.narrow(1,token,1)));
                        if (token + 1 < width) {
                            auto saved = checkpoint(&Qwen4BatchCheckpointPool::ple_conv_, i, row, token, layer.ple->conv_state());
                            saved.copy_(layer.ple->conv_state());
                            saved_conv.push_back(std::move(saved));
                            contexts.push_back(layer.ple->session_context());
                        }
                    }
                    layer.ple->bind_graph_checkpoint(saved_conv, std::move(contexts),
                        requests[row]->state_->verify_start + 1);
                    positional_value = tb::cat(outputs,1);
                } else positional_value = layer.ple->forward(execution,x,row_ids,true,0,row_host);
                positional.push_back(padded(x + positional_value, width));
            }
            hidden = join_request_outputs(positional,0);
        }
        auto mix = block.attention_gr.pre(execution,hidden);
        const bool project_batch=!speculative && batch>1 && block.gdn &&
            runtime_options::batch_attention_projections();
        const auto projections=project_batch ? block.gdn->project_inputs(execution,mix[0]) : std::vector<tb::Tensor>{};
        std::vector<tb::Tensor> attended;
        for (int64_t row = 0; row < batch; ++row) {
            auto& layer = requests[row]->state_->layers[i];
            const auto width = row_width(row);
            auto branch = mix[0].narrow(0,row,1).narrow(1,0,width);
            std::vector<tb::Tensor> projected;
            for(const auto& value:projections)projected.push_back(value.narrow(0,row,1));
            Tensor value;
            if (speculative && width > 1 && layer.gdn) {
                std::vector<Tensor> convolution, recurrent;
                for (int64_t token = 1; token < width; ++token) {
                    convolution.push_back(checkpoint(&Qwen4BatchCheckpointPool::gdn_conv_, i, row, token - 1,
                                                     *layer.gdn->graph_state()[0]));
                    recurrent.push_back(checkpoint(&Qwen4BatchCheckpointPool::gdn_state_, i, row, token - 1,
                                                   *layer.gdn->graph_state()[1]));
                }
                value = layer.gdn->forward_graph(execution,branch,1,convolution,recurrent);
                layer.gdn->bind_graph_checkpoint(convolution,recurrent,requests[row]->state_->verify_start + 1);
            } else value = layer.gdn
                ? layer.gdn->forward_chunk(execution,branch,true,std::move(projected),!project_batch)
                : layer.qsa->forward(execution,branch,current[row],full[row],true);
            attended.push_back(padded(std::move(value), width));
        }
        auto attention=join_request_outputs(attended,0);
        if(project_batch)attention=block.gdn->project_output(execution,attention);
        auto ffn_mix = block.ffn_gr.pre_after(execution,std::move(attention),mix);
        // Tiles share the expert transfer cache across all eligible rows.
        hidden = block.ffn_gr.post(speculative ? block.ffn(execution,ffn_mix[0])
            : block.batch_ffn(execution,ffn_mix[0]),ffn_mix);
    }
    if (raw) *raw = hidden;
    auto logits = model.logits_from_hidden(model.finalize_hidden(std::move(hidden),batch,tokens));
    logits = tokens == 1 ? logits.reshape({batch,-1}) : logits.reshape({batch,tokens,-1});
    for (int64_t row = 0; row < batch; ++row) {
        requests[row]->state_->positions = std::move(full[row]);
        requests[row]->state_->cache_position += row_width(row);
    }
    return logits;
}

void Qwen4Model::adapter_validate_load_options() const {
    weight_loader::validate_load_options(*execution);
}

void Qwen4Model::adapter_load_final_state(const mfq::ModelSource &source,
                                          mfq_tensor_backend::Tensor &output_norm) {
    qwen4_exp::BlockLoader loader{{*execution, source, "qwen4_exp"}};
    final_mixer = loader.final_mixer(
        mfq::models::qwen4_exp::load_residual(loader, config, "model.mhc.pre", false));
    output_norm = mfq_tensor_backend::Tensor();
}

std::unique_ptr<Block> Qwen4Model::adapter_load_block(const mfq::ModelSource &source, int layer,
                                                      int, const std::string &) {
    auto block=std::make_unique<qwen4_exp::Qwen4Block>(*execution, source, config, layer);
    loaded_blocks.push_back(block.get());
    return block;
}

void Qwen4Model::adapter_reset(int64_t new_batch) {
    speculative_prefix_retention=false;
    if(execution && execution->moe_expert_cache)finish_moe_expert_exchanges(execution->moe_expert_cache);
    const bool retain=reuse_decode_graphs && new_batch==1 && graph_registry && graph_registry->reusable();
    if(graph_registry)for(auto* block:graph_registry->blocks)block->retain_graph_storage=retain;
    if(!retain)graph_registry.reset();
    else for(auto& [key,cell]:graph_registry->cells)cell->used_kinds=0;
    positions = mfq_tensor_backend::Tensor();
    batch = new_batch;
}
void Qwen4Model::adapter_begin_forward(bool) {
    for(auto* block:loaded_blocks)block->retained_prefix=false;
    if(graph_registry) {
        if(reuse_decode_graphs && graph_registry->reusable()) {
            for(auto* attention:graph_registry->qsa)attention->leave_graph();
        } else {graph_registry->restore_legacy();graph_registry.reset();}
    }
}

mfq_tensor_backend::Tensor
Qwen4Model::adapter_block_positions(const mfq_tensor_backend::Tensor &full_positions,
                                    const mfq_tensor_backend::Tensor &, int device) const {
    return tensor_to_cuda_device(execution->model_parallel_collectives, full_positions, device);
}

void Qwen4Model::adapter_finish_forward(const mfq_tensor_backend::Tensor &full_positions,
                                        int64_t new_batch, int64_t tokens) {
    positions = full_positions;
    batch = new_batch;
    // Short eager verification retains the same prefix states as its graph.
    // Accepted drafts can be kept without rerunning them as scalar projections.
    speculative_prefix_retention=tokens<=8 && !loaded_blocks.empty() &&
        std::all_of(loaded_blocks.begin(),loaded_blocks.end(),[](const auto* block) {
            return block->retained_prefix;
        });
}

// Keep each compressed expert image resident while advancing all prompt chunks
// through that layer. Attention and PLE still advance in causal token order.
bool qwen4_layer_major_prefill_eligible(const Qwen4CausalLm& model, int64_t tokens) {
    const auto enabled=mfq::cuda::runtime_options::moe_prefill_layer_major();
    const auto whole=mfq::cuda::runtime_options::moe_prefill_layer();
    const bool tq=model.execution && bool(model.execution->tq_expert_cache);
    // TQ already groups ordinary bank rows by expert. Its layer-major path
    // does not require registration in the legacy MFE transfer pipeline.
    // Keep explicit user overrides and the existing legacy selection policy.
    if((enabled && *enabled==0) || (whole && *whole==0) || (!tq && !whole) || tokens<=128 ||
       tokens>16384 || model.speculative_start>=0 || !model.execution ||
       (!tq && !model.execution->config.moe_pipeline) || !model.execution->moe_expert_cache ||
       model.execution->dense_cpu_layer_count || model.execution->tensor_parallel.enabled() ||
       model.execution->expert_parallel.enabled())return false;
    const auto primary=model.execution->layer_placement.primary_device();
    for(const auto& block:model.blocks)
        if(block->cpu_offloaded || block->cuda_device!=primary)return false;
    return true;
}

mfq_tensor_backend::Tensor qwen4_layer_major_prefill_hidden(
        Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int64_t compute_chunk,
        bool last_only,const std::function<bool()>& cancelled,mfq_tensor_backend::Tensor* raw_hidden,
        bool raw_hidden_host) {
    return qwen4_layer_major_prefill_hidden(model,std::move(ids),compute_chunk,last_only,
        cancelled,raw_hidden,raw_hidden_host,{});
}

mfq_tensor_backend::Tensor qwen4_layer_major_prefill_hidden(
        Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int64_t compute_chunk,
        bool last_only,const std::function<bool()>& cancelled,mfq_tensor_backend::Tensor* raw_hidden,
        bool raw_hidden_host,const std::function<void()>& computation_finished) {
    namespace tb=mfq_tensor_backend;
    using Tensor=tb::Tensor;
    MFQ_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 &&
        qwen4_layer_major_prefill_eligible(model,ids.size(1)),
        "layer-major prefill requires a bounded single-device Qwen4 chunk and whole-layer cache");
    MFQ_RUNTIME_CHECK(compute_chunk>8,"layer-major compute chunk must exceed decode width");
    auto scope=model.execution_scope();
    ids=model.device_ids(std::move(ids));
    const auto tokens=ids.size(1);
    const auto cache_start=model.cache_pos;
    auto& profiler=model.execution->profiler;
    const auto* profile_once=std::getenv("MFQ_PROFILE_PREFILL_ONCE");
    static std::atomic<uint64_t> profile_requests{0};
    const bool profile_enabled=profile_once && profile_once[0]=='1';
    const auto* profile_skip=std::getenv("MFQ_PROFILE_PREFILL_SKIP");
    const auto skipped=profile_skip?std::max<long long>(0,std::atoll(profile_skip)):0;
    const auto profile_request=profile_enabled?profile_requests.fetch_add(1):0;
    const bool collect=profile_enabled && profile_request==uint64_t(skipped);
    if(profile_enabled)std::cerr<<"prefill_profile_request index="<<profile_request
        <<" tokens="<<tokens<<" collect="<<int(collect)<<'\n';
    const bool previous_profile=profiler.enabled;
    if(collect){profiler.reset();profiler.enabled=true;}
    struct ProfileRestore {
        CudaProfiler& profiler;bool enabled,previous,collect;
        ~ProfileRestore() {
            if(collect && std::uncaught_exceptions()==0) {
                try {profiler.report("prefill_regression_once");profiler.reset();}
                catch(const std::exception& error){std::cerr<<"prefill_profile_error "<<error.what()<<'\n';}
            }
            profiler.enabled=previous;
        }
    } restore{profiler,profiler.enabled,previous_profile,collect};
    compute_chunk=std::min(compute_chunk,tokens);
    MoePrefillWorkspaceLoan workspace_loan(model.execution->moe_expert_cache);
    bool borrowed_workspace=false;
    {
        // Persistent decode graphs and prefill now coexist. Bound the compute
        // tile by actual remaining workspace instead of discarding those graphs
        // or allocating an 8K projection tile on top of their live buffers.
        const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        const auto reserve=std::size_t(1)<<30;
        const auto row_bytes=std::size_t(model.config.hidden)*model.config.streams*sizeof(float)*8;
        // Whole-prompt residual and FFN injection remain live across tiles.
        // Their previous hidden input and the assembled FFN output can overlap
        // them at a layer boundary. Include that third sequence-sized buffer.
        const auto sequence_reserve=std::size_t(tokens)*model.config.hidden*model.config.streams*sizeof(float)*3;
        const auto budget=moe_prefill_workspace_budget(model.execution->moe_expert_cache,true);
        const auto planned_fixed=mfq::engine::prefill_budget_add(reserve+sequence_reserve,budget.transfer_bytes);
        const auto planned_available=mfq::engine::prefill_budget_add(
            context->available_memory()+context->workspace_available(),budget.lendable_bytes);
        compute_chunk=mfq::engine::select_prefill_chunk(tokens,compute_chunk,planned_available,planned_fixed,row_bytes);
        MFQ_RUNTIME_CHECK(compute_chunk>0,"insufficient CUDA prefill workspace before expert loan");
        borrowed_workspace=workspace_loan.begin(planned_fixed+std::size_t(compute_chunk)*row_bytes)>0;
        const auto available=context->available_memory()+context->workspace_available();
        const auto requested=compute_chunk;
        const auto fixed=reserve+sequence_reserve+moe_prefill_transfer_bytes(model.execution->moe_expert_cache);
        compute_chunk=mfq::engine::select_prefill_chunk(tokens,compute_chunk,available,fixed,row_bytes);
        MFQ_RUNTIME_CHECK(compute_chunk>0,"insufficient CUDA prefill workspace after expert loan");
        if(compute_chunk!=requested)std::cerr<<"runtime_prefill_chunk_budget requested="<<requested
            <<" compute="<<compute_chunk<<" available_bytes="<<available<<'\n';
    }
    mfq::cuda::ResidentAllocationScope persistent_preamble;
    model.adapter_validate_forward(1,tokens,cache_start,false,false,false);
    if(cache_start==0)model.reset(1);
    model.adapter_begin_forward(raw_hidden!=nullptr);
    auto current=tb::arange(cache_start,cache_start+tokens,ids.options());
    if(model.decode_position_delta)current=current+model.decode_position_delta;
    auto prepared=model.adapter_prepare_positions(std::move(current),1,tokens);
    model.adapter_validate_positions(prepared.positions,1,tokens,false);
    auto cpu_ids=ids.to(tb::kCPU,tb::kInt64).contiguous();
    persistent_preamble.finish();
    Tensor hidden=tb::empty({1,tokens,model.hidden_size()*model.hc_mult()},
                           ids.options().dtype(tb::kFloat16));
    for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
        if(cancelled && cancelled())return {};
        const auto count=std::min(compute_chunk,tokens-begin);
        auto input=model.adapter_prepare_hidden(model.embed_tokens(ids.narrow(1,begin,count)),1,count);
        MFQ_RUNTIME_CHECK(input.scalar_type()==hidden.scalar_type() &&
            input.size(-1)==hidden.size(-1),"layer-major embedding geometry changed");
        hidden.narrow(1,begin,count).copy_(input);
    }
    const auto trace_option=mfq::cuda::runtime_options::prefill_layer_major_trace();
    const bool trace=trace_option && *trace_option!=0;
    const auto batch_option=mfq::cuda::runtime_options::moe_prefill_expert_batch();
    const bool batch_experts=!batch_option || *batch_option!=0;
    const auto bulk_gr_option=mfq::cuda::runtime_options::gr_prefill_layer_batch();
    // Long prompts retain the established memory footprint unless explicitly
    // requested; the combined residual tensors have been verified to 16k.
    const bool bulk_gr=batch_experts && (bulk_gr_option?*bulk_gr_option:tokens<=16384);
    const auto bulk_gdn_option=mfq::cuda::runtime_options::gdn_prefill_layer_batch();
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    // Allow 8k chat framing while keeping the extra whole-sequence QKV and
    // gate tensors out of the 16k service path, which also retains MTP weights.
    const bool bulk_gdn=bulk_gr && (bulk_gdn_option?*bulk_gdn_option:tokens<=9216);
#else
    const bool bulk_gdn=bulk_gr && bulk_gdn_option && *bulk_gdn_option;
#endif
    const qwen4_exp::QwenPrefillReferenceRowsScope reference_rows(batch_experts?std::min<int64_t>(1024,compute_chunk):0);
    const auto batch_tokens_option=mfq::cuda::runtime_options::moe_prefill_expert_batch_tokens();
    const int64_t batch_tokens=batch_tokens_option?*batch_tokens_option:8192;
    const auto append_tail_option=mfq::cuda::runtime_options::moe_prefill_append_tail();
    const bool append_tail=append_tail_option && *append_tail_option;
    const auto expert_batch_count=[&](int64_t remaining) {
        // Chat framing can leave a few tokens beyond the batch boundary.
        // Absorb at most one reference chunk instead of running a second
        // almost-empty routed GEMM for every layer.
        return append_tail && remaining<=batch_tokens+1024?remaining:std::min(batch_tokens,remaining);
    };
    MFQ_RUNTIME_CHECK(!batch_experts || batch_tokens>=1024,"expert prefill batch must cover at least 1024 tokens");
    const auto audit_option=mfq::cuda::runtime_options::prefill_expert_batch_audit();
    const bool audit_batch=audit_option && *audit_option!=0;
    const auto started=std::chrono::steady_clock::now();
    int layer=0;
    for(auto& block:model.blocks) {
        const auto decode_reuse_option=mfq::cuda::runtime_options::nint_prefill_decode_reuse();
        NintPrefillDecodeCacheScope decode_reuse(!decode_reuse_option || *decode_reuse_option);
        Tensor next_hidden=hidden;
        Tensor ffn_inputs,ffn_injection;
        std::vector<Tensor> full_mix;
        Tensor full_branch;
        if(bulk_gr) {
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            Tensor source=hidden;
            if(qwen_block.ple) {
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    if(cancelled && cancelled())return {};
                    const auto count=std::min(compute_chunk,tokens-begin);
                    Block::Context context;
                    context.token_ids=ids.narrow(1,begin,count);
                    context.host_token_ids=cpu_ids.narrow(1,begin,count);
                    block->set_token_ids(context.token_ids);
                    block->prefetch_token_ids(context.host_token_ids);
                    auto input=qwen_block.prefill_ple(*model.execution,hidden.narrow(1,begin,count),context);
                    if(begin==0)source=tb::empty(hidden.sizes(),input.options());
                    source.narrow(1,begin,count).copy_(input);
                }
            }
            full_mix=profiler.measure("prefill.gr_attention",[&] {
                return qwen_block.attention_gr.pre(*model.execution,source);
            });
            if(bulk_gdn && qwen_block.gdn) {
                if(cancelled && cancelled())return {};
                const qwen4_exp::QwenGdnPrefillProjectionRowsScope projection_rows(compute_chunk);
                full_branch=profiler.measure("prefill.gdn",[&] {
                    return qwen_block.gdn->forward(*model.execution,full_mix[0],true,0);
                });
            }
        }
        for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
            if(cancelled && cancelled())return {};
            if(bulk_gdn && static_cast<qwen4_exp::Qwen4Block&>(*block).gdn)continue;
            const auto count=std::min(compute_chunk,tokens-begin);
            profiler.enabled=restore.enabled;
            Block::Context context;
            context.token_ids=ids.narrow(1,begin,count);
            context.host_token_ids=cpu_ids.narrow(1,begin,count);
            context.positions=prepared.positions.narrow(-1,begin,count);
            context.full_positions=prepared.full_positions.narrow(-1,0,cache_start+begin+count);
            context.cache_position=cache_start+begin;
            block->set_token_ids(context.token_ids);
            block->prefetch_token_ids(context.host_token_ids);
            auto input=hidden.narrow(1,begin,count);
            Tensor output;
            if(bulk_gr) {
                auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
                auto branch=qwen_block.prefill_attention_core(*model.execution,full_mix[0].narrow(1,begin,count),context);
                if(!full_branch.defined()) {
                    auto shape=branch.sizes().vec();shape[1]=tokens;
                    full_branch=tb::empty(shape,branch.options());
                }
                full_branch.narrow(1,begin,count).copy_(branch);
                continue;
            } else if(batch_experts) {
                auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
                auto mix=qwen_block.prefill_attention(*model.execution,input,context);
                if(!ffn_inputs.defined()) {
                    ffn_inputs=tb::empty({1,tokens,model.hidden_size()},mix[0].options().dtype(tb::kFloat16));
                    ffn_injection=tb::empty({1,tokens,mix[2].size(-1)},mix[2].options());
                }
                // qwen_moe casts the router/FFN source to half at this boundary.
                ffn_inputs.narrow(1,begin,count).copy_(mix[0].to(tb::kFloat16));
                ffn_injection.narrow(1,begin,count).copy_(mix[2]);
                output=std::move(mix[1]);
            } else output=block->forward_context(*model.execution,input,context,model.rope);
            // Retain the original rounding at every layer boundary. The next
            // chunk is independent of this input once its cache writes retire.
            MFQ_RUNTIME_CHECK(output.sizes()==input.sizes(),"layer-major decoder changed hidden shape");
            if(begin==0 && output.scalar_type()!=hidden.scalar_type()) {
                next_hidden=tb::empty(hidden.sizes(),output.options());
                if(trace)std::cout<<"prefill_layer_major_hidden layer="<<layer
                    <<" input_bytes="<<hidden.nbytes()<<" output_bytes="<<next_hidden.nbytes()<<std::endl;
            }
            MFQ_RUNTIME_CHECK(output.scalar_type()==next_hidden.scalar_type(),
                "layer-major decoder output dtype changed within a layer");
            next_hidden.narrow(1,begin,count).copy_(output);
        }
        // Attention's decoded matrices are no longer needed by FFN. Keep
        // reuse within each phase, without retaining both sets at peak memory.
        if(tokens>9216)decode_reuse.clear();
        if(bulk_gr) {
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            auto mix=profiler.measure("prefill.gr_ffn_pre",[&] {
                return qwen_block.ffn_gr.pre_after(*model.execution,full_branch,full_mix);
            });
            ffn_inputs=mix[0].to(tb::kFloat16);ffn_injection=std::move(mix[2]);
            next_hidden=std::move(mix[1]);
            full_mix.clear();full_branch={};
        }
        if(tokens>9216)decode_reuse.clear();
        // The new residual owns the next layer's values. Prior hidden readers
        // have already been submitted before this stream-ordered release.
        if(tokens>9216)hidden={};
        if(batch_experts) {
            if(cancelled && cancelled())return {};
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            profiler.enabled=restore.enabled;
            auto branch=profiler.measure("prefill.ffn",[&] {
                if(expert_batch_count(tokens)==tokens)return qwen_block.ffn(*model.execution,ffn_inputs);
                Tensor result;
                for(int64_t begin=0;begin<tokens;) {
                    if(cancelled && cancelled())return Tensor{};
                    auto count=expert_batch_count(tokens-begin);
                    const auto tail=tokens-begin-count;
                    // A tiny final batch would enter the decode FFN path and
                    // release the shared prefill images halfway through every
                    // layer. Keep both batches in prefill, including chat inputs
                    // just above the expert-batch boundary.
                    if(tail>0 && tail<=8)count-=16-tail;
                    auto part=qwen_block.ffn(*model.execution,ffn_inputs.narrow(1,begin,count));
                    if(!result.defined()) {
                        auto shape=part.sizes().vec();shape[1]=tokens;
                        result=tb::empty(shape,part.options());
                    }
                    result.narrow(1,begin,count).copy_(part);
                    begin+=count;
                }
                return result;
            });
            if(!branch.defined())return {};
            if(audit_batch) {
                auto expected=tb::empty(branch.sizes(),branch.options());
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    const auto count=std::min(compute_chunk,tokens-begin);
                    expected.narrow(1,begin,count).copy_(
                        qwen_block.ffn(*model.execution,ffn_inputs.narrow(1,begin,count)));
                }
                auto a=branch.to(tb::kCPU,tb::kFloat32).contiguous();
                auto b=expected.to(tb::kCPU,tb::kFloat32).contiguous();
                double error=0,scale=0,maximum=0;
                for(int64_t i=0;i<a.numel();++i) {
                    const double delta=double(a.data_ptr<float>()[i])-b.data_ptr<float>()[i];
                    error+=delta*delta;scale+=double(b.data_ptr<float>()[i])*b.data_ptr<float>()[i];
                    maximum=std::max(maximum,std::abs(delta));
                }
                std::cout<<"prefill_expert_batch_layer_oracle layer="<<layer<<" tokens="<<tokens
                    <<" relative_rms="<<std::sqrt(error/std::max(1e-30,scale))
                    <<" maximum_absolute="<<maximum<<std::endl;
                // Diagnostic only: continue with the reference so later
                // layers are tested with their original input distribution.
                branch=std::move(expected);
            }
            Tensor post_hidden=next_hidden;
            for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                const auto count=std::min(compute_chunk,tokens-begin);
                std::vector<Tensor> mix{{},next_hidden.narrow(1,begin,count),ffn_injection.narrow(1,begin,count)};
                auto output=qwen_block.ffn_gr.post(branch.narrow(1,begin,count),mix);
                if(begin==0 && output.scalar_type()!=next_hidden.scalar_type())
                    post_hidden=tb::empty(next_hidden.sizes(),output.options());
                MFQ_RUNTIME_CHECK(output.scalar_type()==post_hidden.scalar_type(),
                    "expert-batched residual dtype changed");
                post_hidden.narrow(1,begin,count).copy_(output);
            }
            next_hidden=std::move(post_hidden);
        }
        hidden=std::move(next_hidden);
        ++layer;
        if(trace && (layer%4==0 || layer==int(model.blocks.size()))) {
            MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
#ifdef MFQ_NATIVE_CUDA_RUNTIME
            const auto memory=mfq::cuda::default_context(ids.get_device())->memory_stats();
            const auto local=mfq::cuda::default_context(ids.get_device())->local_memory_usage();
            cudaMemPool_t pool=nullptr;uint64_t reserved=0,used=0;
            MFQ_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,ids.get_device()));
            MFQ_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&reserved));
            MFQ_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemCurrent,&used));
            std::cout<<"prefill_layer_major_memory tokens="<<tokens<<" chunk="<<compute_chunk
                <<" completed_layers="<<layer<<" owned_bytes="<<memory.allocated
                <<" process_local_bytes="<<local<<" pool_used_bytes="<<used<<" pool_reserved_bytes="<<reserved<<std::endl;
#endif
            std::cout<<"prefill_layer_major_progress tokens="<<tokens<<" chunk="<<compute_chunk
                <<" completed_layers="<<layer<<" total_layers="<<model.blocks.size()
                <<" elapsed_ms="<<std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-started).count()<<std::endl;
        }
    }
    profiler.enabled=restore.enabled;
    if(last_only && !raw_hidden) {
        const auto begin=((tokens-1)/compute_chunk)*compute_chunk;
        Tensor result;
        {
            auto finalized=model.finalize_hidden(hidden.narrow(1,begin,tokens-begin),1,tokens-begin);
            result=model.last_hidden(std::move(finalized)).unsqueeze(1);
        }
        if(computation_finished)computation_finished();
        if(borrowed_workspace) {
            workspace_loan.suspend();
            result=result.clone();
            hidden={};
            workspace_loan.finish();
        }
        model.adapter_finish_forward(prepared.full_positions,1,tokens);
        model.cache_pos=cache_start+tokens;
        return result;
    }
    Tensor finalized;
    if(last_only) {
        // Use the same final GR dispatch and rounding as chunk-major prefill.
        const auto begin=((tokens-1)/compute_chunk)*compute_chunk;
        finalized=model.finalize_hidden(hidden.narrow(1,begin,tokens-begin),1,tokens-begin);
        finalized=model.last_hidden(std::move(finalized)).unsqueeze(1);
    } else {
        for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
            const auto count=std::min(compute_chunk,tokens-begin);
            auto part=model.finalize_hidden(hidden.narrow(1,begin,count),1,count);
            if(!finalized.defined())finalized=tb::empty({1,tokens,part.size(-1)},part.options());
            finalized.narrow(1,begin,count).copy_(part);
        }
    }
    if(raw_hidden && raw_hidden_host)hidden=hidden.to(tb::kCPU);
    if(computation_finished)computation_finished();
    if(borrowed_workspace) {
        // The final mixer also needs sequence-sized scratch. Finish it inside
        // the loan, then retain only the outputs consumed by MTP/the caller.
        workspace_loan.suspend();
        if(raw_hidden) {if(!raw_hidden_host)hidden=hidden.clone();}else hidden={};
        finalized=finalized.clone();
        workspace_loan.finish();
    }
    model.adapter_finish_forward(prepared.full_positions,1,tokens);
    model.cache_pos=cache_start+tokens;
    if(raw_hidden)*raw_hidden=hidden;
    return finalized;
}

mfq_tensor_backend::Tensor Qwen4Model::adapter_finalize_hidden(mfq_tensor_backend::Tensor hidden,
                                                               const mfq_tensor_backend::Tensor &,
                                                               int64_t, int64_t) const {
    auto* window=DecodeWindow::recording();
    GatedResidualMarker marker;
    if(window && runtime_options::mhc_timings())marker=[window](const char* phase) {
        window->mark(-1,(std::string("final_mhc_")+phase).c_str());
    };
    return final_mixer->pre(*execution, hidden,marker)[0];
}

mfq_tensor_backend::Tensor
Qwen4Model::adapter_last_logits(const QuantLinear &lm_head,
                                mfq_tensor_backend::Tensor hidden) const {
    return adapter_logits(lm_head, std::move(hidden));
}

mfq_tensor_backend::Tensor Qwen4Model::adapter_next_token(const QuantLinear &lm_head,
                                                          mfq_tensor_backend::Tensor hidden) const {
    auto* window=DecodeWindow::recording();
    if(window)window->mark(-1,"lm_head_begin");
    auto logits=adapter_logits(lm_head,std::move(hidden));
    if(window)window->mark(-1,"lm_head_projection_end");
    auto output=mfq_tensor_backend::argmax(logits,-1).to(mfq_tensor_backend::kInt64);
    if(window)window->mark(-1,"lm_head_sample_end");
    return output;
}

void Qwen4Model::adapter_rollback_speculative(int64_t keep) {
    speculative_prefix_retention=false;
    if (positions.defined())
        positions = positions.narrow(-1, 0, keep);
}

} // namespace mfq::cuda

namespace mfq::cuda {

TextSessionStateKind CudaSessionCodec<Qwen4Model>::kind(const Qwen4CausalLm& model) {
    return model.execution && !model.execution->qsa_kv_store && !model.blocks.empty()
        ? TextSessionStateKind::Qwen4 : TextSessionStateKind::Unsupported;
}
bool CudaSessionCodec<Qwen4Model>::supports_paged(const Qwen4CausalLm& model) { return kind(model)==TextSessionStateKind::Qwen4; }

TextSessionState CudaSessionCodec<Qwen4Model>::capture(const Qwen4CausalLm& model,const std::vector<int64_t>& tokens) {
    MFQ_RUNTIME_CHECK(kind(model)==TextSessionStateKind::Qwen4 && model.batch==1 && model.cache_pos>0 &&
        std::size_t(model.cache_pos)==tokens.size() && model.speculative_start<0,
        "Qwen4 text session boundary is unavailable");
    TextSessionState state;state.tokens=tokens;state.cache_pos=model.cache_pos;
    std::vector<Qwen4BlockSessionState> blocks;
    for(const auto& block:model.blocks)blocks.push_back(qwen4_exp::capture_block_session(
        static_cast<const qwen4_exp::Qwen4Block&>(*block),model.cache_pos,state.bytes));
    state.payload=std::move(blocks);
    state.qwen4_positions=model.positions.clone();state.bytes+=session_tensor_bytes(state.qwen4_positions);
    return state;
}

void CudaSessionCodec<Qwen4Model>::restore(Qwen4CausalLm& model,const TextSessionState& state) {
    const auto* blocks=std::get_if<std::vector<Qwen4BlockSessionState>>(&state.payload);
    MFQ_RUNTIME_CHECK(kind(model)==TextSessionStateKind::Qwen4 && blocks && blocks->size()==model.blocks.size() &&
        state.cache_pos>0 && std::size_t(state.cache_pos)==state.tokens.size() &&
        state.qwen4_positions.defined() && state.qwen4_positions.size(-1)==state.cache_pos,
        "incompatible Qwen4 text session state");
    model.reset(1);
    for(std::size_t i=0;i<blocks->size();++i)qwen4_exp::restore_block_session(
        static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]),(*blocks)[i],state.cache_pos);
    model.positions=state.qwen4_positions.clone();model.cache_pos=state.cache_pos;
}

} // namespace mfq::cuda

namespace mfq::models {
template struct qwen4_exp::CausalLm<cuda::CudaCausalOps<cuda::Qwen4Model>>;
} // namespace mfq::models
