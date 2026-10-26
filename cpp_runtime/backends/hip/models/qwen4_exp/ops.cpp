#include "hip/hip_runtime.h"
#include "../../runtime/execution_options.h"
#include "mfq_hip_moe_ops.h"
#include "mfq_hip_sampling_ops.h"
#include "ops.h"
#include "model.h"
#include "mtp.h"
#include "storage/weight_loader.h"
#include "storage/moe_ffn_pipeline.h"
#include "storage/moe_expert_cache.h"
#include "runtime/nint_row_pipeline.h"
#include "runtime/decode_window.h"
#include "runtime/moe_pipeline.h"
#include "runtime/router_lookahead_audit.h"
#include <fstream>
#include <iostream>
#include "models/common/gated_mlp.h"
#include "models/common/moe.h"
#include "models/common/transformer_layer.h"
#include "storage/session_codec.h"
#include "gated_residual.h"
#include "gated_residual_fused.h"
#include "float_projection.h"
#include <cstdlib>
#include <cstdio>
#include <tuple>
#include <mutex>
#include <map>

namespace mfq::cuda::qwen4_exp {
using Config = mfq::models::qwen4_exp::Config;

struct Gr {
    Tensor norm;
    weight_loader::ResidualLinear down, up, injection;
    int64_t hidden, streams;
    double eps;
    bool can_fuse_tq(CudaExecutionContext& execution,const Tensor& x,const GatedResidualMarker& marker)const {
        return execution.config.gr_two_stage && execution.config.gr_fused_projections &&
            execution.config.gr_fused_projection_activation && streams>0 && streams<=4 &&
            !marker && down.weight && up.weight && injection.weight &&
            down.weight->is_tq() && up.weight->is_tq() && injection.weight->is_dense() &&
            down.weight->tq.out==320 && down.weight->tq.neuron_len==10240 &&
            injection.weight->dense.is_contiguous() && injection.weight->dense.scalar_type()==tb::kBFloat16 &&
            injection.weight->dense.sizes().vec()==std::vector<int64_t>({streams,hidden*streams}) &&
            x.is_cuda() && x.is_contiguous() && x.scalar_type()==tb::kFloat32 && x.numel()==hidden*streams;
    }
    std::vector<Tensor> fused_tq(const Tensor& input,const Tensor& normalized)const {
        auto projected=tq_gr_down_injection_cuda(down.weight->tq.view,injection.weight->dense,normalized,streams);
        return {tq_gr_mix(up.weight->tq,projected[0],normalized,streams),input,projected[1]};
    }
    bool can_fuse(CudaExecutionContext& execution,const Tensor& x,const GatedResidualMarker& marker)const {
        if(!execution.config.gr_two_stage || marker || !execution.config.gr_fused_projections ||
            !execution.config.gr_native_projection_input || !down.weight || !up.weight || !injection.weight ||
            !down.weight->is_nint() || !up.weight->is_nint() || streams<1 || streams>4 ||
            x.dim()<1 || x.numel()!=hidden*streams || !x.is_cuda() || !x.is_contiguous() ||
            (x.scalar_type()!=tb::kFloat16 && x.scalar_type()!=tb::kFloat32) ||
            !norm.is_contiguous() || norm.scalar_type()!=tb::kFloat32)return false;
        if(!injection.weight->is_nint() && !(execution.config.gr_two_stage_dense_injection &&
            injection.weight->is_dense() && injection.weight->dense.is_contiguous() &&
            injection.weight->dense.scalar_type()==tb::kBFloat16))return false;
        const auto& w=up.weight->nint;
        return w.ng==1 || (w.out+3)/4>=nint_float_projection_resident_blocks(x.get_device());
    }
    std::vector<Tensor> fused(const Tensor& branch,const Tensor& residual,const Tensor& prior,bool after)const {
        const auto& projection=*injection.weight;
        return gated_residual_two_stage_cuda(branch,residual,prior,norm,down.weight->nint,up.weight->nint,
            projection.is_nint()?&projection.nint:nullptr,streams,eps,after,
            projection.is_dense()?projection.dense:Tensor{},injection.prepared_right);
    }
    std::vector<Tensor> pre(CudaExecutionContext &execution, const Tensor &x,
        const GatedResidualMarker& marker={}) const {
        if(can_fuse(execution,x,marker))return fused({},x,{},false);
        if(can_fuse_tq(execution,x,marker))return fused_tq(x,
            mfq_qwen4_exp::grouped_rms_norm(x,norm,hidden,eps));
        return gated_residual_pre_projected(execution, x, norm, down.forward, up.forward, injection.forward,
            hidden, streams, eps,up.mixed,marker,down.activated,injection.activated);
    }
    Tensor post(const Tensor &branch, const std::vector<Tensor> &inputs) const {
        return mfq_qwen4_exp::gated_residual_post(branch, inputs[1], inputs[2], streams);
    }
    std::vector<Tensor> pre_after(CudaExecutionContext& execution,const Tensor& branch,
        const std::vector<Tensor>& inputs,const GatedResidualMarker& marker={}) const {
        if(can_fuse_tq(execution,inputs[1],marker) && execution.config.gr_fused_post_norm &&
           branch.is_contiguous() && inputs[2].is_contiguous() &&
           (branch.scalar_type()==tb::kFloat16 || branch.scalar_type()==tb::kFloat32) &&
           inputs[2].scalar_type()==tb::kFloat32) {
            auto prepared=mfq_qwen4_exp::gated_residual_post_norm(branch,inputs[1],inputs[2],norm,streams,eps);
            return fused_tq(prepared[0],prepared[1]);
        }
        if(can_fuse(execution,inputs[1],marker) && branch.is_contiguous() && inputs[2].is_contiguous() &&
            (branch.scalar_type()==tb::kFloat16 || branch.scalar_type()==tb::kFloat32) &&
            (inputs[2].scalar_type()==tb::kFloat16 || inputs[2].scalar_type()==tb::kFloat32))
            return fused(branch,inputs[1],inputs[2],true);
        return gated_residual_pre_after_projected(execution,branch,inputs[1],inputs[2],norm,
            down.forward,up.forward,injection.forward,hidden,streams,eps,
            up.mixed,marker,down.activated,injection.activated);
    }
};

static Linear qwen_moe(weight_loader::Routed gate_up, weight_loader::Routed down,
                       Linear router, Linear shared_gate, Linear sg, Linear su, Linear sd,
                       const Config &c, MoeFfnPrefetch* prefetch=nullptr) {
    auto sg_weight=weight_loader::linear_weight(sg),su_weight=weight_loader::linear_weight(su);
    auto sd_weight=weight_loader::linear_weight(sd);
    auto router_weight=weight_loader::linear_weight(router),shared_gate_weight=weight_loader::linear_weight(shared_gate);
    auto shared=[sg,su,sd,shared_gate,sg_weight,su_weight](CudaExecutionContext& execution,const Tensor& source) {
        if(sg_weight && su_weight && sg_weight->is_tq() && su_weight->is_tq() &&
           source.is_cuda() && source.is_contiguous() && source.scalar_type()==tb::kFloat16 &&
           source.numel()/source.size(-1)>=1 && source.numel()/source.size(-1)<=6 && !tq_q8_enabled()) {
            auto hidden=tq_shared_swiglu_cuda(sg_weight->tq.view,su_weight->tq.view,source);
            return tb::sigmoid(shared_gate(execution,source))*sd(execution,hidden);
        }
        auto unfused=[](const auto&...){return std::optional<Tensor>{};};
        auto output=mfq::models::gated_mlp(source,false,0.0,unfused,unfused,
            [&](Tensor input){return std::array<Tensor,2>{sg(execution,input),su(execution,input)};},
            [](Tensor g,Tensor u,mfq::models::GatedActivation,double){return (g*tb::sigmoid(g))*u;},
            [&](Tensor hidden){return sd(execution,hidden);},unfused);
        return tb::sigmoid(shared_gate(execution,source))*output;
    };
    MoeFfnSharedWeights shared_weights;
    shared_weights.projections={weight_loader::linear_weight(sg),weight_loader::linear_weight(su),weight_loader::linear_weight(sd)};
    shared_weights.gate=[shared_gate](CudaExecutionContext& execution,const Tensor& input){return tb::sigmoid(shared_gate(execution,input));};
    std::shared_ptr<TqMoeBanks> tq;
    if(!gate_up.tq_projections.empty() || !down.tq_projections.empty()) {
        MFQ_RUNTIME_CHECK(gate_up.tq_projections.size()==2 && down.tq_projections.size()==1 &&
            gate_up.projections.empty() && down.projections.empty(),"TQ MoE requires separate Gate, Up and Down banks");
        tq=std::make_shared<TqMoeBanks>();tq->gate=gate_up.tq_projections[0];
        tq->up=gate_up.tq_projections[1];tq->down=down.tq_projections[0];
    }
    auto pipeline=tq?MoeFfnForward{}:make_moe_ffn_pipeline(gate_up.projections,down.projections,shared,{},std::move(shared_weights),prefetch);
    return [gate_up, down, router, shared, pipeline, tq, c,router_weight,shared_gate_weight,sg_weight,su_weight,sd,sd_weight,shared_gate](CudaExecutionContext &execution,
                                                               const Tensor &x) {
        auto source = x.reshape({-1, c.hidden}).to(tb::kFloat16);
        const auto routing = c.routing();
        if(tq && router_weight && router_weight->is_tq() && shared_gate_weight && shared_gate_weight->is_dense() &&
           shared_gate_weight->dense.scalar_type()==tb::kBFloat16 && shared_gate_weight->dense.is_contiguous() &&
           shared_gate_weight->dense.sizes().vec()==std::vector<int64_t>({1,c.hidden}) &&
           sg_weight && su_weight && sg_weight->is_tq() && su_weight->is_tq() &&
           source.numel()==c.hidden && c.hidden==2560 && !tq_q8_enabled()) {
            auto projected=tq_moe_inputs_cuda(router_weight->tq.view,shared_gate_weight->dense,
                sg_weight->tq.view,su_weight->tq.view,source);
            auto selected=moe_topk_cuda(projected[0].contiguous(),c.topk,
                routing.activation==mfq::models::RouterActivation::sigmoid,
                routing.activation==mfq::models::RouterActivation::sqrt_softplus,
                routing.normalize,routing.delayed_softmax,mfq_nullopt,1e-20,routing.scale);
            auto hidden=projected[2];
            if(sd_weight && sd_weight->is_tq() && tq->gate->shards.empty() &&
               sd_weight->tq.out>=sd_weight->tq.neuron_len && sd_weight->tq.neuron_len%8==0)
                return tq->forward(source,selected[0],selected[1],0.f,&sd_weight->tq,hidden,projected[1]).reshape(x.sizes());
            auto shared_output=sd(execution,hidden);
            auto routed=tq->forward(source,selected[0],selected[1]);
            return tq_shared_add_cuda(routed,shared_output,projected[1]).reshape(x.sizes());
        }
        // fused TQ concurrent ending: preserve the native shared coefficient dtype.
        if(tq && tq->gate->shards.empty() && c.hidden==2560 && source.size(0)>=2 && source.size(0)<=6 &&
           router_weight && router_weight->is_tq() && shared_gate_weight && shared_gate_weight->is_dense() &&
           shared_gate_weight->dense.scalar_type()==tb::kBFloat16 &&
           sg_weight && su_weight && sd_weight && sg_weight->is_tq() && su_weight->is_tq() && sd_weight->is_tq() &&
           sd_weight->tq.out>=sd_weight->tq.neuron_len && sd_weight->tq.neuron_len%8==0 && !tq_q8_enabled()) {
            auto logits=router(execution,source).contiguous();
            auto selected=moe_topk_cuda(logits,c.topk,
                routing.activation==mfq::models::RouterActivation::sigmoid,
                routing.activation==mfq::models::RouterActivation::sqrt_softplus,
                routing.normalize,routing.delayed_softmax,mfq_nullopt,1e-20,routing.scale);
            auto coefficient=tb::sigmoid(shared_gate(execution,source)).contiguous();
            auto hidden=tq_shared_swiglu_cuda(sg_weight->tq.view,su_weight->tq.view,source);
            if(coefficient.scalar_type()==tb::kBFloat16 || coefficient.scalar_type()==tb::kFloat32)
                return tq->forward(source,selected[0],selected[1],0.f,&sd_weight->tq,hidden,coefficient).reshape(x.sizes());
            return (tq->forward(source,selected[0],selected[1])+coefficient*sd(execution,hidden)).reshape(x.sizes());
        }
        return mfq::models::mixture_of_experts(
            [&] { return router(execution, source).contiguous(); },
            [&](Tensor logits) {
                return moe_topk_cuda(
                    logits, c.topk, routing.activation == mfq::models::RouterActivation::sigmoid,
                    routing.activation == mfq::models::RouterActivation::sqrt_softplus,
                    routing.normalize, routing.delayed_softmax, mfq_nullopt, 1e-20, routing.scale);
            },
            [&](const auto &) {
                return pipeline ? Tensor{} : shared(execution,source);
            },
            [&](const auto &selected) {
                if(pipeline)return pipeline(execution,source,selected[0].to(tb::kInt32).contiguous(),selected[1]);
                if(tq)return tq->forward(source,selected[0],selected[1]);
                return mfq::models::routed_experts(
                    [&] { return gate_up(execution, source, selected[0]); },
                    [&](Tensor gu) {
                        return moe_dense_swiglu_hip(gu.contiguous());
                    },
                    [&](Tensor hidden) { return down(execution, hidden, selected[0]); },
                    [&](Tensor pairs) {
                        return moe_dense_weighted_reduce_hip(pairs.contiguous(),selected[1].contiguous());
                    });
            },
            [&](Tensor routed, Tensor shared_output, const auto &) {
                if(pipeline)return routed.reshape(x.sizes());
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
    bool* resident_tq_capture=nullptr;
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
        Linear qkv;
        if(has_tensor(source,p+".qkv.weight"))qkv=linear(p+".qkv.weight");
        else {
            std::vector<Linear> projections{linear(p+".query.weight"),linear(p+".key.weight"),linear(p+".value.weight")};
            auto group=weight_loader::grouped_linear(execution,projections);
            qkv=[projections=std::move(projections),group](CudaExecutionContext& execution,const Tensor& x) {
                if(group)return tb::cat(group(execution,x),-1);
                return tb::cat({projections[0](execution,x),projections[1](execution,x),projections[2](execution,x)},-1);
            };
        }
        return GdnWeights{
            std::move(qkv), linear(p + ".gate.weight"),
            linear(p + ".alpha.weight"), linear(p + ".beta.weight"),
            linear(p + ".output.weight"), dense(p + ".conv.weight"),
            dense(p + ".dt_bias"), dense(p + ".a"), dense(p + ".norm.weight")};
    }
    auto qsa_weights(const std::string &p) const {
        auto result=QsaWeights{
            linear(p + ".query.weight"), linear(p + ".key.weight"),
            linear(p + ".value.weight"), linear(p + ".output.weight"),
            linear(p + ".indexer.query_key.weight"), dense(p + ".query_norm.weight"),
            dense(p + ".key_norm.weight"), dense(p + ".indexer.query_norm.weight"),
            dense(p + ".indexer.key_norm.weight")};
        if(has_tensor(source,p+".gate.weight"))result.gate=linear(p+".gate.weight");
        return result;
    }
    auto gdn(GdnWeights weights, const Config &c) const {
        std::vector<Linear> projections{weights.qkv,weights.gate,weights.alpha,weights.beta};
        weights.input_projection=weight_loader::grouped_linear(execution,projections);
        auto q=weight_loader::linear_weight(projections[0]),g=weight_loader::linear_weight(projections[1]);
        auto a=weight_loader::linear_weight(projections[2]),b=weight_loader::linear_weight(projections[3]);
        const auto tq_input=[&](const auto& w){return w && w->is_tq() && !w->tensor_parallel() &&
            w->tq.image.is_cuda() && w->out()>=w->neuron_len();};
        const auto dense_gate=[&](const auto& w){return w && w->is_dense() && !w->tensor_parallel() &&
            w->dense.is_cuda() && w->dense.is_contiguous() && w->dense.scalar_type()==tb::kBFloat16 &&
            w->dense.dim()==2 && w->out()>0 && w->out()<=64 && w->neuron_len()==c.hidden;};
        if(weights.input_projection && tq_input(q) && tq_input(g) && dense_gate(a) && dense_gate(b)) {
            // GDN promotes these BF16 auxiliaries to FP32 before its core.
            // Keep the generic grouped-linear output dtype contract intact.
            weights.input_projection=[previous=weights.input_projection,q,g,a,b](CudaExecutionContext& ex,const Tensor& x) {
                if(!tq_q8_enabled() && x.is_cuda() && x.scalar_type()==tb::kFloat32 && x.size(1)==1 &&
                        x.numel()/x.size(-1)>=1 && x.numel()/x.size(-1)<=6)
                    return tq_forward_group_bf16({&q->tq,&g->tq},x,a->dense,b->dense);
                return previous(ex,x);
            };
        }
        weights.qkv=std::move(projections[0]);weights.gate=std::move(projections[1]);
        weights.alpha=std::move(projections[2]);weights.beta=std::move(projections[3]);
        return std::make_unique<Gdn>(std::move(weights), c.key_heads, c.value_heads, c.linear_width,
                                     c.kernel, c.eps, c.silu_gate,true,execution.config.gdn_transposed_state,
                                     execution.config.gdn_fused_output,true,execution.config.gdn_fused_preparation,
                                     execution.config.gdn_fused_core);
    }
    auto qsa(QsaWeights weights, const Config &c) const {
        if(weights.gate) {
            std::vector<Linear> separate{weights.query,weights.gate};
            auto group=weight_loader::grouped_linear(execution,separate);
            weights.query=[separate=std::move(separate),group,heads=c.heads,width=c.width](CudaExecutionContext& execution,const Tensor& x) {
                auto values=group?group(execution,x):std::vector<Tensor>{separate[0](execution,x),separate[1](execution,x)};
                auto shape=x.sizes().vec();shape.back()=heads*width*2;
                return tb::cat({values[0].reshape({-1,heads,width}),values[1].reshape({-1,heads,width})},-1).reshape(shape);
            };
        }
        std::vector<Linear> projections{weights.query,weights.key,weights.value,weights.index_query_key};
        weights.input_projection=weight_loader::grouped_linear(execution,projections);
        weights.query=std::move(projections[0]);weights.key=std::move(projections[1]);
        weights.value=std::move(projections[2]);weights.index_query_key=std::move(projections[3]);
        QsaConfig geometry{c.heads, c.kv_heads, c.width, c.index_heads, c.index_width,
                           c.pool, c.budget, c.maximum, c.eps};
        auto rotary = std::make_shared<RotaryEmbedding>(c.rotary, c.maximum, c.rope_base, c.sections, c.interleaved);
        return std::make_unique<Qsa>(std::move(weights), geometry, std::move(rotary));
    }
    auto moe(weight_loader::Routed gate_up, weight_loader::Routed down, Linear router,
                    Linear shared_gate, Linear sg, Linear su, Linear sd, const Config &c) const {
        if(router_capture)*router_capture=router;
        if(resident_tq_capture)*resident_tq_capture=gate_up.tq_projections.size()==2 && down.tq_projections.size()==1;
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
        NgramEmbedding::RowLookup lookup;NgramEmbedding::StageFactory stage;
        if(std::all_of(tq.begin(),tq.end(),[](const auto& table){return bool(table);})) {
            lookup=[tq,rows](const auto& ids,const auto& device){return lookup_tq_shards(tq,rows,ids,device);};
            stage=[tq,rows](const auto& shape,const auto& device){return make_tq_row_stage(tq,rows,shape,device);};
        }
        NgramEmbedding embedding(std::move(functions), rows, width, c.ngram, c.ngram_heads, c.eos,
                                 std::move(multipliers), std::move(offsets), std::move(sizes),std::move(pipeline),std::move(lookup),std::move(stage));
        return std::make_unique<Ple>(std::move(embedding), std::move(weights), c.hidden, c.streams,
                                     c.ngram, c.eps);
    }
    std::unique_ptr<Qwen4Block> block(const Config &c, int layer, bool predictor);
};

struct Qwen4Block final : Block {
    using Tensor = mfq_tensor_backend::Tensor;
    mfq::models::qwen4_exp::Config config;
    Gr attention_gr, ffn_gr;
    Linear ffn,router;
    Linear next_router;
    MoeFfnPrefetch prefetch,next_prefetch;
    bool resident_tq=false,retain_graph_storage=false;
    std::shared_ptr<RouterLookaheadInputs> router_audit;
    std::unique_ptr<Gdn> gdn;
    std::unique_ptr<Qsa> qsa;
    std::unique_ptr<Ple> ple;
    Qwen4Block(CudaExecutionContext &execution, const mfq::ModelSource &file,
               const Config &c, int layer, bool predictor = false) {
        BlockLoader loader{{execution, file, "qwen4_exp"},&router,&prefetch,&resident_tq};
        mfq::models::qwen4_exp::load_block(*this, loader, c, layer, predictor);
    }
    void reset(int64_t) override {
        if (gdn)
            gdn->reset(retain_graph_storage);
        if (qsa)
            qsa->reset(retain_graph_storage);
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
            gdn->rollback();
        if (qsa)
            qsa->truncate(keep);
        if (ple)
            ple->rollback();
    }
    Tensor execute(CudaExecutionContext &execution, Tensor x, const Tensor &ids,
                   const Tensor &positions, const Tensor &full_positions, int64_t confirmed = 0,const Tensor& host_ids = {}) {
        return mfq::models::qwen4_exp::decoder_layer_chained(
            std::move(x), bool(ple), bool(gdn),
            [&](const Tensor &hidden) {
                return execution.profiler.measure("qwen4.ple",[&]{return ple->forward(execution,hidden,ids,true,confirmed,host_ids);});
            },
            [](Tensor hidden, Tensor positional) { return hidden + positional; },
            [&](const Tensor &hidden) { return execution.profiler.measure("qwen4.attention_gr",[&]{return attention_gr.pre(execution,hidden);}); },
            [&](Tensor branch) { return execution.profiler.measure("qwen4.gdn",[&]{return gdn->forward(execution,branch,true,confirmed);}); },
            [&](Tensor branch) {
                return execution.profiler.measure("qwen4.qsa",[&]{return qsa->forward(execution,branch,positions,full_positions,true);});
            },
            [&](Tensor branch, const auto &mix) { return execution.profiler.measure("qwen4.ffn_gr",[&]{return ffn_gr.pre_after(execution,branch,mix);}); },
            [&](Tensor branch) { return execution.profiler.measure("qwen4.moe",[&]{return ffn(execution,branch);}); },
            [&](Tensor branch, const auto &mix) { return execution.profiler.measure("qwen4.post",[&]{return ffn_gr.post(branch,mix);}); });
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
    std::vector<Tensor*> graph_warmup_state() override {
        std::vector<Tensor*> result;
        if(gdn)for(auto* value:gdn->graph_state())if(value->defined())result.push_back(value);
        if(ple)for(auto* value:ple->graph_state())if(value->defined())result.push_back(value);
        return result;
    }
    std::vector<const Tensor*> graph_bindings() const {
        std::vector<const Tensor*> result;
        if(gdn)for(auto* value:gdn->graph_bindings())if(value->defined())result.push_back(value);
        if(ple)for(auto* value:ple->graph_bindings())if(value->defined())result.push_back(value);
        if(qsa)for(auto* value:qsa->graph_bindings())if(value->defined())result.push_back(value);
        return result;
    }
    Tensor execute_graph(CudaExecutionContext& execution,Tensor x,const Tensor& ids,
                         const Tensor& positions,const Tensor& cache_positions,const Tensor& ple_rows,int layer,int64_t confirmed=0) {
        const auto mark=[&](const char* phase){if(auto* window=DecodeWindow::recording())window->mark(layer,phase);};
        const bool detailed=mfq::cuda::runtime_options::mhc_timings();
        const auto marker=[&](const char* role)->GatedResidualMarker {
            if(!detailed || !DecodeWindow::recording())return {};
            return [&,role](const char* phase) {
                const auto name=std::string(role)+phase;mark(name.c_str());
            };
        };
        return mfq::models::qwen4_exp::decoder_layer_chained(std::move(x),bool(ple),bool(gdn),
            [&](const Tensor& hidden){return ple->forward_graph(execution,hidden,ids,ple_rows,confirmed);},
            [](Tensor hidden,Tensor positional){return hidden+positional;},
            [&](const Tensor& hidden){auto result=attention_gr.pre(execution,hidden,marker("attention_mhc_"));mark("attention_core_begin");return result;},
            [&](Tensor branch){auto result=gdn->forward_graph(execution,branch,confirmed);mark("attention_core_end");return result;},
            [&](Tensor branch){auto result=qsa->forward_graph(execution,branch,positions,cache_positions);mark("attention_core_end");return result;},
            [&](Tensor branch,const auto& mix){auto result=ffn_gr.pre_after(execution,branch,mix,marker("ffn_mhc_"));mark("ffn_pre_end");return result;},
            [&](Tensor branch) {
                DecodeWindow::Task prediction{};
                if(next_prefetch && branch.numel()==config.hidden) {
                    const auto routing=config.routing();
                    auto logits=next_router(execution,branch.reshape({1,config.hidden}).to(tb::kFloat16)).to(tb::kFloat32).contiguous();
                    auto selected=moe_topk_cuda(logits,2,
                        routing.activation==mfq::models::RouterActivation::sigmoid,
                        routing.activation==mfq::models::RouterActivation::sqrt_softplus,
                        routing.normalize,routing.delayed_softmax,mfq_nullopt,1e-20,routing.scale);
                    prediction=next_prefetch(selected[0].to(tb::kInt32).contiguous());
                }
                if(router_audit)router_audit->input(layer,branch);
                auto result=ffn(execution,branch);
                // Reserve current-layer readers before selecting speculative
                // slots. The route export itself precedes current FFN math.
                if(prediction.identity)DecodeWindow::recording()->enroll(std::move(prediction));
                return result;
            },
            [&](Tensor branch,const auto& mix){return ffn_gr.post(branch,mix);});
    }
};

std::unique_ptr<Qwen4Block> BlockLoader::block(const Config &c, int layer, bool predictor) {
    return std::make_unique<Qwen4Block>(execution, source, c, layer, predictor);
}

void validate_block_session(const Qwen4Block& block,const Qwen4BlockSessionState& state,int64_t count) {
    if(count<=0 || count>block.config.maximum)throw CudaSessionStateError("invalid Qwen4 session position");
    if(block.qsa)block.qsa->validate_session(state,count);
    else if(state.keys.defined() || state.values.defined() || state.index.defined())
        throw CudaSessionStateError("unexpected Qwen4 attention session tensors");
    if(block.gdn)block.gdn->validate_session(state.convolution,state.recurrent);
    else if(state.convolution.defined() || state.recurrent.defined())
        throw CudaSessionStateError("unexpected Qwen4 recurrent session tensors");
    if(block.ple) {
        block.ple->validate_session(state.ple_convolution,{state.ple_batch,state.ple_tokens});
        for(auto id:state.ple_tokens)if(id<0 || id>=block.config.vocab)
            throw CudaSessionStateError("invalid Qwen4 PLE history token");
    } else if(state.ple_convolution.defined() || state.ple_batch || !state.ple_tokens.empty())
        throw CudaSessionStateError("unexpected Qwen4 PLE session tensors");
}
Qwen4BlockSessionState capture_block_session(const Qwen4Block& block,int64_t position,std::size_t& bytes) {
    Qwen4BlockSessionState state;
    const auto save=[&](const Tensor& tensor) {
        if(!tensor.defined())throw CudaSessionStateError("Qwen4 recurrent session tensor is unavailable");
        auto result=tensor.clone();bytes+=session_tensor_bytes(result);return result;
    };
    if(block.qsa)block.qsa->capture_session(state,position,bytes);
    if(block.gdn) {
        const auto values=block.gdn->graph_state();
        state.convolution=save(*values[0]);state.recurrent=save(*values[1]);
    }
    if(block.ple) {
        state.ple_convolution=save(block.ple->conv_state());
        const auto& context=block.ple->session_context();state.ple_batch=context.batch;state.ple_tokens=context.tokens;
        bytes+=state.ple_tokens.size()*sizeof(int64_t);
    }
    validate_block_session(block,state,position);return state;
}
void restore_block_session(Qwen4Block& block,const Qwen4BlockSessionState& state,int64_t count) {
    block.retain_graph_storage=false;block.reset(1);
    if(block.qsa)block.qsa->restore_session(state,count);
    if(block.gdn)block.gdn->restore_session(state.convolution,state.recurrent);
    if(block.ple)block.ple->restore_session(state.ple_convolution,{state.ple_batch,state.ple_tokens});
}
bool session_positions(const Tensor& positions,int64_t count) {
    return positions.defined() && positions.is_cuda() && positions.dim()>=1 && positions.dim()<=3 &&
        positions.size(-1)==count && (positions.scalar_type()==tb::kInt32 || positions.scalar_type()==tb::kInt64) &&
        (positions.dim()!=3 || positions.size(1)==1);
}

struct ResidentGraphPool {
    std::shared_ptr<mfq::cuda::Context> context;
    hipStream_t stream;
    ResidentGraphPool(int device,hipStream_t value):context(default_context(device)),stream(value) {context->begin_graph_pool(stream);}
    ~ResidentGraphPool() {context->end_graph_pool(stream);}
};
std::shared_ptr<ResidentGraphPool> retain_graph_pool(int device,hipStream_t stream) {
    static std::mutex mutex;
    static std::map<std::pair<int,uintptr_t>,std::weak_ptr<ResidentGraphPool>> pools;
    std::lock_guard lock(mutex);auto& entry=pools[{device,reinterpret_cast<uintptr_t>(stream)}];
    if(auto existing=entry.lock())return existing;
    auto result=std::make_shared<ResidentGraphPool>(device,stream);entry=result;return result;
}

struct Qwen4MtpGraph {
    struct Binding {const Tensor* value;const void* address;std::vector<int64_t> shape;int dtype;};
    struct Cell {
        Tensor ids,hidden,positions,cache_positions,normalized,multi,logits,greedy_host;
        std::vector<Binding> bindings;
        bool used_in_request=false;
        DecodeWindow window;
        Cell(const Tensor& source,const Tensor& tokens,hipStream_t stream)
            :ids(tb::empty_like(tokens)),hidden(tb::empty_like(source)),
             positions(tb::empty({3,1,source.size(1)},tokens.options().dtype(tb::kInt32))),
             cache_positions(tb::empty({source.size(1)},tokens.options().dtype(tb::kInt64))),
             greedy_host(tb::empty({1},tokens.options().device(tb::kCPU).dtype(tb::kInt64).pinned_memory(true))),
             window(stream,mfq::cuda::runtime_options::layer_timings()) {}
    };
    MfqCudaStream stream=mfq_get_current_cuda_stream();
    std::shared_ptr<ResidentGraphPool> resident_pool=retain_graph_pool(stream.device_index(),stream.stream());
    std::map<std::pair<int,int>,std::unique_ptr<Cell>> cells;
    const void* target_binding=nullptr;
    size_t target_kind=SIZE_MAX;
    bool failed=false;
    bool reusable(const Qwen4ExpMtp& predictor) const {
        return !failed && predictor.reuse_decode_graphs && stream.stream()==mfq_current_cuda_stream() &&
            predictor.layers.size()==1 && predictor.layers[0]->resident_tq && predictor.layers[0]->qsa &&
            predictor.layers[0]->qsa->reusable_graph_storage();
    }
    void restore_legacy(Qwen4ExpMtp& predictor) {
        cells.clear();
        if(!predictor.layers.empty() && predictor.layers[0]->qsa)predictor.layers[0]->qsa->leave_graph();
    }
    MtpStep run(Qwen4ExpMtp& predictor,const MtpTarget& target,const Tensor& hidden,const Tensor& ids,const Tensor& supplied) {
        MFQ_RUNTIME_CHECK(!failed && stream.stream()==mfq_current_cuda_stream(),"MTP predictor graph requires session recovery");
        auto& block=*predictor.layers[0];auto& execution=*predictor.execution;
        const auto tokens=ids.size(1),start=predictor.lengths[0];
        MFQ_RUNTIME_CHECK(start>0 && tokens<=predictor.config.maximum-start,"MTP predictor exceeds context capacity");
        try {
            if(target_binding && (target_binding!=target.identity || target_kind!=target.model.index()))cells.clear();
            target_binding=target.identity;target_kind=target.model.index();
            block.qsa->prepare_graph(predictor.positions[0]);
            bool changed=false;
            for(const auto& [key,value]:cells)for(const auto& binding:value->bindings)
                if(!binding.value->defined() || binding.value->data_ptr()!=binding.address ||
                    int(binding.value->scalar_type())!=binding.dtype || binding.value->dim()!=int64_t(binding.shape.size()) ||
                    !std::equal(binding.shape.begin(),binding.shape.end(),binding.value->sizes().begin()))changed=true;
            if(changed)cells.clear();
            auto& owner=cells[{int(tokens),int(hidden.scalar_type())}];
            if(!owner)owner=std::make_unique<Cell>(hidden,ids,stream.stream());
            auto& cell=*owner;cell.hidden.copy_(hidden);cell.ids.copy_(ids);
            mfq_qwen4_exp::fill_decode_positions(cell.cache_positions,cell.positions,start);
            if(supplied.defined()) {
                auto positions=supplied;
                if(positions.dim()==1)positions=positions.reshape({1,1,tokens}).expand({3,1,tokens});
                else if(positions.dim()==2)positions=positions.unsqueeze(1);
                if(positions.dim()==3 && positions.size(0)==4)positions=positions.narrow(0,1,3);
                MFQ_RUNTIME_CHECK(positions.dim()==3 && positions.size(0)==3 && positions.size(1)==1 &&
                    positions.size(2)==tokens,"Qwen4 MTP positions require [T]/[3,T]/[3,1,T]");
                cell.positions.copy_(positions.to(tb::kInt32));
            }
            predictor.graph_initialized=predictor.graph_initialized || !cell.used_in_request;
            if(!cell.window.valid()) {
                DecodeGraphBranchScope branches(execution.decode_graph_serial_branches);
                cell.window.capture([&] {
                    auto embeddings=target.embed(cell.ids);
                    auto e=predictor.embedding_fusion(execution,rms_norm(embeddings,predictor.embedding_norm+1,predictor.config.eps));
                    auto h=predictor.hidden_fusion(execution,rms_norm(cell.hidden,predictor.hidden_norm+1,predictor.config.eps)
                        .reshape({1,tokens,predictor.config.streams,predictor.config.hidden}));
                    auto input=(h+e.unsqueeze(-2)).reshape({1,tokens,predictor.config.streams*predictor.config.hidden});
                    // Preserve the eager predictor's projection/reduction order.
                    cell.multi=mfq::models::qwen4_exp::decoder_layer(
                        std::move(input),false,false,[](const Tensor&){return Tensor{};},
                        [](Tensor value,Tensor){return value;},
                        [&](const Tensor& value){return block.attention_gr.pre(execution,value);},
                        [](Tensor)->Tensor{throw std::logic_error("Qwen4 MTP requires QSA");},
                        [&](Tensor branch){return block.qsa->forward_graph(execution,branch,cell.positions,cell.cache_positions);},
                        [&](Tensor branch,const auto& mix){return block.attention_gr.post(branch,mix);},
                        [&](const Tensor& value){return block.ffn_gr.pre(execution,value);},
                        [&](Tensor branch){return block.ffn(execution,branch);},
                        [&](Tensor branch,const auto& mix){return block.ffn_gr.post(branch,mix);});
                    cell.normalized=predictor.final_mixer->pre(execution,cell.multi)[0];
                    cell.logits=target.logits(cell.normalized.narrow(1,tokens-1,1).contiguous());
                    auto greedy=sample_greedy_cuda(cell.logits.reshape({1,-1}).contiguous());
                    MFQ_CUDA_CHECK(hipMemcpyAsync(cell.greedy_host.data_ptr(),greedy.data_ptr(),sizeof(int64_t),
                        hipMemcpyDeviceToHost,stream.stream()));
                },[]{},[&]{cell.normalized={};cell.multi={};cell.logits={};});
                cell.bindings.clear();
                for(auto* value:block.graph_bindings())cell.bindings.push_back(
                    {value,value->data_ptr(),value->sizes().vec(),int(value->scalar_type())});
                ++predictor.graph_captures;
                if(const auto* trace=std::getenv("MFQ_TRACE_NATIVE_CUDA_GRAPH");trace && trace[0]=='1')
                    std::fprintf(stderr,"hip_mtp_predictor_graph tokens=%lld captures=%llu\n",(long long)tokens,
                        (unsigned long long)predictor.graph_captures);
            }
            cell.window.run();cell.used_in_request=true;++predictor.graph_replays;
            block.qsa->advance_graph(tokens);predictor.lengths[0]=start+tokens;predictor.batch=1;
            predictor.positions[0]=tb::cat({predictor.positions[0],cell.positions},-1);
            return {cell.normalized.clone(),cell.multi.clone(),cell.logits.clone(),
                static_cast<int32_t>(*cell.greedy_host.data_ptr<int64_t>())};
        }catch(...) {failed=true;throw;}
    }
};
bool predictor_graph_eligible(const Qwen4ExpMtp& predictor,const Tensor& hidden,const Tensor& ids) {
    if(const auto* option=std::getenv("MFQ_HIP_MTP_PREDICTOR_GRAPH");option && option[0]=='0')return false;
    if(const auto* option=std::getenv("MFQ_HIP_MTP_GRAPH");option && option[0]=='0')return false;
    if(const auto* option=std::getenv("MFQ_CUDA_GRAPH");option && option[0]=='0')return false;
    return predictor.execution && !predictor.execution->moe_expert_cache &&
        predictor.layers.size()==1 && predictor.layers[0]->resident_tq &&
        predictor.layers[0]->qsa && predictor.lengths[0]>0 && predictor.batch==1 &&
        predictor.positions[0].defined() && predictor.positions[0].dim()==3 &&
        predictor.positions[0].size(0)==3 && predictor.positions[0].size(1)==1 &&
        hidden.is_cuda() && ids.is_cuda() && ids.scalar_type()==tb::kInt64 &&
        (hidden.scalar_type()==tb::kFloat16 || hidden.scalar_type()==tb::kFloat32) && ids.dim()==2 && ids.size(0)==1 && ids.size(1)>0 && ids.size(1)<=8 &&
        hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==ids.size(1) && hidden.size(2)==predictor.hidden_norm.numel();
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
    else for(auto& [key,cell]:graph_registry->cells)cell->used_in_request=false;
    graph_initialized=false;
    for(auto& layer:layers) {layer->retain_graph_storage=retain;layer->reset(next_batch);}
    for (auto &pos : positions)
        pos = Tensor();
    std::fill(lengths.begin(), lengths.end(), 0);
    batch = next_batch;
}

std::pair<Tensor, Tensor> Qwen4ExpMtp::evaluate(const MtpTarget &target, const Tensor &hidden,
                                       const Tensor &ids, int64_t depth, bool cache,
                                       const Tensor &supplied_positions,
                                       const Tensor &supplied_embeddings) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) &&
                          hidden.size(2) == hidden_norm.numel(),
                      "Qwen4-Exp MTP input geometry mismatch");
    if(graph_registry) {
        MFQ_RUNTIME_CHECK(!graph_registry->failed,"MTP predictor requires reset after its previous graph error");
        if(graph_registry->reusable(*this) && cache && ids.size(0)==1)layers[0]->qsa->leave_graph(true);
        else {graph_registry->restore_legacy(*this);graph_registry.reset();}
    }
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
        [&](const Tensor &multi) { return final_mixer->pre(*execution, multi)[0]; },
        [&](int64_t layer, const auto &pos) { positions[layer] = pos[1]; });
}

Tensor Qwen4ExpMtp::forward(const MtpTarget &target, Tensor hidden, Tensor ids) {
    return evaluate(target, hidden, ids).first;
}

MtpStep Qwen4ExpMtp::step(const MtpTarget &target, Tensor hidden, Tensor ids) {
    if(predictor_graph_eligible(*this,hidden,ids)) {
        if(!graph_registry)graph_registry=std::make_unique<Qwen4MtpGraph>();
        return graph_registry->run(*this,target,hidden,ids,{});
    }
    auto result = evaluate(target, hidden, ids);
    return {std::move(result.first), std::move(result.second)};
}

MtpStep Qwen4ExpMtp::step_positioned(const MtpTarget &target, Tensor hidden, Tensor ids, Tensor positions) {
    if(predictor_graph_eligible(*this,hidden,ids)) {
        if(!graph_registry)graph_registry=std::make_unique<Qwen4MtpGraph>();
        return graph_registry->run(*this,target,hidden,ids,positions);
    }
    auto result=evaluate(target,hidden,ids,0,true,positions);
    return {std::move(result.first),std::move(result.second)};
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

bool Qwen4ExpMtp::supports_session_state() const noexcept { return execution && !layers.empty(); }
MtpSessionState Qwen4ExpMtp::capture_session_state(int64_t target_position,const Tensor& hidden) const {
    const auto position=target_position-1;
    if(!supports_session_state() || position<=0 || position>cache_position() || !hidden.defined() ||
        !hidden.is_cuda() || hidden.dim()!=3 || hidden.size(0)!=1 || hidden.size(1)!=1 || hidden.size(2)!=hidden_norm.numel())
        throw CudaSessionStateError("Qwen4 MTP session boundary is unavailable");
    MtpSessionState state;state.cache_pos=position;
    for(size_t i=0;i<layers.size();++i) {
        const auto count=std::min(position,lengths[i]);state.qwen4_lengths.push_back(count);
        state.qwen4_blocks.push_back(count?capture_block_session(*layers[i],count,state.bytes):Qwen4BlockSessionState{});
        auto pos=count?positions[i].narrow(-1,0,count).clone():Tensor{};
        state.bytes+=session_tensor_bytes(pos);state.qwen4_positions.push_back(std::move(pos));
    }
    state.last_target_hidden=hidden.clone();state.bytes+=session_tensor_bytes(state.last_target_hidden);return state;
}
void Qwen4ExpMtp::restore_session_state(const MtpSessionState& state) {
    if(!supports_session_state() || state.cache_pos<=0 || state.cache_pos>config.maximum ||
        state.qwen4_blocks.size()!=layers.size() || state.qwen4_positions.size()!=layers.size() ||
        state.qwen4_lengths.size()!=layers.size() || state.qwen4_lengths.front()!=state.cache_pos ||
        !state.blocks.empty() || !state.deepseek_v41_rings.empty() || !state.last_target_hidden.defined() ||
        !state.last_target_hidden.is_cuda() || state.last_target_hidden.dim()!=3 ||
        state.last_target_hidden.size(0)!=1 || state.last_target_hidden.size(1)!=1 ||
        state.last_target_hidden.size(2)!=hidden_norm.numel())throw CudaSessionStateError("incompatible Qwen4 MTP session state");
    for(size_t i=0;i<layers.size();++i) {
        const auto count=state.qwen4_lengths[i];
        if(count<0 || count>state.cache_pos || (count && !session_positions(state.qwen4_positions[i],count)))
            throw CudaSessionStateError("incompatible Qwen4 MTP session positions");
        if(count)validate_block_session(*layers[i],state.qwen4_blocks[i],count);
    }
    graph_registry.reset();
    reset(1);
    for(size_t i=0;i<layers.size();++i)if(state.qwen4_lengths[i]) {
        restore_block_session(*layers[i],state.qwen4_blocks[i],state.qwen4_lengths[i]);
        positions[i]=state.qwen4_positions[i].clone();lengths[i]=state.qwen4_lengths[i];
    }
}


} // namespace mfq::cuda::qwen4_exp

namespace mfq::cuda {
struct Qwen4DecodeGraph {
    struct PleStage {
        qwen4_exp::Ple* ple;
        mfq_tensor_backend::Tensor rows;
        std::shared_ptr<RowStage> transfer;
        HostBuffer flags{128,true};
        uint32_t* device=nullptr;
        PleStage(qwen4_exp::Ple* p,int tokens,int hidden):ple(p) {
            charge_tensor_host_bytes(128);
            MFQ_CUDA_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&device),flags.data(),0));
            std::memset(flags.data(),0,128);publish_mapped_flag(host()+16);
            transfer=ple->graph_stage(tokens,mfq_tensor_backend::Device{mfq_tensor_backend::kCUDA,0});
            rows=transfer->output().reshape({1,tokens,hidden});
        }
        ~PleStage(){tensor_host_bytes.fetch_sub(128);}
        uint32_t* host(){return static_cast<uint32_t*>(flags.data());}
    };
    struct Cell {
        struct Binding {const mfq_tensor_backend::Tensor* value;const void* address;std::vector<int64_t> shape;int dtype;};
        std::vector<Binding> bindings;
        bool used_in_request=false;
        mfq_tensor_backend::Tensor ids,positions,cache_positions,host_ids,output,raw;
        std::vector<std::unique_ptr<PleStage>> ple;
        DecodeWindow window;
        Cell(int tokens,int hidden,const std::vector<std::unique_ptr<Block>>& blocks,hipStream_t stream)
            :window(stream,mfq::cuda::runtime_options::layer_timings()) {
            namespace tb=mfq_tensor_backend;
            const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
            ids=tb::zeros({1,tokens},options);positions=tb::zeros({tokens},options);cache_positions=tb::zeros({tokens},options);
            host_ids=tb::zeros({1,tokens},options.device(tb::kCPU));
            for(const auto& block:blocks) {
                auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                if(b.ple)ple.push_back(std::make_unique<PleStage>(b.ple.get(),tokens,hidden));
            }
        }
    };
    MfqCudaStream stream,ple_stream;
    std::shared_ptr<qwen4_exp::ResidentGraphPool> resident_pool;
    std::map<std::tuple<int,int,int>,std::unique_ptr<Cell>> cells;
    std::vector<qwen4_exp::Qsa*> qsa;
    std::vector<qwen4_exp::Qwen4Block*> blocks;
    bool failed=false;
    explicit Qwen4DecodeGraph(Qwen4CausalLm& model):stream(mfq_get_current_cuda_stream()),ple_stream(mfq_get_stream_from_pool()) {
        if(!model.execution->moe_expert_cache)
            resident_pool=qwen4_exp::retain_graph_pool(stream.device_index(),stream.stream());
        const bool enabled=model.router_lookahead.value_or(mfq::cuda::runtime_options::router_lookahead());
        for(std::size_t i=0;i<model.blocks.size();++i) {
            auto& current=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
            current.next_router={};current.next_prefetch={};
            if(enabled && i+1<model.blocks.size()) {
                auto& next=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i+1]);
                MFQ_RUNTIME_CHECK(next.router && next.prefetch,"router lookahead requires the tiered FFN pipeline");
                current.next_router=next.router;current.next_prefetch=next.prefetch;
            }
        }
        for(const auto& block:model.blocks) {
            auto* current=static_cast<qwen4_exp::Qwen4Block*>(block.get());blocks.push_back(current);
            if(current->qsa)qsa.push_back(current->qsa.get());
        }
    }
    ~Qwen4DecodeGraph() {(void)hipStreamSynchronize(ple_stream.stream());cells.clear();}
    void restore_legacy() {
        cells.clear();for(auto* attention:qsa)attention->leave_graph();
        for(auto* block:blocks)block->retain_graph_storage=false;
    }
    bool reusable() const {
        return !failed && stream.stream()==mfq_current_cuda_stream() && !qsa.empty() &&
            std::all_of(blocks.begin(),blocks.end(),[](const auto* block){return block->resident_tq;}) &&
            std::all_of(qsa.begin(),qsa.end(),[](const auto* attention){return attention->reusable_graph_storage();});
    }
    void prepare_backings(Qwen4CausalLm& model) {
        finish_moe_expert_exchanges(model.execution->moe_expert_cache);
        for(auto* attention:qsa)attention->prepare_graph(model.positions);
        bool changed=false;
        for(const auto& [key,cell]:cells)for(const auto& binding:cell->bindings)
            if(!binding.value->defined() || binding.value->data_ptr()!=binding.address ||
                binding.value->dim()!=int64_t(binding.shape.size()) ||
                !std::equal(binding.shape.begin(),binding.shape.end(),binding.value->sizes().begin()) ||
                int(binding.value->scalar_type())!=binding.dtype)changed=true;
        if(changed) {cells.clear();++model.graph_binding_invalidations;}
    }

    mfq_tensor_backend::Tensor run(Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int kind,int confirmed=0) {
        namespace tb=mfq_tensor_backend;
        if(failed)throw std::runtime_error("MFQ model session requires recovery after its previous error");
        if(stream.stream()!=mfq_current_cuda_stream())throw std::runtime_error("MFQ model window changed execution stream");
        const int tokens=static_cast<int>(ids.size(1));
        try {prepare_backings(model);}catch(...) {failed=true;throw;}
        auto& owner=cells[{tokens,kind,confirmed}];
        if(!owner)owner=std::make_unique<Cell>(tokens,model.config.hidden,model.blocks,stream.stream());
        auto& cell=*owner;
        const bool already_captured=cell.window.valid();
        cell.host_ids.copy_(ids.to(tb::kCPU,tb::kInt64));cell.ids.copy_(ids);
        mfq_qwen4_exp::fill_decode_positions(cell.cache_positions,cell.positions,model.cache_pos,model.decode_position_delta);
        try {
            if(!cell.window.valid()) {
                DecodeGraphBranchScope branch_scope(
                    model.execution->decode_graph_serial_branches);
                for(const auto& block:model.blocks) {
                    auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                    if(b.qsa)b.qsa->prepare_graph(model.positions);
                    if(confirmed) {
                        if(b.gdn)b.gdn->prepare_graph_checkpoint();
                        if(b.ple)b.ple->prepare_graph_checkpoint();
                    }
                }
                struct State {tb::Tensor* target;tb::Tensor saved;const void* address;};
                std::vector<State> states;
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
                    for(const auto& block:model.blocks) {
                        cell.window.mark(layer,"block_begin");
                        auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);tb::Tensor rows;
                        if(b.ple) {
                            auto* stage=cell.ple[ple_index++].get();rows=stage->rows;
                            cell.window.enroll({stage,
                                [&cell,stage] {std::memset(stage->flags.data(),0,128);stage->ple->prefetch_tokens(cell.host_ids);},
                                [this,&cell,stage] {
                                    MfqCudaStreamGuard guard(ple_stream);
                                    stage->ple->upload_graph(cell.host_ids,*stage->transfer);
                                    // CPU token hashes are available before these layers execute.
                                    // Prepare their rows while the earlier GPU blocks run.
                                    wait_route_publication(stage->host(),stream.stream());
                                    MFQ_CUDA_CHECK(hipLaunchHostFunc(ple_stream.stream(),[](void* p){publish_mapped_flag(static_cast<uint32_t*>(p));},stage->host()+16));
                                    check_stream_progress(ple_stream.stream());
                                },{},[stage]{publish_mapped_flag(stage->host()+16);}});
                            signal_mapped_flag(stage->device,stream.stream());wait_mapped_flag(stage->device+16,stream.stream());
                            stage->transfer->decode();
                        }
                        cell.window.mark(layer,"attention_begin");
                        hidden=b.execute_graph(*model.execution,std::move(hidden),cell.ids,cell.positions,cell.cache_positions,rows,layer,confirmed);
                        cell.window.mark(layer++,"block_end");
                    }
                    if(kind==3)cell.raw=hidden;
                    hidden=model.adapter_finalize_hidden(std::move(hidden),model.output_norm,1,tokens);
                    if(kind==3)cell.output=hidden;
                    else if(kind==2)cell.output=model.adapter_next_token(model.lm_head,model.last_hidden(std::move(hidden)));
                    else if(kind==1)cell.output=model.adapter_last_logits(model.lm_head,model.last_hidden(std::move(hidden)));
                    else cell.output=model.logits_from_hidden(std::move(hidden));
                    cell.window.mark(-1,"window_end");
                    auto audit=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks.front()).router_audit;
                    if(audit)cell.window.enroll({audit.get(),[]{},[]{},[audit]{audit->finish_sample();},{}});
                },restore,[&]{cell.output={};cell.raw={};});
                for(auto* block:blocks)for(auto* value:block->graph_bindings())
                    cell.bindings.push_back({value,value->data_ptr(),value->sizes().vec(),int(value->scalar_type())});
                ++model.graph_captures;
            }
            if(confirmed)for(const auto& block:model.blocks) {
                auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                if(b.gdn)b.gdn->arm_graph_checkpoint();
                if(b.ple)b.ple->arm_graph_checkpoint(cell.host_ids,confirmed);
            }
            cell.window.run();
            if(already_captured && !cell.used_in_request)++model.graph_cross_request_replays;
            cell.used_in_request=true;
            for(const auto& block:model.blocks) {
                auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);if(b.qsa)b.qsa->advance_graph(tokens);
            }
            model.cache_pos+=tokens;
            model.positions=tb::cat({model.positions,cell.positions},-1);model.batch=1;
            return cell.output;
        } catch(...) {failed=true;throw;}
    }
};

std::optional<mfq_tensor_backend::Tensor> qwen4_decode_graph_logits(Qwen4CausalLm& model,mfq_tensor_backend::Tensor ids,int kind) {
    if(model.cache_pos==0 || model.batch!=1 || model.speculative_start>=0)return {};
    const bool resident_tq=std::all_of(model.blocks.begin(),model.blocks.end(),[](const auto& block) {
        const auto& b=static_cast<const qwen4_exp::Qwen4Block&>(*block);
        return b.resident_tq && (!b.ple || b.ple->graph_stage_supported());
    });
    if(!resident_tq && (!model.execution->config.moe_pipeline || !model.execution->moe_expert_cache))return {};
    if(const auto* graph=std::getenv("MFQ_CUDA_GRAPH");graph && std::string_view(graph)=="0")return {};
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 || model.positions.dim()!=1)return {};
    MFQ_RUNTIME_CHECK(model.cache_pos+ids.size(1)<=model.max_position_embeddings(),"MFQ window exceeds context");
    if(!model.graph_registry)model.graph_registry=std::make_unique<Qwen4DecodeGraph>(model);
    return model.graph_registry->run(model,model.device_ids(std::move(ids)),kind);
}

std::optional<mfq_tensor_backend::Tensor> qwen4_decode_graph_hidden(Qwen4CausalLm& model,
        mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor* raw,int64_t confirmed,bool* initialized) {
    namespace tb=mfq_tensor_backend;
    if(initialized)*initialized=false;
    if(model.cache_pos==0 || model.batch!=1 || model.speculative_start>=0 || model.positions.dim()!=1)return {};
    const auto* option=std::getenv("MFQ_HIP_MTP_GRAPH");if(option && option[0]=='0')return {};
    if(const auto* graph=std::getenv("MFQ_CUDA_GRAPH");graph && graph[0]=='0')return {};
    const bool resident=std::all_of(model.blocks.begin(),model.blocks.end(),[](const auto& block) {
        const auto& b=static_cast<const qwen4_exp::Qwen4Block&>(*block);
        return b.resident_tq && (!b.ple || b.ple->graph_stage_supported());
    });
    if(!resident)return {};
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 ||
        confirmed<0 || (confirmed && confirmed>=ids.size(1)))return {};
    MFQ_RUNTIME_CHECK(model.cache_pos+ids.size(1)<=model.max_position_embeddings(),"MTP graph exceeds context");
    if(!model.graph_registry)model.graph_registry=std::make_unique<Qwen4DecodeGraph>(model);
    const auto found=model.graph_registry->cells.find({int(ids.size(1)),3,int(confirmed)});
    if(initialized)*initialized=found==model.graph_registry->cells.end() || !found->second->window.valid() || !found->second->used_in_request;
    if(confirmed) {
        model.speculative_start=model.cache_pos;
        model.speculative_confirmed=confirmed;
    }
    auto hidden=model.graph_registry->run(model,model.device_ids(std::move(ids)),3,int(confirmed)).clone();
    if(raw) {
        const auto tokens=hidden.size(1);
        *raw=model.graph_registry->cells.at({int(tokens),3,int(confirmed)})->raw.clone();
    }
    if(initialized && *initialized) {
        const auto* trace=std::getenv("MFQ_TRACE_NATIVE_CUDA_GRAPH");
        if(trace && trace[0]=='1')std::fprintf(stderr,"hip_mtp_target_graph tokens=%lld confirmed=%lld\n",
            static_cast<long long>(hidden.size(1)),static_cast<long long>(confirmed));
    }
    return hidden;
}

void qwen4_set_rotary_fusion(Qwen4CausalLm& model, bool enabled) {
    MFQ_RUNTIME_CHECK(!model.graph_registry && model.cache_pos == 0,
                      "reset the model before switching rotary execution");
    mfq_hip_synchronize();
    for (const auto& block : model.blocks) {
        auto* attention = static_cast<qwen4_exp::Qwen4Block&>(*block).qsa.get();
        if (attention) attention->set_rotary_fused(enabled);
    }
}

void qwen4_set_attention_grouping(Qwen4CausalLm& model, bool enabled) {
    MFQ_RUNTIME_CHECK(!model.graph_registry && model.cache_pos == 0,
                      "reset the model before switching attention projection groups");
    mfq_hip_synchronize();
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

Qwen4Model::Qwen4Model() = default;
Qwen4Model::~Qwen4Model() = default;
Qwen4Model::Qwen4Model(Qwen4Model&&) noexcept = default;
Qwen4Model& Qwen4Model::operator=(Qwen4Model&&) noexcept = default;

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
    return std::make_unique<qwen4_exp::Qwen4Block>(*execution, source, config, layer);
}

void Qwen4Model::adapter_reset(int64_t new_batch) {
    if(execution && execution->moe_expert_cache)finish_moe_expert_exchanges(execution->moe_expert_cache);
    const bool retain=reuse_decode_graphs && new_batch==1 && graph_registry && graph_registry->reusable();
    if(graph_registry)for(auto* block:graph_registry->blocks)block->retain_graph_storage=retain;
    if(!retain)graph_registry.reset();
    else {
        ++graph_retained_resets;
        for(auto& [key,cell]:graph_registry->cells)cell->used_in_request=false;
        if(const auto* trace=std::getenv("MFQ_TRACE_NATIVE_CUDA_GRAPH");trace && trace[0]=='1')
            std::fprintf(stderr,"hip_decode_graph_reuse retained_cells=%zu\n",graph_registry->cells.size());
    }
    positions = mfq_tensor_backend::Tensor();
    batch = new_batch;
}
void Qwen4Model::adapter_begin_forward(bool) {
    if(graph_registry) {
        if(reuse_decode_graphs && graph_registry->reusable())
            for(auto* attention:graph_registry->qsa)attention->leave_graph(true);
        else {graph_registry->restore_legacy();graph_registry.reset();}
    }
}

mfq_tensor_backend::Tensor
Qwen4Model::adapter_block_positions(const mfq_tensor_backend::Tensor &full_positions,
                                    const mfq_tensor_backend::Tensor &, int device) const {
    return tensor_to_cuda_device(execution->model_parallel_collectives, full_positions, device);
}

void Qwen4Model::adapter_finish_forward(const mfq_tensor_backend::Tensor &full_positions,
                                        int64_t new_batch, int64_t) {
    positions = full_positions;
    batch = new_batch;
}

mfq_tensor_backend::Tensor Qwen4Model::adapter_finalize_hidden(mfq_tensor_backend::Tensor hidden,
                                                               const mfq_tensor_backend::Tensor &,
                                                               int64_t, int64_t) const {
    return final_mixer->pre(*execution, hidden)[0];
}

mfq_tensor_backend::Tensor
Qwen4Model::adapter_last_logits(const QuantLinear &lm_head,
                                mfq_tensor_backend::Tensor hidden) const {
    return adapter_logits(lm_head, std::move(hidden));
}

mfq_tensor_backend::Tensor Qwen4Model::adapter_next_token(const QuantLinear &lm_head,
                                                          mfq_tensor_backend::Tensor hidden) const {
    auto logits=adapter_logits(lm_head,std::move(hidden)).contiguous();
    auto shape=logits.sizes().vec();shape.pop_back();
    return sample_greedy_cuda(logits.reshape({-1,logits.size(-1)})).reshape(shape);
}

void Qwen4Model::adapter_rollback_speculative(int64_t keep) {
    if (positions.defined())
        positions = positions.narrow(-1, 0, keep);
}

} // namespace mfq::cuda

namespace mfq::cuda {

TextSessionStateKind CudaSessionCodec<Qwen4Model>::kind(const Qwen4CausalLm& model) {
    return model.execution && !model.blocks.empty() && std::all_of(model.blocks.begin(),model.blocks.end(),
        [](const auto& block){return dynamic_cast<const qwen4_exp::Qwen4Block*>(block.get())!=nullptr;})
        ?TextSessionStateKind::Qwen4:TextSessionStateKind::Unsupported;
}
bool CudaSessionCodec<Qwen4Model>::supports_paged(const Qwen4CausalLm& model) { return kind(model)==TextSessionStateKind::Qwen4; }
TextSessionState CudaSessionCodec<Qwen4Model>::capture(const Qwen4CausalLm& model,const std::vector<int64_t>& tokens) {
    if(kind(model)!=TextSessionStateKind::Qwen4 || model.batch!=1 || model.cache_pos<=0 ||
        size_t(model.cache_pos)!=tokens.size() || model.speculative_start>=0 ||
        !qwen4_exp::session_positions(model.positions,model.cache_pos))
        throw CudaSessionStateError("Qwen4 text session boundary is unavailable");
    TextSessionState state;state.tokens=tokens;state.cache_pos=model.cache_pos;
    state.decode_position_delta=model.decode_position_delta;std::vector<Qwen4BlockSessionState> blocks;
    for(const auto& block:model.blocks)blocks.push_back(qwen4_exp::capture_block_session(
        static_cast<const qwen4_exp::Qwen4Block&>(*block),model.cache_pos,state.bytes));
    state.payload=std::move(blocks);state.qwen4_positions=model.positions.clone();
    state.bytes+=session_tensor_bytes(state.qwen4_positions);return state;
}
void CudaSessionCodec<Qwen4Model>::restore(Qwen4CausalLm& model,const TextSessionState& state) {
    const auto* blocks=std::get_if<std::vector<Qwen4BlockSessionState>>(&state.payload);
    if(kind(model)!=TextSessionStateKind::Qwen4 || !blocks || blocks->size()!=model.blocks.size() ||
        state.cache_pos<=0 || state.cache_pos>model.max_position_embeddings() ||
        size_t(state.cache_pos)!=state.tokens.size() || !qwen4_exp::session_positions(state.qwen4_positions,state.cache_pos))
        throw CudaSessionStateError("incompatible Qwen4 text session state");
    for(size_t i=0;i<blocks->size();++i)qwen4_exp::validate_block_session(
        static_cast<const qwen4_exp::Qwen4Block&>(*model.blocks[i]),(*blocks)[i],state.cache_pos);
    // Retire captured pointer bindings before replacing saved recurrent tensors.
    model.graph_registry.reset();
    for(const auto& block:model.blocks)static_cast<qwen4_exp::Qwen4Block&>(*block).retain_graph_storage=false;
    model.reset(1);
    for(size_t i=0;i<blocks->size();++i)qwen4_exp::restore_block_session(
        static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]),(*blocks)[i],state.cache_pos);
    model.positions=state.qwen4_positions.clone();model.cache_pos=state.cache_pos;
    model.decode_position_delta=state.decode_position_delta;
}

} // namespace mfq::cuda

namespace mfq::models {
template struct qwen4_exp::CausalLm<cuda::CudaCausalOps<cuda::Qwen4Model>>;
} // namespace mfq::models
