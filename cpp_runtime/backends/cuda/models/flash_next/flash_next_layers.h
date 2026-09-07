#pragma once

// Flash-Next layers use the shared CUDA model loader and the common Block
// context. Architecture-specific execution stays out of the CLI/model loop.
namespace flash_runtime {
namespace tb = mfq_tensor_backend;
using Tensor = tb::Tensor;
using Linear = mfq::flash_next::Linear;
using Routed = std::function<Tensor(const Tensor&, const Tensor&)>;

static Linear linear(const MfqFile& file, const std::string& name) {
    auto weight=std::make_shared<QuantLinear>(load_quant_linear(file,name));
    return [weight](const Tensor& x) { return weight->forward(x); };
}

static Tensor dense(const MfqFile& file, const std::string& name) {
    const auto& dtype=file.record(name).dtype;
    MFQ_RUNTIME_CHECK(dtype=="F32" || dtype=="F16" || dtype=="BF16",
        "Flash-Next requires a dense parameter: ",name);
    auto value=load_dense_gpu(file,name);
    return value.to(dtype=="F16" ? tb::kFloat16 : dtype=="BF16" ? tb::kBFloat16 : tb::kFloat32);
}

static Routed routed(const MfqFile& file, const std::string& name, int layer,
    int64_t experts, int64_t output, int64_t input) {
    const auto& dtype=file.record(name).dtype;
    if (dtype=="F32" || dtype=="F16" || dtype=="BF16") {
        auto w=dense(file,name);
        MFQ_RUNTIME_CHECK(w.sizes().vec()==std::vector<int64_t>({experts,output,input}),
            "Flash-Next dense expert tensor shape mismatch: ",name);
        return [w,output,input](const Tensor& x,const Tensor& ids) {
            const auto rows=ids.size(0), routes=ids.size(1);
            auto selected=w.index_select(0,ids.reshape({-1}).to(tb::kInt64)).reshape({rows,routes,output,input});
            auto source=x.dim()==2 ? x.unsqueeze(1).expand({rows,routes,input}) : x;
            return tb::matmul(selected,source.to(w.scalar_type()).unsqueeze(-1)).squeeze(-1);
        };
    }
    auto w=std::make_shared<NintMoeWeight>(load_nint_moe_gpu(file,name,true,layer,"flash_next"));
    MFQ_RUNTIME_CHECK(w->n_experts==experts && w->out_per_expert==output && w->neuron_len==input,
        "Flash-Next routed tensor shape mismatch: ",name);
    return [w,experts](const Tensor& x,const Tensor& ids) {
        auto route=build_moe_route_plan(ids.to(tb::kInt32).contiguous(),int(experts));
        return w->forward(x.contiguous(),route);
    };
}

static Linear headwise(Routed projection,int64_t heads,int64_t output) {
    return [projection=std::move(projection),heads,output](const Tensor& x) {
        MFQ_RUNTIME_CHECK(x.dim()==4 && x.size(2)==heads,"Flash-Next head-wise projection shape mismatch");
        const auto b=x.size(0),t=x.size(1),rows=b*t*heads;
        auto ids=tb::arange(rows,x.options().dtype(tb::kInt32)).remainder(heads).reshape({rows,1});
        return projection(x.reshape({rows,x.size(-1)}),ids).reshape({b,t,heads,output});
    };
}

static Linear dense_ffn(const MfqFile& file,const std::string& p,double limit) {
    auto gate=linear(file,p+".gate.weight"),up=linear(file,p+".up.weight"),down=linear(file,p+".down.weight");
    return [gate,up,down,limit](const Tensor& x) {
        auto g=tb::clamp_max(gate(x),limit),u=tb::clamp(up(x),-limit,limit);
        return down((g*tb::sigmoid(g))*u);
    };
}

static Linear glm_ffn(const MfqFile& file,const mfq::flash_next::GlmConfig& c,int i,const std::string& root="model") {
    const auto p=root+".block."+std::to_string(i)+".mlp";
    if (root=="model" && c.mlp_types.at(i)=="dense") return dense_ffn(file,p,c.swiglu_limit);
    auto gate_up=routed(file,p+".experts.gate_up.weight",i,c.experts,2*c.moe_intermediate,c.hidden);
    auto down=routed(file,p+".experts.down.weight",i,c.experts,c.hidden,c.moe_intermediate);
    auto router=linear(file,p+".router.weight"),shared=dense_ffn(file,p+".shared_expert",c.swiglu_limit);
    auto bias=dense(file,p+".router.bias").to(tb::kFloat32).contiguous();
    return [gate_up,down,router,shared,bias,c](const Tensor& x) {
        auto source=x.reshape({-1,c.hidden}).to(tb::kFloat16);
        auto selected=moe_topk_cuda(router(source.to(tb::kFloat32)).to(tb::kFloat32).contiguous(),
            c.topk,true,false,c.normalize_routes,false,bias,1e-20,c.router_scale);
        auto gu=gate_up(source,selected[0]);
        auto gate=tb::clamp_max(gu.narrow(-1,0,c.moe_intermediate),c.swiglu_limit);
        auto up=tb::clamp(gu.narrow(-1,c.moe_intermediate,c.moe_intermediate),-c.swiglu_limit,c.swiglu_limit);
        auto pairs=down((gate*tb::sigmoid(gate))*up,selected[0]);
        // Metal accumulates routes in FP32, then returns the pair dtype.
        auto reduced=tb::zeros({source.size(0),c.hidden},source.options().dtype(tb::kFloat32));
        for (int64_t r=0;r<c.topk;++r)
            reduced=reduced+pairs.select(1,r).to(tb::kFloat32)*selected[1].select(1,r).unsqueeze(-1);
        return (reduced.to(pairs.scalar_type())+shared(source)).reshape(x.sizes());
    };
}

struct Gr {
    Tensor norm,down,up,injection;
    int64_t hidden,streams;
    double eps;
    Gr(const MfqFile& file,const mfq::flash_next::QwenConfig& c,const std::string& p,bool combine=true)
        : norm(dense(file,p+".norm.weight").to(tb::kFloat32)),down(dense(file,p+".down.weight")),
          up(dense(file,p+".up.weight")),hidden(c.hidden),streams(c.streams),eps(c.eps) {
        if (combine) injection=dense(file,p.substr(0,p.size()-4)+".post.inject.weight");
    }
    std::vector<Tensor> pre(const Tensor& x) const {
        return mfq_flash_next::qwen4_gated_residual_pre(x,norm,down,up,
            injection.defined()?std::optional<Tensor>(injection):std::nullopt,hidden,streams,eps);
    }
    Tensor post(const Tensor& branch,const std::vector<Tensor>& inputs) const {
        return mfq_flash_next::qwen4_gated_residual_post(branch,inputs[1],inputs[2],streams);
    }
};

static Linear qwen_ffn(const MfqFile& file,const mfq::flash_next::QwenConfig& c,int i,const std::string& root="model") {
    const auto p=root+".block."+std::to_string(i)+".mlp";
    auto gate_up=routed(file,p+".experts.gate_up.weight",i,c.experts,2*c.moe_width,c.hidden);
    auto down=routed(file,p+".experts.down.weight",i,c.experts,c.hidden,c.moe_width);
    auto router=linear(file,p+".router.weight"),shared_gate=linear(file,p+".shared_expert.router.weight");
    auto sg=linear(file,p+".shared_expert.gate.weight"),su=linear(file,p+".shared_expert.up.weight"),sd=linear(file,p+".shared_expert.down.weight");
    return [gate_up,down,router,shared_gate,sg,su,sd,c](const Tensor& x) {
        auto source=x.reshape({-1,c.hidden}).to(tb::kFloat16);
        auto selected=moe_topk_cuda(router(source).to(tb::kFloat32).contiguous(),c.topk,
            false,false,c.normalize_routes,false,mfq_nullopt,1e-20,1.0);
        auto gu=gate_up(source,selected[0]);
        auto gate=gu.narrow(-1,0,c.moe_width),up=gu.narrow(-1,c.moe_width,c.moe_width);
        auto pairs=down((gate*tb::sigmoid(gate))*up,selected[0]);
        auto reduced=tb::zeros({source.size(0),c.hidden},source.options().dtype(tb::kFloat32));
        for (int64_t r=0;r<c.topk;++r)
            reduced=reduced+pairs.select(1,r).to(tb::kFloat32)*selected[1].select(1,r).unsqueeze(-1);
        auto g=sg(source),u=su(source);
        auto shared=tb::sigmoid(shared_gate(source))*sd((g*tb::sigmoid(g))*u);
        return (reduced.to(pairs.scalar_type())+shared).reshape(x.sizes());
    };
}

static std::vector<int64_t> integers(const MfqFile& file,const std::string& name) {
    const auto& type=file.record(name).dtype;
    MFQ_RUNTIME_CHECK(type=="I64" || type=="I32","Qwen4 PLE metadata must be a dense integer array: ",name);
    auto host=load_dense_gpu(file,name).to(tb::kInt64).contiguous().cpu();
    MFQ_RUNTIME_CHECK(host.dim()==1,"Qwen4 PLE metadata must be a vector");
    return {host.data_ptr<int64_t>(),host.data_ptr<int64_t>()+host.numel()};
}

static std::unique_ptr<mfq::flash_next::Ple> qwen_ple(const MfqFile& file,const mfq::flash_next::QwenConfig& c,const std::string& p) {
    std::vector<Linear> shards;
    int64_t rows=0,width=c.hidden/((c.ngram-1)*c.ngram_heads);
    for (int64_t i=0;i<c.shards;++i) {
        const auto name=p+".ngram.shard."+std::to_string(i)+".weight";
        auto weight=std::make_shared<QuantLinear>(load_quant_linear(file,name));
        // QuantLinear reports logical dimensions regardless of storage format.
        std::vector<int64_t> shape{weight->out(),weight->neuron_len()};
        MFQ_RUNTIME_CHECK(shape.size()==2 && shape[1]==width && shape[0]>0 && (!rows || shape[0]==rows),
            "Qwen4 PLE embedding shard dimensions disagree");
        rows=shape[0];
        shards.push_back([weight](const Tensor& ids) {return quant_embedding_lookup(*weight,ids);});
    }
    mfq::flash_next::NgramEmbedding embedding(std::move(shards),rows,width,c.ngram,c.ngram_heads,c.eos,
        integers(file,p+".ngram.layer_multipliers"),integers(file,p+".ngram.head_offsets"),integers(file,p+".ngram.head_vocab_sizes"));
    mfq::flash_next::PleWeights w{linear(file,p+".key.weight"),linear(file,p+".value.weight"),
        dense(file,p+".key_norm.weight"),dense(file,p+".query_norm.weight"),dense(file,p+".conv_norm.weight"),dense(file,p+".conv.weight")};
    return std::make_unique<mfq::flash_next::Ple>(std::move(embedding),std::move(w),c.hidden,c.streams,c.ngram,c.eps);
}

struct Mhc {
    Tensor function,base,scale;
    explicit Mhc(const MfqFile& file,const std::string& p)
        : function(dense(file,p+".function")),base(dense(file,p+".base")),scale(dense(file,p+".scale")) {}
    std::vector<Tensor> pre(const Tensor& x,const mfq::flash_next::GlmConfig& c) const {
        return mfq_flash_next::glm5_mhc_pre(x,function,base,scale,c.sinkhorn,c.hc_eps,c.eps);
    }
};
} // namespace flash_runtime

struct Glm5NextBlock final : Block {
    using Tensor=mfq_tensor_backend::Tensor;
    mfq::flash_next::GlmConfig config;
    flash_runtime::Mhc attention_hc,ffn_hc;
    Tensor attention_norm,ffn_norm;
    mfq::flash_next::Linear ffn;
    std::unique_ptr<mfq::flash_next::Kda> kda;
    std::unique_ptr<mfq::flash_next::SparseMla> mla;

    Glm5NextBlock(const MfqFile& file,const mfq::flash_next::GlmConfig& c,int i)
        : config(c),attention_hc(file,"model.block."+std::to_string(i)+".attention.mhc.pre"),
          ffn_hc(file,"model.block."+std::to_string(i)+".mlp.mhc.pre") {
        using namespace flash_runtime;
        const auto p="model.block."+std::to_string(i);
        attention_norm=dense(file,p+".attention.norm.weight"); ffn_norm=dense(file,p+".mlp.norm.weight");
        ffn=glm_ffn(file,c,i);
        if (c.layer_types.at(i)=="linear_attention") {
            const auto a=p+".linear_attention";
            mfq::flash_next::KdaWeights w{linear(file,a+".query.weight"),linear(file,a+".key.weight"),
                linear(file,a+".value.weight"),linear(file,a+".beta.weight"),linear(file,a+".gate_a.weight"),
                linear(file,a+".gate_b.weight"),linear(file,a+".output.weight"),
                tb::cat({dense(file,a+".query_conv.weight"),dense(file,a+".key_conv.weight"),dense(file,a+".value_conv.weight")},0),
                dense(file,a+".forget_a.weight"),dense(file,a+".forget_b.weight"),dense(file,a+".dt_bias"),
                dense(file,a+".a"),dense(file,a+".output_norm.weight")};
            kda=std::make_unique<mfq::flash_next::Kda>(std::move(w),c.kda_heads,c.kda_width,c.kda_kernel,c.lower_bound,c.eps);
        } else {
            const auto a=p+".attention";
            mfq::flash_next::MlaWeights w{linear(file,a+".query_a.weight"),linear(file,a+".key_value_a.weight"),
                linear(file,a+".query_b.weight"),linear(file,a+".output.weight"),linear(file,a+".indexer.query.weight"),
                linear(file,a+".indexer.key.weight"),linear(file,a+".indexer.score.weight"),
                headwise(routed(file,a+".latent.query_embedding.weight",i,c.heads,c.latent,c.nope),c.heads,c.latent),
                headwise(routed(file,a+".latent.output_unembedding.weight",i,c.heads,c.value_width,c.latent),c.heads,c.value_width),
                dense(file,a+".query_a_norm.weight"),dense(file,a+".key_value_a_norm.weight"),
                dense(file,a+".indexer.key_norm.weight"),dense(file,a+".indexer.key_norm.bias"),
                dense(file,a+".indexer.pool.gate"),dense(file,a+".indexer.pool.position")};
            mfq::flash_next::MlaConfig mc{c.heads,c.nope,c.latent,c.value_width,c.index_heads,c.index_width,c.pool,c.budget,c.maximum,c.tail,c.eps};
            mla=std::make_unique<mfq::flash_next::SparseMla>(std::move(w),mc);
        }
    }
    void reset(int64_t) override { if (kda) kda->reset(); if (mla) mla->reset(); }
    bool supports_speculation() const noexcept override {return true;}
    void commit_speculative() override { if (kda) kda->commit(); }
    void rollback_speculative(int64_t keep) override { if (kda) kda->rollback(); if (mla) mla->truncate(keep); }
    Tensor execute(const Tensor& x,int64_t position,int64_t confirmed=0) {
        if (mla) MFQ_RUNTIME_CHECK(mla->position()==position,"GLM MLA/model cache positions diverged");
        auto first=attention_hc.pre(x,config);
        auto branch=mfq::flash_next::rms_norm(first[2],attention_norm,config.eps);
        branch=kda ? kda->forward(branch,true,confirmed) : mla->forward(branch,true);
        auto hidden=mfq_flash_next::glm5_mhc_post(branch,x,first[0],first[1]);
        auto second=ffn_hc.pre(hidden,config);
        branch=ffn(mfq::flash_next::rms_norm(second[2],ffn_norm,config.eps));
        return mfq_flash_next::glm5_mhc_post(branch,hidden,second[0],second[1]);
    }
    Tensor forward(Tensor x,Tensor,int64_t position,const MfqOptional<Tensor>&,
        const Config&,const RopeCache&,const MfqOptional<Tensor>& = mfq_nullopt,
        const MfqOptional<Tensor>& mask = mfq_nullopt) override {
        MFQ_RUNTIME_CHECK(!mask.has_value(),"GLM Flash-Next requires its causal unpadded attention geometry");
        return execute(x,position);
    }
    Tensor forward_context(Tensor x,const Context& context,const Config&,const RopeCache&) override {
        return execute(x,context.cache_position,context.confirmed_prefix);
    }
};

struct Qwen4Block final : Block {
    using Tensor=mfq_tensor_backend::Tensor;
    mfq::flash_next::QwenConfig config;
    flash_runtime::Gr attention_gr,ffn_gr;
    mfq::flash_next::Linear ffn;
    std::unique_ptr<mfq::flash_next::Gdn> gdn;
    std::unique_ptr<mfq::flash_next::Qsa> qsa;
    std::unique_ptr<mfq::flash_next::Ple> ple;
    Qwen4Block(const MfqFile& file,const mfq::flash_next::QwenConfig& c,int i,const std::string& root="model")
        : config(c),attention_gr(file,c,root+".block."+std::to_string(i)+".attention.mhc.pre"),
          ffn_gr(file,c,root+".block."+std::to_string(i)+".mlp.mhc.pre"),ffn(flash_runtime::qwen_ffn(file,c,i,root)) {
        using namespace flash_runtime;
        const auto p=root+".block."+std::to_string(i);
        if (root=="model" && c.layer_types.at(i)=="linear_attention") {
            const auto a=p+".linear_attention";
            mfq::flash_next::GdnWeights w{linear(file,a+".qkv.weight"),linear(file,a+".gate.weight"),
                linear(file,a+".alpha.weight"),linear(file,a+".beta.weight"),linear(file,a+".output.weight"),
                dense(file,a+".conv.weight"),dense(file,a+".dt_bias"),dense(file,a+".a"),dense(file,a+".norm.weight")};
            gdn=std::make_unique<mfq::flash_next::Gdn>(std::move(w),c.key_heads,c.value_heads,c.linear_width,c.kernel,c.eps,c.silu_gate);
        } else {
            const auto a=p+".attention";
            mfq::flash_next::QsaWeights w{linear(file,a+".query.weight"),linear(file,a+".key.weight"),linear(file,a+".value.weight"),
                linear(file,a+".output.weight"),linear(file,a+".indexer.query_key.weight"),
                dense(file,a+".query_norm.weight"),dense(file,a+".key_norm.weight"),dense(file,a+".indexer.query_norm.weight"),dense(file,a+".indexer.key_norm.weight")};
            mfq::flash_next::QsaConfig qc{c.heads,c.kv_heads,c.width,c.index_heads,c.index_width,c.pool,c.budget,c.maximum,c.eps};
            auto rotary=std::make_shared<mfq::flash_next::Rotary>(c.rotary,c.maximum,c.rope_base,c.sections,c.interleaved);
            qsa=std::make_unique<mfq::flash_next::Qsa>(std::move(w),qc,std::move(rotary));
        }
        if (root=="model" && std::find(c.ple_layers.begin(),c.ple_layers.end(),i+1)!=c.ple_layers.end()) ple=qwen_ple(file,c,p+".position_embedding");
    }
    void reset(int64_t) override {if (gdn) gdn->reset();if (qsa) qsa->reset();if (ple) ple->reset();}
    bool supports_speculation() const noexcept override {return true;}
    void commit_speculative() override {if (gdn) gdn->commit();if (ple) ple->commit();}
    void rollback_speculative(int64_t keep) override {if (gdn) gdn->rollback();if (qsa) qsa->truncate(keep);if (ple) ple->rollback();}
    Tensor execute(Tensor x,const Tensor& ids,const Tensor& positions,const Tensor& full_positions,int64_t confirmed=0) {
        if (ple) x=x+ple->forward(x,ids,true,confirmed);
        auto first=attention_gr.pre(x);
        auto branch=gdn?gdn->forward(first[0],true,confirmed):qsa->forward(first[0],positions,full_positions,true);
        x=attention_gr.post(branch,first);
        auto second=ffn_gr.pre(x);
        return ffn_gr.post(ffn(second[0]),second);
    }
    Tensor forward(Tensor,Tensor,int64_t,const MfqOptional<Tensor>&,
        const Config&,const RopeCache&,const MfqOptional<Tensor>& = mfq_nullopt,
        const MfqOptional<Tensor>& = mfq_nullopt) override {
        throw std::runtime_error("Qwen4 block requires the unified model position/PLE input lifecycle");
    }
    Tensor forward_context(Tensor x,const Context& context,const Config&,const RopeCache&) override {
        return execute(std::move(x),context.token_ids,context.positions,
            context.full_positions,context.confirmed_prefix);
    }
};
