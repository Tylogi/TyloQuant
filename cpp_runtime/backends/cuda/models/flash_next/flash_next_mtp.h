#pragma once

// Appended predictors share the target's embedding and output projection.
// Each depth selects one predictor layer with its own attention/position cache.
struct CudaFlashNextMtp final : CudaMtpModule {
    using Tensor=mfq_tensor_backend::Tensor;
    using Linear=mfq::flash_next::Linear;
    struct GlmLayer {
        Tensor attention_norm,ffn_norm;
        std::unique_ptr<mfq::flash_next::SparseMla> attention;
        Linear ffn;
    };
    Config c;
    Tensor embedding_norm,hidden_norm,output_norm;
    Linear fusion,embedding_fusion,hidden_fusion;
    std::unique_ptr<flash_runtime::Gr> final_mixer;
    std::vector<std::unique_ptr<Qwen4Block>> qwen_layers;
    std::vector<GlmLayer> glm_layers;
    std::vector<Tensor> positions;
    std::vector<int64_t> lengths;
    int64_t batch=0;

    static std::optional<CudaFlashNextMtp> load_if_present(const MfqFile& file,const Config& main) {
        using namespace flash_runtime;
        bool any=false;
        for (const auto& entry:file.records) if (entry.first.rfind("predictor.",0)==0) {any=true;break;}
        for (const auto& entry:file.canonical_aliases) if (entry.first.rfind("predictor.",0)==0) {any=true;break;}
        const auto layers=main.is_qwen4()?main.qwen4->predictor_layers:main.glm5_next->predictor_layers;
        if (!file.has_record("predictor.embedding_norm.weight") || layers<=0) {
            MFQ_RUNTIME_CHECK(!any,"Flash-Next MFQ contains an incomplete or undeclared MTP head");
            return std::nullopt;
        }
        CudaFlashNextMtp result;result.c=main;
        result.embedding_norm=dense(file,"predictor.embedding_norm.weight").to(tb::kFloat32);
        result.hidden_norm=dense(file,"predictor.hidden_norm.weight").to(tb::kFloat32);
        if (main.is_qwen4()) {
            const auto& cfg=*main.qwen4;
            result.embedding_fusion=linear(file,"predictor.fusion.embedding.weight");
            result.hidden_fusion=linear(file,"predictor.fusion.hidden.weight");
            result.final_mixer=std::make_unique<Gr>(file,cfg,"predictor.mhc.pre",false);
            for (int64_t i=0;i<layers;++i) result.qwen_layers.push_back(std::make_unique<Qwen4Block>(file,cfg,int(i),"predictor"));
        } else {
            const auto& cfg=*main.glm5_next;
            result.fusion=linear(file,"predictor.fusion.weight");
            result.output_norm=dense(file,"predictor.output_norm.weight");
            for (int64_t i=0;i<layers;++i) {
                const auto p="predictor.block."+std::to_string(i),a=p+".attention";
                GlmLayer layer;layer.attention_norm=dense(file,a+".norm.weight");layer.ffn_norm=dense(file,p+".mlp.norm.weight");
                layer.ffn=glm_ffn(file,cfg,int(i),"predictor");
                mfq::flash_next::MlaWeights w{linear(file,a+".query_a.weight"),linear(file,a+".key_value_a.weight"),
                    linear(file,a+".query_b.weight"),linear(file,a+".output.weight"),linear(file,a+".indexer.query.weight"),
                    linear(file,a+".indexer.key.weight"),linear(file,a+".indexer.score.weight"),
                    headwise(routed(file,a+".latent.query_embedding.weight",int(i),cfg.heads,cfg.latent,cfg.nope),cfg.heads,cfg.latent),
                    headwise(routed(file,a+".latent.output_unembedding.weight",int(i),cfg.heads,cfg.value_width,cfg.latent),cfg.heads,cfg.value_width),
                    dense(file,a+".query_a_norm.weight"),dense(file,a+".key_value_a_norm.weight"),
                    dense(file,a+".indexer.key_norm.weight"),dense(file,a+".indexer.key_norm.bias"),
                    dense(file,a+".indexer.pool.gate"),dense(file,a+".indexer.pool.position")};
                mfq::flash_next::MlaConfig mc{cfg.heads,cfg.nope,cfg.latent,cfg.value_width,cfg.index_heads,cfg.index_width,
                    cfg.pool,cfg.budget,cfg.maximum,cfg.tail,cfg.eps};
                layer.attention=std::make_unique<mfq::flash_next::SparseMla>(std::move(w),mc);
                result.glm_layers.push_back(std::move(layer));
            }
        }
        const auto hidden_width=main.hidden_size*(main.is_qwen4()?main.qwen4->streams:1);
        MFQ_RUNTIME_CHECK(result.embedding_norm.dim()==1 && result.embedding_norm.numel()==main.hidden_size &&
            result.hidden_norm.dim()==1 && result.hidden_norm.numel()==hidden_width,
            "Flash-Next MTP normalization width disagrees with backbone");
        if (!main.is_qwen4()) MFQ_RUNTIME_CHECK(result.output_norm.numel()==main.hidden_size,"GLM MTP output norm width mismatch");
        result.positions.resize(layers);result.lengths.resize(layers,0);
        return result;
    }
    void reset(int64_t next_batch=1) override {
        MFQ_RUNTIME_CHECK(next_batch>0,"Flash-Next MTP batch must be positive");
        for (auto& layer:qwen_layers) layer->reset(next_batch);
        for (auto& layer:glm_layers) layer.attention->reset();
        for (auto& pos:positions) pos={};
        std::fill(lengths.begin(),lengths.end(),0);batch=next_batch;
    }
    std::pair<Tensor,Tensor> evaluate(Model& main,const Tensor& hidden,const Tensor& ids,int64_t depth=0,
        bool cache=true,const Tensor& supplied_positions={},const Tensor& supplied_embeddings={}) {
        namespace tb=mfq_tensor_backend;
        MFQ_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)>0 && ids.size(1)>0 && hidden.dim()==3 &&
            hidden.size(0)==ids.size(0) && hidden.size(1)==ids.size(1) && hidden.size(2)==hidden_norm.numel(),
            "Flash-Next MTP input geometry mismatch");
        const auto b=ids.size(0),t=ids.size(1),h=c.hidden_size,n=int64_t(lengths.size());
        const auto layer=(depth%n+n)%n;
        if (cache && batch && batch!=b) reset(b);
        const auto start=cache?lengths[layer]:0;
        MFQ_RUNTIME_CHECK(start+t<=c.max_position_embeddings,"Flash-Next MTP history exceeds context capacity");
        auto embeds=supplied_embeddings.defined()?supplied_embeddings:main.embed_forward(ids);
        MFQ_RUNTIME_CHECK(embeds.sizes().vec()==std::vector<int64_t>({b,t,h}),"Flash-Next MTP embedding shape mismatch");
        auto current=supplied_positions.defined()?supplied_positions:tb::arange(start,start+t,ids.options().dtype(tb::kInt32));
        Tensor output,multi;
        if (c.is_qwen4()) {
            const auto& cfg=*c.qwen4;
            if (current.dim()==1) current=current.reshape({1,1,t}).expand({3,1,t}).contiguous();
            else if (current.dim()==2) current=current.unsqueeze(1);
            if (current.dim()==3 && current.size(0)==4) current=current.narrow(0,1,3);
            MFQ_RUNTIME_CHECK(current.dim()==3 && current.size(0)==3 && current.size(-1)==t &&
                (current.size(1)==1 || current.size(1)==b),"Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
            current=current.to(tb::kInt32).expand({3,b,t}).contiguous();
            auto full=cache && positions[layer].defined()?tb::cat({positions[layer],current},-1):current;
            auto e=embedding_fusion(mfq::flash_next::rms_norm(embeds,embedding_norm+1,cfg.eps));
            auto streams=hidden_fusion(mfq::flash_next::rms_norm(hidden,hidden_norm+1,cfg.eps).reshape({b,t,cfg.streams,h}));
            auto x=(streams+e.unsqueeze(-2)).reshape({b,t,cfg.streams*h});
            auto& block=*qwen_layers[layer];
            auto first=block.attention_gr.pre(x);
            auto branch=block.qsa->forward(first[0],current,full,cache);
            x=block.attention_gr.post(branch,first);
            auto second=block.ffn_gr.pre(x);
            multi=block.ffn_gr.post(block.ffn(second[0]),second);
            output=final_mixer->pre(multi)[0];
            if (cache) positions[layer]=full;
        } else {
            const auto eps=c.glm5_next->eps;
            MFQ_RUNTIME_CHECK((current.dim()==1 || current.dim()==2) && current.size(-1)==t &&
                (current.dim()==1 || current.size(0)==b),"GLM MTP positions require [T] or [B,T]");
            current=current.to(tb::kInt32);
            auto mask=(current==0).reshape({current.dim()==1?1:b,t,1});
            auto e=tb::where(mask,tb::zeros_like(embeds),embeds);
            auto x=fusion(tb::cat({mfq::flash_next::rms_norm(e,embedding_norm,eps),mfq::flash_next::rms_norm(hidden,hidden_norm,eps)},-1));
            auto& block=glm_layers[layer];
            auto attention=block.attention->forward(mfq::flash_next::rms_norm(x,block.attention_norm,eps),cache);
            auto residual=x.to(tb::kFloat32)+attention.to(tb::kFloat32);
            const auto dtype=x.scalar_type()==tb::kFloat32?tb::kFloat32:tb::kFloat16;
            auto branch=mfq::flash_next::rms_norm(residual,block.ffn_norm,eps).to(dtype);
            multi=residual+block.ffn(branch).to(tb::kFloat32);
            output=mfq::flash_next::rms_norm(multi,output_norm,eps).to(dtype);
        }
        if (cache) {batch=b;lengths[layer]=start+t;}
        return {output,multi};
    }
    Tensor forward(Model& main,Tensor hidden,Tensor ids) override {return evaluate(main,hidden,ids).first;}
    bool teacher_forced_prompt_prime() const noexcept override {return false;}
    bool preserve_output_dtype() const noexcept override {return true;}
};
