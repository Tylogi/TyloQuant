#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::flash_next {
// Mirrors architectures/flash_next.py. Semantic IDs, layer schedules and
// NoPE/mHC requirements are validated before opening projection weights.
struct GlmConfig {
    int64_t vocab, hidden, intermediate, layers, maximum;
    int64_t streams, sinkhorn, kda_heads, kda_width, kda_kernel;
    int64_t query_rank, latent, heads, nope, value_width;
    int64_t index_heads, index_width, budget, pool;
    int64_t experts, topk, moe_intermediate, shared_experts, predictor_layers;
    double eps, hc_eps, lower_bound, router_scale, swiglu_limit;
    bool tail, normalize_routes, tied_embeddings;
    std::vector<std::string> layer_types, mlp_types;

    static GlmConfig parse(const nlohmann::json& outer) {
        const auto& text = outer.contains("text_config") ? outer.at("text_config") : outer;
        const auto outer_type = outer.value("model_type", std::string{});
        if ((outer_type != "glm5_next" && outer_type != "glm5_next_text") ||
            text.value("model_type", outer_type) != "glm5_next_text")
            throw std::runtime_error("expected glm5_next/glm5_next_text config");
        const auto positive = [](const nlohmann::json& j, const char* key) {
            const auto& v = j.at(key);
            if (!v.is_number_integer() || v.get<int64_t>() <= 0)
                throw std::runtime_error(std::string("invalid GLM positive integer: ") + key);
            return v.get<int64_t>();
        };
        const auto finite = [](const nlohmann::json& j, const char* key, double def) {
            const double value = j.value(key, def);
            if (!std::isfinite(value)) throw std::runtime_error(std::string("nonfinite GLM config: ") + key);
            return value;
        };
        GlmConfig c{};
        c.vocab=positive(text,"vocab_size"); c.hidden=positive(text,"hidden_size");
        c.intermediate=positive(text,"intermediate_size"); c.layers=positive(text,"num_hidden_layers");
        c.maximum=positive(text,"max_position_embeddings");
        c.layer_types=text.at("layer_types").get<std::vector<std::string>>();
        c.mlp_types=text.at("mlp_layer_types").get<std::vector<std::string>>();
        const auto indexers=text.at("indexer_types").get<std::vector<std::string>>();
        if (c.layer_types.size()!=size_t(c.layers) || c.mlp_types.size()!=size_t(c.layers) || indexers.size()!=size_t(c.layers))
            throw std::runtime_error("GLM layer schedules must describe every layer");
        std::set<int64_t> kda, full;
        for (int64_t i=0;i<c.layers;++i) {
            if (c.layer_types[i]=="linear_attention") kda.insert(i);
            else if (c.layer_types[i]=="deepseek_sparse_attention") {
                full.insert(i);
                if (indexers[i]!="full") throw std::runtime_error("GLM sparse layers require full indexers");
            } else throw std::runtime_error("unknown GLM attention layer type");
            if ((c.mlp_types[i]!="dense" && c.mlp_types[i]!="sparse") ||
                (indexers[i]!="full" && indexers[i]!="shared"))
                throw std::runtime_error("unknown GLM MLP/indexer layer type");
        }
        const auto& linear=text.at("linear_attn_config");
        if (linear.at("kda_layers").get<std::set<int64_t>>()!=kda ||
            linear.at("full_attn_layers").get<std::set<int64_t>>()!=full)
            throw std::runtime_error("GLM KDA/sparse layer schedules disagree");
        if (!text.value("mhc",false) || !text.value("mla_use_nope",false) ||
            text.value("qk_rope_head_dim",0)!=0 || text.value("hidden_act",std::string("silu"))!="silu" ||
            text.value("attention_bias",false) || text.value("scoring_func",std::string("sigmoid"))!="sigmoid" ||
            text.value("n_group",1)!=1 || text.value("topk_group",1)!=1)
            throw std::runtime_error("unsupported GLM mHC/NoPE/activation/router semantics");
        c.streams=positive(text,"hc_mult"); c.sinkhorn=positive(text,"hc_sinkhorn_iters");
        c.kda_heads=positive(linear,"num_heads"); c.kda_width=positive(linear,"head_dim");
        c.kda_kernel=positive(linear,"short_conv_kernel_size");
        if (c.kda_kernel<2) throw std::runtime_error("GLM causal convolution requires kernel >= 2");
        c.query_rank=positive(text,"q_lora_rank"); c.latent=positive(text,"kv_lora_rank");
        c.heads=positive(text,"num_attention_heads"); c.nope=positive(text,"qk_nope_head_dim");
        c.value_width=positive(text,"v_head_dim"); c.index_heads=positive(text,"index_n_heads");
        c.index_width=positive(text,"index_head_dim"); c.budget=positive(text,"index_topk");
        c.pool=positive(text,"index_kpool");
        if (c.budget%c.pool) throw std::runtime_error("GLM top-k budget must divide pool size");
        c.experts=positive(text,"n_routed_experts"); c.topk=positive(text,"num_experts_per_tok");
        c.moe_intermediate=positive(text,"moe_intermediate_size"); c.shared_experts=positive(text,"n_shared_experts");
        c.predictor_layers=text.value("num_nextn_predict_layers",int64_t(0));
        if (c.topk>std::min<int64_t>(16,c.experts) || c.experts>4096 || c.predictor_layers<0)
            throw std::runtime_error("GLM expert/predictor counts exceed published routing contract");
        c.eps=finite(text,"rms_norm_eps",1e-5); c.hc_eps=finite(text,"hc_eps",1e-6);
        c.lower_bound=finite(linear,"gate_lower_bound",-5.0);
        c.router_scale=finite(text,"routed_scaling_factor",1.0);
        c.swiglu_limit=finite(text,"swiglu_limit",0.0);
        if (c.eps<=0 || c.hc_eps<=0 || c.swiglu_limit<0) throw std::runtime_error("invalid GLM epsilon/clip limit");
        c.tail=text.value("index_kpool_always_select_tail",false);
        c.normalize_routes=text.value("norm_topk_prob",true);
        c.tied_embeddings=text.value("tie_word_embeddings",false);
        return c;
    }
};
} // namespace mfq::flash_next
