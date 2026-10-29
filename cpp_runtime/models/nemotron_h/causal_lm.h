#pragma once
#include "config.h"
#include "models/common/weight_loading.h"
#include "models/common/causal_model.h"
#include "models/common/predictor.h"
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::models::nemotron_h {

// One pre-normed Mamba-2 mixer. The fused projection keeps z, xBC and dt in
// one matrix; the graph slices them instead of splitting the checkpoint
// tensor, which would requantize the narrow time-step tail.
template <class Block, class Loader>
void load_linear_attention(Block& block, Loader& ops, const Config& config, int layer,
                           std::string_view tensor_root = "model") {
    const auto layer_prefix = std::string(tensor_root) + ".block." + std::to_string(layer) + ".";
    const auto linear = layer_prefix + "linear_attention.";
    block.layer = layer;
    block.rms_norm_eps = config.rms_norm_eps;
    block.mamba_heads = config.mamba_heads;
    block.mamba_head_dim = config.mamba_head_dim;
    block.ssm_state_size = config.ssm_state_size;
    block.mamba_groups = config.mamba_groups;
    block.conv_kernel = config.conv_kernel;
    block.time_step_limit_lo = config.time_step_limit_lo;
    block.time_step_limit_hi = config.time_step_limit_hi;
    block.mixer_norm = ops.dense(layer_prefix + "attention.norm.weight");
    block.in_proj = ops.linear(linear + "in_projection.weight");
    block.conv_weight = ops.dense(linear + "conv.weight");
    if (ops.has(linear + "conv.bias"))
        block.conv_bias = ops.dense(linear + "conv.bias");
    block.a_log = ops.dense(linear + "a");
    block.d = ops.dense(linear + "d");
    block.dt_bias = ops.dense(linear + "dt_bias");
    block.scan_norm = ops.dense(linear + "norm.weight");
    block.out_proj = ops.linear(linear + "output.weight");
    const auto shape = [&](const auto& value) { return ops.shape(value); };
    const std::vector<int64_t> expected_convolution{config.mamba_conv_width(), 1,
                                                    config.conv_kernel};
    if (block.in_proj.out() != config.mamba_projection_width() ||
        block.in_proj.neuron_len() != config.hidden_size ||
        shape(block.conv_weight) != expected_convolution ||
        ops.elements(block.scan_norm) != config.mamba_inner_size() ||
        block.out_proj.out() != config.hidden_size ||
        block.out_proj.neuron_len() != config.mamba_inner_size() ||
        ops.elements(block.mixer_norm) != config.hidden_size ||
        ops.elements(block.a_log) != config.mamba_heads ||
        ops.elements(block.d) != config.mamba_heads ||
        ops.elements(block.dt_bias) != config.mamba_heads ||
        (config.conv_bias && ops.elements(block.conv_bias) != config.mamba_conv_width())) {
        throw std::runtime_error("Nemotron-H mamba tensor shapes disagree with config at layer " +
                                 std::to_string(layer));
    }
}

// One pre-normed NoPE attention mixer. The layer keeps the shared full
// attention block; only the block's FFN tail is absent.
template <class Block, class Loader>
void load_attention(Block& block, Loader& ops, const Config& config, int layer,
                    std::string_view tensor_root = "model") {
    const auto layer_prefix = std::string(tensor_root) + ".block." + std::to_string(layer) + ".";
    models::load_attention_mixer(block, ops, config, layer, layer_prefix, 0.0, false);
}

// One pre-normed latent MoE (or dense relu-squared MLP) mixer. The router and
// the shared expert read the residual stream; only the routed experts run in
// the projected latent space.
template <class Block, class Loader>
void load_moe(Block& block, Loader& ops, const Config& config, int layer,
              std::string_view tensor_root = "model") {
    if (config.num_routed_experts <= 0 || config.moe_latent_size <= 0)
        throw std::runtime_error("Nemotron-H MoE layer has no routed-expert geometry");
    const auto layer_prefix = std::string(tensor_root) + ".block." + std::to_string(layer) + ".";
    const auto mlp = layer_prefix + "mlp.";
    block.layer = layer;
    block.rms_norm_eps = config.rms_norm_eps;
    block.ffn_norm = ops.dense(layer_prefix + "attention.norm.weight");
    auto& ffn = block.ffn;
    ffn = models::load_moe_weights<typename Loader::Ffn>(ops, mlp, layer, true, true, true);
    ffn.moe_top_k = static_cast<int>(config.num_experts_per_tok);
    ffn.moe_use_sigmoid = true;
    ffn.moe_use_sqrt_softplus = false;
    ffn.moe_normalize = config.norm_topk_prob;
    ffn.moe_delayed_softmax = false;
    ffn.moe_router_scale = config.routed_scaling_factor;
    ffn.moe_shared_ungated = true;
    const bool dense_bank = ffn.moe_dense_up.defined() || ffn.moe_dense_down.defined();
    const auto rows = static_cast<int64_t>(config.num_routed_experts);
    bool expert_shapes =
        dense_bank ? ffn.moe_dense_up.dim() == 3 &&
                         ffn.moe_dense_up.size(0) == rows &&
                         ffn.moe_dense_up.size(1) == config.moe_intermediate_size &&
                         ffn.moe_dense_up.size(2) == config.moe_latent_size &&
                         ffn.moe_dense_down.dim() == 3 &&
                         ffn.moe_dense_down.size(0) == rows &&
                         ffn.moe_dense_down.size(1) == config.moe_latent_size &&
                         ffn.moe_dense_down.size(2) == config.moe_intermediate_size
                   : ffn.moe_up.n_experts == config.num_routed_experts &&
                         ffn.moe_up.neuron_len == config.moe_latent_size &&
                         ffn.moe_up.out_per_expert == config.moe_intermediate_size &&
                         ffn.moe_down.n_experts == config.num_routed_experts &&
                         ffn.moe_down.neuron_len == config.moe_intermediate_size &&
                         ffn.moe_down.out_per_expert == config.moe_latent_size;
    if constexpr(requires { ffn.moe_tq; }) {
        if(ffn.moe_tq) {
            const auto& tq=*ffn.moe_tq;
            expert_shapes=tq.up && tq.down && !tq.gate &&
                tq.up->experts==config.num_routed_experts && tq.down->experts==config.num_routed_experts &&
                tq.up->out==config.moe_intermediate_size && tq.up->input==config.moe_latent_size &&
                tq.down->out==config.moe_latent_size && tq.down->input==config.moe_intermediate_size;
        }
    }
    bool router_shape=ffn.moe_router.dim()==2 &&
        ffn.moe_router.size(0)==config.num_routed_experts &&
        ffn.moe_router.size(1)==config.hidden_size;
    if constexpr(requires { ffn.moe_router_projection; })
        if(ffn.moe_router_projection)router_shape=ffn.moe_router_projection->out()==config.num_routed_experts &&
            ffn.moe_router_projection->neuron_len()==config.hidden_size;
    if (!expert_shapes || !router_shape ||
        ops.elements(ffn.moe_router_bias) != config.num_routed_experts ||
        ffn.moe_latent_in.out() != config.moe_latent_size ||
        ffn.moe_latent_in.neuron_len() != config.hidden_size ||
        ffn.moe_latent_out.out() != config.hidden_size ||
        ffn.moe_latent_out.neuron_len() != config.moe_latent_size ||
        ffn.shared->gate_up.outs.size() != 1 ||
        ffn.shared->gate_up.outs[0] != config.moe_shared_intermediate_size ||
        ffn.shared->down.out() != config.hidden_size ||
        ffn.shared->down.neuron_len() != config.moe_shared_intermediate_size ||
        ops.elements(block.ffn_norm) != config.hidden_size) {
        throw std::runtime_error("Nemotron-H MoE tensor shapes disagree with config at layer " +
                                 std::to_string(layer));
    }
}

// The dense relu-squared MLP block of the same family shares the canonical
// non-gated MLP graph, so it needs no separate kernel.
template <class Block, class Loader>
void load_dense_mlp(Block& block, Loader& ops, const Config& config, int layer,
                    std::string_view tensor_root = "model") {
    const auto layer_prefix = std::string(tensor_root) + ".block." + std::to_string(layer) + ".";
    const auto mlp = layer_prefix + "mlp.";
    block.layer = layer;
    block.rms_norm_eps = config.rms_norm_eps;
    block.ffn_norm = ops.dense(layer_prefix + "attention.norm.weight");
    auto& ffn = block.ffn;
    ffn.down = ops.linear(mlp + "down.weight");
    ffn.gate_up = ops.gate_up({mlp + "up.weight"}, ffn.down);
    ffn.activation = models::MlpActivation::relu_squared;
    ops.workspace(ffn);
    if (ffn.gate_up.outs.size() != 1 || ffn.gate_up.outs[0] != config.intermediate_size ||
        ffn.down.out() != config.hidden_size ||
        ffn.down.neuron_len() != config.intermediate_size) {
        throw std::runtime_error("Nemotron-H dense MLP shapes disagree with config at layer " +
                                 std::to_string(layer));
    }
}

enum class PredictorNorm { embedding, hidden, output };

// Nemotron-H's own multi-token-prediction head, matching the three reference
// implementations (llama.cpp src/models/nemotron-h-moe.cpp, vLLM and SGLang
// nemotron_h_mtp.py); no Hugging Face implementation builds these weights:
//   e = enorm(embed(next_ids)); h = hnorm(previous_hidden)
//   x = eh_proj(cat([e, h], -1))                         # embedding first
//   x = x + mixer(mtp_layers_block_type[0])(norm0(x))    # attention, NoPE
//   x = x + mixer(mtp_layers_block_type[1])(norm1(x))    # moe, latent relu^2
//   logits = lm_head(final_layernorm(x))
// Two details differ from the DeepSeek-V3 convention:
//   * previous_hidden is the backbone hidden state *after* output_norm, so the
//     target model hands the predictor its already-normalized hidden (never
//     normalize it twice);
//   * for further draft steps the head's own final_layernorm output is passed
//     back as previous_hidden unchanged, so the backbone norm is not applied
//     again and only hnorm renormalizes it.
//   * the final norm is RMSNorm (llama.cpp uses LayerNorm; vLLM and SGLang use
//     RMSNorm, which is also Megatron's configured normalization).
// The single predictor stage owns one attention history; the MoE sub-block is
// stateless. `next_ids` are the tokens after each hidden state, so the
// predictor history is one row shorter than the target prompt.
template <class Ops>
auto mtp_predictor(Ops &ops, typename Ops::Tensor hidden, typename Ops::Tensor ids,
                   typename Ops::Tensor positions, int64_t depth, bool cache) {
    auto &model = ops.model;
    const auto &config = model.config;
    require_model(
        ops.rank(hidden) == 3 && ops.rank(ids) == 2 && ops.size(hidden, 0) == ops.size(ids, 0) &&
            ops.size(hidden, 1) == ops.size(ids, 1) && ops.size(hidden, 2) == config.hidden_size &&
            ops.size(hidden, 1) > 0,
        "Nemotron-H MTP inputs must be matching [B,T] hidden states and [B,T] next-token IDs");
    require_model(config.num_nextn_predict_layers == 1 && !config.mtp_mixers.empty(),
                  "Nemotron-H MTP requires one declared predictor stage");
    const auto batch = ops.size(hidden, 0), tokens = ops.size(hidden, 1);
    return mfq::models::projected_predictor(
        model, batch, tokens, depth, cache, [&](int64_t next) { model.reset(next); },
        [&] { return ops.embed(std::move(ids), hidden); },
        [&](mfq::models::PredictorCursor cursor) {
            return ops.positions(std::move(positions), cursor.start, tokens);
        },
        [&](typename Ops::Tensor embeds, const typename Ops::Tensor &) {
            return ops.normalize(std::move(embeds), PredictorNorm::embedding);
        },
        [&] { return ops.normalize(hidden, PredictorNorm::hidden); },
        [&](typename Ops::Tensor e, typename Ops::Tensor h) {
            return ops.fuse(std::move(e), std::move(h));
        },
        [&](typename Ops::Tensor x, int64_t stage, const typename Ops::Tensor &pos) {
            return ops.mixer_blocks(std::move(x), stage, pos);
        },
        [&](const typename Ops::Tensor &x) {
            return ops.normalize(x, PredictorNorm::output);
        },
        [&](int64_t, const typename Ops::Tensor &) {});
}

template <class Backend> struct CausalLm : models::CausalModelBase<Backend, CausalLm<Backend>> {
    using Tensor = typename Backend::Tensor;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "nemotron_h"; }
    template <class Graph, class Source>
    void adapter_load_config(std::string_view payload, const Graph& graph, const Source& source) {
        auto& config = this->config;
        auto& metadata = this->metadata;

        config = Config::from_json(payload, graph);
        metadata.vocab_size = config.vocab_size;
        metadata.hidden_size = config.hidden_size;
        metadata.num_hidden_layers = config.num_hidden_layers;
        metadata.num_attention_heads = config.num_attention_heads;
        metadata.num_key_value_heads = config.num_key_value_heads;
        metadata.head_dim = config.head_dim;
        metadata.max_position_embeddings = config.max_position_embeddings;
        metadata.rotary_dim = 0;
        metadata.num_experts = config.num_routed_experts;
        metadata.rope_base = config.rope_base;
        metadata.rms_norm_eps = config.rms_norm_eps;
        metadata.norm_weight_offset = 0.0;
        metadata.tie_word_embeddings = config.tie_word_embeddings;
        metadata.no_rope = true;
        metadata.model_type = config.model_type;
        metadata.layer_types = config.layer_types;
        this->adapter_validate_components(graph);
    }
    void adapter_set_max_position_embeddings(int64_t value) {
        this->config.max_position_embeddings = value;
    }
    bool adapter_supports_speculation() const noexcept { return true; }
};

} // namespace mfq::models::nemotron_h
