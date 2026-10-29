#pragma once

#include "model_config.h"
#include "gated_mlp.h"
#include <memory>
#include <string>
#include <vector>

namespace mfq::models {

enum class ExpertProjection { gate, up, gate_up, down };

// Canonical parameter relationships are shared; the loader materializes native
// weights and chooses packing, projection fusion and device placement.
template <class Ffn, class Loader>
Ffn load_dense_ffn(Loader& ops, const ModelConfig& config, const std::string& prefix) {
    Ffn result;
    const auto down = prefix + "down.weight", gate = prefix + "gate.weight",
               up = prefix + "up.weight";
    result.down = ops.linear(down);
    result.gate_up = ops.gate_up({gate, up}, result.down);
    ops.important_neurons(config.hidden_size, config.intermediate_size, result, down, gate, up);
    ops.workspace(result);
    return result;
}

// Routed banks may be gated (gate/up or fused gate_up) or ungated (up only).
// An ungated expert's activation is relu^2, never a gate projection alias.
template <class Ffn, class Loader>
Ffn load_moe_weights(Loader& ops, const std::string& prefix, int layer,
                    bool router_bias_required = false, bool ungated_experts = false,
                    bool ungated_shared = false) {
    const auto gate_up = prefix + "experts.gate_up.weight", gate = prefix + "experts.gate.weight",
               up = prefix + "experts.up.weight", down = prefix + "experts.down.weight";
    const bool has_up = ops.has(up), has_down = ops.has(down);
    const bool has_gate = ops.has(gate), has_gate_up = ops.has(gate_up);
    Ffn result;
    result.is_moe = true;
    result.moe_layer = layer;
    if (ungated_experts) {
        if (!has_up || !has_down || has_gate || has_gate_up)
            throw std::runtime_error(
                "ungated routed MoE requires only Up and Down at layer " +
                std::to_string(layer));
        if constexpr (requires { result.moe_ungated_experts = true; }) {
            result.moe_ungated_experts = true;
            // Up lives in the split slot so prefetch/cache code sees a real
            // single projection instead of an empty fused pair.
            result.moe_split_gate_up = true;
        } else
            throw std::runtime_error(
                "this FFN type does not support ungated routed experts at layer " +
                std::to_string(layer));
        if constexpr (requires { result.activation = MlpActivation::relu_squared; })
            result.activation = MlpActivation::relu_squared;
        ops.experts(result, ExpertProjection::up, up, layer);
    } else {
        const bool split = has_gate, fused = has_gate_up;
        if (split != ops.has(up) || fused == split || !ops.has(down))
            throw std::runtime_error(
                "routed MoE requires Down and exactly one fused or split "
                "Gate/Up representation at layer " + std::to_string(layer));
        result.moe_split_gate_up = split;
        if (split) {
            ops.experts(result, ExpertProjection::gate, gate, layer);
            ops.experts(result, ExpertProjection::up, up, layer);
        } else {
            ops.experts(result, ExpertProjection::gate_up, gate_up, layer);
        }
    }
    ops.experts(result, ExpertProjection::down, down, layer);
    const auto latent_in = prefix + "latent.down.weight";
    const auto latent_out = prefix + "latent.up.weight";
    if (ops.has(latent_in) != ops.has(latent_out))
        throw std::runtime_error(
            "latent MoE requires both Down and Up latent projections at layer " +
            std::to_string(layer));
    if (ops.has(latent_in)) {
        if constexpr (requires { result.moe_latent_in = ops.linear(latent_in); }) {
            result.moe_latent_in = ops.linear(latent_in);
            result.moe_latent_out = ops.linear(latent_out);
        } else {
            throw std::runtime_error(
                "this FFN type does not support latent MoE projections at layer " +
                std::to_string(layer));
        }
    }
    result.moe_router = ops.router_parameter(prefix + "router.weight");
    if constexpr(requires { result.moe_router_projection = ops.router_projection; })
        result.moe_router_projection=ops.router_projection;
    if (router_bias_required || ops.has(prefix + "router.bias"))
        result.moe_router_bias = ops.router_parameter(prefix + "router.bias");
    result.shared = std::make_unique<Ffn>();
    result.shared->down = ops.linear(prefix + "shared_expert.down.weight");
    const auto shared_gate = prefix + "shared_expert.gate.weight";
    const auto shared_up = prefix + "shared_expert.up.weight";
    if (ungated_shared) {
        if (ops.has(shared_gate) || !ops.has(shared_up))
            throw std::runtime_error(
                "ungated shared expert requires Up without a Gate at layer " +
                std::to_string(layer));
        if constexpr (requires { result.shared->activation = MlpActivation::relu_squared; })
            result.shared->activation = MlpActivation::relu_squared;
        result.shared->gate_up = ops.gate_up({shared_up}, result.shared->down);
    } else {
        result.shared->gate_up = ops.gate_up({shared_gate, shared_up}, result.shared->down);
    }
    ops.workspace(*result.shared);
    return result;
}

// Geometry and projections of one canonical attention mixer. Blocks that also
// carry a separate FFN norm load it through their own adapter.
template <class Block, class Loader>
void load_attention_mixer(Block& block, Loader& ops, const ModelConfig& config, int layer,
                          const std::string& prefix, double norm_offset,
                          bool output_gate = false) {
    block.layer = layer;
    block.attention_heads = config.num_attention_heads;
    block.kv_heads = config.num_key_value_heads;
    block.attention_head_dim = config.head_dim;
    block.max_position_embeddings = config.max_position_embeddings;
    block.rms_norm_eps = config.rms_norm_eps;
    block.norm_weight_offset = norm_offset;
    block.attention_output_gate = output_gate;
    if constexpr (requires { block.no_rope = false; })
        block.no_rope = config.positional_encoding == PositionalEncoding::none;
    block.attn_norm = ops.dense(prefix + "attention.norm.weight");
    const auto attention = prefix + "attention.";
    ops.qkv(block, {attention + "query.weight", attention + "key.weight", attention + "value.weight"});
    block.o = ops.linear(attention + "output.weight");
    if (ops.has(attention + "query_norm.weight"))
        block.q_norm = ops.dense(attention + "query_norm.weight");
    if (ops.has(attention + "key_norm.weight"))
        block.k_norm = ops.dense(attention + "key_norm.weight");
}

template <class Block, class Loader>
void load_full_attention(Block& block, Loader& ops, const ModelConfig& config, int layer,
                         const std::string& prefix, double norm_offset, bool output_gate = false) {
    load_attention_mixer(block, ops, config, layer, prefix, norm_offset, output_gate);
    block.ffn_norm = ops.dense(prefix + "mlp.norm.weight");
}

} // namespace mfq::models
