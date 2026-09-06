#include "mfq_legacy_tensor_names.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace mfq {
namespace {

using json = nlohmann::json;

std::string identity(std::string_view value) {
    std::string result(value);
    std::transform(
        result.begin(), result.end(), result.begin(),
        [](unsigned char character) {
            return character == '-' ? '_' :
                static_cast<char>(std::tolower(character));
        });
    return result;
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.substr(0, prefix.size()) == prefix;
}

bool qwen35_family(
        std::string_view architecture,
        const json& config) {
    const auto stored = identity(architecture);
    const auto config_type = identity(config.value("model_type", std::string{}));
    const json* text = &config;
    const auto found = config.find("text_config");
    if (found != config.end() && found->is_object()) text = &*found;
    const auto text_type = identity(text->value("model_type", std::string{}));
    const auto matches = [](std::string_view value) {
        return starts_with(value, "qwen3_5") ||
            starts_with(value, "qwen35") ||
            starts_with(value, "qwen3_6") ||
            starts_with(value, "qwen3_8");
    };
    return matches(stored) || matches(config_type) || matches(text_type);
}

bool minicpmo_family(
        std::string_view architecture,
        const json& config) {
    const auto stored = identity(architecture);
    const auto config_type = identity(config.value("model_type", std::string{}));
    return starts_with(stored, "minicpmo") || config_type == "minicpmo";
}

bool glm_dsa_family(
        std::string_view architecture,
        const json& config) {
    const auto stored = identity(architecture);
    const auto config_type = identity(config.value("model_type", std::string{}));
    return starts_with(stored, "glm_moe_dsa") ||
        config_type == "glm_moe_dsa";
}

bool deepseek_v4_family(
        std::string_view architecture,
        const json& config) {
    const auto stored = identity(architecture);
    const auto config_type = identity(config.value("model_type", std::string{}));
    return starts_with(stored, "deepseek_v4") ||
        starts_with(config_type, "deepseek_v4");
}

bool gemma4_family(
        std::string_view architecture,
        const json& config) {
    const auto stored = identity(architecture);
    const auto config_type = identity(config.value("model_type", std::string{}));
    return starts_with(stored, "gemma4") || starts_with(config_type, "gemma4");
}

std::int64_t config_integer(
        const json& config,
        const char* key,
        bool vision = false) {
    const json* object = &config;
    const char* section = vision ? "vision_config" : "text_config";
    const auto nested = config.find(section);
    if (nested != config.end() && nested->is_object()) object = &*nested;
    const auto found = object->find(key);
    return found != object->end() && found->is_number_integer()
        ? found->get<std::int64_t>() : 0;
}

void add_alias(
        MfqLegacyTensorAliases& result,
        const std::unordered_set<std::string>& stored_names,
        std::string canonical,
        std::string stored) {
    if (canonical == stored || stored_names.count(canonical) != 0 ||
        stored_names.count(stored) == 0) {
        return;
    }
    const auto [found, inserted] = result.canonical_to_stored.emplace(
        canonical, stored);
    if (!inserted && found->second != stored) {
        throw std::runtime_error(
            "legacy tensor aliases collide at canonical name: " +
            found->first);
    }
}

void add_derived_aliases(
        MfqLegacyTensorAliases& result,
        const std::unordered_set<std::string>& names) {
    const auto base = result.canonical_to_stored;
    for (const auto& [canonical, stored] : base) {
        for (const std::string_view suffix : {std::string_view(".in_high")}) {
            add_alias(
                result,
                names,
                canonical + std::string(suffix),
                stored + std::string(suffix));
        }
    }
}

void add_qwen_hf_aliases(
        MfqLegacyTensorAliases& result,
        const json& config,
        const std::unordered_set<std::string>& names) {
    const auto add = [&](std::string canonical, std::string stored) {
        add_alias(result, names, std::move(canonical), std::move(stored));
    };
    add("model.token_embedding.weight",
        "model.language_model.embed_tokens.weight");
    add("model.output_norm.weight", "model.language_model.norm.weight");
    add("model.output.weight", "lm_head.weight");

    static const std::unordered_map<std::string_view, std::string_view> suffixes{
        {"attention.norm.weight", "input_layernorm.weight"},
        {"attention.query.weight", "self_attn.q_proj.weight"},
        {"attention.key.weight", "self_attn.k_proj.weight"},
        {"attention.value.weight", "self_attn.v_proj.weight"},
        {"attention.output.weight", "self_attn.o_proj.weight"},
        {"attention.query_norm.weight", "self_attn.q_norm.weight"},
        {"attention.key_norm.weight", "self_attn.k_norm.weight"},
        {"mlp.norm.weight", "post_attention_layernorm.weight"},
        {"mlp.gate.weight", "mlp.gate_proj.weight"},
        {"mlp.up.weight", "mlp.up_proj.weight"},
        {"mlp.down.weight", "mlp.down_proj.weight"},
        {"mlp.experts.gate_up.weight", "mlp.experts.gate_up_proj"},
        {"mlp.experts.down.weight", "mlp.experts.down_proj"},
        {"mlp.router.weight", "mlp.gate.weight"},
        {"mlp.shared_expert.gate.weight", "mlp.shared_expert.gate_proj.weight"},
        {"mlp.shared_expert.up.weight", "mlp.shared_expert.up_proj.weight"},
        {"mlp.shared_expert.down.weight", "mlp.shared_expert.down_proj.weight"},
        {"mlp.shared_expert.router.weight", "mlp.shared_expert_gate.weight"},
        {"linear_attention.qkv.weight", "linear_attn.in_proj_qkv.weight"},
        {"linear_attention.qk.weight", "linear_attn.in_proj_qk.weight"},
        {"linear_attention.value.weight", "linear_attn.in_proj_v.weight"},
        {"linear_attention.gate.weight", "linear_attn.in_proj_z.weight"},
        {"linear_attention.alpha.weight", "linear_attn.in_proj_a.weight"},
        {"linear_attention.beta.weight", "linear_attn.in_proj_b.weight"},
        {"linear_attention.conv.weight", "linear_attn.conv1d.weight"},
        {"linear_attention.conv.bias", "linear_attn.conv1d.bias"},
        {"linear_attention.dt_bias", "linear_attn.dt_bias"},
        {"linear_attention.a", "linear_attn.A_log"},
        {"linear_attention.norm.weight", "linear_attn.norm.weight"},
        {"linear_attention.output.weight", "linear_attn.out_proj.weight"},
    };
    const auto text_layers = config_integer(config, "num_hidden_layers");
    for (std::int64_t layer = 0; layer < text_layers; ++layer) {
        const auto canonical = "model.block." + std::to_string(layer) + ".";
        const auto stored =
            "model.language_model.layers." + std::to_string(layer) + ".";
        for (const auto& [canonical_suffix, stored_suffix] : suffixes) {
            add(canonical + std::string(canonical_suffix),
                stored + std::string(stored_suffix));
        }
    }

    static const std::unordered_map<std::string_view, std::string_view>
        predictor_roots{
            {"predictor.fusion.weight", "mtp.fc.weight"},
            {"predictor.fusion.embedding.weight", "mtp.fc_embedding.weight"},
            {"predictor.fusion.hidden.weight", "mtp.fc_hidden.weight"},
            {"predictor.embedding_norm.weight", "mtp.pre_fc_norm_embedding.weight"},
            {"predictor.hidden_norm.weight", "mtp.pre_fc_norm_hidden.weight"},
            {"predictor.output_norm.weight", "mtp.norm.weight"},
            {"predictor.mhc.pre.norm.weight", "mtp.hyper_connection_mixer.hc_norm.weight"},
            {"predictor.mhc.pre.down.weight", "mtp.hyper_connection_mixer.input_mix_weight_down.weight"},
            {"predictor.mhc.pre.up.weight", "mtp.hyper_connection_mixer.input_mix_weight_up.weight"},
        };
    for (const auto& [canonical, stored] : predictor_roots) {
        add(std::string(canonical), std::string(stored));
    }
    const auto predictor_layers = config_integer(
        config, "mtp_num_hidden_layers");
    for (std::int64_t layer = 0; layer < predictor_layers; ++layer) {
        const auto canonical =
            "predictor.block." + std::to_string(layer) + ".";
        const auto stored = "mtp.layers." + std::to_string(layer) + ".";
        for (const auto& [canonical_suffix, stored_suffix] : suffixes) {
            add(canonical + std::string(canonical_suffix),
                stored + std::string(stored_suffix));
        }
    }

    static const std::unordered_map<std::string_view, std::string_view>
        vision_roots{
            {"vision.patch_embedding.weight", "model.visual.patch_embed.proj.weight"},
            {"vision.patch_embedding.bias", "model.visual.patch_embed.proj.bias"},
            {"vision.position_embedding.weight", "model.visual.pos_embed.weight"},
            {"vision.merger.norm.weight", "model.visual.merger.norm.weight"},
            {"vision.merger.norm.bias", "model.visual.merger.norm.bias"},
            {"vision.merger.mlp.up.weight", "model.visual.merger.linear_fc1.weight"},
            {"vision.merger.mlp.up.bias", "model.visual.merger.linear_fc1.bias"},
            {"vision.merger.mlp.down.weight", "model.visual.merger.linear_fc2.weight"},
            {"vision.merger.mlp.down.bias", "model.visual.merger.linear_fc2.bias"},
        };
    for (const auto& [canonical, stored] : vision_roots) {
        add(std::string(canonical), std::string(stored));
    }
    static const std::unordered_map<std::string_view, std::string_view>
        vision_suffixes{
            {"norm1.weight", "norm1.weight"},
            {"norm1.bias", "norm1.bias"},
            {"attention.qkv.weight", "attn.qkv.weight"},
            {"attention.qkv.bias", "attn.qkv.bias"},
            {"attention.output.weight", "attn.proj.weight"},
            {"attention.output.bias", "attn.proj.bias"},
            {"attention.query_norm.weight", "attn.q_norm.weight"},
            {"attention.key_norm.weight", "attn.k_norm.weight"},
            {"norm2.weight", "norm2.weight"},
            {"norm2.bias", "norm2.bias"},
            {"mlp.up.weight", "mlp.linear_fc1.weight"},
            {"mlp.up.bias", "mlp.linear_fc1.bias"},
            {"mlp.down.weight", "mlp.linear_fc2.weight"},
            {"mlp.down.bias", "mlp.linear_fc2.bias"},
        };
    const auto vision_layers = config_integer(config, "depth", true);
    for (std::int64_t layer = 0; layer < vision_layers; ++layer) {
        const auto canonical = "vision.block." + std::to_string(layer) + ".";
        const auto stored = "model.visual.blocks." + std::to_string(layer) + ".";
        for (const auto& [canonical_suffix, stored_suffix] : vision_suffixes) {
            add(canonical + std::string(canonical_suffix),
                stored + std::string(stored_suffix));
        }
    }
}

void add_qwen_gguf_aliases(
        MfqLegacyTensorAliases& result,
        const json& config,
        const std::unordered_set<std::string>& names) {
    result.layout.norm_weight_offset = 0.0;
    result.layout.linear_attention_a_is_log = false;
    result.layout.qwen_gdn_gguf_layout = true;
    const auto add = [&](std::string canonical, std::string stored) {
        add_alias(result, names, std::move(canonical), std::move(stored));
    };
    add("model.token_embedding.weight", "token_embd.weight");
    add("model.output_norm.weight", "output_norm.weight");
    add("model.output.weight", "output.weight");
    static const std::unordered_map<std::string_view, std::string_view> suffixes{
        {"attention.norm.weight", "attn_norm.weight"},
        {"attention.query.weight", "attn_q.weight"},
        {"attention.key.weight", "attn_k.weight"},
        {"attention.value.weight", "attn_v.weight"},
        {"attention.output.weight", "attn_output.weight"},
        {"attention.query_norm.weight", "attn_q_norm.weight"},
        {"attention.key_norm.weight", "attn_k_norm.weight"},
        {"mlp.norm.weight", "post_attention_norm.weight"},
        {"mlp.gate.weight", "ffn_gate.weight"},
        {"mlp.up.weight", "ffn_up.weight"},
        {"mlp.down.weight", "ffn_down.weight"},
        {"linear_attention.qkv.weight", "attn_qkv.weight"},
        {"linear_attention.qk.weight", "ssm_qk.weight"},
        {"linear_attention.value.weight", "ssm_v.weight"},
        {"linear_attention.gate.weight", "attn_gate.weight"},
        {"linear_attention.alpha.weight", "ssm_alpha.weight"},
        {"linear_attention.beta.weight", "ssm_beta.weight"},
        {"linear_attention.conv.weight", "ssm_conv1d.weight"},
        {"linear_attention.conv.bias", "ssm_conv1d.bias"},
        {"linear_attention.dt_bias", "ssm_dt.bias"},
        {"linear_attention.a", "ssm_a"},
        {"linear_attention.norm.weight", "ssm_norm.weight"},
        {"linear_attention.output.weight", "ssm_out.weight"},
    };
    const auto text_layers = config_integer(config, "num_hidden_layers");
    for (std::int64_t layer = 0; layer < text_layers; ++layer) {
        const auto canonical = "model.block." + std::to_string(layer) + ".";
        const auto stored = "blk." + std::to_string(layer) + ".";
        for (const auto& [canonical_suffix, stored_suffix] : suffixes) {
            add(canonical + std::string(canonical_suffix),
                stored + std::string(stored_suffix));
        }
    }
    const auto predictor_layers = config_integer(
        config, "mtp_num_hidden_layers");
    const auto base = text_layers;
    add("predictor.hidden_norm.weight",
        "blk." + std::to_string(base) + ".nextn.hnorm.weight");
    add("predictor.embedding_norm.weight",
        "blk." + std::to_string(base) + ".nextn.enorm.weight");
    add("predictor.fusion.weight",
        "blk." + std::to_string(base) + ".nextn.eh_proj.weight");
    add("predictor.output_norm.weight",
        "blk." + std::to_string(base) + ".nextn.shared_head_norm.weight");
    for (std::int64_t layer = 0; layer < predictor_layers; ++layer) {
        const auto canonical =
            "predictor.block." + std::to_string(layer) + ".";
        const auto stored = "blk." + std::to_string(base + layer) + ".";
        for (const auto& [canonical_suffix, stored_suffix] : suffixes) {
            add(canonical + std::string(canonical_suffix),
                stored + std::string(stored_suffix));
        }
    }
}

void add_minicpmo_text_aliases(
        MfqLegacyTensorAliases& result,
        const json& config,
        const std::unordered_set<std::string>& names) {
    result.layout.norm_weight_offset = 0.0;
    const auto add = [&](std::string canonical, std::string stored) {
        add_alias(result, names, std::move(canonical), std::move(stored));
    };
    add("model.token_embedding.weight", "llm.model.embed_tokens.weight");
    add("model.output_norm.weight", "llm.model.norm.weight");
    add("model.output.weight", "llm.lm_head.weight");
    static const std::unordered_map<std::string_view, std::string_view> suffixes{
        {"attention.norm.weight", "input_layernorm.weight"},
        {"attention.query.weight", "self_attn.q_proj.weight"},
        {"attention.key.weight", "self_attn.k_proj.weight"},
        {"attention.value.weight", "self_attn.v_proj.weight"},
        {"attention.output.weight", "self_attn.o_proj.weight"},
        {"attention.query_norm.weight", "self_attn.q_norm.weight"},
        {"attention.key_norm.weight", "self_attn.k_norm.weight"},
        {"mlp.norm.weight", "post_attention_layernorm.weight"},
        {"mlp.gate.weight", "mlp.gate_proj.weight"},
        {"mlp.up.weight", "mlp.up_proj.weight"},
        {"mlp.down.weight", "mlp.down_proj.weight"},
    };
    const auto layers = config_integer(config, "num_hidden_layers");
    for (std::int64_t layer = 0; layer < layers; ++layer) {
        const auto canonical = "model.block." + std::to_string(layer) + ".";
        const auto stored = "llm.model.layers." + std::to_string(layer) + ".";
        for (const auto& [canonical_suffix, stored_suffix] : suffixes) {
            add(canonical + std::string(canonical_suffix),
                stored + std::string(stored_suffix));
        }
    }

    static const std::unordered_map<std::string_view, std::string_view> roots{
        {"vision.patch_embedding.weight", "vpm.embeddings.patch_embedding.weight"},
        {"vision.patch_embedding.bias", "vpm.embeddings.patch_embedding.bias"},
        {"vision.position_embedding.weight", "vpm.embeddings.position_embedding.weight"},
        {"vision.output_norm.weight", "vpm.post_layernorm.weight"},
        {"vision.output_norm.bias", "vpm.post_layernorm.bias"},
        {"vision.resampler.query", "resampler.query"},
        {"vision.resampler.key_value.weight", "resampler.kv_proj.weight"},
        {"vision.resampler.query_norm.weight", "resampler.ln_q.weight"},
        {"vision.resampler.query_norm.bias", "resampler.ln_q.bias"},
        {"vision.resampler.key_value_norm.weight", "resampler.ln_kv.weight"},
        {"vision.resampler.key_value_norm.bias", "resampler.ln_kv.bias"},
        {"vision.resampler.output_norm.weight", "resampler.ln_post.weight"},
        {"vision.resampler.output_norm.bias", "resampler.ln_post.bias"},
        {"vision.resampler.attention.qkv.weight", "resampler.attn.in_proj_weight"},
        {"vision.resampler.attention.qkv.bias", "resampler.attn.in_proj_bias"},
        {"vision.resampler.attention.output.weight", "resampler.attn.out_proj.weight"},
        {"vision.resampler.attention.output.bias", "resampler.attn.out_proj.bias"},
        {"vision.resampler.output.weight", "resampler.proj"},
        {"audio.patch_embedding.conv1.weight", "apm.conv1.weight"},
        {"audio.patch_embedding.conv1.bias", "apm.conv1.bias"},
        {"audio.patch_embedding.conv2.weight", "apm.conv2.weight"},
        {"audio.patch_embedding.conv2.bias", "apm.conv2.bias"},
        {"audio.position_embedding.weight", "apm.embed_positions.weight"},
        {"audio.output_norm.weight", "apm.layer_norm.weight"},
        {"audio.output_norm.bias", "apm.layer_norm.bias"},
        {"audio.projector.input.weight", "audio_projection_layer.linear1.weight"},
        {"audio.projector.input.bias", "audio_projection_layer.linear1.bias"},
        {"audio.projector.output.weight", "audio_projection_layer.linear2.weight"},
        {"audio.projector.output.bias", "audio_projection_layer.linear2.bias"},
        {"tts.text_embedding.weight", "tts.emb_text.weight"},
        {"tts.token_embedding.weight", "tts.model.embed_tokens.weight"},
        {"tts.output_norm.weight", "tts.model.norm.weight"},
        {"tts.code_embedding.0.weight", "tts.emb_code.0.weight"},
        {"tts.code_output.0.weight_norm.magnitude", "tts.head_code.0.parametrizations.weight.original0"},
        {"tts.code_output.0.weight_norm.direction", "tts.head_code.0.parametrizations.weight.original1"},
        {"tts.semantic_projector.input.weight", "tts.projector_semantic.linear1.weight"},
        {"tts.semantic_projector.input.bias", "tts.projector_semantic.linear1.bias"},
        {"tts.semantic_projector.output.weight", "tts.projector_semantic.linear2.weight"},
        {"tts.semantic_projector.output.bias", "tts.projector_semantic.linear2.bias"},
        {"tts.speaker_projector.input.weight", "tts.projector_spk.linear1.weight"},
        {"tts.speaker_projector.input.bias", "tts.projector_spk.linear1.bias"},
        {"tts.speaker_projector.output.weight", "tts.projector_spk.linear2.weight"},
        {"tts.speaker_projector.output.bias", "tts.projector_spk.linear2.bias"},
    };
    for (const auto& [canonical, stored] : roots) {
        add(std::string(canonical), std::string(stored));
    }

    static const std::unordered_map<std::string_view, std::string_view>
        vision_suffixes{
            {"norm1.weight", "layer_norm1.weight"},
            {"norm1.bias", "layer_norm1.bias"},
            {"norm2.weight", "layer_norm2.weight"},
            {"norm2.bias", "layer_norm2.bias"},
            {"attention.query.weight", "self_attn.q_proj.weight"},
            {"attention.query.bias", "self_attn.q_proj.bias"},
            {"attention.key.weight", "self_attn.k_proj.weight"},
            {"attention.key.bias", "self_attn.k_proj.bias"},
            {"attention.value.weight", "self_attn.v_proj.weight"},
            {"attention.value.bias", "self_attn.v_proj.bias"},
            {"attention.output.weight", "self_attn.out_proj.weight"},
            {"attention.output.bias", "self_attn.out_proj.bias"},
            {"mlp.up.weight", "mlp.fc1.weight"},
            {"mlp.up.bias", "mlp.fc1.bias"},
            {"mlp.down.weight", "mlp.fc2.weight"},
            {"mlp.down.bias", "mlp.fc2.bias"},
        };
    static const std::unordered_map<std::string_view, std::string_view>
        audio_suffixes{
            {"attention.norm.weight", "self_attn_layer_norm.weight"},
            {"attention.norm.bias", "self_attn_layer_norm.bias"},
            {"attention.query.weight", "self_attn.q_proj.weight"},
            {"attention.query.bias", "self_attn.q_proj.bias"},
            {"attention.key.weight", "self_attn.k_proj.weight"},
            {"attention.value.weight", "self_attn.v_proj.weight"},
            {"attention.value.bias", "self_attn.v_proj.bias"},
            {"attention.output.weight", "self_attn.out_proj.weight"},
            {"attention.output.bias", "self_attn.out_proj.bias"},
            {"mlp.norm.weight", "final_layer_norm.weight"},
            {"mlp.norm.bias", "final_layer_norm.bias"},
            {"mlp.up.weight", "fc1.weight"},
            {"mlp.up.bias", "fc1.bias"},
            {"mlp.down.weight", "fc2.weight"},
            {"mlp.down.bias", "fc2.bias"},
        };
    static const std::unordered_map<std::string_view, std::string_view>
        tts_suffixes{
            {"attention.norm.weight", "input_layernorm.weight"},
            {"mlp.norm.weight", "post_attention_layernorm.weight"},
            {"attention.query.weight", "self_attn.q_proj.weight"},
            {"attention.key.weight", "self_attn.k_proj.weight"},
            {"attention.value.weight", "self_attn.v_proj.weight"},
            {"attention.output.weight", "self_attn.o_proj.weight"},
            {"mlp.gate.weight", "mlp.gate_proj.weight"},
            {"mlp.up.weight", "mlp.up_proj.weight"},
            {"mlp.down.weight", "mlp.down_proj.weight"},
        };
    const auto add_numbered = [&](
            std::string_view stored_prefix,
            std::string_view canonical_prefix,
            const auto& suffixes) {
        for (const auto& stored : names) {
            if (!starts_with(stored, stored_prefix)) continue;
            const auto layer_begin = stored_prefix.size();
            const auto layer_end = stored.find('.', layer_begin);
            if (layer_end == std::string::npos) continue;
            const auto layer = stored.substr(layer_begin, layer_end - layer_begin);
            const auto stored_suffix =
                std::string_view(stored).substr(layer_end + 1);
            for (const auto& [canonical_suffix, source_suffix] : suffixes) {
                if (stored_suffix == source_suffix) {
                    add(
                        std::string(canonical_prefix) + layer + "." +
                            std::string(canonical_suffix),
                        stored);
                    break;
                }
            }
        }
    };
    add_numbered("vpm.encoder.layers.", "vision.block.", vision_suffixes);
    add_numbered("apm.layers.", "audio.block.", audio_suffixes);
    add_numbered("tts.model.layers.", "tts.block.", tts_suffixes);
}

void add_glm_dsa_aliases(
        MfqLegacyTensorAliases& result,
        const json& config,
        const std::unordered_set<std::string>& names) {
    result.layout.norm_weight_offset = 0.0;
    const auto add = [&](std::string canonical, std::string stored) {
        add_alias(result, names, std::move(canonical), std::move(stored));
    };
    add("model.token_embedding.weight", "model.embed_tokens.weight");
    add("model.output_norm.weight", "model.norm.weight");
    add("model.output.weight", "lm_head.weight");
    static const std::unordered_map<std::string_view, std::string_view> suffixes{
        {"attention.norm.weight", "input_layernorm.weight"},
        {"mlp.norm.weight", "post_attention_layernorm.weight"},
        {"attention.query_a.weight", "self_attn.q_a_proj.weight"},
        {"attention.query_a_norm.weight", "self_attn.q_a_layernorm.weight"},
        {"attention.query_b.weight", "self_attn.q_b_proj.weight"},
        {"attention.key_value_a.weight", "self_attn.kv_a_proj_with_mqa.weight"},
        {"attention.key_value_a_norm.weight", "self_attn.kv_a_layernorm.weight"},
        {"attention.output.weight", "self_attn.o_proj.weight"},
        {"attention.latent.query_embedding.weight", "self_attn.embed_q"},
        {"attention.latent.output_unembedding.weight", "self_attn.unembed_out"},
        {"attention.indexer.query.weight", "self_attn.indexer.wq_b.weight"},
        {"attention.indexer.key.weight", "self_attn.indexer.wk.weight"},
        {"attention.indexer.score.weight", "self_attn.indexer.weights_proj.weight"},
        {"attention.indexer.key_norm.weight", "self_attn.indexer.k_norm.weight"},
        {"attention.indexer.key_norm.bias", "self_attn.indexer.k_norm.bias"},
        {"mlp.gate.weight", "mlp.gate_proj.weight"},
        {"mlp.up.weight", "mlp.up_proj.weight"},
        {"mlp.down.weight", "mlp.down_proj.weight"},
        {"mlp.router.weight", "mlp.gate.weight"},
        {"mlp.router.bias", "mlp.gate.e_score_correction_bias"},
        {"mlp.experts.gate_up.weight", "mlp.experts.gate_up_proj"},
        {"mlp.experts.down.weight", "mlp.experts.down_proj"},
        {"mlp.shared_expert.gate.weight", "mlp.shared_experts.gate_proj.weight"},
        {"mlp.shared_expert.up.weight", "mlp.shared_experts.up_proj.weight"},
        {"mlp.shared_expert.down.weight", "mlp.shared_experts.down_proj.weight"},
    };
    const auto layers = config_integer(config, "num_hidden_layers");
    for (std::int64_t layer = 0; layer < layers; ++layer) {
        const auto canonical = "model.block." + std::to_string(layer) + ".";
        const auto stored = "model.layers." + std::to_string(layer) + ".";
        for (const auto& [canonical_suffix, stored_suffix] : suffixes) {
            add(canonical + std::string(canonical_suffix),
                stored + std::string(stored_suffix));
        }
    }
}

void add_deepseek_v4_aliases(
        MfqLegacyTensorAliases& result,
        const std::unordered_set<std::string>& names) {
    result.layout.norm_weight_offset = 0.0;
    const auto add = [&](std::string canonical, std::string stored) {
        add_alias(result, names, std::move(canonical), std::move(stored));
    };
    add("model.token_embedding.weight", "token_embd.weight");
    add("model.token_embedding.weight", "embed.weight");
    add("model.output_norm.weight", "output_norm.weight");
    add("model.output_norm.weight", "norm.weight");
    add("model.output.weight", "output.weight");
    add("model.output.weight", "head.weight");
    add("model.mhc.output.function", "output_hc_fn.weight");
    add("model.mhc.output.function", "hc_head_fn");
    add("model.mhc.output.base", "output_hc_base.weight");
    add("model.mhc.output.base", "hc_head_base");
    add("model.mhc.output.scale", "output_hc_scale.weight");
    add("model.mhc.output.scale", "hc_head_scale");
    static const std::unordered_map<std::string_view, std::string_view>
        vision_roots{
            {"vision.patch_embedding.weight", "vision.patch_embed.proj.weight"},
            {"vision.patch_embedding.bias", "vision.patch_embed.proj.bias"},
            {"vision.output_norm.weight", "vision.norm.weight"},
            {"vision.aligner.input.weight", "aligner.w1.weight"},
            {"vision.aligner.input.bias", "aligner.w1.bias"},
            {"vision.aligner.output.weight", "aligner.w2.weight"},
            {"vision.aligner.output.bias", "aligner.w2.bias"},
            {"vision.special_token.start", "image_start"},
            {"vision.special_token.pad", "image_pad"},
            {"vision.special_token.newline", "image_newline"},
            {"vision.special_token.end", "image_end"},
        };
    for (const auto& [canonical, stored] : vision_roots) {
        add(std::string(canonical), std::string(stored));
    }
    static const std::unordered_map<std::string_view, std::string_view>
        vision_suffixes{
            {"norm1.weight", "norm1.weight"},
            {"norm2.weight", "norm2.weight"},
            {"attention.qkv.weight", "attn.wqkv.weight"},
            {"attention.qkv.bias", "attn.wqkv.bias"},
            {"attention.output.weight", "attn.wo.weight"},
            {"attention.output.bias", "attn.wo.bias"},
            {"mlp.gate_up.weight", "mlp.w1.weight"},
            {"mlp.down.weight", "mlp.w2.weight"},
        };
    for (const auto& stored : names) {
        constexpr std::string_view prefix = "vision.blocks.";
        if (!starts_with(stored, prefix)) continue;
        const auto layer_end = stored.find('.', prefix.size());
        if (layer_end == std::string::npos) continue;
        const auto layer = stored.substr(prefix.size(), layer_end - prefix.size());
        const auto stored_suffix =
            std::string_view(stored).substr(layer_end + 1);
        for (const auto& [canonical_suffix, source_suffix] : vision_suffixes) {
            if (stored_suffix == source_suffix) {
                add(
                    "vision.block." + layer + "." +
                        std::string(canonical_suffix),
                    stored);
                break;
            }
        }
    }
    static const std::unordered_map<std::string_view, std::string_view> suffixes{
        {"attention.norm.weight", "attn_norm.weight"},
        {"mlp.norm.weight", "ffn_norm.weight"},
        {"attention.query_a.weight", "attn_q_a.weight"},
        {"attention.query_a_norm.weight", "attn_q_a_norm.weight"},
        {"attention.query_b.weight", "attn_q_b.weight"},
        {"attention.key_value_a.weight", "attn_kv.weight"},
        {"attention.key_value_a_norm.weight", "attn_kv_a_norm.weight"},
        {"attention.sink", "attn_sinks.weight"},
        {"attention.output_a.weight", "attn_output_a.weight"},
        {"attention.output_b.weight", "attn_output_b.weight"},
        {"attention.mhc.pre.function", "hc_attn_fn.weight"},
        {"attention.mhc.pre.base", "hc_attn_base.weight"},
        {"attention.mhc.pre.scale", "hc_attn_scale.weight"},
        {"mlp.mhc.pre.function", "hc_ffn_fn.weight"},
        {"mlp.mhc.pre.base", "hc_ffn_base.weight"},
        {"mlp.mhc.pre.scale", "hc_ffn_scale.weight"},
        {"attention.compressor.key_value.weight", "attn_compressor_kv.weight"},
        {"attention.compressor.gate.weight", "attn_compressor_gate.weight"},
        {"attention.compressor.position", "attn_compressor_ape.weight"},
        {"attention.compressor.norm.weight", "attn_compressor_norm.weight"},
        {"attention.indexer.query.weight", "indexer.attn_q_b.weight"},
        {"attention.indexer.score.weight", "indexer.proj.weight"},
        {"attention.indexer.compressor.key_value.weight", "indexer_compressor_kv.weight"},
        {"attention.indexer.compressor.gate.weight", "indexer_compressor_gate.weight"},
        {"attention.indexer.compressor.position", "indexer_compressor_ape.weight"},
        {"attention.indexer.compressor.norm.weight", "indexer_compressor_norm.weight"},
        {"mlp.router.weight", "ffn_gate_inp.weight"},
        {"mlp.router.bias", "exp_probs_b.bias"},
        {"mlp.router.token_to_expert", "ffn_gate_tid2eid.weight"},
        {"mlp.experts.gate_up.weight", "ffn_gate_up_exps.weight"},
        {"mlp.experts.gate.weight", "ffn_gate_exps.weight"},
        {"mlp.experts.up.weight", "ffn_up_exps.weight"},
        {"mlp.experts.down.weight", "ffn_down_exps.weight"},
        {"mlp.shared_expert.gate.weight", "ffn_gate_shexp.weight"},
        {"mlp.shared_expert.up.weight", "ffn_up_shexp.weight"},
        {"mlp.shared_expert.down.weight", "ffn_down_shexp.weight"},
    };
    for (const auto& stored : names) {
        if (!starts_with(stored, "blk.")) continue;
        const auto layer_end = stored.find('.', 4);
        if (layer_end == std::string::npos) continue;
        const auto layer = stored.substr(4, layer_end - 4);
        const auto stored_suffix = std::string_view(stored).substr(layer_end + 1);
        for (const auto& [canonical_suffix, legacy_suffix] : suffixes) {
            if (stored_suffix == legacy_suffix) {
                add(
                    "model.block." + layer + "." +
                        std::string(canonical_suffix),
                    stored);
                break;
            }
        }
    }
    static const std::unordered_map<std::string_view, std::string_view>
        private_suffixes{
            {"attention.norm.weight", "attn_norm.weight"},
            {"mlp.norm.weight", "ffn_norm.weight"},
            {"attention.query_a.weight", "attn.wq_a.weight"},
            {"attention.query_a_norm.weight", "attn.q_norm.weight"},
            {"attention.query_b.weight", "attn.wq_b.weight"},
            {"attention.key_value_a.weight", "attn.wkv.weight"},
            {"attention.key_value_a_norm.weight", "attn.kv_norm.weight"},
            {"attention.sink", "attn.attn_sink"},
            {"attention.output_a.weight", "attn.wo_a.weight"},
            {"attention.output_b.weight", "attn.wo_b.weight"},
            {"attention.mhc.pre.function", "hc_attn_fn"},
            {"attention.mhc.pre.base", "hc_attn_base"},
            {"attention.mhc.pre.scale", "hc_attn_scale"},
            {"mlp.mhc.pre.function", "hc_ffn_fn"},
            {"mlp.mhc.pre.base", "hc_ffn_base"},
            {"mlp.mhc.pre.scale", "hc_ffn_scale"},
            {"attention.compressor.key_value.weight", "attn.compressor.wkv.weight"},
            {"attention.compressor.gate.weight", "attn.compressor.wgate.weight"},
            {"attention.compressor.position", "attn.compressor.ape"},
            {"attention.compressor.norm.weight", "attn.compressor.norm.weight"},
            {"attention.indexer.query.weight", "attn.indexer.wq_b.weight"},
            {"attention.indexer.score.weight", "attn.indexer.weights_proj.weight"},
            {"attention.indexer.compressor.key_value.weight", "attn.indexer.compressor.wkv.weight"},
            {"attention.indexer.compressor.gate.weight", "attn.indexer.compressor.wgate.weight"},
            {"attention.indexer.compressor.position", "attn.indexer.compressor.ape"},
            {"attention.indexer.compressor.norm.weight", "attn.indexer.compressor.norm.weight"},
            {"mlp.router.weight", "ffn.gate.weight"},
            {"mlp.router.bias", "ffn.gate.bias"},
            {"mlp.router.token_to_expert", "ffn.gate.tid2eid"},
            {"mlp.experts.gate_up.weight", "ffn.experts.gate_up.weight"},
            {"mlp.experts.gate.weight", "ffn.experts.gate.weight"},
            {"mlp.experts.up.weight", "ffn.experts.up.weight"},
            {"mlp.experts.down.weight", "ffn.experts.down.weight"},
            {"mlp.shared_expert.gate.weight", "ffn.shared_experts.w1.weight"},
            {"mlp.shared_expert.up.weight", "ffn.shared_experts.w3.weight"},
            {"mlp.shared_expert.down.weight", "ffn.shared_experts.w2.weight"},
        };
    for (const auto& stored : names) {
        if (!starts_with(stored, "layers.")) continue;
        const auto layer_end = stored.find('.', 7);
        if (layer_end == std::string::npos) continue;
        const auto layer = stored.substr(7, layer_end - 7);
        const auto stored_suffix = std::string_view(stored).substr(layer_end + 1);
        for (const auto& [canonical_suffix, private_suffix] : private_suffixes) {
            if (stored_suffix == private_suffix) {
                add(
                    "model.block." + layer + "." +
                        std::string(canonical_suffix),
                    stored);
                break;
            }
        }
    }

    static const std::unordered_map<std::string_view, std::string_view>
        predictor_suffixes{
            {"main_norm.weight", "main_norm.weight"},
            {"main_projection.weight", "main_proj.weight"},
            {"output_norm.weight", "norm.weight"},
            {"hc_head_fn", "mhc.output.function"},
            {"hc_head_base", "mhc.output.base"},
            {"hc_head_scale", "mhc.output.scale"},
            {"confidence_head.proj.weight", "confidence.projection.weight"},
            {"markov_head.markov_w1.weight", "markov.input.weight"},
            {"markov_head.markov_w2.weight", "markov.output.weight"},
        };
    for (const auto& stored : names) {
        if (!starts_with(stored, "mtp.")) continue;
        const auto stage_end = stored.find('.', 4);
        if (stage_end == std::string::npos) continue;
        const auto stage = stored.substr(4, stage_end - 4);
        const auto stored_suffix =
            std::string_view(stored).substr(stage_end + 1);
        const auto prefix = "predictor.stage." + stage + ".";
        bool mapped = false;
        for (const auto& [canonical_suffix, private_suffix] : private_suffixes) {
            if (stored_suffix == private_suffix) {
                add(prefix + std::string(canonical_suffix), stored);
                mapped = true;
                break;
            }
        }
        if (mapped) continue;
        const auto found = predictor_suffixes.find(stored_suffix);
        if (found != predictor_suffixes.end()) {
            add(prefix + std::string(found->second), stored);
        }
    }
}

void add_gemma4_aliases(
        MfqLegacyTensorAliases& result,
        const json& config,
        const std::unordered_set<std::string>& names) {
    result.layout.norm_weight_offset = 0.0;
    const auto add = [&](std::string canonical, std::string stored) {
        add_alias(result, names, std::move(canonical), std::move(stored));
    };
    add("model.token_embedding.weight",
        "model.language_model.embed_tokens.weight");
    add("model.output_norm.weight", "model.language_model.norm.weight");
    add("model.output.weight", "lm_head.weight");
    static const std::unordered_map<std::string_view, std::string_view> suffixes{
        {"attention.norm.weight", "input_layernorm.weight"},
        {"attention.output_norm.weight", "post_attention_layernorm.weight"},
        {"attention.query.weight", "self_attn.q_proj.weight"},
        {"attention.key.weight", "self_attn.k_proj.weight"},
        {"attention.value.weight", "self_attn.v_proj.weight"},
        {"attention.output.weight", "self_attn.o_proj.weight"},
        {"attention.query_norm.weight", "self_attn.q_norm.weight"},
        {"attention.key_norm.weight", "self_attn.k_norm.weight"},
        {"mlp.dense.input_norm.weight", "pre_feedforward_layernorm.weight"},
        {"mlp.output_norm.weight", "post_feedforward_layernorm.weight"},
        {"mlp.dense.output_norm.weight", "post_feedforward_layernorm_1.weight"},
        {"mlp.experts.input_norm.weight", "pre_feedforward_layernorm_2.weight"},
        {"mlp.experts.output_norm.weight", "post_feedforward_layernorm_2.weight"},
        {"output_scale", "layer_scalar"},
        {"mlp.gate.weight", "mlp.gate_proj.weight"},
        {"mlp.up.weight", "mlp.up_proj.weight"},
        {"mlp.down.weight", "mlp.down_proj.weight"},
        {"mlp.experts.gate_up.weight", "experts.gate_up_proj"},
        {"mlp.experts.down.weight", "experts.down_proj"},
        {"mlp.router.weight", "router.proj.weight"},
        {"mlp.router.norm.weight", "router.scale"},
        {"mlp.router.expert_scale", "router.per_expert_scale"},
    };
    const auto layers = config_integer(config, "num_hidden_layers");
    for (std::int64_t layer = 0; layer < layers; ++layer) {
        const auto canonical = "model.block." + std::to_string(layer) + ".";
        const auto stored =
            "model.language_model.layers." + std::to_string(layer) + ".";
        for (const auto& [canonical_suffix, stored_suffix] : suffixes) {
            add(canonical + std::string(canonical_suffix),
                stored + std::string(stored_suffix));
        }
    }
}

} // namespace

MfqLegacyTensorAliases make_legacy_tensor_aliases(
        std::string_view artifact_architecture,
        std::string_view model_config_json,
        const std::vector<std::string>& stored_names) {
    json config;
    try {
        config = json::parse(model_config_json);
    } catch (const json::exception& error) {
        throw std::invalid_argument(
            std::string("invalid legacy MFQ model config: ") + error.what());
    }
    if (!config.is_object()) {
        throw std::invalid_argument("legacy MFQ model config must be an object");
    }
    const std::unordered_set<std::string> names(
        stored_names.begin(), stored_names.end());
    MfqLegacyTensorAliases result;
    if (gemma4_family(artifact_architecture, config)) {
        add_gemma4_aliases(result, config, names);
        add_derived_aliases(result, names);
        return result;
    }
    if (deepseek_v4_family(artifact_architecture, config)) {
        add_deepseek_v4_aliases(result, names);
        add_derived_aliases(result, names);
        return result;
    }
    if (glm_dsa_family(artifact_architecture, config)) {
        add_glm_dsa_aliases(result, config, names);
        add_derived_aliases(result, names);
        return result;
    }
    if (minicpmo_family(artifact_architecture, config)) {
        add_minicpmo_text_aliases(result, config, names);
        add_derived_aliases(result, names);
        return result;
    }
    if (!qwen35_family(artifact_architecture, config)) return result;
    if (names.count("model.language_model.embed_tokens.weight") != 0) {
        add_qwen_hf_aliases(result, config, names);
    } else if (names.count("token_embd.weight") != 0) {
        add_qwen_gguf_aliases(result, config, names);
    }
    add_derived_aliases(result, names);
    return result;
}

} // namespace mfq
