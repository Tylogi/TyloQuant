#include "models/nemotron_h/config.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace mfq::models::nemotron_h {
namespace {

using json = nlohmann::json;

const json& text_section(const json& root) {
    for (const char* key : {"llm_config", "text_config"}) {
        const auto found = root.find(key);
        if (found == root.end())
            continue;
        if (!found->is_object())
            throw std::runtime_error(std::string("Nemotron-H model config ") + key +
                                     " must be an object");
        return *found;
    }
    return root;
}

// The released checkpoint encodes Infinity as {"__float__": "Infinity"}.
double number(const json& value, const char* name) {
    if (value.is_number())
        return value.get<double>();
    if (value.is_object() && value.contains("__float__")) {
        const auto& raw = value.at("__float__");
        if (raw.is_number())
            return raw.get<double>();
        if (raw.is_string()) {
            const auto text = raw.get<std::string>();
            if (text == "Infinity" || text == "inf")
                return std::numeric_limits<double>::infinity();
            if (text == "-Infinity" || text == "-inf")
                return -std::numeric_limits<double>::infinity();
        }
    }
    throw std::runtime_error(std::string("Nemotron-H config ") + name +
                             " must be a number");
}

std::int64_t integer(const json& object, const char* name, std::int64_t fallback) {
    const auto found = object.find(name);
    if (found == object.end() || found->is_null())
        return fallback;
    if (!found->is_number_integer())
        throw std::runtime_error(std::string("Nemotron-H config ") + name +
                                 " must be an integer");
    return found->get<std::int64_t>();
}

std::vector<std::string> strings(const json& object, const char* name) {
    const auto found = object.find(name);
    if (found == object.end())
        throw std::runtime_error(std::string("Nemotron-H config requires ") + name);
    if (!found->is_array())
        throw std::runtime_error(std::string("Nemotron-H config ") + name +
                                 " must be an array");
    return found->get<std::vector<std::string>>();
}

} // namespace

MixerKind Config::mixer_kind(std::string_view type) {
    if (type == "mamba" || type == "linear_attention")
        return MixerKind::linear_attention;
    if (type == "attention" || type == "full_attention")
        return MixerKind::full_attention;
    if (type == "moe")
        return MixerKind::moe;
    if (type == "mlp")
        return MixerKind::mlp;
    throw std::runtime_error("unsupported Nemotron-H mixer type: " + std::string(type));
}

std::string_view Config::mixer_name(MixerKind kind) {
    switch (kind) {
        case MixerKind::linear_attention:
            return "linear_attention";
        case MixerKind::full_attention:
            return "full_attention";
        case MixerKind::moe:
            return "moe";
        case MixerKind::mlp:
            return "mlp";
    }
    throw std::runtime_error("unknown Nemotron-H mixer kind");
}

Config Config::from_json(std::string_view payload, const MfqModelGraph&) {
    const auto document = json::parse(payload.begin(), payload.end(), nullptr, false);
    if (!document.is_object())
        throw std::runtime_error("Nemotron-H model config must be an object");
    const auto& text = text_section(document);

    // Normalize the nested language-model section into the shared model config
    // contract: the block schedule is authoritative for the layer count and
    // the mixer labels are the canonical names the backend dispatch uses.
    json normalized = text;
    std::vector<std::string> schedule;
    if(text.contains("layers_block_type"))schedule=strings(text,"layers_block_type");
    else if(text.contains("layer_types"))schedule=strings(text,"layer_types");
    else if(text.contains("hybrid_override_pattern")) {
        if(!text.at("hybrid_override_pattern").is_string())
            throw std::runtime_error("hybrid_override_pattern must be a string");
        for(const char type:text.at("hybrid_override_pattern").get<std::string>()) {
            switch(type) {
                case 'M':schedule.emplace_back("linear_attention");break;
                case '*':schedule.emplace_back("full_attention");break;
                case '-':schedule.emplace_back("mlp");break;
                case 'E':schedule.emplace_back("moe");break;
                default:throw std::runtime_error("hybrid_override_pattern must contain only M, *, -, E");
            }
        }
    }
    if (schedule.empty())
        throw std::runtime_error("Nemotron-H config requires a non-empty layer schedule");

    Config config;
    std::vector<std::string> runtime_types;
    runtime_types.reserve(schedule.size());
    for (const auto& type : schedule) {
        const auto kind = mixer_kind(type);
        config.mixers.push_back(kind);
        runtime_types.emplace_back(mixer_name(kind));
    }
    const auto declared_layers = integer(text, "num_hidden_layers", 0);
    if (declared_layers > 0 && declared_layers != static_cast<std::int64_t>(schedule.size()))
        throw std::runtime_error(
            "Nemotron-H num_hidden_layers disagrees with layers_block_type");
    normalized["num_hidden_layers"] = static_cast<std::int64_t>(schedule.size());
    normalized["layer_types"] = runtime_types;
    // Released checkpoints carry no positional encoding on any attention layer.
    normalized["positional_encoding"] = "none";
    if (!text.contains("rms_norm_eps")) {
        if (!text.contains("layer_norm_epsilon"))
            throw std::runtime_error("Nemotron-H config requires layer_norm_epsilon");
        normalized["rms_norm_eps"] = text.at("layer_norm_epsilon");
    }
    if (!text.contains("tie_word_embeddings") && document.contains("tie_word_embeddings"))
        normalized["tie_word_embeddings"] = document.at("tie_word_embeddings");
    static_cast<ModelConfig&>(config) = ModelConfig::from_json(normalized.dump());

    // One predictor stage holds every declared sub-block; the schedule alone
    // decides whether that stage contains an attention, MoE or dense mixer.
    config.num_nextn_predict_layers = integer(text, "num_nextn_predict_layers", 0);
    if (config.num_nextn_predict_layers < 0)
        throw std::runtime_error("Nemotron-H num_nextn_predict_layers must not be negative");
    if (config.num_nextn_predict_layers > 1)
        throw std::runtime_error(
            "Nemotron-H supports one multi-token-prediction stage; its sub-blocks are listed "
            "in mtp_layers_block_type");
    if (config.num_nextn_predict_layers > 0) {
        for (const auto& type : strings(text, "mtp_layers_block_type")) {
            const auto kind = mixer_kind(type);
            if (kind == MixerKind::linear_attention)
                throw std::runtime_error(
                    "Nemotron-H MTP cannot declare a mamba sub-block: the predictor keeps no "
                    "recurrent state in the shared predictor graph");
            config.mtp_mixers.push_back(kind);
        }
        if (config.mtp_mixers.empty())
            throw std::runtime_error("Nemotron-H mtp_layers_block_type must not be empty");
    } else if (text.contains("mtp_layers_block_type")) {
        throw std::runtime_error(
            "Nemotron-H mtp_layers_block_type requires num_nextn_predict_layers");
    }

    if (text.value("mamba_hidden_act", std::string("silu")) != "silu")
        throw std::runtime_error("Nemotron-H mamba activation must be silu");
    if (text.value("mamba_ssm_cache_dtype", std::string("float32")) != "float32")
        throw std::runtime_error("Nemotron-H requires a float32 SSM cache");
    config.mamba_heads = integer(text, "mamba_num_heads", 0);
    config.mamba_head_dim = integer(text, "mamba_head_dim", 0);
    config.ssm_state_size = integer(text, "ssm_state_size", 0);
    config.mamba_groups = integer(text, "n_groups", 0);
    config.conv_kernel = integer(text, "conv_kernel", 0);
    config.conv_bias = text.value("use_conv_bias", true);
    const bool has_mamba = std::any_of(config.mixers.begin(), config.mixers.end(), [](auto kind) {
        return kind == MixerKind::linear_attention;
    });
    if (has_mamba) {
        if (config.mamba_heads <= 0 || config.mamba_head_dim <= 0 ||
            config.ssm_state_size <= 0 || config.mamba_groups <= 0 || config.conv_kernel <= 0)
            throw std::runtime_error("Nemotron-H mamba geometry must be positive");
        if (config.mamba_heads % config.mamba_groups != 0)
            throw std::runtime_error("Nemotron-H mamba heads must divide into n_groups");
        // The shared selective scan contracts one warp lane per state column.
        if (config.ssm_state_size % 32 != 0)
            throw std::runtime_error("Nemotron-H ssm_state_size must be a multiple of 32");
        const auto limit = text.find("time_step_limit");
        if(limit==text.end()) {
            config.time_step_limit_lo=0.0;
            config.time_step_limit_hi=std::numeric_limits<double>::infinity();
        } else {
            if(!limit->is_array() || limit->size()!=2)
                throw std::runtime_error("Nemotron-H config requires time_step_limit [low, high]");
            config.time_step_limit_lo=number((*limit)[0],"time_step_limit[0]");
            config.time_step_limit_hi=number((*limit)[1],"time_step_limit[1]");
        }
        if (!(config.time_step_limit_lo <= config.time_step_limit_hi))
            throw std::runtime_error("Nemotron-H time_step_limit low must not exceed high");
    }

    const auto needs_moe = std::any_of(config.mixers.begin(), config.mixers.end(), [](auto kind) {
        return kind == MixerKind::moe;
    }) || std::any_of(config.mtp_mixers.begin(), config.mtp_mixers.end(), [](auto kind) {
        return kind == MixerKind::moe;
    });
    if (!needs_moe)
        return config;
    if (text.value("mlp_hidden_act", std::string("relu2")) != "relu2")
        throw std::runtime_error(
            "Nemotron-H routed experts require the relu-squared activation");
    config.num_routed_experts = integer(text, "n_routed_experts", 0);
    config.num_experts_per_tok = integer(text, "num_experts_per_tok", 0);
    config.moe_intermediate_size = integer(text, "moe_intermediate_size", 0);
    config.moe_latent_size = integer(text, "moe_latent_size", 0);
    config.moe_shared_intermediate_size =
        integer(text, "moe_shared_expert_intermediate_size", 0);
    config.norm_topk_prob = text.value("norm_topk_prob", true);
    config.routed_scaling_factor = text.value("routed_scaling_factor", 1.0);
    if (config.num_routed_experts <= 0 || config.num_experts_per_tok <= 0 ||
        config.num_experts_per_tok > config.num_routed_experts ||
        config.moe_intermediate_size <= 0 || config.moe_latent_size <= 0 ||
        config.moe_shared_intermediate_size <= 0)
        throw std::runtime_error("Nemotron-H latent MoE geometry must be positive");
    const auto shared_experts = integer(text, "n_shared_experts", 0);
    if (shared_experts != 1)
        throw std::runtime_error("Nemotron-H latent MoE requires exactly one shared expert");
    const auto expert_groups = integer(text, "n_group", 1);
    const auto expert_group_topk = integer(text, "topk_group", 1);
    if (expert_groups != 1 || expert_group_topk != 1)
        throw std::runtime_error(
            "Nemotron-H group-limited routing is not supported: n_group and topk_group "
            "must both be 1");
    return config;
}

} // namespace mfq::models::nemotron_h
