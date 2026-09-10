#include "mfq_legacy_model_graph.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

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

const json& object_or(const json& parent, const char* key) {
    const auto found = parent.find(key);
    return found != parent.end() && found->is_object() ? *found : parent;
}

std::string string_value(const json& value, const char* key) {
    const auto found = value.find(key);
    return found != value.end() && found->is_string()
        ? found->get<std::string>() : std::string{};
}

std::int64_t integer_value(
        const json& value, const char* key, std::int64_t fallback = 0) {
    const auto found = value.find(key);
    return found != value.end() && found->is_number_integer()
        ? found->get<std::int64_t>() : fallback;
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.substr(0, prefix.size()) == prefix;
}

} // namespace

MfqModelGraph synthesize_legacy_model_graph(
    std::string_view artifact_architecture,
    std::string_view model_config_json,
    const std::function<bool(std::string_view)>& contains_tensor) {
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
    const auto& text = object_or(config, "text_config");
    const auto& vision = object_or(config, "vision_config");
    const auto config_family = identity(string_value(config, "model_type"));
    const auto text_family = identity(string_value(text, "model_type"));
    const auto stored_family = identity(artifact_architecture);

    std::string family;
    std::string backbone;
    if (starts_with(text_family, "qwen3_5") ||
        starts_with(stored_family, "qwen3_5") ||
        starts_with(stored_family, "qwen35") ||
        starts_with(stored_family, "qwen3_6") ||
        starts_with(stored_family, "qwen3_8")) {
        family = backbone = "qwen3_5";
    } else if (starts_with(text_family, "qwen4_exp") ||
               starts_with(config_family, "qwen4_exp") ||
               starts_with(stored_family, "qwen4_exp")) {
        family = backbone = "qwen4_exp";
    } else if (starts_with(text_family, "glm5_next") ||
               starts_with(config_family, "glm5_next") ||
               starts_with(stored_family, "glm5_next")) {
        family = backbone = "glm5_next";
    } else if (starts_with(text_family, "deepseek_v41") ||
               starts_with(config_family, "deepseek_v41") ||
               starts_with(stored_family, "deepseek_v41")) {
        family = backbone = "deepseek_v41";
    } else if (starts_with(text_family, "deepseek_v4") ||
               starts_with(config_family, "deepseek_v4") ||
               starts_with(stored_family, "deepseek_v4")) {
        family = backbone = "deepseek_v4";
    } else if (starts_with(config_family, "minicpmtts") ||
               starts_with(stored_family, "minicpmtts")) {
        family = backbone = "minicpmo_tts";
    } else if (starts_with(config_family, "minicpmo") ||
               starts_with(stored_family, "minicpmo")) {
        family = "minicpmo";
        backbone = "minicpmo45";
    } else if (starts_with(text_family, "glm_moe_dsa") ||
               starts_with(stored_family, "glm_moe_dsa")) {
        family = backbone = "glm_dsa";
    } else if (starts_with(text_family, "gemma4") ||
               starts_with(stored_family, "gemma4")) {
        family = backbone = "gemma4";
    } else {
        // Pre-schema dense runtimes historically accepted Qwen-compatible
        // configs without registering every checkpoint alias.  Keep that
        // behaviour inside this compatibility boundary, but normalize it to a
        // backend-neutral semantic implementation.
        family = !config_family.empty() ? config_family : stored_family;
        if (family.empty()) family = "legacy_causal_lm";
        backbone = "generic_qwen";
    }

    MfqModelGraph graph;
    graph.schema_version = 1;
    graph.architecture = family;
    graph.graph_kind = "causal_lm";
    graph.backbone = backbone;
    graph.tensor_namespace = std::string(kMfqCanonicalTensorNamespace);
    graph.tensor_naming_version = 1;
    graph.topology.text_layers = std::max<std::int64_t>(
        1, integer_value(text, "num_hidden_layers", 1));
    graph.component_roots.push_back("model");
    graph.components.push_back({
        "text", "model", backbone, "decoder", {}, {}, {}});
    graph.capabilities.push_back("text");

    auto add_optional = [&](
            std::string kind, std::string root, std::string implementation,
            std::string input_contract = {},
            std::string position_policy = {}) mutable {
        graph.component_roots.push_back(root);
        graph.components.push_back({
            kind, std::move(root), std::move(implementation), "optional",
            std::move(input_contract), std::move(position_policy), {}});
        if (kind == "vision") graph.capabilities.push_back("vision");
        if (kind == "predictor") {
            graph.capabilities.push_back("speculative_prediction");
        }
    };

    const bool qwen_grid = family == "qwen3_5" || family == "qwen4_exp";
    const bool glm_grid = family == "glm5_next";
    const bool has_grid_vision =
        contains_tensor("vision.patch_embedding.weight") ||
        contains_tensor("model.visual.patch_embed.proj.weight");
    if ((qwen_grid || glm_grid) && has_grid_vision) {
        graph.topology.vision_layers = std::max<std::int64_t>(
            1, integer_value(vision, "depth",
                integer_value(vision, "num_hidden_layers", 1)));
        add_optional(
            "vision", "vision", "grid_vit",
            std::string(kMfqGridVisionInputContract),
            std::string(kMfqGridMropePositionPolicy));
    }

    std::int64_t predictor_layers = integer_value(
        text, "mtp_num_hidden_layers",
        integer_value(text, "num_nextn_predict_layers", 0));
    bool has_predictor = contains_tensor("predictor.fusion.weight") ||
        contains_tensor("mtp.fc.weight") ||
        contains_tensor("mtp.fc_embedding.weight");
    if (family == "glm5_next" && predictor_layers > 0) {
        const auto prefix = "model.language_model.layers." +
            std::to_string(graph.topology.text_layers) + ".eh_proj.weight";
        has_predictor = has_predictor || contains_tensor(prefix);
    }
    if ((qwen_grid || glm_grid) && has_predictor) {
        graph.topology.predictor_layers = std::max<std::int64_t>(
            1, predictor_layers);
        add_optional(
            "predictor", "predictor", "next_token_prediction");
    }

    if (family == "deepseek_v4") {
        if (contains_tensor("vision.patch_embedding.weight") ||
            contains_tensor("vision.patch_embed.weight")) {
            graph.topology.vision_layers = 1;
            add_optional(
                "vision", "vision", "deepseek_v4_vision",
                "deepseek_v4_vision.v1", "deepseek_v4_positions");
        }
        if (contains_tensor("predictor.stage.0.main_proj.weight") ||
            contains_tensor("mtp.0.main_proj.weight")) {
            graph.topology.predictor_layers = 1;
            add_optional("predictor", "predictor", "dspark");
        }
    }

    if (family == "deepseek_v41") {
        if (contains_tensor("vision.patch_embedding.weight")) {
            graph.topology.vision_layers = 1;
            add_optional(
                "vision", "vision", "deepseek_v41_vision",
                "deepseek_v41_vision.v1", "deepseek_v41_positions");
        }
        if (contains_tensor("predictor.stage.0.main_projection.weight")) {
            graph.topology.predictor_layers = std::max<std::int64_t>(
                1, predictor_layers);
            add_optional(
                "predictor", "predictor", "deepseek_v41_dspark");
        }
    }

    if (family == "minicpmo") {
        const bool has_vision =
            contains_tensor("vision.patch_embedding.weight") ||
            contains_tensor("vpm.embeddings.patch_embedding.weight");
        const bool has_audio = contains_tensor("audio.patch_embedding.weight") ||
            contains_tensor("apm.conv1.weight");
        const bool has_tts = contains_tensor("tts.token_embedding.weight") ||
            contains_tensor("tts.emb_text.weight");
        if (has_vision) {
            graph.topology.vision_layers = 1;
            add_optional(
                "vision", "vision", "minicpmo45_vision",
                "minicpmo45.v1", "minicpmo45_positions");
        }
        if (has_audio) {
            add_optional("audio_input", "audio", "minicpmo45_audio");
        }
        if (has_tts) {
            add_optional("audio_output", "tts", "minicpmo45_tts");
        }
        if (has_audio && has_tts) {
            add_optional("duplex", "runtime", "minicpmo45_duplex");
        }
    }
    return graph;
}

} // namespace mfq
