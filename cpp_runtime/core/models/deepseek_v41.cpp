#include "models/deepseek_v41.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mfq::models::deepseek_v41 {
namespace {

using json = nlohmann::json;

const json& object_at(const json& root, const char* name) {
    const auto found = root.find(name);
    if (found == root.end() || !found->is_object()) {
        throw std::runtime_error(
            std::string("DeepSeek-V4.1 config requires object ") + name);
    }
    return *found;
}

std::int64_t positive(const json& value, const char* name) {
    const auto found = value.find(name);
    if (found == value.end() || !found->is_number_integer() ||
        found->get<std::int64_t>() <= 0) {
        throw std::runtime_error(
            std::string("DeepSeek-V4.1 config requires positive ") + name);
    }
    return found->get<std::int64_t>();
}

std::vector<std::int64_t> integers(
    const json& value,
    const char* name) {
    const auto found = value.find(name);
    if (found == value.end() || !found->is_array()) {
        throw std::runtime_error(
            std::string("DeepSeek-V4.1 config requires array ") + name);
    }
    std::vector<std::int64_t> result;
    result.reserve(found->size());
    for (const auto& item : *found) {
        if (!item.is_number_integer()) {
            throw std::runtime_error(
                std::string("DeepSeek-V4.1 config array is not integral: ") + name);
        }
        result.push_back(item.get<std::int64_t>());
    }
    return result;
}

std::vector<std::int64_t> eos_ids(const json& root) {
    const auto found = root.find("eos_token_id");
    if (found == root.end() || found->is_null()) return {};
    if (found->is_number_integer()) {
        return {found->get<std::int64_t>()};
    }
    if (!found->is_array()) {
        throw std::runtime_error(
            "DeepSeek-V4.1 eos_token_id must be an integer or array");
    }
    std::vector<std::int64_t> result;
    for (const auto& item : *found) {
        if (!item.is_number_integer()) {
            throw std::runtime_error(
                "DeepSeek-V4.1 eos_token_id array is not integral");
        }
        result.push_back(item.get<std::int64_t>());
    }
    return result;
}

bool contains(
    const std::vector<std::int64_t>& values,
    std::int64_t value) noexcept {
    return std::find(values.begin(), values.end(), value) != values.end();
}

void require_layers(
    const std::vector<std::int64_t>& values,
    std::int64_t layers,
    const char* description) {
    if (std::any_of(
            values.begin(), values.end(),
            [layers](std::int64_t value) {
                return value < 0 || value >= layers;
            })) {
        throw std::runtime_error(
            std::string("DeepSeek-V4.1 ") + description +
            " contains an invalid layer");
    }
}

} // namespace

Config Config::from_json(std::string_view payload) {
    json root;
    try {
        root = json::parse(payload.begin(), payload.end());
    } catch (const json::exception& error) {
        throw std::runtime_error(
            std::string("invalid DeepSeek-V4.1 config JSON: ") + error.what());
    }
    if (!root.is_object() ||
        root.value("model_type", std::string{}) != "deepseek_v41") {
        throw std::runtime_error(
            "DeepSeek-V4.1 runtime requires model_type=deepseek_v41");
    }
    const auto& text = object_at(root, "text_config");
    if (text.value("model_type", std::string{}) != "deepseek_v41_text") {
        throw std::runtime_error(
            "DeepSeek-V4.1 runtime requires text model_type=deepseek_v41_text");
    }
    const auto& vision = object_at(root, "vision_config");
    if (vision.value("model_type", std::string{}) != "deepseek_v41_vision") {
        throw std::runtime_error(
            "DeepSeek-V4.1 runtime requires its native vision tower");
    }
    const auto& quantization = object_at(root, "quantization_config");
    const auto blocks = integers(quantization, "weight_block_size");
    if (quantization.value("quant_method", std::string{}) != "fp8" ||
        blocks != std::vector<std::int64_t>{32, 32} ||
        quantization.value("scale_fmt", std::string{}) != "ue8m0" ||
        quantization.value("expert_dtype", std::string{}) != "fp4") {
        throw std::runtime_error(
            "DeepSeek-V4.1 runtime does not support this source encoding");
    }

    Config config;
    config.vocab = positive(text, "vocab_size");
    config.hidden = positive(text, "hidden_size");
    config.n_layers = positive(text, "num_hidden_layers");
    config.n_heads = positive(text, "num_attention_heads");
    config.n_kv_heads = positive(text, "num_key_value_heads");
    config.head_dim = positive(text, "head_dim");
    config.rope_head_dim = positive(text, "qk_rope_head_dim");
    config.q_lora_rank = positive(text, "q_lora_rank");
    config.o_lora_rank = positive(text, "o_lora_rank");
    config.o_groups = positive(text, "o_groups");
    config.moe_inter = positive(text, "moe_intermediate_size");
    config.n_experts = positive(text, "n_routed_experts");
    config.n_shared = positive(text, "n_shared_experts");
    config.top_k = positive(text, "num_experts_per_tok");
    config.rms_eps = text.value("rms_norm_eps", 1e-20);
    config.routed_scaling = text.value("routed_scaling_factor", 1.0);
    config.swiglu_limit = text.value("swiglu_limit", 0.0);
    config.norm_topk_prob = text.value("norm_topk_prob", true);
    config.sliding_window = positive(text, "sliding_window");
    config.max_position_embeddings = positive(text, "max_position_embeddings");
    config.rope_theta = text.value("rope_theta", 10'000.0);
    config.compress_rope_theta = text.value("compress_rope_theta", 160'000.0);
    config.compress_ratios = integers(text, "compress_ratios");
    const auto decoder = std::find(
        config.compress_ratios.begin(), config.compress_ratios.end(), 1);
    config.causal_encoder_layers = decoder == config.compress_ratios.end()
        ? 0
        : std::distance(config.compress_ratios.begin(), decoder);
    config.kv_source_layers = integers(text, "kv_source_layer_ids");
    config.index_source_layers = integers(text, "index_source_layer_ids");
    config.index_n_heads = positive(text, "index_n_heads");
    config.index_head_dim = positive(text, "index_head_dim");
    config.index_topk = positive(text, "index_topk");
    config.candidate_source_layer = text.value("candidate_source_layer_id", -1);
    config.candidate_topk_blocks = positive(text, "candidate_topk_blocks");
    config.candidate_block_size = positive(text, "candidate_block_size");
    config.hc_mult = positive(text, "hc_mult");
    config.hc_sinkhorn_iters = positive(text, "hc_sinkhorn_iters");
    config.hc_eps = text.value("hc_eps", 1e-6);
    config.engram_layer_ids = integers(text, "engram_layer_ids");
    config.engram_num_embeddings = integers(text, "engram_num_embeddings");
    config.engram_max_ngram_size = positive(text, "engram_max_ngram_size");
    config.engram_vocab_size = positive(text, "engram_vocab_size");
    config.engram_n_heads = positive(text, "engram_n_heads");
    config.engram_head_dim = positive(text, "engram_head_dim");
    config.engram_pad_token_id = text.value("engram_pad_token_id", 0);
    config.engram_compressed_vocab_size = positive(
        text, "engram_compressed_vocab_size");
    config.n_mtp_layers = positive(text, "num_nextn_predict_layers");
    config.dspark_block_size = positive(text, "dspark_block_size");
    config.dspark_noise_token_id = text.value("dspark_noise_token_id", 0);
    config.dspark_target_layer_ids = integers(text, "dspark_target_layer_ids");
    config.dspark_markov_rank = positive(text, "dspark_markov_rank");
    config.dspark_n_experts = positive(text, "dspark_n_routed_experts");
    config.dspark_top_k = positive(text, "dspark_num_experts_per_tok");
    config.image_token_id = root.value("image_token_id", -1);
    config.eos_token_ids = eos_ids(root);

    const auto& rope = object_at(text, "rope_scaling");
    if (rope.value("rope_type", std::string{}) != "yarn") {
        throw std::runtime_error("DeepSeek-V4.1 runtime requires YaRN RoPE");
    }
    config.rope_scaling.factor = rope.value("factor", 1.0);
    config.rope_scaling.beta_fast = rope.value("beta_fast", 32.0);
    config.rope_scaling.beta_slow = rope.value("beta_slow", 1.0);
    config.rope_scaling.original_max_position_embeddings =
        positive(rope, "original_max_position_embeddings");

    config.vision.n_layers = positive(vision, "num_hidden_layers");
    config.vision.hidden = positive(vision, "hidden_size");
    config.vision.n_heads = positive(vision, "num_attention_heads");
    config.vision.intermediate = positive(vision, "intermediate_size");
    config.vision.patch_size = positive(vision, "patch_size");
    config.vision.downsample_ratio = positive(vision, "downsample_ratio");
    config.vision.max_image_tokens = positive(vision, "max_image_tokens");
    config.vision.min_pixels = positive(vision, "min_pixels");
    config.vision.rope_theta = vision.value("rope_theta", 10'000.0);
    config.validate();
    return config;
}

void Config::validate() const {
    if (vocab <= 0 || hidden <= 0 || n_layers <= 0 || n_heads <= 0 ||
        n_kv_heads <= 0 || head_dim <= 0 || rope_head_dim <= 0 ||
        q_lora_rank <= 0 || o_lora_rank <= 0 || o_groups <= 0 ||
        moe_inter <= 0 || n_experts <= 0 || n_shared <= 0 || top_k <= 0 ||
        top_k > n_experts || n_heads % n_kv_heads || n_heads % o_groups ||
        rope_head_dim > head_dim || rope_head_dim % 2 || rms_eps <= 0.0 ||
        sliding_window <= 0 || max_position_embeddings <= 0 ||
        hc_mult <= 0 || hc_sinkhorn_iters <= 0 || hc_eps <= 0.0) {
        throw std::runtime_error("invalid DeepSeek-V4.1 model dimensions");
    }
    if (compress_ratios.size() !=
        static_cast<std::size_t>(n_layers + n_mtp_layers)) {
        throw std::runtime_error(
            "DeepSeek-V4.1 compression schedule has the wrong length");
    }
    const auto decoder = std::find(
        compress_ratios.begin(),
        compress_ratios.begin() + n_layers,
        1);
    if (decoder == compress_ratios.begin() + n_layers) {
        throw std::runtime_error("DeepSeek-V4.1 CED schedule has no decoder");
    }
    const auto derived_encoder = std::distance(compress_ratios.begin(), decoder);
    if (causal_encoder_layers != 0 && causal_encoder_layers != derived_encoder) {
        throw std::runtime_error("DeepSeek-V4.1 CED split is inconsistent");
    }
    for (std::int64_t layer = 0; layer < n_layers; ++layer) {
        const auto ratio = compress_ratios[static_cast<std::size_t>(layer)];
        if (ratio < 0 || ratio > 2 ||
            (layer >= derived_encoder && ratio != 1)) {
            throw std::runtime_error("DeepSeek-V4.1 CED compression ratio is invalid");
        }
    }
    for (std::int64_t layer = n_layers;
         layer < n_layers + n_mtp_layers; ++layer) {
        if (compress_ratios[static_cast<std::size_t>(layer)] != 0) {
            throw std::runtime_error(
                "DeepSeek-V4.1 DSpark compression ratio must be zero");
        }
    }
    require_layers(kv_source_layers, n_layers, "KV source schedule");
    require_layers(index_source_layers, n_layers, "index source schedule");
    require_layers(engram_layer_ids, n_layers, "Engram schedule");
    require_layers(dspark_target_layer_ids, n_layers, "DSpark target schedule");
    if (kv_source_layers.empty() ||
        !std::all_of(
            kv_source_layers.begin(), kv_source_layers.end(),
            [this](std::int64_t layer) { return is_index_source(layer); }) ||
        !is_index_source(candidate_source_layer) ||
        engram_layer_ids.size() != engram_num_embeddings.size() ||
        dspark_target_layer_ids.size() != static_cast<std::size_t>(n_mtp_layers)) {
        throw std::runtime_error("DeepSeek-V4.1 architecture schedules disagree");
    }
    std::int64_t active_ratio = 0;
    for (std::int64_t layer = 0; layer < n_layers; ++layer) {
        const auto ratio = compress_ratios[static_cast<std::size_t>(layer)];
        if (is_kv_source(layer)) {
            if (ratio <= 0) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 KV source has no compressed stream");
            }
            active_ratio = ratio;
        }
        if (ratio > 0 && active_ratio != ratio) {
            throw std::runtime_error(
                "DeepSeek-V4.1 CSA2 consumer has no compatible KV source");
        }
    }
}

bool Config::has_engram(std::int64_t layer) const noexcept {
    return contains(engram_layer_ids, layer);
}

bool Config::is_kv_source(std::int64_t layer) const noexcept {
    return contains(kv_source_layers, layer);
}

bool Config::is_index_source(std::int64_t layer) const noexcept {
    return contains(index_source_layers, layer);
}

std::string TensorNames::layer(
    std::size_t index,
    std::string_view suffix) {
    return "model.block." + std::to_string(index) + "." + std::string(suffix);
}

std::string TensorNames::predictor(
    std::size_t index,
    std::string_view suffix) {
    return "predictor.stage." + std::to_string(index) + "." + std::string(suffix);
}

} // namespace mfq::models::deepseek_v41
