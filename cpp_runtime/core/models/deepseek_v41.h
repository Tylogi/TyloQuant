#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mfq::models::deepseek_v41 {

struct VisionConfig {
    std::int64_t n_layers = 0;
    std::int64_t hidden = 0;
    std::int64_t n_heads = 0;
    std::int64_t intermediate = 0;
    std::int64_t patch_size = 0;
    std::int64_t downsample_ratio = 0;
    std::int64_t max_image_tokens = 0;
    std::int64_t min_pixels = 0;
    double rope_theta = 10'000.0;
};

struct RopeScaling {
    double factor = 1.0;
    double beta_fast = 32.0;
    double beta_slow = 1.0;
    std::int64_t original_max_position_embeddings = 0;
};

// Backend-neutral, normalized contract for the CED/CSA2 DeepSeek-V4.1
// family. Backends share this parser so schedule validation cannot drift.
struct Config {
    std::string model_type = "deepseek_v41";
    std::string text_model_type = "deepseek_v41_text";
    std::int64_t vocab = 0;
    std::int64_t hidden = 0;
    std::int64_t n_layers = 0;
    std::int64_t causal_encoder_layers = 0;
    std::int64_t n_heads = 0;
    std::int64_t n_kv_heads = 0;
    std::int64_t head_dim = 0;
    std::int64_t rope_head_dim = 0;
    std::int64_t q_lora_rank = 0;
    std::int64_t o_lora_rank = 0;
    std::int64_t o_groups = 0;
    std::int64_t moe_inter = 0;
    std::int64_t n_experts = 0;
    std::int64_t n_shared = 0;
    std::int64_t top_k = 0;
    double rms_eps = 1e-20;
    double routed_scaling = 1.0;
    double swiglu_limit = 0.0;
    bool norm_topk_prob = true;
    std::int64_t sliding_window = 0;
    std::int64_t max_position_embeddings = 0;
    double rope_theta = 10'000.0;
    RopeScaling rope_scaling;
    double compress_rope_theta = 160'000.0;
    std::vector<std::int64_t> compress_ratios;
    std::vector<std::int64_t> kv_source_layers;
    std::vector<std::int64_t> index_source_layers;
    std::int64_t index_n_heads = 0;
    std::int64_t index_head_dim = 0;
    std::int64_t index_topk = 0;
    std::int64_t candidate_source_layer = -1;
    std::int64_t candidate_topk_blocks = 0;
    std::int64_t candidate_block_size = 0;
    std::int64_t hc_mult = 0;
    std::int64_t hc_sinkhorn_iters = 0;
    double hc_eps = 1e-6;
    std::vector<std::int64_t> engram_layer_ids;
    std::vector<std::int64_t> engram_num_embeddings;
    std::int64_t engram_max_ngram_size = 0;
    std::int64_t engram_vocab_size = 0;
    std::int64_t engram_n_heads = 0;
    std::int64_t engram_head_dim = 0;
    std::int64_t engram_pad_token_id = 0;
    std::int64_t engram_compressed_vocab_size = 0;
    std::int64_t n_mtp_layers = 0;
    std::int64_t dspark_block_size = 0;
    std::int64_t dspark_noise_token_id = 0;
    std::vector<std::int64_t> dspark_target_layer_ids;
    std::int64_t dspark_markov_rank = 0;
    std::int64_t dspark_n_experts = 0;
    std::int64_t dspark_top_k = 0;
    std::int64_t image_token_id = -1;
    std::vector<std::int64_t> eos_token_ids;
    VisionConfig vision;

    static Config from_json(std::string_view payload);

    template <class Container>
    static Config from_mfq(const Container& model) {
        const auto graph = model.model_graph();
        if (!graph || graph->backbone != "deepseek_v41") {
            throw std::runtime_error(
                "DeepSeek-V4.1 C++ loading requires a deepseek_v41 model graph");
        }
        constexpr const char* asset = "__mfq_asset__/model_config.json";
        if (!model.contains(asset)) {
            throw std::runtime_error(
                "DeepSeek-V4.1 MFQ has no embedded model_config.json asset");
        }
        return from_json(model.read_text(asset));
    }

    void validate() const;

    bool has_vision() const noexcept { return vision.n_layers > 0; }
    bool has_dspark() const noexcept {
        return n_mtp_layers > 0 && dspark_block_size > 0;
    }
    bool has_engram(std::int64_t layer) const noexcept;
    bool is_kv_source(std::int64_t layer) const noexcept;
    bool is_index_source(std::int64_t layer) const noexcept;
};

struct TensorNames {
    static std::string layer(std::size_t index, std::string_view suffix);
    static std::string predictor(std::size_t index, std::string_view suffix);
};

} // namespace mfq::models::deepseek_v41
