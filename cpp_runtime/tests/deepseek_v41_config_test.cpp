#include "models/deepseek_v41.h"

#include "nlohmann/json.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using mfq::models::deepseek_v41::Config;
using mfq::models::deepseek_v41::TensorNames;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

const char* valid_config = R"json({
  "model_type": "deepseek_v41",
  "eos_token_id": [1, 2],
  "image_token_id": 129264,
  "quantization_config": {
    "quant_method": "fp8",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4"
  },
  "text_config": {
    "model_type": "deepseek_v41_text",
    "vocab_size": 129280,
    "hidden_size": 5120,
    "moe_intermediate_size": 2304,
    "num_hidden_layers": 4,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "rms_norm_eps": 1e-20,
    "max_position_embeddings": 1048576,
    "rope_theta": 10000,
    "rope_scaling": {
      "rope_type": "yarn",
      "factor": 16,
      "original_max_position_embeddings": 65536
    },
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "norm_topk_prob": true,
    "routed_scaling_factor": 1.5,
    "swiglu_limit": 10.0,
    "sliding_window": 128,
    "compress_ratios": [0, 2, 1, 1, 0, 0, 0],
    "compress_rope_theta": 160000,
    "kv_source_layer_ids": [1],
    "index_source_layer_ids": [1, 2],
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "candidate_source_layer_id": 2,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "engram_layer_ids": [1],
    "engram_num_embeddings": [384006168],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [1, 2, 3],
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3
  },
  "vision_config": {
    "model_type": "deepseek_v41_vision",
    "num_hidden_layers": 32,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "intermediate_size": 2816,
    "patch_size": 14,
    "rope_theta": 10000,
    "downsample_ratio": 3,
    "max_image_tokens": 1024,
    "min_pixels": 295936
  }
})json";

template <class Mutator>
void require_rejected(Mutator mutate, const char* message) {
    auto document = nlohmann::json::parse(valid_config);
    mutate(document);
    try {
        static_cast<void>(Config::from_json(document.dump()));
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        const auto config = Config::from_json(valid_config);
        require(config.causal_encoder_layers == 2, "CED split changed");
        require(config.compress_ratios.size() == 7, "compression schedule changed");
        require(config.is_kv_source(1), "KV source schedule was lost");
        require(config.is_index_source(2), "index source schedule was lost");
        require(config.has_engram(1), "Engram schedule was lost");
        require(config.has_dspark(), "DSpark schedule was lost");
        require(config.has_vision(), "vision contract was lost");
        require(config.eos_token_ids == std::vector<std::int64_t>({1, 2}),
                "EOS array was not preserved");
        require(TensorNames::layer(2, "attention.query.weight") ==
                    "model.block.2.attention.query.weight",
                "layer tensor name changed");
        require(TensorNames::predictor(1, "main_projection.weight") ==
                    "predictor.stage.1.main_projection.weight",
                "predictor tensor name changed");

        require_rejected(
            [](auto& root) { root["model_type"] = "deepseek_v4"; },
            "legacy DeepSeek-V4 alias was accepted");
        require_rejected(
            [](auto& root) {
                root["text_config"]["compress_ratios"] = {0, 2, 1, 0, 0, 0, 0};
            },
            "non-decoder compression after the CED split was accepted");
        require_rejected(
            [](auto& root) {
                root["text_config"]["kv_source_layer_ids"] = {0};
            },
            "KV source absent from the index schedule was accepted");
        require_rejected(
            [](auto& root) {
                root["quantization_config"]["weight_block_size"] = {128, 128};
            },
            "unsupported source encoding was accepted");

        std::cout << "DeepSeek-V4.1 common config tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DeepSeek-V4.1 common config test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
