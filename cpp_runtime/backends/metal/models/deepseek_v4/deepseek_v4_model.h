#pragma once

#include "mfq_container.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mfq::metal {

struct DeepseekV4RopeScaling {
    bool enabled = false;
    std::string type;
    double factor = 1.0;
    double beta_fast = 32.0;
    double beta_slow = 1.0;
    std::int64_t original_max_position_embeddings = 0;
};

// Normalized configuration shared by manifest-form TPQ archives and raw
// Hugging Face DeepSeek-V4 config.json files.
struct DeepseekV4Config {
    std::string model_type = "deepseek_v4";
    std::int64_t n_layers = 0;
    std::int64_t hidden = 0;
    std::int64_t n_experts = 0;
    std::int64_t top_k = 0;
    std::int64_t moe_inter = 0;
    std::int64_t n_shared = 1;
    std::int64_t n_heads = 0;
    std::int64_t head_dim = 512;
    std::int64_t q_lora_rank = 0;
    std::int64_t o_lora_rank = 0;
    std::int64_t o_groups = 1;
    std::int64_t kv_dim = 0;
    std::int64_t qk_rope_head_dim = 0;
    std::int64_t n_kv_heads = 1;
    std::int64_t vocab = 0;
    double rms_eps = 1e-6;
    std::string scoring_func = "sqrtsoftplus";
    bool norm_topk_prob = true;
    double routed_scaling = 1.0;
    double swiglu_limit = 0.0;
    std::int64_t n_hash_layers = 0;
    std::int64_t sliding_window = 128;
    double rope_theta = 10'000.0;
    DeepseekV4RopeScaling rope_scaling;
    std::vector<std::int64_t> eos_token_id;
    std::int64_t index_n_heads = 64;
    std::int64_t index_head_dim = 128;
    std::int64_t index_topk = 512;
    std::vector<std::int64_t> kv_source_layer_ids;
    std::vector<std::int64_t> index_source_layer_ids;
    std::int64_t candidate_source_layer_id = -1;
    std::int64_t candidate_topk_blocks = 0;
    std::int64_t candidate_block_size = 0;
    std::int64_t max_position_embeddings = 1'048'576;
    std::int64_t hc_mult = 4;
    double hc_eps = 1e-6;
    std::int64_t hc_sinkhorn_iters = 20;
    double compress_rope_theta = 160'000.0;
    std::vector<std::int64_t> compress_ratios;

    // V4.1 sparse n-gram memory. The two checkpoint embedding tables remain
    // on SSD; only the rows selected for the current request are admitted to
    // a bounded host/UMA cache.
    std::vector<std::int64_t> engram_layer_ids;
    std::vector<std::int64_t> engram_num_embeddings;
    std::int64_t engram_max_ngram_size = 1;
    std::int64_t engram_vocab_size = 0;
    std::int64_t engram_n_heads = 0;
    std::int64_t engram_head_dim = 0;
    std::int64_t engram_pad_token_id = 2;
    std::int64_t engram_compressed_vocab_size = 0;

    // DSpark speculative decoder.  Unlike Qwen's dense MTP head, every stage
    // is a complete DeepSeek hyper-connection + attention + MoE block.
    std::int64_t n_mtp_layers = 0;
    std::int64_t dspark_block_size = 0;
    std::int64_t dspark_noise_token_id = 0;
    std::vector<std::int64_t> dspark_target_layer_ids;
    std::int64_t dspark_markov_rank = 256;
    std::int64_t dspark_n_experts = 0;
    std::int64_t dspark_top_k = 0;
    std::vector<std::int64_t> mtp_compress_ratios;

    // Optional native DeepSeek-V4 vision tower and aligner.
    std::int64_t vision_n_layers = 0;
    std::int64_t vision_dim = 1024;
    std::int64_t vision_n_heads = 16;
    std::int64_t vision_inter_dim = 2816;
    std::int64_t vision_patch_size = 14;
    double vision_rope_theta = 10'000.0;
    std::int64_t vision_downsample_ratio = 3;
    std::int64_t vision_max_n_token = 384;
    std::int64_t vision_min_pixels = 147'456;
    std::int64_t vision_max_wh_ratio = 8;
    std::int64_t image_token_id = -1;

    // Accepts a normalized manifest config, a complete TPQ manifest, or a
    // Hugging Face config. Alias fields are normalized to the members above.
    static DeepseekV4Config from_json(std::string_view payload);

    // Requires a native DeepSeek-V4 TPQ MFQ header and reads the normalized
    // Canonical artifacts load the shared model_config.json asset. The
    // header TPQ manifest is accepted only as pre-schema compatibility.
    static DeepseekV4Config from_mfq(const MfqContainer& model);

    void validate() const;

    std::int64_t attention_size() const noexcept {
        return n_heads * head_dim;
    }
    std::int64_t shared_intermediate_size() const noexcept {
        return n_shared * moe_inter;
    }
    std::int64_t hyper_connection_projection_size() const noexcept {
        return hc_mult * hc_mult + 2 * hc_mult;
    }
    bool fast_attention() const noexcept {
        return n_heads == 64 && head_dim == 512;
    }
    bool fast_hyper_connections() const noexcept {
        return hidden == 4096;
    }
    bool fast_indexer() const noexcept {
        return index_n_heads == 64 &&
            index_head_dim == 128;
    }
    bool has_vision() const noexcept {
        return vision_n_layers > 0;
    }
    bool has_dspark() const noexcept {
        return n_mtp_layers > 0 && dspark_block_size > 0;
    }
    bool is_v41() const noexcept {
        return model_type == "deepseek_v41";
    }
    bool has_engram() const noexcept {
        return !engram_layer_ids.empty();
    }
    bool is_kv_source(std::size_t layer) const noexcept;
    bool is_index_source(std::size_t layer) const noexcept;
    std::int64_t experts_for_layer(std::size_t layer) const noexcept {
        return layer < static_cast<std::size_t>(n_layers) || dspark_n_experts <= 0
            ? n_experts
            : dspark_n_experts;
    }
};

struct DeepseekV4TensorNames {
    std::string embedding = "model.token_embedding.weight";
    std::string output_norm = "model.output_norm.weight";
    std::string output = "model.output.weight";
    std::string hc_head_fn = "model.mhc.output.function";
    std::string hc_head_base = "model.mhc.output.base";
    std::string hc_head_scale = "model.mhc.output.scale";

    static std::string layer(
        std::size_t index,
        std::string_view suffix);

    static std::vector<std::string> required(
        const DeepseekV4Config& config);
};

enum class DeepseekV4TensorKind {
    embedding,
    linear,
    dense_float,
    dense_integer,
    routed_experts,
};

struct DeepseekV4TensorBinding {
    std::string name;
    std::vector<std::int64_t> shape;
    DeepseekV4TensorKind kind = DeepseekV4TensorKind::linear;
};

struct DeepseekV4TensorMetadata {
    std::string dtype;
    std::vector<std::int64_t> shape;
    bool packed = false;
};

// The canonical schedule is ratio-dependent:
//   0   -> local attention only
//   4   -> main compressor plus Indexer compressor/query/weights
//   128 -> main compressor without the Indexer branch
std::vector<DeepseekV4TensorBinding>
deepseek_v4_required_bindings(
    const DeepseekV4Config& config,
    const DeepseekV4TensorNames& names = {});

DeepseekV4TensorMetadata inspect_deepseek_v4_tensor_metadata(
    const MfqContainer& model,
    const std::string& name);

void validate_deepseek_v4_model_bindings(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    const DeepseekV4TensorNames& names = {});

using MlxDeepseekV4Config = DeepseekV4Config;
using MlxDeepseekV4Names = DeepseekV4TensorNames;

} // namespace mfq::metal
