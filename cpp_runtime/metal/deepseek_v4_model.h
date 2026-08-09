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
    std::int64_t max_position_embeddings = 1'048'576;
    std::int64_t hc_mult = 4;
    double hc_eps = 1e-6;
    std::int64_t hc_sinkhorn_iters = 20;
    double compress_rope_theta = 160'000.0;
    std::vector<std::int64_t> compress_ratios;

    // Accepts a normalized manifest config, a complete TPQ manifest, or a
    // Hugging Face config. Alias fields are normalized to the members above.
    static DeepseekV4Config from_json(std::string_view payload);

    // Requires a native DeepSeek-V4 TPQ MFQ header and reads the normalized
    // configuration from header.extra_json["tpq_manifest"]["config"].
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
};

struct DeepseekV4TensorNames {
    std::string embedding = "embed.weight";
    std::string output_norm = "norm.weight";
    std::string output = "head.weight";
    std::string hc_head_fn = "hc_head_fn";
    std::string hc_head_base = "hc_head_base";
    std::string hc_head_scale = "hc_head_scale";

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
