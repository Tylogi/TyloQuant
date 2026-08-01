#pragma once

#include "mfq_container.h"
#include "mlx_tensor.h"
#include "mlx_transformer.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

inline constexpr std::string_view kMfqModelConfigAsset =
    "__mfq_asset__/model_config.json";

struct Qwen35Config {
    std::string model_type;
    std::string text_model_type;
    std::int64_t vocab_size = 0;
    std::int64_t hidden_size = 0;
    std::int64_t intermediate_size = 0;
    std::int64_t num_hidden_layers = 0;
    std::int64_t num_attention_heads = 0;
    std::int64_t num_key_value_heads = 0;
    std::int64_t max_position_embeddings = 0;
    std::int64_t head_dim = 0;
    double rope_base = 1'000'000.0;
    std::int64_t rotary_dim = 0;
    std::vector<std::int64_t> rope_sections;
    double rms_norm_eps = 1e-6;
    double norm_weight_offset = 1.0;
    bool tie_word_embeddings = false;
    bool attention_output_gate = false;
    std::string output_gate_type;
    std::vector<std::string> layer_types;
    std::int64_t full_attention_interval = 0;
    std::int64_t linear_conv_kernel_dim = 4;
    std::int64_t linear_key_head_dim = 128;
    std::int64_t linear_value_head_dim = 128;
    std::int64_t linear_num_key_heads = 0;
    std::int64_t linear_num_value_heads = 0;
    bool linear_a_is_log = true;
    bool mrope_interleaved = false;
    std::int64_t mtp_num_hidden_layers = 0;
    bool mtp_use_dedicated_embeddings = false;

    static Qwen35Config from_json(std::string_view payload);
    static Qwen35Config from_mfq(const MfqContainer& model);

    std::int64_t attention_size() const noexcept {
        return num_attention_heads * head_dim;
    }
    std::int64_t kv_size() const noexcept {
        return num_key_value_heads * head_dim;
    }
    std::int64_t query_projection_size() const noexcept {
        return attention_size() * (attention_output_gate ? 2 : 1);
    }
    std::int64_t linear_key_heads() const noexcept {
        return linear_num_key_heads > 0
            ? linear_num_key_heads
            : num_key_value_heads;
    }
    std::int64_t linear_value_heads() const noexcept {
        return linear_num_value_heads > 0
            ? linear_num_value_heads
            : num_attention_heads;
    }
    std::int64_t linear_key_size() const noexcept {
        return linear_key_heads() * linear_key_head_dim;
    }
    std::int64_t linear_value_size() const noexcept {
        return linear_value_heads() * linear_value_head_dim;
    }
    std::int64_t linear_qkv_size() const noexcept {
        return 2 * linear_key_size() + linear_value_size();
    }
};

struct Qwen35TensorMetadata {
    std::string dtype;
    std::vector<std::int64_t> shape;
    bool packed = false;
    std::int32_t axis = -1;
    std::int64_t neuron_len = 0;
    std::int32_t bits = 0;
    std::int32_t sub_bits = 0;
    std::int32_t group_size = 0;
    std::uint32_t output_size = 0;
    std::uint32_t groups = 0;
};

struct Qwen35ResolvedLayerNames {
    std::string attention_norm;
    std::string attention_query;
    std::string attention_key;
    std::string attention_value;
    std::string attention_output;
    std::string attention_query_norm;
    std::string attention_key_norm;
    std::string ffn_norm;
    std::string ffn_gate;
    std::string ffn_up;
    std::string ffn_down;
    std::string linear_qkv;
    std::optional<std::string> linear_qk;
    std::optional<std::string> linear_value;
    std::string linear_z;
    std::string linear_alpha;
    std::string linear_beta;
    std::string linear_conv;
    std::optional<std::string> linear_conv_bias;
    std::string linear_dt_bias;
    std::string linear_a;
    std::string linear_norm;
    std::string linear_output;
};

struct Qwen35TensorNames {
    std::string token_embedding = "token_embd.weight";
    std::string attention_norm = "blk.{i}.attn_norm.weight";
    std::string attention_query = "blk.{i}.attn_q.weight";
    std::string attention_key = "blk.{i}.attn_k.weight";
    std::string attention_value = "blk.{i}.attn_v.weight";
    std::string attention_output = "blk.{i}.attn_output.weight";
    std::string attention_query_norm = "blk.{i}.attn_q_norm.weight";
    std::string attention_key_norm = "blk.{i}.attn_k_norm.weight";
    std::string ffn_norm = "blk.{i}.ffn_norm.weight";
    std::string ffn_gate = "blk.{i}.ffn_gate.weight";
    std::string ffn_up = "blk.{i}.ffn_up.weight";
    std::string ffn_down = "blk.{i}.ffn_down.weight";
    std::string output_norm = "output_norm.weight";
    std::string output = "output.weight";
    std::string linear_qkv = "blk.{i}.ssm_qkv.weight";
    std::optional<std::string> linear_qk = "blk.{i}.ssm_qk.weight";
    std::optional<std::string> linear_value = "blk.{i}.ssm_v.weight";
    std::string linear_z = "blk.{i}.ssm_z.weight";
    std::string linear_alpha = "blk.{i}.ssm_alpha.weight";
    std::string linear_beta = "blk.{i}.ssm_beta.weight";
    std::string linear_conv = "blk.{i}.ssm_conv1d.weight";
    std::optional<std::string> linear_conv_bias;
    std::string linear_dt_bias = "blk.{i}.ssm_dt.bias";
    std::string linear_a = "blk.{i}.ssm_a";
    std::string linear_norm = "blk.{i}.ssm_norm.weight";
    std::string linear_output = "blk.{i}.ssm_out.weight";

    static Qwen35TensorNames gguf();
    static Qwen35TensorNames hugging_face();
    static Qwen35TensorNames detect(const MfqContainer& model);

    Qwen35ResolvedLayerNames layer(std::size_t index) const;
};

// GGUF Qwen3.5 exports store already-shifted RMSNorm weights and the
// materialized negative SSM decay instead of the original A_log. Match the
// CUDA runtime's layout adaptation before constructing model components.
Qwen35Config adapt_qwen35_config_for_tensor_names(
    Qwen35Config config,
    const Qwen35TensorNames& names);

Qwen35TensorMetadata inspect_qwen35_tensor_metadata(
    const MfqContainer& model,
    const std::string& name);

void validate_qwen35_model_bindings(
    const MfqContainer& model,
    const Qwen35Config& config,
    const Qwen35TensorNames& names);

using Qwen35RmsNorm = MlxRmsNorm;

MlxRmsNorm load_qwen35_rms_norm(
    const MfqContainer& model,
    const std::string& name,
    double eps,
    double weight_offset = 1.0);

class Qwen35Linear {
public:
    static Qwen35Linear load(
        const MfqContainer& model,
        const std::string& name);

    explicit Qwen35Linear(MlxLinear linear);

    mlx::core::array operator()(const mlx::core::array& input) const;

    int input_size() const noexcept {
        return linear_.input_size();
    }
    int output_size() const noexcept {
        return linear_.output_size();
    }
    bool packed() const noexcept {
        return linear_.packed();
    }

private:
    MlxLinear linear_;
};

class Qwen35Embedding {
public:
    static Qwen35Embedding load(
        const MfqContainer& model,
        const std::string& name);

    explicit Qwen35Embedding(MlxEmbedding embedding);

    mlx::core::array operator()(
        const mlx::core::array& token_ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;

    mlx::core::array project(
        const mlx::core::array& input) const;

    int vocabulary_size() const noexcept {
        return embedding_.vocabulary_size();
    }
    int hidden_size() const noexcept {
        return embedding_.hidden_size();
    }

private:
    MlxEmbedding embedding_;
};

} // namespace mfq::metal
