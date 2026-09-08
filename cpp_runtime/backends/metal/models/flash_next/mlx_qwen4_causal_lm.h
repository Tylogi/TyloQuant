#pragma once

// Native Qwen4-Exp model graph and generation lifecycle.

#include "mfq_container.h"
#include "mlx_sampling.h"

#include "mfq/token_constraint.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct Qwen4Config {
    std::string model_type;
    std::string text_model_type;
    std::int64_t vocab_size = 0;
    std::int64_t hidden_size = 0;
    std::int64_t num_hidden_layers = 0;
    std::int64_t max_position_embeddings = 0;
    std::int64_t num_attention_heads = 0;
    std::int64_t num_key_value_heads = 0;
    std::int64_t head_dim = 0;
    std::int64_t rotary_dim = 0;
    std::int64_t full_attention_interval = 0;
    std::int64_t hc_count = 0;
    std::int64_t hc_lowrank = 0;
    std::int64_t linear_num_key_heads = 0;
    std::int64_t linear_num_value_heads = 0;
    std::int64_t linear_key_head_dim = 0;
    std::int64_t linear_value_head_dim = 0;
    std::int64_t linear_conv_kernel_dim = 0;
    std::int64_t num_experts = 0;
    std::int64_t num_experts_per_tok = 0;
    std::int64_t moe_intermediate_size = 0;
    std::int64_t shared_expert_intermediate_size = 0;
    std::int64_t indexer_n_heads = 0;
    std::int64_t indexer_head_dim = 0;
    std::int64_t indexer_compress_ratio = 0;
    std::int64_t indexer_budget = 0;
    std::int64_t ple_conv_kernel_size = 0;
    std::int64_t ngram_size = 0;
    std::int64_t heads_per_ngram = 0;
    std::int64_t split_ngram_parts = 0;
    std::int64_t mtp_num_hidden_layers = 0;
    std::int64_t eos_token_id = 0;
    double rms_norm_eps = 1e-6;
    double rope_theta = 1e7;
    bool mrope_interleaved = false;
    bool norm_topk_prob = true;
    bool output_gate_silu = true;
    bool tie_word_embeddings = false;
    bool mtp_use_dedicated_embeddings = false;
    std::vector<std::int64_t> rope_sections;
    std::vector<std::int64_t> ple_layer_ids;
    std::vector<std::string> layer_types;

    static Qwen4Config from_json(std::string_view payload);
    static Qwen4Config from_mfq(const MfqContainer& model);
};

// The first native landing keeps persistent session snapshots disabled.  The
// type still satisfies the common server contract without routing generation
// through a Python worker.
struct MlxQwen4TextSessionState {
    std::vector<std::int64_t> tokens;
    std::size_t bytes = 0;
};

class MlxQwen4CausalLm {
public:
    static MlxQwen4CausalLm load(
        const MfqContainer& model,
        int max_context = 4096);

    ~MlxQwen4CausalLm();
    MlxQwen4CausalLm(MlxQwen4CausalLm&&) noexcept;
    MlxQwen4CausalLm& operator=(MlxQwen4CausalLm&&) noexcept;
    MlxQwen4CausalLm(const MlxQwen4CausalLm&) = delete;
    MlxQwen4CausalLm& operator=(const MlxQwen4CausalLm&) = delete;

    mlx::core::array forward(
        const mlx::core::array& token_ids,
        bool use_cache = true);

    void reset_cache(int batch = 1);
    void clear_cache() noexcept;

    std::int32_t generate(
        const std::vector<std::int64_t>& prompt,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const std::function<bool(std::int64_t)>& callback = {},
        const std::function<void(std::size_t, double)>& prefill_callback = {},
        const MfqTokenConstraintPtr& token_constraint = {},
        std::optional<std::size_t> stable_prefix_tokens = std::nullopt);

    const Qwen4Config& config() const noexcept;
    std::size_t layer_count() const noexcept;
    int cache_position() const noexcept;
    bool supports_mtp() const noexcept { return false; }
    bool supports_multimodal() const noexcept { return false; }
    bool supports_text_session_state() const noexcept { return false; }
    MlxQwen4TextSessionState capture_text_session_state(
        const std::vector<std::int64_t>& tokens) const;
    void restore_text_session_state(const MlxQwen4TextSessionState& state);

private:
    struct Impl;
    explicit MlxQwen4CausalLm(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::metal
