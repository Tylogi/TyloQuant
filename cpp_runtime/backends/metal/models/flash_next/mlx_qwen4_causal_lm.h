#pragma once

// Native Qwen4-Exp model graph and generation lifecycle.

#include "mfq_container.h"
#include "mlx_mtp.h"
#include "mlx_prefix_cache.h"
#include "mlx_transformer.h"
#include "mlx_sampling.h"
#include "mlx_ssd_expert_cache.h"
#include "mlx_qsa_kv_offload.h"
#include "mlx_grid_vision.h"

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
    MlxYarnScaling yarn;
    double yarn_max_factor = 4.0;
    bool mrope_interleaved = false;
    bool norm_topk_prob = true;
    bool output_gate_silu = true;
    bool tie_word_embeddings = false;
    bool mtp_use_dedicated_embeddings = false;
    std::vector<std::int64_t> rope_sections;
    std::vector<std::int64_t> ple_layer_ids;
    std::vector<std::string> layer_types;
    std::optional<GridVisionConfig> vision;
    std::optional<std::int64_t> image_token_id;
    std::optional<std::int64_t> video_token_id;

    static Qwen4Config from_json(std::string_view payload);
    static Qwen4Config from_mfq(const MfqContainer& model);
};

struct MlxQwen4LayerCacheSnapshot {
    int position = 0;
    int batch = 0;
    std::optional<MlxKvCacheSnapshot> kv;
    std::optional<MlxQsaKvSnapshot> offloaded_kv;
    std::optional<MlxQsaIndexSnapshot> offloaded_pooled_keys;
    int index_start = 0;
    int index_ratio = 0;
    std::optional<mlx::core::array> index_keys;
    std::optional<mlx::core::array> pooled_keys;
    std::optional<mlx::core::array> convolution;
    std::optional<mlx::core::array> recurrent;
    std::optional<mlx::core::array> ple_convolution;
    std::vector<std::int64_t> ple_context;
};

struct MlxQwen4TextSessionState {
    std::vector<std::int64_t> tokens;
    std::vector<MlxQwen4LayerCacheSnapshot> layers;
    std::vector<MlxQwen4LayerCacheSnapshot> mtp_layers;
    std::optional<mlx::core::array> last_hidden;
    int cache_position = 0;
    int cache_batch = 0;
    std::size_t bytes = 0;
};

struct MlxQwen4RequestCache;

std::pair<std::int32_t, mlx::core::array> qwen4_mtp_logits_from_state(
    const MfqContainer& model, const MfqContainer& predictor_source,
    const MlxQwen4TextSessionState& state, bool fp16);

class MlxQwen4CausalLm {
public:
    static MlxQwen4CausalLm load(
        const MfqContainer& model,
        int max_context = 4096,
        std::optional<std::size_t> expert_cache_bytes = std::nullopt);

    ~MlxQwen4CausalLm();
    MlxQwen4CausalLm(MlxQwen4CausalLm&&) noexcept;
    MlxQwen4CausalLm& operator=(MlxQwen4CausalLm&&) noexcept;
    MlxQwen4CausalLm(const MlxQwen4CausalLm&) = delete;
    MlxQwen4CausalLm& operator=(const MlxQwen4CausalLm&) = delete;

    mlx::core::array forward(
        const mlx::core::array& token_ids,
        bool use_cache = true);

    std::shared_ptr<MlxQwen4RequestCache> make_request_cache();
    void adopt_request_cache(MlxQwen4RequestCache& cache);
    mlx::core::array prefill_request(MlxQwen4RequestCache& cache,
        const std::vector<std::int64_t>& tokens, bool mtp = false);
    std::vector<mlx::core::array> prefill_requests(
        const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
        const std::vector<std::vector<std::int64_t>>& tokens,
        const std::vector<bool>& mtp);
    mlx::core::array decode_requests(
        const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
        const std::vector<std::int64_t>& tokens);
    std::pair<mlx::core::array, mlx::core::array> forward_request_tokens(
        const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
        const std::vector<std::int64_t>& tokens);
    std::pair<mlx::core::array, mlx::core::array> verify_requests(
        const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
        const std::vector<std::vector<std::int64_t>>& tokens);
    std::vector<std::pair<mlx::core::array, mlx::core::array>> forward_request_windows(
        const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
        const std::vector<std::vector<std::int64_t>>& tokens,
        std::optional<mlx::core::array>* packed_logits = nullptr);
    void resolve_request(MlxQwen4RequestCache& cache, int accepted_drafts);
    int request_cache_position(const MlxQwen4RequestCache& cache) const;
    MlxMtpEngineCallbacks request_mtp_callbacks(const std::shared_ptr<MlxQwen4RequestCache>& cache,
        bool batched_drafts = false);
    void draft_request_heads(const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
        const std::vector<std::optional<mlx::core::array>>& previous_tokens);
    MlxQwen4TextSessionState capture_request_cache(MlxQwen4RequestCache& cache,
        const std::vector<std::int64_t>& tokens);
    void restore_request_cache(MlxQwen4RequestCache& cache,
        const MlxQwen4TextSessionState& state);

    void reset_cache(int batch = 1);
    mlx::core::array score_forward(const mlx::core::array& token_ids, bool last_token_only);
    void clear_cache() noexcept;
    void reset_generation_state() noexcept;

    std::int32_t generate(
        const std::vector<std::int64_t>& prompt,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const std::function<bool(std::int64_t)>& callback = {},
        const std::function<void(std::size_t, double)>& prefill_callback = {},
        const MfqTokenConstraintPtr& token_constraint = {},
        std::optional<std::size_t> stable_prefix_tokens = std::nullopt,
        int prefill_chunk_size = 2048,
        const MlxPrefixCacheHooks& prefix_cache = {});

    MlxPreparedPrompt prepare_multimodal_prompt(
        const std::vector<std::int64_t>& prompt,
        const MlxGridMediaInput& media) const;
    std::int32_t generate_prepared(
        const MlxPreparedPrompt& prompt,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const std::function<bool(std::int64_t)>& callback = {},
        const std::function<void(std::size_t, double)>& prefill_callback = {},
        const MfqTokenConstraintPtr& token_constraint = {});
    std::string_view multimodal_input_contract() const noexcept;

    const Qwen4Config& config() const noexcept;
    std::size_t layer_count() const noexcept;
    int cache_position() const noexcept;
    bool supports_mtp() const noexcept;
    bool supports_batch_mtp() const noexcept;
    const MlxMtpGenerationStats& last_mtp_stats() const noexcept;
    std::size_t expert_cache_limit_bytes() const noexcept;
    std::optional<MlxSsdExpertCacheStats> ssd_expert_cache_stats() const;
    void prewarm_ssd_expert_arena();
    void clear_expert_cache();
    std::size_t set_expert_cache_limit(std::size_t bytes);
    std::optional<std::size_t> trim_preallocated_expert_cache(std::size_t bytes);
    std::size_t reclaimable_expert_bytes() const noexcept;
    std::size_t resident_full_expert_bytes() const noexcept;
    // Telemetry reads metadata only: no evaluation, copies, or device sync.
    std::size_t kv_cache_bytes() const noexcept;
    std::optional<QsaKvStoreStats> qsa_kv_offload_stats() const;
    std::size_t kv_cache_contexts() const noexcept;
    std::size_t dynamic_weight_bytes() const noexcept;
    std::size_t ssd_ple_payload_bytes() const noexcept;
    std::size_t ssd_expert_payload_bytes() const noexcept;
    bool supports_multimodal() const noexcept;
    bool supports_text_session_state() const noexcept { return true; }
    MlxQwen4TextSessionState capture_text_session_state(
        const std::vector<std::int64_t>& tokens, bool detached = true) const;
    void restore_text_session_state(const MlxQwen4TextSessionState& state);
    const std::string& mtp_cache_fingerprint() const noexcept;
    void prepare_mtp_ttt(const std::filesystem::path& directory);
    void begin_mtp_session(const std::string& session);
    void end_mtp_session();
    bool close_mtp_session(const std::string& session);
    bool fork_mtp_session(const std::string& source, const std::string& target);
    void clear_mtp_sessions();
    std::size_t trim_mtp_sessions(std::size_t bytes);
    std::size_t mtp_session_bytes() const;
    std::vector<std::pair<std::string, double>> mtp_session_metrics() const;

private:
    std::int32_t generate_prepared_impl(
        const MlxPreparedPrompt& prompt,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const std::function<bool(std::int64_t)>& callback,
        const std::function<void(std::size_t, double)>& prefill_callback,
        const MfqTokenConstraintPtr& token_constraint,
        std::optional<std::size_t> stable_prefix_tokens,
        int prefill_chunk_size,
        const MlxPrefixCacheHooks& prefix_cache);
    std::pair<mlx::core::array, mlx::core::array> forward_requests(
        const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
        const std::vector<std::int64_t>& tokens, int rows, bool speculative);
    std::pair<mlx::core::array, mlx::core::array> forward_packed_requests(
        const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
        const std::vector<std::int64_t>& tokens, const std::vector<int>& offsets, bool speculative);
    struct Impl;
    explicit MlxQwen4CausalLm(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    std::shared_ptr<QsaKvStore> kv_offload_store_;
};

} // namespace mfq::metal
