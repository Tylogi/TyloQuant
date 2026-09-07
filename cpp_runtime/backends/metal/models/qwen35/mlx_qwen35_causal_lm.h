#pragma once

#include "mfq/token_constraint.h"
#include "mlx_qwen35_full_attention.h"
#include "mlx_qwen35_linear_attention.h"
#include "mlx_multimodal.h"
#include "mlx_sampling.h"
#include "qwen35_model.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

namespace detail {

// Generation only needs a dense vocabulary-sized count vector when a
// configured penalty consumes it. Returning nullopt for the common
// no-penalty path also prevents an otherwise-unused per-token graph chain.
std::optional<mlx::core::array>
qwen35_generation_token_counts(
    const MlxSamplingParams& sampling,
    const mlx::core::array& prompt_ids,
    int vocab);

} // namespace detail

using MlxQwen35Layer = std::variant<
    MlxQwen35FullAttentionBlock,
    MlxQwen35LinearAttentionBlock>;

using MlxQwen35LayerCacheSnapshot = std::variant<
    MlxKvCacheSnapshot,
    MlxQwen35LinearAttentionCacheSnapshot>;

struct MlxQwen35TextSessionState {
    std::vector<std::int64_t> tokens;
    std::vector<MlxQwen35LayerCacheSnapshot> layers;
    int cache_position = 0;
    int cache_batch = 0;
    std::size_t bytes = 0;
};

struct MlxMtpGenerationStats {
    bool available = false;
    bool used = false;
    std::uint64_t cycles = 0;
    std::uint64_t drafted_tokens = 0;
    std::uint64_t accepted_tokens = 0;
    std::array<std::uint64_t, 6> depth_cycles{};
    std::array<std::uint64_t, 5> position_drafted{};
    std::array<std::uint64_t, 5> position_accepted{};
    std::array<double, 6> measured_depth_ms{};
    int selected_depth = 0;
};

using MlxTokenCallback = std::function<bool(std::int64_t)>;

class MlxQwen35MtpModule {
public:
    static std::optional<MlxQwen35MtpModule> load_if_present(
        const MfqContainer& model,
        const Qwen35Config& config,
        const Qwen35TensorNames& names);

    MlxQwen35MtpModule(
        Qwen35Config config,
        MlxRmsNorm hidden_norm,
        MlxRmsNorm embedding_norm,
        Qwen35Linear fusion,
        std::vector<MlxQwen35FullAttentionBlock> layers,
        MlxRmsNorm output_norm);

    mlx::core::array forward(
        const mlx::core::array& hidden_states,
        const mlx::core::array& next_token_ids,
        const Qwen35Embedding& embedding,
        bool use_cache = true);
    mlx::core::array forward(
        const mlx::core::array& hidden_states,
        const mlx::core::array& next_token_ids,
        const mlx::core::array& positions,
        const Qwen35Embedding& embedding,
        bool use_cache = true);
    void reset_cache(int batch = 1, int initial_capacity = 16);
    void materialize_cache();
    void trim_cache_to(int position);
    void clear_cache() noexcept;
    int cache_position() const noexcept;
    std::size_t layer_count() const noexcept {
        return layers_.size();
    }

private:
    void validate_components() const;
    mlx::core::array forward_impl(
        const mlx::core::array& hidden_states,
        const mlx::core::array& next_token_ids,
        const mlx::core::array* positions,
        const Qwen35Embedding& embedding,
        bool use_cache);

    Qwen35Config config_;
    MlxRmsNorm hidden_norm_;
    MlxRmsNorm embedding_norm_;
    Qwen35Linear fusion_;
    std::vector<MlxQwen35FullAttentionBlock> layers_;
    MlxRmsNorm output_norm_;
};

class MlxQwen35CausalLm {
public:
    static MlxQwen35CausalLm load(const MfqContainer& model);

    static MlxQwen35CausalLm load(
        const MfqContainer& model,
        const Qwen35Config& config,
        const Qwen35TensorNames& names);

    MlxQwen35CausalLm(
        Qwen35Config config,
        Qwen35Embedding embedding,
        std::vector<MlxQwen35Layer> layers,
        MlxRmsNorm output_norm,
        std::optional<Qwen35Linear> output,
        mlx::core::Dtype activation_dtype = mlx::core::float16,
        std::optional<MlxQwen35MtpModule> mtp = std::nullopt,
        std::optional<MlxGridVisionPromptComponent> vision = std::nullopt);

    // token_ids must be a non-empty [batch,tokens] array. Cached calls append
    // contiguously; uncached calls do not mutate existing cache state.
    mlx::core::array forward(
        const mlx::core::array& token_ids,
        bool use_cache = true);

    // Explicit text or three-axis multimodal RoPE coordinates. positions
    // accepts [tokens], [1,tokens], or [3,tokens]. Cache storage still
    // advances contiguously by token count.
    mlx::core::array forward(
        const mlx::core::array& token_ids,
        const mlx::core::array& positions,
        bool use_cache = true);

    // Run precomputed or multimodally injected [batch,tokens,hidden]
    // embeddings through the same backbone and cache used by token forwards.
    // Without explicit positions, text positions advance contiguously from
    // the current cache position. Explicit positions accept [tokens],
    // [1,tokens], or Qwen MRoPE [3,tokens] coordinates.
    mlx::core::array forward_embeddings(
        const mlx::core::array& embeddings,
        bool use_cache = true);

    mlx::core::array forward_embeddings(
        const mlx::core::array& embeddings,
        const mlx::core::array& positions,
        bool use_cache = true);

    std::pair<mlx::core::array, mlx::core::array>
    forward_embeddings_with_hidden(
        const mlx::core::array& embeddings,
        bool use_cache = true);

    std::pair<mlx::core::array, mlx::core::array>
    forward_embeddings_with_hidden(
        const mlx::core::array& embeddings,
        const mlx::core::array& positions,
        bool use_cache = true);

    mlx::core::array operator()(
        const mlx::core::array& token_ids,
        bool use_cache = true) {
        return forward(token_ids, use_cache);
    }

    mlx::core::array operator()(
        const mlx::core::array& token_ids,
        const mlx::core::array& positions,
        bool use_cache = true) {
        return forward(token_ids, positions, use_cache);
    }

    void reset_cache(int batch = 1);
    void clear_cache() noexcept;

    // Returns the number of sampled tokens. A token is counted before its
    // callback is invoked, so callback=false still returns a count including
    // that token and stops immediately without an extra decode.
    std::int32_t generate(
        const std::vector<std::int64_t>& prompt,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const MlxTokenCallback& callback = {},
        const std::function<void(std::size_t, double)>&
            prefill_callback = {},
        const MfqTokenConstraintPtr& token_constraint = {},
        std::optional<std::size_t> stable_prefix_tokens =
            std::nullopt);

    // All optional input components converge here. Sampling, penalties,
    // cache management and MTP are deliberately shared with text generation.
    std::int32_t generate_prepared(
        const MlxPreparedPrompt& prompt,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const MlxTokenCallback& callback = {},
        const std::function<void(std::size_t, double)>&
            prefill_callback = {},
        const MfqTokenConstraintPtr& token_constraint = {},
        std::optional<std::size_t> stable_prefix_tokens =
            std::nullopt);

    mlx::core::array embed_tokens(
        const mlx::core::array& token_ids) const;

    MlxPreparedPrompt prepare_multimodal_prompt(
        const std::vector<std::int64_t>& prompt,
        const MlxGridMediaInput& media) const;

    MlxQwen35TextSessionState capture_text_session_state(
        const std::vector<std::int64_t>& tokens) const;
    void restore_text_session_state(
        const MlxQwen35TextSessionState& state);
    bool supports_text_session_state() const noexcept {
        return true;
    }

    const Qwen35Config& config() const noexcept {
        return config_;
    }
    std::size_t layer_count() const noexcept {
        return layers_.size();
    }
    std::string_view layer_type(std::size_t index) const;
    int cache_position() const noexcept {
        return cache_position_;
    }
    int cache_batch() const noexcept {
        return cache_batch_;
    }
    bool supports_mtp() const noexcept {
        return mtp_.has_value();
    }
    bool supports_multimodal() const noexcept {
        return vision_.has_value();
    }
    const MlxMtpGenerationStats& last_mtp_stats() const noexcept {
        return last_mtp_stats_;
    }
    std::string_view multimodal_input_contract() const noexcept {
        return vision_ ? vision_->input_contract() : std::string_view{};
    }

private:
    void validate_components() const;
    void prepare_cache_for_prefill(
        int batch,
        int prompt_tokens);
    std::pair<mlx::core::array, mlx::core::array>
    forward_embeddings_impl(
        const mlx::core::array& embeddings,
        const mlx::core::array* positions,
        bool use_cache,
        int speculative_confirmed = 0);
    std::pair<mlx::core::array, mlx::core::array>
    forward_with_hidden(
        const mlx::core::array& token_ids,
        bool use_cache,
        int speculative_confirmed = 0);
    void commit_speculative();
    void rollback_speculative(
        int accepted_tokens,
        int draft_tokens);
    mlx::core::array project_logits(
        const mlx::core::array& hidden) const;
    mlx::core::array project_normalized(
        const mlx::core::array& hidden) const;

    Qwen35Config config_;
    Qwen35Embedding embedding_;
    std::vector<MlxQwen35Layer> layers_;
    MlxRmsNorm output_norm_;
    std::optional<Qwen35Linear> output_;
    mlx::core::Dtype activation_dtype_;
    std::optional<MlxQwen35MtpModule> mtp_;
    std::optional<MlxGridVisionPromptComponent> vision_;
    MlxMtpGenerationStats last_mtp_stats_;
    int cache_position_ = 0;
    int cache_batch_ = 0;
    std::vector<std::int64_t> stable_cache_tokens_;
};

} // namespace mfq::metal
