#pragma once

#include "deepseek_v41_model.h"
#include "mlx_deepseek_v41_attention.h"
#include "mlx_deepseek_v41_dspark.h"
#include "mlx_deepseek_v41_engram.h"
#include "mlx_deepseek_v41_mhc.h"
#include "mlx_deepseek_v41_moe.h"
#include "mlx_deepseek_v41_vision.h"
#include "mlx_sampling.h"
#include "mlx_tensor.h"
#include "mlx_transformer.h"

#include "mfq/token_constraint.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV41TextSessionState {
    std::vector<std::int64_t> tokens;
    std::size_t bytes = 0;
};

struct MlxDeepseekV41LayerState {
    MlxDeepseekV41AttentionState attention;
};

struct MlxDeepseekV41LayerResult {
    mlx::core::array hidden;
    mlx::core::array next_pre;
    // Residual-stream mean at the exact attention input. DSpark taps this
    // value rather than the layer output.
    mlx::core::array attention_input;
};

// One native V4.1 layer. Engram injection and delayed Mega-mHC semantics are
// explicit here; the older V4 layer state machine is not reused.
class MlxDeepseekV41Layer {
public:
    static MlxDeepseekV41Layer load(
        const MfqContainer& model,
        const DeepseekV41Config& config,
        int index,
        int max_context,
        std::pair<mlx::core::array, mlx::core::array> rope_base,
        std::pair<mlx::core::array, mlx::core::array> rope_compressed,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache = nullptr);

    MlxDeepseekV41LayerResult forward(
        const mlx::core::array& hidden,
        const mlx::core::array& previous_pre,
        const DeepseekV41EngramHashBatch* hashes,
        const std::optional<mlx::core::array>& image_mask,
        MlxDeepseekV41LayerState& state,
        MlxDeepseekV41SharedAttentionState& shared_attention,
        int pos0) const;

    std::vector<mlx::core::array> begin_speculative(
        MlxDeepseekV41LayerState& state,
        int confirmed_tokens,
        int total_tokens) const;
    void commit_speculative(
        MlxDeepseekV41LayerState& state) const noexcept;
    std::vector<mlx::core::array> rollback_speculative(
        MlxDeepseekV41LayerState& state,
        int accepted_drafts) const;

    int index() const noexcept { return index_; }
    bool has_engram() const noexcept { return engram_ != nullptr; }

private:
    MlxDeepseekV41Layer(
        DeepseekV41Config config,
        int index,
        MlxDeepseekV41Attention attention,
        MlxDeepseekV41Mhc attention_mhc,
        MlxDeepseekV41Mhc ffn_mhc,
        MlxDeepseekV41Moe moe,
        std::unique_ptr<MlxDeepseekV41Engram> engram);

    DeepseekV41Config config_;
    int index_;
    MlxDeepseekV41Attention attention_;
    MlxDeepseekV41Mhc attention_mhc_;
    MlxDeepseekV41Mhc ffn_mhc_;
    MlxDeepseekV41Moe moe_;
    std::unique_ptr<MlxDeepseekV41Engram> engram_;
};

// Native text backbone for CED/CSA2 DeepSeek-V4.1. Vision replacement and
// the three-stage DSpark verifier attach to this class without changing the
// layer implementation.
class MlxDeepseekV41CausalLm {
public:
    static MlxDeepseekV41CausalLm load(
        const MfqContainer& model,
        int max_context = 4096,
        std::optional<std::size_t> expert_cache_bytes = std::nullopt);

    MlxDeepseekV41CausalLm(
        DeepseekV41Config config,
        MlxEmbedding embedding,
        std::vector<MlxDeepseekV41Layer> layers,
        mlx::core::array output_norm,
        MlxLinear output,
        MlxDeepseekV41EngramHashState engram_hash,
        int max_context,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache = nullptr,
        std::optional<MlxDeepseekV41Vision> vision = std::nullopt,
        std::optional<MlxDeepseekV41DSpark> dspark = std::nullopt);

    // Accept [tokens] or [batch,tokens]. use_cache=false resets the native
    // cache first and, like the other MFQ runtimes, leaves it ready to decode.
    mlx::core::array forward(
        const mlx::core::array& token_ids,
        bool use_cache = false);

    mlx::core::array prefill(
        const mlx::core::array& token_ids,
        int chunk_size = 512,
        bool full_logits = false);

    mlx::core::array decode(const mlx::core::array& token_ids);

    // V4.1 image spans must be present in the first, unchunked prefill. The
    // returned logits are [batch,vocab] for the final prompt position.
    mlx::core::array prefill_multimodal(
        const std::vector<std::int64_t>& token_ids,
        const std::vector<MlxDeepseekV41ImageInput>& images);

    MlxDeepseekV41DSparkDraft draft_mtp(
        const mlx::core::array& anchor_ids,
        const MlxMtpTokenSelector& select_token,
        int width = 0);

    std::int32_t generate(
        const std::vector<std::int64_t>& prompt,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const std::function<bool(std::int64_t)>& callback = {},
        const std::function<void(std::size_t, double)>& prefill_callback = {},
        const MfqTokenConstraintPtr& token_constraint = {},
        std::optional<std::size_t> stable_prefix_tokens = std::nullopt,
        int prefill_chunk_size = 2048);

    std::int32_t generate_multimodal(
        const std::vector<std::int64_t>& prompt,
        const std::vector<MlxDeepseekV41ImageInput>& images,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const std::function<bool(std::int64_t)>& callback = {},
        const std::function<void(std::size_t, double)>& prefill_callback = {},
        const MfqTokenConstraintPtr& token_constraint = {});

    void reset_cache(int batch = 1);
    void clear_cache() noexcept;

    const DeepseekV41Config& config() const noexcept { return config_; }
    int max_context() const noexcept { return max_context_; }
    int cache_position() const noexcept { return cache_position_; }
    int cache_batch() const noexcept { return cache_batch_; }
    std::size_t layer_count() const noexcept { return layers_.size(); }
    bool supports_multimodal() const noexcept { return vision_.has_value(); }
    bool supports_mtp() const noexcept { return dspark_.has_value(); }
    std::size_t expert_cache_limit_bytes() const noexcept;
    std::optional<MlxSsdExpertCacheStats> ssd_expert_cache_stats() const;
    void prewarm_ssd_expert_arena();
    void clear_expert_cache();
    const MlxMtpGenerationStats& last_mtp_stats() const noexcept {
        return last_mtp_stats_;
    }
    bool supports_text_session_state() const noexcept { return false; }
    MlxDeepseekV41TextSessionState capture_text_session_state(
        const std::vector<std::int64_t>& tokens) const;
    void restore_text_session_state(
        const MlxDeepseekV41TextSessionState& state);

private:
    mlx::core::array normalized_ids(const mlx::core::array& token_ids) const;
    mlx::core::array forward_impl(
        const mlx::core::array& token_ids,
        const std::optional<mlx::core::array>& image_mask,
        bool reuse_cache,
        const std::optional<mlx::core::array>& input_embeddings = std::nullopt,
        mlx::core::array* dspark_hidden = nullptr,
        bool update_dspark = true);
    void begin_speculative_target(
        const mlx::core::array& token_ids,
        int confirmed_tokens);
    void commit_speculative_target() noexcept;
    void rollback_speculative_target(
        int accepted_drafts,
        int draft_tokens);
    void abort_speculative_target() noexcept;
    std::int32_t generate_from_prefill(
        mlx::core::array logits,
        const std::vector<std::int64_t>& history,
        const MlxSamplingParams& sampling,
        std::int32_t limit,
        const std::function<bool(std::int64_t)>& callback,
        const MfqTokenConstraintPtr& token_constraint);

    DeepseekV41Config config_;
    MlxEmbedding embedding_;
    std::vector<MlxDeepseekV41Layer> layers_;
    MlxRmsNorm output_norm_;
    MlxLinear output_;
    MlxDeepseekV41EngramHashState engram_hash_;
    std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache_;
    std::optional<MlxDeepseekV41Vision> vision_;
    std::optional<MlxDeepseekV41DSpark> dspark_;
    std::optional<MlxDeepseekV41DSparkState> dspark_state_;
    int max_context_;
    int cache_position_ = 0;
    int cache_batch_ = 0;
    std::vector<MlxDeepseekV41LayerState> states_;
    std::optional<DeepseekV41EngramHashSnapshot>
        speculative_engram_snapshot_;
    std::optional<mlx::core::array> speculative_token_ids_;
    int speculative_cache_start_ = -1;
    bool mtp_context_requested_ = false;
    MlxMtpGenerationStats last_mtp_stats_;
};

} // namespace mfq::metal
