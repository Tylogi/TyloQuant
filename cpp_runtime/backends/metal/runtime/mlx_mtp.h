#pragma once

#include "mlx_sampling.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// Storage/statistics capacity of the common engine, not a model capability.
// Each predictor adapter supplies its own maximum depth in the request.
inline constexpr int kMlxMtpEngineMaximumDraftDepth = 5;

struct MlxMtpVerification {
    std::size_t accepted_drafts = 0;
    std::int32_t next_token = -1;
    bool bonus = false;
};

// One backend-wide accounting contract for every speculative predictor.
// Model adapters expose this value but never define their own statistics
// schema or scheduling policy.
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

using MlxGenerationTokenCallback =
    std::function<bool(std::int64_t)>;

// Information delivered to a predictor adapter before it proposes a chain.
// On the first call verified_target is null. On later calls it owns the target
// rows from the preceding verification; next_token_ids pairs those rows with
// the shifted tokens expected by recurrent predictors such as Qwen MTP.
struct MlxMtpDraftContext {
    bool initial = false;
    std::int32_t pending_token = -1;
    int requested_depth = 0;
    int target_cache_start = 0;
    int accepted_drafts = 0;
    const mlx::core::array* verified_hidden = nullptr;
    std::span<const std::int32_t> next_token_ids;
};

// Predictor math remains architecture-specific, but token selection is owned
// by the common engine. Calling this function once per draft position applies
// the same penalties, RNG stream and speculative distribution policy for all
// predictor implementations.
using MlxMtpTokenSelector = std::function<mlx::core::array(
    const mlx::core::array& logits)>;

// Fold teacher-forced (target_hidden[t], token[t + 1]) pairs into any
// predictor cache.  Architecture adapters supply only the predictor forward;
// slicing, alignment, chunking and graph materialization stay common.
using MlxMtpHistoryFold = std::function<mlx::core::array(
    const mlx::core::array& hidden_rows,
    const mlx::core::array& shifted_token_ids,
    int pair_offset)>;

std::size_t mlx_prime_mtp_history(
    const mlx::core::array& target_hidden,
    const mlx::core::array& prompt_ids,
    const MlxMtpHistoryFold& fold,
    int chunk_size = 512);

struct MlxMtpTargetBatch {
    // [drafts + 1, vocab], one target row for each draft and the bonus row.
    mlx::core::array logits;
    // [1, drafts + 1, ...] adapter payload used to seed the next proposal.
    mlx::core::array hidden;
};

struct MlxMtpEngineCallbacks {
    // Current number of tokens committed in the target-model cache.
    std::function<int()> target_cache_position;

    // Fold verified history into the predictor and invoke select_token once
    // for every requested draft. Depth zero still calls prepare_draft so a
    // recurrent predictor can commit its shifted history.
    std::function<void(
        const MlxMtpDraftContext& context,
        const MlxMtpTokenSelector& select_token)> prepare_draft;

    // Verify [pending_token, draft_tokens...] with the target model. The
    // target adapter opens any cache transaction needed by resolve_target.
    std::function<MlxMtpTargetBatch(
        std::int32_t pending_token,
        const mlx::core::array& draft_tokens,
        int draft_count)> verify_target;

    // Keep pending_token plus accepted_drafts and discard the rejected draft
    // suffix. This closes the target cache transaction for every verify call.
    std::function<void(int accepted_drafts, int draft_count)> resolve_target;

    // Ordinary one-token target decode used after adaptive MTP exits.
    std::function<mlx::core::array(std::int32_t pending_token)> plain_decode;
};

struct MlxMtpEngineRequest {
    int vocab = 0;
    int generation_limit = 0;
    int maximum_context = 0;
    int predictor_maximum_depth = 0;
    mlx::core::array initial_logits;
    MlxSamplingParams sampling;
    std::optional<mlx::core::array> token_counts;
    std::span<const std::int64_t> eos_token_ids;
    MlxGenerationTokenCallback callback;
    std::uint64_t sampler_draws_consumed = 0;
};

// Avoid allocating a vocabulary-sized count vector on the common no-penalty
// path. The helper is shared by every generation adapter.
std::optional<mlx::core::array> mlx_generation_token_counts(
    const MlxSamplingParams& sampling,
    const mlx::core::array& prompt_ids,
    int vocab);

// Architecture-independent speculative generation state machine. Adapters
// provide only predictor math and target-cache mechanics; this function owns
// sampling, verification, dynamic depth, emission, stopping and statistics.
std::int32_t run_mlx_mtp_generation(
    MlxMtpEngineRequest request,
    const MlxMtpEngineCallbacks& callbacks,
    MlxMtpGenerationStats& stats);

// Runtime draft-depth controller adapted from oMLX's production MTP loop.
// It learns conditional acceptance by draft position and the measured wall
// time of every verify width, then maximizes expected emitted tokens / ms.
// Depth zero is a measured plain-decode escape hatch.
class MlxMtpDepthController {
public:
    explicit MlxMtpDepthController(int maximum_depth = 3);

    int depth() const noexcept {
        return current_depth_;
    }
    int maximum_depth() const noexcept {
        return maximum_depth_;
    }
    void observe(
        int used_depth,
        int accepted_drafts,
        double cycle_ms,
        bool time_sample = true);
    bool should_exit() const noexcept;

    double conditional_acceptance(int position) const;
    std::optional<double> measured_cycle_ms(int depth) const;

private:
    void update_time(int depth, double cycle_ms);
    double time_estimate(int depth) const;
    double marginal_estimate() const;
    double score(int depth) const;
    bool speculation_losing() const;
    int best_depth() const;
    std::optional<int> best_rival() const;
    std::optional<int> most_stale() const;

    int maximum_depth_ = 1;
    int current_depth_ = 1;
    int cycles_ = 0;
    int probe_left_ = 0;
    int exit_streak_ = 0;
    double milliseconds_since_probe_ = 0.0;
    double milliseconds_since_explore_ = 0.0;
    std::vector<double> acceptance_;
    std::vector<int> warmup_accepts_;
    std::vector<int> warmup_trials_;
    std::vector<std::optional<double>> cycle_ms_;
    std::vector<std::optional<double>> cycle_age_ms_;
    std::vector<int> warmup_;
};

// Architecture-independent greedy speculative verification.  targets must
// contain one prediction for every draft plus the all-accepted bonus row.
MlxMtpVerification verify_greedy_mtp(
    std::span<const std::int32_t> drafts,
    std::span<const std::int32_t> targets);

// Exact one-draft stochastic speculative verification. proposal/target/bonus
// are already filtered and normalized by the ordinary sampling policy.
MlxMtpVerification verify_stochastic_mtp(
    std::int32_t draft,
    const std::vector<float>& proposal,
    const std::vector<float>& target,
    const std::vector<float>& bonus,
    double acceptance_uniform,
    double sample_uniform);

// GPU-resident one-draft stochastic verification over compact top-k
// distributions. Returns int32 [accepted_drafts, next_token, draft_token].
// Keeping the result compact lets the caller resolve an entire speculative
// cycle with one host synchronization.
mlx::core::array verify_stochastic_mtp_top_k_device(
    const mlx::core::array& proposal_indices,
    const mlx::core::array& proposal_probabilities,
    const mlx::core::array& target_indices,
    const mlx::core::array& target_probabilities,
    const mlx::core::array& bonus_indices,
    const mlx::core::array& bonus_probabilities,
    const mlx::core::array& draft_token,
    const mlx::core::array& random,
    int top_k);

// Multi-draft form. Proposal tensors contain [drafts, top_k], target tensors
// contain [drafts + 1, top_k], random contains one acceptance roll per draft
// followed by one final sampling roll. Returns
// [accepted_drafts, next_token, draft_0, ... draft_n].
mlx::core::array verify_stochastic_mtp_top_k_chain_device(
    const mlx::core::array& proposal_indices,
    const mlx::core::array& proposal_probabilities,
    const mlx::core::array& target_indices,
    const mlx::core::array& target_probabilities,
    const mlx::core::array& draft_tokens,
    const mlx::core::array& random,
    int drafts,
    int top_k);

} // namespace mfq::metal
