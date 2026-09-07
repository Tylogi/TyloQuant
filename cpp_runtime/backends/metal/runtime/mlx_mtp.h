#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxMtpVerification {
    std::size_t accepted_drafts = 0;
    std::int32_t next_token = -1;
    bool bonus = false;
};

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
