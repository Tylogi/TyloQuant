#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mfq::metal {

struct MlxMtpVerification {
    std::size_t accepted_drafts = 0;
    std::int32_t next_token = -1;
    bool bonus = false;
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

} // namespace mfq::metal
