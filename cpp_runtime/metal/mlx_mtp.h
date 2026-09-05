#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

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

} // namespace mfq::metal
