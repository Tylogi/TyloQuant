#include "mlx_mtp.h"

#include "mlx_sampling.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mfq::metal {

MlxMtpVerification verify_greedy_mtp(
    std::span<const std::int32_t> drafts,
    std::span<const std::int32_t> targets) {
    if (drafts.empty() || targets.size() != drafts.size() + 1) {
        throw std::invalid_argument(
            "MTP verification requires N drafts and N+1 targets");
    }
    for (std::size_t index = 0; index < drafts.size(); ++index) {
        if (targets[index] != drafts[index]) {
            return {index, targets[index], false};
        }
    }
    return {drafts.size(), targets.back(), true};
}

MlxMtpVerification verify_stochastic_mtp(
    std::int32_t draft,
    const std::vector<float>& proposal,
    const std::vector<float>& target,
    const std::vector<float>& bonus,
    double acceptance_uniform,
    double sample_uniform) {
    if (proposal.empty() || proposal.size() != target.size() ||
        target.size() != bonus.size() || draft < 0 ||
        static_cast<std::size_t>(draft) >= proposal.size() ||
        !std::isfinite(acceptance_uniform)) {
        throw std::invalid_argument(
            "stochastic MTP verification inputs are incompatible");
    }
    const double q = proposal[static_cast<std::size_t>(draft)];
    const double p = target[static_cast<std::size_t>(draft)];
    if (!(q > 0.0) || p < 0.0 || !std::isfinite(q) || !std::isfinite(p)) {
        throw std::invalid_argument(
            "stochastic MTP draft has an invalid probability");
    }
    const double acceptance = std::min(1.0, p / q);
    if (std::clamp(acceptance_uniform, 0.0, std::nextafter(1.0, 0.0)) <
        acceptance) {
        return {
            1,
            sample_host_distribution(bonus, sample_uniform),
            true,
        };
    }

    std::vector<float> correction(proposal.size());
    double total = 0.0;
    for (std::size_t token = 0; token < correction.size(); ++token) {
        correction[token] = std::max(0.0f, target[token] - proposal[token]);
        total += correction[token];
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        correction = target;
    } else {
        for (auto& probability : correction) {
            probability = static_cast<float>(probability / total);
        }
    }
    return {
        0,
        sample_host_distribution(correction, sample_uniform),
        false,
    };
}

} // namespace mfq::metal
