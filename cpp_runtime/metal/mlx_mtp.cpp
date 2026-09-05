#include "mlx_mtp.h"

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

} // namespace mfq::metal
