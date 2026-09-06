#include "mlx_mtp.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    using mfq::metal::verify_greedy_mtp;
    using mfq::metal::verify_stochastic_mtp;
    try {
        const std::array<std::int32_t, 4> drafts{11, 12, 13, 14};
        {
            const std::array<std::int32_t, 5> targets{11, 12, 99, 14, 15};
            const auto result = verify_greedy_mtp(drafts, targets);
            if (result.accepted_drafts != 2 || result.next_token != 99 ||
                result.bonus) {
                throw std::runtime_error("partial MTP acceptance mismatch");
            }
        }
        {
            const std::vector<float> proposal{0.8f, 0.2f};
            const std::vector<float> target{0.4f, 0.6f};
            const std::vector<float> bonus{0.1f, 0.9f};
            const auto accepted = verify_stochastic_mtp(
                0, proposal, target, bonus, 0.25, 0.2);
            if (accepted.accepted_drafts != 1 ||
                accepted.next_token != 1 || !accepted.bonus) {
                throw std::runtime_error(
                    "stochastic MTP acceptance mismatch");
            }
            const auto rejected = verify_stochastic_mtp(
                0, proposal, target, bonus, 0.75, 0.0);
            if (rejected.accepted_drafts != 0 ||
                rejected.next_token != 1 || rejected.bonus) {
                throw std::runtime_error(
                    "stochastic MTP correction mismatch");
            }
        }
        {
            const std::array<std::int32_t, 5> targets{11, 12, 13, 14, 15};
            const auto result = verify_greedy_mtp(drafts, targets);
            if (result.accepted_drafts != 4 || result.next_token != 15 ||
                !result.bonus) {
                throw std::runtime_error("MTP bonus acceptance mismatch");
            }
        }
        std::cout << "MFQ generic MTP verification tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ generic MTP verification tests failed: "
                  << error.what() << '\n';
        return 1;
    }
}
