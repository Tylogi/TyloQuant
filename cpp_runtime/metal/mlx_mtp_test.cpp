#include "mlx_mtp.h"

#include <array>
#include <iostream>
#include <stdexcept>

int main() {
    using mfq::metal::verify_greedy_mtp;
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
