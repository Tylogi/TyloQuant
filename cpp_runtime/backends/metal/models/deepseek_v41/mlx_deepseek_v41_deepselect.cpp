#include "mlx_deepseek_v41_deepselect.h"

#include "mlx_deepseek_sparse.h"

#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace mfq::metal {

bool deepseek_v41_deepselect_preferred(int width, int rows) noexcept {
    // Crossovers include downstream gather, validation, and sort. They were
    // measured on M3 Ultra; decode and short prefill stay on argpartition.
    const bool favorable_shape =
        (width >= 16384 && rows >= 64) ||
        (width >= 8192 && rows >= 128);
    if (!favorable_shape) {
        return false;
    }
    if (const auto* requested = std::getenv(
            "MFQ_METAL_DSV41_DEEPSELECT")) {
        return std::strcmp(requested, "0") != 0 &&
            std::strcmp(requested, "false") != 0 &&
            std::strcmp(requested, "off") != 0;
    }
#if defined(__APPLE__)
    static const bool is_m3_ultra = [] {
        std::size_t size = 0;
        if (::sysctlbyname(
                "machdep.cpu.brand_string",
                nullptr,
                &size,
                nullptr,
                0) != 0 || size <= 1) {
            return false;
        }
        std::string name(size, '\0');
        if (::sysctlbyname(
                "machdep.cpu.brand_string",
                name.data(),
                &size,
                nullptr,
                0) != 0) {
            return false;
        }
        if (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }
        return name == "Apple M3 Ultra";
    }();
    return is_m3_ultra;
#else
    return false;
#endif
}

mlx::core::array deepseek_v41_deepselect_topk512(
    const mlx::core::array& scores,
    const std::optional<mlx::core::array>& valid_keys) {
    return mlx_deepselect_topk512(scores, valid_keys);
}

} // namespace mfq::metal
