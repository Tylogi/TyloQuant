#pragma once

#include <cstdlib>
#include <string_view>

namespace mfq::metal {

// Numerical-audit mode. Packed weights stay resident, but every projection is
// independently unpacked to FP16 and evaluated with MLX dense matmul. Native
// MXFP4/MXFP8 and dense BF16/F16/F32 tensors retain their original execution.
inline bool mlx_reference_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv(
            "MFQ_METAL_UNPACK_REFERENCE");
        return value != nullptr
            && std::string_view(value) != "0";
    }();
    return enabled;
}

} // namespace mfq::metal
