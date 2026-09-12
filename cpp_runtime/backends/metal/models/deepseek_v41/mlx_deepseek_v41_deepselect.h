#pragma once

#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

// V4.1 owns the device/shape policy for selecting the generic DeepSelect op.
bool deepseek_v41_deepselect_preferred(int width, int rows) noexcept;

mlx::core::array deepseek_v41_deepselect_topk512(
    const mlx::core::array& scores,
    const std::optional<mlx::core::array>& valid_keys = std::nullopt);

} // namespace mfq::metal
