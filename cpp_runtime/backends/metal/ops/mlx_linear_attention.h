#pragma once

#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxGatedDeltaNetResult {
    mlx::core::array output;
    mlx::core::array state;
};

struct MlxLinearConvQkvResult {
    mlx::core::array query;
    mlx::core::array key;
    mlx::core::array value;
    mlx::core::array state;
};

MlxGatedDeltaNetResult gated_delta_net(
    const mlx::core::array& query,
    const mlx::core::array& key,
    const mlx::core::array& value,
    const mlx::core::array& gate,
    const mlx::core::array& beta,
    const std::optional<mlx::core::array>& state = std::nullopt,
    bool transposed_state = false,
    bool tiled_heads = false);

mlx::core::array ssm_conv_silu(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    int tokens,
    const std::optional<mlx::core::array>& bias = std::nullopt);

MlxLinearConvQkvResult linear_conv_qkv(
    const mlx::core::array& state,
    const mlx::core::array& qk,
    const mlx::core::array& value,
    const mlx::core::array& weight,
    int key_heads,
    int value_heads,
    int key_head_dimension,
    int value_head_dimension,
    const std::optional<mlx::core::array>& bias = std::nullopt,
    float eps = 1e-5f);

} // namespace mfq::metal
