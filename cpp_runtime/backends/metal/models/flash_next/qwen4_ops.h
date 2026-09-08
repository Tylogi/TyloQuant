#pragma once

// Qwen4-Exp model semantics implemented with native MLX/Metal primitives.

#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxQwen4GatedResidualPre {
    mlx::core::array branch;
    mlx::core::array residual;
    std::optional<mlx::core::array> injection;
};

mlx::core::array qwen4_grouped_rms_norm(
    const mlx::core::array& value,
    const mlx::core::array& weight,
    int group_size,
    float eps = 1e-6f);

MlxQwen4GatedResidualPre qwen4_gated_residual_pre(
    const mlx::core::array& hyper_input,
    const mlx::core::array& norm_weight,
    const mlx::core::array& down_weight,
    const mlx::core::array& up_weight,
    const std::optional<mlx::core::array>& inject_weight,
    int hidden_size,
    int hc_count,
    float eps = 1e-6f);

mlx::core::array qwen4_gated_residual_post(
    const mlx::core::array& branch,
    const mlx::core::array& residual,
    const mlx::core::array& injection,
    int hc_count);

mlx::core::array qwen4_qsa_block_scores(
    const mlx::core::array& query,
    const mlx::core::array& pooled_keys);

mlx::core::array qwen4_dense_gqa_attention(
    const mlx::core::array& query,
    const mlx::core::array& key,
    const mlx::core::array& value,
    int query_offset);

} // namespace mfq::metal
