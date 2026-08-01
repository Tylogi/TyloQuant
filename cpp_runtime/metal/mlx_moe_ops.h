#pragma once

#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxMoeTopKResult {
    mlx::core::array ids;
    mlx::core::array weights;
};

MlxMoeTopKResult moe_topk(
    const mlx::core::array& logits,
    int top_k,
    bool use_sigmoid = false,
    bool use_sqrt_softplus = false,
    bool normalize = false,
    bool delayed_softmax = false,
    const std::optional<mlx::core::array>& bias = std::nullopt,
    const std::optional<mlx::core::array>& available = std::nullopt,
    float norm_floor = 1e-20f,
    float scale = 1.0f);

mlx::core::array moe_sqrtsoftplus_weights(
    const mlx::core::array& logits,
    const mlx::core::array& ids,
    float norm_floor = 1e-20f,
    float scale = 1.0f);

mlx::core::array moe_selected_sqrtsoftplus_weights(
    const mlx::core::array& logits,
    const mlx::core::array& ids,
    bool normalize,
    float norm_floor = 1e-20f,
    float scale = 1.0f);

mlx::core::array moe_repair_hash_ids(
    const mlx::core::array& static_ids,
    const mlx::core::array& candidate_ids,
    const mlx::core::array& available);

mlx::core::array moe_weighted_reduce(
    const mlx::core::array& pair_output,
    const mlx::core::array& weights);

mlx::core::array moe_swiglu_split(
    const mlx::core::array& gate_up);

mlx::core::array moe_limited_swiglu_split(
    const mlx::core::array& gate_up,
    float limit);

mlx::core::array moe_geglu_split(
    const mlx::core::array& gate_up);

mlx::core::array moe_add_shared_gate(
    const mlx::core::array& routed,
    const mlx::core::array& shared,
    const mlx::core::array& gate_logits);

mlx::core::array moe_weighted_reduce_shared_gate(
    const mlx::core::array& pair_output,
    const mlx::core::array& weights,
    const mlx::core::array& shared,
    const mlx::core::array& gate_logits);

mlx::core::array moe_apply_expert_scale(
    const mlx::core::array& weights,
    const mlx::core::array& ids,
    const mlx::core::array& scales);

} // namespace mfq::metal
