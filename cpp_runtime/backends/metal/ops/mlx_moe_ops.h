#pragma once

#include <optional>
#include <cstdint>
#include <span>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxMoeTopKResult {
    mlx::core::array ids;
    mlx::core::array weights;
    std::optional<mlx::core::array> groups;
};

std::optional<mlx::core::array> moe_route_groups_host(
    std::span<const std::int32_t> ids, int tokens, int routes);

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
    float scale = 1.0f,
    bool route_groups = true);

// One-row dense MoE router GEMV. A SIMD group owns each expert row so the
// small [experts, hidden] projection uses the whole GPU instead of the stock
// narrow-output matmul schedule. FP16 and BF16 weights are supported.
bool moe_dense_router_logits_supported(
    const mlx::core::array& input,
    const mlx::core::array& weight) noexcept;

mlx::core::array moe_dense_router_logits(
    const mlx::core::array& input,
    const mlx::core::array& weight);

// Small-M normalized top-6 sqrt-softplus router hot path. Compute up to 16
// FP16/BF16 router rows and selection weights in one Metal dispatch.
bool moe_dense_router_topk_supported(
    const mlx::core::array& input,
    const mlx::core::array& weight) noexcept;

MlxMoeTopKResult moe_dense_router_topk(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    const std::optional<mlx::core::array>& bias = std::nullopt,
    const std::optional<mlx::core::array>& available = std::nullopt,
    float norm_floor = 1e-20f,
    float scale = 1.0f);

MlxMoeTopKResult moe_dense_router_topk_packed(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    const mlx::core::array& expert_map,
    const std::optional<mlx::core::array>& bias = std::nullopt,
    const std::optional<mlx::core::array>& available = std::nullopt,
    float norm_floor = 1e-20f,
    float scale = 1.0f);

// Hash-routed models already know the selected expert IDs from the token
// table.  Compute only those dense router rows and their normalized
// sqrt-softplus weights in one dispatch instead of materializing every
// expert logit and running a redundant Top-K pass.
bool moe_dense_hash_router_supported(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    const mlx::core::array& token_ids,
    const mlx::core::array& token_experts) noexcept;

MlxMoeTopKResult moe_dense_hash_router(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    const mlx::core::array& token_ids,
    const mlx::core::array& token_experts,
    float norm_floor = 1e-20f,
    float scale = 1.0f);

MlxMoeTopKResult moe_dense_hash_router_packed(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    const mlx::core::array& token_ids,
    const mlx::core::array& token_experts,
    const mlx::core::array& expert_map,
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

// Reduce expert-contiguous routed output directly through the inverse route
// permutation. This avoids materializing a second token-major routed tensor
// before the weighted sum.
mlx::core::array moe_weighted_reduce_sorted(
    const mlx::core::array& sorted_pair_output,
    const mlx::core::array& inverse_route_order,
    const mlx::core::array& weights);

// Invert a valid one-dimensional int32 permutation in one linear Metal pass.
// This is substantially cheaper than argsort(order), especially for the
// token*top_k route permutations used by grouped MoE prefill.
mlx::core::array moe_inverse_permutation(
    const mlx::core::array& order);

mlx::core::array moe_restore_routed_rows(
    const mlx::core::array& sorted_output,
    const mlx::core::array& route_order,
    const mlx::core::array& expert_ids);

mlx::core::array moe_swiglu_split(
    const mlx::core::array& gate_up);

mlx::core::array moe_limited_swiglu_split(
    const mlx::core::array& gate_up,
    float limit);

mlx::core::array moe_swiglu_pair(
    const mlx::core::array& gate,
    const mlx::core::array& up);

// Apply limited SwiGLU to separate Gate and Up projections without first
// materializing their concatenation.
mlx::core::array moe_limited_swiglu_pair(
    const mlx::core::array& gate,
    const mlx::core::array& up,
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

mlx::core::array moe_weighted_reduce_shared_gate_sorted(
    const mlx::core::array& sorted_pair_output,
    const mlx::core::array& inverse_route_order,
    const mlx::core::array& weights,
    const mlx::core::array& shared,
    const mlx::core::array& gate_logits);

mlx::core::array moe_apply_expert_scale(
    const mlx::core::array& weights,
    const mlx::core::array& ids,
    const mlx::core::array& scales);

} // namespace mfq::metal
