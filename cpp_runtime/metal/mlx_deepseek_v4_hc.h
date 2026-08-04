#pragma once

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV4HcPreResult {
    mlx::core::array reduced;
    mlx::core::array post;
    mlx::core::array combination;
};

// Reduce four DeepSeek-V4 hyper-connection streams to one branch input and
// compute the post gates plus the doubly-stochastic residual mixing matrix.
MlxDeepseekV4HcPreResult deepseek_v4_hc_pre(
    const mlx::core::array& residual,
    const mlx::core::array& mixes,
    const mlx::core::array& scale,
    const mlx::core::array& base,
    int sinkhorn_iterations = 20,
    float eps = 1e-6f);

// Decode/prefill fast path: perform the following learned RMSNorm inside the
// same threadgroup as HC collapse. The reduced field contains the normalized
// branch while post and combination retain their usual meanings.
MlxDeepseekV4HcPreResult deepseek_v4_hc_pre_norm(
    const mlx::core::array& residual,
    const mlx::core::array& mixes,
    const mlx::core::array& scale,
    const mlx::core::array& base,
    const mlx::core::array& norm,
    int sinkhorn_iterations = 20,
    float hc_eps = 1e-6f,
    float norm_eps = 1e-6f,
    bool normalize_mixes_from_residual = false);

// Expand one transformed branch back into four hyper-connection streams.
mlx::core::array deepseek_v4_hc_post(
    const mlx::core::array& branch,
    const mlx::core::array& residual,
    const mlx::core::array& post,
    const mlx::core::array& combination);

// HC expansion with a fused routed + shared MoE branch sum. This avoids a
// full hidden-width temporary and one elementwise dispatch per layer.
mlx::core::array deepseek_v4_hc_post_sum(
    const mlx::core::array& routed,
    const mlx::core::array& shared,
    const mlx::core::array& residual,
    const mlx::core::array& post,
    const mlx::core::array& combination);

} // namespace mfq::metal
