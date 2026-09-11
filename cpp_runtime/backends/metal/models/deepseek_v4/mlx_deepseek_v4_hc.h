#pragma once

#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV4HcPreResult {
    mlx::core::array reduced;
    mlx::core::array post;
    mlx::core::array combination;
    std::optional<mlx::core::array> packed_metadata;
    // V4.1 carries the pre coefficients to the following sublayer instead of
    // consuming them at the site where they are produced. Generic callers
    // may omit this value, hence the optional field.
    std::optional<mlx::core::array> pre;
};

struct MlxDeepseekV41HcMetadataResult {
    mlx::core::array post;
    mlx::core::array combination;
    mlx::core::array pre;
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

// V4.1 carries four HC pre coefficients from the preceding sublayer. Collapse
// those streams and apply the following BF16 RMSNorm without materializing the
// generic graph's FP32 [batch,tokens,4,hidden] product tensor.
mlx::core::array deepseek_v41_hc_collapse_norm(
    const mlx::core::array& residual,
    const mlx::core::array& pre,
    const mlx::core::array& norm,
    float norm_eps = 1e-6f);

// Exact single-row V4.1 HC gate and Sinkhorn path. This preserves the FP32
// generic graph while collapsing the launch-bound Sinkhorn iterations into
// one Metal dispatch.
MlxDeepseekV41HcMetadataResult deepseek_v41_hc_metadata_exact(
    const mlx::core::array& normalized_mixes,
    const mlx::core::array& scale,
    const mlx::core::array& base,
    int sinkhorn_iterations = 20,
    float eps = 1e-6f);

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

// Fast-path variant consuming the single packed FP32 metadata output from
// deepseek_v4_hc_pre[_norm]. Layout: post[4], combination[4x4].
mlx::core::array deepseek_v4_hc_post_packed(
    const mlx::core::array& branch,
    const mlx::core::array& residual,
    const mlx::core::array& metadata);

mlx::core::array deepseek_v4_hc_post_sum_packed(
    const mlx::core::array& routed,
    const mlx::core::array& shared,
    const mlx::core::array& residual,
    const mlx::core::array& metadata);

} // namespace mfq::metal
