#pragma once

#include "cpp_runtime/backends/cuda/include/mfq_tensor_backend.h"

// Shared native/Torch entry points. Equations and dtype boundaries follow
// mfq/kernels/metal/flash_next.py; these are not aliases for older backbones.
namespace mfq_flash_next {
using Tensor = mfq_tensor_backend::Tensor;

Tensor qwen4_grouped_rms_norm(const Tensor&, const Tensor&, int64_t, double = 1e-6);
std::vector<Tensor> qwen4_gated_residual_pre(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const std::optional<Tensor>&, int64_t, int64_t = 4, double = 1e-6);
Tensor qwen4_gated_residual_post(const Tensor&, const Tensor&, const Tensor&, int64_t = 4);
std::vector<Tensor> glm5_mhc_pre(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    int64_t = 20, double = 1e-6, double = 1e-5);
Tensor glm5_mhc_post(const Tensor&, const Tensor&, const Tensor&, const Tensor&);
Tensor glm5_kda_forget_gate(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    int64_t, int64_t, double = -5.0);
Tensor qsa_block_scores(const Tensor&, const Tensor&);
Tensor glm5_kpool_scores(const Tensor&, const Tensor&, const Tensor&);
Tensor glm5_kpool_states(const Tensor&, const Tensor&, const Tensor&, int64_t = 4);
std::vector<Tensor> qwen4_ple_dilated_conv_silu(
    const Tensor&, const Tensor&, const std::optional<Tensor>&, int64_t);
Tensor qwen4_dense_gqa_attention(const Tensor&, const Tensor&, const Tensor&, int64_t);
Tensor qwen4_sparse_gqa_attention(const Tensor&, const Tensor&, const Tensor&, const Tensor&);
Tensor glm5_dense_mla_attention(
    const Tensor&, const Tensor&, int64_t, std::optional<double> = std::nullopt);
Tensor glm5_sparse_mla_attention(
    const Tensor&, const Tensor&, const Tensor&, std::optional<double> = std::nullopt);
} // namespace mfq_flash_next
