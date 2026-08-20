#pragma once

#include "mfq_cuda_tensor_view.h"

#include <cuda_runtime_api.h>

namespace mfq::cuda::kernels {

void silu_mul(
    const TensorView& gate,
    const TensorView& up,
    const TensorView& output,
    cudaStream_t stream);

void gelu_mul(
    const TensorView& gate,
    const TensorView& up,
    const TensorView& output,
    cudaStream_t stream);

void linear_gate_beta(
    const TensorView& alpha,
    const TensorView& beta,
    const TensorView& dt_bias,
    const TensorView& a_log,
    const TensorView& gate_t,
    const TensorView& beta_t,
    cudaStream_t stream);

}  // namespace mfq::cuda::kernels
