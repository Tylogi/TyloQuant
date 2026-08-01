// GLU activation helpers used by materialized prefill paths.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <algorithm>
#include <type_traits>
#include <vector>

#include "glu.cuh"

__global__ void silu_mul_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    size_t n)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void silu_mul_f16_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    size_t n)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        out[i] = __float2half((g / (1.0f + expf(-g))) * u);
    }
}

torch::Tensor silu_mul_cuda(torch::Tensor gate, torch::Tensor up)
{
    TORCH_CHECK(gate.is_cuda() && gate.is_contiguous(), "silu_mul: gate must be cuda contiguous");
    TORCH_CHECK(up.is_cuda() && up.is_contiguous(), "silu_mul: up must be cuda contiguous");
    TORCH_CHECK(gate.sizes() == up.sizes(), "silu_mul: gate/up shapes must match");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "silu_mul: gate/up dtype must match");
    TORCH_CHECK(gate.scalar_type() == torch::kFloat32 || gate.scalar_type() == torch::kFloat16,
                "silu_mul: dtype must be f32 or f16");

    auto out = torch::empty_like(gate);
    size_t n = (size_t)gate.numel();
    constexpr int BD = 256;
    int grid = (int)((n + BD - 1) / BD);
    grid = grid > 4096 ? 4096 : grid;
    if (gate.scalar_type() == torch::kFloat32) {
        silu_mul_f32_kernel<<<grid, BD, 0, at::cuda::getCurrentCUDAStream()>>>(
            gate.data_ptr<float>(), up.data_ptr<float>(), out.data_ptr<float>(), n);
    } else {
        silu_mul_f16_kernel<<<grid, BD, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            n);
    }
    return out;
}

template <typename scalar_t>
__global__ void gelu_mul_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ out,
    size_t n)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float value = mfq_glu<true>((float)gate[i], (float)up[i]);
        if constexpr (std::is_same_v<scalar_t, at::Half>) {
            value = fminf(65504.0f, fmaxf(-65504.0f, value));
        }
        out[i] = (scalar_t)value;
    }
}

torch::Tensor gelu_mul_cuda(torch::Tensor gate, torch::Tensor up)
{
    TORCH_CHECK(gate.is_cuda() && gate.is_contiguous(), "gelu_mul: gate must be cuda contiguous");
    TORCH_CHECK(up.is_cuda() && up.is_contiguous(), "gelu_mul: up must be cuda contiguous");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gelu_mul: gate/up shapes must match");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gelu_mul: gate/up dtype must match");
    TORCH_CHECK(gate.scalar_type() == torch::kFloat32 || gate.scalar_type() == torch::kFloat16,
                "gelu_mul: dtype must be f32 or f16");

    auto out = torch::empty_like(gate);
    const size_t n = (size_t)gate.numel();
    constexpr int block = 256;
    const int grid = (int)std::min<size_t>(4096, (n + block - 1) / block);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gate.scalar_type(), "gelu_mul_cuda", [&] {
        gelu_mul_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
    return out;
}

template <typename scalar_t>
__global__ void linear_gate_beta_kernel(
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ beta,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    float* __restrict__ gate_t,
    float* __restrict__ beta_t,
    int64_t B, int64_t T, int64_t V,
    int64_t as0, int64_t as1, int64_t as2,
    int64_t bs0, int64_t bs1, int64_t bs2)
{
    size_t n = (size_t)B * (size_t)T * (size_t)V;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        int64_t v = (int64_t)(idx % (size_t)V);
        int64_t t = (int64_t)((idx / (size_t)V) % (size_t)T);
        int64_t b = (int64_t)(idx / ((size_t)T * (size_t)V));
        float a = (float)alpha[b * as0 + t * as1 + v * as2] + dt_bias[v];
        float sp = a > 20.0f ? a : log1pf(expf(a));
        float g = sp * -expf(a_log[v]);
        float bv = (float)beta[b * bs0 + t * bs1 + v * bs2];
        float sig = 1.0f / (1.0f + expf(-bv));
        size_t out = ((size_t)b * (size_t)V + (size_t)v) * (size_t)T + (size_t)t;
        gate_t[out] = g;
        beta_t[out] = sig;
    }
}

std::vector<torch::Tensor> linear_gate_beta_cuda(
    torch::Tensor alpha, torch::Tensor beta, torch::Tensor dt_bias, torch::Tensor a_log)
{
    TORCH_CHECK(alpha.is_cuda() && beta.is_cuda(), "linear_gate_beta: alpha/beta must be cuda tensors");
    TORCH_CHECK(dt_bias.is_cuda() && a_log.is_cuda(), "linear_gate_beta: dt_bias/a_log must be cuda tensors");
    TORCH_CHECK(alpha.dim() == 3 && beta.dim() == 3, "linear_gate_beta: alpha/beta must be [B,T,V]");
    TORCH_CHECK(alpha.sizes() == beta.sizes(), "linear_gate_beta: alpha/beta shape mismatch");
    TORCH_CHECK(alpha.scalar_type() == beta.scalar_type(), "linear_gate_beta: alpha/beta dtype mismatch");
    TORCH_CHECK(alpha.scalar_type() == torch::kFloat32 || alpha.scalar_type() == torch::kFloat16,
                "linear_gate_beta: alpha/beta dtype must be f32 or f16");
    TORCH_CHECK(dt_bias.is_contiguous() && a_log.is_contiguous(), "linear_gate_beta: dt_bias/a_log must be contiguous");
    TORCH_CHECK(dt_bias.scalar_type() == torch::kFloat32 && a_log.scalar_type() == torch::kFloat32,
                "linear_gate_beta: dt_bias/a_log must be f32");
    int64_t B = alpha.size(0);
    int64_t T = alpha.size(1);
    int64_t V = alpha.size(2);
    TORCH_CHECK(dt_bias.numel() == V && a_log.numel() == V, "linear_gate_beta: parameter size mismatch");

    auto opts = torch::TensorOptions().device(alpha.device()).dtype(torch::kFloat32);
    auto gate_t = torch::empty({B, V, T}, opts);
    auto beta_t = torch::empty({B, V, T}, opts);
    size_t n = (size_t)B * (size_t)T * (size_t)V;
    constexpr int BD = 256;
    int grid = (int)((n + BD - 1) / BD);
    grid = grid > 4096 ? 4096 : grid;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(alpha.scalar_type(), "linear_gate_beta_cuda", [&] {
        linear_gate_beta_kernel<scalar_t><<<grid, BD, 0, at::cuda::getCurrentCUDAStream()>>>(
            alpha.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            dt_bias.data_ptr<float>(),
            a_log.data_ptr<float>(),
            gate_t.data_ptr<float>(),
            beta_t.data_ptr<float>(),
            B, T, V,
            alpha.stride(0), alpha.stride(1), alpha.stride(2),
            beta.stride(0), beta.stride(1), beta.stride(2));
    });
    return {gate_t, beta_t};
}
