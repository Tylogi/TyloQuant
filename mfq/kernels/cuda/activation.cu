// GLU activation helpers used by materialized prefill paths.

#include <cuda_fp16.h>
#include "../../../cpp_runtime/cuda/mfq_tensor_backend.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "glu.cuh"
#include "../../../cpp_runtime/cuda/mfq_cuda_kernels.h"

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

__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    size_t n)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        const __nv_bfloat16 activated = __float2bfloat16(
            g / (1.0f + expf(-g)));
        out[i] = __float2bfloat16(
            __bfloat162float(activated) * u);
    }
}

mfq_tensor_backend::Tensor silu_mul_cuda(mfq_tensor_backend::Tensor gate, mfq_tensor_backend::Tensor up)
{
    MFQ_RUNTIME_CHECK(gate.is_cuda() && gate.is_contiguous(), "silu_mul: gate must be cuda contiguous");
    MFQ_RUNTIME_CHECK(up.is_cuda() && up.is_contiguous(), "silu_mul: up must be cuda contiguous");
    MFQ_RUNTIME_CHECK(gate.sizes() == up.sizes(), "silu_mul: gate/up shapes must match");
    MFQ_RUNTIME_CHECK(gate.scalar_type() == up.scalar_type(), "silu_mul: gate/up dtype must match");
    MFQ_RUNTIME_CHECK(
        gate.scalar_type() == mfq_tensor_backend::kFloat32 ||
        gate.scalar_type() == mfq_tensor_backend::kFloat16 ||
        gate.scalar_type() == mfq_tensor_backend::kBFloat16,
        "silu_mul: dtype must be f32, f16, or bf16");

    auto out = mfq_tensor_backend::empty_like(gate);
    const auto dtype = gate.scalar_type() == mfq_tensor_backend::kFloat32
        ? mfq::cuda::ScalarType::float32
        : gate.scalar_type() == mfq_tensor_backend::kFloat16
            ? mfq::cuda::ScalarType::float16
            : mfq::cuda::ScalarType::bfloat16;
    const auto shape = gate.sizes().vec();
    mfq::cuda::kernels::silu_mul(
        mfq::cuda::make_contiguous_view(
            gate.data_ptr(), shape, dtype,
            mfq::cuda::Device{mfq::cuda::DeviceType::cuda, gate.get_device()}),
        mfq::cuda::make_contiguous_view(
            up.data_ptr(), shape, dtype,
            mfq::cuda::Device{mfq::cuda::DeviceType::cuda, up.get_device()}),
        mfq::cuda::make_contiguous_view(
            out.data_ptr(), shape, dtype,
            mfq::cuda::Device{mfq::cuda::DeviceType::cuda, out.get_device()}),
        mfq_current_cuda_stream());
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
        if constexpr (!std::is_same_v<scalar_t, float>) {
            value = fminf(65504.0f, fmaxf(-65504.0f, value));
        }
        out[i] = (scalar_t)value;
    }
}

mfq_tensor_backend::Tensor gelu_mul_cuda(mfq_tensor_backend::Tensor gate, mfq_tensor_backend::Tensor up)
{
    MFQ_RUNTIME_CHECK(gate.is_cuda() && gate.is_contiguous(), "gelu_mul: gate must be cuda contiguous");
    MFQ_RUNTIME_CHECK(up.is_cuda() && up.is_contiguous(), "gelu_mul: up must be cuda contiguous");
    MFQ_RUNTIME_CHECK(gate.sizes() == up.sizes(), "gelu_mul: gate/up shapes must match");
    MFQ_RUNTIME_CHECK(gate.scalar_type() == up.scalar_type(), "gelu_mul: gate/up dtype must match");
    MFQ_RUNTIME_CHECK(gate.scalar_type() == mfq_tensor_backend::kFloat32 || gate.scalar_type() == mfq_tensor_backend::kFloat16,
                "gelu_mul: dtype must be f32 or f16");

    auto out = mfq_tensor_backend::empty_like(gate);
    const auto dtype = gate.scalar_type() == mfq_tensor_backend::kFloat32
        ? mfq::cuda::ScalarType::float32
        : mfq::cuda::ScalarType::float16;
    const auto shape = gate.sizes().vec();
    mfq::cuda::kernels::gelu_mul(
        mfq::cuda::make_contiguous_view(
            gate.data_ptr(), shape, dtype,
            mfq::cuda::Device{mfq::cuda::DeviceType::cuda, gate.get_device()}),
        mfq::cuda::make_contiguous_view(
            up.data_ptr(), shape, dtype,
            mfq::cuda::Device{mfq::cuda::DeviceType::cuda, up.get_device()}),
        mfq::cuda::make_contiguous_view(
            out.data_ptr(), shape, dtype,
            mfq::cuda::Device{mfq::cuda::DeviceType::cuda, out.get_device()}),
        mfq_current_cuda_stream());
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

std::vector<mfq_tensor_backend::Tensor> linear_gate_beta_cuda(
    mfq_tensor_backend::Tensor alpha, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor dt_bias, mfq_tensor_backend::Tensor a_log)
{
    MFQ_RUNTIME_CHECK(alpha.is_cuda() && beta.is_cuda(), "linear_gate_beta: alpha/beta must be cuda tensors");
    MFQ_RUNTIME_CHECK(dt_bias.is_cuda() && a_log.is_cuda(), "linear_gate_beta: dt_bias/a_log must be cuda tensors");
    MFQ_RUNTIME_CHECK(alpha.dim() == 3 && beta.dim() == 3, "linear_gate_beta: alpha/beta must be [B,T,V]");
    MFQ_RUNTIME_CHECK(alpha.sizes() == beta.sizes(), "linear_gate_beta: alpha/beta shape mismatch");
    MFQ_RUNTIME_CHECK(alpha.scalar_type() == beta.scalar_type(), "linear_gate_beta: alpha/beta dtype mismatch");
    MFQ_RUNTIME_CHECK(alpha.scalar_type() == mfq_tensor_backend::kFloat32 || alpha.scalar_type() == mfq_tensor_backend::kFloat16,
                "linear_gate_beta: alpha/beta dtype must be f32 or f16");
    MFQ_RUNTIME_CHECK(dt_bias.is_contiguous() && a_log.is_contiguous(), "linear_gate_beta: dt_bias/a_log must be contiguous");
    MFQ_RUNTIME_CHECK(dt_bias.scalar_type() == mfq_tensor_backend::kFloat32 && a_log.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_gate_beta: dt_bias/a_log must be f32");
    int64_t B = alpha.size(0);
    int64_t T = alpha.size(1);
    int64_t V = alpha.size(2);
    MFQ_RUNTIME_CHECK(dt_bias.numel() == V && a_log.numel() == V, "linear_gate_beta: parameter size mismatch");

    auto opts = mfq_tensor_backend::TensorOptions().device(alpha.device()).dtype(mfq_tensor_backend::kFloat32);
    auto gate_t = mfq_tensor_backend::empty({B, V, T}, opts);
    auto beta_t = mfq_tensor_backend::empty({B, V, T}, opts);
    const auto dtype = alpha.scalar_type() == mfq_tensor_backend::kFloat32
        ? mfq::cuda::ScalarType::float32
        : mfq::cuda::ScalarType::float16;
    const auto device = mfq::cuda::Device{mfq::cuda::DeviceType::cuda, alpha.get_device()};
    mfq::cuda::TensorView alpha_view;
    alpha_view.data = alpha.data_ptr();
    alpha_view.rank = 3;
    alpha_view.scalar_type = dtype;
    alpha_view.device = device;
    mfq::cuda::TensorView beta_view = alpha_view;
    beta_view.data = beta.data_ptr();
    for (int index = 0; index < 3; ++index) {
        alpha_view.sizes[index] = alpha.size(index);
        alpha_view.strides[index] = alpha.stride(index);
        beta_view.sizes[index] = beta.size(index);
        beta_view.strides[index] = beta.stride(index);
    }
    const std::array<std::int64_t, 1> parameter_shape = {V};
    const std::array<std::int64_t, 3> output_shape = {B, V, T};
    mfq::cuda::kernels::linear_gate_beta(
        alpha_view,
        beta_view,
        mfq::cuda::make_contiguous_view(
            dt_bias.data_ptr(), parameter_shape, mfq::cuda::ScalarType::float32, device),
        mfq::cuda::make_contiguous_view(
            a_log.data_ptr(), parameter_shape, mfq::cuda::ScalarType::float32, device),
        mfq::cuda::make_contiguous_view(
            gate_t.data_ptr(), output_shape, mfq::cuda::ScalarType::float32, device),
        mfq::cuda::make_contiguous_view(
            beta_t.data_ptr(), output_shape, mfq::cuda::ScalarType::float32, device),
        mfq_current_cuda_stream());
    return {gate_t, beta_t};
}


namespace mfq::cuda::kernels {

void silu_mul(
    const TensorView& gate,
    const TensorView& up,
    const TensorView& output,
    cudaStream_t stream) {
    if (!gate.is_contiguous() || !up.is_contiguous() || !output.is_contiguous() ||
        gate.numel() != up.numel() || gate.numel() != output.numel() ||
        gate.scalar_type != up.scalar_type || gate.scalar_type != output.scalar_type) {
        throw std::invalid_argument("native silu_mul tensor mismatch");
    }
    const auto elements = static_cast<std::size_t>(gate.numel());
    if (elements == 0) return;
    constexpr int block = 256;
    const int grid = static_cast<int>(std::min<std::size_t>(
        4096, (elements + block - 1) / block));
    if (gate.scalar_type == ScalarType::float32) {
        silu_mul_f32_kernel<<<grid, block, 0, stream>>>(
            gate.data_as<float>(), up.data_as<float>(), output.data_as<float>(), elements);
    } else if (gate.scalar_type == ScalarType::float16) {
        silu_mul_f16_kernel<<<grid, block, 0, stream>>>(
            gate.data_as<half>(), up.data_as<half>(), output.data_as<half>(), elements);
    } else if (gate.scalar_type == ScalarType::bfloat16) {
        silu_mul_bf16_kernel<<<grid, block, 0, stream>>>(
            gate.data_as<__nv_bfloat16>(), up.data_as<__nv_bfloat16>(),
            output.data_as<__nv_bfloat16>(), elements);
    } else {
        throw std::invalid_argument("native silu_mul dtype must be f32, f16, or bf16");
    }
}

void gelu_mul(
    const TensorView& gate,
    const TensorView& up,
    const TensorView& output,
    cudaStream_t stream) {
    if (!gate.is_contiguous() || !up.is_contiguous() || !output.is_contiguous() ||
        gate.numel() != up.numel() || gate.numel() != output.numel() ||
        gate.scalar_type != up.scalar_type || gate.scalar_type != output.scalar_type) {
        throw std::invalid_argument("native gelu_mul tensor mismatch");
    }
    const auto elements = static_cast<std::size_t>(gate.numel());
    if (elements == 0) return;
    constexpr int block = 256;
    const int grid = static_cast<int>(std::min<std::size_t>(
        4096, (elements + block - 1) / block));
    if (gate.scalar_type == ScalarType::float32) {
        gelu_mul_kernel<float><<<grid, block, 0, stream>>>(
            gate.data_as<float>(), up.data_as<float>(), output.data_as<float>(), elements);
    } else if (gate.scalar_type == ScalarType::float16) {
        gelu_mul_kernel<half><<<grid, block, 0, stream>>>(
            gate.data_as<half>(), up.data_as<half>(), output.data_as<half>(), elements);
    } else {
        throw std::invalid_argument("native gelu_mul dtype must be f32 or f16");
    }
}

void linear_gate_beta(
    const TensorView& alpha,
    const TensorView& beta,
    const TensorView& dt_bias,
    const TensorView& a_log,
    const TensorView& gate_t,
    const TensorView& beta_t,
    cudaStream_t stream) {
    if (alpha.rank != 3 || beta.rank != 3 || alpha.scalar_type != beta.scalar_type ||
        alpha.sizes != beta.sizes || dt_bias.scalar_type != ScalarType::float32 ||
        a_log.scalar_type != ScalarType::float32 ||
        gate_t.scalar_type != ScalarType::float32 || beta_t.scalar_type != ScalarType::float32) {
        throw std::invalid_argument("native linear_gate_beta tensor mismatch");
    }
    const auto batch = alpha.sizes[0];
    const auto tokens = alpha.sizes[1];
    const auto width = alpha.sizes[2];
    if (dt_bias.numel() != width || a_log.numel() != width ||
        gate_t.numel() != alpha.numel() || beta_t.numel() != alpha.numel()) {
        throw std::invalid_argument("native linear_gate_beta shape mismatch");
    }
    const auto elements = static_cast<std::size_t>(alpha.numel());
    if (elements == 0) return;
    constexpr int block = 256;
    const int grid = static_cast<int>(std::min<std::size_t>(
        4096, (elements + block - 1) / block));
    if (alpha.scalar_type == ScalarType::float32) {
        linear_gate_beta_kernel<float><<<grid, block, 0, stream>>>(
            alpha.data_as<float>(), beta.data_as<float>(),
            dt_bias.data_as<float>(), a_log.data_as<float>(),
            gate_t.data_as<float>(), beta_t.data_as<float>(),
            batch, tokens, width,
            alpha.strides[0], alpha.strides[1], alpha.strides[2],
            beta.strides[0], beta.strides[1], beta.strides[2]);
    } else if (alpha.scalar_type == ScalarType::float16) {
        linear_gate_beta_kernel<half><<<grid, block, 0, stream>>>(
            alpha.data_as<half>(), beta.data_as<half>(),
            dt_bias.data_as<float>(), a_log.data_as<float>(),
            gate_t.data_as<float>(), beta_t.data_as<float>(),
            batch, tokens, width,
            alpha.strides[0], alpha.strides[1], alpha.strides[2],
            beta.strides[0], beta.strides[1], beta.strides[2]);
    } else {
        throw std::invalid_argument("native linear_gate_beta dtype must be f32 or f16");
    }
}

}  // namespace mfq::cuda::kernels
