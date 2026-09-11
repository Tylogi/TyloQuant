#include "mfq/kernels/cuda/deepseek_v41.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

__device__ __forceinline__ float e4m3(float value) {
    const float bounded = fminf(448.0f, fmaxf(-448.0f, value));
    return static_cast<float>(__nv_fp8_e4m3(bounded));
}

__device__ __forceinline__ float pow2_ceil(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t exponent = (bits >> 23u) & 0xffu;
    const bool has_mantissa = (bits & 0x7fffffu) != 0u;
    return __uint_as_float(
        (exponent + static_cast<std::uint32_t>(has_mantissa)) << 23u);
}

__device__ __forceinline__ float e2m1(float value, float scale) {
    const float normalized = fminf(6.0f, fmaxf(-6.0f, value / scale));
    const float magnitude = fabsf(normalized);
    float quantized = 0.0f;
    if (magnitude <= 0.25f) quantized = 0.0f;
    else if (magnitude < 0.75f) quantized = 0.5f;
    else if (magnitude <= 1.25f) quantized = 1.0f;
    else if (magnitude < 1.75f) quantized = 1.5f;
    else if (magnitude <= 2.5f) quantized = 2.0f;
    else if (magnitude < 3.5f) quantized = 3.0f;
    else if (magnitude <= 5.0f) quantized = 4.0f;
    else quantized = 6.0f;
    return copysignf(quantized * scale, normalized);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__global__ void mxfp8_sim_kernel(
    const half* __restrict__ input,
    half* __restrict__ output) {
    const int lane = threadIdx.x;
    const std::int64_t offset =
        static_cast<std::int64_t>(blockIdx.x) * 32 + lane;
    const float value = __half2float(input[offset]);
    const float maximum = warp_max(fabsf(value));
    const float scale = pow2_ceil(
        fmaxf(maximum, 448.0f * 0x1p-126f) / 448.0f);
    output[offset] = __float2half_rn(e4m3(value / scale) * scale);
}

__global__ void mxfp4_e4m3_scale_sim_kernel(
    const half* __restrict__ input,
    half* __restrict__ output) {
    const int lane = threadIdx.x;
    const std::int64_t offset =
        static_cast<std::int64_t>(blockIdx.x) * 16 +
        static_cast<std::int64_t>(lane < 16 ? lane : 0);
    const float value = lane < 16 ? __half2float(input[offset]) : 0.0f;
    const float maximum = warp_max(fabsf(value));
    const float raw_scale = fmaxf(maximum, 6.0f * 0x1p-9f) / 6.0f;
    const float scale = fmaxf(e4m3(raw_scale), 0x1p-9f);
    if (lane < 16) {
        output[offset] = __float2half_rn(e2m1(value, scale));
    }
}

void require_input(
    const mfq_tensor_backend::Tensor& input,
    std::int64_t group,
    const char* operation) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
            input.scalar_type() == mfq_tensor_backend::kFloat16 &&
            input.dim() >= 1 && input.numel() > 0 &&
            input.size(-1) % group == 0,
        operation,
        ": expected contiguous CUDA f16 with a grouped last dimension");
    MFQ_RUNTIME_CHECK(
        input.numel() / group <= std::numeric_limits<int>::max(),
        operation, ": input is too large");
}

} // namespace

mfq_tensor_backend::Tensor deepseek_v41_mxfp8_e4m3_sim_cuda(
    mfq_tensor_backend::Tensor input) {
    require_input(input, 32, "deepseek_v41_mxfp8_e4m3_sim");
    auto output = mfq_tensor_backend::empty_like(input);
    mxfp8_sim_kernel<<<
        static_cast<int>(input.numel() / 32), 32, 0,
        mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<mfq_half>()),
        reinterpret_cast<half*>(output.data_ptr<mfq_half>()));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

mfq_tensor_backend::Tensor deepseek_v41_mxfp4_e4m3_scale_sim_cuda(
    mfq_tensor_backend::Tensor input) {
    require_input(input, 16, "deepseek_v41_mxfp4_e4m3_scale_sim");
    auto output = mfq_tensor_backend::empty_like(input);
    mxfp4_e4m3_scale_sim_kernel<<<
        static_cast<int>(input.numel() / 16), 32, 0,
        mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<mfq_half>()),
        reinterpret_cast<half*>(output.data_ptr<mfq_half>()));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
