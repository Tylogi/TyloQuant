// RMSNorm / L2 Norm (ggml norm.cu: ggml_rms_norm / ggml_l2_norm).
// One block normalizes one row (last dim D). fp32. x viewed as [N, D] (caller flattens).

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "reduce.cuh"

constexpr int NORM_BD = 256;

template <int BD>
__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                float* __restrict__ out, int N, int D, float eps, float weight_offset)
{
    int row = blockIdx.x;
    if (row >= N) {
        return;
    }
    const float* xr = x + (size_t)row * D;
    float* or_ = out + (size_t)row * D;
    int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BD) {
        float xi = xr[i];
        ssq += xi * xi;
    }
    ssq = block_sum<NORM_BD / 32>(ssq);

    float rinv = rsqrtf(ssq / (float)D + eps);
    for (int i = tid; i < D; i += BD) {
        or_[i] = xr[i] * rinv * (w[i] + weight_offset);
    }
}

template <int BD>
__global__ void rms_norm_f16_kernel(const at::Half* __restrict__ x,
                                    const float* __restrict__ w,
                                    at::Half* __restrict__ out,
                                    int N, int D, float eps, float weight_offset)
{
    int row = blockIdx.x;
    if (row >= N) {
        return;
    }
    const at::Half* xr = x + (size_t)row * D;
    at::Half* or_ = out + (size_t)row * D;
    int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BD) {
        float xi = (float)xr[i];
        ssq += xi * xi;
    }
    ssq = block_sum<NORM_BD / 32>(ssq);

    float rinv = rsqrtf(ssq / (float)D + eps);
    for (int i = tid; i < D; i += BD) {
        or_[i] = (at::Half)((float)xr[i] * rinv * (w[i] + weight_offset));
    }
}

template <int BD>
__global__ void l2_norm_kernel(const float* __restrict__ x, float* __restrict__ out,
                               int N, int D, float eps)
{
    int row = blockIdx.x;
    if (row >= N) {
        return;
    }
    const float* xr = x + (size_t)row * D;
    float* or_ = out + (size_t)row * D;
    int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BD) {
        float xi = xr[i];
        ssq += xi * xi;
    }
    ssq = block_sum<NORM_BD / 32>(ssq);

    float nrm = sqrtf(ssq);
    float denom = fmaxf(nrm, eps);   // matches F.normalize clamp_min(eps)
    float inv = 1.0f / denom;
    for (int i = tid; i < D; i += BD) {
        or_[i] = xr[i] * inv;
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, double eps)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32,
                "rms_norm: x must be cuda contiguous f32");
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kFloat32, "weight must be cuda f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    auto out = torch::empty_like(x);
    rms_norm_kernel<NORM_BD><<<N, NORM_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), N, D, (float)eps, 0.0f);
    return out;
}

torch::Tensor rms_norm_offset_cuda(torch::Tensor x, torch::Tensor weight, double eps, double weight_offset)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32,
                "rms_norm_offset: x must be cuda contiguous f32");
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kFloat32, "weight must be cuda f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    auto out = torch::empty_like(x);
    rms_norm_kernel<NORM_BD><<<N, NORM_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), N, D,
        (float)eps, (float)weight_offset);
    return out;
}

torch::Tensor rms_norm_f16_cuda(torch::Tensor x, torch::Tensor weight, double eps,
                                double weight_offset)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat16,
                "rms_norm_f16: x must be cuda contiguous f16");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                    weight.scalar_type() == torch::kFloat32,
                "rms_norm_f16: weight must be cuda contiguous f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    TORCH_CHECK(weight.numel() == D, "rms_norm_f16: weight length mismatch");
    auto out = torch::empty_like(x);
    rms_norm_f16_kernel<NORM_BD><<<N, NORM_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<at::Half>(), weight.data_ptr<float>(), out.data_ptr<at::Half>(),
        N, D, (float)eps, (float)weight_offset);
    return out;
}

torch::Tensor l2_norm_cuda(torch::Tensor x, double eps)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32,
                "l2_norm: x must be cuda contiguous f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    auto out = torch::empty_like(x);
    l2_norm_kernel<NORM_BD><<<N, NORM_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N, D, (float)eps);
    return out;
}
