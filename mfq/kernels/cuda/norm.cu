// RMSNorm / L2 Norm (ggml norm.cu: ggml_rms_norm / ggml_l2_norm).
// One block normalizes one row (last dim D). fp32. x viewed as [N, D] (caller flattens).

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>

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
__global__ void rms_norm_pair_f16_f32_kernel(
    const at::Half* __restrict__ first,
    const at::Half* __restrict__ second,
    const float* __restrict__ first_weight,
    const float* __restrict__ second_weight,
    float* __restrict__ first_out,
    float* __restrict__ second_out,
    int first_rows,
    int second_rows,
    int D,
    float eps,
    float weight_offset)
{
    const int combined_row = blockIdx.x;
    if (combined_row >= first_rows + second_rows) return;
    const bool use_second = combined_row >= first_rows;
    const int row = use_second ? combined_row - first_rows : combined_row;
    const at::Half* input = use_second ? second : first;
    const float* weight = use_second ? second_weight : first_weight;
    float* output = use_second ? second_out : first_out;
    const at::Half* input_row = input + (size_t)row * D;
    float* output_row = output + (size_t)row * D;
    const int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BD) {
        const float value = (float)input_row[i];
        ssq += value * value;
    }
    ssq = block_sum<NORM_BD / 32>(ssq);

    const float inverse = rsqrtf(ssq / (float)D + eps);
    for (int i = tid; i < D; i += BD) {
        output_row[i] = (float)input_row[i] * inverse *
            (weight[i] + weight_offset);
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

std::vector<torch::Tensor> rms_norm_pair_f16_f32_offset_cuda(
    torch::Tensor first,
    torch::Tensor second,
    torch::Tensor first_weight,
    torch::Tensor second_weight,
    double eps,
    double weight_offset)
{
    TORCH_CHECK(
        first.is_cuda() && first.is_contiguous() &&
        first.scalar_type() == torch::kFloat16,
        "paired RMSNorm first input must be CUDA contiguous fp16");
    TORCH_CHECK(
        second.is_cuda() && second.is_contiguous() &&
        second.scalar_type() == torch::kFloat16,
        "paired RMSNorm second input must be CUDA contiguous fp16");
    TORCH_CHECK(
        first_weight.is_cuda() && first_weight.is_contiguous() &&
        first_weight.scalar_type() == torch::kFloat32,
        "paired RMSNorm first weight must be CUDA contiguous fp32");
    TORCH_CHECK(
        second_weight.is_cuda() && second_weight.is_contiguous() &&
        second_weight.scalar_type() == torch::kFloat32,
        "paired RMSNorm second weight must be CUDA contiguous fp32");
    const int D = (int)first.size(-1);
    TORCH_CHECK(second.size(-1) == D, "paired RMSNorm widths must match");
    TORCH_CHECK(
        first_weight.numel() == D && second_weight.numel() == D,
        "paired RMSNorm weight lengths must match the input width");
    const int first_rows = (int)(first.numel() / D);
    const int second_rows = (int)(second.numel() / D);
    auto output_options = first.options().dtype(torch::kFloat32);
    auto first_out = torch::empty(first.sizes(), output_options);
    auto second_out = torch::empty(second.sizes(), output_options);
    rms_norm_pair_f16_f32_kernel<NORM_BD><<<
        first_rows + second_rows, NORM_BD, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        first.data_ptr<at::Half>(), second.data_ptr<at::Half>(),
        first_weight.data_ptr<float>(), second_weight.data_ptr<float>(),
        first_out.data_ptr<float>(), second_out.data_ptr<float>(),
        first_rows, second_rows, D, (float)eps, (float)weight_offset);
    return {first_out, second_out};
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
