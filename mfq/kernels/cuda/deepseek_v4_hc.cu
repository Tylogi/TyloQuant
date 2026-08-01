#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kHyperConnections = 4;
constexpr int kHyperMixWidth =
    (2 + kHyperConnections) * kHyperConnections;
constexpr int kHyperSinkhornIterations = 20;

__device__ __forceinline__ float dsv4_sigmoid(float value) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
}

__device__ __forceinline__ float dsv4_sum4(
    float x0, float x1, float x2, float x3) {
    return __fadd_rn(__fadd_rn(__fadd_rn(x0, x1), x2), x3);
}

__device__ __forceinline__ float dsv4_row_sum4(
    float x0, float x1, float x2, float x3) {
    return __fadd_rn(
        __fadd_rn(x0, x1),
        __fadd_rn(x2, x3));
}

__device__ __forceinline__ float dsv4_softmax_sum4(
    float x0, float x1, float x2, float x3) {
    return __fadd_rn(
        __fadd_rn(x0, x2),
        __fadd_rn(x1, x3));
}

__global__ void dsv4_hc_pre_finalize_kernel(
    const half * __restrict__ x,
    const float * __restrict__ mixes,
    const float * __restrict__ scale,
    const float * __restrict__ base,
    half * __restrict__ reduced,
    float * __restrict__ post,
    float * __restrict__ combination,
    int rows,
    int hidden,
    float eps)
{
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    if (row >= rows) return;

    __shared__ float shared_pre[kHyperConnections];
    __shared__ float shared_comb[
        kHyperConnections * kHyperConnections];

    if (thread < 32) {
        if (thread < kHyperConnections) {
            const float pre_affine = __fadd_rn(
                __fmul_rn(
                    mixes[static_cast<int64_t>(row) * kHyperMixWidth +
                           thread],
                    scale[0]),
                base[thread]);
            shared_pre[thread] = __fadd_rn(
                dsv4_sigmoid(pre_affine), eps);

            const int post_index = kHyperConnections + thread;
            const float post_affine = __fadd_rn(
                __fmul_rn(
                    mixes[static_cast<int64_t>(row) * kHyperMixWidth +
                           post_index],
                    scale[1]),
                base[post_index]);
            post[static_cast<int64_t>(row) * kHyperConnections + thread] =
                __fmul_rn(2.0f, dsv4_sigmoid(post_affine));
        }
        if (thread < kHyperConnections * kHyperConnections) {
            const int mix_index = 2 * kHyperConnections + thread;
            shared_comb[thread] = __fadd_rn(
                __fmul_rn(
                    mixes[static_cast<int64_t>(row) * kHyperMixWidth +
                           mix_index],
                    scale[2]),
                base[mix_index]);
        }
        __syncwarp();

        if (thread < kHyperConnections) {
            const int offset = thread * kHyperConnections;
            const float maximum = fmaxf(
                fmaxf(shared_comb[offset], shared_comb[offset + 1]),
                fmaxf(shared_comb[offset + 2], shared_comb[offset + 3]));
            const float e0 = expf(shared_comb[offset] - maximum);
            const float e1 = expf(shared_comb[offset + 1] - maximum);
            const float e2 = expf(shared_comb[offset + 2] - maximum);
            const float e3 = expf(shared_comb[offset + 3] - maximum);
            const float sum = dsv4_softmax_sum4(e0, e1, e2, e3);
            shared_comb[offset] = __fadd_rn(__fdiv_rn(e0, sum), eps);
            shared_comb[offset + 1] =
                __fadd_rn(__fdiv_rn(e1, sum), eps);
            shared_comb[offset + 2] =
                __fadd_rn(__fdiv_rn(e2, sum), eps);
            shared_comb[offset + 3] =
                __fadd_rn(__fdiv_rn(e3, sum), eps);
        }
        __syncwarp();

        if (thread < kHyperConnections) {
            const int column = thread;
            const float denominator = __fadd_rn(
                dsv4_sum4(
                    shared_comb[column],
                    shared_comb[kHyperConnections + column],
                    shared_comb[2 * kHyperConnections + column],
                    shared_comb[3 * kHyperConnections + column]),
                eps);
#pragma unroll
            for (int source = 0; source < kHyperConnections; ++source) {
                const int index =
                    source * kHyperConnections + column;
                shared_comb[index] =
                    __fdiv_rn(shared_comb[index], denominator);
            }
        }
        __syncwarp();

#pragma unroll
        for (int iteration = 1;
             iteration < kHyperSinkhornIterations;
             ++iteration) {
            if (thread < kHyperConnections) {
                const int offset = thread * kHyperConnections;
                const float denominator = __fadd_rn(
                    dsv4_row_sum4(
                        shared_comb[offset],
                        shared_comb[offset + 1],
                        shared_comb[offset + 2],
                        shared_comb[offset + 3]),
                    eps);
                shared_comb[offset] =
                    __fdiv_rn(shared_comb[offset], denominator);
                shared_comb[offset + 1] =
                    __fdiv_rn(shared_comb[offset + 1], denominator);
                shared_comb[offset + 2] =
                    __fdiv_rn(shared_comb[offset + 2], denominator);
                shared_comb[offset + 3] =
                    __fdiv_rn(shared_comb[offset + 3], denominator);
            }
            __syncwarp();
            if (thread < kHyperConnections) {
                const int column = thread;
                const float denominator = __fadd_rn(
                    dsv4_sum4(
                        shared_comb[column],
                        shared_comb[kHyperConnections + column],
                        shared_comb[2 * kHyperConnections + column],
                        shared_comb[3 * kHyperConnections + column]),
                    eps);
#pragma unroll
                for (int source = 0;
                     source < kHyperConnections;
                     ++source) {
                    const int index =
                        source * kHyperConnections + column;
                    shared_comb[index] =
                        __fdiv_rn(shared_comb[index], denominator);
                }
            }
            __syncwarp();
        }

        if (thread < kHyperConnections * kHyperConnections) {
            combination[
                static_cast<int64_t>(row) *
                    kHyperConnections * kHyperConnections +
                thread] = shared_comb[thread];
        }
    }
    __syncthreads();

    for (int feature = thread; feature < hidden; feature += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int source = 0; source < kHyperConnections; ++source) {
            value = __fadd_rn(
                value,
                __fmul_rn(
                    shared_pre[source],
                    __half2float(
                        x[(static_cast<int64_t>(row) *
                               kHyperConnections +
                           source) *
                              hidden +
                          feature])));
        }
        reduced[static_cast<int64_t>(row) * hidden + feature] =
            __float2half_rn(value);
    }
}

__global__ void dsv4_hc_post_kernel(
    const half * __restrict__ x,
    const half * __restrict__ residual,
    const float * __restrict__ post,
    const float * __restrict__ combination,
    half * __restrict__ output,
    int rows,
    int hidden)
{
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    const int destination = blockIdx.z;
    if (feature >= hidden || row >= rows) return;

    float residual_sum = 0.0f;
#pragma unroll
    for (int source = 0; source < kHyperConnections; ++source) {
        residual_sum = __fadd_rn(
            residual_sum,
            __fmul_rn(
                combination[
                    (static_cast<int64_t>(row) *
                         kHyperConnections +
                     source) *
                        kHyperConnections +
                    destination],
                __half2float(
                    residual[
                        (static_cast<int64_t>(row) *
                             kHyperConnections +
                         source) *
                            hidden +
                        feature])));
    }
    const float direct = __fmul_rn(
        post[static_cast<int64_t>(row) * kHyperConnections + destination],
        __half2float(x[static_cast<int64_t>(row) * hidden + feature]));
    output[
        (static_cast<int64_t>(row) * kHyperConnections + destination) *
            hidden +
        feature] = __float2half_rn(__fadd_rn(direct, residual_sum));
}

} // namespace

std::vector<torch::Tensor> dsv4_hc_pre_cuda(
    torch::Tensor x,
    torch::Tensor mixes,
    torch::Tensor scale,
    torch::Tensor base,
    int64_t iterations,
    double eps)
{
    TORCH_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == torch::kFloat16 &&
            x.dim() == 4 && x.size(2) == kHyperConnections &&
            x.size(3) == 4096,
        "dsv4_hc_pre: x must be contiguous CUDA f16 [B,T,4,4096]");
    TORCH_CHECK(
        mixes.is_cuda() && mixes.is_contiguous() &&
            mixes.scalar_type() == torch::kFloat32 &&
            mixes.dim() == 3 && mixes.size(0) == x.size(0) &&
            mixes.size(1) == x.size(1) &&
            mixes.size(2) == kHyperMixWidth,
        "dsv4_hc_pre: mixes must be contiguous CUDA f32 [B,T,24]");
    TORCH_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
            scale.scalar_type() == torch::kFloat32 &&
            scale.numel() == 3 &&
            base.is_cuda() && base.is_contiguous() &&
            base.scalar_type() == torch::kFloat32 &&
            base.numel() == kHyperMixWidth &&
            iterations == kHyperSinkhornIterations &&
            std::isfinite(eps) && eps > 0.0,
        "dsv4_hc_pre: invalid scale, base, iterations, or epsilon");
    const int64_t rows64 = x.size(0) * x.size(1);
    TORCH_CHECK(
        rows64 > 0 && rows64 <= INT_MAX,
        "dsv4_hc_pre: row count is out of range");
    const int rows = static_cast<int>(rows64);
    const int hidden = static_cast<int>(x.size(3));
    auto reduced = torch::empty(
        {x.size(0), x.size(1), hidden}, x.options());
    auto post = torch::empty(
        {x.size(0), x.size(1), kHyperConnections},
        mixes.options());
    auto combination = torch::empty(
        {x.size(0), x.size(1),
         kHyperConnections, kHyperConnections},
        mixes.options());
    dsv4_hc_pre_finalize_kernel<<<
        rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
        mixes.data_ptr<float>(), scale.data_ptr<float>(),
        base.data_ptr<float>(),
        reinterpret_cast<half *>(reduced.data_ptr<at::Half>()),
        post.data_ptr<float>(), combination.data_ptr<float>(),
        rows, hidden, static_cast<float>(eps));
    const cudaError_t status = cudaGetLastError();
    TORCH_CHECK(
        status == cudaSuccess,
        "dsv4_hc_pre launch failed: ", cudaGetErrorString(status));
    return {reduced, post, combination};
}

torch::Tensor dsv4_hc_post_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor post,
    torch::Tensor combination)
{
    TORCH_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == torch::kFloat16 &&
            x.dim() == 3 && x.size(2) == 4096,
        "dsv4_hc_post: x must be contiguous CUDA f16 [B,T,4096]");
    TORCH_CHECK(
        residual.is_cuda() && residual.is_contiguous() &&
            residual.scalar_type() == torch::kFloat16 &&
            residual.dim() == 4 &&
            residual.size(0) == x.size(0) &&
            residual.size(1) == x.size(1) &&
            residual.size(2) == kHyperConnections &&
            residual.size(3) == x.size(2),
        "dsv4_hc_post: residual shape mismatch");
    TORCH_CHECK(
        post.is_cuda() && post.is_contiguous() &&
            post.scalar_type() == torch::kFloat32 &&
            post.dim() == 3 &&
            post.size(0) == x.size(0) &&
            post.size(1) == x.size(1) &&
            post.size(2) == kHyperConnections &&
            combination.is_cuda() && combination.is_contiguous() &&
            combination.scalar_type() == torch::kFloat32 &&
            combination.dim() == 4 &&
            combination.size(0) == x.size(0) &&
            combination.size(1) == x.size(1) &&
            combination.size(2) == kHyperConnections &&
            combination.size(3) == kHyperConnections,
        "dsv4_hc_post: post or combination shape mismatch");
    const int64_t rows64 = x.size(0) * x.size(1);
    TORCH_CHECK(
        rows64 > 0 && rows64 <= 65535,
        "dsv4_hc_post: row count is out of range");
    const int rows = static_cast<int>(rows64);
    const int hidden = static_cast<int>(x.size(2));
    auto output = torch::empty(
        {x.size(0), x.size(1), kHyperConnections, hidden},
        residual.options());
    dsv4_hc_post_kernel<<<
        dim3((hidden + 255) / 256, rows, kHyperConnections),
        256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(residual.data_ptr<at::Half>()),
        post.data_ptr<float>(), combination.data_ptr<float>(),
        reinterpret_cast<half *>(output.data_ptr<at::Half>()),
        rows, hidden);
    const cudaError_t status = cudaGetLastError();
    TORCH_CHECK(
        status == cudaSuccess,
        "dsv4_hc_post launch failed: ", cudaGetErrorString(status));
    return output;
}
