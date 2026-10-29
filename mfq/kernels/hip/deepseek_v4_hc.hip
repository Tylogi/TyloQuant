#include "hip/hip_runtime.h"
#include "mfq_hip_compat.h"
#include <hip/hip_fp16.h>
#include "deepseek_v4_hc.h"
#include "mfq/kernels/cuda/deepseek_v41.h"
#include "mfq_tensor_backend.h"
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <cstdint>
#include <vector>

namespace {

constexpr int kHyperConnections = 4;
constexpr int kHyperMixWidth =
    (2 + kHyperConnections) * kHyperConnections;

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

// Keep the original FP32 reduction order while assigning the sixteen
// Sinkhorn entries to lanes instead of serializing four divides per lane.
template<bool Fast>
__device__ __forceinline__ float dsv41_hc_divide(float numerator,float denominator) {
    if constexpr(Fast) {
        float reciprocal;
        asm("v_rcp_f32 %0, %1" : "=v"(reciprocal) : "v"(denominator));
        const float quotient=__fmul_rn(numerator,reciprocal);
        const float remainder=__fmaf_rn(-quotient,denominator,numerator);
        return __fmaf_rn(remainder,reciprocal,quotient);
    }
    return __fdiv_rn(numerator,denominator);
}

template<int Column,bool Fast>
__device__ __forceinline__ float dsv41_hc_row_value(float value,int entry) {
    if constexpr(Fast)
        return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(value),Column*0x55,15,15,false));
    return __shfl(value,(entry&~3)+Column,32);
}

template<int Row,bool Fast>
__device__ __forceinline__ float dsv41_hc_column_value(float value,int entry) {
    if constexpr(Fast)
        return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(value),3|((Row*4)<<5)));
    return __shfl(value,(entry&3)+Row*4,32);
}

template<bool Fast=false>
__device__ __forceinline__ float dsv41_hc_sinkhorn_warp(
        float value, int entry, int iterations, float eps) {
    const float maximum = fmaxf(
        fmaxf(dsv41_hc_row_value<0,Fast>(value,entry),dsv41_hc_row_value<1,Fast>(value,entry)),
        fmaxf(dsv41_hc_row_value<2,Fast>(value,entry),dsv41_hc_row_value<3,Fast>(value,entry)));
    value = expf(value - maximum);
    float denominator = dsv4_softmax_sum4(
        dsv41_hc_row_value<0,Fast>(value,entry),dsv41_hc_row_value<1,Fast>(value,entry),
        dsv41_hc_row_value<2,Fast>(value,entry),dsv41_hc_row_value<3,Fast>(value,entry));
    value = __fadd_rn(dsv41_hc_divide<Fast>(value, denominator), eps);
    denominator = __fadd_rn(dsv4_sum4(
        dsv41_hc_column_value<0,Fast>(value,entry),dsv41_hc_column_value<1,Fast>(value,entry),
        dsv41_hc_column_value<2,Fast>(value,entry),dsv41_hc_column_value<3,Fast>(value,entry)), eps);
    value = dsv41_hc_divide<Fast>(value, denominator);
    for (int iteration = 1; iteration < iterations; ++iteration) {
        denominator = __fadd_rn(dsv4_row_sum4(
            dsv41_hc_row_value<0,Fast>(value,entry),dsv41_hc_row_value<1,Fast>(value,entry),
            dsv41_hc_row_value<2,Fast>(value,entry),dsv41_hc_row_value<3,Fast>(value,entry)), eps);
        value = dsv41_hc_divide<Fast>(value, denominator);
        denominator = __fadd_rn(dsv4_sum4(
            dsv41_hc_column_value<0,Fast>(value,entry),dsv41_hc_column_value<1,Fast>(value,entry),
            dsv41_hc_column_value<2,Fast>(value,entry),dsv41_hc_column_value<3,Fast>(value,entry)), eps);
        value = dsv41_hc_divide<Fast>(value, denominator);
    }
    return value;
}

template<bool OverlapIndependent=false,bool ParallelPartials=false>
__global__ void dsv4_hc_pre_finalize_kernel(
    const half * __restrict__ x,
    const float * __restrict__ mixes,
    const float * __restrict__ scale,
    const float * __restrict__ base,
    half * __restrict__ reduced,
    float * __restrict__ post,
    float * __restrict__ combination,
    float * __restrict__ next_pre,
    int rows,
    int hidden,
    int iterations,
    float eps,
    const float* partials, int parts, const float* previous, const float* norm, float rms_epsilon,
    bool fast_sinkhorn)
{
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    if (row >= rows) return;

    __shared__ float shared_pre[kHyperConnections];
    __shared__ float shared_comb[
        kHyperConnections * kHyperConnections];

    __shared__ float fused_mixes[24];
    __shared__ float fused_inverse[2];
    extern __shared__ float cached_partials[];
    float sum=0;
    if(partials) {
        const auto* row_partials=partials+int64_t(row)*parts*26;
        if constexpr(ParallelPartials) {
            // Eight coalesced groups reduce independent portions of all 26
            // statistics. Only the final eight partial sums pass through LDS.
            __shared__ float parallel_partials[8][32];
            const int component=thread&31,group=thread>>5;
            float value=0.f;
            if(component<26) {
                #pragma unroll 4
                for(int part=group;part<parts;part+=8)
                    value=__fadd_rn(value,row_partials[part*26+component]);
            }
            parallel_partials[group][component]=value;
            __syncthreads();
            if(thread<26) {
                float total=0.f;
                #pragma unroll
                for(int group=0;group<8;++group)
                    total=__fadd_rn(total,parallel_partials[group][thread]);
                sum=total;
            }
        } else {
        if(parts<=320) {
            for(int i=thread;i<parts*26;i+=blockDim.x)cached_partials[i]=row_partials[i];
            __syncthreads();
            if(thread<26) {
                if(parts==320) {
                    // Pipeline independent LDS reads; retain the original serial FP32 sum.
#pragma unroll 32
                    for(int part=0;part<320;++part)
                        sum=__fadd_rn(sum,cached_partials[part*26+thread]);
                } else {
                    for(int part=0;part<parts;++part)
                        sum=__fadd_rn(sum,cached_partials[part*26+thread]);
                }
            }
        }else if(thread<26)for(int part=0;part<parts;++part)
            sum=__fadd_rn(sum,row_partials[part*26+thread]);
        }
        if(thread==24)fused_inverse[0]=rsqrtf(sum/(4.f*hidden)+rms_epsilon);
        if(thread==25)fused_inverse[1]=rsqrtf(sum/hidden+rms_epsilon);
        __syncthreads();
        if(thread<24)fused_mixes[thread]=__fmul_rn(sum,fused_inverse[0]);
        __syncthreads();
    }

    if constexpr(OverlapIndependent) {
        // V4.1 uses previous_pre for the residual branch. Its normalized
        // output does not depend on the new Sinkhorn coefficients. Let three
        // physical waves produce it while wave zero computes those coefficients.
        if(partials && previous && reduced && thread>=64) {
            for(int feature=thread-64;feature<hidden;feature+=blockDim.x-64) {
                float value=0.f;
                #pragma unroll
                for(int source=0;source<4;++source) {
                    float product=__fmul_rn(previous[row*4+source],
                        __half2float(x[(int64_t(row)*4+source)*hidden+feature]));
                    value=__fadd_rn(value,product);
                }
                value=__fmul_rn(__fmul_rn(value,fused_inverse[1]),norm[feature]);
                asm volatile("" : "+v"(value));
                reduced[int64_t(row)*hidden+feature]=__float2half_rn(value);
            }
        }
    }
    const float* row_mixes=partials?fused_mixes:mixes+int64_t(row)*kHyperMixWidth;
    if (partials && thread < 32) {
        if (thread < 4) {
            const float pre_affine = __fadd_rn(__fmul_rn(row_mixes[thread], scale[0]), base[thread]);
            shared_pre[thread] = __fadd_rn(dsv4_sigmoid(pre_affine), eps);
            if (next_pre) next_pre[int64_t(row) * 4 + thread] = shared_pre[thread];
            const float post_affine = __fadd_rn(__fmul_rn(row_mixes[4 + thread], scale[1]), base[4 + thread]);
            post[int64_t(row) * 4 + thread] = __fmul_rn(2.f, dsv4_sigmoid(post_affine));
        }
        const int entry = thread & 15;
        float value = __fadd_rn(__fmul_rn(row_mixes[8 + entry], scale[2]), base[8 + entry]);
        value = fast_sinkhorn?dsv41_hc_sinkhorn_warp<true>(value, entry, iterations, eps):
                              dsv41_hc_sinkhorn_warp<false>(value, entry, iterations, eps);
        if (thread < 16) combination[int64_t(row) * 16 + thread] = value;
    } else if (thread < 32) {
        if (thread < kHyperConnections) {
            const float pre_affine = __fadd_rn(
                __fmul_rn(
                    row_mixes[thread],
                    scale[0]),
                base[thread]);
            shared_pre[thread] = __fadd_rn(
                dsv4_sigmoid(pre_affine), eps);
            if (next_pre)
                next_pre[static_cast<int64_t>(row) * kHyperConnections + thread] = shared_pre[thread];

            const int post_index = kHyperConnections + thread;
            const float post_affine = __fadd_rn(
                __fmul_rn(
                    row_mixes[post_index],
                    scale[1]),
                base[post_index]);
            post[static_cast<int64_t>(row) * kHyperConnections + thread] =
                __fmul_rn(2.0f, dsv4_sigmoid(post_affine));
        }
        if (thread < kHyperConnections * kHyperConnections) {
            const int mix_index = 2 * kHyperConnections + thread;
            shared_comb[thread] = __fadd_rn(
                __fmul_rn(
                    row_mixes[mix_index],
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
             iteration < iterations;
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
    if constexpr(OverlapIndependent) {if(partials && previous && reduced)return;}
    __syncthreads();

    if (!reduced) return;

    for (int feature = thread; feature < hidden; feature += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int source = 0; source < kHyperConnections; ++source) {
            value = __fadd_rn(
                value,
                __fmul_rn(
                    previous ? previous[row*4+source] : shared_pre[source],
                    __half2float(
                        x[(static_cast<int64_t>(row) *
                               kHyperConnections +
                           source) *
                              hidden +
                          feature])));
        }
        if(partials)value=__fmul_rn(__fmul_rn(value,fused_inverse[1]),norm[feature]);
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

std::vector<mfq_tensor_backend::Tensor> dsv4_hc_pre_cuda(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor mixes,
    mfq_tensor_backend::Tensor scale,
    mfq_tensor_backend::Tensor base,
    int64_t iterations,
    double eps)
{
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == mfq_tensor_backend::kFloat16 &&
            x.dim() == 4 && x.size(2) == kHyperConnections &&
            x.size(3) > 0 && x.size(3) <= INT_MAX,
        "dsv4_hc_pre: x must be contiguous CUDA f16 [B,T,4,H]");
    MFQ_RUNTIME_CHECK(
        mixes.is_cuda() && mixes.is_contiguous() &&
            mixes.scalar_type() == mfq_tensor_backend::kFloat32 &&
            mixes.dim() == 3 && mixes.size(0) == x.size(0) &&
            mixes.size(1) == x.size(1) &&
            mixes.size(2) == kHyperMixWidth,
        "dsv4_hc_pre: mixes must be contiguous CUDA f32 [B,T,24]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
            scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
            scale.numel() == 3 &&
            base.is_cuda() && base.is_contiguous() &&
            base.scalar_type() == mfq_tensor_backend::kFloat32 &&
            base.numel() == kHyperMixWidth &&
            iterations > 0 && iterations <= INT_MAX &&
            std::isfinite(eps) && eps > 0.0,
        "dsv4_hc_pre: invalid scale, base, iterations, or epsilon");
    const int64_t rows64 = x.size(0) * x.size(1);
    MFQ_RUNTIME_CHECK(
        rows64 > 0 && rows64 <= INT_MAX,
        "dsv4_hc_pre: row count is out of range");
    const int rows = static_cast<int>(rows64);
    const int hidden = static_cast<int>(x.size(3));
    auto reduced = mfq_tensor_backend::empty(
        {x.size(0), x.size(1), hidden}, x.options());
    auto post = mfq_tensor_backend::empty(
        {x.size(0), x.size(1), kHyperConnections},
        mixes.options());
    auto combination = mfq_tensor_backend::empty(
        {x.size(0), x.size(1),
         kHyperConnections, kHyperConnections},
        mixes.options());
    dsv4_hc_pre_finalize_kernel<false><<<
        rows, 256, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half *>(x.data_ptr<mfq_half>()),
        mixes.data_ptr<float>(), scale.data_ptr<float>(),
        base.data_ptr<float>(),
        reinterpret_cast<half *>(reduced.data_ptr<mfq_half>()),
        post.data_ptr<float>(), combination.data_ptr<float>(),
        nullptr,
        rows, hidden, static_cast<int>(iterations), static_cast<float>(eps),
        nullptr, 0, nullptr, nullptr, 0.f, false);
    const hipError_t status = hipGetLastError();
    MFQ_RUNTIME_CHECK(
        status == hipSuccess,
        "dsv4_hc_pre launch failed: ", hipGetErrorString(status));
    return {reduced, post, combination};
}

std::vector<mfq_tensor_backend::Tensor> dsv4_hc_coefficients_cuda(
    mfq_tensor_backend::Tensor mixes,
    mfq_tensor_backend::Tensor scale,
    mfq_tensor_backend::Tensor base,
    int64_t iterations,
    double eps) {
    MFQ_RUNTIME_CHECK(mixes.is_cuda() && mixes.is_contiguous() &&
                          mixes.scalar_type() == mfq_tensor_backend::kFloat32 &&
                          mixes.dim() == 3 && mixes.size(2) == kHyperMixWidth,
                      "dsv4_hc_coefficients: expected CUDA f32 [B,T,24]");
    MFQ_RUNTIME_CHECK(scale.is_cuda() && scale.is_contiguous() &&
                          scale.scalar_type() == mfq_tensor_backend::kFloat32 && scale.numel() == 3 &&
                          base.is_cuda() && base.is_contiguous() &&
                          base.scalar_type() == mfq_tensor_backend::kFloat32 &&
                          base.numel() == kHyperMixWidth && iterations > 0 && iterations <= INT_MAX &&
                          std::isfinite(eps) && eps > 0,
                      "dsv4_hc_coefficients: invalid coefficient parameters");
    const auto rows = mixes.size(0) * mixes.size(1);
    MFQ_RUNTIME_CHECK(rows > 0 && rows <= INT_MAX,
                      "dsv4_hc_coefficients: row count is out of range");
    auto pre = mfq_tensor_backend::empty({mixes.size(0), mixes.size(1), kHyperConnections}, mixes.options());
    auto post = mfq_tensor_backend::empty_like(pre);
    auto combination = mfq_tensor_backend::empty(
        {mixes.size(0), mixes.size(1), kHyperConnections, kHyperConnections}, mixes.options());
    dsv4_hc_pre_finalize_kernel<false><<<static_cast<int>(rows), 32, 0, mfq_current_cuda_stream()>>>(
        nullptr, mixes.data_ptr<float>(), scale.data_ptr<float>(), base.data_ptr<float>(), nullptr,
        post.data_ptr<float>(), combination.data_ptr<float>(), pre.data_ptr<float>(),
        static_cast<int>(rows), 0, static_cast<int>(iterations), static_cast<float>(eps),
        nullptr, 0, nullptr, nullptr, 0.f, false);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return {pre, post, combination};
}

mfq_tensor_backend::Tensor dsv4_hc_post_cuda(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor residual,
    mfq_tensor_backend::Tensor post,
    mfq_tensor_backend::Tensor combination)
{
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == mfq_tensor_backend::kFloat16 &&
            x.dim() == 3 && x.size(2) > 0 && x.size(2) <= INT_MAX,
        "dsv4_hc_post: x must be contiguous CUDA f16 [B,T,H]");
    MFQ_RUNTIME_CHECK(
        residual.is_cuda() && residual.is_contiguous() &&
            residual.scalar_type() == mfq_tensor_backend::kFloat16 &&
            residual.dim() == 4 &&
            residual.size(0) == x.size(0) &&
            residual.size(1) == x.size(1) &&
            residual.size(2) == kHyperConnections &&
            residual.size(3) == x.size(2),
        "dsv4_hc_post: residual shape mismatch");
    MFQ_RUNTIME_CHECK(
        post.is_cuda() && post.is_contiguous() &&
            post.scalar_type() == mfq_tensor_backend::kFloat32 &&
            post.dim() == 3 &&
            post.size(0) == x.size(0) &&
            post.size(1) == x.size(1) &&
            post.size(2) == kHyperConnections &&
            combination.is_cuda() && combination.is_contiguous() &&
            combination.scalar_type() == mfq_tensor_backend::kFloat32 &&
            combination.dim() == 4 &&
            combination.size(0) == x.size(0) &&
            combination.size(1) == x.size(1) &&
            combination.size(2) == kHyperConnections &&
            combination.size(3) == kHyperConnections,
        "dsv4_hc_post: post or combination shape mismatch");
    const int64_t rows64 = x.size(0) * x.size(1);
    MFQ_RUNTIME_CHECK(
        rows64 > 0 && rows64 <= 65535,
        "dsv4_hc_post: row count is out of range");
    const int rows = static_cast<int>(rows64);
    const int hidden = static_cast<int>(x.size(2));
    auto output = mfq_tensor_backend::empty(
        {x.size(0), x.size(1), kHyperConnections, hidden},
        residual.options());
    dsv4_hc_post_kernel<<<
        dim3((hidden + 255) / 256, rows, kHyperConnections),
        256, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half *>(x.data_ptr<mfq_half>()),
        reinterpret_cast<const half *>(residual.data_ptr<mfq_half>()),
        post.data_ptr<float>(), combination.data_ptr<float>(),
        reinterpret_cast<half *>(output.data_ptr<mfq_half>()),
        rows, hidden);
    const hipError_t status = hipGetLastError();
    MFQ_RUNTIME_CHECK(
        status == hipSuccess,
        "dsv4_hc_post launch failed: ", hipGetErrorString(status));
    return output;
}

std::vector<mfq_tensor_backend::Tensor> deepseek_v41_hc_finalize_cuda(
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor previous,
    mfq_tensor_backend::Tensor norm, mfq_tensor_backend::Tensor partials,
    mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor base,
    int64_t iterations, double rms_epsilon, double hc_epsilon) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(residual.is_cuda() && residual.is_contiguous() && residual.scalar_type()==tb::kFloat16 &&
        residual.dim()==4 && residual.size(2)==4 && residual.size(3)>0 && residual.size(3)<=INT_MAX/4,
        "invalid mHC finalize residual");
    const auto rows=residual.size(0)*residual.size(1),hidden=residual.size(3),parts=(4*hidden+63)/64;
    MFQ_RUNTIME_CHECK(rows>0 && rows<=INT_MAX && partials.is_contiguous() && partials.is_cuda() &&
        partials.scalar_type()==tb::kFloat32 && partials.dim()==3 && partials.size(0)==rows &&
        partials.size(1)==parts && partials.size(2)==26 && previous.is_cuda() && previous.is_contiguous() &&
        previous.scalar_type()==tb::kFloat32 && previous.dim()==3 && previous.size(0)==residual.size(0) &&
        previous.size(1)==residual.size(1) && previous.size(2)==4 && norm.is_cuda() && norm.is_contiguous() &&
        norm.scalar_type()==tb::kFloat32 && norm.numel()==hidden && scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type()==tb::kFloat32 && scale.numel()==3 && base.is_cuda() && base.is_contiguous() &&
        base.scalar_type()==tb::kFloat32 && base.numel()==24 && iterations>0 && iterations<=INT_MAX &&
        std::isfinite(rms_epsilon) && rms_epsilon>0 && std::isfinite(hc_epsilon) && hc_epsilon>0 &&
        previous.device()==residual.device() && norm.device()==residual.device() && partials.device()==residual.device() &&
        scale.device()==residual.device() && base.device()==residual.device(),"invalid mHC finalize inputs");
    auto branch=tb::empty({residual.size(0),residual.size(1),hidden},residual.options());
    auto pre=tb::empty({residual.size(0),residual.size(1),4},previous.options()),post=tb::empty_like(pre);
    auto combination=tb::empty({residual.size(0),residual.size(1),4,4},previous.options());
    const size_t shared_bytes=parts<=320?size_t(parts)*26*sizeof(float):0;
    static const bool fast_sinkhorn=[] {
        const char* value=std::getenv("MFQ_HIP_HC_FAST_SINKHORN");
        return !value || std::atoi(value)!=0;
    }();
    static const bool overlap=[] {
        const char* value=std::getenv("MFQ_HIP_HC_OVERLAP");return !value || std::atoi(value)!=0;
    }();
    static const bool parallel_partials=[] {
        const char* value=std::getenv("MFQ_HIP_HC_PARTIALS_PARALLEL");return !value || std::atoi(value)!=0;
    }();
    if(overlap && parallel_partials && rows==1 && parts==320) {
    dsv4_hc_pre_finalize_kernel<true,true><<<int(rows),256,0,mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(residual.data_ptr<mfq_half>()),nullptr,scale.data_ptr<float>(),base.data_ptr<float>(),
        reinterpret_cast<half*>(branch.data_ptr<mfq_half>()),post.data_ptr<float>(),combination.data_ptr<float>(),
        pre.data_ptr<float>(),int(rows),int(hidden),int(iterations),float(hc_epsilon),partials.data_ptr<float>(),int(parts),
        previous.data_ptr<float>(),norm.data_ptr<float>(),float(rms_epsilon),fast_sinkhorn);
        static const bool log=std::getenv("MFQ_HIP_HC_PARTIALS_LOG")!=nullptr;
        static std::atomic_flag logged=ATOMIC_FLAG_INIT;
        if(log && !logged.test_and_set(std::memory_order_relaxed))
            std::fprintf(stderr,"native_hc_parallel_partials parts=320 groups=8 fp32=1\n");
    } else if(overlap && rows==1) {
    dsv4_hc_pre_finalize_kernel<true><<<int(rows),256,shared_bytes,mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(residual.data_ptr<mfq_half>()),nullptr,scale.data_ptr<float>(),base.data_ptr<float>(),
        reinterpret_cast<half*>(branch.data_ptr<mfq_half>()),post.data_ptr<float>(),combination.data_ptr<float>(),
        pre.data_ptr<float>(),int(rows),int(hidden),int(iterations),float(hc_epsilon),partials.data_ptr<float>(),int(parts),
        previous.data_ptr<float>(),norm.data_ptr<float>(),float(rms_epsilon),fast_sinkhorn);
    } else {
    dsv4_hc_pre_finalize_kernel<false><<<int(rows),256,shared_bytes,mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(residual.data_ptr<mfq_half>()),nullptr,scale.data_ptr<float>(),base.data_ptr<float>(),
        reinterpret_cast<half*>(branch.data_ptr<mfq_half>()),post.data_ptr<float>(),combination.data_ptr<float>(),
        pre.data_ptr<float>(),int(rows),int(hidden),int(iterations),float(hc_epsilon),partials.data_ptr<float>(),int(parts),
        previous.data_ptr<float>(),norm.data_ptr<float>(),float(rms_epsilon),fast_sinkhorn);
    }

    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return {branch,pre,post,combination};
}
