// RMSNorm / L2 Norm (ggml norm.cu: ggml_rms_norm / ggml_l2_norm).
// One block normalizes one row (last dim D). fp32. x viewed as [N, D] (caller flattens).

#include <cuda_runtime.h>
#include "../../../cpp_runtime/cuda/mfq_tensor_backend.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <vector>

#include "reduce.cuh"

constexpr int NORM_BD = 256;

__global__ void qwen_rms_norm_bf16_square_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ squared,
    int64_t count)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (int64_t)blockDim.x * gridDim.x) {
        const float value = __bfloat162float(input[i]);
        squared[i] = value * value;
    }
}

__global__ void qwen_rms_norm_bf16_finalize_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int64_t count,
    int D,
    float eps,
    float weight_offset)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (int64_t)blockDim.x * gridDim.x) {
        const float inverse = rsqrtf(mean[i / D] + eps);
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(input[i]) * inverse);
        __nv_bfloat16 scale = __float2bfloat16_rn(weight[i % D]);
        if (weight_offset != 0.0f) {
            scale = __float2bfloat16_rn(
                __bfloat162float(scale) + weight_offset);
        }
        output[i] = __float2bfloat16_rn(
            __bfloat162float(scale) * __bfloat162float(normalized));
    }
}

__global__ void qwen_rms_norm_pair_bf16_square_kernel(
    const __nv_bfloat16* __restrict__ first,
    const __nv_bfloat16* __restrict__ second,
    float* __restrict__ squared,
    int64_t first_count,
    int64_t total_count)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total_count; i += (int64_t)blockDim.x * gridDim.x) {
        const float value = i < first_count
            ? __bfloat162float(first[i])
            : __bfloat162float(second[i - first_count]);
        squared[i] = value * value;
    }
}

__global__ void qwen_rms_norm_pair_bf16_finalize_kernel(
    const __nv_bfloat16* __restrict__ first,
    const __nv_bfloat16* __restrict__ second,
    const float* __restrict__ mean,
    const float* __restrict__ first_weight,
    const float* __restrict__ second_weight,
    __nv_bfloat16* __restrict__ first_output,
    __nv_bfloat16* __restrict__ second_output,
    int64_t first_count,
    int64_t total_count,
    int D,
    float eps,
    float weight_offset)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total_count; i += (int64_t)blockDim.x * gridDim.x) {
        const bool use_second = i >= first_count;
        const int64_t local_i = use_second ? i - first_count : i;
        const __nv_bfloat16 value = use_second ? second[local_i] : first[local_i];
        const float* weight = use_second ? second_weight : first_weight;
        __nv_bfloat16* output = use_second ? second_output : first_output;
        const float inverse = rsqrtf(mean[i / D] + eps);
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(value) * inverse);
        __nv_bfloat16 scale = __float2bfloat16_rn(weight[local_i % D]);
        if (weight_offset != 0.0f) {
            scale = __float2bfloat16_rn(
                __bfloat162float(scale) + weight_offset);
        }
        output[local_i] = __float2bfloat16_rn(
            __bfloat162float(scale) * __bfloat162float(normalized));
    }
}

__device__ __forceinline__ __nv_bfloat16 qwen_rms_norm_bf16_value(
    __nv_bfloat16 value, float weight, float inverse, float weight_offset)
{
    const __nv_bfloat16 normalized = __float2bfloat16_rn(
        __bfloat162float(value) * inverse);
    __nv_bfloat16 scale = __float2bfloat16_rn(weight);
    if (weight_offset != 0.0f) {
        scale = __float2bfloat16_rn(
            __bfloat162float(scale) + weight_offset);
    }
    return __float2bfloat16_rn(
        __bfloat162float(scale) * __bfloat162float(normalized));
}

__global__ void minicpm_qk_norm_rope_cache_write_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ mean,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const int64_t* __restrict__ rope_pos,
    const int64_t* __restrict__ write_pos,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int B, int Hq, int Hk, int D, int half,
    int table_len, int max_seq, float eps, float weight_offset)
{
    const int q_row = blockIdx.x;
    if (q_row >= B * Hq) return;
    const int hq = q_row % Hq;
    const int b = q_row / Hq;
    const int tid = threadIdx.x;
    int64_t p = rope_pos[0];
    p = p < 0 ? 0 : (p >= table_len ? table_len - 1 : p);
    const int64_t wp = write_pos[0];
    const float q_inverse = rsqrtf(mean[q_row] + eps);
    const size_t q_offset = (size_t)q_row * D;

    if (tid < half) {
        const __nv_bfloat16 cs = __float2bfloat16_rn(
            cos[(size_t)p * half + tid]);
        const __nv_bfloat16 sn = __float2bfloat16_rn(
            sin[(size_t)p * half + tid]);
        const __nv_bfloat16 x0 = qwen_rms_norm_bf16_value(
            q[q_offset + tid], q_weight[tid], q_inverse, weight_offset);
        const __nv_bfloat16 x1 = qwen_rms_norm_bf16_value(
            q[q_offset + tid + half], q_weight[tid + half],
            q_inverse, weight_offset);
        const __nv_bfloat16 x0c = __float2bfloat16_rn(
            __bfloat162float(x0) * __bfloat162float(cs));
        const __nv_bfloat16 x1s = __float2bfloat16_rn(
            __bfloat162float(x1) * __bfloat162float(sn));
        const __nv_bfloat16 x1c = __float2bfloat16_rn(
            __bfloat162float(x1) * __bfloat162float(cs));
        const __nv_bfloat16 x0s = __float2bfloat16_rn(
            __bfloat162float(x0) * __bfloat162float(sn));
        q_out[q_offset + tid] = __float2bfloat16_rn(
            __bfloat162float(x0c) - __bfloat162float(x1s));
        q_out[q_offset + tid + half] = __float2bfloat16_rn(
            __bfloat162float(x1c) + __bfloat162float(x0s));
    }

    if (hq < Hk && wp >= 0 && wp < max_seq) {
        const int k_row = b * Hk + hq;
        const float k_inverse = rsqrtf(mean[B * Hq + k_row] + eps);
        const size_t src = (size_t)k_row * D;
        const size_t dst = ((size_t)k_row * max_seq + wp) * D;
        if (tid < half) {
            const __nv_bfloat16 cs = __float2bfloat16_rn(
                cos[(size_t)p * half + tid]);
            const __nv_bfloat16 sn = __float2bfloat16_rn(
                sin[(size_t)p * half + tid]);
            const __nv_bfloat16 x0 = qwen_rms_norm_bf16_value(
                k[src + tid], k_weight[tid], k_inverse, weight_offset);
            const __nv_bfloat16 x1 = qwen_rms_norm_bf16_value(
                k[src + tid + half], k_weight[tid + half],
                k_inverse, weight_offset);
            const __nv_bfloat16 x0c = __float2bfloat16_rn(
                __bfloat162float(x0) * __bfloat162float(cs));
            const __nv_bfloat16 x1s = __float2bfloat16_rn(
                __bfloat162float(x1) * __bfloat162float(sn));
            const __nv_bfloat16 x1c = __float2bfloat16_rn(
                __bfloat162float(x1) * __bfloat162float(cs));
            const __nv_bfloat16 x0s = __float2bfloat16_rn(
                __bfloat162float(x0) * __bfloat162float(sn));
            k_cache[dst + tid] = __float2bfloat16_rn(
                __bfloat162float(x0c) - __bfloat162float(x1s));
            k_cache[dst + tid + half] = __float2bfloat16_rn(
                __bfloat162float(x1c) + __bfloat162float(x0s));
        }
        if (tid < D) {
            v_cache[dst + tid] = v[src + tid];
        }
    }
}

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
__global__ void rms_norm_f16_kernel(const mfq_half* __restrict__ x,
                                    const float* __restrict__ w,
                                    mfq_half* __restrict__ out,
                                    int N, int D, float eps, float weight_offset)
{
    int row = blockIdx.x;
    if (row >= N) {
        return;
    }
    const mfq_half* xr = x + (size_t)row * D;
    mfq_half* or_ = out + (size_t)row * D;
    int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BD) {
        float xi = (float)xr[i];
        ssq += xi * xi;
    }
    ssq = block_sum<NORM_BD / 32>(ssq);

    float rinv = rsqrtf(ssq / (float)D + eps);
    for (int i = tid; i < D; i += BD) {
        or_[i] = (mfq_half)((float)xr[i] * rinv * (w[i] + weight_offset));
    }
}

template <int BD>
__global__ void rms_norm_pair_f16_f32_kernel(
    const mfq_half* __restrict__ first,
    const mfq_half* __restrict__ second,
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
    const mfq_half* input = use_second ? second : first;
    const float* weight = use_second ? second_weight : first_weight;
    float* output = use_second ? second_out : first_out;
    const mfq_half* input_row = input + (size_t)row * D;
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

mfq_tensor_backend::Tensor rms_norm_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, double eps)
{
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kFloat32,
                "rms_norm: x must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.scalar_type() == mfq_tensor_backend::kFloat32, "weight must be cuda f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    auto out = mfq_tensor_backend::empty_like(x);
    rms_norm_kernel<NORM_BD><<<N, NORM_BD, 0, mfq_current_cuda_stream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), N, D, (float)eps, 0.0f);
    return out;
}

mfq_tensor_backend::Tensor rms_norm_offset_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, double eps, double weight_offset)
{
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kFloat32,
                "rms_norm_offset: x must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.scalar_type() == mfq_tensor_backend::kFloat32, "weight must be cuda f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    auto out = mfq_tensor_backend::empty_like(x);
    rms_norm_kernel<NORM_BD><<<N, NORM_BD, 0, mfq_current_cuda_stream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), N, D,
        (float)eps, (float)weight_offset);
    return out;
}

mfq_tensor_backend::Tensor rms_norm_f16_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, double eps,
                                double weight_offset)
{
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kFloat16,
                "rms_norm_f16: x must be cuda contiguous f16");
    MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                    weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "rms_norm_f16: weight must be cuda contiguous f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    MFQ_RUNTIME_CHECK(weight.numel() == D, "rms_norm_f16: weight length mismatch");
    auto out = mfq_tensor_backend::empty_like(x);
    rms_norm_f16_kernel<NORM_BD><<<N, NORM_BD, 0, mfq_current_cuda_stream()>>>(
        x.data_ptr<mfq_half>(), weight.data_ptr<float>(), out.data_ptr<mfq_half>(),
        N, D, (float)eps, (float)weight_offset);
    return out;
}

mfq_tensor_backend::Tensor qwen_rms_norm_bf16_cuda(
    mfq_tensor_backend::Tensor input, mfq_tensor_backend::Tensor weight, double eps,
    double weight_offset)
{
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() &&
                    input.scalar_type() == mfq_tensor_backend::kBFloat16,
                "qwen_rms_norm_bf16: input must be cuda contiguous bf16");
    MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                    weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "qwen_rms_norm_bf16: weight must be cuda contiguous f32");
    const int D = (int)input.size(-1);
    const int64_t count = input.numel();
    MFQ_RUNTIME_CHECK(D > 0 && weight.numel() == D,
                "qwen_rms_norm_bf16: weight length mismatch");
    auto squared = mfq_tensor_backend::empty(input.sizes(), input.options().dtype(mfq_tensor_backend::kFloat32));
    const int blocks = (int)std::min<int64_t>((count + NORM_BD - 1) / NORM_BD, 65535);
    auto stream = mfq_current_cuda_stream();
    qwen_rms_norm_bf16_square_kernel<<<blocks, NORM_BD, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<mfq_bfloat16>()),
        squared.data_ptr<float>(), count);
    auto mean = squared.mean(-1, true);
    auto output = mfq_tensor_backend::empty_like(input);
    qwen_rms_norm_bf16_finalize_kernel<<<blocks, NORM_BD, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<mfq_bfloat16>()),
        mean.data_ptr<float>(), weight.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<mfq_bfloat16>()),
        count, D, (float)eps, (float)weight_offset);
    return output;
}

std::vector<mfq_tensor_backend::Tensor> qwen_rms_norm_pair_bf16_cuda(
    mfq_tensor_backend::Tensor first, mfq_tensor_backend::Tensor second,
    mfq_tensor_backend::Tensor first_weight, mfq_tensor_backend::Tensor second_weight,
    double eps, double weight_offset)
{
    MFQ_RUNTIME_CHECK(first.is_cuda() && first.is_contiguous() &&
                    first.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                    second.is_cuda() && second.is_contiguous() &&
                    second.scalar_type() == mfq_tensor_backend::kBFloat16,
                "paired Qwen RMSNorm inputs must be cuda contiguous bf16");
    MFQ_RUNTIME_CHECK(first_weight.is_cuda() && first_weight.is_contiguous() &&
                    first_weight.scalar_type() == mfq_tensor_backend::kFloat32 &&
                    second_weight.is_cuda() && second_weight.is_contiguous() &&
                    second_weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "paired Qwen RMSNorm weights must be cuda contiguous f32");
    const int D = (int)first.size(-1);
    MFQ_RUNTIME_CHECK(D > 0 && second.size(-1) == D &&
                    first_weight.numel() == D && second_weight.numel() == D,
                "paired Qwen RMSNorm widths must match");
    const int64_t first_count = first.numel();
    const int64_t total_count = first_count + second.numel();
    const int64_t total_rows = total_count / D;
    auto squared = mfq_tensor_backend::empty(
        {total_rows, D}, first.options().dtype(mfq_tensor_backend::kFloat32));
    const int blocks = (int)std::min<int64_t>(
        (total_count + NORM_BD - 1) / NORM_BD, 65535);
    auto stream = mfq_current_cuda_stream();
    qwen_rms_norm_pair_bf16_square_kernel<<<blocks, NORM_BD, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(first.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(second.data_ptr<mfq_bfloat16>()),
        squared.data_ptr<float>(), first_count, total_count);
    auto mean = squared.mean(-1, true);
    auto first_output = mfq_tensor_backend::empty_like(first);
    auto second_output = mfq_tensor_backend::empty_like(second);
    qwen_rms_norm_pair_bf16_finalize_kernel<<<blocks, NORM_BD, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(first.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(second.data_ptr<mfq_bfloat16>()),
        mean.data_ptr<float>(), first_weight.data_ptr<float>(),
        second_weight.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(
            first_output.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            second_output.data_ptr<mfq_bfloat16>()),
        first_count, total_count, D, (float)eps, (float)weight_offset);
    return {first_output, second_output};
}

mfq_tensor_backend::Tensor minicpm_qk_norm_rope_cache_write_bf16_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor q_weight, mfq_tensor_backend::Tensor k_weight,
    mfq_tensor_backend::Tensor rope_pos, mfq_tensor_backend::Tensor write_pos,
    mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin,
    mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    double eps, double weight_offset)
{
    const auto bf16 = mfq_tensor_backend::kBFloat16;
    MFQ_RUNTIME_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() &&
                    q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                    q.scalar_type() == bf16 && k.scalar_type() == bf16 &&
                    v.scalar_type() == bf16,
                "minicpm_qk_norm_rope_kv: q/k/v must be cuda contiguous bf16");
    MFQ_RUNTIME_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 &&
                    q.size(0) == k.size(0) && q.size(1) == 32 &&
                    k.size(1) == 8 && q.size(2) == 1 && k.size(2) == 1 &&
                    q.size(3) == 128 && k.size(3) == 128 && k.sizes() == v.sizes(),
                "minicpm_qk_norm_rope_kv: expected Bx32x1x128 Q and Bx8x1x128 K/V");
    MFQ_RUNTIME_CHECK(q_weight.is_cuda() && k_weight.is_cuda() &&
                    q_weight.is_contiguous() && k_weight.is_contiguous() &&
                    q_weight.scalar_type() == mfq_tensor_backend::kFloat32 &&
                    k_weight.scalar_type() == mfq_tensor_backend::kFloat32 &&
                    q_weight.numel() == 128 && k_weight.numel() == 128,
                "minicpm_qk_norm_rope_kv: norm weights must be cuda contiguous f32[128]");
    MFQ_RUNTIME_CHECK(k_cache.is_cuda() && v_cache.is_cuda() &&
                    k_cache.is_contiguous() && v_cache.is_contiguous() &&
                    k_cache.scalar_type() == bf16 && v_cache.scalar_type() == bf16 &&
                    k_cache.sizes() == v_cache.sizes() && k_cache.dim() == 4 &&
                    k_cache.size(0) == k.size(0) && k_cache.size(1) == 8 &&
                    k_cache.size(3) == 128,
                "minicpm_qk_norm_rope_kv: cache shape mismatch");
    MFQ_RUNTIME_CHECK(rope_pos.is_cuda() && write_pos.is_cuda() &&
                    rope_pos.is_contiguous() && write_pos.is_contiguous() &&
                    rope_pos.scalar_type() == mfq_tensor_backend::kInt64 &&
                    write_pos.scalar_type() == mfq_tensor_backend::kInt64 &&
                    rope_pos.numel() == 1 && write_pos.numel() == 1,
                "minicpm_qk_norm_rope_kv: positions must be cuda contiguous int64[1]");
    MFQ_RUNTIME_CHECK(cos.is_cuda() && sin.is_cuda() && cos.is_contiguous() &&
                    sin.is_contiguous() && cos.scalar_type() == mfq_tensor_backend::kFloat32 &&
                    sin.scalar_type() == mfq_tensor_backend::kFloat32 && cos.sizes() == sin.sizes() &&
                    cos.dim() == 2 && cos.size(1) == 64,
                "minicpm_qk_norm_rope_kv: cos/sin must be matching f32 tables");

    constexpr int D = 128;
    constexpr int block = 256;
    const int64_t first_count = q.numel();
    const int64_t total_count = first_count + k.numel();
    const int64_t total_rows = total_count / D;
    auto squared = mfq_tensor_backend::empty(
        {total_rows, D}, q.options().dtype(mfq_tensor_backend::kFloat32));
    const int blocks = (int)std::min<int64_t>(
        (total_count + block - 1) / block, 65535);
    auto stream = mfq_current_cuda_stream();
    qwen_rms_norm_pair_bf16_square_kernel<<<blocks, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<mfq_bfloat16>()),
        squared.data_ptr<float>(), first_count, total_count);
    auto mean = squared.mean(-1, true);
    auto q_out = mfq_tensor_backend::empty_like(q);
    const int B = (int)q.size(0);
    minicpm_qk_norm_rope_cache_write_bf16_kernel<<<
        B * 32, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<mfq_bfloat16>()),
        mean.data_ptr<float>(), q_weight.data_ptr<float>(),
        k_weight.data_ptr<float>(), rope_pos.data_ptr<int64_t>(),
        write_pos.data_ptr<int64_t>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr<mfq_bfloat16>()),
        B, 32, 8, D, 64, (int)cos.size(0), (int)k_cache.size(2),
        (float)eps, (float)weight_offset);
    return q_out;
}

std::vector<mfq_tensor_backend::Tensor> rms_norm_pair_f16_f32_offset_cuda(
    mfq_tensor_backend::Tensor first,
    mfq_tensor_backend::Tensor second,
    mfq_tensor_backend::Tensor first_weight,
    mfq_tensor_backend::Tensor second_weight,
    double eps,
    double weight_offset)
{
    MFQ_RUNTIME_CHECK(
        first.is_cuda() && first.is_contiguous() &&
        first.scalar_type() == mfq_tensor_backend::kFloat16,
        "paired RMSNorm first input must be CUDA contiguous fp16");
    MFQ_RUNTIME_CHECK(
        second.is_cuda() && second.is_contiguous() &&
        second.scalar_type() == mfq_tensor_backend::kFloat16,
        "paired RMSNorm second input must be CUDA contiguous fp16");
    MFQ_RUNTIME_CHECK(
        first_weight.is_cuda() && first_weight.is_contiguous() &&
        first_weight.scalar_type() == mfq_tensor_backend::kFloat32,
        "paired RMSNorm first weight must be CUDA contiguous fp32");
    MFQ_RUNTIME_CHECK(
        second_weight.is_cuda() && second_weight.is_contiguous() &&
        second_weight.scalar_type() == mfq_tensor_backend::kFloat32,
        "paired RMSNorm second weight must be CUDA contiguous fp32");
    const int D = (int)first.size(-1);
    MFQ_RUNTIME_CHECK(second.size(-1) == D, "paired RMSNorm widths must match");
    MFQ_RUNTIME_CHECK(
        first_weight.numel() == D && second_weight.numel() == D,
        "paired RMSNorm weight lengths must match the input width");
    const int first_rows = (int)(first.numel() / D);
    const int second_rows = (int)(second.numel() / D);
    auto output_options = first.options().dtype(mfq_tensor_backend::kFloat32);
    auto first_out = mfq_tensor_backend::empty(first.sizes(), output_options);
    auto second_out = mfq_tensor_backend::empty(second.sizes(), output_options);
    rms_norm_pair_f16_f32_kernel<NORM_BD><<<
        first_rows + second_rows, NORM_BD, 0,
        mfq_current_cuda_stream()>>>(
        first.data_ptr<mfq_half>(), second.data_ptr<mfq_half>(),
        first_weight.data_ptr<float>(), second_weight.data_ptr<float>(),
        first_out.data_ptr<float>(), second_out.data_ptr<float>(),
        first_rows, second_rows, D, (float)eps, (float)weight_offset);
    return {first_out, second_out};
}

mfq_tensor_backend::Tensor l2_norm_cuda(mfq_tensor_backend::Tensor x, double eps)
{
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kFloat32,
                "l2_norm: x must be cuda contiguous f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    auto out = mfq_tensor_backend::empty_like(x);
    l2_norm_kernel<NORM_BD><<<N, NORM_BD, 0, mfq_current_cuda_stream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N, D, (float)eps);
    return out;
}
