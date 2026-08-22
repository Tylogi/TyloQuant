// Residual add a + b (ggml acc.cu: ggml_acc). fp16/fp32, any shape flattened.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../../../cpp_runtime/cuda/mfq_tensor_backend.h"
#include <vector>

#include "reduce.cuh"

template <typename scalar_t>
__global__ void acc_kernel(const scalar_t* __restrict__ a, const scalar_t* __restrict__ b,
                           scalar_t* __restrict__ out, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = a[i] + b[i];
    }
}

mfq_tensor_backend::Tensor acc_cuda(mfq_tensor_backend::Tensor a, mfq_tensor_backend::Tensor b)
{
    MFQ_RUNTIME_CHECK(a.is_cuda() && a.is_contiguous(), "acc: a must be cuda contiguous");
    MFQ_RUNTIME_CHECK(b.is_cuda() && b.is_contiguous(), "acc: b must be cuda contiguous");
    MFQ_RUNTIME_CHECK(a.scalar_type() == b.scalar_type(), "acc: a/b dtype mismatch");
    MFQ_RUNTIME_CHECK(a.scalar_type() == mfq_tensor_backend::kFloat16 || a.scalar_type() == mfq_tensor_backend::kFloat32,
                "acc: dtype must be f16 or f32");
    MFQ_RUNTIME_CHECK(a.sizes() == b.sizes(), "acc: a/b shape mismatch");
    int n = (int)a.numel();
    auto out = mfq_tensor_backend::empty_like(a);
    constexpr int BD = 256;
    MFQ_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "acc_cuda", [&] {
        acc_kernel<scalar_t><<<(n + BD - 1) / BD, BD, 0, mfq_current_cuda_stream()>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
    return out;
}

constexpr int ACC_RMS_BD = 256;

template <int BD>
__global__ void acc_rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ weight,
    __nv_bfloat16* __restrict__ sum_out,
    __nv_bfloat16* __restrict__ norm_out,
    int N, int D, float eps, float weight_offset)
{
    const int row = blockIdx.x;
    if (row >= N) {
        return;
    }
    const size_t row_offset = (size_t)row * D;
    float square_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const __nv_bfloat16 stored = __float2bfloat16_rn(
            __bfloat162float(a[row_offset + i]) +
            __bfloat162float(b[row_offset + i]));
        sum_out[row_offset + i] = stored;
        const float value = __bfloat162float(stored);
        square_sum += value * value;
    }
    square_sum = block_sum<BD / 32>(square_sum);
    const float inverse = rsqrtf(square_sum / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += BD) {
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(sum_out[row_offset + i]) * inverse);
        __nv_bfloat16 scale = __float2bfloat16_rn(weight[i]);
        if (weight_offset != 0.0f) {
            scale = __float2bfloat16_rn(
                __bfloat162float(scale) + weight_offset);
        }
        norm_out[row_offset + i] = __float2bfloat16_rn(
            __bfloat162float(normalized) * __bfloat162float(scale));
    }
}

template <typename scalar_t, typename norm_t, int BD>
__global__ void acc_rms_norm_kernel(const scalar_t* __restrict__ a,
                                    const scalar_t* __restrict__ b,
                                    const float* __restrict__ w,
                                    scalar_t* __restrict__ sum_out,
                                    norm_t* __restrict__ norm_out,
                                    int N, int D, float eps, float weight_offset)
{
    int row = blockIdx.x;
    if (row >= N) {
        return;
    }

    const scalar_t* ar = a + (size_t)row * D;
    const scalar_t* br = b + (size_t)row * D;
    scalar_t* sr = sum_out + (size_t)row * D;
    norm_t* nr = norm_out + (size_t)row * D;
    int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BD) {
        float sf = (float)ar[i] + (float)br[i];
        scalar_t stored = (scalar_t)sf;
        sr[i] = stored;
        float sn = (float)stored;
        ssq += sn * sn;
    }
    ssq = block_sum<BD / 32>(ssq);

    float rinv = rsqrtf(ssq / (float)D + eps);
    for (int i = tid; i < D; i += BD) {
        float sf = (float)ar[i] + (float)br[i];
        scalar_t stored = (scalar_t)sf;
        float sn = (float)stored;
        nr[i] = (norm_t)(sn * rinv * (w[i] + weight_offset));
    }
}

std::vector<mfq_tensor_backend::Tensor> acc_rms_norm_cuda(mfq_tensor_backend::Tensor a, mfq_tensor_backend::Tensor b,
                                             mfq_tensor_backend::Tensor weight, double eps,
                                             double weight_offset)
{
    MFQ_RUNTIME_CHECK(a.is_cuda() && a.is_contiguous(), "acc_rms_norm: a must be cuda contiguous");
    MFQ_RUNTIME_CHECK(b.is_cuda() && b.is_contiguous(), "acc_rms_norm: b must be cuda contiguous");
    MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "acc_rms_norm: weight must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(a.scalar_type() == b.scalar_type(), "acc_rms_norm: a/b dtype mismatch");
    MFQ_RUNTIME_CHECK(a.scalar_type() == mfq_tensor_backend::kFloat16 || a.scalar_type() == mfq_tensor_backend::kFloat32,
                "acc_rms_norm: dtype must be f16 or f32");
    MFQ_RUNTIME_CHECK(a.sizes() == b.sizes(), "acc_rms_norm: a/b shape mismatch");
    MFQ_RUNTIME_CHECK(a.dim() >= 1, "acc_rms_norm: a must have at least one dim");
    int D = (int)a.size(-1);
    int N = (int)(a.numel() / D);
    MFQ_RUNTIME_CHECK(weight.numel() == D, "acc_rms_norm: weight length mismatch");

    auto sum = mfq_tensor_backend::empty_like(a);
    auto norm = mfq_tensor_backend::empty(a.sizes(), a.options().dtype(mfq_tensor_backend::kFloat32));
    MFQ_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "acc_rms_norm_cuda", [&] {
        acc_rms_norm_kernel<scalar_t, float, ACC_RMS_BD><<<N, ACC_RMS_BD, 0, mfq_current_cuda_stream()>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), weight.data_ptr<float>(),
            sum.data_ptr<scalar_t>(), norm.data_ptr<float>(), N, D,
            (float)eps, (float)weight_offset);
    });
    return {sum, norm};
}

std::vector<mfq_tensor_backend::Tensor> acc_rms_norm_f16_cuda(mfq_tensor_backend::Tensor a, mfq_tensor_backend::Tensor b,
                                                 mfq_tensor_backend::Tensor weight, double eps,
                                                 double weight_offset)
{
    MFQ_RUNTIME_CHECK(a.is_cuda() && a.is_contiguous(), "acc_rms_norm_f16: a must be cuda contiguous");
    MFQ_RUNTIME_CHECK(b.is_cuda() && b.is_contiguous(), "acc_rms_norm_f16: b must be cuda contiguous");
    MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "acc_rms_norm_f16: weight must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(a.scalar_type() == mfq_tensor_backend::kFloat16 && b.scalar_type() == mfq_tensor_backend::kFloat16,
                "acc_rms_norm_f16: a/b must be f16");
    MFQ_RUNTIME_CHECK(a.sizes() == b.sizes(), "acc_rms_norm_f16: a/b shape mismatch");
    MFQ_RUNTIME_CHECK(a.dim() >= 1, "acc_rms_norm_f16: a must have at least one dim");
    int D = (int)a.size(-1);
    int N = (int)(a.numel() / D);
    MFQ_RUNTIME_CHECK(weight.numel() == D, "acc_rms_norm_f16: weight length mismatch");

    auto sum = mfq_tensor_backend::empty_like(a);
    auto norm = mfq_tensor_backend::empty_like(a);
    acc_rms_norm_kernel<mfq_half, mfq_half, ACC_RMS_BD><<<N, ACC_RMS_BD, 0, mfq_current_cuda_stream()>>>(
        a.data_ptr<mfq_half>(), b.data_ptr<mfq_half>(), weight.data_ptr<float>(),
        sum.data_ptr<mfq_half>(), norm.data_ptr<mfq_half>(), N, D,
        (float)eps, (float)weight_offset);
    return {sum, norm};
}

std::vector<mfq_tensor_backend::Tensor> acc_rms_norm_bf16_cuda(
    mfq_tensor_backend::Tensor a, mfq_tensor_backend::Tensor b,
    mfq_tensor_backend::Tensor weight, double eps, double weight_offset)
{
    MFQ_RUNTIME_CHECK(a.is_cuda() && a.is_contiguous() &&
                    a.scalar_type() == mfq_tensor_backend::kBFloat16,
                "acc_rms_norm_bf16: a must be cuda contiguous bf16");
    MFQ_RUNTIME_CHECK(b.is_cuda() && b.is_contiguous() &&
                    b.scalar_type() == mfq_tensor_backend::kBFloat16,
                "acc_rms_norm_bf16: b must be cuda contiguous bf16");
    MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                    weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "acc_rms_norm_bf16: weight must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(a.sizes() == b.sizes() && a.dim() >= 1,
                "acc_rms_norm_bf16: a/b shape mismatch");
    const int D = (int)a.size(-1);
    const int N = (int)(a.numel() / D);
    MFQ_RUNTIME_CHECK(D > 0 && weight.numel() == D,
                "acc_rms_norm_bf16: weight length mismatch");
    auto sum = mfq_tensor_backend::empty_like(a);
    auto norm = mfq_tensor_backend::empty_like(a);
    acc_rms_norm_bf16_kernel<ACC_RMS_BD><<<
        N, ACC_RMS_BD, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<mfq_bfloat16>()),
        weight.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(sum.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<__nv_bfloat16*>(norm.data_ptr<mfq_bfloat16>()),
        N, D, (float)eps, (float)weight_offset);
    return {sum, norm};
}

template <int BD>
__global__ void gemma4_attn_residual_pre_norms_f16_kernel(
    const mfq_half* __restrict__ residual,
    const mfq_half* __restrict__ attn,
    const float* __restrict__ attn_post_weight,
    const float* __restrict__ dense_pre_weight,
    const float* __restrict__ router_weight,
    const float* __restrict__ moe_pre_weight,
    mfq_half* __restrict__ residual_out,
    mfq_half* __restrict__ dense_out,
    float* __restrict__ router_out,
    mfq_half* __restrict__ moe_out,
    int N, int D, float eps)
{
    const int row = blockIdx.x;
    if (row >= N) {
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    auto* shared_x = reinterpret_cast<mfq_half*>(shared_bytes);
    const size_t row_offset = (size_t)row * D;
    const mfq_half* rr = residual + row_offset;
    const mfq_half* ar = attn + row_offset;
    mfq_half* xr = residual_out + row_offset;
    mfq_half* dr = dense_out + row_offset;
    float* router_r = router_out + row_offset;
    mfq_half* mr = moe_out + row_offset;

    float attn_ssq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const float value = (float)ar[i];
        attn_ssq += value * value;
    }
    attn_ssq = block_sum<BD / 32>(attn_ssq);
    const float attn_rinv = rsqrtf(attn_ssq / (float)D + eps);

    float x_ssq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const mfq_half attn_norm = (mfq_half)(
            (float)ar[i] * attn_rinv * attn_post_weight[i]);
        const mfq_half x_value = (mfq_half)((float)rr[i] + (float)attn_norm);
        xr[i] = x_value;
        shared_x[i] = x_value;
        const float value = (float)x_value;
        x_ssq += value * value;
    }
    x_ssq = block_sum<BD / 32>(x_ssq);
    const float x_rinv = rsqrtf(x_ssq / (float)D + eps);

    for (int i = threadIdx.x; i < D; i += BD) {
        const float value = (float)shared_x[i] * x_rinv;
        dr[i] = (mfq_half)(value * dense_pre_weight[i]);
        router_r[i] = value * router_weight[i];
        mr[i] = (mfq_half)(value * moe_pre_weight[i]);
    }
}

std::vector<mfq_tensor_backend::Tensor> gemma4_attn_residual_pre_norms_f16_cuda(
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor attn,
    mfq_tensor_backend::Tensor attn_post_weight, mfq_tensor_backend::Tensor dense_pre_weight,
    mfq_tensor_backend::Tensor router_weight, mfq_tensor_backend::Tensor moe_pre_weight, double eps)
{
    MFQ_RUNTIME_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                    residual.scalar_type() == mfq_tensor_backend::kFloat16,
                "gemma4 fused pre norms: residual must be cuda contiguous f16");
    MFQ_RUNTIME_CHECK(attn.is_cuda() && attn.is_contiguous() &&
                    attn.scalar_type() == mfq_tensor_backend::kFloat16,
                "gemma4 fused pre norms: attention output must be cuda contiguous f16");
    MFQ_RUNTIME_CHECK(residual.sizes() == attn.sizes(),
                "gemma4 fused pre norms: activation shape mismatch");
    MFQ_RUNTIME_CHECK(residual.dim() >= 1,
                "gemma4 fused pre norms: activation must have at least one dimension");
    const int D = (int)residual.size(-1);
    const int N = (int)(residual.numel() / D);
    for (const auto& weight : {attn_post_weight, dense_pre_weight, router_weight, moe_pre_weight}) {
        MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                        weight.scalar_type() == mfq_tensor_backend::kFloat32 && weight.numel() == D,
                    "gemma4 fused pre norms: weights must be cuda contiguous f32[D]");
    }

    auto residual_out = mfq_tensor_backend::empty_like(residual);
    auto dense_out = mfq_tensor_backend::empty_like(residual);
    auto router_out = mfq_tensor_backend::empty(residual.sizes(), residual.options().dtype(mfq_tensor_backend::kFloat32));
    auto moe_out = mfq_tensor_backend::empty_like(residual);
    constexpr int BD = ACC_RMS_BD;
    const size_t shared_bytes = (size_t)D * sizeof(mfq_half);
    gemma4_attn_residual_pre_norms_f16_kernel<BD>
        <<<N, BD, shared_bytes, mfq_current_cuda_stream()>>>(
            residual.data_ptr<mfq_half>(), attn.data_ptr<mfq_half>(),
            attn_post_weight.data_ptr<float>(), dense_pre_weight.data_ptr<float>(),
            router_weight.data_ptr<float>(), moe_pre_weight.data_ptr<float>(),
            residual_out.data_ptr<mfq_half>(), dense_out.data_ptr<mfq_half>(),
            router_out.data_ptr<float>(), moe_out.data_ptr<mfq_half>(),
            N, D, (float)eps);
    return {residual_out, dense_out, router_out, moe_out};
}

template <int BD>
__global__ void gemma4_ffn_merge_f16_kernel(
    const mfq_half* __restrict__ dense,
    const mfq_half* __restrict__ moe,
    const mfq_half* __restrict__ residual,
    const float* __restrict__ dense_post_weight,
    const float* __restrict__ moe_post_weight,
    const float* __restrict__ final_post_weight,
    const mfq_half* __restrict__ layer_scale,
    mfq_half* __restrict__ out,
    int N, int D, float eps)
{
    const int row = blockIdx.x;
    if (row >= N) {
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    auto* combined = reinterpret_cast<mfq_half*>(shared_bytes);
    const size_t row_offset = (size_t)row * D;
    const mfq_half* dr = dense + row_offset;
    const mfq_half* mr = moe + row_offset;
    const mfq_half* rr = residual + row_offset;
    mfq_half* orow = out + row_offset;

    float dense_ssq = 0.0f;
    float moe_ssq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const float dense_value = (float)dr[i];
        const float moe_value = (float)mr[i];
        dense_ssq += dense_value * dense_value;
        moe_ssq += moe_value * moe_value;
    }
    dense_ssq = block_sum<BD / 32>(dense_ssq);
    moe_ssq = block_sum<BD / 32>(moe_ssq);
    const float dense_rinv = rsqrtf(dense_ssq / (float)D + eps);
    const float moe_rinv = rsqrtf(moe_ssq / (float)D + eps);

    float combined_ssq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const mfq_half dense_norm = (mfq_half)(
            (float)dr[i] * dense_rinv * dense_post_weight[i]);
        const mfq_half moe_norm = (mfq_half)(
            (float)mr[i] * moe_rinv * moe_post_weight[i]);
        const mfq_half value = (mfq_half)((float)dense_norm + (float)moe_norm);
        combined[i] = value;
        const float value_f = (float)value;
        combined_ssq += value_f * value_f;
    }
    combined_ssq = block_sum<BD / 32>(combined_ssq);
    const float combined_rinv = rsqrtf(combined_ssq / (float)D + eps);
    const float scale = (float)layer_scale[0];

    for (int i = threadIdx.x; i < D; i += BD) {
        const mfq_half post = (mfq_half)(
            (float)combined[i] * combined_rinv * final_post_weight[i]);
        const mfq_half residual_sum = (mfq_half)((float)rr[i] + (float)post);
        orow[i] = (mfq_half)((float)residual_sum * scale);
    }
}

mfq_tensor_backend::Tensor gemma4_ffn_merge_f16_cuda(
    mfq_tensor_backend::Tensor dense, mfq_tensor_backend::Tensor moe, mfq_tensor_backend::Tensor residual,
    mfq_tensor_backend::Tensor dense_post_weight, mfq_tensor_backend::Tensor moe_post_weight,
    mfq_tensor_backend::Tensor final_post_weight, mfq_tensor_backend::Tensor layer_scale, double eps)
{
    for (const auto& value : {dense, moe, residual}) {
        MFQ_RUNTIME_CHECK(value.is_cuda() && value.is_contiguous() &&
                        value.scalar_type() == mfq_tensor_backend::kFloat16,
                    "gemma4 fused FFN merge: activations must be cuda contiguous f16");
    }
    MFQ_RUNTIME_CHECK(dense.sizes() == moe.sizes() && dense.sizes() == residual.sizes(),
                "gemma4 fused FFN merge: activation shape mismatch");
    MFQ_RUNTIME_CHECK(dense.dim() >= 1,
                "gemma4 fused FFN merge: activation must have at least one dimension");
    const int D = (int)dense.size(-1);
    const int N = (int)(dense.numel() / D);
    for (const auto& weight : {dense_post_weight, moe_post_weight, final_post_weight}) {
        MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                        weight.scalar_type() == mfq_tensor_backend::kFloat32 && weight.numel() == D,
                    "gemma4 fused FFN merge: weights must be cuda contiguous f32[D]");
    }
    MFQ_RUNTIME_CHECK(layer_scale.is_cuda() && layer_scale.is_contiguous() &&
                    layer_scale.scalar_type() == mfq_tensor_backend::kFloat16 && layer_scale.numel() == 1,
                "gemma4 fused FFN merge: layer scale must be cuda contiguous f16[1]");

    auto out = mfq_tensor_backend::empty_like(dense);
    constexpr int BD = ACC_RMS_BD;
    const size_t shared_bytes = (size_t)D * sizeof(mfq_half);
    gemma4_ffn_merge_f16_kernel<BD>
        <<<N, BD, shared_bytes, mfq_current_cuda_stream()>>>(
            dense.data_ptr<mfq_half>(), moe.data_ptr<mfq_half>(), residual.data_ptr<mfq_half>(),
            dense_post_weight.data_ptr<float>(), moe_post_weight.data_ptr<float>(),
            final_post_weight.data_ptr<float>(), layer_scale.data_ptr<mfq_half>(),
            out.data_ptr<mfq_half>(), N, D, (float)eps);
    return out;
}

__global__ void decode_graph_commit_kernel(const int64_t* __restrict__ next,
                                           int64_t* __restrict__ generated,
                                           int64_t* __restrict__ step,
                                           int64_t* __restrict__ input,
                                           int64_t* __restrict__ pos,
                                           int64_t* __restrict__ len)
{
    int64_t idx = step[0];
    int64_t tok = next[0];
    generated[idx] = tok;
    input[0] = tok;
    pos[0] += 1;
    len[0] += 1;
    step[0] = idx + 1;
}

void decode_graph_commit_cuda(mfq_tensor_backend::Tensor next, mfq_tensor_backend::Tensor generated, mfq_tensor_backend::Tensor step,
                              mfq_tensor_backend::Tensor input, mfq_tensor_backend::Tensor pos, mfq_tensor_backend::Tensor len)
{
    MFQ_RUNTIME_CHECK(next.is_cuda() && next.is_contiguous() && next.scalar_type() == mfq_tensor_backend::kInt64,
                "decode_graph_commit: next must be cuda int64 contiguous");
    MFQ_RUNTIME_CHECK(generated.is_cuda() && generated.is_contiguous() && generated.scalar_type() == mfq_tensor_backend::kInt64,
                "decode_graph_commit: generated must be cuda int64 contiguous");
    MFQ_RUNTIME_CHECK(step.is_cuda() && step.is_contiguous() && step.scalar_type() == mfq_tensor_backend::kInt64 && step.numel() == 1,
                "decode_graph_commit: step must be cuda int64[1]");
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type() == mfq_tensor_backend::kInt64 && input.numel() == 1,
                "decode_graph_commit: input must be cuda int64[1]");
    MFQ_RUNTIME_CHECK(pos.is_cuda() && pos.is_contiguous() && pos.scalar_type() == mfq_tensor_backend::kInt64 && pos.numel() == 1,
                "decode_graph_commit: pos must be cuda int64[1]");
    MFQ_RUNTIME_CHECK(len.is_cuda() && len.is_contiguous() && len.scalar_type() == mfq_tensor_backend::kInt64 && len.numel() == 1,
                "decode_graph_commit: len must be cuda int64[1]");
    decode_graph_commit_kernel<<<1, 1, 0, mfq_current_cuda_stream()>>>(
        next.data_ptr<int64_t>(), generated.data_ptr<int64_t>(), step.data_ptr<int64_t>(),
        input.data_ptr<int64_t>(), pos.data_ptr<int64_t>(), len.data_ptr<int64_t>());
}
