// Residual add a + b (ggml acc.cu: ggml_acc). fp16/fp32, any shape flattened.

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
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

torch::Tensor acc_cuda(torch::Tensor a, torch::Tensor b)
{
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(), "acc: a must be cuda contiguous");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous(), "acc: b must be cuda contiguous");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "acc: a/b dtype mismatch");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16 || a.scalar_type() == torch::kFloat32,
                "acc: dtype must be f16 or f32");
    TORCH_CHECK(a.sizes() == b.sizes(), "acc: a/b shape mismatch");
    int n = (int)a.numel();
    auto out = torch::empty_like(a);
    constexpr int BD = 256;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "acc_cuda", [&] {
        acc_kernel<scalar_t><<<(n + BD - 1) / BD, BD, 0, at::cuda::getCurrentCUDAStream()>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
    return out;
}

constexpr int ACC_RMS_BD = 256;

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

std::vector<torch::Tensor> acc_rms_norm_cuda(torch::Tensor a, torch::Tensor b,
                                             torch::Tensor weight, double eps,
                                             double weight_offset)
{
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(), "acc_rms_norm: a must be cuda contiguous");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous(), "acc_rms_norm: b must be cuda contiguous");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kFloat32,
                "acc_rms_norm: weight must be cuda contiguous f32");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "acc_rms_norm: a/b dtype mismatch");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16 || a.scalar_type() == torch::kFloat32,
                "acc_rms_norm: dtype must be f16 or f32");
    TORCH_CHECK(a.sizes() == b.sizes(), "acc_rms_norm: a/b shape mismatch");
    TORCH_CHECK(a.dim() >= 1, "acc_rms_norm: a must have at least one dim");
    int D = (int)a.size(-1);
    int N = (int)(a.numel() / D);
    TORCH_CHECK(weight.numel() == D, "acc_rms_norm: weight length mismatch");

    auto sum = torch::empty_like(a);
    auto norm = torch::empty(a.sizes(), a.options().dtype(torch::kFloat32));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "acc_rms_norm_cuda", [&] {
        acc_rms_norm_kernel<scalar_t, float, ACC_RMS_BD><<<N, ACC_RMS_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), weight.data_ptr<float>(),
            sum.data_ptr<scalar_t>(), norm.data_ptr<float>(), N, D,
            (float)eps, (float)weight_offset);
    });
    return {sum, norm};
}

std::vector<torch::Tensor> acc_rms_norm_f16_cuda(torch::Tensor a, torch::Tensor b,
                                                 torch::Tensor weight, double eps,
                                                 double weight_offset)
{
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(), "acc_rms_norm_f16: a must be cuda contiguous");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous(), "acc_rms_norm_f16: b must be cuda contiguous");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kFloat32,
                "acc_rms_norm_f16: weight must be cuda contiguous f32");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16 && b.scalar_type() == torch::kFloat16,
                "acc_rms_norm_f16: a/b must be f16");
    TORCH_CHECK(a.sizes() == b.sizes(), "acc_rms_norm_f16: a/b shape mismatch");
    TORCH_CHECK(a.dim() >= 1, "acc_rms_norm_f16: a must have at least one dim");
    int D = (int)a.size(-1);
    int N = (int)(a.numel() / D);
    TORCH_CHECK(weight.numel() == D, "acc_rms_norm_f16: weight length mismatch");

    auto sum = torch::empty_like(a);
    auto norm = torch::empty_like(a);
    acc_rms_norm_kernel<at::Half, at::Half, ACC_RMS_BD><<<N, ACC_RMS_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<at::Half>(), b.data_ptr<at::Half>(), weight.data_ptr<float>(),
        sum.data_ptr<at::Half>(), norm.data_ptr<at::Half>(), N, D,
        (float)eps, (float)weight_offset);
    return {sum, norm};
}

template <int BD>
__global__ void gemma4_attn_residual_pre_norms_f16_kernel(
    const at::Half* __restrict__ residual,
    const at::Half* __restrict__ attn,
    const float* __restrict__ attn_post_weight,
    const float* __restrict__ dense_pre_weight,
    const float* __restrict__ router_weight,
    const float* __restrict__ moe_pre_weight,
    at::Half* __restrict__ residual_out,
    at::Half* __restrict__ dense_out,
    float* __restrict__ router_out,
    at::Half* __restrict__ moe_out,
    int N, int D, float eps)
{
    const int row = blockIdx.x;
    if (row >= N) {
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    auto* shared_x = reinterpret_cast<at::Half*>(shared_bytes);
    const size_t row_offset = (size_t)row * D;
    const at::Half* rr = residual + row_offset;
    const at::Half* ar = attn + row_offset;
    at::Half* xr = residual_out + row_offset;
    at::Half* dr = dense_out + row_offset;
    float* router_r = router_out + row_offset;
    at::Half* mr = moe_out + row_offset;

    float attn_ssq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const float value = (float)ar[i];
        attn_ssq += value * value;
    }
    attn_ssq = block_sum<BD / 32>(attn_ssq);
    const float attn_rinv = rsqrtf(attn_ssq / (float)D + eps);

    float x_ssq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BD) {
        const at::Half attn_norm = (at::Half)(
            (float)ar[i] * attn_rinv * attn_post_weight[i]);
        const at::Half x_value = (at::Half)((float)rr[i] + (float)attn_norm);
        xr[i] = x_value;
        shared_x[i] = x_value;
        const float value = (float)x_value;
        x_ssq += value * value;
    }
    x_ssq = block_sum<BD / 32>(x_ssq);
    const float x_rinv = rsqrtf(x_ssq / (float)D + eps);

    for (int i = threadIdx.x; i < D; i += BD) {
        const float value = (float)shared_x[i] * x_rinv;
        dr[i] = (at::Half)(value * dense_pre_weight[i]);
        router_r[i] = value * router_weight[i];
        mr[i] = (at::Half)(value * moe_pre_weight[i]);
    }
}

std::vector<torch::Tensor> gemma4_attn_residual_pre_norms_f16_cuda(
    torch::Tensor residual, torch::Tensor attn,
    torch::Tensor attn_post_weight, torch::Tensor dense_pre_weight,
    torch::Tensor router_weight, torch::Tensor moe_pre_weight, double eps)
{
    TORCH_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                    residual.scalar_type() == torch::kFloat16,
                "gemma4 fused pre norms: residual must be cuda contiguous f16");
    TORCH_CHECK(attn.is_cuda() && attn.is_contiguous() &&
                    attn.scalar_type() == torch::kFloat16,
                "gemma4 fused pre norms: attention output must be cuda contiguous f16");
    TORCH_CHECK(residual.sizes() == attn.sizes(),
                "gemma4 fused pre norms: activation shape mismatch");
    TORCH_CHECK(residual.dim() >= 1,
                "gemma4 fused pre norms: activation must have at least one dimension");
    const int D = (int)residual.size(-1);
    const int N = (int)(residual.numel() / D);
    for (const auto& weight : {attn_post_weight, dense_pre_weight, router_weight, moe_pre_weight}) {
        TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                        weight.scalar_type() == torch::kFloat32 && weight.numel() == D,
                    "gemma4 fused pre norms: weights must be cuda contiguous f32[D]");
    }

    auto residual_out = torch::empty_like(residual);
    auto dense_out = torch::empty_like(residual);
    auto router_out = torch::empty(residual.sizes(), residual.options().dtype(torch::kFloat32));
    auto moe_out = torch::empty_like(residual);
    constexpr int BD = ACC_RMS_BD;
    const size_t shared_bytes = (size_t)D * sizeof(at::Half);
    gemma4_attn_residual_pre_norms_f16_kernel<BD>
        <<<N, BD, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            residual.data_ptr<at::Half>(), attn.data_ptr<at::Half>(),
            attn_post_weight.data_ptr<float>(), dense_pre_weight.data_ptr<float>(),
            router_weight.data_ptr<float>(), moe_pre_weight.data_ptr<float>(),
            residual_out.data_ptr<at::Half>(), dense_out.data_ptr<at::Half>(),
            router_out.data_ptr<float>(), moe_out.data_ptr<at::Half>(),
            N, D, (float)eps);
    return {residual_out, dense_out, router_out, moe_out};
}

template <int BD>
__global__ void gemma4_ffn_merge_f16_kernel(
    const at::Half* __restrict__ dense,
    const at::Half* __restrict__ moe,
    const at::Half* __restrict__ residual,
    const float* __restrict__ dense_post_weight,
    const float* __restrict__ moe_post_weight,
    const float* __restrict__ final_post_weight,
    const at::Half* __restrict__ layer_scale,
    at::Half* __restrict__ out,
    int N, int D, float eps)
{
    const int row = blockIdx.x;
    if (row >= N) {
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    auto* combined = reinterpret_cast<at::Half*>(shared_bytes);
    const size_t row_offset = (size_t)row * D;
    const at::Half* dr = dense + row_offset;
    const at::Half* mr = moe + row_offset;
    const at::Half* rr = residual + row_offset;
    at::Half* orow = out + row_offset;

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
        const at::Half dense_norm = (at::Half)(
            (float)dr[i] * dense_rinv * dense_post_weight[i]);
        const at::Half moe_norm = (at::Half)(
            (float)mr[i] * moe_rinv * moe_post_weight[i]);
        const at::Half value = (at::Half)((float)dense_norm + (float)moe_norm);
        combined[i] = value;
        const float value_f = (float)value;
        combined_ssq += value_f * value_f;
    }
    combined_ssq = block_sum<BD / 32>(combined_ssq);
    const float combined_rinv = rsqrtf(combined_ssq / (float)D + eps);
    const float scale = (float)layer_scale[0];

    for (int i = threadIdx.x; i < D; i += BD) {
        const at::Half post = (at::Half)(
            (float)combined[i] * combined_rinv * final_post_weight[i]);
        const at::Half residual_sum = (at::Half)((float)rr[i] + (float)post);
        orow[i] = (at::Half)((float)residual_sum * scale);
    }
}

torch::Tensor gemma4_ffn_merge_f16_cuda(
    torch::Tensor dense, torch::Tensor moe, torch::Tensor residual,
    torch::Tensor dense_post_weight, torch::Tensor moe_post_weight,
    torch::Tensor final_post_weight, torch::Tensor layer_scale, double eps)
{
    for (const auto& value : {dense, moe, residual}) {
        TORCH_CHECK(value.is_cuda() && value.is_contiguous() &&
                        value.scalar_type() == torch::kFloat16,
                    "gemma4 fused FFN merge: activations must be cuda contiguous f16");
    }
    TORCH_CHECK(dense.sizes() == moe.sizes() && dense.sizes() == residual.sizes(),
                "gemma4 fused FFN merge: activation shape mismatch");
    TORCH_CHECK(dense.dim() >= 1,
                "gemma4 fused FFN merge: activation must have at least one dimension");
    const int D = (int)dense.size(-1);
    const int N = (int)(dense.numel() / D);
    for (const auto& weight : {dense_post_weight, moe_post_weight, final_post_weight}) {
        TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                        weight.scalar_type() == torch::kFloat32 && weight.numel() == D,
                    "gemma4 fused FFN merge: weights must be cuda contiguous f32[D]");
    }
    TORCH_CHECK(layer_scale.is_cuda() && layer_scale.is_contiguous() &&
                    layer_scale.scalar_type() == torch::kFloat16 && layer_scale.numel() == 1,
                "gemma4 fused FFN merge: layer scale must be cuda contiguous f16[1]");

    auto out = torch::empty_like(dense);
    constexpr int BD = ACC_RMS_BD;
    const size_t shared_bytes = (size_t)D * sizeof(at::Half);
    gemma4_ffn_merge_f16_kernel<BD>
        <<<N, BD, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            dense.data_ptr<at::Half>(), moe.data_ptr<at::Half>(), residual.data_ptr<at::Half>(),
            dense_post_weight.data_ptr<float>(), moe_post_weight.data_ptr<float>(),
            final_post_weight.data_ptr<float>(), layer_scale.data_ptr<at::Half>(),
            out.data_ptr<at::Half>(), N, D, (float)eps);
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

void decode_graph_commit_cuda(torch::Tensor next, torch::Tensor generated, torch::Tensor step,
                              torch::Tensor input, torch::Tensor pos, torch::Tensor len)
{
    TORCH_CHECK(next.is_cuda() && next.is_contiguous() && next.scalar_type() == torch::kInt64,
                "decode_graph_commit: next must be cuda int64 contiguous");
    TORCH_CHECK(generated.is_cuda() && generated.is_contiguous() && generated.scalar_type() == torch::kInt64,
                "decode_graph_commit: generated must be cuda int64 contiguous");
    TORCH_CHECK(step.is_cuda() && step.is_contiguous() && step.scalar_type() == torch::kInt64 && step.numel() == 1,
                "decode_graph_commit: step must be cuda int64[1]");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type() == torch::kInt64 && input.numel() == 1,
                "decode_graph_commit: input must be cuda int64[1]");
    TORCH_CHECK(pos.is_cuda() && pos.is_contiguous() && pos.scalar_type() == torch::kInt64 && pos.numel() == 1,
                "decode_graph_commit: pos must be cuda int64[1]");
    TORCH_CHECK(len.is_cuda() && len.is_contiguous() && len.scalar_type() == torch::kInt64 && len.numel() == 1,
                "decode_graph_commit: len must be cuda int64[1]");
    decode_graph_commit_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        next.data_ptr<int64_t>(), generated.data_ptr<int64_t>(), step.data_ptr<int64_t>(),
        input.data_ptr<int64_t>(), pos.data_ptr<int64_t>(), len.data_ptr<int64_t>());
}
