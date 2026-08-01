// Fused causal depthwise SSM conv + SiLU.
// conv_input [B, K - 1 + T, C], weight [C,1,K], [C,K], or [K,C], output [B,T,C].

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>

#include "reduce.cuh"

constexpr int SSM_CONV_BD = 256;

__device__ inline float silu_f32(float x)
{
    return x / (1.0f + expf(-x));
}

__global__ void ssm_conv_silu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B,
    int T,
    int C,
    int K,
    int weight_layout,
    int has_bias)
{
    int b = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) {
        return;
    }
    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            float xv = x[((size_t)b * (T + K - 1) + t + j) * C + c];
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        if (has_bias) {
            sum += bias[c];
        }
        out[((size_t)b * T + t) * C + c] = silu_f32(sum);
    }
}

torch::Tensor ssm_conv_silu_cuda(torch::Tensor conv_input, torch::Tensor weight, torch::Tensor bias, int64_t n_tokens)
{
    TORCH_CHECK(conv_input.is_cuda() && conv_input.is_contiguous() && conv_input.scalar_type() == torch::kFloat32,
                "ssm_conv_silu: conv_input must be cuda contiguous f32");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kFloat32,
                "ssm_conv_silu: weight must be cuda contiguous f32");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == torch::kFloat32,
                "ssm_conv_silu: bias must be cuda contiguous f32");
    TORCH_CHECK(conv_input.dim() == 3, "ssm_conv_silu: conv_input must be [B,K-1+T,C]");
    int B = (int)conv_input.size(0);
    int C = (int)conv_input.size(2);
    int has_bias = (int)(bias.numel() > 0);
    TORCH_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "ssm_conv_silu: bias must be [C]");
    int K = 0;
    int layout = 0; // 0: channel-major [C,1,K]/[C,K], 1: [K,C]
    if (weight.dim() == 3) {
        TORCH_CHECK(weight.size(0) == C && weight.size(1) == 1, "ssm_conv_silu: [C,1,K] weight mismatch");
        K = (int)weight.size(2);
        layout = 0;
    } else {
        TORCH_CHECK(weight.dim() == 2, "ssm_conv_silu: weight must be [C,1,K], [C,K], or [K,C]");
        if (weight.size(0) == C) {
            K = (int)weight.size(1);
            layout = 0;
        } else {
            TORCH_CHECK(weight.size(1) == C, "ssm_conv_silu: 2D weight must be [C,K] or [K,C]");
            K = (int)weight.size(0);
            layout = 1;
        }
    }
    int T = (int)n_tokens;
    TORCH_CHECK(T > 0 && conv_input.size(1) == T + K - 1, "ssm_conv_silu: conv_input length must be K-1+T");
    auto out = torch::empty({B, T, C}, conv_input.options());
    dim3 blocks(B, (C + SSM_CONV_BD - 1) / SSM_CONV_BD);
    ssm_conv_silu_kernel<<<blocks, SSM_CONV_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
        conv_input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
        B, T, C, K, layout, has_bias);
    return out;
}

__global__ void ssm_conv_silu_decode_kernel(
    float* __restrict__ state,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B,
    int C,
    int K,
    int weight_layout,
    int has_bias)
{
    int b = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) {
        return;
    }
    size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
    float sum = 0.0f;
    for (int j = 0; j < K - 1; ++j) {
        float xv = state[state_base + (size_t)j * (size_t)C];
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    float x_cur = x[(size_t)b * (size_t)C + (size_t)c];
    float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
    sum += x_cur * wv;
    if (has_bias) {
        sum += bias[c];
    }
    out[(size_t)b * (size_t)C + (size_t)c] = silu_f32(sum);
    for (int j = 0; j < K - 2; ++j) {
        state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
    }
    state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
}

torch::Tensor ssm_conv_silu_decode_cuda(torch::Tensor state, torch::Tensor x, torch::Tensor weight, torch::Tensor bias)
{
    TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == torch::kFloat32,
                "ssm_conv_silu_decode: state must be cuda contiguous f32");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32,
                "ssm_conv_silu_decode: x must be cuda contiguous f32");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kFloat32,
                "ssm_conv_silu_decode: weight must be cuda contiguous f32");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == torch::kFloat32,
                "ssm_conv_silu_decode: bias must be cuda contiguous f32");
    TORCH_CHECK(state.dim() == 3, "ssm_conv_silu_decode: state must be [B,K-1,C]");
    TORCH_CHECK(x.dim() == 3 && x.size(1) == 1, "ssm_conv_silu_decode: x must be [B,1,C]");
    int B = (int)state.size(0);
    int K = (int)state.size(1) + 1;
    int C = (int)state.size(2);
    TORCH_CHECK(x.size(0) == B && x.size(2) == C, "ssm_conv_silu_decode: x shape mismatch");
    int has_bias = (int)(bias.numel() > 0);
    TORCH_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "ssm_conv_silu_decode: bias must be [C]");
    int layout = 0;
    if (weight.dim() == 3) {
        TORCH_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "ssm_conv_silu_decode: [C,1,K] weight mismatch");
        layout = 0;
    } else {
        TORCH_CHECK(weight.dim() == 2,
                    "ssm_conv_silu_decode: weight must be [C,1,K], [C,K], or [K,C]");
        if (weight.size(0) == C && weight.size(1) == K) {
            layout = 0;
        } else {
            TORCH_CHECK(weight.size(0) == K && weight.size(1) == C,
                        "ssm_conv_silu_decode: 2D weight must be [C,K] or [K,C]");
            layout = 1;
        }
    }
    auto out = torch::empty({B, 1, C}, x.options());
    dim3 blocks(B, (C + SSM_CONV_BD - 1) / SSM_CONV_BD);
    ssm_conv_silu_decode_kernel<<<blocks, SSM_CONV_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
        state.data_ptr<float>(), x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), B, C, K, layout, has_bias);
    return out;
}

__device__ inline float ssm_decode_cur(
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    int b,
    int c,
    int qkC,
    int vsz)
{
    if (c < qkC) {
        return __half2float(qk[(size_t)b * (size_t)qkC + (size_t)c]);
    }
    return __half2float(v[(size_t)b * (size_t)vsz + (size_t)(c - qkC)]);
}

template <int BD>
__global__ void ssm_conv_qk_norm_decode_kernel(
    float* __restrict__ state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ q,
    float* __restrict__ k,
    int B,
    int nk,
    int nv,
    int dk,
    int dv,
    int K,
    int weight_layout,
    int has_bias,
    float eps)
{
    int row = blockIdx.x;
    int b = row / (2 * nk);
    int rem = row - b * 2 * nk;
    int which = rem / nk;
    int h = rem - which * nk;
    int tid = threadIdx.x;
    int qkC = 2 * nk * dk;
    int vsz = nv * dv;
    int C = qkC + nv * dv;
    int base_c = (which * nk + h) * dk;

    __shared__ float vals[BD];
    __shared__ float sums[BD];
    float val = 0.0f;
    if (tid < dk) {
        int c = base_c + tid;
        size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
        float sum = 0.0f;
        for (int j = 0; j < K - 1; ++j) {
            float xv = state[state_base + (size_t)j * (size_t)C];
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        float x_cur = ssm_decode_cur(qk, v, b, c, qkC, vsz);
        float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
        sum += x_cur * wv;
        if (has_bias) {
            sum += bias[c];
        }
        val = silu_f32(sum);
        vals[tid] = val;
        for (int j = 0; j < K - 2; ++j) {
            state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
        }
        state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
    }
    float ssq = (tid < dk) ? val * val : 0.0f;
    sums[tid] = ssq;
    __syncthreads();
    for (int stride = BD / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        __syncthreads();
    }
    float inv = 1.0f / fmaxf(sqrtf(sums[0]), eps);
    if (tid < dk) {
        float outv = vals[tid] * inv;
        float* dst = which == 0 ? q : k;
        dst[((size_t)b * nk + h) * dk + tid] = outv;
    }
}

__global__ void ssm_conv_v_decode_kernel(
    float* __restrict__ state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v_in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ v_out,
    int B,
    int nk,
    int nv,
    int dk,
    int dv,
    int K,
    int weight_layout,
    int has_bias)
{
    int b = blockIdx.x;
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    int vsz = nv * dv;
    if (b >= B || idx >= vsz) {
        return;
    }
    int qkC = 2 * nk * dk;
    int C = qkC + vsz;
    int c = qkC + idx;
    size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
    float sum = 0.0f;
    for (int j = 0; j < K - 1; ++j) {
        float xv = state[state_base + (size_t)j * (size_t)C];
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    float x_cur = __half2float(v_in[(size_t)b * (size_t)vsz + (size_t)idx]);
    float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
    sum += x_cur * wv;
    if (has_bias) {
        sum += bias[c];
    }
    v_out[(size_t)b * (size_t)vsz + (size_t)idx] = silu_f32(sum);
    for (int j = 0; j < K - 2; ++j) {
        state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
    }
    state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
}

std::vector<torch::Tensor> linear_conv_qkv_decode_cuda(
    torch::Tensor state,
    torch::Tensor qk,
    torch::Tensor v,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t nk,
    int64_t nv,
    int64_t dk,
    int64_t dv,
    double eps)
{
    TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == torch::kFloat32,
                "linear_conv_qkv_decode: state must be cuda contiguous f32");
    TORCH_CHECK(qk.is_cuda() && qk.is_contiguous() && qk.scalar_type() == torch::kHalf,
                "linear_conv_qkv_decode: qk must be cuda contiguous f16");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == torch::kHalf,
                "linear_conv_qkv_decode: v must be cuda contiguous f16");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kFloat32,
                "linear_conv_qkv_decode: weight must be cuda contiguous f32");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == torch::kFloat32,
                "linear_conv_qkv_decode: bias must be cuda contiguous f32");
    TORCH_CHECK(state.dim() == 3, "linear_conv_qkv_decode: state must be [B,K-1,C]");
    TORCH_CHECK(qk.dim() == 3 && qk.size(1) == 1, "linear_conv_qkv_decode: qk must be [B,1,2*nk*dk]");
    TORCH_CHECK(v.dim() == 3 && v.size(1) == 1, "linear_conv_qkv_decode: v must be [B,1,nv*dv]");
    int B = (int)state.size(0);
    int K = (int)state.size(1) + 1;
    int qkC = (int)(2 * nk * dk);
    int vsz = (int)(nv * dv);
    int C = qkC + vsz;
    TORCH_CHECK(qk.size(0) == B && qk.size(2) == qkC, "linear_conv_qkv_decode: qk shape mismatch");
    TORCH_CHECK(v.size(0) == B && v.size(2) == vsz, "linear_conv_qkv_decode: v shape mismatch");
    TORCH_CHECK(state.size(2) == C, "linear_conv_qkv_decode: state width mismatch");
    TORCH_CHECK(nv % nk == 0, "linear_conv_qkv_decode: nv must be divisible by nk");
    TORCH_CHECK(dk <= 256, "linear_conv_qkv_decode: dk > 256 is unsupported");
    int has_bias = (int)(bias.numel() > 0);
    TORCH_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "linear_conv_qkv_decode: bias must be [C]");
    int layout = 0;
    if (weight.dim() == 3) {
        TORCH_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "linear_conv_qkv_decode: [C,1,K] weight mismatch");
        layout = 0;
    } else {
        TORCH_CHECK(weight.dim() == 2 && weight.size(0) == K && weight.size(1) == C,
                    "linear_conv_qkv_decode: weight must be [C,1,K] or [K,C]");
        layout = 1;
    }
    auto opts = state.options();
    auto q = torch::empty({B, (int)nk, 1, (int)dk}, opts);
    auto k = torch::empty({B, (int)nk, 1, (int)dk}, opts);
    auto vo = torch::empty({B, (int)nv, 1, (int)dv}, opts);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ssm_conv_qk_norm_decode_kernel<256><<<B * 2 * (int)nk, 256, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const __half*>(qk.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()), weight.data_ptr<float>(),
        bias.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(), B, (int)nk, (int)nv,
        (int)dk, (int)dv, K, layout, has_bias, (float)eps);
    dim3 v_blocks(B, (vsz + SSM_CONV_BD - 1) / SSM_CONV_BD);
    ssm_conv_v_decode_kernel<<<v_blocks, SSM_CONV_BD, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const __half*>(qk.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()), weight.data_ptr<float>(),
        bias.data_ptr<float>(), vo.data_ptr<float>(), B, (int)nk, (int)nv, (int)dk, (int)dv,
        K, layout, has_bias);
    return {q, k, vo};
}

template <int BD>
__global__ void linear_conv_qk_norm_prefill_kernel(
    const float* __restrict__ state,
    const __half* __restrict__ qk,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ q,
    float* __restrict__ k,
    int B, int T, int nk, int dk, int C, int K,
    int weight_layout, int has_bias, float eps)
{
    int row = blockIdx.x;
    int h = row % nk;
    int r = row / nk;
    int which = r & 1;
    r >>= 1;
    int t = r % T;
    int b = r / T;
    int tid = threadIdx.x;
    int qkC = 2 * nk * dk;
    int c = (which * nk + h) * dk + tid;
    float val = 0.0f;
    if (tid < dk) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j >= K) break;
            int token = t + j - (K - 1);
            float xv;
            if (token < 0) {
                xv = state[((size_t)b * (K - 1) + (token + K - 1)) * C + c];
            } else {
                xv = __half2float(qk[((size_t)b * T + token) * qkC + c]);
            }
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        if (has_bias) sum += bias[c];
        val = silu_f32(sum);
    }
    float ssq = block_sum<BD / 32>(tid < dk ? val * val : 0.0f);
    float inv = 1.0f / fmaxf(sqrtf(ssq), eps);
    if (tid < dk) {
        float* dst = which == 0 ? q : k;
        dst[(((size_t)b * nk + h) * T + t) * dk + tid] = val * inv;
    }
}

__global__ void linear_conv_v_prefill_kernel(
    const float* __restrict__ state,
    const __half* __restrict__ v_in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ v_out,
    int B, int T, int nk, int nv, int dk, int dv, int C, int K,
    int weight_layout, int has_bias)
{
    int bt = blockIdx.x;
    int b = bt / T;
    int t = bt - b * T;
    int vi = blockIdx.y * blockDim.x + threadIdx.x;
    int vsz = nv * dv;
    if (b >= B || vi >= vsz) return;
    int qkC = 2 * nk * dk;
    int c = qkC + vi;
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j >= K) break;
        int token = t + j - (K - 1);
        float xv;
        if (token < 0) {
            xv = state[((size_t)b * (K - 1) + (token + K - 1)) * C + c];
        } else {
            xv = __half2float(v_in[((size_t)b * T + token) * vsz + vi]);
        }
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    if (has_bias) sum += bias[c];
    int h = vi / dv;
    int d = vi - h * dv;
    v_out[(((size_t)b * nv + h) * T + t) * dv + d] = silu_f32(sum);
}

__device__ __forceinline__ float linear_prefill_input(
    const __half* qk, const __half* v, int b, int t, int c,
    int T, int qkC, int vsz)
{
    if (c < qkC) return __half2float(qk[((size_t)b * T + t) * qkC + c]);
    return __half2float(v[((size_t)b * T + t) * vsz + (c - qkC)]);
}

__global__ void linear_conv_state_prefill_kernel(
    const float* __restrict__ old_state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    float* __restrict__ new_state,
    int B, int T, int qkC, int vsz, int C, int K)
{
    size_t total = (size_t)B * (K - 1) * C;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        int c = (int)(idx % C);
        size_t br = idx / C;
        int j = (int)(br % (K - 1));
        int b = (int)(br / (K - 1));
        int combined = T + j;
        if (combined < K - 1) {
            new_state[idx] = old_state[((size_t)b * (K - 1) + combined) * C + c];
        } else {
            int token = combined - (K - 1);
            new_state[idx] = linear_prefill_input(qk, v, b, token, c, T, qkC, vsz);
        }
    }
}

std::vector<torch::Tensor> linear_conv_qkv_prefill_cuda(
    torch::Tensor state,
    torch::Tensor qk,
    torch::Tensor v,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t nk,
    int64_t nv,
    int64_t dk,
    int64_t dv,
    double eps)
{
    TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == torch::kFloat32,
                "linear_conv_qkv_prefill: state must be contiguous CUDA f32");
    TORCH_CHECK(qk.is_cuda() && qk.is_contiguous() && qk.scalar_type() == torch::kHalf,
                "linear_conv_qkv_prefill: qk must be contiguous CUDA f16");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == torch::kHalf,
                "linear_conv_qkv_prefill: v must be contiguous CUDA f16");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kFloat32,
                "linear_conv_qkv_prefill: weight must be contiguous CUDA f32");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == torch::kFloat32,
                "linear_conv_qkv_prefill: bias must be contiguous CUDA f32");
    TORCH_CHECK(state.dim() == 3 && qk.dim() == 3 && v.dim() == 3,
                "linear_conv_qkv_prefill: state/qk/v must be rank 3");
    int B = (int)qk.size(0);
    int T = (int)qk.size(1);
    int K = (int)state.size(1) + 1;
    int qkC = (int)(2 * nk * dk);
    int vsz = (int)(nv * dv);
    int C = qkC + vsz;
    TORCH_CHECK(T > 1 && qk.size(2) == qkC && v.size(0) == B && v.size(1) == T && v.size(2) == vsz,
                "linear_conv_qkv_prefill: qk/v shape mismatch");
    TORCH_CHECK(state.size(0) == B && state.size(2) == C, "linear_conv_qkv_prefill: state shape mismatch");
    TORCH_CHECK(dk <= 256 && K <= 8, "linear_conv_qkv_prefill: requires dk<=256 and K<=8");
    int layout = 0;
    if (weight.dim() == 3) {
        TORCH_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "linear_conv_qkv_prefill: [C,1,K] weight mismatch");
    } else {
        TORCH_CHECK(weight.dim() == 2 && weight.size(0) == K && weight.size(1) == C,
                    "linear_conv_qkv_prefill: weight must be [C,1,K] or [K,C]");
        layout = 1;
    }
    int has_bias = (int)(bias.numel() > 0);
    TORCH_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "linear_conv_qkv_prefill: bias shape mismatch");

    auto opts = state.options();
    auto qo = torch::empty({B, (int)nk, T, (int)dk}, opts);
    auto ko = torch::empty_like(qo);
    auto vo = torch::empty({B, (int)nv, T, (int)dv}, opts);
    auto new_state = torch::empty_like(state);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    linear_conv_qk_norm_prefill_kernel<256><<<B * T * 2 * (int)nk, 256, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const __half*>(qk.data_ptr<at::Half>()),
        weight.data_ptr<float>(), bias.data_ptr<float>(), qo.data_ptr<float>(), ko.data_ptr<float>(),
        B, T, (int)nk, (int)dk, C, K, layout, has_bias, (float)eps);
    linear_conv_v_prefill_kernel<<<dim3(B * T, (vsz + 255) / 256), 256, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        weight.data_ptr<float>(), bias.data_ptr<float>(), vo.data_ptr<float>(),
        B, T, (int)nk, (int)nv, (int)dk, (int)dv, C, K, layout, has_bias);
    int state_total = B * (K - 1) * C;
    linear_conv_state_prefill_kernel<<<(state_total + 255) / 256, 256, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const __half*>(qk.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()), new_state.data_ptr<float>(),
        B, T, qkC, vsz, C, K);
    return {qo, ko, vo, new_state};
}
