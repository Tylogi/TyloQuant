// Rotary Position Embedding, rotate-half (HF/GPT-J style, Qwen convention).
// Supports full RoPE, partial RoPE, and MRoPE position sections.
// x [..., T, D], pos [T] or [A,T]. rotary_dim <= D.
// freq_j = base^(-2j/rotary_dim). One block per (m, t) row, threads cover j in [0, rotary_dim/2):
//   out[t,j]      = x0*cos - x1*sin
//   out[t,j+half] = x1*cos + x0*sin

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

constexpr int ROPE_BD = 256;

torch::Tensor rope_ext_cuda(torch::Tensor x, torch::Tensor pos, double base, int64_t rotary_dim, torch::Tensor sections);
torch::Tensor rope_table_cuda(torch::Tensor x, torch::Tensor pos, torch::Tensor cos, torch::Tensor sin,
                              int64_t rotary_dim, torch::Tensor sections);

__device__ int rope_axis_for_pair(int j, int s0, int s1, int s2)
{
    if (s0 <= 0 && s1 <= 0 && s2 <= 0) {
        return 0;
    }
    if (j < s0) {
        return 0;
    }
    if (j < s0 + s1) {
        return 1;
    }
    return 2;
}

__global__ void rope_kernel(const float* __restrict__ x, const float* __restrict__ pos,
                            float* __restrict__ out, int MT, int T, int D, int rotary_dim,
                            int pos_axes, int s0, int s1, int s2, float base)
{
    int mt = blockIdx.x;
    if (mt >= MT) {
        return;
    }
    int m = mt / T;
    int t = mt % T;
    int half = rotary_dim / 2;
    size_t base0 = ((size_t)m * T + t) * D;
    int tid = threadIdx.x;

    for (int i = tid; i < D; i += ROPE_BD) {
        out[base0 + i] = x[base0 + i];
    }
    __syncthreads();

    for (int j = tid; j < half; j += ROPE_BD) {
        int axis = rope_axis_for_pair(j, s0, s1, s2);
        if (axis >= pos_axes) {
            axis = 0;
        }
        float p = pos_axes == 1 ? pos[t] : pos[(size_t)axis * T + t];
        float freq = powf(base, -2.0f * (float)j / (float)rotary_dim);
        float ang = p * freq;
        float cs = cosf(ang);
        float sn = sinf(ang);
        float x0 = x[base0 + j];
        float x1 = x[base0 + j + half];
        out[base0 + j] = x0 * cs - x1 * sn;
        out[base0 + j + half] = x1 * cs + x0 * sn;
    }
}

__global__ void rope_table_kernel(const float* __restrict__ x, const int64_t* __restrict__ pos,
                                  const float* __restrict__ cos, const float* __restrict__ sin,
                                  float* __restrict__ out, int MT, int T, int D, int rotary_dim,
                                  int table_len, int pos_axes, int s0, int s1, int s2)
{
    int mt = blockIdx.x;
    if (mt >= MT) {
        return;
    }
    int m = mt / T;
    int t = mt % T;
    int half = rotary_dim / 2;
    size_t base0 = ((size_t)m * T + t) * D;
    int tid = threadIdx.x;

    for (int i = tid; i < D; i += ROPE_BD) {
        out[base0 + i] = x[base0 + i];
    }
    __syncthreads();

    for (int j = tid; j < half; j += ROPE_BD) {
        int axis = rope_axis_for_pair(j, s0, s1, s2);
        if (axis >= pos_axes) {
            axis = 0;
        }
        int64_t p = pos_axes == 1 ? pos[t] : pos[(size_t)axis * T + t];
        if (p < 0) {
            p = 0;
        }
        if (p >= table_len) {
            p = table_len - 1;
        }
        float cs = cos[(size_t)p * half + j];
        float sn = sin[(size_t)p * half + j];
        float x0 = x[base0 + j];
        float x1 = x[base0 + j + half];
        out[base0 + j] = x0 * cs - x1 * sn;
        out[base0 + j + half] = x1 * cs + x0 * sn;
    }
}

torch::Tensor rope_cuda(torch::Tensor x, torch::Tensor pos, double base)
{
    return rope_ext_cuda(
        x, pos, base, x.size(-1),
        torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)));
}

torch::Tensor rope_ext_cuda(torch::Tensor x, torch::Tensor pos, double base, int64_t rotary_dim, torch::Tensor sections)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32,
                "rope: x must be cuda contiguous f32");
    TORCH_CHECK(pos.is_cuda() && pos.is_contiguous() && pos.scalar_type() == torch::kFloat32,
                "rope: pos must be cuda contiguous f32");
    int T = (int)x.size(-2);
    int D = (int)x.size(-1);
    int RD = (int)rotary_dim;
    TORCH_CHECK(RD > 0 && RD <= D && RD % 2 == 0, "rope: rotary_dim must be positive even and <= D");
    TORCH_CHECK(pos.dim() == 1 || pos.dim() == 2, "rope: pos must be [T] or [A,T]");
    int pos_axes = pos.dim() == 1 ? 1 : (int)pos.size(0);
    TORCH_CHECK(pos.size(-1) == T, "rope: pos last dim must match T");
    int s0 = 0, s1 = 0, s2 = 0;
    if (sections.numel() > 0) {
        TORCH_CHECK(!sections.is_cuda() && sections.is_contiguous() && sections.scalar_type() == torch::kInt64,
                    "rope: sections must be CPU contiguous int64");
        TORCH_CHECK(sections.numel() == 3, "rope: sections must have 3 entries");
        const int64_t* sp = sections.data_ptr<int64_t>();
        s0 = (int)sp[0];
        s1 = (int)sp[1];
        s2 = (int)sp[2];
        TORCH_CHECK(s0 + s1 + s2 == RD / 2, "rope: sections must sum to rotary_dim/2");
    }
    int MT = (int)(x.numel() / ((size_t)T * D));
    auto out = torch::empty_like(x);
    rope_kernel<<<MT * T, ROPE_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), pos.data_ptr<float>(), out.data_ptr<float>(),
        MT * T, T, D, RD, pos_axes, s0, s1, s2, (float)base);
    return out;
}

torch::Tensor rope_table_cuda(torch::Tensor x, torch::Tensor pos, torch::Tensor cos, torch::Tensor sin,
                              int64_t rotary_dim, torch::Tensor sections)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32,
                "rope_table: x must be cuda contiguous f32");
    TORCH_CHECK(pos.is_cuda() && pos.is_contiguous() && pos.scalar_type() == torch::kInt64,
                "rope_table: pos must be cuda contiguous int64");
    TORCH_CHECK(cos.is_cuda() && cos.is_contiguous() && cos.scalar_type() == torch::kFloat32,
                "rope_table: cos must be cuda contiguous f32");
    TORCH_CHECK(sin.is_cuda() && sin.is_contiguous() && sin.scalar_type() == torch::kFloat32,
                "rope_table: sin must be cuda contiguous f32");
    int T = (int)x.size(-2);
    int D = (int)x.size(-1);
    int RD = (int)rotary_dim;
    int half = RD / 2;
    TORCH_CHECK(RD > 0 && RD <= D && RD % 2 == 0, "rope_table: rotary_dim must be positive even and <= D");
    TORCH_CHECK(pos.dim() == 1 || pos.dim() == 2, "rope_table: pos must be [T] or [A,T]");
    int pos_axes = pos.dim() == 1 ? 1 : (int)pos.size(0);
    TORCH_CHECK(pos.size(-1) == T, "rope_table: pos last dim must match T");
    TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2 && cos.sizes() == sin.sizes(),
                "rope_table: cos/sin must be [table_len, rotary_dim/2]");
    TORCH_CHECK(cos.size(1) == half, "rope_table: cos/sin width mismatch");
    int table_len = (int)cos.size(0);
    int s0 = 0, s1 = 0, s2 = 0;
    if (sections.numel() > 0) {
        TORCH_CHECK(!sections.is_cuda() && sections.is_contiguous() && sections.scalar_type() == torch::kInt64,
                    "rope_table: sections must be CPU contiguous int64");
        TORCH_CHECK(sections.numel() == 3, "rope_table: sections must have 3 entries");
        const int64_t* sp = sections.data_ptr<int64_t>();
        s0 = (int)sp[0];
        s1 = (int)sp[1];
        s2 = (int)sp[2];
        TORCH_CHECK(s0 + s1 + s2 == half, "rope_table: sections must sum to rotary_dim/2");
    }
    int MT = (int)(x.numel() / ((size_t)T * D));
    auto out = torch::empty_like(x);
    rope_table_kernel<<<MT * T, ROPE_BD, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), pos.data_ptr<int64_t>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
        out.data_ptr<float>(), MT * T, T, D, RD, table_len, pos_axes, s0, s1, s2);
    return out;
}
