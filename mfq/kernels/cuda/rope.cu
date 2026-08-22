// Rotary Position Embedding, rotate-half (HF/GPT-J style, Qwen convention).
// Supports full RoPE, partial RoPE, and MRoPE position sections.
// x [..., T, D], pos [T] or [A,T]. rotary_dim <= D.
// freq_j = base^(-2j/rotary_dim). One block per (m, t) row, threads cover j in [0, rotary_dim/2):
//   out[t,j]      = x0*cos - x1*sin
//   out[t,j+half] = x1*cos + x0*sin

#include <cuda_runtime.h>
#include "../../../cpp_runtime/cuda/mfq_tensor_backend.h"
#include <cuda_bf16.h>

constexpr int ROPE_BD = 256;

mfq_tensor_backend::Tensor rope_ext_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos, double base, int64_t rotary_dim, mfq_tensor_backend::Tensor sections);
mfq_tensor_backend::Tensor rope_table_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos, mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin,
                              int64_t rotary_dim, mfq_tensor_backend::Tensor sections);
mfq_tensor_backend::Tensor rope_table_bf16_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos,
                                   mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin,
                                   int64_t rotary_dim);
mfq_tensor_backend::Tensor minicpm_bf16_rope_cache_write_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor rope_pos, mfq_tensor_backend::Tensor write_pos,
    mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin,
    mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    int64_t rotary_dim);

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

__global__ void rope_table_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ pos,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    __nv_bfloat16* __restrict__ out,
    int rows, int T, int D, int rotary_dim, int table_len)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int t = row % T;
    const int half = rotary_dim / 2;
    const size_t offset = (size_t)row * D;
    int64_t p = pos[t];
    p = p < 0 ? 0 : (p >= table_len ? table_len - 1 : p);

    for (int i = threadIdx.x; i < D; i += ROPE_BD) {
        out[offset + i] = x[offset + i];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < half; j += ROPE_BD) {
        const __nv_bfloat16 cs = __float2bfloat16_rn(
            cos[(size_t)p * half + j]);
        const __nv_bfloat16 sn = __float2bfloat16_rn(
            sin[(size_t)p * half + j]);
        const __nv_bfloat16 x0 = x[offset + j];
        const __nv_bfloat16 x1 = x[offset + j + half];
        const __nv_bfloat16 x0c = __float2bfloat16_rn(
            __bfloat162float(x0) * __bfloat162float(cs));
        const __nv_bfloat16 x1s = __float2bfloat16_rn(
            __bfloat162float(x1) * __bfloat162float(sn));
        const __nv_bfloat16 x1c = __float2bfloat16_rn(
            __bfloat162float(x1) * __bfloat162float(cs));
        const __nv_bfloat16 x0s = __float2bfloat16_rn(
            __bfloat162float(x0) * __bfloat162float(sn));
        out[offset + j] = __float2bfloat16_rn(
            __bfloat162float(x0c) - __bfloat162float(x1s));
        out[offset + j + half] = __float2bfloat16_rn(
            __bfloat162float(x1c) + __bfloat162float(x0s));
    }
}

__global__ void minicpm_bf16_rope_cache_write_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int64_t* __restrict__ rope_pos,
    const int64_t* __restrict__ write_pos,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int Hq, int Hk, int D, int half, int table_len, int max_seq)
{
    const int row = blockIdx.x;
    const int hq = row % Hq;
    const int b = row / Hq;
    const int tid = threadIdx.x;
    int64_t p = rope_pos[0];
    p = p < 0 ? 0 : (p >= table_len ? table_len - 1 : p);
    const int64_t wp = write_pos[0];

    const size_t q_offset = (size_t)row * D;
    if (tid < half) {
        const __nv_bfloat16 cs = __float2bfloat16_rn(
            cos[(size_t)p * half + tid]);
        const __nv_bfloat16 sn = __float2bfloat16_rn(
            sin[(size_t)p * half + tid]);
        const __nv_bfloat16 x0 = q[q_offset + tid];
        const __nv_bfloat16 x1 = q[q_offset + tid + half];
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
        const size_t src = ((size_t)b * Hk + hq) * D;
        const size_t dst = (((size_t)b * Hk + hq) * max_seq + wp) * D;
        if (tid < half) {
            const __nv_bfloat16 cs = __float2bfloat16_rn(
                cos[(size_t)p * half + tid]);
            const __nv_bfloat16 sn = __float2bfloat16_rn(
                sin[(size_t)p * half + tid]);
            const __nv_bfloat16 x0 = k[src + tid];
            const __nv_bfloat16 x1 = k[src + tid + half];
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

mfq_tensor_backend::Tensor rope_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos, double base)
{
    return rope_ext_cuda(
        x, pos, base, x.size(-1),
        mfq_tensor_backend::empty({0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCPU)));
}

mfq_tensor_backend::Tensor rope_ext_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos, double base, int64_t rotary_dim, mfq_tensor_backend::Tensor sections)
{
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kFloat32,
                "rope: x must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(pos.is_cuda() && pos.is_contiguous() && pos.scalar_type() == mfq_tensor_backend::kFloat32,
                "rope: pos must be cuda contiguous f32");
    int T = (int)x.size(-2);
    int D = (int)x.size(-1);
    int RD = (int)rotary_dim;
    MFQ_RUNTIME_CHECK(RD > 0 && RD <= D && RD % 2 == 0, "rope: rotary_dim must be positive even and <= D");
    MFQ_RUNTIME_CHECK(pos.dim() == 1 || pos.dim() == 2, "rope: pos must be [T] or [A,T]");
    int pos_axes = pos.dim() == 1 ? 1 : (int)pos.size(0);
    MFQ_RUNTIME_CHECK(pos.size(-1) == T, "rope: pos last dim must match T");
    int s0 = 0, s1 = 0, s2 = 0;
    if (sections.numel() > 0) {
        MFQ_RUNTIME_CHECK(!sections.is_cuda() && sections.is_contiguous() && sections.scalar_type() == mfq_tensor_backend::kInt64,
                    "rope: sections must be CPU contiguous int64");
        MFQ_RUNTIME_CHECK(sections.numel() == 3, "rope: sections must have 3 entries");
        const int64_t* sp = sections.data_ptr<int64_t>();
        s0 = (int)sp[0];
        s1 = (int)sp[1];
        s2 = (int)sp[2];
        MFQ_RUNTIME_CHECK(s0 + s1 + s2 == RD / 2, "rope: sections must sum to rotary_dim/2");
    }
    int MT = (int)(x.numel() / ((size_t)T * D));
    auto out = mfq_tensor_backend::empty_like(x);
    rope_kernel<<<MT * T, ROPE_BD, 0, mfq_current_cuda_stream()>>>(
        x.data_ptr<float>(), pos.data_ptr<float>(), out.data_ptr<float>(),
        MT * T, T, D, RD, pos_axes, s0, s1, s2, (float)base);
    return out;
}

mfq_tensor_backend::Tensor rope_table_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos, mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin,
                              int64_t rotary_dim, mfq_tensor_backend::Tensor sections)
{
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kFloat32,
                "rope_table: x must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(pos.is_cuda() && pos.is_contiguous() && pos.scalar_type() == mfq_tensor_backend::kInt64,
                "rope_table: pos must be cuda contiguous int64");
    MFQ_RUNTIME_CHECK(cos.is_cuda() && cos.is_contiguous() && cos.scalar_type() == mfq_tensor_backend::kFloat32,
                "rope_table: cos must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(sin.is_cuda() && sin.is_contiguous() && sin.scalar_type() == mfq_tensor_backend::kFloat32,
                "rope_table: sin must be cuda contiguous f32");
    int T = (int)x.size(-2);
    int D = (int)x.size(-1);
    int RD = (int)rotary_dim;
    int half = RD / 2;
    MFQ_RUNTIME_CHECK(RD > 0 && RD <= D && RD % 2 == 0, "rope_table: rotary_dim must be positive even and <= D");
    MFQ_RUNTIME_CHECK(pos.dim() == 1 || pos.dim() == 2, "rope_table: pos must be [T] or [A,T]");
    int pos_axes = pos.dim() == 1 ? 1 : (int)pos.size(0);
    MFQ_RUNTIME_CHECK(pos.size(-1) == T, "rope_table: pos last dim must match T");
    MFQ_RUNTIME_CHECK(cos.dim() == 2 && sin.dim() == 2 && cos.sizes() == sin.sizes(),
                "rope_table: cos/sin must be [table_len, rotary_dim/2]");
    MFQ_RUNTIME_CHECK(cos.size(1) == half, "rope_table: cos/sin width mismatch");
    int table_len = (int)cos.size(0);
    int s0 = 0, s1 = 0, s2 = 0;
    if (sections.numel() > 0) {
        MFQ_RUNTIME_CHECK(!sections.is_cuda() && sections.is_contiguous() && sections.scalar_type() == mfq_tensor_backend::kInt64,
                    "rope_table: sections must be CPU contiguous int64");
        MFQ_RUNTIME_CHECK(sections.numel() == 3, "rope_table: sections must have 3 entries");
        const int64_t* sp = sections.data_ptr<int64_t>();
        s0 = (int)sp[0];
        s1 = (int)sp[1];
        s2 = (int)sp[2];
        MFQ_RUNTIME_CHECK(s0 + s1 + s2 == half, "rope_table: sections must sum to rotary_dim/2");
    }
    int MT = (int)(x.numel() / ((size_t)T * D));
    auto out = mfq_tensor_backend::empty_like(x);
    rope_table_kernel<<<MT * T, ROPE_BD, 0, mfq_current_cuda_stream()>>>(
        x.data_ptr<float>(), pos.data_ptr<int64_t>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
        out.data_ptr<float>(), MT * T, T, D, RD, table_len, pos_axes, s0, s1, s2);
    return out;
}

mfq_tensor_backend::Tensor rope_table_bf16_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos,
                                   mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin,
                                   int64_t rotary_dim)
{
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() &&
                    x.scalar_type() == mfq_tensor_backend::kBFloat16,
                "rope_table_bf16: x must be cuda contiguous bf16");
    MFQ_RUNTIME_CHECK(pos.is_cuda() && pos.is_contiguous() &&
                    pos.scalar_type() == mfq_tensor_backend::kInt64 && pos.dim() == 1,
                "rope_table_bf16: pos must be cuda contiguous int64 [T]");
    MFQ_RUNTIME_CHECK(cos.is_cuda() && cos.is_contiguous() &&
                    cos.scalar_type() == mfq_tensor_backend::kFloat32,
                "rope_table_bf16: cos must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(sin.is_cuda() && sin.is_contiguous() &&
                    sin.scalar_type() == mfq_tensor_backend::kFloat32 &&
                    cos.sizes() == sin.sizes(),
                "rope_table_bf16: sin must match cos");
    const int T = (int)x.size(-2);
    const int D = (int)x.size(-1);
    const int RD = (int)rotary_dim;
    MFQ_RUNTIME_CHECK(pos.numel() == T,
                "rope_table_bf16: position count must match T");
    MFQ_RUNTIME_CHECK(RD > 0 && RD <= D && RD % 2 == 0 &&
                    cos.dim() == 2 && cos.size(1) == RD / 2,
                "rope_table_bf16: invalid rotary geometry");
    const int rows = (int)(x.numel() / D);
    auto out = mfq_tensor_backend::empty_like(x);
    rope_table_bf16_kernel<<<
        rows, ROPE_BD, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            x.data_ptr<mfq_bfloat16>()),
        pos.data_ptr<int64_t>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(
            out.data_ptr<mfq_bfloat16>()),
        rows, T, D, RD, (int)cos.size(0));
    return out;
}

mfq_tensor_backend::Tensor minicpm_bf16_rope_cache_write_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor rope_pos, mfq_tensor_backend::Tensor write_pos,
    mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin,
    mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    int64_t rotary_dim)
{
    const auto bf16 = mfq_tensor_backend::kBFloat16;
    MFQ_RUNTIME_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() &&
                    q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                    q.scalar_type() == bf16 && k.scalar_type() == bf16 &&
                    v.scalar_type() == bf16,
                "minicpm_rope_kv: q/k/v must be cuda contiguous bf16");
    MFQ_RUNTIME_CHECK(k_cache.is_cuda() && v_cache.is_cuda() &&
                    k_cache.is_contiguous() && v_cache.is_contiguous() &&
                    k_cache.scalar_type() == bf16 && v_cache.scalar_type() == bf16,
                "minicpm_rope_kv: caches must be cuda contiguous bf16");
    MFQ_RUNTIME_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 &&
                    k_cache.dim() == 4 && v_cache.dim() == 4,
                "minicpm_rope_kv: tensors must be rank four");
    MFQ_RUNTIME_CHECK(k.sizes() == v.sizes() && k_cache.sizes() == v_cache.sizes(),
                "minicpm_rope_kv: k/v shapes must match");
    MFQ_RUNTIME_CHECK(q.size(0) == k.size(0) && q.size(2) == 1 && k.size(2) == 1 &&
                    q.size(1) == 32 && k.size(1) == 8 &&
                    q.size(3) == 128 && k.size(3) == 128,
                "minicpm_rope_kv: expected Bx32x1x128 Q and Bx8x1x128 K/V");
    MFQ_RUNTIME_CHECK(k_cache.size(0) == k.size(0) &&
                    k_cache.size(1) == k.size(1) &&
                    k_cache.size(3) == k.size(3),
                "minicpm_rope_kv: cache shape mismatch");
    MFQ_RUNTIME_CHECK(rope_pos.is_cuda() && write_pos.is_cuda() &&
                    rope_pos.is_contiguous() && write_pos.is_contiguous() &&
                    rope_pos.scalar_type() == mfq_tensor_backend::kInt64 &&
                    write_pos.scalar_type() == mfq_tensor_backend::kInt64 &&
                    rope_pos.numel() == 1 && write_pos.numel() == 1,
                "minicpm_rope_kv: positions must be cuda contiguous int64[1]");
    MFQ_RUNTIME_CHECK(cos.is_cuda() && sin.is_cuda() &&
                    cos.is_contiguous() && sin.is_contiguous() &&
                    cos.scalar_type() == mfq_tensor_backend::kFloat32 &&
                    sin.scalar_type() == mfq_tensor_backend::kFloat32 &&
                    cos.sizes() == sin.sizes() && cos.dim() == 2,
                "minicpm_rope_kv: cos/sin must be matching cuda contiguous f32 tables");
    MFQ_RUNTIME_CHECK(rotary_dim == 128 && cos.size(1) == 64,
                "minicpm_rope_kv: expected rotary_dim 128");

    auto q_out = mfq_tensor_backend::empty_like(q);
    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int Hk = (int)k.size(1);
    minicpm_bf16_rope_cache_write_kernel<<<
        B * Hq, 128, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<mfq_bfloat16>()),
        rope_pos.data_ptr<int64_t>(), write_pos.data_ptr<int64_t>(),
        cos.data_ptr<float>(), sin.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr<mfq_bfloat16>()),
        reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr<mfq_bfloat16>()),
        Hq, Hk, 128, 64, (int)cos.size(0), (int)k_cache.size(2));
    return q_out;
}
