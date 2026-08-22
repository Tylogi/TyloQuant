// Full / Grouped-Query Attention (ggml fattn.cu).
//
//   out[b,hq,tq,:] = softmax_tq( (q . k^T) * scale ) . v
//
// One block per query vector (b, hq, tq); blockDim = warp-rounded to >= D. q row in shared
// mem; per key s, online softmax (flash-v1 single-query): running max m, norm-sum l,
// accumulator O[tid]. One block_sum reduces the q . k_s dot and returns it to all threads.
// causal mask follows torch SDPA for Tq != Tk: query tq sees key s iff s <= tq + (Tk - T).
// q/k/v may be fp16 or fp32; dot/softmax accumulation stays fp32.

#include <cuda_runtime.h>
#include "../../../cpp_runtime/cuda/mfq_tensor_backend.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cfloat>
#include <climits>
#include <cstdlib>

#include "reduce.cuh"

namespace wmma = nvcuda::wmma;

// Qwen3.5 full-attention prefill: FP16 GQA, head_dim=256, causal self-attention.
// QK is recomputed in the second pass so the full T x T score matrix is never materialized.
__global__ void attention_flash256_f16_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ out,
    int Hq, int Hk, int T, float scale)
{
    constexpr int D = 256;
    constexpr int QT = 128;
    constexpr int KT = 16;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int q0 = blockIdx.x * QT;
    const int bhq = blockIdx.y;
    const int hq = bhq % Hq;
    const int b = bhq / Hq;
    const int hk = hq / (Hq / Hk);

    extern __shared__ __align__(16) unsigned char smem[];
    half* q_s = reinterpret_cast<half*>(smem);
    half* kv_s = q_s + QT * D;
    float* scores = reinterpret_cast<float*>(kv_s + KT * D);
    half* probs = reinterpret_cast<half*>(scores + QT * KT);
    float* row_max = reinterpret_cast<float*>(probs + QT * KT);
    float* row_sum = row_max + QT;

    const size_t q_base = (size_t)bhq * T * D;
    const size_t kv_base = ((size_t)b * Hk + hk) * T * D;
    for (int i = tid; i < QT * D; i += blockDim.x) {
        const int qr = i / D;
        const int d = i - qr * D;
        q_s[i] = q0 + qr < T ? q[q_base + (size_t)(q0 + qr) * D + d] : __float2half(0.0f);
    }
    if (tid < QT) {
        row_max[tid] = -FLT_MAX;
        row_sum[tid] = 0.0f;
    }
    __syncthreads();

    for (int k0 = 0; k0 < T; k0 += KT) {
        for (int i = tid; i < KT * D; i += blockDim.x) {
            const int kr = i / D;
            const int d = i - kr * D;
            kv_s[i] = k0 + kr < T ? k[kv_base + (size_t)(k0 + kr) * D + d] : __float2half(0.0f);
        }
        __syncthreads();

        if (warp < 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            const int qr = warp;
            for (int d0 = 0; d0 < D; d0 += 16) {
                wmma::load_matrix_sync(a_frag, q_s + qr * 16 * D + d0, D);
                wmma::load_matrix_sync(b_frag, kv_s + d0, D);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores + qr * 16 * KT, c_frag, KT, wmma::mem_row_major);
        }
        __syncthreads();

        if (warp < 8) {
        #pragma unroll
        for (int rr = 0; rr < 16; ++rr) {
            const int r = warp * 16 + rr;
            const int qg = q0 + r;
            if (qg >= T) continue;
            const int kg = k0 + lane;
            float s = lane < KT && kg < T && kg <= qg ? scores[r * KT + lane] * scale : -FLT_MAX;
            const float tile_max = warp_max(s);
            const float old_max = row_max[r];
            const float new_max = fmaxf(old_max, tile_max);
            float add = s == -FLT_MAX ? 0.0f : expf(s - new_max);
            add = warp_sum(add);
            if (lane == 0) {
                const float old_scaled = old_max == -FLT_MAX ? 0.0f : row_sum[r] * expf(old_max - new_max);
                row_max[r] = new_max;
                row_sum[r] = old_scaled + add;
            }
        }
        }
        __syncthreads();
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) wmma::fill_fragment(o_frag[i], 0.0f);

    for (int k0 = 0; k0 < T; k0 += KT) {
        for (int i = tid; i < KT * D; i += blockDim.x) {
            const int kr = i / D;
            const int d = i - kr * D;
            kv_s[i] = k0 + kr < T ? k[kv_base + (size_t)(k0 + kr) * D + d] : __float2half(0.0f);
        }
        __syncthreads();

        if (warp < 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            const int qr = warp;
            for (int d0 = 0; d0 < D; d0 += 16) {
                wmma::load_matrix_sync(a_frag, q_s + qr * 16 * D + d0, D);
                wmma::load_matrix_sync(b_frag, kv_s + d0, D);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores + qr * 16 * KT, c_frag, KT, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < QT * KT; i += blockDim.x) {
            const int r = i / KT;
            const int c = i - r * KT;
            const int qg = q0 + r;
            const int kg = k0 + c;
            float p = 0.0f;
            if (qg < T && kg < T && kg <= qg && row_sum[r] > 0.0f) {
                p = expf(scores[i] * scale - row_max[r]) / row_sum[r];
            }
            probs[i] = __float2half_rn(p);
        }
        __syncthreads();

        for (int i = tid; i < D * KT; i += blockDim.x) {
            const int d = i / KT;
            const int kr = i - d * KT;
            kv_s[i] = k0 + kr < T ? v[kv_base + (size_t)(k0 + kr) * D + d] : __float2half(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> v_frag;
        const int qr = warp >> 1;
        const int parity = warp & 1;
        for (int kk = 0; kk < KT; kk += 16) {
            wmma::load_matrix_sync(p_frag, probs + qr * 16 * KT + kk, KT);
            #pragma unroll
            for (int fi = 0; fi < 8; ++fi) {
                const int ni = parity + fi * 2;
                wmma::load_matrix_sync(v_frag, kv_s + ni * 16 * KT + kk, KT);
                wmma::mma_sync(o_frag[fi], p_frag, v_frag, o_frag[fi]);
            }
        }
        __syncthreads();
    }

    float* out_s = reinterpret_cast<float*>(smem) + warp * 256;
    const int qr = warp >> 1;
    const int parity = warp & 1;
    #pragma unroll
    for (int fi = 0; fi < 8; ++fi) {
        const int ni = parity + fi * 2;
        wmma::store_matrix_sync(out_s, o_frag[fi], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int r = e / 16;
            const int c = e - r * 16;
            if (q0 + qr * 16 + r < T) {
                out[q_base + (size_t)(q0 + qr * 16 + r) * D + ni * 16 + c] = __float2half_rn(out_s[e]);
            }
        }
        __syncwarp();
    }
}

mfq_tensor_backend::Tensor attention_flash256_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v, double scale)
{
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == mfq_tensor_backend::kFloat16,
                "attention_flash256: q must be contiguous CUDA fp16");
    MFQ_RUNTIME_CHECK(k.is_cuda() && k.is_contiguous() && k.scalar_type() == mfq_tensor_backend::kFloat16,
                "attention_flash256: k must be contiguous CUDA fp16");
    MFQ_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == mfq_tensor_backend::kFloat16,
                "attention_flash256: v must be contiguous CUDA fp16");
    MFQ_RUNTIME_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "attention_flash256: q/k/v must be rank 4");
    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int T = (int)q.size(2);
    const int Hk = (int)k.size(1);
    MFQ_RUNTIME_CHECK(q.size(3) == 256 && k.size(3) == 256 && v.size(3) == 256,
                "attention_flash256: head_dim must be 256");
    MFQ_RUNTIME_CHECK(k.size(0) == B && v.size(0) == B && k.size(2) == T && v.size(2) == T && v.size(1) == Hk,
                "attention_flash256: only self-attention with matching k/v shapes is supported");
    MFQ_RUNTIME_CHECK(Hq % Hk == 0, "attention_flash256: Hq must be divisible by Hkv");
    auto out = mfq_tensor_backend::empty_like(q);
    const dim3 grid((T + 127) / 128, B * Hq);
    constexpr int shmem = 96 * 1024;
    cudaFuncSetAttribute(attention_flash256_f16_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    attention_flash256_f16_kernel<<<grid, 512, shmem, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(q.data_ptr<mfq_half>()),
        reinterpret_cast<const half*>(k.data_ptr<mfq_half>()),
        reinterpret_cast<const half*>(v.data_ptr<mfq_half>()),
        reinterpret_cast<half*>(out.data_ptr<mfq_half>()),
        Hq, Hk, T, (float)scale);
    return out;
}

template <int BD, typename scalar_t>
__global__ void attention_kernel(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k, const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int B, int Hq, int Hk, int T, int Tk, int D, int rep, float scale, int causal, int window)
{
    constexpr int NW = BD / 32;
    int qv = blockIdx.x;
    int tq = qv % T;
    int rem = qv / T;
    int hq = rem % Hq;
    int b = rem / Hq;
    int hk = hq / rep;
    int tid = threadIdx.x;

    extern __shared__ float buf[];
    float* q_s = buf;        // [D]

    if (tid < D) {
        q_s[tid] = (float)q[((size_t)b * Hq + hq) * T * D + (size_t)tq * D + tid];
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float O = 0.0f;
    const int offset = Tk - T;
    const int end = causal ? min(Tk, tq + offset + 1) : Tk;
    const int start = window > 0 ? max(0, end - window) : 0;

    for (int s = start; s < end; ++s) {
        float kd = (tid < D) ? (float)k[((size_t)b * Hk + hk) * Tk * D + (size_t)s * D + tid] : 0.0f;
        float dot = (tid < D) ? q_s[tid] * kd : 0.0f;
        dot = block_sum<NW>(dot);

        float score = dot * scale;
        float m_new = fmaxf(m, score);
        float fac = expf(m - m_new);
        float p = expf(score - m_new);
        float vd = (tid < D) ? (float)v[((size_t)b * Hk + hk) * Tk * D + (size_t)s * D + tid] : 0.0f;
        O = O * fac + p * vd;
        l = l * fac + p;
        m = m_new;
    }

    if (tid < D) {
        float lo = (l > 0.0f) ? l : 1.0f;
        out[((size_t)b * Hq + hq) * T * D + (size_t)tq * D + tid] = (scalar_t)(O / lo);
    }
}

template <int BD, typename scalar_t>
__global__ void attention_split_part_kernel(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k, const scalar_t* __restrict__ v,
    float* __restrict__ partial_o, float* __restrict__ partial_m, float* __restrict__ partial_l,
    int B, int Hq, int Hk, int T, int Tk, int D, int rep, int parts,
    float scale, int causal, int window)
{
    constexpr int NW = BD / 32;
    int part = blockIdx.x % parts;
    int qv = blockIdx.x / parts;
    int tq = qv % T;
    int rem = qv / T;
    int hq = rem % Hq;
    int b = rem / Hq;
    int hk = hq / rep;
    int tid = threadIdx.x;
    const int offset = Tk - T;
    const int visible_end = causal ? min(Tk, tq + offset + 1) : Tk;
    const int visible_start = window > 0 ? max(0, visible_end - window) : 0;
    const int visible = visible_end - visible_start;
    const int start = visible_start + (int)(((int64_t)visible * part) / parts);
    const int end = visible_start + (int)(((int64_t)visible * (part + 1)) / parts);

    extern __shared__ float q_s[];
    if (tid < D) {
        q_s[tid] = (float)q[((size_t)b * Hq + hq) * T * D + (size_t)tq * D + tid];
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float O = 0.0f;
    int seen = 0;

    for (int s = start; s < end; ++s) {
        float kd = (tid < D) ? (float)k[((size_t)b * Hk + hk) * Tk * D + (size_t)s * D + tid] : 0.0f;
        float dot = (tid < D) ? q_s[tid] * kd : 0.0f;
        dot = block_sum<NW>(dot);
        float score = dot * scale;
        float m_new = fmaxf(m, score);
        float fac = expf(m - m_new);
        float p = expf(score - m_new);
        float vd = (tid < D) ? (float)v[((size_t)b * Hk + hk) * Tk * D + (size_t)s * D + tid] : 0.0f;
        O = O * fac + p * vd;
        l = l * fac + p;
        m = m_new;
        seen = 1;
    }

    size_t po = ((size_t)qv * parts + part) * D;
    if (tid < D) {
        partial_o[po + tid] = seen ? O : 0.0f;
    }
    if (tid == 0) {
        partial_m[(size_t)qv * parts + part] = seen ? m : -1e30f;
        partial_l[(size_t)qv * parts + part] = seen ? l : 0.0f;
    }
}

template <int BD, typename scalar_t>
__global__ void attention_split_reduce_kernel(
    const float* __restrict__ partial_o, const float* __restrict__ partial_m, const float* __restrict__ partial_l,
    scalar_t* __restrict__ out, int total, int parts, int D)
{
    int qv = blockIdx.x;
    int tid = threadIdx.x;
    if (qv >= total) {
        return;
    }
    float m = -1e30f;
    for (int p = 0; p < parts; ++p) {
        m = fmaxf(m, partial_m[(size_t)qv * parts + p]);
    }
    float l = 0.0f;
    float O = 0.0f;
    for (int p = 0; p < parts; ++p) {
        float lp = partial_l[(size_t)qv * parts + p];
        float w = lp > 0.0f ? expf(partial_m[(size_t)qv * parts + p] - m) : 0.0f;
        l += w * lp;
        if (tid < D) {
            O += w * partial_o[((size_t)qv * parts + p) * D + tid];
        }
    }
    if (tid < D) {
        float lo = (l > 0.0f) ? l : 1.0f;
        out[(size_t)qv * D + tid] = (scalar_t)(O / lo);
    }
}

static mfq_tensor_backend::Tensor attention_impl_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
                                         double scale, bool causal, int window)
{
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous(), "attention: q must be cuda contiguous");
    MFQ_RUNTIME_CHECK(k.is_cuda() && k.is_contiguous(), "attention: k must be cuda contiguous");
    MFQ_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous(), "attention: v must be cuda contiguous");
    MFQ_RUNTIME_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "attention: q/k/v dtype mismatch");
    MFQ_RUNTIME_CHECK(q.scalar_type() == mfq_tensor_backend::kFloat16 || q.scalar_type() == mfq_tensor_backend::kFloat32,
                "attention: dtype must be f16 or f32");
    int B = (int)q.size(0), Hq = (int)q.size(1), T = (int)q.size(2), D = (int)q.size(3);
    int Hk = (int)k.size(1), Tk = (int)k.size(2);
    MFQ_RUNTIME_CHECK(D == (int)k.size(3) && D == (int)v.size(3), "q/k/v last dim must match");
    MFQ_RUNTIME_CHECK(Hq % Hk == 0, "GQA requires H_q % H_kv == 0");
    int rep = Hq / Hk;
    auto out = mfq_tensor_backend::empty_like(q);
    int total = B * Hq * T;
    int shmem = D * (int)sizeof(float);
    cudaStream_t stream = mfq_current_cuda_stream();
    int parts = 1;
    const int visible_keys = window > 0 ? std::min(Tk, window) : Tk;
    const char* split_env = std::getenv("MFQ_ATTENTION_SPLITK");
    if (total < 128 && visible_keys >= 512 && !(split_env && split_env[0] == '0')) {
        parts = (visible_keys + 255) / 256;
        parts = parts < 2 ? 1 : parts;
        parts = parts > 16 ? 16 : parts;
    }

#define ATT(BD) attention_kernel<BD, scalar_t><<<total, BD, shmem, stream>>>(                       \
    q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), \
    B, Hq, Hk, T, Tk, D, rep, (float)scale, (int)causal, window)
#define ATT_SPLIT(BD)                                                                                 \
    do {                                                                                              \
        auto po = mfq_tensor_backend::empty({total, parts, D}, q.options().dtype(mfq_tensor_backend::kFloat32));                 \
        auto pm = mfq_tensor_backend::empty({total, parts}, q.options().dtype(mfq_tensor_backend::kFloat32));                    \
        auto pl = mfq_tensor_backend::empty({total, parts}, q.options().dtype(mfq_tensor_backend::kFloat32));                    \
        attention_split_part_kernel<BD, scalar_t><<<total * parts, BD, shmem, stream>>>(               \
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),                    \
            po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),                          \
            B, Hq, Hk, T, Tk, D, rep, parts, (float)scale, (int)causal, window);                       \
        attention_split_reduce_kernel<BD, scalar_t><<<total, BD, 0, stream>>>(                         \
            po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),                          \
            out.data_ptr<scalar_t>(), total, parts, D);                                                \
    } while (0)
    MFQ_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "attention_cuda", [&] {
        if (D <= 64) {
            if (parts > 1) { ATT_SPLIT(64); } else { ATT(64); }
        } else if (D <= 128) {
            if (parts > 1) { ATT_SPLIT(128); } else { ATT(128); }
        } else if (D <= 256) {
            if (parts > 1) { ATT_SPLIT(256); } else { ATT(256); }
        } else if (D <= 512) {
            if (parts > 1) { ATT_SPLIT(512); } else { ATT(512); }
        } else {
            MFQ_RUNTIME_CHECK(false, "attention: D>512 unsupported, got ", D);
        }
    });
#undef ATT
#undef ATT_SPLIT
    return out;
}

mfq_tensor_backend::Tensor attention_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
                             double scale, bool causal)
{
    return attention_impl_cuda(q, k, v, scale, causal, 0);
}

mfq_tensor_backend::Tensor attention_swa_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor v,
                                 double scale, int64_t window)
{
    MFQ_RUNTIME_CHECK(window > 0 && window <= INT_MAX, "attention_swa: window must be in [1, INT_MAX]");
    return attention_impl_cuda(q, k, v, scale, true, (int)window);
}

template <int BD, typename scalar_t>
__global__ void attention_cache_decode_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ seq_len,
    scalar_t* __restrict__ out,
    int B, int Hq, int Hk, int max_seq, int D, int rep, float scale)
{
    constexpr int NW = BD / 32;
    int qv = blockIdx.x;
    int hq = qv % Hq;
    int b = qv / Hq;
    int hk = hq / rep;
    int tid = threadIdx.x;
    int Tk = (int)seq_len[0];
    if (Tk < 0) {
        Tk = 0;
    }
    if (Tk > max_seq) {
        Tk = max_seq;
    }

    extern __shared__ float q_s[];
    if (tid < D) {
        q_s[tid] = (float)q[((size_t)b * Hq + hq) * D + tid];
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float O = 0.0f;

    for (int s = 0; s < Tk; ++s) {
        float kd = (tid < D) ? (float)k[((size_t)b * Hk + hk) * max_seq * D + (size_t)s * D + tid] : 0.0f;
        float dot = (tid < D) ? q_s[tid] * kd : 0.0f;
        dot = block_sum<NW>(dot);
        float score = dot * scale;
        float m_new = fmaxf(m, score);
        float fac = expf(m - m_new);
        float p = expf(score - m_new);
        float vd = (tid < D) ? (float)v[((size_t)b * Hk + hk) * max_seq * D + (size_t)s * D + tid] : 0.0f;
        O = O * fac + p * vd;
        l = l * fac + p;
        m = m_new;
    }

    if (tid < D) {
        float lo = (l > 0.0f) ? l : 1.0f;
        out[((size_t)b * Hq + hq) * D + tid] = (scalar_t)(O / lo);
    }
}

mfq_tensor_backend::Tensor attention_cache_decode_cuda(mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
                                          mfq_tensor_backend::Tensor seq_len, double scale)
{
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous(), "attention_cache_decode: q must be cuda contiguous");
    MFQ_RUNTIME_CHECK(k_cache.is_cuda() && k_cache.is_contiguous(), "attention_cache_decode: k_cache must be cuda contiguous");
    MFQ_RUNTIME_CHECK(v_cache.is_cuda() && v_cache.is_contiguous(), "attention_cache_decode: v_cache must be cuda contiguous");
    MFQ_RUNTIME_CHECK(seq_len.is_cuda() && seq_len.is_contiguous() && seq_len.scalar_type() == mfq_tensor_backend::kInt64 &&
                seq_len.numel() == 1, "attention_cache_decode: seq_len must be cuda int64[1]");
    MFQ_RUNTIME_CHECK(q.scalar_type() == k_cache.scalar_type() && q.scalar_type() == v_cache.scalar_type(),
                "attention_cache_decode: q/k/v dtype mismatch");
    MFQ_RUNTIME_CHECK(q.scalar_type() == mfq_tensor_backend::kFloat16 || q.scalar_type() == mfq_tensor_backend::kBFloat16 ||
                q.scalar_type() == mfq_tensor_backend::kFloat32,
                "attention_cache_decode: dtype must be f16, bf16, or f32");
    MFQ_RUNTIME_CHECK(q.dim() == 4 && q.size(2) == 1, "attention_cache_decode: q must be [B,Hq,1,D]");
    MFQ_RUNTIME_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
                "attention_cache_decode: caches must be [B,Hk,max_seq,D]");
    int B = (int)q.size(0);
    int Hq = (int)q.size(1);
    int D = (int)q.size(3);
    int Hk = (int)k_cache.size(1);
    int max_seq = (int)k_cache.size(2);
    MFQ_RUNTIME_CHECK(k_cache.size(0) == B && k_cache.size(3) == D, "attention_cache_decode: cache shape mismatch");
    MFQ_RUNTIME_CHECK(Hq % Hk == 0, "attention_cache_decode: GQA requires Hq % Hk == 0");
    int rep = Hq / Hk;
    auto out = mfq_tensor_backend::empty_like(q);
    int total = B * Hq;
    int shmem = D * (int)sizeof(float);
    cudaStream_t stream = mfq_current_cuda_stream();
#define ATT_CACHE(BD) attention_cache_decode_kernel<BD, scalar_t><<<total, BD, shmem, stream>>>( \
    q.data_ptr<scalar_t>(), k_cache.data_ptr<scalar_t>(), v_cache.data_ptr<scalar_t>(),          \
    seq_len.data_ptr<int64_t>(), out.data_ptr<scalar_t>(),                                       \
    B, Hq, Hk, max_seq, D, rep, (float)scale)
    MFQ_DISPATCH_FLOATING_TYPES_AND2(
        mfq_dispatch_half, mfq_dispatch_bfloat16,
        q.scalar_type(), "attention_cache_decode_cuda", [&] {
        if (D <= 64) { ATT_CACHE(64); }
        else if (D <= 128) { ATT_CACHE(128); }
        else if (D <= 256) { ATT_CACHE(256); }
        else if (D <= 512) { ATT_CACHE(512); }
        else { MFQ_RUNTIME_CHECK(false, "attention_cache_decode: D>512 unsupported, got ", D); }
    });
#undef ATT_CACHE
    return out;
}

template <int BD, typename scalar_t>
__global__ void attention_cache_decode_split_part_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ seq_len,
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int B, int Hq, int Hk, int max_seq, int D, int rep,
    int parts, int workspace_parts, float scale)
{
    constexpr int NW = BD / 32;
    const int part = blockIdx.x % parts;
    const int qv = blockIdx.x / parts;
    const int hq = qv % Hq;
    const int b = qv / Hq;
    const int hk = hq / rep;
    const int tid = threadIdx.x;
    int Tk = (int)seq_len[0];
    Tk = Tk < 0 ? 0 : (Tk > max_seq ? max_seq : Tk);
    const int start = (int)(((int64_t)Tk * part) / parts);
    const int end = (int)(((int64_t)Tk * (part + 1)) / parts);

    extern __shared__ float q_s[];
    if (tid < D) {
        q_s[tid] = (float)q[(size_t)qv * D + tid];
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float O = 0.0f;
    for (int s = start; s < end; ++s) {
        const size_t kv_idx = ((size_t)b * Hk + hk) * max_seq * D + (size_t)s * D + tid;
        const float kd = tid < D ? (float)k[kv_idx] : 0.0f;
        float dot = tid < D ? q_s[tid] * kd : 0.0f;
        dot = block_sum<NW>(dot);
        const float score = dot * scale;
        const float m_new = fmaxf(m, score);
        const float fac = expf(m - m_new);
        const float p = expf(score - m_new);
        const float vd = tid < D ? (float)v[kv_idx] : 0.0f;
        O = O * fac + p * vd;
        l = l * fac + p;
        m = m_new;
    }

    const size_t stat_idx = (size_t)qv * workspace_parts + part;
    if (tid < D) {
        partial_o[stat_idx * D + tid] = O;
    }
    if (tid == 0) {
        partial_m[stat_idx] = start < end ? m : -1e30f;
        partial_l[stat_idx] = start < end ? l : 0.0f;
    }
}

template <typename scalar_t>
__global__ void attention_cache_decode_split_gqa4_d128_part_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ seq_len,
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int B, int Hq, int Hk, int max_seq,
    int parts, int workspace_parts, float scale)
{
    constexpr int D = 128;
    constexpr int REP = 4;
    constexpr int NW = D / 32;
    const int part = blockIdx.x % parts;
    const int kv = blockIdx.x / parts;
    const int hk = kv % Hk;
    const int b = kv / Hk;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int hq0 = hk * REP;
    int Tk = (int)seq_len[0];
    Tk = Tk < 0 ? 0 : (Tk > max_seq ? max_seq : Tk);
    const int start = (int)(((int64_t)Tk * part) / parts);
    const int end = (int)(((int64_t)Tk * (part + 1)) / parts);

    float qv[REP];
    float m[REP];
    float l[REP];
    float O[REP];
    #pragma unroll
    for (int r = 0; r < REP; ++r) {
        qv[r] = (float)q[((size_t)b * Hq + hq0 + r) * D + tid];
        m[r] = -1e30f;
        l[r] = 0.0f;
        O[r] = 0.0f;
    }

    __shared__ float warp_dot[REP][NW];
    __shared__ float score[REP];
    const size_t kv_base = ((size_t)b * Hk + hk) * max_seq * D;
    for (int s = start; s < end; ++s) {
        const size_t idx = kv_base + (size_t)s * D + tid;
        const float kd = (float)k[idx];
        const float vd = (float)v[idx];
        #pragma unroll
        for (int r = 0; r < REP; ++r) {
            const float dot = warp_sum(qv[r] * kd);
            if (lane == 0) {
                warp_dot[r][warp] = dot;
            }
        }
        __syncthreads();
        if (warp == 0) {
            #pragma unroll
            for (int r = 0; r < REP; ++r) {
                float dot = lane < NW ? warp_dot[r][lane] : 0.0f;
                dot = warp_sum(dot);
                if (lane == 0) {
                    score[r] = dot * scale;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < REP; ++r) {
            const float m_new = fmaxf(m[r], score[r]);
            const float fac = expf(m[r] - m_new);
            const float p = expf(score[r] - m_new);
            O[r] = O[r] * fac + p * vd;
            l[r] = l[r] * fac + p;
            m[r] = m_new;
        }
    }

    #pragma unroll
    for (int r = 0; r < REP; ++r) {
        const int qv_idx = b * Hq + hq0 + r;
        const size_t stat_idx = (size_t)qv_idx * workspace_parts + part;
        partial_o[stat_idx * D + tid] = O[r];
        if (tid == 0) {
            partial_m[stat_idx] = start < end ? m[r] : -1e30f;
            partial_l[stat_idx] = start < end ? l[r] : 0.0f;
        }
    }
}

template <int BD, typename scalar_t>
__global__ void attention_cache_decode_split_reduce_kernel(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    scalar_t* __restrict__ out,
    int total, int parts, int workspace_parts, int D)
{
    const int qv = blockIdx.x;
    const int tid = threadIdx.x;
    if (qv >= total) {
        return;
    }

    float m = -1e30f;
    for (int p = 0; p < parts; ++p) {
        m = fmaxf(m, partial_m[(size_t)qv * workspace_parts + p]);
    }
    float l = 0.0f;
    float O = 0.0f;
    for (int p = 0; p < parts; ++p) {
        const size_t stat_idx = (size_t)qv * workspace_parts + p;
        const float lp = partial_l[stat_idx];
        const float w = lp > 0.0f ? expf(partial_m[stat_idx] - m) : 0.0f;
        l += w * lp;
        if (tid < D) {
            O += w * partial_o[stat_idx * D + tid];
        }
    }
    if (tid < D) {
        out[(size_t)qv * D + tid] = (scalar_t)(O / (l > 0.0f ? l : 1.0f));
    }
}

mfq_tensor_backend::Tensor attention_cache_decode_split_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor seq_len, double scale,
    mfq_tensor_backend::Tensor partial_o, mfq_tensor_backend::Tensor partial_m, mfq_tensor_backend::Tensor partial_l,
    int64_t parts)
{
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous(), "attention_cache_decode_split: q must be cuda contiguous");
    MFQ_RUNTIME_CHECK(k_cache.is_cuda() && k_cache.is_contiguous(), "attention_cache_decode_split: k_cache must be cuda contiguous");
    MFQ_RUNTIME_CHECK(v_cache.is_cuda() && v_cache.is_contiguous(), "attention_cache_decode_split: v_cache must be cuda contiguous");
    MFQ_RUNTIME_CHECK(seq_len.is_cuda() && seq_len.is_contiguous() && seq_len.scalar_type() == mfq_tensor_backend::kInt64 &&
                seq_len.numel() == 1, "attention_cache_decode_split: seq_len must be cuda int64[1]");
    MFQ_RUNTIME_CHECK(q.scalar_type() == k_cache.scalar_type() && q.scalar_type() == v_cache.scalar_type(),
                "attention_cache_decode_split: q/k/v dtype mismatch");
    MFQ_RUNTIME_CHECK(q.scalar_type() == mfq_tensor_backend::kFloat16 || q.scalar_type() == mfq_tensor_backend::kBFloat16 ||
                q.scalar_type() == mfq_tensor_backend::kFloat32,
                "attention_cache_decode_split: dtype must be f16, bf16, or f32");
    MFQ_RUNTIME_CHECK(q.dim() == 4 && q.size(2) == 1, "attention_cache_decode_split: q must be [B,Hq,1,D]");
    MFQ_RUNTIME_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
                "attention_cache_decode_split: caches must be [B,Hk,max_seq,D]");
    MFQ_RUNTIME_CHECK(partial_o.is_cuda() && partial_o.is_contiguous() && partial_o.scalar_type() == mfq_tensor_backend::kFloat32,
                "attention_cache_decode_split: partial_o must be contiguous CUDA f32");
    MFQ_RUNTIME_CHECK(partial_m.is_cuda() && partial_m.is_contiguous() && partial_m.scalar_type() == mfq_tensor_backend::kFloat32 &&
                partial_l.is_cuda() && partial_l.is_contiguous() && partial_l.scalar_type() == mfq_tensor_backend::kFloat32,
                "attention_cache_decode_split: partial_m/l must be contiguous CUDA f32");

    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int D = (int)q.size(3);
    const int Hk = (int)k_cache.size(1);
    const int max_seq = (int)k_cache.size(2);
    const int total = B * Hq;
    MFQ_RUNTIME_CHECK(k_cache.size(0) == B && k_cache.size(3) == D, "attention_cache_decode_split: cache shape mismatch");
    MFQ_RUNTIME_CHECK(Hq % Hk == 0, "attention_cache_decode_split: GQA requires Hq % Hk == 0");
    MFQ_RUNTIME_CHECK(partial_o.dim() == 3 && partial_o.size(0) == total && partial_o.size(2) == D,
                "attention_cache_decode_split: partial_o shape mismatch");
    const int workspace_parts = (int)partial_o.size(1);
    MFQ_RUNTIME_CHECK(parts >= 2 && parts <= workspace_parts, "attention_cache_decode_split: invalid parts");
    MFQ_RUNTIME_CHECK(partial_m.sizes() == mfq_tensor_backend::IntArrayRef({total, workspace_parts}) &&
                partial_l.sizes() == partial_m.sizes(), "attention_cache_decode_split: partial_m/l shape mismatch");

    auto out = mfq_tensor_backend::empty_like(q);
    const int shmem = D * (int)sizeof(float);
    const int rep = Hq / Hk;
    cudaStream_t stream = mfq_current_cuda_stream();
#define ATT_CACHE_SPLIT(BD) do {                                                                    \
    if (D == 128 && rep == 4) {                                                                       \
        attention_cache_decode_split_gqa4_d128_part_kernel<scalar_t>                                 \
            <<<B * Hk * (int)parts, 128, 0, stream>>>(                                               \
                q.data_ptr<scalar_t>(), k_cache.data_ptr<scalar_t>(),                                 \
                v_cache.data_ptr<scalar_t>(), seq_len.data_ptr<int64_t>(),                            \
                partial_o.data_ptr<float>(), partial_m.data_ptr<float>(),                             \
                partial_l.data_ptr<float>(), B, Hq, Hk, max_seq, (int)parts,                         \
                workspace_parts, (float)scale);                                                       \
    } else {                                                                                           \
        attention_cache_decode_split_part_kernel<BD, scalar_t>                                       \
            <<<total * (int)parts, BD, shmem, stream>>>(                                             \
                q.data_ptr<scalar_t>(), k_cache.data_ptr<scalar_t>(), v_cache.data_ptr<scalar_t>(),   \
                seq_len.data_ptr<int64_t>(), partial_o.data_ptr<float>(), partial_m.data_ptr<float>(), \
                partial_l.data_ptr<float>(), B, Hq, Hk, max_seq, D, rep, (int)parts, workspace_parts, \
                (float)scale);                                                                         \
    }                                                                                                  \
    attention_cache_decode_split_reduce_kernel<BD, scalar_t><<<total, BD, 0, stream>>>(               \
        partial_o.data_ptr<float>(), partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),          \
        out.data_ptr<scalar_t>(), total, (int)parts, workspace_parts, D);                              \
} while (0)
    MFQ_DISPATCH_FLOATING_TYPES_AND2(
        mfq_dispatch_half, mfq_dispatch_bfloat16,
        q.scalar_type(), "attention_cache_decode_split_cuda", [&] {
        if (D <= 64) { ATT_CACHE_SPLIT(64); }
        else if (D <= 128) { ATT_CACHE_SPLIT(128); }
        else if (D <= 256) { ATT_CACHE_SPLIT(256); }
        else if (D <= 512) { ATT_CACHE_SPLIT(512); }
        else { MFQ_RUNTIME_CHECK(false, "attention_cache_decode_split: D>512 unsupported, got ", D); }
    });
#undef ATT_CACHE_SPLIT
    return out;
}

template <int BD, typename scalar_t>
__global__ void attention_cache_swa_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ seq_len,
    scalar_t* __restrict__ out,
    int Hq, int Hk, int T, int capacity, int D, int rep,
    int window, float scale)
{
    constexpr int NW = BD / 32;
    const int qv = blockIdx.x;
    const int tq = qv % T;
    const int rem = qv / T;
    const int hq = rem % Hq;
    const int b = rem / Hq;
    const int hk = hq / rep;
    const int tid = threadIdx.x;
    const int64_t length = seq_len[0] > 0 ? seq_len[0] : 0;
    const int64_t qpos = length - T + tq;
    const int64_t end = qpos + 1 < length ? qpos + 1 : length;
    const int64_t start = end > window ? end - window : 0;

    extern __shared__ float q_s[];
    if (tid < D) q_s[tid] = (float)q[(size_t)qv * D + tid];
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float O = 0.0f;
    for (int64_t pos = start; pos < end; ++pos) {
        const int slot = (int)(pos % capacity);
        const size_t kv_idx = ((size_t)b * Hk + hk) * capacity * D + (size_t)slot * D + tid;
        const float kd = tid < D ? (float)k[kv_idx] : 0.0f;
        float dot = tid < D ? q_s[tid] * kd : 0.0f;
        dot = block_sum<NW>(dot);
        const float score = dot * scale;
        const float m_new = fmaxf(m, score);
        const float fac = expf(m - m_new);
        const float p = expf(score - m_new);
        const float vd = tid < D ? (float)v[kv_idx] : 0.0f;
        O = O * fac + p * vd;
        l = l * fac + p;
        m = m_new;
    }
    if (tid < D) out[(size_t)qv * D + tid] = (scalar_t)(O / (l > 0.0f ? l : 1.0f));
}

template <int BD, typename scalar_t>
__global__ void attention_cache_swa_split_part_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ seq_len,
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int Hq, int Hk, int T, int capacity, int D, int rep,
    int window, int parts, float scale)
{
    constexpr int NW = BD / 32;
    const int part = blockIdx.x % parts;
    const int qv = blockIdx.x / parts;
    const int tq = qv % T;
    const int rem = qv / T;
    const int hq = rem % Hq;
    const int b = rem / Hq;
    const int hk = hq / rep;
    const int tid = threadIdx.x;
    const int64_t length = seq_len[0] > 0 ? seq_len[0] : 0;
    const int64_t qpos = length - T + tq;
    const int64_t visible_end = qpos + 1 < length ? qpos + 1 : length;
    const int64_t visible_start = visible_end > window ? visible_end - window : 0;
    const int64_t visible = visible_end - visible_start;
    const int64_t start = visible_start + visible * part / parts;
    const int64_t end = visible_start + visible * (part + 1) / parts;

    extern __shared__ float q_s[];
    if (tid < D) q_s[tid] = (float)q[(size_t)qv * D + tid];
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float O = 0.0f;
    for (int64_t pos = start; pos < end; ++pos) {
        const int slot = (int)(pos % capacity);
        const size_t kv_idx = ((size_t)b * Hk + hk) * capacity * D + (size_t)slot * D + tid;
        const float kd = tid < D ? (float)k[kv_idx] : 0.0f;
        float dot = tid < D ? q_s[tid] * kd : 0.0f;
        dot = block_sum<NW>(dot);
        const float score = dot * scale;
        const float m_new = fmaxf(m, score);
        const float fac = expf(m - m_new);
        const float p = expf(score - m_new);
        const float vd = tid < D ? (float)v[kv_idx] : 0.0f;
        O = O * fac + p * vd;
        l = l * fac + p;
        m = m_new;
    }
    const size_t stat = (size_t)qv * parts + part;
    if (tid < D) partial_o[stat * D + tid] = O;
    if (tid == 0) {
        partial_m[stat] = start < end ? m : -1e30f;
        partial_l[stat] = start < end ? l : 0.0f;
    }
}

static mfq_tensor_backend::Tensor attention_cache_swa_impl_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor seq_len, double scale, int64_t window, int64_t planned_length)
{
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous(), "attention_cache_swa: q must be cuda contiguous");
    MFQ_RUNTIME_CHECK(k_cache.is_cuda() && k_cache.is_contiguous() &&
                v_cache.is_cuda() && v_cache.is_contiguous(),
                "attention_cache_swa: caches must be cuda contiguous");
    MFQ_RUNTIME_CHECK(seq_len.is_cuda() && seq_len.is_contiguous() &&
                seq_len.scalar_type() == mfq_tensor_backend::kInt64 && seq_len.numel() == 1,
                "attention_cache_swa: seq_len must be cuda int64[1]");
    MFQ_RUNTIME_CHECK(q.scalar_type() == k_cache.scalar_type() && q.scalar_type() == v_cache.scalar_type(),
                "attention_cache_swa: q/k/v dtype mismatch");
    MFQ_RUNTIME_CHECK(q.scalar_type() == mfq_tensor_backend::kFloat16 || q.scalar_type() == mfq_tensor_backend::kFloat32,
                "attention_cache_swa: dtype must be f16 or f32");
    MFQ_RUNTIME_CHECK(q.dim() == 4 && k_cache.dim() == 4 && v_cache.dim() == 4 &&
                k_cache.sizes() == v_cache.sizes(),
                "attention_cache_swa: expected q[B,Hq,T,D] and cache[B,Hk,capacity,D]");
    MFQ_RUNTIME_CHECK(window > 0 && window <= INT_MAX, "attention_cache_swa: invalid window");

    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int T = (int)q.size(2);
    const int D = (int)q.size(3);
    const int Hk = (int)k_cache.size(1);
    const int capacity = (int)k_cache.size(2);
    MFQ_RUNTIME_CHECK(T > 0 && capacity >= window, "attention_cache_swa: cache capacity must cover the window");
    MFQ_RUNTIME_CHECK(k_cache.size(0) == B && k_cache.size(3) == D, "attention_cache_swa: cache shape mismatch");
    MFQ_RUNTIME_CHECK(Hq % Hk == 0, "attention_cache_swa: GQA requires Hq % Hk == 0");

    auto out = mfq_tensor_backend::empty_like(q);
    const int total = B * Hq * T;
    const int rep = Hq / Hk;
    const int shmem = D * (int)sizeof(float);
    const int visible_keys = planned_length > 0
        ? std::min<int64_t>(window, planned_length)
        : static_cast<int>(window);
    int parts = total < 128 && visible_keys >= 512 ? (visible_keys + 255) / 256 : 1;
    parts = std::max(1, std::min(parts, 16));
    cudaStream_t stream = mfq_current_cuda_stream();

#define ATT_CACHE_SWA(BD) do {                                                                  \
    if (parts == 1) {                                                                            \
        attention_cache_swa_kernel<BD, scalar_t><<<total, BD, shmem, stream>>>(                  \
            q.data_ptr<scalar_t>(), k_cache.data_ptr<scalar_t>(), v_cache.data_ptr<scalar_t>(),  \
            seq_len.data_ptr<int64_t>(), out.data_ptr<scalar_t>(),                               \
            Hq, Hk, T, capacity, D, rep, (int)window, (float)scale);                             \
    } else {                                                                                     \
        auto po = mfq_tensor_backend::empty({total, parts, D}, q.options().dtype(mfq_tensor_backend::kFloat32));            \
        auto pm = mfq_tensor_backend::empty({total, parts}, q.options().dtype(mfq_tensor_backend::kFloat32));               \
        auto pl = mfq_tensor_backend::empty({total, parts}, q.options().dtype(mfq_tensor_backend::kFloat32));               \
        attention_cache_swa_split_part_kernel<BD, scalar_t><<<total * parts, BD, shmem, stream>>>( \
            q.data_ptr<scalar_t>(), k_cache.data_ptr<scalar_t>(), v_cache.data_ptr<scalar_t>(),  \
            seq_len.data_ptr<int64_t>(), po.data_ptr<float>(), pm.data_ptr<float>(),             \
            pl.data_ptr<float>(), Hq, Hk, T, capacity, D, rep, (int)window, parts, (float)scale); \
        attention_split_reduce_kernel<BD, scalar_t><<<total, BD, 0, stream>>>(                   \
            po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),                    \
            out.data_ptr<scalar_t>(), total, parts, D);                                          \
    }                                                                                            \
} while (0)
    MFQ_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "attention_cache_swa_cuda", [&] {
        if (D <= 64) { ATT_CACHE_SWA(64); }
        else if (D <= 128) { ATT_CACHE_SWA(128); }
        else if (D <= 256) { ATT_CACHE_SWA(256); }
        else if (D <= 512) { ATT_CACHE_SWA(512); }
        else { MFQ_RUNTIME_CHECK(false, "attention_cache_swa: D>512 unsupported, got ", D); }
    });
#undef ATT_CACHE_SWA
    return out;
}

mfq_tensor_backend::Tensor attention_cache_swa_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor seq_len, double scale, int64_t window)
{
    return attention_cache_swa_impl_cuda(
        q, k_cache, v_cache, seq_len, scale, window, 0);
}

mfq_tensor_backend::Tensor attention_cache_swa_planned_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k_cache, mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor seq_len, double scale, int64_t window, int64_t planned_length)
{
    MFQ_RUNTIME_CHECK(planned_length > 0 && planned_length <= INT_MAX,
        "attention_cache_swa_planned: planned length must be in [1, INT_MAX]");
    return attention_cache_swa_impl_cuda(
        q, k_cache, v_cache, seq_len, scale, window, planned_length);
}
