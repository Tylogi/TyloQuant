// Gated DeltaNet CUDA kernel (ggml gated_delta_net.cu).
//
// grid = B*H (one block per (batch, head)); state S[D,D] in shared memory, T loop in-kernel.
// Recurrence (matches ops.cpp / Python reference):
//     S <- decay*S ; delta = (v - S^T k) * beta ; S <- S + k (x) delta ; o = scale * (S^T q)
// scale = 1/sqrt(D). Scalar gate and per-dim (KDA) gate; D in {32,64,128} (D=128 needs the
// 64KB shared-mem opt-in via cudaFuncSetAttribute).

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include <cstdlib>

#include "reduce.cuh"

template <int D>
__global__ void gdn_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* __restrict__ s_in,
    float* __restrict__ out, float* __restrict__ s_out,
    int H, int T, int kda)
{
    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    int j = threadIdx.x;
    if (j >= D) return;

    extern __shared__ float S[];   // S[i*D + j] = state[i][j]

    if (s_in) {
        const float* src = s_in + ((size_t)(b * H + h)) * D * D;
        #pragma unroll 4
        for (int i = 0; i < D; ++i) S[i * D + j] = src[i * D + j];
    } else {
        #pragma unroll 4
        for (int i = 0; i < D; ++i) S[i * D + j] = 0.0f;
    }

    const float scale = 1.0f / sqrtf((float)D);
    const size_t bhoff = (size_t)(b * H + h) * T;
    const float* qb = q + bhoff * D;
    const float* kb = k + bhoff * D;
    const float* vb = v + bhoff * D;
    const float* gb = kda ? g + bhoff * D : g + bhoff;
    const float* bb = beta + bhoff;
    float* ob = out + bhoff * D;

    for (int t = 0; t < T; ++t) {
        const float* qt = qb + (size_t)t * D;
        const float* kt = kb + (size_t)t * D;
        const float* vt = vb + (size_t)t * D;
        float bt = bb[t];

        // gated decay
        if (kda) {
            #pragma unroll 4
            for (int i = 0; i < D; ++i) S[i * D + j] *= expf(gb[t * D + i]);
        } else {
            float dec = expf(gb[t]);
            #pragma unroll 4
            for (int i = 0; i < D; ++i) S[i * D + j] *= dec;
        }
        __syncthreads();

        // delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta
        float stk = 0.f;
        #pragma unroll 4
        for (int i = 0; i < D; ++i) stk += S[i * D + j] * kt[i];
        float delta = (vt[j] - stk) * bt;

        // S[i][j] += k[i] * delta[j]
        #pragma unroll 4
        for (int i = 0; i < D; ++i) S[i * D + j] += kt[i] * delta;
        __syncthreads();

        // o[j] = scale * sum_i S[i][j]*q[i]
        float o = 0.f;
        #pragma unroll 4
        for (int i = 0; i < D; ++i) o += S[i * D + j] * qt[i];
        ob[(size_t)t * D + j] = o * scale;
    }

    float* dst = s_out + ((size_t)(b * H + h)) * D * D;
    #pragma unroll 4
    for (int i = 0; i < D; ++i) dst[i * D + j] = S[i * D + j];
}

template <int D, int BD>
__global__ void gdn_column_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* __restrict__ s_in,
    float* __restrict__ out, float* __restrict__ s_out,
    int H, int T, int kda)
{
    constexpr int NW = BD / 32;
    int bid = blockIdx.x;
    int j = bid % D;
    int bh = bid / D;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    const float scale = 1.0f / sqrtf((float)D);
    const size_t state_off = ((size_t)(b * H + h)) * D * D;
    const size_t bhoff = (size_t)(b * H + h) * T;
    const float* qb = q + bhoff * D;
    const float* kb = k + bhoff * D;
    const float* vb = v + bhoff * D;
    const float* gb = kda ? g + bhoff * D : g + bhoff;
    const float* bb = beta + bhoff;
    float* ob = out + bhoff * D;
    float* so = s_out + state_off;

    for (int i = tid; i < D; i += BD) {
        so[i * D + j] = s_in ? s_in[state_off + i * D + j] : 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        const float* qt = qb + (size_t)t * D;
        const float* kt = kb + (size_t)t * D;
        const float* vt = vb + (size_t)t * D;
        float bt = bb[t];

        float local = 0.0f;
        for (int i = tid; i < D; i += BD) {
            float sij = so[i * D + j];
            float dec = kda ? expf(gb[(size_t)t * D + i]) : expf(gb[t]);
            sij *= dec;
            so[i * D + j] = sij;
            local += sij * kt[i];
        }
        float stk = block_sum<NW>(local);
        float delta = (vt[j] - stk) * bt;
        __syncthreads();

        for (int i = tid; i < D; i += BD) {
            so[i * D + j] += kt[i] * delta;
        }
        __syncthreads();

        local = 0.0f;
        for (int i = tid; i < D; i += BD) {
            local += so[i * D + j] * qt[i];
        }
        float o = block_sum<NW>(local);
        if (tid == 0) {
            ob[(size_t)t * D + j] = o * scale;
        }
        __syncthreads();
    }
}

template <int D, bool KDA, int WARPS, bool TRANSPOSED_STATE=false, bool TILED_HEADS=false>
__global__ void gdn_warp_column_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* __restrict__ s_in,
    float* __restrict__ out, float* __restrict__ s_out,
    int Hq, int Hv, int T)
{
    constexpr int WARP = 32;
    constexpr int ROWS = (D + WARP - 1) / WARP;
    int bh = blockIdx.x;
    int b = bh / Hv;
    int hv = bh % Hv;
    int hq = TILED_HEADS ? hv % Hq : hv / (Hv / Hq);
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int col = blockIdx.y * WARPS + warp;
    if (col >= D) {
        return;
    }

    const size_t state_off = ((size_t)(b * Hv + hv)) * D * D;
    const size_t q_seq_off = ((size_t)(b * Hq + hq)) * T;
    const size_t v_seq_off = ((size_t)(b * Hv + hv)) * T;
    float s_shard[ROWS];

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        int i = r * WARP + lane;
        size_t state_idx = TRANSPOSED_STATE ? (size_t)col * D + i : (size_t)i * D + col;
        s_shard[r] = (i < D && s_in) ? s_in[state_off + state_idx] : 0.0f;
    }

    const float scale = 1.0f / sqrtf((float)D);
    for (int t = 0; t < T; ++t) {
        const float* qt = q + (q_seq_off + t) * D;
        const float* kt = k + (q_seq_off + t) * D;
        const float* vt = v + (v_seq_off + t) * D;
        const float* gt = KDA ? g + (v_seq_off + t) * D : g + (v_seq_off + t);
        float bt = beta[v_seq_off + t];

        float k_reg[ROWS];
        float q_reg[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            int i = r * WARP + lane;
            k_reg[r] = (i < D) ? kt[i] : 0.0f;
            q_reg[r] = (i < D) ? qt[i] : 0.0f;
        }

        float kv = 0.0f;
        if constexpr (KDA) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                int i = r * WARP + lane;
                float dec = (i < D) ? expf(gt[i]) : 0.0f;
                kv += dec * s_shard[r] * k_reg[r];
            }
        } else {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                kv += s_shard[r] * k_reg[r];
            }
        }
        kv = warp_sum(kv);

        float delta;
        if constexpr (KDA) {
            delta = (vt[col] - kv) * bt;
        } else {
            float dec = expf(*gt);
            delta = (vt[col] - dec * kv) * bt;
        }

        float o = 0.0f;
        if constexpr (KDA) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                int i = r * WARP + lane;
                float dec = (i < D) ? expf(gt[i]) : 0.0f;
                s_shard[r] = dec * s_shard[r] + k_reg[r] * delta;
                o += s_shard[r] * q_reg[r];
            }
        } else {
            float dec = expf(*gt);
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                s_shard[r] = dec * s_shard[r] + k_reg[r] * delta;
                o += s_shard[r] * q_reg[r];
            }
        }
        o = warp_sum(o);
        if (lane == 0) {
            out[(v_seq_off + t) * D + col] = o * scale;
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        int i = r * WARP + lane;
        if (i < D) {
            size_t state_idx = TRANSPOSED_STATE ? (size_t)col * D + i : (size_t)i * D + col;
            s_out[state_off + state_idx] = s_shard[r];
        }
    }
}

#define LAUNCH_GDN(DVAL)                                                            \
    do {                                                                            \
        if (shmem > 49152)                                                          \
            cudaFuncSetAttribute((const void*)gdn_kernel<DVAL>,                     \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);               \
        gdn_kernel<DVAL><<<B * H, DVAL, shmem, stream>>>(                           \
            qd, kd, vd, gd, bd, sd, od, sod, H, T, kda);                            \
    } while (0)

#define LAUNCH_GDN_COL(DVAL)                                                        \
    do {                                                                            \
        gdn_column_kernel<DVAL, 128><<<B * H * DVAL, 128, 0, stream>>>(              \
            qd, kd, vd, gd, bd, sd, od, sod, H, T, kda);                            \
    } while (0)

#define LAUNCH_GDN_WARP(DVAL, KDA_VAL, TRANSPOSED_VAL, TILED_VAL)                     \
    do {                                                                              \
        constexpr int WARPS = 4;                                                      \
        dim3 grid(B * Hv, (DVAL + WARPS - 1) / WARPS, 1);                             \
        dim3 block(32, WARPS, 1);                                                     \
        gdn_warp_column_kernel<DVAL, KDA_VAL, WARPS, TRANSPOSED_VAL, TILED_VAL><<<grid, block, 0, stream>>>( \
            qd, kd, vd, gd, bd, sd, od, sod, Hq, Hv, T);                              \
    } while (0)

#define DISPATCH_GDN_WARP(TILED_VAL)                                                   \
    do {                                                                               \
        if (transposed_state) {                                                        \
            if (kda) {                                                                 \
                if (D == 32) { LAUNCH_GDN_WARP(32, true, true, TILED_VAL); }           \
                else if (D == 64) { LAUNCH_GDN_WARP(64, true, true, TILED_VAL); }      \
                else if (D == 128) { LAUNCH_GDN_WARP(128, true, true, TILED_VAL); }    \
                else TORCH_CHECK(false, "GDN CUDA: D must be in {32,64,128}, got ", D); \
            } else {                                                                   \
                if (D == 32) { LAUNCH_GDN_WARP(32, false, true, TILED_VAL); }          \
                else if (D == 64) { LAUNCH_GDN_WARP(64, false, true, TILED_VAL); }     \
                else if (D == 128) { LAUNCH_GDN_WARP(128, false, true, TILED_VAL); }   \
                else TORCH_CHECK(false, "GDN CUDA: D must be in {32,64,128}, got ", D); \
            }                                                                          \
        } else if (kda) {                                                              \
            if (D == 32) { LAUNCH_GDN_WARP(32, true, false, TILED_VAL); }              \
            else if (D == 64) { LAUNCH_GDN_WARP(64, true, false, TILED_VAL); }         \
            else if (D == 128) { LAUNCH_GDN_WARP(128, true, false, TILED_VAL); }       \
            else TORCH_CHECK(false, "GDN CUDA: D must be in {32,64,128}, got ", D);   \
        } else {                                                                       \
            if (D == 32) { LAUNCH_GDN_WARP(32, false, false, TILED_VAL); }             \
            else if (D == 64) { LAUNCH_GDN_WARP(64, false, false, TILED_VAL); }        \
            else if (D == 128) { LAUNCH_GDN_WARP(128, false, false, TILED_VAL); }      \
            else TORCH_CHECK(false, "GDN CUDA: D must be in {32,64,128}, got ", D);   \
        }                                                                              \
    } while (0)

static std::vector<torch::Tensor> gdn_cuda_impl(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, c10::optional<torch::Tensor> state,
    bool inplace_state, bool transposed_state=false, bool tiled_heads=false)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kFloat32,
                "q must be cuda contiguous f32");
    int B = q.size(0), Hq = q.size(1), T = q.size(2), D = q.size(3);
    int Hv = v.size(1);
    int H = Hv;
    int kda = (g.dim() == 4) ? 1 : 0;
    auto opts = q.options();
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() && k.scalar_type() == torch::kFloat32,
                "k must be cuda contiguous f32");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == torch::kFloat32,
                "v must be cuda contiguous f32");
    TORCH_CHECK(k.size(0) == B && k.size(1) == Hq && k.size(2) == T && k.size(3) == D,
                "GDN CUDA: k shape mismatch");
    TORCH_CHECK(v.size(0) == B && v.size(2) == T && v.size(3) == D,
                "GDN CUDA: v shape mismatch");
    TORCH_CHECK(Hv % Hq == 0, "GDN CUDA: value heads must be divisible by key heads");
    TORCH_CHECK(beta.is_cuda() && beta.is_contiguous() && beta.scalar_type() == torch::kFloat32,
                "beta must be cuda contiguous f32");
    TORCH_CHECK(beta.dim() == 3 && beta.size(0) == B && beta.size(1) == Hv && beta.size(2) == T,
                "GDN CUDA: beta must be [B,Hv,T]");
    TORCH_CHECK(g.is_cuda() && g.is_contiguous() && g.scalar_type() == torch::kFloat32,
                "g must be cuda contiguous f32");
    if (kda) {
        TORCH_CHECK(g.size(0) == B && g.size(1) == Hv && g.size(2) == T && g.size(3) == D,
                    "GDN CUDA: KDA g must be [B,Hv,T,D]");
    } else {
        TORCH_CHECK(g.dim() == 3 && g.size(0) == B && g.size(1) == Hv && g.size(2) == T,
                    "GDN CUDA: g must be [B,Hv,T]");
    }
    auto out = torch::empty({B, Hv, T, D}, opts);
    TORCH_CHECK(!inplace_state || (state.has_value() && state->defined() && state->numel() > 0),
                "GDN CUDA inplace: state is required");
    auto s_out = inplace_state ? state.value() : torch::empty({B, Hv, D, D}, opts);
    TORCH_CHECK(s_out.is_cuda() && s_out.is_contiguous() && s_out.scalar_type() == torch::kFloat32,
                "GDN CUDA: output state must be cuda contiguous f32");
    TORCH_CHECK(s_out.size(0) == B && s_out.size(1) == Hv && s_out.size(2) == D && s_out.size(3) == D,
                "GDN CUDA: state shape must be [B,Hv,D,D]");

    const float *qd = q.data_ptr<float>(), *kd = k.data_ptr<float>(), *vd = v.data_ptr<float>();
    const float *gd = g.data_ptr<float>(), *bd = beta.data_ptr<float>();
    const float* sd = state.has_value() && state->defined() && state->numel() > 0
                          ? state->contiguous().data_ptr<float>() : nullptr;
    float* od = out.data_ptr<float>();
    float* sod = s_out.data_ptr<float>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int shmem = D * D * (int)sizeof(float);

    const char* col_env = std::getenv("MFQ_GDN_COLUMN");
    const char* warp_env = std::getenv("MFQ_GDN_WARP");
    bool use_warp = !(warp_env && warp_env[0] == '0');
    if (use_warp) {
        if (tiled_heads) { DISPATCH_GDN_WARP(true); }
        else { DISPATCH_GDN_WARP(false); }
    } else if (T <= 4 && col_env && col_env[0] == '1') {
        TORCH_CHECK(Hq == Hv, "GDN CUDA column path requires repeated q/k heads");
        if (D == 32) { LAUNCH_GDN_COL(32); }
        else if (D == 64) { LAUNCH_GDN_COL(64); }
        else if (D == 128) { LAUNCH_GDN_COL(128); }
        else TORCH_CHECK(false, "GDN CUDA: D must be in {32,64,128}, got ", D);
    } else {
        TORCH_CHECK(Hq == Hv, "GDN CUDA shared-memory path requires repeated q/k heads");
        if (D == 32) { LAUNCH_GDN(32); }
        else if (D == 64) { LAUNCH_GDN(64); }
        else if (D == 128) { LAUNCH_GDN(128); }
        else TORCH_CHECK(false, "GDN CUDA: D must be in {32,64,128}, got ", D);
    }

    return {out, s_out};
}

std::vector<torch::Tensor> gdn_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, c10::optional<torch::Tensor> state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, false, false);
}

std::vector<torch::Tensor> gdn_inplace_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, false);
}

std::vector<torch::Tensor> gdn_inplace_transposed_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, true);
}

std::vector<torch::Tensor> gdn_inplace_tiled_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, false, true);
}

std::vector<torch::Tensor> gdn_inplace_transposed_tiled_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, true, true);
}
