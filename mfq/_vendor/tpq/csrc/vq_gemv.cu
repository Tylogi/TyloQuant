// TPQ fused kernels:
//   1) VQ grouped GEMV (single-kernel codebook lookup + dot), u8/u16 index.
//   2) HC sinkhorn (Hyper-Connections 4x4 doubly-stochastic iteration, one
//      kernel for softmax + 20 rounds, replacing ~85 tiny launches per call).
//   3) DSV4 decode attention core (score + masked sink-softmax + value reduce
//      + inverse RoPE, one block per head).
// Chinese docs live in tpq/fusedext.py (this file stays pure ASCII: CJK
// comments make MSVC emit C4819 and echo source lines into the build log,
// which crashes the torch builder with UnicodeDecodeError on GBK consoles).
//
// vq_gemv math: y[n, r] = sum_b dot(cb[n, idx[n, r, b], :], x[n, b*D:(b+1)*D])
//   Element-wise identical to the LUT algorithm of VQWeight.matmul_T in
//   tpq/kernels.py, but avoids materializing the [N, B, R] gather buffer.
//   cb batch broadcast supported (cbStrideN==0: all N experts share the layer
//   codebook -- saves the per-call N-fold stack copy of grouped.py).
//
// Parallelization: one warp per (n, r) row; block = 32x8 (8 rows),
// grid = (ceil(R/8), N). The x row is staged in dynamic shared memory
// (<=16KB at C=4096), then each warp walks b blocks: fetch codebook row,
// lane-strided dot, warp reduce via __shfl_down_sync. x batch broadcast is
// supported (xStrideN==0: all N experts share one input row at T=1 decode).
//
// hc_sinkhorn math (per row of mixes [N,24], hc=4; mirrors CCCP/dsv4.hc_split):
//   pre[j]  = sigmoid(m[j]*scale[0] + base[j]) + eps
//   post[j] = 2*sigmoid(m[4+j]*scale[1] + base[4+j])
//   comb[j][k] = m[8+4j+k]*scale[2] + base[8+4j+k]
//   comb = softmax(comb, dim=-1) + eps; col-normalize; then (iters-1) rounds
//   of (row-normalize, col-normalize), each divide by (sum + eps).
//   One thread per row, 4x4 kept in registers; output packed [N,24]:
//   pre | post | comb(row-major).
//
// Build: python -c "from tpq import fusedext; fusedext.prebuild()"
//        (needs CUDA Toolkit + MSVC Build Tools + ninja).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || \
    defined(_M_X64)
#include <immintrin.h>
#endif

#define ROWS_PER_BLOCK 32  // warps per block = output rows per block

template <typename idx_t>
__global__ void vq_gemv_kernel(
    const float* __restrict__ x,     // [Nx, C], batch-broadcast when xStrideN==0
    const idx_t* __restrict__ idx,   // [Ni, R, B], batch-broadcast when idxStrideN==0
    const float* __restrict__ cb,    // [N, K, D], batch-broadcast when cbStrideN==0
    float* __restrict__ out,         // [N, R]
    const int R, const int B, const int D,
    const long xStrideN, const long cbStrideN, const long idxStrideN)
{
    const int n = blockIdx.y;
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    extern __shared__ float xs[];                 // [C] staged x row
    const float* xrow = x + (long)n * xStrideN;
    const int C = B * D;
    for (int i = threadIdx.y * 32 + threadIdx.x; i < C; i += 32 * ROWS_PER_BLOCK)
        xs[i] = xrow[i];
    __syncthreads();
    if (r >= R) return;

    const idx_t* irow = idx + (long)n * idxStrideN + (long)r * B;
    const float* cbn = cb + (long)n * cbStrideN;
    // lane-parallel over b blocks: each lane finishes its own D-dim dot locally
    // (D=4/8, no cross-lane work), one single warp reduce at the end (the old
    // version reduced per b-block = 5B shuffles per row, with most lanes idle
    // when D<32).
    float acc = 0.f;
    for (int b = threadIdx.x; b < B; b += 32) {
        const float* crow = cbn + (long)irow[b] * D;   // codebook row (D floats)
        const float* xb = xs + b * D;
        float part = 0.f;
        #pragma unroll 8
        for (int i = 0; i < D; ++i)
            part += crow[i] * xb[i];
        acc += part;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (threadIdx.x == 0)
        out[(long)n * R + r] = acc;
}

torch::Tensor vq_gemv(torch::Tensor x, torch::Tensor idx, torch::Tensor cb) {
    TORCH_CHECK(x.is_cuda() && idx.is_cuda() && cb.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat && cb.scalar_type() == at::kFloat,
                "x/cb must be float32");
    TORCH_CHECK(idx.scalar_type() == at::kByte || idx.scalar_type() == at::kUInt16,
                "idx must be uint8 or uint16");
    TORCH_CHECK(x.is_contiguous() && idx.is_contiguous() && cb.is_contiguous(),
                "tensors must be contiguous");
    const long R = idx.size(1), B = idx.size(2);
    const long D = cb.size(2);
    // batch N = the non-broadcast side of x/idx: (x[N],idx[N]) | (x[1],idx[N]) | (x[N],idx[1])
    const long N = x.size(0) > idx.size(0) ? x.size(0) : idx.size(0);
    TORCH_CHECK(x.size(0) == 1 || x.size(0) == N, "x batch must be 1 or N");
    TORCH_CHECK(idx.size(0) == 1 || idx.size(0) == N, "idx batch must be 1 or N");
    TORCH_CHECK(cb.size(0) == N || cb.size(0) == 1, "cb batch must be 1 or N");
    TORCH_CHECK(x.size(1) == B * D, "x cols must equal B*D");

    auto out = torch::empty({N, R}, x.options());
    const long xStrideN = x.size(0) == 1 ? 0 : x.stride(0);
    const long cbStrideN = cb.size(0) == 1 ? 0 : cb.stride(0);
    const long idxStrideN = idx.size(0) == 1 ? 0 : (long)R * B;
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid((unsigned)((R + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK), (unsigned)N);
    const size_t smem = (size_t)B * D * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (idx.scalar_type() == at::kByte) {
        vq_gemv_kernel<uint8_t><<<grid, block, smem, stream>>>(
            x.data_ptr<float>(), idx.data_ptr<uint8_t>(), cb.data_ptr<float>(),
            out.data_ptr<float>(), (int)R, (int)B, (int)D, xStrideN, cbStrideN, idxStrideN);
    } else {
        vq_gemv_kernel<uint16_t><<<grid, block, smem, stream>>>(
            x.data_ptr<float>(), (const uint16_t*)idx.data_ptr(), cb.data_ptr<float>(),
            out.data_ptr<float>(), (int)R, (int)B, (int)D, xStrideN, cbStrideN, idxStrideN);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---- Kimi KDA one-token recurrent update (V-first FP32 state) ----

template <typename weight_t>
__device__ __forceinline__ float kimi_conv_weight(
    const weight_t* value)
{
    return static_cast<float>(*value);
}

template <>
__device__ __forceinline__ float kimi_conv_weight(
    const __nv_bfloat16* value)
{
    return __bfloat162float(*value);
}

template <typename weight_t>
__global__ void kimi_short_conv3_kernel(
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ query_state,
    __nv_bfloat16* __restrict__ key_state,
    __nv_bfloat16* __restrict__ value_state,
    const weight_t* __restrict__ query_weight,
    const weight_t* __restrict__ key_weight,
    const weight_t* __restrict__ value_weight,
    const int channels,
    const int history)
{
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int stream = blockIdx.y;
    if (channel >= channels || stream >= 3) return;
    __nv_bfloat16* input = stream == 0 ? query : (
        stream == 1 ? key : value);
    __nv_bfloat16* state = stream == 0 ? query_state : (
        stream == 1 ? key_state : value_state);
    const weight_t* weight = stream == 0 ? query_weight : (
        stream == 1 ? key_weight : value_weight);
    __nv_bfloat16* state_row =
        state + static_cast<long>(channel) * history;
    const weight_t* weight_row =
        weight + static_cast<long>(channel) * (history + 1);
    float result = 0.0f;
    for (int item = 0; item < history; ++item) {
        result = fmaf(
            __bfloat162float(state_row[item]),
            kimi_conv_weight(weight_row + item),
            result);
    }
    const float current = __bfloat162float(input[channel]);
    result = fmaf(
        current,
        kimi_conv_weight(weight_row + history),
        result);
    for (int item = 0; item + 1 < history; ++item)
        state_row[item] = state_row[item + 1];
    if (history > 0)
        state_row[history - 1] = input[channel];
    const float silu = result / (1.0f + expf(-result));
    input[channel] = __float2bfloat16_rn(silu);
}

bool kimi_short_conv3(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor query_state,
    torch::Tensor key_state,
    torch::Tensor value_state,
    torch::Tensor query_weight,
    torch::Tensor key_weight,
    torch::Tensor value_weight)
{
    TORCH_CHECK(
        query.is_cuda() && key.is_cuda() && value.is_cuda() &&
        query_state.is_cuda() && key_state.is_cuda() &&
        value_state.is_cuda() && query_weight.is_cuda() &&
        key_weight.is_cuda() && value_weight.is_cuda(),
        "Kimi short convolution tensors must be CUDA");
    TORCH_CHECK(
        query.scalar_type() == at::kBFloat16 &&
        key.scalar_type() == at::kBFloat16 &&
        value.scalar_type() == at::kBFloat16 &&
        query_state.scalar_type() == at::kBFloat16 &&
        key_state.scalar_type() == at::kBFloat16 &&
        value_state.scalar_type() == at::kBFloat16 &&
        (
            query_weight.scalar_type() == at::kBFloat16 ||
            query_weight.scalar_type() == at::kFloat
        ) &&
        key_weight.scalar_type() == query_weight.scalar_type() &&
        value_weight.scalar_type() == query_weight.scalar_type(),
        "Kimi short convolution requires BF16 state and matching "
        "BF16/FP32 weights");
    TORCH_CHECK(
        query.is_contiguous() && key.is_contiguous() &&
        value.is_contiguous() && query_state.is_contiguous() &&
        key_state.is_contiguous() && value_state.is_contiguous() &&
        query_weight.is_contiguous() && key_weight.is_contiguous() &&
        value_weight.is_contiguous(),
        "Kimi short convolution tensors must be contiguous");
    TORCH_CHECK(
        query.dim() == 1 && key.sizes() == query.sizes() &&
        value.sizes() == query.sizes() &&
        query_state.dim() == 2 &&
        key_state.sizes() == query_state.sizes() &&
        value_state.sizes() == query_state.sizes() &&
        query_state.size(0) == query.size(0),
        "Kimi short convolution input/state shapes do not match");
    const int channels = static_cast<int>(query.numel());
    const int history = static_cast<int>(query_state.size(1));
    const long weight_items =
        static_cast<long>(channels) * (history + 1);
    TORCH_CHECK(
        query_weight.numel() == weight_items &&
        key_weight.numel() == weight_items &&
        value_weight.numel() == weight_items,
        "Kimi short convolution weight shapes do not match");
    const int device = query.get_device();
    TORCH_CHECK(
        key.get_device() == device && value.get_device() == device &&
        query_state.get_device() == device &&
        key_state.get_device() == device &&
        value_state.get_device() == device &&
        query_weight.get_device() == device &&
        key_weight.get_device() == device &&
        value_weight.get_device() == device,
        "Kimi short convolution tensors must share one device");
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((channels + 255) / 256, 3);
    if (query_weight.scalar_type() == at::kBFloat16) {
        kimi_short_conv3_kernel<<<grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(
                query.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                key.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                value.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                query_state.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                key_state.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                value_state.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                query_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                key_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                value_weight.data_ptr<at::BFloat16>()),
            channels,
            history);
    } else {
        kimi_short_conv3_kernel<<<grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(
                query.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                key.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                value.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                query_state.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                key_state.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                value_state.data_ptr<at::BFloat16>()),
            query_weight.data_ptr<float>(),
            key_weight.data_ptr<float>(),
            value_weight.data_ptr<float>(),
            channels,
            history);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

__global__ void kimi_kda_prepare_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ gate,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query_norm,
    float* __restrict__ key_norm,
    float* __restrict__ decay,
    const int heads,
    const int key_dim,
    const float lower_bound)
{
    const int head = blockIdx.x;
    const int item = threadIdx.x;
    if (head >= heads) return;
    extern __shared__ float shared[];
    float* q_square = shared;
    float* k_square = shared + blockDim.x;
    float q = 0.f;
    float k = 0.f;
    if (item < key_dim) {
        const long offset = (long)head * key_dim + item;
        q = __bfloat162float(query[offset]);
        k = __bfloat162float(key[offset]);
    }
    q_square[item] = q * q;
    k_square[item] = k * k;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (item < stride) {
            q_square[item] += q_square[item + stride];
            k_square[item] += k_square[item + stride];
        }
        __syncthreads();
    }
    if (item < key_dim) {
        const long offset = (long)head * key_dim + item;
        const float q_scale = rsqrtf(q_square[0] + 1e-6f);
        const float k_scale = rsqrtf(k_square[0] + 1e-6f);
        query_norm[offset] = q * q_scale;
        key_norm[offset] = k * k_scale;
        const float a = expf(a_log[head]);
        const float raw = a * (
            __bfloat162float(gate[offset]) + dt_bias[offset]);
        const float sigmoid = 1.f / (1.f + expf(-raw));
        decay[offset] = expf(lower_bound * sigmoid);
    }
}

constexpr int KIMI_KDA_VALUES_PER_BLOCK = 4;

__global__ void kimi_kda_update_kernel(
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ query_norm,
    const float* __restrict__ key_norm,
    const float* __restrict__ decay,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    const int heads,
    const int key_dim,
    const int value_dim)
{
    const int head = blockIdx.y;
    const int value_start = blockIdx.x * KIMI_KDA_VALUES_PER_BLOCK;
    const int item = threadIdx.x;
    if (head >= heads) return;
    extern __shared__ float shared[];
    float* prediction = shared;
    float* old_output =
        prediction + KIMI_KDA_VALUES_PER_BLOCK * blockDim.x;
    float* key_query =
        old_output + KIMI_KDA_VALUES_PER_BLOCK * blockDim.x;
    float* deltas = key_query + blockDim.x;

    const long qk_offset = (long)head * key_dim + item;
    const float q = item < key_dim ? query_norm[qk_offset] : 0.f;
    const float k = item < key_dim ? key_norm[qk_offset] : 0.f;
    const float d = item < key_dim ? decay[qk_offset] : 0.f;
    key_query[item] = q * k;
    #pragma unroll
    for (int row = 0; row < KIMI_KDA_VALUES_PER_BLOCK; ++row) {
        const int value_index = value_start + row;
        float current = 0.f;
        if (value_index < value_dim && item < key_dim) {
            const long state_offset =
                ((long)head * value_dim + value_index) * key_dim + item;
            current = state[state_offset] * d;
            state[state_offset] = current;
        }
        prediction[row * blockDim.x + item] = current * k;
        old_output[row * blockDim.x + item] = current * q;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (item < stride) {
            key_query[item] += key_query[item + stride];
            #pragma unroll
            for (int row = 0; row < KIMI_KDA_VALUES_PER_BLOCK; ++row) {
                const int base = row * blockDim.x + item;
                prediction[base] += prediction[base + stride];
                old_output[base] += old_output[base + stride];
            }
        }
        __syncthreads();
    }

    if (item < KIMI_KDA_VALUES_PER_BLOCK) {
        const int value_index = value_start + item;
        float delta = 0.f;
        if (value_index < value_dim) {
            const float beta_value = 1.f / (1.f + expf(-beta[head]));
            const float source = __bfloat162float(
                value[(long)head * value_dim + value_index]);
            delta = (source - prediction[item * blockDim.x]) * beta_value;
        }
        deltas[item] = delta;
    }
    __syncthreads();

    #pragma unroll
    for (int row = 0; row < KIMI_KDA_VALUES_PER_BLOCK; ++row) {
        const int value_index = value_start + row;
        if (value_index < value_dim && item < key_dim) {
            const long state_offset =
                ((long)head * value_dim + value_index) * key_dim + item;
            state[state_offset] += deltas[row] * k;
        }
        if (item == 0 && value_index < value_dim) {
            const float result = (
                old_output[row * blockDim.x]
                + deltas[row] * key_query[0]
            ) * rsqrtf((float)key_dim);
            output[(long)head * value_dim + value_index] =
                __float2bfloat16_rn(result);
        }
    }
}

torch::Tensor kimi_kda_recurrent(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor a_log,
    torch::Tensor dt_bias,
    torch::Tensor state,
    torch::Tensor workspace,
    torch::Tensor output,
    double lower_bound)
{
    TORCH_CHECK(
        query.is_cuda() && key.is_cuda() && value.is_cuda() &&
        gate.is_cuda() && beta.is_cuda() && a_log.is_cuda() &&
        dt_bias.is_cuda() && state.is_cuda() && workspace.is_cuda() &&
        output.is_cuda(),
        "Kimi KDA tensors must be CUDA");
    TORCH_CHECK(
        query.scalar_type() == at::kBFloat16 &&
        key.scalar_type() == at::kBFloat16 &&
        value.scalar_type() == at::kBFloat16 &&
        gate.scalar_type() == at::kBFloat16 &&
        output.scalar_type() == at::kBFloat16,
        "Kimi KDA activations must be BF16");
    TORCH_CHECK(
        beta.scalar_type() == at::kFloat &&
        a_log.scalar_type() == at::kFloat &&
        dt_bias.scalar_type() == at::kFloat &&
        state.scalar_type() == at::kFloat &&
        workspace.scalar_type() == at::kFloat,
        "Kimi KDA parameters/state/workspace must be FP32");
    TORCH_CHECK(
        query.is_contiguous() && key.is_contiguous() &&
        value.is_contiguous() && gate.is_contiguous() &&
        beta.is_contiguous() && a_log.is_contiguous() &&
        dt_bias.is_contiguous() && state.is_contiguous() &&
        workspace.is_contiguous() && output.is_contiguous(),
        "Kimi KDA tensors must be contiguous");
    TORCH_CHECK(
        query.dim() == 2 && key.sizes() == query.sizes() &&
        gate.sizes() == query.sizes() && value.dim() == 2,
        "Kimi KDA q/k/g must be [H,K] and v must be [H,V]");
    const int heads = (int)query.size(0);
    const int key_dim = (int)query.size(1);
    const int value_dim = (int)value.size(1);
    TORCH_CHECK(
        value.size(0) == heads &&
        state.dim() == 3 &&
        state.size(0) == heads &&
        state.size(1) == value_dim &&
        state.size(2) == key_dim &&
        output.sizes() == value.sizes(),
        "Kimi KDA value/state/output shape mismatch");
    TORCH_CHECK(
        key_dim > 0 && key_dim <= 256 &&
        (key_dim & (key_dim - 1)) == 0,
        "Kimi KDA key dimension must be a power of two <= 256");
    TORCH_CHECK(
        workspace.numel() >= 3LL * heads * key_dim &&
        beta.numel() >= heads && a_log.numel() >= heads &&
        dt_bias.numel() >= (long)heads * key_dim,
        "Kimi KDA workspace or parameter shape mismatch");

    auto query_norm = workspace.narrow(
        0, 0, (long)heads * key_dim).view({heads, key_dim});
    auto key_norm = workspace.narrow(
        0, (long)heads * key_dim,
        (long)heads * key_dim).view({heads, key_dim});
    auto decay = workspace.narrow(
        0, 2LL * heads * key_dim,
        (long)heads * key_dim).view({heads, key_dim});
    auto stream = at::cuda::getCurrentCUDAStream();
    kimi_kda_prepare_kernel<<<
        heads,
        key_dim,
        2LL * key_dim * sizeof(float),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                query.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                key.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                gate.data_ptr<at::BFloat16>()),
            a_log.data_ptr<float>(),
            dt_bias.data_ptr<float>(),
            query_norm.data_ptr<float>(),
            key_norm.data_ptr<float>(),
            decay.data_ptr<float>(),
            heads,
            key_dim,
            (float)lower_bound);
    const int value_blocks =
        (value_dim + KIMI_KDA_VALUES_PER_BLOCK - 1)
        / KIMI_KDA_VALUES_PER_BLOCK;
    const size_t update_smem = (
        2 * KIMI_KDA_VALUES_PER_BLOCK * key_dim
        + key_dim
        + KIMI_KDA_VALUES_PER_BLOCK
    ) * sizeof(float);
    kimi_kda_update_kernel<<<
        dim3(value_blocks, heads),
        key_dim,
        update_smem,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                value.data_ptr<at::BFloat16>()),
            beta.data_ptr<float>(),
            query_norm.data_ptr<float>(),
            key_norm.data_ptr<float>(),
            decay.data_ptr<float>(),
            state.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            heads,
            key_dim,
            value_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void kimi_gated_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    const int heads,
    const int width,
    const float eps)
{
    const int head = blockIdx.x;
    const int item = threadIdx.x;
    if (head >= heads) return;
    extern __shared__ float reduction[];
    float square = 0.0f;
    if (item < width) {
        const float value = __bfloat162float(
            input[static_cast<long>(head) * width + item]);
        square = value * value;
    }
    reduction[item] = square;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (item < stride)
            reduction[item] += reduction[item + stride];
        __syncthreads();
    }
    if (item < width) {
        const long offset = static_cast<long>(head) * width + item;
        const float scale = rsqrtf(
            reduction[0] / static_cast<float>(width) + eps);
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(input[offset]) * scale);
        const __nv_bfloat16 weighted = __float2bfloat16_rn(
            __bfloat162float(normalized)
            * __bfloat162float(weight[item]));
        const float gate_value = __bfloat162float(
            __float2bfloat16_rn(
                1.0f / (
                    1.0f
                    + expf(-__bfloat162float(gate[offset])))));
        output[offset] = __float2bfloat16_rn(
            __bfloat162float(weighted) * gate_value);
    }
}

torch::Tensor kimi_gated_rmsnorm(
    torch::Tensor input,
    torch::Tensor gate,
    torch::Tensor weight,
    torch::Tensor output,
    double eps)
{
    TORCH_CHECK(
        input.is_cuda() && gate.is_cuda() &&
        weight.is_cuda() && output.is_cuda(),
        "Kimi gated RMSNorm tensors must be CUDA");
    TORCH_CHECK(
        input.scalar_type() == at::kBFloat16 &&
        gate.scalar_type() == at::kBFloat16 &&
        weight.scalar_type() == at::kBFloat16 &&
        output.scalar_type() == at::kBFloat16,
        "Kimi gated RMSNorm currently requires BF16");
    TORCH_CHECK(
        input.is_contiguous() && gate.is_contiguous() &&
        weight.is_contiguous() && output.is_contiguous(),
        "Kimi gated RMSNorm tensors must be contiguous");
    TORCH_CHECK(
        input.dim() == 2 && gate.sizes() == input.sizes() &&
        output.sizes() == input.sizes() &&
        weight.dim() == 1 && weight.size(0) == input.size(1),
        "Kimi gated RMSNorm shapes do not match");
    const int heads = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    TORCH_CHECK(
        width > 0 && width <= 256 &&
        (width & (width - 1)) == 0,
        "Kimi gated RMSNorm width must be a power of two <= 256");
    const int device = input.get_device();
    TORCH_CHECK(
        gate.get_device() == device &&
        weight.get_device() == device &&
        output.get_device() == device,
        "Kimi gated RMSNorm tensors must share one device");
    auto stream = at::cuda::getCurrentCUDAStream();
    kimi_gated_rmsnorm_kernel<<<
        heads,
        width,
        width * sizeof(float),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                gate.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            heads,
            width,
            static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// ---- Stable-slot grouped VQ MLP (top-k <= 8, BF16 I/O) ----

constexpr int MAX_SLOT_EXPERTS = 16;

template <typename idx_t>
struct IndexPointerPack {
    const idx_t* ptrs[MAX_SLOT_EXPERTS];
};

template <typename scalar_t>
struct CodebookPointerPack {
    const scalar_t* ptrs[MAX_SLOT_EXPERTS];
};

struct SlotIntPack {
    int values[MAX_SLOT_EXPERTS];
};

template <typename scalar_t>
__device__ __forceinline__ float vq_scalar_to_float(const scalar_t* p) {
    return (float)(*p);
}

template <>
__device__ __forceinline__ float vq_scalar_to_float<__nv_bfloat16>(
    const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t vq_float_to_scalar(float v) {
    return (scalar_t)v;
}

template <>
__device__ __forceinline__ __nv_bfloat16
vq_float_to_scalar<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

template <typename scalar_t>
__device__ __forceinline__ float vq_block_dot(
    const scalar_t* cb, const scalar_t* x, int D) {
    float part = 0.f;
    #pragma unroll 8
    for (int i = 0; i < D; ++i)
        part = fmaf(
            vq_scalar_to_float(cb + i),
            vq_scalar_to_float(x + i),
            part);
    return part;
}

template <>
__device__ __forceinline__ float vq_block_dot<__nv_bfloat16>(
    const __nv_bfloat16* cb, const __nv_bfloat16* x, int D) {
    const auto* cb2 = reinterpret_cast<const __nv_bfloat162*>(cb);
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    float part = 0.f;
    #pragma unroll 4
    for (int i = 0; i < D / 2; ++i) {
        const float2 cv = __bfloat1622float2(cb2[i]);
        const float2 xv = __bfloat1622float2(x2[i]);
        part = fmaf(cv.x, xv.x, part);
        part = fmaf(cv.y, xv.y, part);
    }
    return part;
}

template <typename idx_t, typename scalar_t>
__global__ void vq_gemv_slots_kernel(
    const scalar_t* __restrict__ x,
    const IndexPointerPack<idx_t> indices,
    const CodebookPointerPack<scalar_t> codebooks,
    const SlotIntPack blocks,
    const SlotIntPack code_dims,
    scalar_t* __restrict__ out,
    const int N, const int R, const int C,
    const long x_stride_n)
{
    const int n = blockIdx.y;
    if (n >= N) return;
    const int B = blocks.values[n];
    const int D = code_dims.values[n];
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    extern __shared__ unsigned char raw_smem[];
    scalar_t* xs = reinterpret_cast<scalar_t*>(raw_smem);
    const scalar_t* xrow = x + (long)n * x_stride_n;
    for (int i = threadIdx.y * 32 + threadIdx.x;
         i < C; i += 32 * ROWS_PER_BLOCK)
        xs[i] = xrow[i];
    __syncthreads();
    if (r >= R) return;

    const idx_t* irow = indices.ptrs[n] + (long)r * B;
    const scalar_t* cb = codebooks.ptrs[n];
    float acc = 0.f;
    for (int b = threadIdx.x; b < B; b += 32) {
        const scalar_t* crow = cb + (long)irow[b] * D;
        const scalar_t* xb = xs + b * D;
        acc += vq_block_dot(crow, xb, D);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (threadIdx.x == 0)
        out[(long)n * R + r] = vq_float_to_scalar<scalar_t>(acc);
}

template <typename idx_t, typename scalar_t>
void launch_vq_gemv_slots(
    const torch::Tensor& x,
    const std::vector<torch::Tensor>& indices,
    const std::vector<torch::Tensor>& codebooks,
    torch::Tensor& out)
{
    const int N = (int)indices.size();
    const int R = (int)indices[0].size(0);
    const int C = (int)x.size(1);
    IndexPointerPack<idx_t> index_pack{};
    CodebookPointerPack<scalar_t> codebook_pack{};
    SlotIntPack block_pack{};
    SlotIntPack dim_pack{};
    for (int n = 0; n < N; ++n) {
        index_pack.ptrs[n] =
            reinterpret_cast<const idx_t*>(indices[n].data_ptr());
        codebook_pack.ptrs[n] =
            reinterpret_cast<const scalar_t*>(codebooks[n].data_ptr());
        block_pack.values[n] = (int)indices[n].size(1);
        dim_pack.values[n] = (int)codebooks[n].size(1);
    }
    const long x_stride_n = x.size(0) == 1 ? 0 : x.stride(0);
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid(
        (unsigned)((R + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
        (unsigned)N);
    const size_t smem = (size_t)C * sizeof(scalar_t);
    auto stream = at::cuda::getCurrentCUDAStream();
    vq_gemv_slots_kernel<idx_t, scalar_t><<<grid, block, smem, stream>>>(
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        index_pack,
        codebook_pack,
        block_pack,
        dim_pack,
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        N, R, C, x_stride_n);
}

void vq_gemv_slots_out(
    torch::Tensor x,
    const std::vector<torch::Tensor>& indices,
    const std::vector<torch::Tensor>& codebooks,
    torch::Tensor out)
{
    TORCH_CHECK(!indices.empty() && indices.size() <= MAX_SLOT_EXPERTS,
                "slot expert count must be in [1,8]");
    TORCH_CHECK(codebooks.size() == indices.size(),
                "slot codebook count mismatch");
    TORCH_CHECK(x.is_cuda() && out.is_cuda(),
                "x/out must be CUDA");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16 &&
        out.scalar_type() == at::kBFloat16,
        "slot VQ x/out must be bfloat16");
    TORCH_CHECK(x.stride(1) == 1 && out.is_contiguous(),
                "slot VQ x rows and out must be contiguous");
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2,
                "slot VQ tensors must be 2D");
    const auto dtype = indices[0].scalar_type();
    TORCH_CHECK(dtype == at::kByte || dtype == at::kUInt16,
                "slot indices must be uint8 or uint16");
    const auto rows = indices[0].size(0);
    TORCH_CHECK((long)indices.size() == out.size(0) && rows == out.size(1),
                "slot VQ output shape mismatch");
    TORCH_CHECK(x.size(0) == 1 || x.size(0) == (long)indices.size(),
                "slot VQ x batch mismatch");
    for (int n = 0; n < (int)indices.size(); ++n) {
        const auto& idx = indices[n];
        const auto& cb = codebooks[n];
        TORCH_CHECK(
            idx.is_cuda() && idx.is_contiguous() &&
            idx.scalar_type() == dtype &&
            idx.dim() == 2 && idx.size(0) == rows,
            "slot index row/dtype mismatch");
        TORCH_CHECK(
            cb.is_cuda() && cb.is_contiguous() &&
            cb.scalar_type() == at::kBFloat16 &&
            cb.dim() == 2,
            "slot codebook must be contiguous CUDA BF16 [K,D]");
        TORCH_CHECK(
            idx.size(1) * cb.size(1) == x.size(1),
            "slot VQ input width mismatch");
    }
    if (dtype == at::kByte) {
        launch_vq_gemv_slots<uint8_t, __nv_bfloat16>(
            x, indices, codebooks, out);
    } else {
        launch_vq_gemv_slots<uint16_t, __nv_bfloat16>(
            x, indices, codebooks, out);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void swiglu_bf16_inplace_kernel(
    __nv_bfloat16* __restrict__ h,
    const int N, const int inter, const float limit)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * inter;
    if (i >= total) return;
    const int n = i / inter;
    const int m = i - n * inter;
    const long row = (long)n * 2 * inter;
    float gate = __bfloat162float(h[row + m]);
    float up = __bfloat162float(h[row + inter + m]);
    if (limit > 0.f) {
        gate = fminf(gate, limit);
        up = fminf(fmaxf(up, -limit), limit);
    }
    const float silu = gate / (1.f + expf(-gate));
    h[row + m] = __float2bfloat16_rn(silu * up);
}

__global__ void gated_activation_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    const int count,
    const int activation,
    const float beta,
    const float linear_beta)
{
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count)
        return;
    const float gate_value = __bfloat162float(gate[item]);
    const float up_value = __bfloat162float(up[item]);
    if (activation == 0) {
        const __nv_bfloat16 silu = __float2bfloat16_rn(
            gate_value / (1.0f + expf(-gate_value)));
        output[item] = __float2bfloat16_rn(
            __bfloat162float(silu) * up_value);
        return;
    }
    const float activated = (
        beta
        * tanhf(gate_value / beta)
        / (1.0f + expf(-gate_value)));
    float bounded_up = up_value;
    if (linear_beta > 0.0f)
        bounded_up = linear_beta * tanhf(up_value / linear_beta);
    output[item] = __float2bfloat16_rn(activated * bounded_up);
}

torch::Tensor gated_activation_bf16(
    torch::Tensor gate,
    torch::Tensor up,
    long activation,
    double beta,
    double linear_beta,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        gate.is_cuda() && up.is_cuda() &&
        gate.scalar_type() == at::kBFloat16 &&
        up.scalar_type() == at::kBFloat16 &&
        gate.is_contiguous() && up.is_contiguous() &&
        gate.sizes() == up.sizes() &&
        gate.get_device() == up.get_device(),
        "Gated activation needs colocated contiguous BF16 tensors");
    TORCH_CHECK(
        activation == 0 || activation == 1,
        "Gated activation kind must be 0 (SiLU) or 1 (SiTU)");
    TORCH_CHECK(
        activation == 0 || beta > 0.0,
        "SiTU beta must be positive");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty_like(gate);
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kBFloat16 &&
        output.is_contiguous() &&
        output.sizes() == gate.sizes() &&
        output.get_device() == gate.get_device(),
        "Gated activation output must match input");
    const int count = static_cast<int>(gate.numel());
    auto stream = at::cuda::getCurrentCUDAStream();
    gated_activation_bf16_kernel<<<
        (count + 255) / 256,
        256,
        0,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                gate.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                up.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            count,
            static_cast<int>(activation),
            static_cast<float>(beta),
            static_cast<float>(linear_beta));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void weighted_sum_bf16_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ result,
    const int N, const int D)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    #pragma unroll
    for (int n = 0; n < MAX_SLOT_EXPERTS; ++n) {
        if (n < N)
            acc = fmaf(
                __bfloat162float(rows[(long)n * D + d]),
                weights[n],
                acc);
    }
    result[d] = __float2bfloat16_rn(acc);
}

__global__ void weighted_sum_f32_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const float* __restrict__ weights,
    float* __restrict__ result,
    const int N, const int D)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    #pragma unroll
    for (int n = 0; n < MAX_SLOT_EXPERTS; ++n) {
        if (n < N)
            acc = fmaf(
                __bfloat162float(rows[(long)n * D + d]),
                weights[n],
                acc);
    }
    result[d] = acc;
}

torch::Tensor moe_mlp_slots(
    torch::Tensor x,
    const std::vector<torch::Tensor>& gu_indices,
    const std::vector<torch::Tensor>& gu_codebooks,
    const std::vector<torch::Tensor>& dn_indices,
    const std::vector<torch::Tensor>& dn_codebooks,
    torch::Tensor weights,
    double limit,
    torch::Tensor hidden_workspace,
    torch::Tensor out_workspace,
    torch::Tensor result)
{
    const int N = (int)gu_indices.size();
    TORCH_CHECK(N > 0 && N <= MAX_SLOT_EXPERTS &&
                dn_indices.size() == gu_indices.size(),
                "GU/DN slot expert count mismatch");
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.numel() == N,
        "slot route weights must be contiguous float32 [N]");
    TORCH_CHECK(
        hidden_workspace.is_cuda() &&
        hidden_workspace.scalar_type() == at::kBFloat16 &&
        hidden_workspace.is_contiguous() &&
        hidden_workspace.size(0) == N,
        "hidden workspace must be contiguous BF16 [N,2I]");
    TORCH_CHECK(
        out_workspace.is_cuda() &&
        out_workspace.scalar_type() == at::kBFloat16 &&
        out_workspace.is_contiguous() &&
        out_workspace.size(0) == N,
        "out workspace must be contiguous BF16 [N,D]");
    TORCH_CHECK(
        result.is_cuda() &&
        (
            result.scalar_type() == at::kBFloat16 ||
            result.scalar_type() == at::kFloat
        ) &&
        result.is_contiguous() && result.dim() == 1,
        "result must be contiguous BF16 or float32 [D]");

    vq_gemv_slots_out(x, gu_indices, gu_codebooks, hidden_workspace);
    const int inter = (int)hidden_workspace.size(1) / 2;
    const int activation_items = N * inter;
    auto stream = at::cuda::getCurrentCUDAStream();
    swiglu_bf16_inplace_kernel<<<
        (activation_items + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(
            hidden_workspace.data_ptr<at::BFloat16>()),
        N, inter, (float)limit);
    auto activation = hidden_workspace.narrow(1, 0, inter);
    vq_gemv_slots_out(
        activation, dn_indices, dn_codebooks, out_workspace);
    const int D = (int)out_workspace.size(1);
    TORCH_CHECK(result.numel() == D, "slot result width mismatch");
    if (result.scalar_type() == at::kFloat) {
        weighted_sum_f32_kernel<<<
            (D + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    out_workspace.data_ptr<at::BFloat16>()),
                weights.data_ptr<float>(),
                result.data_ptr<float>(),
                N, D);
    } else {
        weighted_sum_bf16_kernel<<<
            (D + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    out_workspace.data_ptr<at::BFloat16>()),
                weights.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(
                    result.data_ptr<at::BFloat16>()),
                N, D);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

// ---- Device-routed stable VQ MLP for full-resident Expert Parallel ----
//
// metadata is [10,E] int64 on the owner GPU:
//   GU index pointer, codebook pointer, blocks, code dim, index dtype tag,
//   DN index pointer, codebook pointer, blocks, code dim, index dtype tag.
// A zero index pointer means that the expert belongs to another rank.  The
// Top-K IDs stay on CUDA; every rank processes its owned positions directly.

constexpr int ROUTED_META_ROWS = 10;

void ensure_peer_access(
    const int current,
    const int peer,
    const char* operation)
{
    if (current == peer) return;
    constexpr int MAX_CACHED_DEVICES = 64;
    static bool peer_enabled[
        MAX_CACHED_DEVICES][MAX_CACHED_DEVICES] = {};
    TORCH_CHECK(
        current >= 0 && current < MAX_CACHED_DEVICES &&
        peer >= 0 && peer < MAX_CACHED_DEVICES,
        operation, " CUDA device index out of cache range");
    if (peer_enabled[current][peer]) return;
    int can_access = 0;
    const auto query_status = cudaDeviceCanAccessPeer(
        &can_access, current, peer);
    TORCH_CHECK(
        query_status == cudaSuccess && can_access,
        operation, " requires CUDA peer access");
    const auto enable_status = cudaDeviceEnablePeerAccess(peer, 0);
    if (enable_status == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
    } else {
        TORCH_CHECK(
            enable_status == cudaSuccess,
            "failed to enable ", operation, " peer access: ",
            cudaGetErrorString(enable_status));
    }
    peer_enabled[current][peer] = true;
}

#include "codegemm_vq.cuh"

// One peer-reading launch replaces three tiny cross-device copies for the
// full-resident expert path. x keeps the model's FP32 -> BF16 boundary.
template <typename input_t>
__global__ void expert_dispatch_pack_kernel(
    const input_t* __restrict__ x,
    const int64_t* __restrict__ route_ids,
    const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ x_out,
    int64_t* __restrict__ route_ids_out,
    float* __restrict__ weights_out,
    const int hidden,
    const int K)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden)
        x_out[i] = __float2bfloat16_rn(
            vq_scalar_to_float(x + i));
    if (i < K) {
        route_ids_out[i] = route_ids[i];
        weights_out[i] = weights[i];
    }
}

void expert_dispatch_pack(
    torch::Tensor x,
    torch::Tensor route_ids,
    torch::Tensor weights,
    torch::Tensor x_out,
    torch::Tensor route_ids_out,
    torch::Tensor weights_out)
{
    TORCH_CHECK(
        x.is_cuda() &&
        (
            x.scalar_type() == at::kFloat ||
            x.scalar_type() == at::kBFloat16
        ) &&
        x.is_contiguous() && x.dim() == 2 && x.size(0) == 1,
        "expert dispatch x must be contiguous CUDA FP32/BF16 [1,D]");
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == at::kLong &&
        route_ids.is_contiguous() && route_ids.dim() == 1,
        "expert dispatch IDs must be contiguous CUDA int64 [K]");
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.sizes() == route_ids.sizes(),
        "expert dispatch weights must be contiguous CUDA FP32 [K]");
    TORCH_CHECK(
        x_out.is_cuda() && x_out.scalar_type() == at::kBFloat16 &&
        x_out.is_contiguous() && x_out.sizes() == x.sizes(),
        "expert dispatch x output must be contiguous CUDA BF16 [1,D]");
    TORCH_CHECK(
        route_ids_out.is_cuda() &&
        route_ids_out.scalar_type() == at::kLong &&
        route_ids_out.is_contiguous() &&
        route_ids_out.sizes() == route_ids.sizes(),
        "expert dispatch ID output shape mismatch");
    TORCH_CHECK(
        weights_out.is_cuda() &&
        weights_out.scalar_type() == at::kFloat &&
        weights_out.is_contiguous() &&
        weights_out.sizes() == weights.sizes(),
        "expert dispatch weight output shape mismatch");
    const int source = x.get_device();
    const int target = x_out.get_device();
    TORCH_CHECK(
        route_ids.get_device() == source &&
        weights.get_device() == source,
        "expert dispatch sources must share one CUDA device");
    TORCH_CHECK(
        route_ids_out.get_device() == target &&
        weights_out.get_device() == target,
        "expert dispatch outputs must share one CUDA device");

    int current = -1;
    const auto current_status = cudaGetDevice(&current);
    TORCH_CHECK(
        current_status == cudaSuccess && current == target,
        "expert dispatch must run under the output CUDA device");
    ensure_peer_access(target, source, "expert dispatch");
    const int hidden = static_cast<int>(x.numel());
    const int K = static_cast<int>(route_ids.numel());
    const int count = std::max(hidden, K);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kBFloat16) {
        expert_dispatch_pack_kernel<<<
            (count + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    x.data_ptr<at::BFloat16>()),
                route_ids.data_ptr<int64_t>(),
                weights.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(
                    x_out.data_ptr<at::BFloat16>()),
                route_ids_out.data_ptr<int64_t>(),
                weights_out.data_ptr<float>(),
                hidden,
                K);
    } else {
        expert_dispatch_pack_kernel<<<
            (count + 255) / 256, 256, 0, stream>>>(
                x.data_ptr<float>(),
                route_ids.data_ptr<int64_t>(),
                weights.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(
                    x_out.data_ptr<at::BFloat16>()),
                route_ids_out.data_ptr<int64_t>(),
                weights_out.data_ptr<float>(),
                hidden,
                K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
__global__ void tp_peer_copy_kernel(
    const scalar_t* __restrict__ source,
    scalar_t* __restrict__ destination,
    const long count)
{
    for (
        long index =
            static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < count;
        index += static_cast<long>(blockDim.x) * gridDim.x
    ) {
        destination[index] = source[index];
    }
}

void tp_peer_copy(
    torch::Tensor source,
    torch::Tensor destination)
{
    TORCH_CHECK(
        source.is_cuda() && destination.is_cuda() &&
        source.is_contiguous() && destination.is_contiguous() &&
        source.sizes() == destination.sizes() &&
        source.scalar_type() == destination.scalar_type(),
        "TP peer copy tensors must be matching contiguous CUDA tensors");
    TORCH_CHECK(
        source.scalar_type() == at::kFloat ||
        source.scalar_type() == at::kLong ||
        source.scalar_type() == at::kBFloat16,
        "TP peer copy currently supports float32, bfloat16 and int64");
    const int source_device = source.get_device();
    const int target_device = destination.get_device();
    int current = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    TORCH_CHECK(
        current == target_device || current == source_device,
        "TP peer copy must run under its source or destination CUDA device");
    ensure_peer_access(
        current,
        current == target_device ? source_device : target_device,
        "TP peer copy");
    const long count = source.numel();
    const int blocks = static_cast<int>(
        std::min<long>((count + 255) / 256, 4096));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (source.scalar_type() == at::kFloat) {
        tp_peer_copy_kernel<<<blocks, 256, 0, stream>>>(
            source.data_ptr<float>(),
            destination.data_ptr<float>(),
            count);
    } else if (source.scalar_type() == at::kLong) {
        tp_peer_copy_kernel<<<blocks, 256, 0, stream>>>(
            source.data_ptr<int64_t>(),
            destination.data_ptr<int64_t>(),
            count);
    } else {
        tp_peer_copy_kernel<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                source.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                destination.data_ptr<at::BFloat16>()),
            count);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool POSITION_FROM_POINTER>
__global__ void tp_attention_dispatch_kernel(
    const float* __restrict__ source_q,
    float* __restrict__ destination_q,
    const long q_count,
    const float* __restrict__ source_c,
    float* __restrict__ destination_c,
    const long c_count,
    const float* __restrict__ source_k,
    float* __restrict__ destination_k,
    const long k_count,
    const int64_t* __restrict__ source_position,
    int64_t* __restrict__ destination_position,
    const int64_t position_value)
{
    for (
        long index =
            static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < q_count;
        index += static_cast<long>(blockDim.x) * gridDim.x
    ) {
        destination_q[index] = source_q[index];
    }
    for (
        long index =
            static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < c_count;
        index += static_cast<long>(blockDim.x) * gridDim.x
    ) {
        destination_c[index] = source_c[index];
    }
    for (
        long index =
            static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < k_count;
        index += static_cast<long>(blockDim.x) * gridDim.x
    ) {
        destination_k[index] = source_k[index];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        destination_position[0] = (
            POSITION_FROM_POINTER
                ? source_position[0]
                : position_value);
    }
}

void tp_attention_peer_dispatch(
    torch::Tensor source_q,
    torch::Tensor source_c,
    torch::Tensor source_k,
    torch::Tensor source_position,
    torch::Tensor destination_q,
    torch::Tensor destination_c,
    torch::Tensor destination_k,
    torch::Tensor destination_position)
{
    const torch::Tensor sources[] = {
        source_q,
        source_c,
        source_k,
    };
    const torch::Tensor destinations[] = {
        destination_q,
        destination_c,
        destination_k,
    };
    for (int index = 0; index < 3; ++index) {
        TORCH_CHECK(
            sources[index].is_cuda() &&
            destinations[index].is_cuda() &&
            sources[index].scalar_type() == at::kFloat &&
            destinations[index].scalar_type() == at::kFloat &&
            sources[index].is_contiguous() &&
            destinations[index].is_contiguous() &&
            sources[index].sizes() == destinations[index].sizes(),
            "Attention TP peer tensors must be matching contiguous "
            "CUDA float32 tensors");
    }
    TORCH_CHECK(
        source_position.is_cuda() &&
        destination_position.is_cuda() &&
        source_position.scalar_type() == at::kLong &&
        destination_position.scalar_type() == at::kLong &&
        source_position.is_contiguous() &&
        destination_position.is_contiguous() &&
        source_position.numel() == 1 &&
        destination_position.numel() == 1,
        "Attention TP position tensors must be scalar CUDA int64");
    const int source_device = source_q.get_device();
    const int target_device = destination_q.get_device();
    TORCH_CHECK(
        source_c.get_device() == source_device &&
        source_k.get_device() == source_device &&
        source_position.get_device() == source_device &&
        destination_c.get_device() == target_device &&
        destination_k.get_device() == target_device &&
        destination_position.get_device() == target_device,
        "Attention TP source and destination tensors must each share "
        "one device");
    int current = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    TORCH_CHECK(
        current == target_device,
        "Attention TP peer dispatch must run under the destination device");
    ensure_peer_access(
        target_device,
        source_device,
        "Attention TP source dispatch");
    const long q_count = (
        source_q.data_ptr() == destination_q.data_ptr()
            ? 0
            : source_q.numel());
    const long c_count = (
        source_c.data_ptr() == destination_c.data_ptr()
            ? 0
            : source_c.numel());
    const long k_count = (
        source_k.data_ptr() == destination_k.data_ptr()
            ? 0
            : source_k.numel());
    const long count = std::max(q_count, std::max(c_count, k_count));
    const int blocks = static_cast<int>(
        std::min<long>((count + 255) / 256, 4096));
    auto stream = at::cuda::getCurrentCUDAStream();
    tp_attention_dispatch_kernel<true><<<blocks, 256, 0, stream>>>(
        source_q.data_ptr<float>(),
        destination_q.data_ptr<float>(),
        q_count,
        source_c.data_ptr<float>(),
        destination_c.data_ptr<float>(),
        c_count,
        source_k.data_ptr<float>(),
        destination_k.data_ptr<float>(),
        k_count,
        source_position.data_ptr<int64_t>(),
        destination_position.data_ptr<int64_t>(),
        0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tp_attention_source_pack(
    torch::Tensor source_q,
    torch::Tensor source_c,
    torch::Tensor source_k,
    torch::Tensor destination_q,
    torch::Tensor destination_c,
    torch::Tensor destination_k,
    torch::Tensor destination_position,
    int64_t position)
{
    const torch::Tensor sources[] = {
        source_q,
        source_c,
        source_k,
    };
    const torch::Tensor destinations[] = {
        destination_q,
        destination_c,
        destination_k,
    };
    const int device = source_q.get_device();
    for (int index = 0; index < 3; ++index) {
        TORCH_CHECK(
            sources[index].is_cuda() &&
            destinations[index].is_cuda() &&
            sources[index].scalar_type() == at::kFloat &&
            destinations[index].scalar_type() == at::kFloat &&
            sources[index].is_contiguous() &&
            destinations[index].is_contiguous() &&
            sources[index].sizes() == destinations[index].sizes() &&
            sources[index].get_device() == device &&
            destinations[index].get_device() == device,
            "Attention TP source-pack tensors must be matching contiguous "
            "CUDA float32 tensors on one device");
    }
    TORCH_CHECK(
        destination_position.is_cuda() &&
        destination_position.scalar_type() == at::kLong &&
        destination_position.is_contiguous() &&
        destination_position.numel() == 1 &&
        destination_position.get_device() == device,
        "Attention TP source-pack position must be scalar CUDA int64 "
        "on the input device");
    int current = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    TORCH_CHECK(
        current == device,
        "Attention TP source pack must run under the input device");
    const long q_count = (
        source_q.data_ptr() == destination_q.data_ptr()
            ? 0
            : source_q.numel());
    const long c_count = (
        source_c.data_ptr() == destination_c.data_ptr()
            ? 0
            : source_c.numel());
    const long k_count = (
        source_k.data_ptr() == destination_k.data_ptr()
            ? 0
            : source_k.numel());
    const long count = std::max(q_count, std::max(c_count, k_count));
    // All three sources may already alias their fixed graph buffers.  We
    // still launch one block to publish the new position scalar; a zero-block
    // launch is an invalid CUDA configuration.
    const int blocks = static_cast<int>(
        std::max<long>(
            1,
            std::min<long>((count + 255) / 256, 4096)));
    auto stream = at::cuda::getCurrentCUDAStream();
    tp_attention_dispatch_kernel<false><<<blocks, 256, 0, stream>>>(
        source_q.data_ptr<float>(),
        destination_q.data_ptr<float>(),
        q_count,
        source_c.data_ptr<float>(),
        destination_c.data_ptr<float>(),
        c_count,
        source_k.data_ptr<float>(),
        destination_k.data_ptr<float>(),
        k_count,
        nullptr,
        destination_position.data_ptr<int64_t>(),
        position);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ int routed_index_value(
    const int64_t address,
    const int dtype_tag,
    const long offset)
{
    const uintptr_t raw = static_cast<uintptr_t>(address);
    if (dtype_tag == 0)
        return static_cast<int>(
            reinterpret_cast<const uint8_t*>(raw)[offset]);
    if (dtype_tag == 1)
        return static_cast<int>(
            reinterpret_cast<const uint16_t*>(raw)[offset]);
    const auto* bytes = reinterpret_cast<const uint8_t*>(raw);
    if (dtype_tag == 2) {
        // Two little-endian 12-bit indices are stored in three bytes.
        const long base = (offset >> 1) * 3;
        if ((offset & 1) == 0)
            return static_cast<int>(
                bytes[base] | ((bytes[base + 1] & 0x0f) << 8));
        return static_cast<int>(
            (bytes[base + 1] >> 4) | (bytes[base + 2] << 4));
    }
    // Four little-endian 14-bit indices are stored in seven bytes.  Assemble
    // explicitly so the read remains valid for arbitrary byte alignment.
    const long base = (offset >> 2) * 7;
    unsigned long long word = 0;
    #pragma unroll
    for (int byte = 0; byte < 7; ++byte)
        word |= static_cast<unsigned long long>(bytes[base + byte])
                << (8 * byte);
    return static_cast<int>(
        (word >> (14 * (offset & 3))) & 0x3fffu);
}

__device__ __forceinline__ float vq_block_dot4_bf16(
    const __nv_bfloat16* cb,
    const __nv_bfloat16* x)
{
    const auto* cb2 = reinterpret_cast<const __nv_bfloat162*>(cb);
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const float2 cv0 = __bfloat1622float2(cb2[0]);
    const float2 xv0 = __bfloat1622float2(x2[0]);
    const float2 cv1 = __bfloat1622float2(cb2[1]);
    const float2 xv1 = __bfloat1622float2(x2[1]);
    float part = fmaf(cv0.x, xv0.x, 0.f);
    part = fmaf(cv0.y, xv0.y, part);
    part = fmaf(cv1.x, xv1.x, part);
    part = fmaf(cv1.y, xv1.y, part);
    return part;
}

__device__ __forceinline__ float vq_gemv_routed_row(
    const int64_t index_address,
    const __nv_bfloat16* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ input,
    const int blocks,
    const int vector,
    const int dtype_tag,
    const long index_row)
{
    float value = 0.f;
    if (dtype_tag == 0 && vector == 4) {
        const auto* indices = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(index_address));
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const int code = static_cast<int>(
                indices[index_row + block]);
            value += vq_block_dot4_bf16(
                codebook + (long)code * 4,
                input + block * 4);
        }
    } else if (dtype_tag == 2 && (blocks & 1) == 0) {
        // Packed-12 stores two adjacent indices in three bytes.  One even
        // lane loads each pair and broadcasts the assembled word.
        const auto* indices = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(index_address));
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const unsigned active = __activemask();
            unsigned packed = 0;
            const int leader = threadIdx.x & ~1;
            if ((threadIdx.x & 1) == 0) {
                const long base = ((index_row + block) >> 1) * 3;
                packed =
                    static_cast<unsigned>(indices[base]) |
                    (static_cast<unsigned>(indices[base + 1]) << 8) |
                    (static_cast<unsigned>(indices[base + 2]) << 16);
            }
            packed = __shfl_sync(active, packed, leader);
            const int code = static_cast<int>(
                (packed >> ((threadIdx.x & 1) * 12)) & 0xfffu);
            value += vq_block_dot(
                codebook + (long)code * vector,
                input + block * vector,
                vector);
        }
    } else if (dtype_tag == 3 && (blocks & 3) == 0) {
        // Packed-14 stores four adjacent indices in seven bytes.  One lane
        // loads each group and broadcasts the same 56-bit word.
        const auto* indices = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(index_address));
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const unsigned active = __activemask();
            const int leader = threadIdx.x & ~3;
            unsigned low = 0;
            unsigned high = 0;
            if ((threadIdx.x & 3) == 0) {
                const long base = ((index_row + block) >> 2) * 7;
                unsigned long long packed = 0;
                #pragma unroll
                for (int byte = 0; byte < 7; ++byte)
                    packed |=
                        static_cast<unsigned long long>(
                            indices[base + byte])
                        << (8 * byte);
                low = static_cast<unsigned>(packed);
                high = static_cast<unsigned>(packed >> 32);
            }
            low = __shfl_sync(active, low, leader);
            high = __shfl_sync(active, high, leader);
            const unsigned long long packed =
                static_cast<unsigned long long>(low) |
                (static_cast<unsigned long long>(high) << 32);
            const int code = static_cast<int>(
                (packed >> (14 * (threadIdx.x & 3))) & 0x3fffu);
            value += vq_block_dot(
                codebook + (long)code * vector,
                input + block * vector,
                vector);
        }
    } else {
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const int code = routed_index_value(
                index_address,
                dtype_tag,
                index_row + block);
            value += vq_block_dot(
                codebook + (long)code * vector,
                input + block * vector,
                vector);
        }
    }
    return value;
}

struct RoutedBlockMetadata {
    int64_t index_address;
    int64_t codebook_address;
    int blocks;
    int vector;
    int dtype_tag;
    int valid;
};

template <int WARPS>
__global__ void vq_gemv_routed_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    __nv_bfloat16* __restrict__ out,
    const int K,
    const int E,
    const int meta_base,
    const int R,
    const int C,
    const long x_stride_n,
    const bool skip_p12,
    const int route_offset,
    const bool vector_input_copy)
{
    const int n = blockIdx.y + route_offset;
    if (n >= K) return;

    const int row =
        blockIdx.x * WARPS + threadIdx.y;
    extern __shared__ unsigned char raw_smem[];
    __shared__ RoutedBlockMetadata route_meta;
    auto* xs = reinterpret_cast<__nv_bfloat16*>(raw_smem);
    const __nv_bfloat16* xrow = x + (long)n * x_stride_n;
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    if (linear_thread == 0) {
        const int expert_id = static_cast<int>(route_ids[n]);
        route_meta.valid = 0;
        if (expert_id >= 0 && expert_id < E) {
            route_meta.index_address =
                metadata[(long)(meta_base + 0) * E + expert_id];
            if (route_meta.index_address != 0) {
                route_meta.codebook_address =
                    metadata[(long)(meta_base + 1) * E + expert_id];
                route_meta.blocks = static_cast<int>(
                    metadata[
                        (long)(meta_base + 2) * E + expert_id]);
                route_meta.vector = static_cast<int>(
                    metadata[
                        (long)(meta_base + 3) * E + expert_id]);
                route_meta.dtype_tag = static_cast<int>(
                    metadata[
                        (long)(meta_base + 4) * E + expert_id]);
                route_meta.valid = !(
                    skip_p12 &&
                    route_meta.dtype_tag == 2 &&
                    (
                        route_meta.vector == 4 ||
                        route_meta.vector == 8
                    )
                );
            }
        }
    }
    if (
        vector_input_copy &&
        (C & 7) == 0 &&
        (
            reinterpret_cast<uintptr_t>(xrow) &
            (alignof(uint4) - 1)
        ) == 0
    ) {
        const auto* x4 = reinterpret_cast<const uint4*>(xrow);
        auto* xs4 = reinterpret_cast<uint4*>(xs);
        for (int i = linear_thread;
             i < C / 8; i += 32 * WARPS)
            xs4[i] = x4[i];
    } else {
        for (int i = linear_thread; i < C; i += 32 * WARPS)
            xs[i] = xrow[i];
    }
    __syncthreads();
    if (!route_meta.valid || row >= R) return;

    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(route_meta.codebook_address));
    float value = vq_gemv_routed_row(
        route_meta.index_address,
        codebook,
        xs,
        route_meta.blocks,
        route_meta.vector,
        route_meta.dtype_tag,
        (long)row * route_meta.blocks);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, off);
    if (threadIdx.x == 0)
        out[(long)n * R + row] = __float2bfloat16_rn(value);
}

template <int WARPS>
inline void launch_vq_gemv_routed(
    const __nv_bfloat16* input,
    const int64_t* route_ids,
    const int64_t* metadata,
    __nv_bfloat16* output,
    const int top_k,
    const int expert_count,
    const int metadata_base,
    const int output_rows,
    const int input_cols,
    const long input_stride,
    const bool skip_p12,
    const int route_offset,
    const int active_count,
    const bool vector_input_copy,
    cudaStream_t stream)
{
    vq_gemv_routed_kernel<WARPS><<<
        dim3(
            (unsigned)((output_rows + WARPS - 1) / WARPS),
            (unsigned)active_count),
        dim3(32, WARPS),
        (size_t)input_cols * sizeof(__nv_bfloat16),
        stream>>>(
            input,
            route_ids,
            metadata,
            output,
            top_k,
            expert_count,
            metadata_base,
            output_rows,
            input_cols,
            input_stride,
            skip_p12,
            route_offset,
            vector_input_copy);
}

constexpr int TPQ_P12_CODES = 4096;
constexpr int TPQ_P12_WARPS = 32;
constexpr int TPQ_P12_ROWS_PER_WARP = 4;
constexpr int TPQ_P12_ROWS_PER_BLOCK =
    TPQ_P12_WARPS * TPQ_P12_ROWS_PER_WARP;
// The registered p12 operator only accepts 4D and 8D codebooks.  Keeping a
// 10-element stride wasted 8 KiB of shared memory per CTA and reduced
// occupancy on decode-sized GEMVs.
constexpr int TPQ_P12_SHARED_STRIDE = 8;

// Kimi x/vv are packed 12-bit K4096 tiers.  Each CTA stages the input and
// one codebook in shared memory, then computes four rows per warp.  This
// replaces repeated input loads and random L2 codebook reads for roughly
// ninety percent of routed traffic while preserving the original p12 bytes.
__global__ void vq_gemv_routed_p12_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    __nv_bfloat16* __restrict__ out,
    const int top_k,
    const int expert_count,
    const int metadata_base,
    const int output_rows,
    const int input_cols,
    const long input_stride,
    const int active_count,
    const int route_offset)
{
    if (blockIdx.y >= active_count)
        return;
    const int position = blockIdx.y + route_offset;
    if (position >= top_k)
        return;
    const int expert = static_cast<int>(route_ids[position]);
    if (expert < 0 || expert >= expert_count)
        return;

    const int64_t index_address =
        metadata[(long)(metadata_base + 0) * expert_count + expert];
    if (index_address == 0)
        return;
    const int64_t codebook_address =
        metadata[(long)(metadata_base + 1) * expert_count + expert];
    const int blocks = static_cast<int>(
        metadata[(long)(metadata_base + 2) * expert_count + expert]);
    const int vector = static_cast<int>(
        metadata[(long)(metadata_base + 3) * expert_count + expert]);
    const int dtype_tag = static_cast<int>(
        metadata[(long)(metadata_base + 4) * expert_count + expert]);
    if (dtype_tag != 2 || (vector != 4 && vector != 8))
        return;

    extern __shared__ __nv_bfloat16 p12_shared[];
    auto* shared_input = p12_shared;
    auto* shared_codebook = p12_shared + input_cols;
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    constexpr int block_threads = 32 * TPQ_P12_WARPS;
    const __nv_bfloat16* input_row =
        x + (long)position * input_stride;
    for (
        int item = linear_thread;
        item < input_cols;
        item += block_threads
    )
        shared_input[item] = input_row[item];
    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(codebook_address));
    const int codebook_items = TPQ_P12_CODES * vector;
    for (
        int item = linear_thread;
        item < codebook_items;
        item += block_threads
    ) {
        const int code = item / vector;
        const int component = item - code * vector;
        shared_codebook[
            code * TPQ_P12_SHARED_STRIDE + component
        ] = codebook[item];
    }
    __syncthreads();

    float values[TPQ_P12_ROWS_PER_WARP] = {};
    for (int block = threadIdx.x; block < blocks; block += 32) {
        #pragma unroll
        for (
            int item = 0;
            item < TPQ_P12_ROWS_PER_WARP;
            ++item
        ) {
            const int row =
                blockIdx.x * TPQ_P12_ROWS_PER_BLOCK +
                threadIdx.y +
                item * TPQ_P12_WARPS;
            if (row < output_rows) {
                const int code = routed_index_value(
                    index_address,
                    dtype_tag,
                    (long)row * blocks + block);
                const __nv_bfloat16* code_row =
                    shared_codebook +
                    (long)code * TPQ_P12_SHARED_STRIDE;
                const __nv_bfloat16* input_block =
                    shared_input + block * vector;
                values[item] += (
                    vector == 4
                    ? vq_block_dot4_bf16(code_row, input_block)
                    : vq_block_dot(code_row, input_block, 8)
                );
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (
            int item = 0;
            item < TPQ_P12_ROWS_PER_WARP;
            ++item
        )
            values[item] += __shfl_down_sync(
                0xffffffffu,
                values[item],
                offset);
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (
            int item = 0;
            item < TPQ_P12_ROWS_PER_WARP;
            ++item
        ) {
            const int row =
                blockIdx.x * TPQ_P12_ROWS_PER_BLOCK +
                threadIdx.y +
                item * TPQ_P12_WARPS;
            if (row < output_rows)
                out[(long)position * output_rows + row] =
                    __float2bfloat16_rn(values[item]);
        }
    }
}

template <int WARPS>
__global__ void vq_gemv_routed_p12_l2_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    __nv_bfloat16* __restrict__ out,
    const int top_k,
    const int expert_count,
    const int metadata_base,
    const int output_rows,
    const int input_cols,
    const long input_stride,
    const int active_count,
    const int route_offset)
{
    if (blockIdx.y >= active_count)
        return;
    const int position = blockIdx.y + route_offset;
    if (position >= top_k)
        return;
    const int expert = static_cast<int>(route_ids[position]);
    if (expert < 0 || expert >= expert_count)
        return;

    const int64_t index_address =
        metadata[(long)(metadata_base + 0) * expert_count + expert];
    if (index_address == 0)
        return;
    const int64_t codebook_address =
        metadata[(long)(metadata_base + 1) * expert_count + expert];
    const int blocks = static_cast<int>(
        metadata[(long)(metadata_base + 2) * expert_count + expert]);
    const int vector = static_cast<int>(
        metadata[(long)(metadata_base + 3) * expert_count + expert]);
    const int dtype_tag = static_cast<int>(
        metadata[(long)(metadata_base + 4) * expert_count + expert]);
    if (dtype_tag != 2 || (vector != 4 && vector != 8))
        return;

    extern __shared__ __nv_bfloat16 p12_l2_input[];
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    const __nv_bfloat16* input_row =
        x + (long)position * input_stride;
    for (
        int item = linear_thread;
        item < input_cols;
        item += 32 * WARPS
    )
        p12_l2_input[item] = input_row[item];
    __syncthreads();

    const int row = blockIdx.x * WARPS + threadIdx.y;
    if (row >= output_rows)
        return;
    const auto* indices = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(index_address));
    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(codebook_address));
    const long row_offset = (long)row * blocks;
    float value = 0.f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        int code;
        if ((blocks & 1) == 0) {
            unsigned packed = 0;
            if ((threadIdx.x & 1) == 0) {
                const long base = ((row_offset + block) >> 1) * 3;
                packed =
                    static_cast<unsigned>(indices[base]) |
                    (static_cast<unsigned>(indices[base + 1]) << 8) |
                    (static_cast<unsigned>(indices[base + 2]) << 16);
            }
            packed = __shfl_sync(
                __activemask(),
                packed,
                threadIdx.x & ~1);
            code = static_cast<int>(
                (packed >> ((threadIdx.x & 1) * 12)) & 0xfffu);
        } else {
            code = routed_index_value(
                index_address,
                dtype_tag,
                row_offset + block);
        }
        const __nv_bfloat16* code_row =
            codebook + (long)code * vector;
        const __nv_bfloat16* input_block =
            p12_l2_input + block * vector;
        value += (
            vector == 4
            ? vq_block_dot4_bf16(code_row, input_block)
            : vq_block_dot(code_row, input_block, 8)
        );
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(
            0xffffffffu,
            value,
            offset);
    if (threadIdx.x == 0)
        out[(long)position * output_rows + row] =
            __float2bfloat16_rn(value);
}

__global__ void routed_swiglu_bf16_inplace_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    const int K,
    const int E,
    const int inter,
    const float limit)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = K * inter;
    if (i >= total) return;
    const int n = i / inter;
    const int expert_id = static_cast<int>(route_ids[n]);
    if (
        expert_id < 0 || expert_id >= E ||
        metadata[expert_id] == 0
    ) return;
    const int m = i - n * inter;
    const long row = (long)n * 2 * inter;
    float gate = __bfloat162float(hidden[row + m]);
    float up = __bfloat162float(hidden[row + inter + m]);
    if (limit > 0.f) {
        gate = fminf(gate, limit);
        up = fminf(fmaxf(up, -limit), limit);
    }
    const float silu = gate / (1.f + expf(-gate));
    hidden[row + m] = __float2bfloat16_rn(silu * up);
}

__global__ void routed_situ_bf16_inplace_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    const int top_k,
    const int expert_count,
    const int intermediate,
    const float beta,
    const float linear_beta)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = top_k * intermediate;
    if (i >= total) return;
    const int position = i / intermediate;
    const int expert = static_cast<int>(route_ids[position]);
    if (
        expert < 0 || expert >= expert_count ||
        metadata[expert] == 0
    ) return;
    const int column = i - position * intermediate;
    const long row = (long)position * 2 * intermediate;
    const float gate = __bfloat162float(hidden[row + column]);
    float up = __bfloat162float(
        hidden[row + intermediate + column]);
    const float activated =
        beta * tanhf(gate / beta) / (1.f + expf(-gate));
    if (linear_beta > 0.f)
        up = linear_beta * tanhf(up / linear_beta);
    hidden[row + column] =
        __float2bfloat16_rn(activated * up);
}

__global__ void routed_weighted_sum_f32_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const int64_t* __restrict__ route_ids,
    const float* __restrict__ weights,
    const int64_t* __restrict__ metadata,
    float* __restrict__ result,
    const int K,
    const int E,
    const int D,
    const bool accumulate);

constexpr int TPQ_DOWN_REDUCE_ROWS = 1;

// Decode only needs the route-weighted sum of the expert down projections.
// One output row per expert warp minimizes the random-codebook dependency
// chain on H20.  Reducing Top-K inside the same CTA avoids materialising
// [Top-K, hidden] and launching a second reduction kernel.  The BF16 round
// before route weighting intentionally matches the unfused reference path.
__global__ void vq_gemv_routed_down_reduce_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t* __restrict__ route_ids,
    const float* __restrict__ route_weights,
    const int64_t* __restrict__ metadata,
    float* __restrict__ result,
    const int top_k,
    const int expert_count,
    const int output_rows,
    const int input_cols,
    const long input_stride)
{
    const int position = threadIdx.y;
    const int row_base = blockIdx.x * TPQ_DOWN_REDUCE_ROWS;
    int expert = -1;
    int64_t index_address = 0;
    int64_t codebook_address = 0;
    int blocks = 0;
    int vector = 0;
    int dtype_tag = 0;
    if (position < top_k) {
        expert = static_cast<int>(route_ids[position]);
        if (expert >= 0 && expert < expert_count) {
            index_address =
                metadata[(long)5 * expert_count + expert];
            codebook_address =
                metadata[(long)6 * expert_count + expert];
            blocks = static_cast<int>(
                metadata[(long)7 * expert_count + expert]);
            vector = static_cast<int>(
                metadata[(long)8 * expert_count + expert]);
            dtype_tag = static_cast<int>(
                metadata[(long)9 * expert_count + expert]);
        }
    }
    const bool valid = (
        index_address != 0 &&
        codebook_address != 0 &&
        blocks > 0 &&
        (vector == 4 || vector == 8)
    );
    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(codebook_address));
    const __nv_bfloat16* input_row =
        input + (long)position * input_stride;
    float values[TPQ_DOWN_REDUCE_ROWS] = {};
    if (valid) {
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const __nv_bfloat16* input_block =
                input_row + block * vector;
            #pragma unroll
            for (int item = 0; item < TPQ_DOWN_REDUCE_ROWS; ++item) {
                const int row = row_base + item;
                if (row < output_rows) {
                    const int code = routed_index_value(
                        index_address,
                        dtype_tag,
                        (long)row * blocks + block);
                    const __nv_bfloat16* code_row =
                        codebook + (long)code * vector;
                    values[item] += (
                        vector == 4
                        ? vq_block_dot4_bf16(code_row, input_block)
                        : vq_block_dot(code_row, input_block, 8)
                    );
                }
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int item = 0; item < TPQ_DOWN_REDUCE_ROWS; ++item)
            values[item] += __shfl_down_sync(
                0xffffffffu,
                values[item],
                offset);
    }
    __shared__ float partial[TPQ_DOWN_REDUCE_ROWS][MAX_SLOT_EXPERTS];
    if (threadIdx.x == 0) {
        const float weight = (
            position < top_k ? route_weights[position] : 0.0f);
        #pragma unroll
        for (int item = 0; item < TPQ_DOWN_REDUCE_ROWS; ++item) {
            const float rounded = __bfloat162float(
                __float2bfloat16_rn(values[item]));
            partial[item][position] = rounded * weight;
        }
    }
    __syncthreads();
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    if (linear_thread < TPQ_DOWN_REDUCE_ROWS) {
        const int row = row_base + linear_thread;
        if (row < output_rows) {
            float value = 0.0f;
            #pragma unroll
            for (int item = 0; item < MAX_SLOT_EXPERTS; ++item) {
                if (item < top_k)
                    value += partial[linear_thread][item];
            }
            result[row] = value;
        }
    }
}

inline void launch_vq_gemv_routed_down_reduce(
    const __nv_bfloat16* input,
    const int64_t* route_ids,
    const float* route_weights,
    const int64_t* metadata,
    float* result,
    const int top_k,
    const int expert_count,
    const int output_rows,
    const int input_cols,
    const long input_stride,
    cudaStream_t stream)
{
    vq_gemv_routed_down_reduce_kernel<<<
        (output_rows + TPQ_DOWN_REDUCE_ROWS - 1) /
            TPQ_DOWN_REDUCE_ROWS,
        dim3(32, MAX_SLOT_EXPERTS),
        0,
        stream>>>(
            input,
            route_ids,
            route_weights,
            metadata,
            result,
            top_k,
            expert_count,
            output_rows,
            input_cols,
            input_stride);
}

torch::Tensor packed_moe_topk(
    torch::Tensor input,
    torch::Tensor route_ids,
    torch::Tensor weights,
    torch::Tensor metadata,
    double beta,
    double linear_beta,
      torch::Tensor hidden_workspace,
      torch::Tensor out_workspace,
      torch::Tensor result,
      int64_t p12_count_value)
{
    TORCH_CHECK(
        input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
        input.is_contiguous() && input.dim() == 2 && input.size(0) == 1,
        "packed MoE input must be contiguous CUDA BF16 [1,D]");
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == at::kLong &&
        route_ids.is_contiguous() && route_ids.dim() == 1,
        "packed MoE route IDs must be CUDA int64 [K]");
      const int top_k = static_cast<int>(route_ids.numel());
    TORCH_CHECK(
          top_k > 0 && top_k <= MAX_SLOT_EXPERTS,
          "packed MoE Top-K must be in [1,16]");
      TORCH_CHECK(
          p12_count_value >= -1 && p12_count_value <= top_k,
          "packed MoE p12 count must be -1 or in [0,Top-K]");
      const int p12_count = static_cast<int>(p12_count_value);
      const bool p12_grouped = p12_count >= 0;
      const char* p12_setting = std::getenv("TPQ_P12_SHARED");
      const std::string p12_mode = (
          p12_setting == nullptr ? "direct" : std::string(p12_setting)
      );
      const bool use_p12_shared = (
          p12_mode == "1" || p12_mode == "shared"
      );
      const bool use_p12_l2 = p12_mode == "l2";
      const bool use_p12_specialized =
          use_p12_shared || use_p12_l2;
      const char* p12_warps_setting =
          std::getenv("TPQ_P12_L2_WARPS");
      int p12_l2_warps = (
          p12_warps_setting == nullptr
          ? 16
          : std::atoi(p12_warps_setting)
      );
      if (
          p12_l2_warps != 8 &&
          p12_l2_warps != 16 &&
          p12_l2_warps != 32
      )
          p12_l2_warps = 16;
      const char* routed_warps_setting =
          std::getenv("TPQ_ROUTED_WARPS");
      int routed_warps = (
          routed_warps_setting == nullptr
          ? ROWS_PER_BLOCK
          : std::atoi(routed_warps_setting)
      );
      if (
          routed_warps != 8 &&
          routed_warps != 16 &&
          routed_warps != 32
      )
          routed_warps = ROWS_PER_BLOCK;
      const char* vector_copy_setting =
          std::getenv("TPQ_ROUTED_VECTOR_COPY");
      const bool vector_input_copy = (
          vector_copy_setting == nullptr ||
          std::string(vector_copy_setting) != "0"
      );
      const int generic_count = (
          use_p12_specialized
          ? (p12_grouped ? top_k - p12_count : top_k)
          : top_k
      );
      const int generic_offset = (
          use_p12_specialized && p12_grouped ? p12_count : 0
      );
      const int p12_active = (
          use_p12_specialized
          ? (p12_grouped ? p12_count : top_k)
          : 0
      );
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.sizes() == route_ids.sizes(),
        "packed MoE route weights must be CUDA float32 [K]");
    TORCH_CHECK(
        metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        metadata.size(0) == ROUTED_META_ROWS,
        "packed MoE metadata must be CUDA int64 [10,E]");
    TORCH_CHECK(
        input.get_device() == route_ids.get_device() &&
        input.get_device() == weights.get_device() &&
        input.get_device() == metadata.get_device() &&
        input.get_device() == hidden_workspace.get_device() &&
        input.get_device() == out_workspace.get_device(),
        "packed MoE inputs and workspaces must share one CUDA device");
    const int expert_count = static_cast<int>(metadata.size(1));
    const int hidden = static_cast<int>(input.size(1));
    TORCH_CHECK(
        hidden_workspace.scalar_type() == at::kBFloat16 &&
        hidden_workspace.is_contiguous() &&
        hidden_workspace.dim() == 2 &&
        hidden_workspace.size(0) == top_k &&
        hidden_workspace.size(1) % 2 == 0,
        "packed MoE hidden workspace must be BF16 [K,2I]");
    const int intermediate =
        static_cast<int>(hidden_workspace.size(1) / 2);
    TORCH_CHECK(
        out_workspace.scalar_type() == at::kBFloat16 &&
        out_workspace.is_contiguous() &&
        out_workspace.sizes() ==
            torch::IntArrayRef({top_k, hidden}),
        "packed MoE output workspace must be BF16 [K,D]");
    TORCH_CHECK(
        result.scalar_type() == at::kFloat &&
        result.is_contiguous() && result.dim() == 1 &&
        result.numel() == hidden,
        "packed MoE result must be float32 [D]");
    TORCH_CHECK(
        beta > 0.0,
        "packed MoE activation beta must be positive");

    int current = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    TORCH_CHECK(
        current == input.get_device(),
        "packed MoE kernel must run under the input device");
      if (result.get_device() != current)
          ensure_peer_access(
              current,
              result.get_device(),
              "packed MoE direct result");
      dim3 p12_block(32, TPQ_P12_WARPS);
      auto stream = at::cuda::getCurrentCUDAStream();
      const size_t gu_p12_shared = static_cast<size_t>(
          hidden + (
              use_p12_shared
              ? TPQ_P12_CODES * TPQ_P12_SHARED_STRIDE
              : 0
          )
      ) * sizeof(__nv_bfloat16);
      const size_t down_p12_shared = static_cast<size_t>(
          intermediate + (
              use_p12_shared
              ? TPQ_P12_CODES * TPQ_P12_SHARED_STRIDE
              : 0
          )
      ) * sizeof(__nv_bfloat16);
      const size_t max_p12_shared =
          gu_p12_shared > down_p12_shared
          ? gu_p12_shared
          : down_p12_shared;
      if (use_p12_shared && max_p12_shared > 48 * 1024) {
          static size_t configured_p12_shared[16] = {};
          if (configured_p12_shared[current] < max_p12_shared) {
              const auto attribute_status = cudaFuncSetAttribute(
                  vq_gemv_routed_p12_kernel,
                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                  static_cast<int>(max_p12_shared));
              TORCH_CHECK(
                  attribute_status == cudaSuccess,
                  "failed to configure Kimi p12 shared memory: ",
                  cudaGetErrorString(attribute_status));
              configured_p12_shared[current] = max_p12_shared;
          }
      }
      if (generic_count > 0) {
          const auto* input_pointer =
              reinterpret_cast<const __nv_bfloat16*>(
                  input.data_ptr<at::BFloat16>());
          auto* output_pointer =
              reinterpret_cast<__nv_bfloat16*>(
                  hidden_workspace.data_ptr<at::BFloat16>());
          if (routed_warps == 8)
              launch_vq_gemv_routed<8>(
                  input_pointer, route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(), output_pointer,
                  top_k, expert_count, 0, 2 * intermediate, hidden, 0,
                  use_p12_specialized, generic_offset, generic_count,
                  vector_input_copy, stream);
          else if (routed_warps == 16)
              launch_vq_gemv_routed<16>(
                  input_pointer, route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(), output_pointer,
                  top_k, expert_count, 0, 2 * intermediate, hidden, 0,
                  use_p12_specialized, generic_offset, generic_count,
                  vector_input_copy, stream);
          else
              launch_vq_gemv_routed<32>(
                  input_pointer, route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(), output_pointer,
                  top_k, expert_count, 0, 2 * intermediate, hidden, 0,
                  use_p12_specialized, generic_offset, generic_count,
                  vector_input_copy, stream);
      }
      if (p12_active > 0) {
          if (use_p12_shared) {
              vq_gemv_routed_p12_kernel<<<
                  dim3(
                      (unsigned)(
                          (
                              2 * intermediate +
                              TPQ_P12_ROWS_PER_BLOCK - 1
                          ) / TPQ_P12_ROWS_PER_BLOCK),
                      (unsigned)p12_active),
                  p12_block,
                  gu_p12_shared,
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          input.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 0, 2 * intermediate, hidden,
                      0, p12_active, 0);
          } else if (p12_l2_warps == 8) {
              vq_gemv_routed_p12_l2_kernel<8><<<
                  dim3(
                      (unsigned)((2 * intermediate + 7) / 8),
                      (unsigned)p12_active),
                  dim3(32, 8),
                  (size_t)hidden * sizeof(__nv_bfloat16),
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          input.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 0, 2 * intermediate, hidden,
                      0, p12_active, 0);
          } else if (p12_l2_warps == 16) {
              vq_gemv_routed_p12_l2_kernel<16><<<
                  dim3(
                      (unsigned)((2 * intermediate + 15) / 16),
                      (unsigned)p12_active),
                  dim3(32, 16),
                  (size_t)hidden * sizeof(__nv_bfloat16),
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          input.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 0, 2 * intermediate, hidden,
                      0, p12_active, 0);
          } else {
              vq_gemv_routed_p12_l2_kernel<32><<<
                  dim3(
                      (unsigned)((2 * intermediate + 31) / 32),
                      (unsigned)p12_active),
                  dim3(32, 32),
                  (size_t)hidden * sizeof(__nv_bfloat16),
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          input.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 0, 2 * intermediate, hidden,
                      0, p12_active, 0);
          }
      }
    routed_situ_bf16_inplace_kernel<<<
        (top_k * intermediate + 255) / 256,
        256,
        0,
        stream>>>(
            reinterpret_cast<__nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            top_k,
            expert_count,
            intermediate,
            static_cast<float>(beta),
            static_cast<float>(linear_beta));
      const char* fused_down_setting =
          std::getenv("TPQ_FUSED_DOWN_REDUCE");
      const bool fused_down_forced = (
          fused_down_setting != nullptr &&
          fused_down_setting[0] == '1' &&
          fused_down_setting[1] == '\0'
      );
      const bool fused_down_disabled = (
          fused_down_setting != nullptr &&
          fused_down_setting[0] == '0' &&
          fused_down_setting[1] == '\0'
      );
      const bool fused_down_reduce = (
          !fused_down_disabled &&
          (
              fused_down_forced ||
              (
                  // Kimi TP8 uses 448 routed-intermediate columns per
                  // rank.  The one-row fused kernel was faster there in
                  // the v63 H20 A/B, while TP4's 896 columns remain on the
                  // materialized path.
                  intermediate <= 448 &&
                  result.get_device() == current
              )
          )
      );
      if (fused_down_reduce) {
          launch_vq_gemv_routed_down_reduce(
              reinterpret_cast<const __nv_bfloat16*>(
                  hidden_workspace.data_ptr<at::BFloat16>()),
              route_ids.data_ptr<int64_t>(),
              weights.data_ptr<float>(),
              metadata.data_ptr<int64_t>(),
              result.data_ptr<float>(),
              top_k,
              expert_count,
              hidden,
              intermediate,
              2 * intermediate,
              stream);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          return result;
      }
      if (generic_count > 0) {
          const auto* input_pointer =
              reinterpret_cast<const __nv_bfloat16*>(
                  hidden_workspace.data_ptr<at::BFloat16>());
          auto* output_pointer =
              reinterpret_cast<__nv_bfloat16*>(
                  out_workspace.data_ptr<at::BFloat16>());
          if (routed_warps == 8)
              launch_vq_gemv_routed<8>(
                  input_pointer, route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(), output_pointer,
                  top_k, expert_count, 5, hidden, intermediate,
                  2 * intermediate, use_p12_specialized, generic_offset,
                  generic_count, vector_input_copy, stream);
          else if (routed_warps == 16)
              launch_vq_gemv_routed<16>(
                  input_pointer, route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(), output_pointer,
                  top_k, expert_count, 5, hidden, intermediate,
                  2 * intermediate, use_p12_specialized, generic_offset,
                  generic_count, vector_input_copy, stream);
          else
              launch_vq_gemv_routed<32>(
                  input_pointer, route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(), output_pointer,
                  top_k, expert_count, 5, hidden, intermediate,
                  2 * intermediate, use_p12_specialized, generic_offset,
                  generic_count, vector_input_copy, stream);
      }
      if (p12_active > 0) {
          if (use_p12_shared) {
              vq_gemv_routed_p12_kernel<<<
                  dim3(
                      (unsigned)(
                          (hidden + TPQ_P12_ROWS_PER_BLOCK - 1) /
                          TPQ_P12_ROWS_PER_BLOCK),
                      (unsigned)p12_active),
                  p12_block,
                  down_p12_shared,
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          out_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 5, hidden, intermediate,
                      2 * intermediate, p12_active, 0);
          } else if (p12_l2_warps == 8) {
              vq_gemv_routed_p12_l2_kernel<8><<<
                  dim3(
                      (unsigned)((hidden + 7) / 8),
                      (unsigned)p12_active),
                  dim3(32, 8),
                  (size_t)intermediate * sizeof(__nv_bfloat16),
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          out_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 5, hidden, intermediate,
                      2 * intermediate, p12_active, 0);
          } else if (p12_l2_warps == 16) {
              vq_gemv_routed_p12_l2_kernel<16><<<
                  dim3(
                      (unsigned)((hidden + 15) / 16),
                      (unsigned)p12_active),
                  dim3(32, 16),
                  (size_t)intermediate * sizeof(__nv_bfloat16),
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          out_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 5, hidden, intermediate,
                      2 * intermediate, p12_active, 0);
          } else {
              vq_gemv_routed_p12_l2_kernel<32><<<
                  dim3(
                      (unsigned)((hidden + 31) / 32),
                      (unsigned)p12_active),
                  dim3(32, 32),
                  (size_t)intermediate * sizeof(__nv_bfloat16),
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          out_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 5, hidden, intermediate,
                      2 * intermediate, p12_active, 0);
          }
      }
    routed_weighted_sum_f32_kernel<<<
        (hidden + 255) / 256,
        256,
        0,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                out_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            metadata.data_ptr<int64_t>(),
            result.data_ptr<float>(),
            top_k,
            expert_count,
            hidden,
            false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

__global__ void routed_weighted_sum_f32_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const int64_t* __restrict__ route_ids,
    const float* __restrict__ weights,
    const int64_t* __restrict__ metadata,
    float* __restrict__ result,
    const int K,
    const int E,
    const int D,
    const bool accumulate)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    #pragma unroll
    for (int n = 0; n < MAX_SLOT_EXPERTS; ++n) {
        if (n >= K) continue;
        const int expert_id = static_cast<int>(route_ids[n]);
        if (
            expert_id >= 0 && expert_id < E &&
            metadata[(long)5 * E + expert_id] != 0
        ) {
            acc = fmaf(
                __bfloat162float(rows[(long)n * D + d]),
                weights[n],
                acc);
        }
    }
    result[d] = accumulate ? result[d] + acc : acc;
}

torch::Tensor moe_mlp_routed_slots(
    torch::Tensor x,
    torch::Tensor route_ids,
    torch::Tensor weights,
    torch::Tensor metadata,
    double limit,
    torch::Tensor hidden_workspace,
    torch::Tensor out_workspace,
    torch::Tensor result,
    bool d4_specialized,
    bool accumulate)
{
    TORCH_CHECK(
        x.is_cuda() && x.scalar_type() == at::kBFloat16 &&
        x.is_contiguous() && x.dim() == 2 && x.size(0) == 1,
        "routed slot input must be contiguous CUDA BF16 [1,D]");
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == at::kLong &&
        route_ids.is_contiguous() && route_ids.dim() == 1,
        "route IDs must be contiguous CUDA int64 [K]");
    const int K = static_cast<int>(route_ids.numel());
    TORCH_CHECK(
        K > 0 && K <= MAX_SLOT_EXPERTS,
        "routed slot Top-K must be in [1,8]");
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.dim() == 1 &&
        weights.numel() == K,
        "route weights must be contiguous CUDA float32 [K]");
    TORCH_CHECK(
        metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        metadata.size(0) == ROUTED_META_ROWS,
        "routed metadata must be contiguous CUDA int64 [10,E]");
    TORCH_CHECK(
        x.get_device() == route_ids.get_device() &&
        x.get_device() == weights.get_device() &&
        x.get_device() == metadata.get_device() &&
        x.get_device() == hidden_workspace.get_device() &&
        x.get_device() == out_workspace.get_device(),
        "routed slot compute tensors must be on one CUDA device");
    TORCH_CHECK(
        result.is_cuda(),
        "routed partial result must be a CUDA tensor");
    int current = -1;
    const auto current_status = cudaGetDevice(&current);
    TORCH_CHECK(
        current_status == cudaSuccess &&
        current == x.get_device(),
        "routed slot kernel must run on its input CUDA device");
    ensure_peer_access(
        current,
        result.get_device(),
        "expert direct return");
    const int E = static_cast<int>(metadata.size(1));
    const int hidden = static_cast<int>(x.size(1));
    TORCH_CHECK(
        hidden_workspace.scalar_type() == at::kBFloat16 &&
        hidden_workspace.is_contiguous() &&
        hidden_workspace.dim() == 2 &&
        hidden_workspace.size(0) == K &&
        hidden_workspace.size(1) % 2 == 0,
        "hidden workspace must be contiguous BF16 [K,2I]");
    const int inter = static_cast<int>(hidden_workspace.size(1) / 2);
    TORCH_CHECK(
        out_workspace.scalar_type() == at::kBFloat16 &&
        out_workspace.is_contiguous() &&
        out_workspace.dim() == 2 &&
        out_workspace.size(0) == K &&
        out_workspace.size(1) == hidden,
        "output workspace must be contiguous BF16 [K,D]");
    TORCH_CHECK(
        result.scalar_type() == at::kFloat &&
        result.is_contiguous() && result.dim() == 1 &&
        result.numel() == hidden,
        "routed partial result must be contiguous float32 [D]");

    dim3 block(32, ROWS_PER_BLOCK);
    auto stream = at::cuda::getCurrentCUDAStream();
    vq_gemv_routed_kernel<ROWS_PER_BLOCK><<<
        dim3(
            (unsigned)((2 * inter + ROWS_PER_BLOCK - 1) /
                       ROWS_PER_BLOCK),
            (unsigned)K),
        block,
        (size_t)hidden * sizeof(__nv_bfloat16),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            K, E, 0, 2 * inter, hidden, 0,
            d4_specialized, 0, true);
    routed_swiglu_bf16_inplace_kernel<<<
        (K * inter + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            K, E, inter, static_cast<float>(limit));
    vq_gemv_routed_kernel<ROWS_PER_BLOCK><<<
        dim3(
            (unsigned)((hidden + ROWS_PER_BLOCK - 1) /
                       ROWS_PER_BLOCK),
            (unsigned)K),
        block,
        (size_t)inter * sizeof(__nv_bfloat16),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                out_workspace.data_ptr<at::BFloat16>()),
            K, E, 5, hidden, inter, 2 * inter,
            d4_specialized, 0, true);
    routed_weighted_sum_f32_kernel<<<
        (hidden + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                out_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            metadata.data_ptr<int64_t>(),
            result.data_ptr<float>(),
            K, E, hidden, accumulate);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

// ---- Dedicated D4/K4096 routed VQ path ---------------------------------
//
// An independent ``vv`` expert has a 4096x4 BF16 codebook and uint16
// indices.  The generic kernel re-reads four random BF16 values for every
// matrix index.  The complete codebook is only 32 KiB, so one CTA can stage
// it once in shared memory and reuse it across 32 output rows.  v256 experts
// remain on the CodeGEMM Psumbook path; metadata for those experts is zero in
// this kernel and returns before the first barrier.

constexpr int TPQ_VV_CODES = 4096;
constexpr int TPQ_VV_VECTOR = 4;
// Six BF16 slots per code make consecutive code starts advance by three
// shared-memory banks instead of two. Random K4096 lookups can then use all
// 32 banks rather than only the 16 even/odd start banks.
constexpr int TPQ_VV_SHARED_STRIDE = 6;
constexpr int TPQ_VV_WARPS_PER_BLOCK = 32;
constexpr int TPQ_VV_ROWS_PER_WARP = 2;
constexpr int TPQ_VV_ROWS_PER_BLOCK =
    TPQ_VV_WARPS_PER_BLOCK * TPQ_VV_ROWS_PER_WARP;

__global__ void vq_gemv_routed_vv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    __nv_bfloat16* __restrict__ out,
    const int top_k,
    const int expert_count,
    const int metadata_base,
    const int output_rows,
    const int input_cols,
    const long input_stride)
{
    const int position = blockIdx.y;
    if (position >= top_k)
        return;
    const int expert = static_cast<int>(route_ids[position]);
    if (expert < 0 || expert >= expert_count)
        return;

    const int64_t index_address =
        metadata[(long)(metadata_base + 0) * expert_count + expert];
    if (index_address == 0)
        return;
    const int64_t codebook_address =
        metadata[(long)(metadata_base + 1) * expert_count + expert];
    const int blocks = static_cast<int>(
        metadata[(long)(metadata_base + 2) * expert_count + expert]);
    const int vector = static_cast<int>(
        metadata[(long)(metadata_base + 3) * expert_count + expert]);
    const int dtype_tag = static_cast<int>(
        metadata[(long)(metadata_base + 4) * expert_count + expert]);
    if (vector != TPQ_VV_VECTOR || dtype_tag != 1)
        return;

    extern __shared__ __nv_bfloat16 vv_shared[];
    auto* shared_input = vv_shared;
    auto* shared_codebook = vv_shared + input_cols;
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    const int block_threads = 32 * TPQ_VV_WARPS_PER_BLOCK;
    const __nv_bfloat16* input_row =
        x + (long)position * input_stride;
    for (
        int item = linear_thread;
        item < input_cols;
        item += block_threads
    )
        shared_input[item] = input_row[item];
    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(codebook_address));
    constexpr int codebook_items =
        TPQ_VV_CODES * TPQ_VV_VECTOR;
    for (
        int item = linear_thread;
        item < codebook_items;
        item += block_threads
    )
    {
        const int code = item / TPQ_VV_VECTOR;
        const int component = item - code * TPQ_VV_VECTOR;
        shared_codebook[
            code * TPQ_VV_SHARED_STRIDE + component
        ] = codebook[item];
    }
    __syncthreads();

    const auto* indices = reinterpret_cast<const uint16_t*>(
        static_cast<uintptr_t>(index_address));
    float values[TPQ_VV_ROWS_PER_WARP] = {};
    for (int block = threadIdx.x; block < blocks; block += 32) {
        #pragma unroll
        for (int item = 0; item < TPQ_VV_ROWS_PER_WARP; ++item) {
            const int row =
                blockIdx.x * TPQ_VV_ROWS_PER_BLOCK +
                threadIdx.y +
                item * TPQ_VV_WARPS_PER_BLOCK;
            if (row < output_rows) {
                const int code = static_cast<int>(
                    indices[(long)row * blocks + block]);
                values[item] += vq_block_dot4_bf16(
                    shared_codebook +
                        (long)code * TPQ_VV_SHARED_STRIDE,
                    shared_input + block * TPQ_VV_VECTOR);
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int item = 0; item < TPQ_VV_ROWS_PER_WARP; ++item)
            values[item] += __shfl_down_sync(
                0xffffffffu,
                values[item],
                offset);
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int item = 0; item < TPQ_VV_ROWS_PER_WARP; ++item) {
            const int row =
                blockIdx.x * TPQ_VV_ROWS_PER_BLOCK +
                threadIdx.y +
                item * TPQ_VV_WARPS_PER_BLOCK;
            if (row < output_rows)
                out[(long)position * output_rows + row] =
                    __float2bfloat16_rn(values[item]);
        }
    }
}

torch::Tensor moe_mlp_routed_vv(
    torch::Tensor input,
    torch::Tensor route_ids,
    torch::Tensor weights,
    torch::Tensor metadata,
    double limit,
    torch::Tensor hidden_workspace,
    torch::Tensor out_workspace,
    torch::Tensor result,
    bool accumulate)
{
    TORCH_CHECK(
        input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
        input.is_contiguous() && input.dim() == 2 && input.size(0) == 1,
        "vv input must be contiguous CUDA BF16 [1,D]");
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == at::kLong &&
        route_ids.is_contiguous() && route_ids.dim() == 1,
        "vv route IDs must be contiguous CUDA int64 [K]");
    const int top_k = static_cast<int>(route_ids.numel());
    TORCH_CHECK(
        top_k > 0 && top_k <= MAX_SLOT_EXPERTS,
        "vv Top-K must be in [1,8]");
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.sizes() == route_ids.sizes(),
        "vv weights must be contiguous CUDA float32 [K]");
    TORCH_CHECK(
        metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        metadata.size(0) == ROUTED_META_ROWS,
        "vv metadata must be contiguous CUDA int64 [10,E]");
    TORCH_CHECK(
        input.get_device() == route_ids.get_device() &&
        input.get_device() == weights.get_device() &&
        input.get_device() == metadata.get_device() &&
        input.get_device() == hidden_workspace.get_device() &&
        input.get_device() == out_workspace.get_device(),
        "vv compute tensors must share one device");
    TORCH_CHECK(
        result.is_cuda() && result.scalar_type() == at::kFloat &&
        result.is_contiguous() && result.dim() == 1,
        "vv result must be contiguous CUDA float32 [D]");

    const int hidden = static_cast<int>(input.size(1));
    TORCH_CHECK(
        hidden_workspace.scalar_type() == at::kBFloat16 &&
        hidden_workspace.is_contiguous() &&
        hidden_workspace.dim() == 2 &&
        hidden_workspace.size(0) == top_k &&
        hidden_workspace.size(1) % 2 == 0,
        "vv hidden workspace must be BF16 [K,2I]");
    const int intermediate =
        static_cast<int>(hidden_workspace.size(1) / 2);
    TORCH_CHECK(
        out_workspace.scalar_type() == at::kBFloat16 &&
        out_workspace.is_contiguous() &&
        out_workspace.sizes() ==
            torch::IntArrayRef({top_k, hidden}),
        "vv output workspace must be BF16 [K,D]");
    TORCH_CHECK(
        result.numel() == hidden,
        "vv result width mismatch");

    int current = -1;
    const auto status = cudaGetDevice(&current);
    TORCH_CHECK(
        status == cudaSuccess && current == input.get_device(),
        "vv kernel must run on its input CUDA device");
    ensure_peer_access(current, result.get_device(), "vv direct return");

    const int expert_count = static_cast<int>(metadata.size(1));
    dim3 block(32, TPQ_VV_WARPS_PER_BLOCK);
    auto stream = at::cuda::getCurrentCUDAStream();
    const size_t gu_shared = static_cast<size_t>(
        hidden + TPQ_VV_CODES * TPQ_VV_SHARED_STRIDE
    ) * sizeof(__nv_bfloat16);
    const size_t dn_shared = static_cast<size_t>(
        intermediate + TPQ_VV_CODES * TPQ_VV_SHARED_STRIDE
    ) * sizeof(__nv_bfloat16);
    const size_t max_shared =
        gu_shared > dn_shared ? gu_shared : dn_shared;
    if (max_shared > 48 * 1024) {
        static size_t configured_shared[16] = {};
        if (configured_shared[current] < max_shared) {
            const auto attribute_status = cudaFuncSetAttribute(
                vq_gemv_routed_vv_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(max_shared));
            TORCH_CHECK(
                attribute_status == cudaSuccess,
                "failed to configure vv shared memory: ",
                cudaGetErrorString(attribute_status));
            configured_shared[current] = max_shared;
        }
    }
    vq_gemv_routed_vv_kernel<<<
        dim3(
            (2 * intermediate + TPQ_VV_ROWS_PER_BLOCK - 1) /
                TPQ_VV_ROWS_PER_BLOCK,
            top_k),
        block,
        gu_shared,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            top_k,
            expert_count,
            0,
            2 * intermediate,
            hidden,
            0);
    routed_swiglu_bf16_inplace_kernel<<<
        (top_k * intermediate + 255) / 256,
        256,
        0,
        stream>>>(
            reinterpret_cast<__nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            top_k,
            expert_count,
            intermediate,
            static_cast<float>(limit));
    vq_gemv_routed_vv_kernel<<<
        dim3(
            (hidden + TPQ_VV_ROWS_PER_BLOCK - 1) /
                TPQ_VV_ROWS_PER_BLOCK,
            top_k),
        block,
        dn_shared,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                out_workspace.data_ptr<at::BFloat16>()),
            top_k,
            expert_count,
            5,
            hidden,
            intermediate,
            2 * intermediate);
    routed_weighted_sum_f32_kernel<<<
        (hidden + 255) / 256,
        256,
        0,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                out_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            metadata.data_ptr<int64_t>(),
            result.data_ptr<float>(),
            top_k,
            expert_count,
            hidden,
            accumulate);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

// ---- HC sinkhorn (hc = 4 fixed: DSV4 Hyper-Connections) ----

__global__ void hc_sinkhorn_kernel(
    const float* __restrict__ mixes,  // [N, 24]
    const float* __restrict__ scale,  // [3]
    const float* __restrict__ base,   // [24]
    float* __restrict__ out,          // [N, 24]: pre[4] | post[4] | comb[16]
    const int N, const int iters, const float eps)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const float* m = mixes + (long)row * 24;
    float* o = out + (long)row * 24;
    const float s0 = scale[0], s1 = scale[1], s2 = scale[2];

    #pragma unroll
    for (int j = 0; j < 4; ++j)
        o[j] = 1.f / (1.f + expf(-(m[j] * s0 + base[j]))) + eps;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        o[4 + j] = 2.f / (1.f + expf(-(m[4 + j] * s1 + base[4 + j])));

    float c[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float mx = -INFINITY;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            c[j][k] = m[8 + 4 * j + k] * s2 + base[8 + 4 * j + k];
            mx = fmaxf(mx, c[j][k]);
        }
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) { c[j][k] = expf(c[j][k] - mx); sum += c[j][k]; }
        #pragma unroll
        for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / sum + eps;
    }
    // first column normalize (after softmax), then iters-1 rounds of row+col
    for (int it = 0; it < iters; ++it) {
        if (it > 0) {  // row normalize (skip on round 0: softmax already row-stochastic)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float rs = c[j][0] + c[j][1] + c[j][2] + c[j][3];
                const float inv = 1.f / (rs + eps);
                #pragma unroll
                for (int k = 0; k < 4; ++k) c[j][k] *= inv;
            }
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {  // column normalize
            float cs = c[0][k] + c[1][k] + c[2][k] + c[3][k];
            const float inv = 1.f / (cs + eps);
            #pragma unroll
            for (int j = 0; j < 4; ++j) c[j][k] *= inv;
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            o[8 + 4 * j + k] = c[j][k];
}

torch::Tensor hc_sinkhorn(torch::Tensor mixes, torch::Tensor scale,
                          torch::Tensor base, long iters, double eps) {
    TORCH_CHECK(mixes.is_cuda() && scale.is_cuda() && base.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(mixes.scalar_type() == at::kFloat, "mixes must be float32");
    TORCH_CHECK(mixes.size(-1) == 24, "mixes last dim must be 24 (hc=4)");
    auto m2 = mixes.contiguous().view({-1, 24});
    const int N = (int)m2.size(0);
    auto out = torch::empty_like(m2);
    const int threads = 128;
    const int blocks = (N + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    hc_sinkhorn_kernel<<<blocks, threads, 0, stream>>>(
        m2.data_ptr<float>(), scale.contiguous().data_ptr<float>(),
        base.contiguous().data_ptr<float>(), out.data_ptr<float>(),
        N, (int)iters, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view(mixes.sizes());
}

// ---- RMSNorm (fused: one block per row, f32) ----
// out[r, i] = w[i] * x[r, i] * rsqrt(mean_i(x[r]^2) + eps)

__global__ void rmsnorm_kernel(
    const float* __restrict__ x,   // [N, D]
    const float* __restrict__ w,   // [D]
    float* __restrict__ out,       // [N, D]
    const int D, const float eps)
{
    const int r = blockIdx.x;
    const float* xr = x + (long)r * D;
    float* orow = out + (long)r * D;
    float acc = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = xr[i];
        acc += v * v;
    }
    __shared__ float red[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? red[threadIdx.x] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float scale = rsqrtf(red[0] / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        orow[i] = w[i] * (xr[i] * scale);
}

torch::Tensor rmsnorm(
    torch::Tensor x,
    torch::Tensor w,
    double eps,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat,
                "x/w must be float32");
    auto xc = x.contiguous();
    const int D = (int)xc.size(-1);
    auto x2 = xc.view({-1, D});
    const int N = (int)x2.size(0);
    auto out = output_buffer.has_value()
        ? output_buffer.value().view({-1, D})
        : torch::empty_like(x2);
    TORCH_CHECK(
        out.is_cuda() &&
        out.scalar_type() == at::kFloat &&
        out.is_contiguous() &&
        out.sizes() == x2.sizes() &&
        out.get_device() == x.get_device(),
        "RMSNorm output buffer must be contiguous float32 and match input");
    auto stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_kernel<<<N, 256, 0, stream>>>(
        x2.data_ptr<float>(), w.contiguous().data_ptr<float>(),
        out.data_ptr<float>(), D, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view(xc.sizes());
}

template <typename weight_t>
__device__ __forceinline__ float rmsnorm_weight_float(weight_t value)
{
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float rmsnorm_weight_float(
    __nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename weight_t>
__global__ void rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const weight_t* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    const int D,
    const float eps)
{
    const int row = blockIdx.x;
    const auto* input = x + static_cast<long>(row) * D;
    auto* output = out + static_cast<long>(row) * D;
    float sum = 0.0f;
    for (int item = threadIdx.x; item < D; item += blockDim.x) {
        const float value = __bfloat162float(input[item]);
        sum += value * value;
    }
    __shared__ float reduction[32];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if ((threadIdx.x & 31) == 0)
        reduction[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = (
            threadIdx.x < (blockDim.x + 31) / 32
            ? reduction[threadIdx.x]
            : 0.0f);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(
                0xffffffffu,
                value,
                offset);
        if (threadIdx.x == 0)
            reduction[0] = value;
    }
    __syncthreads();
    const float scale = rsqrtf(reduction[0] / static_cast<float>(D) + eps);
    for (int item = threadIdx.x; item < D; item += blockDim.x) {
        // Match the reference's two BF16 boundaries:
        // weight.to(BF16) * normalized.to(BF16).
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(input[item]) * scale);
        const __nv_bfloat16 weight = __float2bfloat16_rn(
            rmsnorm_weight_float(w[item]));
        output[item] = __float2bfloat16_rn(
            __bfloat162float(normalized)
            * __bfloat162float(weight));
    }
}

torch::Tensor rmsnorm_bf16(
    torch::Tensor x,
    torch::Tensor w,
    double eps,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        x.is_cuda() && w.is_cuda(),
        "BF16 RMSNorm tensors must be CUDA");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16 &&
        (
            w.scalar_type() == at::kBFloat16 ||
            w.scalar_type() == at::kFloat
        ),
        "BF16 RMSNorm requires BF16 input and BF16/FP32 weight");
    auto input = x.contiguous();
    auto weight = w.contiguous();
    const int width = static_cast<int>(input.size(-1));
    auto rows = input.view({-1, width});
    auto output = output_buffer.has_value()
        ? output_buffer.value().view({-1, width})
        : torch::empty_like(rows);
    TORCH_CHECK(
        weight.dim() == 1 &&
        weight.numel() == width &&
        output.is_cuda() &&
        output.scalar_type() == at::kBFloat16 &&
        output.is_contiguous() &&
        output.sizes() == rows.sizes() &&
        output.get_device() == input.get_device() &&
        weight.get_device() == input.get_device(),
        "BF16 RMSNorm shapes/devices do not match");
    auto stream = at::cuda::getCurrentCUDAStream();
    if (weight.scalar_type() == at::kBFloat16) {
        rmsnorm_bf16_kernel<<<rows.size(0), 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                rows.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            width,
            static_cast<float>(eps));
    } else {
        rmsnorm_bf16_kernel<<<rows.size(0), 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                rows.data_ptr<at::BFloat16>()),
            weight.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            width,
            static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output.view(input.sizes());
}

constexpr int TPQ_RESIDUAL_MAX_ROWS = 16;
constexpr int TPQ_RESIDUAL_STAGED_MAX_ROWS = 32;
constexpr int TPQ_RESIDUAL_THREADS = 256;
constexpr int TPQ_RESIDUAL_WARPS = TPQ_RESIDUAL_THREADS / 32;

__global__ void attention_residual_bf16_kernel(
    const __nv_bfloat16* __restrict__ prefix,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ projection,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ post_norm_weight,
    float* __restrict__ residual_inverse,
    __nv_bfloat16* __restrict__ output,
    const int residual_rows,
    const int width,
    const float eps)
{
    const int rows = residual_rows + 1;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float partial[
        TPQ_RESIDUAL_MAX_ROWS * TPQ_RESIDUAL_WARPS];
    __shared__ float row_values[TPQ_RESIDUAL_MAX_ROWS];

    if (threadIdx.x < rows) {
        const int row = threadIdx.x;
        row_values[row] = (
            row < residual_rows && residual_inverse != nullptr
            ? residual_inverse[row]
            : 0.0f);
    }
    __syncthreads();
    for (int row = 0; row < rows; ++row) {
        if (row_values[row] > 0.0f)
            continue;
        const auto* source = (
            row < residual_rows
            ? residual + static_cast<long>(row) * width
            : prefix);
        float sum = 0.0f;
        for (
            int item = threadIdx.x;
            item < width;
            item += blockDim.x
        ) {
            const float value = __bfloat162float(source[item]);
            sum += value * value;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0)
            partial[row * TPQ_RESIDUAL_WARPS + warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        for (int row = lane; row < rows; row += 32) {
            if (row_values[row] > 0.0f)
                continue;
            float sum = 0.0f;
            #pragma unroll
            for (int item = 0; item < TPQ_RESIDUAL_WARPS; ++item)
                sum += partial[row * TPQ_RESIDUAL_WARPS + item];
            row_values[row] = rsqrtf(
                sum / static_cast<float>(width) + eps);
            if (row < residual_rows && residual_inverse != nullptr)
                residual_inverse[row] = row_values[row];
        }
    }
    __syncthreads();

    for (int row = 0; row < rows; ++row) {
        const auto* source = (
            row < residual_rows
            ? residual + static_cast<long>(row) * width
            : prefix);
        float score = 0.0f;
        const float inverse = row_values[row];
        for (
            int item = threadIdx.x;
            item < width;
            item += blockDim.x
        ) {
            score += (
                __bfloat162float(source[item])
                * inverse
                * __bfloat162float(norm_weight[item])
                * __bfloat162float(projection[item]));
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            score += __shfl_down_sync(
                0xffffffffu,
                score,
                offset);
        if (lane == 0)
            partial[row * TPQ_RESIDUAL_WARPS + warp] = score;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = -1.0e30f;
        for (int row = 0; row < rows; ++row) {
            float score = 0.0f;
            #pragma unroll
            for (int item = 0; item < TPQ_RESIDUAL_WARPS; ++item)
                score += partial[row * TPQ_RESIDUAL_WARPS + item];
            row_values[row] = score;
            maximum = fmaxf(maximum, score);
        }
        float denominator = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const float value = expf(row_values[row] - maximum);
            row_values[row] = value;
            denominator += value;
        }
        for (int row = 0; row < rows; ++row)
            row_values[row] /= denominator;
    }
    __syncthreads();
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        float mixed = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const auto* source = (
                row < residual_rows
                ? residual + static_cast<long>(row) * width
                : prefix);
            mixed += row_values[row] * __bfloat162float(source[item]);
        }
        output[item] = __float2bfloat16_rn(mixed);
    }
    if (post_norm_weight == nullptr)
        return;
    __syncthreads();
    float sum = 0.0f;
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        const float value = __bfloat162float(output[item]);
        sum += value * value;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float value = (
            lane < TPQ_RESIDUAL_WARPS ? partial[lane] : 0.0f);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(
                0xffffffffu,
                value,
                offset);
        if (lane == 0)
            row_values[0] = value;
    }
    __syncthreads();
    const float scale = rsqrtf(
        row_values[0] / static_cast<float>(width) + eps);
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(output[item]) * scale);
        output[item] = __float2bfloat16_rn(
            __bfloat162float(normalized)
            * __bfloat162float(post_norm_weight[item]));
    }
}

// The one-CTA kernel above is fastest for short residual lists, but it
// serializes every row.  Deep residual blocks first calculate their row
// scores with one CTA per row, then use one deterministic CTA for softmax,
// weighted mixing and the optional following RMSNorm.  The reduction and row
// accumulation orders match the short kernel.
__global__ void attention_residual_scores_bf16_kernel(
    const __nv_bfloat16* __restrict__ prefix,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ projection,
    const __nv_bfloat16* __restrict__ norm_weight,
    float* __restrict__ residual_inverse,
    float* __restrict__ scores,
    const int residual_rows,
    const int width,
    const float eps)
{
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const auto* source = (
        row < residual_rows
        ? residual + static_cast<long>(row) * width
        : prefix);
    __shared__ float partial[TPQ_RESIDUAL_WARPS];
    __shared__ float inverse;

    if (threadIdx.x == 0) {
        inverse = (
            row < residual_rows && residual_inverse != nullptr
            ? residual_inverse[row]
            : 0.0f);
    }
    __syncthreads();
    if (inverse <= 0.0f) {
        float sum = 0.0f;
        for (
            int item = threadIdx.x;
            item < width;
            item += blockDim.x
        ) {
            const float value = __bfloat162float(source[item]);
            sum += value * value;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0)
            partial[warp] = sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.0f;
            #pragma unroll
            for (int item = 0; item < TPQ_RESIDUAL_WARPS; ++item)
                total += partial[item];
            inverse = rsqrtf(
                total / static_cast<float>(width) + eps);
            if (row < residual_rows && residual_inverse != nullptr)
                residual_inverse[row] = inverse;
        }
        __syncthreads();
    }

    float score = 0.0f;
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        score += (
            __bfloat162float(source[item])
            * inverse
            * __bfloat162float(norm_weight[item])
            * __bfloat162float(projection[item]));
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        score += __shfl_down_sync(
            0xffffffffu,
            score,
            offset);
    if (lane == 0)
        partial[warp] = score;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int item = 0; item < TPQ_RESIDUAL_WARPS; ++item)
            total += partial[item];
        scores[row] = total;
    }
}

__global__ void attention_residual_mix_bf16_kernel(
    const __nv_bfloat16* __restrict__ prefix,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ post_norm_weight,
    float* __restrict__ scores,
    __nv_bfloat16* __restrict__ output,
    const int residual_rows,
    const int width,
    const float eps)
{
    const int rows = residual_rows + 1;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float probabilities[TPQ_RESIDUAL_STAGED_MAX_ROWS];
    __shared__ float partial[TPQ_RESIDUAL_WARPS];

    if (threadIdx.x == 0) {
        float maximum = -1.0e30f;
        for (int row = 0; row < rows; ++row)
            maximum = fmaxf(maximum, scores[row]);
        float denominator = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const float value = expf(scores[row] - maximum);
            probabilities[row] = value;
            denominator += value;
        }
        for (int row = 0; row < rows; ++row)
            probabilities[row] /= denominator;
    }
    __syncthreads();
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        float mixed = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const auto* source = (
                row < residual_rows
                ? residual + static_cast<long>(row) * width
                : prefix);
            mixed += (
                probabilities[row]
                * __bfloat162float(source[item]));
        }
        output[item] = __float2bfloat16_rn(mixed);
    }
    if (post_norm_weight == nullptr)
        return;
    __syncthreads();
    float sum = 0.0f;
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        const float value = __bfloat162float(output[item]);
        sum += value * value;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float value = (
            lane < TPQ_RESIDUAL_WARPS ? partial[lane] : 0.0f);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(
                0xffffffffu,
                value,
                offset);
        if (lane == 0)
            scores[0] = value;
    }
    __syncthreads();
    const float scale = rsqrtf(
        scores[0] / static_cast<float>(width) + eps);
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(output[item]) * scale);
        output[item] = __float2bfloat16_rn(
            __bfloat162float(normalized)
            * __bfloat162float(post_norm_weight[item]));
    }
}

torch::Tensor attention_residual_bf16(
    torch::Tensor prefix,
    torch::Tensor residual,
    torch::Tensor projection,
    torch::Tensor norm_weight,
    c10::optional<torch::Tensor> post_norm_weight,
    double eps,
    c10::optional<torch::Tensor> output_buffer,
    c10::optional<torch::Tensor> score_workspace,
    long single_cta_max_rows,
    c10::optional<torch::Tensor> residual_inverse)
{
    TORCH_CHECK(
        prefix.is_cuda() && residual.is_cuda() &&
        projection.is_cuda() && norm_weight.is_cuda(),
        "Attention residual tensors must be CUDA");
    TORCH_CHECK(
        prefix.scalar_type() == at::kBFloat16 &&
        residual.scalar_type() == at::kBFloat16 &&
        projection.scalar_type() == at::kBFloat16 &&
        norm_weight.scalar_type() == at::kBFloat16,
        "Attention residual currently requires BF16");
    TORCH_CHECK(
        prefix.dim() == 2 && prefix.size(0) == 1 &&
        residual.dim() == 3 && residual.size(0) == 1 &&
        residual.size(2) == prefix.size(1) &&
        projection.numel() == prefix.size(1) &&
        norm_weight.numel() == prefix.size(1) &&
        residual.size(1) > 0 &&
        residual.size(1) + 1 <= TPQ_RESIDUAL_STAGED_MAX_ROWS,
        "Attention residual shapes do not match");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty_like(prefix);
    TORCH_CHECK(
        prefix.is_contiguous() && residual.is_contiguous() &&
        projection.is_contiguous() && norm_weight.is_contiguous() &&
        output.is_contiguous() &&
        output.scalar_type() == at::kBFloat16 &&
        output.sizes() == prefix.sizes() &&
        output.get_device() == prefix.get_device() &&
        residual.get_device() == prefix.get_device() &&
        projection.get_device() == prefix.get_device() &&
        norm_weight.get_device() == prefix.get_device(),
        "Attention residual buffers must be contiguous and colocated");
    const __nv_bfloat16* post_norm_ptr = nullptr;
    if (post_norm_weight.has_value()) {
        const auto post = post_norm_weight.value();
        TORCH_CHECK(
            post.is_cuda() &&
            post.scalar_type() == at::kBFloat16 &&
            post.is_contiguous() &&
            post.numel() == prefix.size(1) &&
            post.get_device() == prefix.get_device(),
            "Attention residual post-norm weight must be colocated BF16");
        post_norm_ptr = reinterpret_cast<const __nv_bfloat16*>(
            post.data_ptr<at::BFloat16>());
    }
    float* residual_inverse_ptr = nullptr;
    if (residual_inverse.has_value()) {
        const auto inverse = residual_inverse.value();
        TORCH_CHECK(
            inverse.is_cuda() &&
            inverse.scalar_type() == at::kFloat &&
            inverse.is_contiguous() &&
            inverse.numel() >= residual.size(1) &&
            inverse.get_device() == prefix.get_device(),
            "Attention residual inverse cache must be colocated "
            "contiguous float32[>=residual_rows]");
        residual_inverse_ptr = inverse.data_ptr<float>();
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = static_cast<int>(residual.size(1)) + 1;
    TORCH_CHECK(
        single_cta_max_rows >= 1 &&
        single_cta_max_rows <= TPQ_RESIDUAL_MAX_ROWS,
        "Attention residual single-CTA threshold must be in [1,16]");
    if (rows <= single_cta_max_rows) {
        attention_residual_bf16_kernel<<<
            1,
            TPQ_RESIDUAL_THREADS,
            0,
            stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    prefix.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    residual.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    projection.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    norm_weight.data_ptr<at::BFloat16>()),
                post_norm_ptr,
                residual_inverse_ptr,
                reinterpret_cast<__nv_bfloat16*>(
                    output.data_ptr<at::BFloat16>()),
                static_cast<int>(residual.size(1)),
                static_cast<int>(prefix.size(1)),
                static_cast<float>(eps));
    } else {
        TORCH_CHECK(
            score_workspace.has_value(),
            "deep Attention residual requires a score workspace");
        const auto workspace = score_workspace.value();
        TORCH_CHECK(
            workspace.is_cuda() &&
            workspace.scalar_type() == at::kFloat &&
            workspace.is_contiguous() &&
            workspace.numel() >= TPQ_RESIDUAL_STAGED_MAX_ROWS &&
            workspace.get_device() == prefix.get_device(),
            "Attention residual score workspace must be colocated "
            "contiguous float32[>=32]");
        attention_residual_scores_bf16_kernel<<<
            rows,
            TPQ_RESIDUAL_THREADS,
            0,
            stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    prefix.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    residual.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    projection.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    norm_weight.data_ptr<at::BFloat16>()),
                residual_inverse_ptr,
                workspace.data_ptr<float>(),
                static_cast<int>(residual.size(1)),
                static_cast<int>(prefix.size(1)),
                static_cast<float>(eps));
        attention_residual_mix_bf16_kernel<<<
            1,
            TPQ_RESIDUAL_THREADS,
            0,
            stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    prefix.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    residual.data_ptr<at::BFloat16>()),
                post_norm_ptr,
                workspace.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(
                    output.data_ptr<at::BFloat16>()),
                static_cast<int>(residual.size(1)),
                static_cast<int>(prefix.size(1)),
                static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void tp_hidden_add_bf16_kernel(
    const __nv_bfloat16* __restrict__ left,
    const __nv_bfloat16* __restrict__ right,
    __nv_bfloat16* __restrict__ output,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    ) {
        output[index] = __float2bfloat16_rn(
            __bfloat162float(left[index])
            + __bfloat162float(right[index]));
    }
}

std::vector<torch::Tensor> tp_hidden_add_batch(
    std::vector<torch::Tensor> left,
    std::vector<int64_t> left_events,
    std::vector<torch::Tensor> right,
    std::vector<int64_t> right_events,
    std::vector<torch::Tensor> outputs,
    std::vector<int64_t> output_events)
{
    const size_t count = outputs.size();
    TORCH_CHECK(
        count > 0 &&
        left.size() == count &&
        left_events.size() == count &&
        right.size() == count &&
        right_events.size() == count &&
        output_events.size() == count,
        "TPHidden add vectors must be non-empty and size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    for (size_t rank = 0; rank < count; ++rank) {
        const int target = outputs[rank].get_device();
        TORCH_CHECK(
            left[rank].is_cuda() &&
            right[rank].is_cuda() &&
            outputs[rank].is_cuda() &&
            left[rank].get_device() == target &&
            right[rank].get_device() == target &&
            left[rank].scalar_type() == at::kBFloat16 &&
            right[rank].scalar_type() == at::kBFloat16 &&
            outputs[rank].scalar_type() == at::kBFloat16 &&
            left[rank].is_contiguous() &&
            right[rank].is_contiguous() &&
            outputs[rank].is_contiguous() &&
            left[rank].sizes() == outputs[rank].sizes() &&
            right[rank].sizes() == outputs[rank].sizes(),
            "TPHidden add requires matching colocated BF16 tensors");
        C10_CUDA_CHECK(cudaSetDevice(target));
        const auto stream = at::cuda::getCurrentCUDAStream(target);
        C10_CUDA_CHECK(cudaStreamWaitEvent(
            stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(left_events[rank])),
            0));
        C10_CUDA_CHECK(cudaStreamWaitEvent(
            stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(right_events[rank])),
            0));
        const int items = static_cast<int>(outputs[rank].numel());
        const int blocks = std::min(32, (items + 255) / 256);
        tp_hidden_add_bf16_kernel<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                left[rank].data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                right[rank].data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                outputs[rank].data_ptr<at::BFloat16>()),
            items);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(output_events[rank])),
            stream));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
    return outputs;
}

std::vector<torch::Tensor> tp_hidden_rmsnorm_batch(
    std::vector<torch::Tensor> inputs,
    std::vector<int64_t> input_events,
    std::vector<torch::Tensor> weights,
    double eps,
    std::vector<torch::Tensor> outputs,
    std::vector<int64_t> output_events)
{
    const size_t count = outputs.size();
    TORCH_CHECK(
        count > 0 &&
        inputs.size() == count &&
        input_events.size() == count &&
        weights.size() == count &&
        output_events.size() == count,
        "TPHidden RMSNorm vectors must be non-empty and size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    for (size_t rank = 0; rank < count; ++rank) {
        const int target = outputs[rank].get_device();
        TORCH_CHECK(
            inputs[rank].get_device() == target &&
            weights[rank].get_device() == target,
            "TPHidden RMSNorm tensors must be colocated");
        C10_CUDA_CHECK(cudaSetDevice(target));
        const auto stream = at::cuda::getCurrentCUDAStream(target);
        C10_CUDA_CHECK(cudaStreamWaitEvent(
            stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(input_events[rank])),
            0));
        rmsnorm_bf16(
            inputs[rank],
            weights[rank],
            eps,
            outputs[rank]);
        C10_CUDA_CHECK(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(output_events[rank])),
            stream));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
    return outputs;
}

std::vector<torch::Tensor> tp_hidden_residual_mix_batch(
    std::vector<torch::Tensor> prefixes,
    std::vector<int64_t> prefix_events,
    std::vector<torch::Tensor> residuals,
    std::vector<int64_t> residual_events,
    std::vector<torch::Tensor> projections,
    std::vector<torch::Tensor> norm_weights,
    std::vector<torch::Tensor> post_norm_weights,
    std::vector<torch::Tensor> workspaces,
    std::vector<torch::Tensor> residual_inverses,
    double eps,
    long single_cta_max_rows,
    std::vector<torch::Tensor> outputs,
    std::vector<int64_t> output_events)
{
    const size_t count = outputs.size();
    TORCH_CHECK(
        count > 0 &&
        prefixes.size() == count &&
        prefix_events.size() == count &&
        residuals.size() == count &&
        residual_events.size() == count &&
        projections.size() == count &&
        norm_weights.size() == count &&
        post_norm_weights.size() == count &&
        workspaces.size() == count &&
        residual_inverses.size() == count &&
        output_events.size() == count,
        "TPHidden residual vectors must be non-empty and size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    for (size_t rank = 0; rank < count; ++rank) {
        const int target = outputs[rank].get_device();
        TORCH_CHECK(
            prefixes[rank].get_device() == target &&
            residuals[rank].get_device() == target &&
            projections[rank].get_device() == target &&
            norm_weights[rank].get_device() == target &&
            post_norm_weights[rank].get_device() == target &&
            workspaces[rank].get_device() == target &&
            residual_inverses[rank].get_device() == target,
            "TPHidden residual tensors must be colocated");
        C10_CUDA_CHECK(cudaSetDevice(target));
        const auto stream = at::cuda::getCurrentCUDAStream(target);
        C10_CUDA_CHECK(cudaStreamWaitEvent(
            stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(prefix_events[rank])),
            0));
        C10_CUDA_CHECK(cudaStreamWaitEvent(
            stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(residual_events[rank])),
            0));
        attention_residual_bf16(
            prefixes[rank],
            residuals[rank],
            projections[rank],
            norm_weights[rank],
            post_norm_weights[rank],
            eps,
            outputs[rank],
            workspaces[rank],
            single_cta_max_rows,
            residual_inverses[rank]);
        C10_CUDA_CHECK(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(output_events[rank])),
            stream));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
    return outputs;
}

torch::Tensor glm_mla_bmm_decode(
    torch::Tensor input,
    torch::Tensor weight,
    bool transpose_weight,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        input.is_cuda() && weight.is_cuda(),
        "MLA decode GEMM inputs must be CUDA");
    TORCH_CHECK(
        input.scalar_type() == at::kBFloat16 &&
        weight.scalar_type() == at::kBFloat16,
        "MLA decode GEMM inputs must be BF16");
    TORCH_CHECK(
        input.dim() == 3 &&
        input.size(1) == 1 &&
        input.stride(2) == 1 &&
        weight.dim() == 3 &&
        weight.is_contiguous() &&
        input.size(0) == weight.size(0),
        "MLA decode GEMM expects input[H,1,K] and contiguous weight");
    const int heads = static_cast<int>(input.size(0));
    const int inner = static_cast<int>(input.size(2));
    const int output_width = static_cast<int>(
        transpose_weight ? weight.size(1) : weight.size(2));
    TORCH_CHECK(
        (
            transpose_weight
            ? weight.size(2) == inner
            : weight.size(1) == inner
        ) &&
        input.get_device() == weight.get_device(),
        "MLA decode GEMM shapes/devices do not match");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {heads, 1, output_width},
            input.options());
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kBFloat16 &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef(
            {heads, 1, output_width}) &&
        output.get_device() == input.get_device(),
        "MLA decode GEMM output must be contiguous BF16 [H,1,N]");

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasOperation_t weight_op = transpose_weight
        ? CUBLAS_OP_T
        : CUBLAS_OP_N;
    const int lda = transpose_weight
        ? inner
        : output_width;
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
        handle,
        weight_op,
        CUBLAS_OP_N,
        output_width,
        1,
        inner,
        &alpha,
        weight.data_ptr<at::BFloat16>(),
        CUDA_R_16BF,
        lda,
        static_cast<long long>(weight.stride(0)),
        input.data_ptr<at::BFloat16>(),
        CUDA_R_16BF,
        inner,
        static_cast<long long>(input.stride(0)),
        &beta,
        output.data_ptr<at::BFloat16>(),
        CUDA_R_16BF,
        output_width,
        static_cast<long long>(output.stride(0)),
        heads,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return output;
}

// ---- RoPE (interleaved pairs, decode T=1 fast path) ----
// rows share one (cos, sin) phase: out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
//                                  out[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
// inverse=true: conjugate (sin negated).

__global__ void rope1_kernel(
    const float* __restrict__ x,    // [N, rd]
    const float* __restrict__ cs,   // [rd/2]
    const float* __restrict__ sn,   // [rd/2]
    float* __restrict__ out,        // [N, rd]
    const int rd2, const int inverse)
{
    const int r = blockIdx.x;
    const float* xr = x + (long)r * rd2 * 2;
    float* orow = out + (long)r * rd2 * 2;
    for (int i = threadIdx.x; i < rd2; i += blockDim.x) {
        const float c = cs[i], s = inverse ? -sn[i] : sn[i];
        const float x1 = xr[2 * i], x2 = xr[2 * i + 1];
        orow[2 * i] = x1 * c - x2 * s;
        orow[2 * i + 1] = x1 * s + x2 * c;
    }
}

torch::Tensor rope1(torch::Tensor x, torch::Tensor cs, torch::Tensor sn,
                    bool inverse) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat, "x must be CUDA f32");
    auto xc = x.contiguous();
    const int rd = (int)xc.size(-1);
    auto x2 = xc.view({-1, rd});
    const int N = (int)x2.size(0);
    auto out = torch::empty_like(x2);
    auto stream = at::cuda::getCurrentCUDAStream();
    rope1_kernel<<<N, 64, 0, stream>>>(
        x2.data_ptr<float>(), cs.contiguous().data_ptr<float>(),
        sn.contiguous().data_ptr<float>(), out.data_ptr<float>(),
        rd / 2, inverse ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view(xc.sizes());
}

// ---- GLM MLA RoPE (Q and shared K in one launch, HF cat layout) ----
// The reference first computes two rounded multiplies and then a rounded
// add/sub in separate ATen kernels.  Explicit round-to-nearest intrinsics keep
// this fused implementation from contracting the expression into an FMA.

__global__ void glm_rope_qk_kernel(
    const float* __restrict__ q,     // [H*T, rd], interleaved input
    const float* __restrict__ k,     // [T, rd], interleaved input
    const float* __restrict__ cs,    // [T, rd/2]
    const float* __restrict__ sn,    // [T, rd/2]
    float* __restrict__ qo,          // [H*T, rd], cat output
    float* __restrict__ ko,          // [T, rd], cat output
    const int nq, const int T, const int rd2)
{
    const int row = blockIdx.x;
    const bool is_q = row < nq;
    const int local_row = is_q ? row : row - nq;
    const int phase = is_q ? local_row % T : local_row;
    const float* xr = (is_q ? q : k) + (long)local_row * rd2 * 2;
    float* yr = (is_q ? qo : ko) + (long)local_row * rd2 * 2;
    const float* cr = cs + (long)phase * rd2;
    const float* sr = sn + (long)phase * rd2;
    for (int i = threadIdx.x; i < rd2; i += blockDim.x) {
        const float x1 = xr[2 * i];
        const float x2 = xr[2 * i + 1];
        const float a = __fmul_rn(x1, cr[i]);
        const float b = __fmul_rn(x2, sr[i]);
        const float c = __fmul_rn(x2, cr[i]);
        const float d = __fmul_rn(x1, sr[i]);
        yr[i] = __fsub_rn(a, b);
        yr[i + rd2] = __fadd_rn(c, d);
    }
}

std::vector<torch::Tensor> glm_rope_qk(
    torch::Tensor q, torch::Tensor k, torch::Tensor cs, torch::Tensor sn) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && cs.is_cuda() && sn.is_cuda(),
                "GLM RoPE tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                cs.scalar_type() == at::kFloat && sn.scalar_type() == at::kFloat,
                "GLM RoPE tensors must be float32");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && cs.dim() == 2 && sn.dim() == 2,
                "GLM RoPE expects q[H,T,D], k[1,T,D], cos/sin[T,D/2]");
    const int T = (int)q.size(1);
    const int rd = (int)q.size(2);
    TORCH_CHECK(k.size(0) == 1 && k.size(1) == T && k.size(2) == rd,
                "GLM RoPE q/k shape mismatch");
    TORCH_CHECK(rd % 2 == 0 && cs.size(0) == T && sn.size(0) == T &&
                cs.size(1) * 2 == rd && sn.size(1) * 2 == rd,
                "GLM RoPE phase shape mismatch");
    auto qc = q.contiguous();
    auto kc = k.contiguous();
    auto cc = cs.contiguous();
    auto sc = sn.contiguous();
    auto qo = torch::empty_like(qc);
    auto ko = torch::empty_like(kc);
    const int nq = (int)(q.size(0) * T);
    auto stream = at::cuda::getCurrentCUDAStream();
    glm_rope_qk_kernel<<<nq + T, 64, 0, stream>>>(
        qc.data_ptr<float>(), kc.data_ptr<float>(),
        cc.data_ptr<float>(), sc.data_ptr<float>(),
        qo.data_ptr<float>(), ko.data_ptr<float>(),
        nq, T, rd / 2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {qo, ko};
}

// GLM decode-only latent KV preparation.  The reference path launches:
// RMSNorm(c_kv), a BF16 KV copy, a strided Q copy, RoPE(Q/K), a BF16 Q
// conversion and a BF16 K copy.  Decode has T=1 and fixed destination rows,
// so two kernels can preserve the same FP32 arithmetic and BF16 boundaries.

__global__ void glm_ckv_rms_write_bf16_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    __nv_bfloat16* __restrict__ output_base,
    const int64_t* __restrict__ position_ptr,
    const int capacity,
    const int width,
    const float eps)
{
    float acc = 0.f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = x[i];
        acc += value * value;
    }
    __shared__ float reduction[32];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    if ((threadIdx.x & 31) == 0)
        reduction[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = threadIdx.x < (blockDim.x + 31) / 32
            ? reduction[threadIdx.x]
            : 0.f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(
                0xffffffffu, value, offset);
        if (threadIdx.x == 0)
            reduction[0] = value;
    }
    __syncthreads();
    const float scale = rsqrtf(
        reduction[0] / static_cast<float>(width) + eps);
    const int64_t position = position_ptr[0];
    if (position < 0 || position >= capacity)
        return;
    __nv_bfloat16* output =
        output_base + position * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = w[i] * (x[i] * scale);
        output[i] = __float2bfloat16_rn(value);
    }
}

__global__ void glm_rope_qk_write_bf16_kernel(
    const float* __restrict__ q,
    const long q_row_stride,
    const float* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ q_output,
    __nv_bfloat16* __restrict__ k_output_base,
    const int64_t* __restrict__ position_ptr,
    const int capacity,
    const int heads,
    const int rope_half)
{
    const int64_t position = position_ptr[0];
    if (position < 0 || position >= capacity)
        return;
    const float* cs =
        cos_cache + position * rope_half;
    const float* sn =
        sin_cache + position * rope_half;
    const int row = blockIdx.x;
    const bool is_q = row < heads;
    const float* input = is_q
        ? q + static_cast<long>(row) * q_row_stride
        : k;
    __nv_bfloat16* output = is_q
        ? q_output + static_cast<long>(row) * rope_half * 2
        : k_output_base + position * rope_half * 2;
    for (
        int index = threadIdx.x;
        index < rope_half;
        index += blockDim.x
    ) {
        const float x1 = input[2 * index];
        const float x2 = input[2 * index + 1];
        const float a = __fmul_rn(x1, cs[index]);
        const float b = __fmul_rn(x2, sn[index]);
        const float c = __fmul_rn(x2, cs[index]);
        const float d = __fmul_rn(x1, sn[index]);
        output[index] = __float2bfloat16_rn(
            __fsub_rn(a, b));
        output[index + rope_half] = __float2bfloat16_rn(
            __fadd_rn(c, d));
    }
}

torch::Tensor glm_latent_kv_decode_prepare(
    torch::Tensor c_raw,
    torch::Tensor c_weight,
    torch::Tensor q_rot,
    torch::Tensor k_rot,
    torch::Tensor cos_cache,
    torch::Tensor sin_cache,
    torch::Tensor ckv_buffer,
    torch::Tensor krot_buffer,
    torch::Tensor position,
    double eps,
    c10::optional<torch::Tensor> q_output_buffer)
{
    TORCH_CHECK(
        c_raw.is_cuda() && c_weight.is_cuda() &&
        q_rot.is_cuda() && k_rot.is_cuda() &&
        cos_cache.is_cuda() && sin_cache.is_cuda() &&
        ckv_buffer.is_cuda() && krot_buffer.is_cuda() &&
        position.is_cuda(),
        "GLM latent decode tensors must be CUDA");
    TORCH_CHECK(
        c_raw.scalar_type() == at::kFloat &&
        c_weight.scalar_type() == at::kFloat &&
        q_rot.scalar_type() == at::kFloat &&
        k_rot.scalar_type() == at::kFloat &&
        cos_cache.scalar_type() == at::kFloat &&
        sin_cache.scalar_type() == at::kFloat,
        "GLM latent decode inputs must be float32");
    TORCH_CHECK(
        ckv_buffer.scalar_type() == at::kBFloat16 &&
        krot_buffer.scalar_type() == at::kBFloat16 &&
        position.scalar_type() == at::kLong &&
        position.numel() == 1 &&
        position.is_contiguous(),
        "GLM latent KV buffers/position have invalid dtypes");
    TORCH_CHECK(
        c_raw.dim() == 2 && c_raw.size(0) == 1 &&
        c_weight.dim() == 1 &&
        c_raw.size(1) == c_weight.size(0),
        "GLM latent C shapes do not match");
    TORCH_CHECK(
        q_rot.dim() == 3 && q_rot.size(1) == 1 &&
        k_rot.dim() == 3 && k_rot.size(0) == 1 &&
        k_rot.size(1) == 1 &&
        q_rot.size(2) == k_rot.size(2) &&
        q_rot.size(2) % 2 == 0,
        "GLM decode RoPE expects Q[H,1,D] and K[1,1,D]");
    const int latent = static_cast<int>(c_raw.size(1));
    const int heads = static_cast<int>(q_rot.size(0));
    const int rope = static_cast<int>(q_rot.size(2));
    TORCH_CHECK(
        ckv_buffer.dim() == 2 &&
        ckv_buffer.size(1) == latent &&
        krot_buffer.dim() == 2 &&
        krot_buffer.size(1) == rope &&
        ckv_buffer.size(0) == krot_buffer.size(0),
        "GLM latent KV destination shape mismatch");
    TORCH_CHECK(
        cos_cache.dim() == 2 && sin_cache.dim() == 2 &&
        cos_cache.sizes() == sin_cache.sizes() &&
        cos_cache.size(1) * 2 == rope &&
        cos_cache.size(0) > 0,
        "GLM RoPE cache shape/position mismatch");
    const int device = c_raw.get_device();
    TORCH_CHECK(
        c_weight.get_device() == device &&
        q_rot.get_device() == device &&
        k_rot.get_device() == device &&
        cos_cache.get_device() == device &&
        sin_cache.get_device() == device &&
        ckv_buffer.get_device() == device &&
        krot_buffer.get_device() == device &&
        position.get_device() == device,
        "GLM latent decode tensors must share one device");

    auto q_output = q_output_buffer.has_value()
        ? q_output_buffer.value()
        : torch::empty(
            q_rot.sizes(),
            q_rot.options().dtype(at::kBFloat16));
    TORCH_CHECK(
        q_output.is_cuda() &&
        q_output.scalar_type() == at::kBFloat16 &&
        q_output.is_contiguous() &&
        q_output.sizes() == q_rot.sizes() &&
        q_output.get_device() == device,
        "GLM latent Q output must be contiguous BF16 and match Q shape");
    auto stream = at::cuda::getCurrentCUDAStream();
    glm_ckv_rms_write_bf16_kernel<<<1, 256, 0, stream>>>(
        c_raw.data_ptr<float>(),
        c_weight.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(
            ckv_buffer.data_ptr<at::BFloat16>()),
        position.data_ptr<int64_t>(),
        static_cast<int>(std::min(
            ckv_buffer.size(0),
            cos_cache.size(0))),
        latent,
        static_cast<float>(eps));
    glm_rope_qk_write_bf16_kernel<<<
        heads + 1,
        64,
        0,
        stream>>>(
            q_rot.data_ptr<float>(),
            q_rot.stride(0),
            k_rot.data_ptr<float>(),
            cos_cache.data_ptr<float>(),
            sin_cache.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                q_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                krot_buffer.data_ptr<at::BFloat16>()),
            position.data_ptr<int64_t>(),
            static_cast<int>(std::min(
                krot_buffer.size(0),
                cos_cache.size(0))),
            heads,
            rope / 2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q_output;
}

// Merge the two latent-MLA score components and their scale in one launch.
// Inputs are BF16 GEMM outputs; the explicit operations mirror
// a.float()/scale + b.float()/scale without changing either GEMM.
__global__ void glm_merge_scores_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    float* __restrict__ out,
    const long n, const float scale)
{
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (long)blockDim.x * gridDim.x) {
        const float av = __fdiv_rn(__bfloat162float(a[i]), scale);
        const float bv = __fdiv_rn(__bfloat162float(b[i]), scale);
        out[i] = __fadd_rn(av, bv);
    }
}

torch::Tensor glm_merge_scores(
    torch::Tensor a, torch::Tensor b, double scale) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "GLM scores must be CUDA");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 &&
                b.scalar_type() == at::kBFloat16,
                "GLM score merge currently requires BF16");
    TORCH_CHECK(a.sizes() == b.sizes(), "GLM score shapes must match");
    auto ac = a.contiguous();
    auto bc = b.contiguous();
    auto out = torch::empty(ac.sizes(), ac.options().dtype(at::kFloat));
    const long n = ac.numel();
    const int blocks = (int)std::min<long>((n + 255) / 256, 4096);
    auto stream = at::cuda::getCurrentCUDAStream();
    glm_merge_scores_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            ac.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            bc.data_ptr<at::BFloat16>()),
        out.data_ptr<float>(), n, (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---- DSV4 decode attention core (B=1, T=1, f32) ----
// Window and compressed KV are separate contiguous views. Scores are kept in
// shared memory so score matvec, sink-softmax and value reduction need one
// launch instead of a chain of tiny ATen kernels.

__device__ __forceinline__ float warp_max_f32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float warp_sum_f32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---- Latent MLA decode core (BF16, dynamic device-side length) ----
//
// Q-B, latent preparation, this core, Wuv and O-projection are captured in
// one rank-local CUDA Graph. Reading the device position inside the kernel
// keeps the graph valid for every context length without a host-side plan.

__global__ void latent_mla_attention_scores_kernel(
    const __nv_bfloat16* __restrict__ qa,
    const __nv_bfloat16* __restrict__ qrot,
    const __nv_bfloat16* __restrict__ ckv,
    const __nv_bfloat16* __restrict__ krot,
    const int64_t* __restrict__ position,
    float* __restrict__ scores,
    const int heads,
    const int latent,
    const int rope,
    const int capacity,
    const float scale,
    const bool scores_only)
{
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;
    const int length = min(
        max(static_cast<int>(position[0]) + 1, 1),
        capacity);
    __shared__ float reduced[8];
    __shared__ float score_max;
    __shared__ float denominator;

    const __nv_bfloat16* qah =
        qa + static_cast<long>(head) * latent;
    const __nv_bfloat16* qrh =
        qrot + static_cast<long>(head) * rope;
    float local_max = -INFINITY;
    for (int token = tid; token < length; token += blockDim.x) {
        const __nv_bfloat16* ck =
            ckv + static_cast<long>(token) * latent;
        const __nv_bfloat16* kr =
            krot + static_cast<long>(token) * rope;
        float nope_score = 0.0f;
        float rope_score = 0.0f;
        for (int dim = 0; dim < latent; ++dim) {
            nope_score = fmaf(
                __bfloat162float(qah[dim]),
                __bfloat162float(ck[dim]),
                nope_score);
        }
        for (int dim = 0; dim < rope; ++dim) {
            rope_score = fmaf(
                __bfloat162float(qrh[dim]),
                __bfloat162float(kr[dim]),
                rope_score);
        }
        // Match eager decode exactly: both GEMMs produce BF16, their sum and
        // scalar multiply are rounded to BF16, and only then softmax promotes
        // the score tensor to FP32.
        nope_score = __bfloat162float(
            __float2bfloat16_rn(nope_score));
        rope_score = __bfloat162float(
            __float2bfloat16_rn(rope_score));
        const float merged = __bfloat162float(
            __float2bfloat16_rn(nope_score + rope_score));
        const float score = __bfloat162float(
            __float2bfloat16_rn(merged / scale));
        scores[static_cast<long>(head) * capacity + token] = score;
        local_max = fmaxf(local_max, score);
    }
    if (scores_only)
        return;
    local_max = warp_max_f32(local_max);
    if (lane == 0) reduced[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float value = lane < nwarps ? reduced[lane] : -INFINITY;
        value = warp_max_f32(value);
        if (lane == 0) score_max = value;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int token = tid; token < length; token += blockDim.x) {
        float value = expf(
            scores[static_cast<long>(head) * capacity + token]
            - score_max);
        scores[static_cast<long>(head) * capacity + token] = value;
        local_sum += value;
    }
    local_sum = warp_sum_f32(local_sum);
    if (lane == 0) reduced[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float value = lane < nwarps ? reduced[lane] : 0.0f;
        value = warp_sum_f32(value);
        if (lane == 0) denominator = value;
    }
    __syncthreads();
    for (int token = tid; token < length; token += blockDim.x) {
        scores[static_cast<long>(head) * capacity + token] =
            __bfloat162float(
                __float2bfloat16_rn(
                    scores[
                        static_cast<long>(head) * capacity + token
                    ] / denominator));
    }
}

__global__ void latent_mla_attention_value_kernel(
    const float* __restrict__ scores,
    const __nv_bfloat16* __restrict__ ckv,
    const int64_t* __restrict__ position,
    __nv_bfloat16* __restrict__ output,
    const int heads,
    const int latent,
    const int capacity)
{
    const int head = blockIdx.x;
    const int dim = blockIdx.y * blockDim.x + threadIdx.x;
    if (head >= heads || dim >= latent) return;
    const int length = min(
        max(static_cast<int>(position[0]) + 1, 1),
        capacity);
    const float* weights =
        scores + static_cast<long>(head) * capacity;
    float value = 0.0f;
    for (int token = 0; token < length; ++token) {
        value = fmaf(
            weights[token],
            __bfloat162float(
                ckv[static_cast<long>(token) * latent + dim]),
            value);
    }
    output[static_cast<long>(head) * latent + dim] =
        __float2bfloat16_rn(value);
}

torch::Tensor latent_mla_attention_decode(
    torch::Tensor qa,
    torch::Tensor qrot,
    torch::Tensor ckv,
    torch::Tensor krot,
    torch::Tensor position,
    double scale,
    torch::Tensor score_workspace,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        qa.is_cuda() && qrot.is_cuda() && ckv.is_cuda() &&
        krot.is_cuda() && position.is_cuda() &&
        score_workspace.is_cuda(),
        "latent MLA tensors must be CUDA");
    TORCH_CHECK(
        qa.scalar_type() == at::kBFloat16 &&
        qrot.scalar_type() == at::kBFloat16 &&
        ckv.scalar_type() == at::kBFloat16 &&
        krot.scalar_type() == at::kBFloat16 &&
        position.scalar_type() == at::kLong &&
        score_workspace.scalar_type() == at::kFloat,
        "latent MLA requires BF16 state, int64 position and FP32 scores");
    TORCH_CHECK(
        qa.is_contiguous() && qrot.is_contiguous() &&
        ckv.is_contiguous() && krot.is_contiguous() &&
        position.is_contiguous() && score_workspace.is_contiguous(),
        "latent MLA tensors must be contiguous");
    TORCH_CHECK(
        qa.dim() == 3 && qa.size(1) == 1 &&
        qrot.dim() == 3 && qrot.size(1) == 1 &&
        qa.size(0) == qrot.size(0) &&
        ckv.dim() == 2 && krot.dim() == 2 &&
        ckv.size(0) == krot.size(0) &&
        ckv.size(1) == qa.size(2) &&
        krot.size(1) == qrot.size(2) &&
        score_workspace.sizes() == torch::IntArrayRef(
            {qa.size(0), ckv.size(0)}) &&
        position.numel() == 1 && scale > 0.0,
        "latent MLA shapes do not match");
    const auto device = qa.get_device();
    TORCH_CHECK(
        qrot.get_device() == device &&
        ckv.get_device() == device &&
        krot.get_device() == device &&
        position.get_device() == device &&
        score_workspace.get_device() == device,
        "latent MLA tensors must share one device");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty_like(qa);
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kBFloat16 &&
        output.is_contiguous() &&
        output.sizes() == qa.sizes() &&
        output.get_device() == device,
        "latent MLA output must be contiguous BF16 and match Q-A");
    const int heads = static_cast<int>(qa.size(0));
    const int latent = static_cast<int>(qa.size(2));
    const int rope = static_cast<int>(qrot.size(2));
    const int capacity = static_cast<int>(ckv.size(0));
    auto stream = at::cuda::getCurrentCUDAStream();
    latent_mla_attention_scores_kernel<<<heads, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            qa.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            qrot.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            ckv.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            krot.data_ptr<at::BFloat16>()),
        position.data_ptr<int64_t>(),
        score_workspace.data_ptr<float>(),
        heads,
        latent,
        rope,
        capacity,
        static_cast<float>(scale),
        false);
    const dim3 value_grid(
        heads,
        (latent + 255) / 256);
    latent_mla_attention_value_kernel<<<
        value_grid,
        256,
        0,
        stream>>>(
            score_workspace.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(
                ckv.data_ptr<at::BFloat16>()),
            position.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            heads,
            latent,
            capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor latent_mla_attention_scores(
    torch::Tensor qa,
    torch::Tensor qrot,
    torch::Tensor ckv,
    torch::Tensor krot,
    torch::Tensor position,
    double scale,
    torch::Tensor score_workspace)
{
    TORCH_CHECK(
        qa.is_cuda() && qrot.is_cuda() && ckv.is_cuda() &&
        krot.is_cuda() && position.is_cuda() &&
        score_workspace.is_cuda() &&
        qa.scalar_type() == at::kBFloat16 &&
        qrot.scalar_type() == at::kBFloat16 &&
        ckv.scalar_type() == at::kBFloat16 &&
        krot.scalar_type() == at::kBFloat16 &&
        position.scalar_type() == at::kLong &&
        score_workspace.scalar_type() == at::kFloat,
        "latent MLA score tensors have invalid device or dtype");
    TORCH_CHECK(
        qa.is_contiguous() && qrot.is_contiguous() &&
        ckv.is_contiguous() && krot.is_contiguous() &&
        position.is_contiguous() && score_workspace.is_contiguous() &&
        qa.dim() == 3 && qa.size(1) == 1 &&
        qrot.dim() == 3 && qrot.size(1) == 1 &&
        qa.size(0) == qrot.size(0) &&
        ckv.dim() == 2 && krot.dim() == 2 &&
        ckv.size(0) == krot.size(0) &&
        ckv.size(1) == qa.size(2) &&
        krot.size(1) == qrot.size(2) &&
        score_workspace.sizes() == torch::IntArrayRef(
            {qa.size(0), ckv.size(0)}) &&
        position.numel() == 1 && scale > 0.0,
        "latent MLA score shapes do not match");
    const int device = qa.get_device();
    TORCH_CHECK(
        qrot.get_device() == device &&
        ckv.get_device() == device &&
        krot.get_device() == device &&
        position.get_device() == device &&
        score_workspace.get_device() == device,
        "latent MLA score tensors must share one device");
    auto stream = at::cuda::getCurrentCUDAStream();
    latent_mla_attention_scores_kernel<<<
        qa.size(0),
        256,
        0,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                qa.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                qrot.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                ckv.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                krot.data_ptr<at::BFloat16>()),
            position.data_ptr<int64_t>(),
            score_workspace.data_ptr<float>(),
            static_cast<int>(qa.size(0)),
            static_cast<int>(qa.size(2)),
            static_cast<int>(qrot.size(2)),
            static_cast<int>(ckv.size(0)),
            static_cast<float>(scale),
            true);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return score_workspace;
}

__global__ void dsv4_attn_decode_kernel(
    const float* __restrict__ q,          // [H,D]
    const float* __restrict__ win_kv,     // [W,D]
    const int64_t* __restrict__ win_pos,  // [W], negative means invalid
    const float* __restrict__ comp_kv,    // [C,D]
    const float* __restrict__ sink,       // [H]
    const float* __restrict__ cs,         // [rd/2]
    const float* __restrict__ sn,         // [rd/2]
    float* __restrict__ out,              // [H,D]
    const int H, const int D, const int W, const int C, const int rd,
    const float scale)
{
    const int h = blockIdx.x;
    if (h >= H) return;
    const int tid = threadIdx.x;
    const int S = W + C;
    extern __shared__ float smem[];
    float* qsh = smem;             // D
    float* scores = qsh + D;       // S
    float* osh = scores + S;       // D
    float* red = osh + D;          // one value per warp
    __shared__ float score_max;
    __shared__ float denom;

    const float* qh = q + (long)h * D;
    for (int d = tid; d < D; d += blockDim.x)
        qsh[d] = qh[d];
    __syncthreads();

    for (int s = tid; s < S; s += blockDim.x) {
        const bool valid = s >= W || win_pos[s] >= 0;
        const float* kv = s < W ? win_kv + (long)s * D
                                : comp_kv + (long)(s - W) * D;
        float acc = 0.f;
        if (valid) {
            for (int d = 0; d < D; ++d)
                acc = fmaf(qsh[d], kv[d], acc);
        }
        scores[s] = valid ? acc * scale : -INFINITY;
    }
    __syncthreads();

    float mx = -INFINITY;
    for (int s = tid; s < S; s += blockDim.x)
        mx = fmaxf(mx, scores[s]);
    mx = warp_max_f32(mx);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        float v = lane < nwarps ? red[lane] : -INFINITY;
        v = warp_max_f32(v);
        if (lane == 0) score_max = v;
    }
    __syncthreads();

    float z = 0.f;
    for (int s = tid; s < S; s += blockDim.x) {
        const float e = expf(scores[s] - score_max);
        scores[s] = e;
        z += e;
    }
    z = warp_sum_f32(z);
    if (lane == 0) red[warp] = z;
    __syncthreads();
    if (warp == 0) {
        float v = lane < nwarps ? red[lane] : 0.f;
        v = warp_sum_f32(v);
        if (lane == 0)
            denom = v + expf(sink[h] - score_max);
    }
    __syncthreads();

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < S; ++s) {
            const float* kv = s < W ? win_kv + (long)s * D
                                    : comp_kv + (long)(s - W) * D;
            acc = fmaf(scores[s], kv[d], acc);
        }
        osh[d] = acc / denom;
    }
    __syncthreads();

    const int plain = D - rd;
    for (int d = tid; d < D; d += blockDim.x) {
        float v;
        if (d < plain) {
            v = osh[d];
        } else {
            const int r = d - plain;
            const int pair = r >> 1;
            const float x0 = osh[plain + 2 * pair];
            const float x1 = osh[plain + 2 * pair + 1];
            v = (r & 1) ? (-x0 * sn[pair] + x1 * cs[pair])
                        : ( x0 * cs[pair] + x1 * sn[pair]);
        }
        out[(long)h * D + d] = v;
    }
}

torch::Tensor dsv4_attn_decode(
    torch::Tensor q, torch::Tensor win_kv, torch::Tensor win_pos,
    torch::Tensor comp_kv, torch::Tensor sink, torch::Tensor cs,
    torch::Tensor sn, double scale) {
    TORCH_CHECK(q.is_cuda() && win_kv.is_cuda() && win_pos.is_cuda()
                && comp_kv.is_cuda() && sink.is_cuda() && cs.is_cuda() && sn.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == at::kFloat && win_kv.scalar_type() == at::kFloat
                && comp_kv.scalar_type() == at::kFloat && sink.scalar_type() == at::kFloat
                && cs.scalar_type() == at::kFloat && sn.scalar_type() == at::kFloat,
                "attention tensors must be float32");
    TORCH_CHECK(win_pos.scalar_type() == at::kLong, "win_pos must be int64");
    TORCH_CHECK(q.dim() == 3 && q.size(0) == 1, "q must be [1,H,D]");
    TORCH_CHECK(win_kv.dim() == 3 && win_kv.size(0) == 1, "win_kv must be [1,W,D]");
    TORCH_CHECK(comp_kv.dim() == 3 && comp_kv.size(0) == 1, "comp_kv must be [1,C,D]");
    const int H = (int)q.size(1), D = (int)q.size(2);
    const int W = (int)win_kv.size(1), C = (int)comp_kv.size(1);
    const int rd = (int)cs.numel() * 2;
    TORCH_CHECK(win_kv.size(2) == D && comp_kv.size(2) == D, "KV head dim mismatch");
    TORCH_CHECK(win_pos.numel() == W, "win_pos size mismatch");
    TORCH_CHECK(sink.numel() == H, "sink size mismatch");
    TORCH_CHECK(sn.numel() * 2 == rd && rd <= D, "RoPE size mismatch");
    TORCH_CHECK(W + C > 0 && W + C <= 4096, "fused sequence length out of range");

    auto qc = q.contiguous();
    auto wc = win_kv.contiguous();
    auto pc = win_pos.contiguous();
    auto cc = comp_kv.contiguous();
    auto sk = sink.contiguous();
    auto csc = cs.contiguous();
    auto snc = sn.contiguous();
    auto out = torch::empty_like(qc);
    const int threads = 128;
    const size_t smem = (size_t)(2 * D + W + C + 4) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    dsv4_attn_decode_kernel<<<H, threads, smem, stream>>>(
        qc.data_ptr<float>(), wc.data_ptr<float>(), pc.data_ptr<int64_t>(),
        cc.data_ptr<float>(), sk.data_ptr<float>(), csc.data_ptr<float>(),
        snc.data_ptr<float>(), out.data_ptr<float>(),
        H, D, W, C, rd, (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---- DSV4 HC pre (RMS + 24-row GEMV + sinkhorn + channel reduce) ----

template <typename wt_t>
__device__ __forceinline__ float hc_weight(const wt_t* p) {
    return (float)(*p);
}

template <>
__device__ __forceinline__ float hc_weight<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

// ---- DSV4 HC post (4-channel residual mix, BF16 state) ----
//
// result[n,k,d] = post[n,k] * out[n,d]
//               + sum_j comb[n,j,k] * residual[n,j,d]
//
// Decode has N=1.  Four blocks run the output channels in parallel while a
// single launch replaces the casts, broadcasts, multiply, reduction and add
// sequence emitted by the PyTorch reference.

template <typename out_t>
__global__ void dsv4_hc_post_bf16_kernel(
    const out_t* __restrict__ out,                 // [N,D]
    const __nv_bfloat16* __restrict__ residual,    // [N,4,D]
    const __nv_bfloat16* __restrict__ post,        // [N,4]
    const __nv_bfloat16* __restrict__ comb,        // [N,4,4]
    __nv_bfloat16* __restrict__ result,            // [N,4,D]
    const int D)
{
    const int n = blockIdx.x >> 2;
    const int k = blockIdx.x & 3;
    __shared__ float coeff[5];
    if (threadIdx.x == 0) {
        coeff[0] = __bfloat162float(post[(long)n * 4 + k]);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            coeff[1 + j] =
                __bfloat162float(comb[(long)n * 16 + 4 * j + k]);
    }
    __syncthreads();

    const out_t* on = out + (long)n * D;
    const __nv_bfloat16* rn = residual + (long)n * 4 * D;
    __nv_bfloat16* dst = result + ((long)n * 4 + k) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = coeff[0] * hc_weight(on + d);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc = fmaf(
                coeff[1 + j],
                __bfloat162float(rn[(long)j * D + d]),
                acc);
        dst[d] = __float2bfloat16_rn(acc);
    }
}

torch::Tensor dsv4_hc_post(
    torch::Tensor out, torch::Tensor residual,
    torch::Tensor post, torch::Tensor comb)
{
    TORCH_CHECK(
        out.is_cuda() && residual.is_cuda() && post.is_cuda() && comb.is_cuda(),
        "all tensors must be CUDA");
    TORCH_CHECK(
        out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16,
        "out must be float32 or bfloat16");
    TORCH_CHECK(
        residual.scalar_type() == at::kBFloat16 &&
        post.scalar_type() == at::kBFloat16 &&
        comb.scalar_type() == at::kBFloat16,
        "residual/post/comb must be bfloat16");
    TORCH_CHECK(
        residual.dim() >= 2 && residual.size(-2) == 4,
        "residual must end in [4,D]");
    const int D = (int)residual.size(-1);
    const int N = (int)(residual.numel() / (4L * D));
    TORCH_CHECK(out.numel() == (long)N * D, "out must contain N*D values");
    TORCH_CHECK(post.numel() == (long)N * 4, "post must contain N*4 values");
    TORCH_CHECK(comb.numel() == (long)N * 16, "comb must contain N*16 values");

    auto oc = out.contiguous();
    auto rc = residual.contiguous();
    auto pc = post.contiguous();
    auto cc = comb.contiguous();
    auto result = torch::empty_like(rc);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = N * 4;
    if (out.scalar_type() == at::kBFloat16) {
        dsv4_hc_post_bf16_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                oc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                rc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                pc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                cc.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                result.data_ptr<at::BFloat16>()),
            D);
    } else {
        dsv4_hc_post_bf16_kernel<float><<<blocks, 256, 0, stream>>>(
            oc.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(
                rc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                pc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                cc.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                result.data_ptr<at::BFloat16>()),
            D);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result.view(residual.sizes());
}

template <typename wt_t>
__global__ void dsv4_hc_pre_kernel(
    const float* __restrict__ x,       // [N,4,D]
    const wt_t* __restrict__ fn,       // [24,4D]
    const float* __restrict__ scale,   // [3]
    const float* __restrict__ base,    // [24]
    float* __restrict__ y,             // [N,D]
    float* __restrict__ post_out,      // [N,4]
    float* __restrict__ comb_out,      // [N,16]
    const int D, const int iters, const float eps)
{
    const int n = blockIdx.x;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int flatD = 4 * D;
    const float* xn = x + (long)n * flatD;
    __shared__ float red[8];
    __shared__ float inv_rms;
    __shared__ float mixes[24];
    __shared__ float pre[4];
    __shared__ float post[4];
    __shared__ float comb[16];

    float ss = 0.f;
    for (int i = tid; i < flatD; i += 256) {
        const float v = xn[i];
        ss = fmaf(v, v, ss);
    }
    ss = warp_sum_f32(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8 ? red[lane] : 0.f;
        v = warp_sum_f32(v);
        if (lane == 0) inv_rms = rsqrtf(v / (float)flatD + eps);
    }
    __syncthreads();

    #pragma unroll
    for (int batch = 0; batch < 3; ++batch) {
        const int m = batch * 8 + warp;
        const wt_t* fm = fn + (long)m * flatD;
        float acc = 0.f;
        for (int i = lane; i < flatD; i += 32)
            acc = fmaf(xn[i], hc_weight(fm + i), acc);
        acc = warp_sum_f32(acc);
        if (lane == 0) mixes[m] = acc * inv_rms;
    }
    __syncthreads();

    if (tid == 0) {
        const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            pre[j] = 1.f / (1.f + expf(-(mixes[j] * s0 + base[j]))) + eps;
            post[j] = 2.f / (1.f + expf(-(mixes[4 + j] * s1 + base[4 + j])));
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float mx = -INFINITY;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[4 * j + k] = mixes[8 + 4 * j + k] * s2 + base[8 + 4 * j + k];
                mx = fmaxf(mx, comb[4 * j + k]);
            }
            float sum = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[4 * j + k] = expf(comb[4 * j + k] - mx);
                sum += comb[4 * j + k];
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                comb[4 * j + k] = comb[4 * j + k] / sum + eps;
        }
        for (int it = 0; it < iters; ++it) {
            if (it > 0) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float sum = comb[4*j] + comb[4*j+1] + comb[4*j+2] + comb[4*j+3];
                    const float inv = 1.f / (sum + eps);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) comb[4*j+k] *= inv;
                }
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float sum = comb[k] + comb[4+k] + comb[8+k] + comb[12+k];
                const float inv = 1.f / (sum + eps);
                #pragma unroll
                for (int j = 0; j < 4; ++j) comb[4*j+k] *= inv;
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) post_out[(long)n * 4 + j] = post[j];
        #pragma unroll
        for (int j = 0; j < 16; ++j) comb_out[(long)n * 16 + j] = comb[j];
    }
    __syncthreads();

    for (int d = tid; d < D; d += 256) {
        float v = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v = fmaf(pre[j], xn[(long)j * D + d], v);
        y[(long)n * D + d] = v;
    }
}

template <typename wt_t, typename norm_t>
__global__ void dsv4_hc_pre_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,  // [N,4,D]
    const wt_t* __restrict__ fn,          // [24,4D]
    const float* __restrict__ scale,      // [3]
    const float* __restrict__ base,       // [24]
    const norm_t* __restrict__ norm,      // [D]
    __nv_bfloat16* __restrict__ y,        // [N,D]
    __nv_bfloat16* __restrict__ post_out, // [N,4]
    __nv_bfloat16* __restrict__ comb_out, // [N,16]
    const int D, const int iters, const float eps)
{
    const int n = blockIdx.x;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int flatD = 4 * D;
    const __nv_bfloat16* xn = x + (long)n * flatD;
    __shared__ float red[8];
    __shared__ float inv_rms;
    __shared__ float inv_y_rms;
    __shared__ float mixes[24];
    __shared__ float pre[4];
    __shared__ float post[4];
    __shared__ float comb[16];

    float ss = 0.f;
    for (int i = tid; i < flatD; i += 256) {
        const float v = __bfloat162float(xn[i]);
        ss = fmaf(v, v, ss);
    }
    ss = warp_sum_f32(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8 ? red[lane] : 0.f;
        v = warp_sum_f32(v);
        if (lane == 0) inv_rms = rsqrtf(v / (float)flatD + eps);
    }
    __syncthreads();

    #pragma unroll
    for (int batch = 0; batch < 3; ++batch) {
        const int m = batch * 8 + warp;
        const wt_t* fm = fn + (long)m * flatD;
        float acc = 0.f;
        for (int i = lane; i < flatD; i += 32)
            acc = fmaf(__bfloat162float(xn[i]), hc_weight(fm + i), acc);
        acc = warp_sum_f32(acc);
        if (lane == 0) mixes[m] = acc * inv_rms;
    }
    __syncthreads();

    if (tid == 0) {
        const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            pre[j] = 1.f / (1.f + expf(-(mixes[j] * s0 + base[j]))) + eps;
            post[j] = 2.f / (1.f + expf(-(mixes[4 + j] * s1 + base[4 + j])));
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float mx = -INFINITY;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[4 * j + k] =
                    mixes[8 + 4 * j + k] * s2 + base[8 + 4 * j + k];
                mx = fmaxf(mx, comb[4 * j + k]);
            }
            float sum = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[4 * j + k] = expf(comb[4 * j + k] - mx);
                sum += comb[4 * j + k];
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                comb[4 * j + k] = comb[4 * j + k] / sum + eps;
        }
        for (int it = 0; it < iters; ++it) {
            if (it > 0) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float sum =
                        comb[4*j] + comb[4*j+1] + comb[4*j+2] + comb[4*j+3];
                    const float inv = 1.f / (sum + eps);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) comb[4*j+k] *= inv;
                }
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float sum =
                    comb[k] + comb[4+k] + comb[8+k] + comb[12+k];
                const float inv = 1.f / (sum + eps);
                #pragma unroll
                for (int j = 0; j < 4; ++j) comb[4*j+k] *= inv;
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            post_out[(long)n * 4 + j] = __float2bfloat16_rn(post[j]);
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            comb_out[(long)n * 16 + j] = __float2bfloat16_rn(comb[j]);
    }
    __syncthreads();

    float yss = 0.f;
    for (int d = tid; d < D; d += 256) {
        float v = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v = fmaf(pre[j], __bfloat162float(xn[(long)j * D + d]), v);
        yss = fmaf(v, v, yss);
    }
    yss = warp_sum_f32(yss);
    if (lane == 0) red[warp] = yss;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8 ? red[lane] : 0.f;
        v = warp_sum_f32(v);
        if (lane == 0) inv_y_rms = rsqrtf(v / (float)D + eps);
    }
    __syncthreads();

    for (int d = tid; d < D; d += 256) {
        float v = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v = fmaf(pre[j], __bfloat162float(xn[(long)j * D + d]), v);
        y[(long)n * D + d] =
            __float2bfloat16_rn(v * inv_y_rms * hc_weight(norm + d));
    }
}

// SM120 decode fast path.  The original all-in-one HC kernel launches one
// block per token and evaluates the 24-row GEMV in three serial batches of
// eight warps.  For N=1 that leaves almost the whole GPU idle.  Split the
// operation into 24 independent GEMV blocks followed by one finish block.
// The 24 float mixes temporarily occupy the first 96 bytes of the BF16 y
// output; the finish kernel loads them into shared memory before overwriting y.

__global__ void dsv4_hc_mix_parallel_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,       // [N,4,D]
    const __nv_bfloat16* __restrict__ fn,      // [24,4D]
    __nv_bfloat16* __restrict__ scratch,       // y [N,D], temporary mixes
    const int D, const float eps)
{
    const int n = blockIdx.x / 24;
    const int m = blockIdx.x - n * 24;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int flatD = 4 * D;
    const __nv_bfloat16* xn = x + (long)n * flatD;
    const __nv_bfloat16* fm = fn + (long)m * flatD;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(xn);
    const auto* f2 = reinterpret_cast<const __nv_bfloat162*>(fm);
    const int pairs = flatD / 2;
    float dot = 0.f;
    float ss = 0.f;
    for (int i = tid; i < pairs; i += blockDim.x) {
        const float2 xv = __bfloat1622float2(x2[i]);
        const float2 fv = __bfloat1622float2(f2[i]);
        dot = fmaf(xv.x, fv.x, dot);
        dot = fmaf(xv.y, fv.y, dot);
        ss = fmaf(xv.x, xv.x, ss);
        ss = fmaf(xv.y, xv.y, ss);
    }
    dot = warp_sum_f32(dot);
    ss = warp_sum_f32(ss);
    __shared__ float dot_warp[8];
    __shared__ float ss_warp[8];
    if (lane == 0) {
        dot_warp[warp] = dot;
        ss_warp[warp] = ss;
    }
    __syncthreads();
    if (warp == 0) {
        float dv = lane < 8 ? dot_warp[lane] : 0.f;
        float sv = lane < 8 ? ss_warp[lane] : 0.f;
        dv = warp_sum_f32(dv);
        sv = warp_sum_f32(sv);
        if (lane == 0) {
            float* mixes = reinterpret_cast<float*>(
                scratch + (long)n * D);
            mixes[m] = dv * rsqrtf(sv / (float)flatD + eps);
        }
    }
}

__global__ void dsv4_hc_finish_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,       // [N,4,D]
    const float* __restrict__ scale,            // [3]
    const float* __restrict__ base,             // [24]
    const __nv_bfloat16* __restrict__ norm,     // [D]
    __nv_bfloat16* __restrict__ y,              // [N,D], starts as scratch
    __nv_bfloat16* __restrict__ post_out,       // [N,4]
    __nv_bfloat16* __restrict__ comb_out,       // [N,16]
    const int D, const int iters, const float eps)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const __nv_bfloat16* xn = x + (long)n * 4 * D;
    __nv_bfloat16* yn = y + (long)n * D;
    __shared__ float red[8];
    __shared__ float inv_y_rms;
    __shared__ float mixes[24];
    __shared__ float pre[4];
    __shared__ float post[4];
    __shared__ float comb[16];

    if (tid == 0) {
        const float* mix_scratch = reinterpret_cast<const float*>(yn);
        #pragma unroll
        for (int i = 0; i < 24; ++i)
            mixes[i] = mix_scratch[i];

        const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            pre[j] = 1.f / (1.f + expf(-(mixes[j] * s0 + base[j]))) + eps;
            post[j] = 2.f / (1.f + expf(-(mixes[4 + j] * s1 + base[4 + j])));
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float mx = -INFINITY;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[4 * j + k] =
                    mixes[8 + 4 * j + k] * s2 + base[8 + 4 * j + k];
                mx = fmaxf(mx, comb[4 * j + k]);
            }
            float sum = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[4 * j + k] = expf(comb[4 * j + k] - mx);
                sum += comb[4 * j + k];
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                comb[4 * j + k] = comb[4 * j + k] / sum + eps;
        }
        for (int it = 0; it < iters; ++it) {
            if (it > 0) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float sum =
                        comb[4*j] + comb[4*j+1] + comb[4*j+2] + comb[4*j+3];
                    const float inv = 1.f / (sum + eps);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k)
                        comb[4*j+k] *= inv;
                }
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float sum =
                    comb[k] + comb[4+k] + comb[8+k] + comb[12+k];
                const float inv = 1.f / (sum + eps);
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    comb[4*j+k] *= inv;
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            post_out[(long)n * 4 + j] = __float2bfloat16_rn(post[j]);
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            comb_out[(long)n * 16 + j] = __float2bfloat16_rn(comb[j]);
    }
    __syncthreads();

    float yss = 0.f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v = fmaf(pre[j], __bfloat162float(xn[(long)j * D + d]), v);
        yss = fmaf(v, v, yss);
    }
    yss = warp_sum_f32(yss);
    if (lane == 0) red[warp] = yss;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8 ? red[lane] : 0.f;
        v = warp_sum_f32(v);
        if (lane == 0)
            inv_y_rms = rsqrtf(v / (float)D + eps);
    }
    __syncthreads();

    for (int d = tid; d < D; d += blockDim.x) {
        float v = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v = fmaf(pre[j], __bfloat162float(xn[(long)j * D + d]), v);
        yn[d] = __float2bfloat16_rn(
            v * inv_y_rms * __bfloat162float(norm[d]));
    }
}

void launch_dsv4_hc_pre_norm_bf16_parallel(
    const torch::Tensor& x, const torch::Tensor& fn,
    const torch::Tensor& scale, const torch::Tensor& base,
    const torch::Tensor& norm, torch::Tensor& y,
    torch::Tensor& post, torch::Tensor& comb,
    int N, int D, int iters, float eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    dsv4_hc_mix_parallel_bf16_kernel<<<N * 24, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            fn.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            y.data_ptr<at::BFloat16>()),
        D, eps);
    dsv4_hc_finish_norm_bf16_kernel<<<N, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            x.data_ptr<at::BFloat16>()),
        scale.data_ptr<float>(),
        base.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(
            norm.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            y.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            post.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            comb.data_ptr<at::BFloat16>()),
        D, iters, eps);
}

template <typename wt_t, typename norm_t>
void launch_dsv4_hc_pre_norm_bf16(
    const torch::Tensor& x, const torch::Tensor& fn,
    const torch::Tensor& scale, const torch::Tensor& base,
    const torch::Tensor& norm, torch::Tensor& y,
    torch::Tensor& post, torch::Tensor& comb,
    int N, int D, int iters, float eps)
{
    dim3 block(32, 8);
    auto stream = at::cuda::getCurrentCUDAStream();
    dsv4_hc_pre_norm_bf16_kernel<wt_t, norm_t><<<N, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const wt_t*>(fn.data_ptr()),
        scale.data_ptr<float>(), base.data_ptr<float>(),
        reinterpret_cast<const norm_t*>(norm.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(post.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(comb.data_ptr<at::BFloat16>()),
        D, iters, eps);
}

std::vector<torch::Tensor> dsv4_hc_pre_norm(
    torch::Tensor x, torch::Tensor fn, torch::Tensor scale,
    torch::Tensor base, torch::Tensor norm, long iters, double eps)
{
    TORCH_CHECK(
        x.is_cuda() && fn.is_cuda() && scale.is_cuda() &&
        base.is_cuda() && norm.is_cuda(),
        "all tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(
        fn.scalar_type() == at::kFloat || fn.scalar_type() == at::kBFloat16,
        "fn must be float32 or bfloat16");
    TORCH_CHECK(
        norm.scalar_type() == at::kFloat || norm.scalar_type() == at::kBFloat16,
        "norm must be float32 or bfloat16");
    TORCH_CHECK(
        scale.scalar_type() == at::kFloat && base.scalar_type() == at::kFloat,
        "scale/base must be float32");
    TORCH_CHECK(x.dim() >= 2 && x.size(-2) == 4, "x must end in [4,D]");
    const int D = (int)x.size(-1);
    const int N = (int)(x.numel() / (4L * D));
    TORCH_CHECK(fn.numel() == 24L * 4L * D, "fn must be [24,4D]");
    TORCH_CHECK(norm.numel() == D, "norm must be [D]");
    TORCH_CHECK(scale.numel() == 3 && base.numel() == 24,
                "HC parameter size mismatch");

    auto xc = x.contiguous();
    auto fc = fn.contiguous();
    auto sc = scale.contiguous();
    auto bc = base.contiguous();
    auto nc = norm.contiguous();
    auto y = torch::empty({N, D}, x.options());
    auto post = torch::empty({N, 4}, x.options());
    auto comb = torch::empty({N, 16}, x.options());

    if (fn.scalar_type() == at::kBFloat16 &&
        norm.scalar_type() == at::kBFloat16 && D >= 48) {
        launch_dsv4_hc_pre_norm_bf16_parallel(
            xc, fc, sc, bc, nc, y, post, comb,
            N, D, (int)iters, (float)eps);
    } else if (fn.scalar_type() == at::kBFloat16 &&
               norm.scalar_type() == at::kBFloat16) {
        launch_dsv4_hc_pre_norm_bf16<__nv_bfloat16, __nv_bfloat16>(
            xc, fc, sc, bc, nc, y, post, comb,
            N, D, (int)iters, (float)eps);
    } else if (fn.scalar_type() == at::kBFloat16) {
        launch_dsv4_hc_pre_norm_bf16<__nv_bfloat16, float>(
            xc, fc, sc, bc, nc, y, post, comb, N, D, (int)iters, (float)eps);
    } else if (norm.scalar_type() == at::kBFloat16) {
        launch_dsv4_hc_pre_norm_bf16<float, __nv_bfloat16>(
            xc, fc, sc, bc, nc, y, post, comb, N, D, (int)iters, (float)eps);
    } else {
        launch_dsv4_hc_pre_norm_bf16<float, float>(
            xc, fc, sc, bc, nc, y, post, comb, N, D, (int)iters, (float)eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, post, comb};
}

std::vector<torch::Tensor> dsv4_hc_pre(
    torch::Tensor x, torch::Tensor fn, torch::Tensor scale, torch::Tensor base,
    long iters, double eps) {
    TORCH_CHECK(x.is_cuda() && fn.is_cuda() && scale.is_cuda() && base.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(fn.scalar_type() == at::kFloat || fn.scalar_type() == at::kBFloat16,
                "fn must be float32 or bfloat16");
    TORCH_CHECK(scale.scalar_type() == at::kFloat && base.scalar_type() == at::kFloat,
                "scale/base must be float32");
    TORCH_CHECK(x.dim() >= 2 && x.size(-2) == 4, "x must end in [4,D]");
    const int D = (int)x.size(-1);
    const int N = (int)(x.numel() / (4L * D));
    TORCH_CHECK(fn.numel() == 24L * 4L * D, "fn must be [24,4D]");
    TORCH_CHECK(scale.numel() == 3 && base.numel() == 24, "HC parameter size mismatch");
    auto xc = x.contiguous();
    auto fc = fn.contiguous();
    auto sc = scale.contiguous();
    auto bc = base.contiguous();
    auto y = torch::empty({N, D}, x.options());
    auto post = torch::empty({N, 4}, x.options());
    auto comb = torch::empty({N, 16}, x.options());
    dim3 block(32, 8);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (fn.scalar_type() == at::kFloat) {
        dsv4_hc_pre_kernel<float><<<N, block, 0, stream>>>(
            xc.data_ptr<float>(), fc.data_ptr<float>(), sc.data_ptr<float>(),
            bc.data_ptr<float>(), y.data_ptr<float>(), post.data_ptr<float>(),
            comb.data_ptr<float>(), D, (int)iters, (float)eps);
    } else {
        dsv4_hc_pre_kernel<__nv_bfloat16><<<N, block, 0, stream>>>(
            xc.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(fc.data_ptr<at::BFloat16>()),
            sc.data_ptr<float>(), bc.data_ptr<float>(), y.data_ptr<float>(),
            post.data_ptr<float>(), comb.data_ptr<float>(),
            D, (int)iters, (float)eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, post, comb};
}

__global__ void dsv4_route_post_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ bias,
    const bool* __restrict__ mask,
    float* __restrict__ weights,
    int64_t* __restrict__ indices,
    int experts,
    int top_k) {
    extern __shared__ float choices[];
    __shared__ float route_warp_values[32];
    __shared__ int route_warp_indices[32];
    __shared__ int route_selected[16];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        choices[expert] = mask[expert] ? scores[expert] + bias[expert]
                                       : -INFINITY;
    }
    __syncthreads();

    for (int rank = 0; rank < top_k; ++rank) {
        float best = -INFINITY;
        int best_expert = -1;
        for (int expert = tid; expert < experts; expert += blockDim.x) {
            const float value = choices[expert];
            if (best_expert < 0 || value > best ||
                (value == best && expert < best_expert)) {
                best = value;
                best_expert = expert;
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value =
                __shfl_down_sync(0xffffffffu, best, offset);
            const int other_expert =
                __shfl_down_sync(0xffffffffu, best_expert, offset);
            if (other_expert >= 0 &&
                (best_expert < 0 || other_value > best ||
                 (other_value == best && other_expert < best_expert))) {
                best = other_value;
                best_expert = other_expert;
            }
        }
        if (lane == 0) {
            route_warp_values[warp] = best;
            route_warp_indices[warp] = best_expert;
        }
        __syncthreads();

        if (warp == 0) {
            best = lane < warps ? route_warp_values[lane] : -INFINITY;
            best_expert = lane < warps ? route_warp_indices[lane] : -1;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_value =
                    __shfl_down_sync(0xffffffffu, best, offset);
                const int other_expert =
                    __shfl_down_sync(0xffffffffu, best_expert, offset);
                if (other_expert >= 0 &&
                    (best_expert < 0 || other_value > best ||
                     (other_value == best && other_expert < best_expert))) {
                    best = other_value;
                    best_expert = other_expert;
                }
            }
            if (lane == 0) {
                route_selected[rank] = best_expert;
                indices[rank] = best_expert;
                weights[rank] = scores[best_expert];
            }
        }
        __syncthreads();

        const int selected = route_selected[rank];
        for (int expert = tid; expert < experts; expert += blockDim.x) {
            if (expert == selected) {
                choices[expert] = -INFINITY;
            }
        }
        __syncthreads();
    }
}

std::vector<torch::Tensor> dsv4_route_post(
    torch::Tensor scores,
    torch::Tensor bias,
    torch::Tensor mask,
    long top_k) {
    TORCH_CHECK(scores.is_cuda() && bias.is_cuda() && mask.is_cuda(),
                "scores/bias/mask must be CUDA");
    TORCH_CHECK(scores.scalar_type() == at::kFloat &&
                bias.scalar_type() == at::kFloat,
                "scores/bias must be float32");
    TORCH_CHECK(mask.scalar_type() == at::kBool, "mask must be bool");
    TORCH_CHECK(scores.dim() == 2 && scores.size(0) == 1,
                "scores must be [1,E]");
    const int experts = (int)scores.size(1);
    TORCH_CHECK(bias.numel() == experts && mask.numel() == experts,
                "bias/mask size must match experts");
    TORCH_CHECK(experts > 0 && experts <= 1024,
                "experts must be in [1,1024]");
    TORCH_CHECK(top_k > 0 && top_k <= 16 && top_k <= experts,
                "top_k must be in [1,min(16,E)]");

    auto sc = scores.contiguous();
    auto bc = bias.contiguous();
    auto mc = mask.contiguous();
    auto weights = torch::empty({1, top_k}, scores.options());
    auto indices = torch::empty(
        {1, top_k}, scores.options().dtype(at::kLong));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = experts < 256 ? 128 : 256;
    dsv4_route_post_kernel<<<1, threads, experts * sizeof(float), stream>>>(
        sc.data_ptr<float>(),
        bc.data_ptr<float>(),
        mc.data_ptr<bool>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        experts,
        (int)top_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {weights, indices};
}

// ---- GLM sigmoid router + corrected Top-K + normalized route weights ----

__global__ void sigmoid_route_select_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const bool* __restrict__ mask,
    float* __restrict__ weights,
    int64_t* __restrict__ indices,
    int experts,
    int top_k,
    float routed_scaling) {
    extern __shared__ float choices[];
    __shared__ float route_warp_values[32];
    __shared__ int route_warp_indices[32];
    __shared__ int route_selected[16];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    for (int expert = tid; expert < experts; expert += blockDim.x) {
        const float probability =
            1.0f / (1.0f + expf(-logits[expert]));
        choices[expert] = mask[expert]
            ? probability + bias[expert]
            : -INFINITY;
    }
    __syncthreads();

    for (int rank = 0; rank < top_k; ++rank) {
        float best = -INFINITY;
        int best_expert = -1;
        for (int expert = tid; expert < experts; expert += blockDim.x) {
            const float value = choices[expert];
            if (
                best_expert < 0 ||
                value > best ||
                (value == best && expert < best_expert)
            ) {
                best = value;
                best_expert = expert;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value =
                __shfl_down_sync(0xffffffffu, best, offset);
            const int other_expert =
                __shfl_down_sync(0xffffffffu, best_expert, offset);
            if (
                other_expert >= 0 &&
                (
                    best_expert < 0 ||
                    other_value > best ||
                    (
                        other_value == best &&
                        other_expert < best_expert
                    )
                )
            ) {
                best = other_value;
                best_expert = other_expert;
            }
        }
        if (lane == 0) {
            route_warp_values[warp] = best;
            route_warp_indices[warp] = best_expert;
        }
        __syncthreads();
        if (warp == 0) {
            best = lane < warps
                ? route_warp_values[lane]
                : -INFINITY;
            best_expert = lane < warps
                ? route_warp_indices[lane]
                : -1;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_value =
                    __shfl_down_sync(0xffffffffu, best, offset);
                const int other_expert =
                    __shfl_down_sync(
                        0xffffffffu,
                        best_expert,
                        offset);
                if (
                    other_expert >= 0 &&
                    (
                        best_expert < 0 ||
                        other_value > best ||
                        (
                            other_value == best &&
                            other_expert < best_expert
                        )
                    )
                ) {
                    best = other_value;
                    best_expert = other_expert;
                }
            }
            if (lane == 0) {
                route_selected[rank] = best_expert;
                indices[rank] = best_expert;
                weights[rank] =
                    1.0f / (1.0f + expf(-logits[best_expert]));
            }
        }
        __syncthreads();
        const int selected = route_selected[rank];
        for (int expert = tid; expert < experts; expert += blockDim.x) {
            if (expert == selected) {
                choices[expert] = -INFINITY;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sum = 1.0e-20f;
        for (int rank = 0; rank < top_k; ++rank) {
            sum += weights[rank];
        }
        const float factor = routed_scaling / sum;
        for (int rank = 0; rank < top_k; ++rank) {
            weights[rank] *= factor;
        }
    }
}

__device__ __forceinline__ uint32_t route_ordered_float(float value)
{
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// One radix sort replaces Top-K rounds of block-wide reduction and removal.
// The 64-bit key preserves the reference ordering exactly: corrected score
// descending, then expert ID ascending for ties.
__global__ void sigmoid_route_radix_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const bool* __restrict__ mask,
    float* __restrict__ weights,
    int64_t* __restrict__ indices,
    const int experts,
    const int top_k,
    const float routed_scaling)
{
    constexpr int kThreads = 256;
    constexpr int kItems = 4;
    using Sort = cub::BlockRadixSort<
        unsigned long long,
        kThreads,
        kItems,
        int>;
    __shared__ typename Sort::TempStorage sort_storage;
    __shared__ int selected[16];
    unsigned long long keys[kItems];
    int values[kItems];
    #pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int expert = threadIdx.x * kItems + item;
        float choice = -INFINITY;
        if (expert < experts && mask[expert]) {
            const float probability =
                1.0f / (1.0f + expf(-logits[expert]));
            choice = probability + bias[expert];
        }
        const uint32_t score_key = route_ordered_float(choice);
        keys[item] =
            (static_cast<unsigned long long>(score_key) << 32) |
            static_cast<unsigned long long>(
                0xffffffffu - static_cast<uint32_t>(expert));
        values[item] = expert;
    }
    Sort(sort_storage).SortDescending(keys, values);
    #pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int rank = threadIdx.x * kItems + item;
        if (rank < top_k)
            selected[rank] = values[item];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 1.0e-20f;
        for (int rank = 0; rank < top_k; ++rank) {
            const int expert = selected[rank];
            const float probability =
                1.0f / (1.0f + expf(-logits[expert]));
            indices[rank] = expert;
            weights[rank] = probability;
            sum += probability;
        }
        const float factor = routed_scaling / sum;
        for (int rank = 0; rank < top_k; ++rank)
            weights[rank] *= factor;
    }
}

void launch_sigmoid_route(
    const float* logits,
    const float* bias,
    const bool* mask,
    float* weights,
    int64_t* indices,
    const int experts,
    const int top_k,
    const float routed_scaling,
    cudaStream_t stream)
{
    const char* radix_setting = std::getenv("TPQ_ROUTE_RADIX");
    const bool use_radix = (
        radix_setting == nullptr ||
        (radix_setting[0] == '1' && radix_setting[1] == '\0'));
    if (use_radix) {
        sigmoid_route_radix_kernel<<<1, 256, 0, stream>>>(
            logits,
            bias,
            mask,
            weights,
            indices,
            experts,
            top_k,
            routed_scaling);
    } else {
        const int threads = experts < 256 ? 128 : 256;
        sigmoid_route_select_kernel<<<
            1,
            threads,
            experts * sizeof(float),
            stream>>>(
                logits,
                bias,
                mask,
                weights,
                indices,
                experts,
                top_k,
                routed_scaling);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> sigmoid_route(
    torch::Tensor logits,
    torch::Tensor bias,
    torch::Tensor mask,
    long top_k,
    double routed_scaling) {
    TORCH_CHECK(
        logits.is_cuda() && bias.is_cuda() && mask.is_cuda(),
        "sigmoid router logits/bias/mask must be CUDA");
    TORCH_CHECK(
        logits.scalar_type() == at::kFloat &&
        bias.scalar_type() == at::kFloat &&
        mask.scalar_type() == at::kBool,
        "sigmoid router logits/bias/mask dtype mismatch");
    TORCH_CHECK(
        logits.dim() == 2 && logits.size(0) == 1,
        "sigmoid router logits must be [1,E]");
    const int experts = static_cast<int>(logits.size(1));
    TORCH_CHECK(
        bias.numel() == experts && mask.numel() == experts,
        "sigmoid router bias/mask size mismatch");
    TORCH_CHECK(
        experts > 0 && experts <= 1024,
        "sigmoid router experts must be in [1,1024]");
    TORCH_CHECK(
        top_k > 0 && top_k <= 16 && top_k <= experts,
        "sigmoid router top_k must be in [1,min(16,E)]");
    TORCH_CHECK(
        logits.get_device() == bias.get_device() &&
        logits.get_device() == mask.get_device(),
        "sigmoid router tensors must be on one device");

    auto lc = logits.contiguous();
    auto bc = bias.contiguous();
    auto mc = mask.contiguous();
    auto weights = torch::empty({1, top_k}, logits.options());
    auto indices = torch::empty(
        {1, top_k},
        logits.options().dtype(at::kLong));
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_sigmoid_route(
        lc.data_ptr<float>(),
        bc.data_ptr<float>(),
        mc.data_ptr<bool>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        experts,
        static_cast<int>(top_k),
        static_cast<float>(routed_scaling),
        stream);
    return {weights, indices};
}

std::vector<torch::Tensor> sigmoid_route_out(
    torch::Tensor logits,
    torch::Tensor bias,
    torch::Tensor mask,
    long top_k,
    double routed_scaling,
    torch::Tensor weights,
    torch::Tensor indices)
{
    TORCH_CHECK(
        logits.is_cuda() && bias.is_cuda() && mask.is_cuda() &&
        weights.is_cuda() && indices.is_cuda(),
        "sigmoid router tensors must be CUDA");
    TORCH_CHECK(
        logits.scalar_type() == at::kFloat &&
        bias.scalar_type() == at::kFloat &&
        mask.scalar_type() == at::kBool &&
        weights.scalar_type() == at::kFloat &&
        indices.scalar_type() == at::kLong,
        "sigmoid router output-buffer dtype mismatch");
    TORCH_CHECK(
        logits.is_contiguous() && bias.is_contiguous() &&
        mask.is_contiguous() && weights.is_contiguous() &&
        indices.is_contiguous() &&
        logits.dim() == 2 && logits.size(0) == 1,
        "sigmoid router output-buffer tensors must be contiguous");
    const int experts = static_cast<int>(logits.size(1));
    TORCH_CHECK(
        experts > 0 && experts <= 1024 &&
        bias.numel() == experts && mask.numel() == experts &&
        top_k > 0 && top_k <= 16 && top_k <= experts &&
        weights.sizes() == torch::IntArrayRef({1, top_k}) &&
        indices.sizes() == torch::IntArrayRef({1, top_k}),
        "sigmoid router output-buffer shapes do not match");
    const int device = logits.get_device();
    TORCH_CHECK(
        bias.get_device() == device &&
        mask.get_device() == device &&
        weights.get_device() == device &&
        indices.get_device() == device,
        "sigmoid router output-buffer tensors must share one device");
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_sigmoid_route(
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        mask.data_ptr<bool>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        experts,
        static_cast<int>(top_k),
        static_cast<float>(routed_scaling),
        stream);
    return {weights, indices};
}

template <typename input_t, int rows_per_block>
__global__ void linear_route_logits_kernel(
    const input_t* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ logits,
    const int experts,
    const int hidden)
{
    extern __shared__ float shared_input[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (
        int column = linear_thread;
        column < hidden;
        column += 32 * rows_per_block
    )
        shared_input[column] = vq_scalar_to_float(input + column);
    __syncthreads();
    const int expert =
        static_cast<int>(blockIdx.x) * rows_per_block + threadIdx.y;
    if (expert >= experts)
        return;
    const float* row =
        weight + static_cast<long>(expert) * hidden;
    float value = 0.0f;
    for (int column = lane; column < hidden; column += 32)
        value = __fmaf_rn(
            shared_input[column],
            __ldg(row + column),
            value);
    value = warp_sum_f32(value);
    if (lane == 0)
        logits[expert] = value;
}

std::vector<torch::Tensor> linear_sigmoid_route_out(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mask,
    long top_k,
    double routed_scaling,
    torch::Tensor logits,
    torch::Tensor weights,
    torch::Tensor indices)
{
    TORCH_CHECK(
        input.is_cuda() && weight.is_cuda() &&
        bias.is_cuda() && mask.is_cuda() &&
        logits.is_cuda() && weights.is_cuda() && indices.is_cuda(),
        "linear sigmoid router tensors must be CUDA");
    TORCH_CHECK(
        (
            input.scalar_type() == at::kBFloat16 ||
            input.scalar_type() == at::kFloat
        ) &&
        weight.scalar_type() == at::kFloat &&
        bias.scalar_type() == at::kFloat &&
        mask.scalar_type() == at::kBool &&
        logits.scalar_type() == at::kFloat &&
        weights.scalar_type() == at::kFloat &&
        indices.scalar_type() == at::kLong,
        "linear sigmoid router dtype mismatch");
    TORCH_CHECK(
        input.is_contiguous() && weight.is_contiguous() &&
        bias.is_contiguous() && mask.is_contiguous() &&
        logits.is_contiguous() && weights.is_contiguous() &&
        indices.is_contiguous() &&
        input.dim() == 2 && input.size(0) == 1 &&
        weight.dim() == 2,
        "linear sigmoid router tensors must be contiguous matrices");
    const int experts = static_cast<int>(weight.size(0));
    const int hidden = static_cast<int>(weight.size(1));
    TORCH_CHECK(
        input.size(1) == hidden &&
        experts > 0 && experts <= 1024 &&
        bias.numel() == experts && mask.numel() == experts &&
        logits.sizes() == torch::IntArrayRef({1, experts}) &&
        top_k > 0 && top_k <= 16 && top_k <= experts &&
        weights.sizes() == torch::IntArrayRef({1, top_k}) &&
        indices.sizes() == torch::IntArrayRef({1, top_k}),
        "linear sigmoid router shapes do not match");
    const int device = input.get_device();
    TORCH_CHECK(
        weight.get_device() == device &&
        bias.get_device() == device &&
        mask.get_device() == device &&
        logits.get_device() == device &&
        weights.get_device() == device &&
        indices.get_device() == device,
        "linear sigmoid router tensors must share one device");
    constexpr int rows_per_block = 32;
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(32, rows_per_block);
    const int grid = (experts + rows_per_block - 1) / rows_per_block;
    const size_t shared_bytes =
        static_cast<size_t>(hidden) * sizeof(float);
    if (input.scalar_type() == at::kBFloat16) {
        linear_route_logits_kernel<
            __nv_bfloat16,
            rows_per_block><<<grid, block, shared_bytes, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    input.data_ptr<at::BFloat16>()),
                weight.data_ptr<float>(),
                logits.data_ptr<float>(),
                experts,
                hidden);
    } else {
        linear_route_logits_kernel<
            float,
            rows_per_block><<<grid, block, shared_bytes, stream>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                logits.data_ptr<float>(),
                experts,
                hidden);
    }
    launch_sigmoid_route(
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        mask.data_ptr<bool>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        experts,
        static_cast<int>(top_k),
        static_cast<float>(routed_scaling),
        stream);
    return {weights, indices};
}

__global__ void paged_gather_bf16_kernel(
    const int64_t* __restrict__ page_ptrs,
    const int64_t* __restrict__ indices,
    __nv_bfloat16* __restrict__ output,
    int64_t items,
    int page_items,
    int dim) {
    const int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = items * static_cast<int64_t>(dim);
    if (linear >= total) {
        return;
    }
    const int64_t output_item = linear / dim;
    const int feature = static_cast<int>(linear % dim);
    const int64_t source_item = indices[output_item];
    const int64_t page_index = source_item / page_items;
    const int64_t page_offset = source_item % page_items;
    const auto* page = reinterpret_cast<const __nv_bfloat16*>(
        page_ptrs[page_index]);
    output[linear] = page[
        page_offset * static_cast<int64_t>(dim) + feature
    ];
}

torch::Tensor paged_gather_bf16(
    torch::Tensor page_ptrs,
    torch::Tensor indices,
    long page_items,
    long dim) {
    TORCH_CHECK(page_ptrs.is_cuda() && indices.is_cuda(),
                "page_ptrs and indices must be CUDA");
    TORCH_CHECK(page_ptrs.scalar_type() == at::kLong,
                "page_ptrs must be int64");
    TORCH_CHECK(indices.scalar_type() == at::kLong,
                "indices must be int64");
    TORCH_CHECK(page_ptrs.dim() == 1 && page_ptrs.numel() > 0,
                "page_ptrs must be a non-empty vector");
    TORCH_CHECK(page_items > 0 && dim > 0,
                "page_items and dim must be positive");

    auto pc = page_ptrs.contiguous();
    auto ic = indices.contiguous().view({-1});
    auto output = torch::empty(
        {ic.numel(), dim},
        ic.options().dtype(at::kBFloat16));
    if (ic.numel() == 0) {
        return output;
    }
    constexpr int threads = 256;
    const int64_t total = ic.numel() * dim;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    paged_gather_bf16_kernel<<<blocks, threads, 0, stream>>>(
        pc.data_ptr<int64_t>(),
        ic.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(
            output.data_ptr<at::BFloat16>()),
        ic.numel(),
        static_cast<int>(page_items),
        static_cast<int>(dim));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void hadamard_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int width,
    float scale) {
    extern __shared__ float values[];
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t base = static_cast<int64_t>(row) * width;
    if (lane < width) {
        values[lane] = __bfloat162float(input[base + lane]);
    }
    __syncthreads();

    for (int span = 1; span < width; span <<= 1) {
        if (lane < width / 2) {
            const int group = lane / span;
            const int offset = lane - group * span;
            const int left = group * (span << 1) + offset;
            const int right = left + span;
            const float a = values[left];
            const float b = values[right];
            values[left] = a + b;
            values[right] = a - b;
        }
        __syncthreads();
    }
    if (lane < width) {
        output[base + lane] = __float2bfloat16_rn(values[lane] * scale);
    }
}

torch::Tensor hadamard_bf16(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16,
                "input must be bfloat16");
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    auto x = input.contiguous();
    const int width = static_cast<int>(x.size(-1));
    TORCH_CHECK(
        width > 0 && width <= 256 && (width & (width - 1)) == 0,
        "last dimension must be a power of two up to 256");
    const int64_t rows = x.numel() / width;
    auto output = torch::empty_like(x);
    if (rows == 0) {
        return output.view(input.sizes());
    }
    const float scale = static_cast<float>(
        1.0 / std::sqrt(static_cast<double>(width)));
    auto stream = at::cuda::getCurrentCUDAStream();
    hadamard_bf16_kernel<<<
        static_cast<int>(rows),
        width,
        static_cast<size_t>(width) * sizeof(float),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            width,
            scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output.view(input.sizes());
}

// ---- Direct packed INT4-G64 GEMV for single-token decode ----
// Mirrors the USSR VQ kernel layout: one CTA stages the activation row once,
// then eight warps consume packed weights directly. No floating-point weight
// matrix is materialized. Each warp loads one G64 scale, broadcasts it, and
// unpacks one coalesced 32-byte group into registers.

constexpr int INT4_ROWS_PER_BLOCK = 32;

__global__ void int4_embedding_lookup_kernel(
    const uint8_t* __restrict__ packed_row,
    const __half* __restrict__ scale_row,
    float* __restrict__ output,
    int cols)
{
    for (
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        col < cols;
        col += blockDim.x * gridDim.x
    ) {
        const uint8_t code = __ldg(packed_row + col / 2);
        const int quantized = (col & 1)
            ? static_cast<int>(code >> 4) - 8
            : static_cast<int>(code & 15) - 8;
        const float scale = __half2float(
            __ldg(scale_row + col / 64));
        output[col] = __fmul_rn(
            static_cast<float>(quantized),
            scale);
    }
}

__global__ void int4_embedding_lookup_device_row_kernel(
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    const int64_t* __restrict__ row_ptr,
    float* __restrict__ output,
    int rows,
    int packed_cols,
    int scale_cols,
    int cols)
{
    const int64_t row = row_ptr[0];
    if (row < 0 || row >= rows)
        return;
    const uint8_t* packed_row =
        packed + row * packed_cols;
    const __half* scale_row =
        scales + row * scale_cols;
    for (
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        col < cols;
        col += blockDim.x * gridDim.x
    ) {
        const uint8_t code = __ldg(packed_row + col / 2);
        const int quantized = (col & 1)
            ? static_cast<int>(code >> 4) - 8
            : static_cast<int>(code & 15) - 8;
        const float scale = __half2float(
            __ldg(scale_row + col / 64));
        output[col] = __fmul_rn(
            static_cast<float>(quantized),
            scale);
    }
}

torch::Tensor int4_embedding_lookup(
    torch::Tensor packed,
    torch::Tensor scales,
    long row,
    long cols,
    long group_size,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        packed.is_cuda() && scales.is_cuda(),
        "INT4 embedding weights must be CUDA");
    TORCH_CHECK(
        packed.scalar_type() == at::kByte &&
        scales.scalar_type() == at::kHalf &&
        packed.is_contiguous() &&
        scales.is_contiguous() &&
        packed.dim() == 2 &&
        scales.dim() == 2,
        "INT4 embedding weights must be contiguous uint8/float16 matrices");
    TORCH_CHECK(
        group_size == 64 &&
        cols > 0 &&
        cols % 64 == 0 &&
        packed.size(1) * 2 == cols &&
        scales.size(0) == packed.size(0) &&
        scales.size(1) == cols / group_size &&
        row >= 0 &&
        row < packed.size(0),
        "INT4 embedding shape, group size or row is invalid");
    const int device = packed.get_device();
    TORCH_CHECK(
        scales.get_device() == device,
        "INT4 embedding tensors must share one device");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {1, cols},
            scales.options().dtype(at::kFloat));
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kFloat &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({1, cols}) &&
        output.get_device() == device,
        "INT4 embedding output must be contiguous float32 [1,cols]");
    const int threads = 256;
    const int blocks = std::min(
        32,
        (static_cast<int>(cols) + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    int4_embedding_lookup_kernel<<<
        blocks,
        threads,
        0,
        stream>>>(
            packed.data_ptr<uint8_t>() +
                row * packed.size(1),
            reinterpret_cast<const __half*>(
                scales.data_ptr<at::Half>()) +
                row * scales.size(1),
            output.data_ptr<float>(),
            static_cast<int>(cols));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor int4_embedding_lookup_device_row(
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor row,
    long cols,
    long group_size,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        packed.is_cuda() &&
        scales.is_cuda() &&
        row.is_cuda(),
        "device-row INT4 embedding inputs must be CUDA");
    TORCH_CHECK(
        packed.scalar_type() == at::kByte &&
        scales.scalar_type() == at::kHalf &&
        row.scalar_type() == at::kLong &&
        packed.is_contiguous() &&
        scales.is_contiguous() &&
        row.is_contiguous() &&
        packed.dim() == 2 &&
        scales.dim() == 2 &&
        row.numel() == 1,
        "device-row INT4 embedding input layouts do not match");
    TORCH_CHECK(
        group_size == 64 &&
        cols > 0 &&
        cols % 64 == 0 &&
        packed.size(1) * 2 == cols &&
        scales.size(0) == packed.size(0) &&
        scales.size(1) == cols / group_size,
        "device-row INT4 embedding shapes do not match");
    const int device = packed.get_device();
    TORCH_CHECK(
        scales.get_device() == device &&
        row.get_device() == device,
        "device-row INT4 embedding inputs must share one device");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {1, cols},
            scales.options().dtype(at::kFloat));
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kFloat &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({1, cols}) &&
        output.get_device() == device,
        "device-row INT4 embedding output must be float32 [1,cols]");
    const int threads = 256;
    const int blocks = std::min(
        32,
        (static_cast<int>(cols) + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    int4_embedding_lookup_device_row_kernel<<<
        blocks,
        threads,
        0,
        stream>>>(
            packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(
                scales.data_ptr<at::Half>()),
            row.data_ptr<int64_t>(),
            output.data_ptr<float>(),
            static_cast<int>(packed.size(0)),
            static_cast<int>(packed.size(1)),
            static_cast<int>(scales.size(1)),
            static_cast<int>(cols));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename input_t, int rows_per_block>
__global__ void int4_gemv_packed_f32_kernel(
    const input_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    float* __restrict__ output,
    int rows,
    int cols,
    int groups) {
    extern __shared__ float shared_x[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (int col = linear_thread;
         col < cols;
         col += 32 * rows_per_block) {
        shared_x[col] = vq_scalar_to_float(x + col);
    }
    __syncthreads();

    const int row =
        blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= rows) {
        return;
    }
    const int packed_cols = cols / 2;
    const uint8_t* qrow =
        packed + static_cast<int64_t>(row) * packed_cols;
    const __half* srow =
        scales + static_cast<int64_t>(row) * groups;
    float acc = 0.0f;
    for (int group = 0; group < groups; ++group) {
        float scale = lane == 0 ? __half2float(srow[group]) : 0.0f;
        scale = __shfl_sync(0xffffffffu, scale, 0);
        const int byte_index = group * 32 + lane;
        const uint8_t q = __ldg(qrow + byte_index);
        const int col = group * 64 + lane * 2;
        const float low = __fmul_rn(
            static_cast<float>((q & 15) - 8),
            scale);
        const float high = __fmul_rn(
            static_cast<float>((q >> 4) - 8),
            scale);
        acc = __fmaf_rn(low, shared_x[col], acc);
        acc = __fmaf_rn(high, shared_x[col + 1], acc);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        output[row] = acc;
    }
}

template <typename input_t, int rows_per_block>
__global__ void block_fp8_gemv_f32_kernel(
    const input_t* __restrict__ input,
    const uint8_t* __restrict__ weights,
    const float* __restrict__ scales,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const int scale_cols)
{
    extern __shared__ unsigned char fp8_shared_raw[];
    auto* fp8_shared_input =
        reinterpret_cast<__nv_bfloat16*>(fp8_shared_raw);
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (
        int column = linear_thread;
        column < cols;
        column += 32 * rows_per_block
    ) {
        fp8_shared_input[column] = __float2bfloat16_rn(
            vq_scalar_to_float(input + column));
    }
    __syncthreads();

    const int row =
        blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= rows)
        return;
    const auto* weight_row =
        weights + static_cast<long>(row) * cols;
    const auto* scale_row =
        scales + static_cast<long>(row / 128) * scale_cols;
    float accumulator = 0.f;
    for (
        int column_block = 0;
        column_block < scale_cols;
        ++column_block
    ) {
        float scale = lane == 0
            ? scale_row[column_block]
            : 0.f;
        scale = __shfl_sync(0xffffffffu, scale, 0);
        const float rounded_scale = __bfloat162float(
            __float2bfloat16_rn(scale));
        const int begin = column_block * 128;
        const int end = min(begin + 128, cols);
        const int column = begin + lane * 4;
        if (column + 3 < end) {
            __nv_fp8x4_e4m3 fp8_values;
            fp8_values.__x = __ldg(
                reinterpret_cast<const uint32_t*>(
                    weight_row + column));
            const float4 values =
                static_cast<float4>(fp8_values);
            const float rounded0 = __bfloat162float(
                __float2bfloat16_rn(values.x));
            const float rounded1 = __bfloat162float(
                __float2bfloat16_rn(values.y));
            const float rounded2 = __bfloat162float(
                __float2bfloat16_rn(values.z));
            const float rounded3 = __bfloat162float(
                __float2bfloat16_rn(values.w));
            const float scaled0 = __bfloat162float(
                __float2bfloat16_rn(rounded0 * rounded_scale));
            const float scaled1 = __bfloat162float(
                __float2bfloat16_rn(rounded1 * rounded_scale));
            const float scaled2 = __bfloat162float(
                __float2bfloat16_rn(rounded2 * rounded_scale));
            const float scaled3 = __bfloat162float(
                __float2bfloat16_rn(rounded3 * rounded_scale));
            accumulator = __fmaf_rn(
                scaled0,
                __bfloat162float(fp8_shared_input[column]),
                accumulator);
            accumulator = __fmaf_rn(
                scaled1,
                __bfloat162float(fp8_shared_input[column + 1]),
                accumulator);
            accumulator = __fmaf_rn(
                scaled2,
                __bfloat162float(fp8_shared_input[column + 2]),
                accumulator);
            accumulator = __fmaf_rn(
                scaled3,
                __bfloat162float(fp8_shared_input[column + 3]),
                accumulator);
        } else {
            for (int tail = column; tail < end; ++tail) {
                __nv_fp8_e4m3 fp8_value;
                fp8_value.__x = weight_row[tail];
                const float rounded_value = __bfloat162float(
                    __float2bfloat16_rn(
                        static_cast<float>(fp8_value)));
                const float scaled_value = __bfloat162float(
                    __float2bfloat16_rn(
                        rounded_value * rounded_scale));
                accumulator = __fmaf_rn(
                    scaled_value,
                    __bfloat162float(fp8_shared_input[tail]),
                    accumulator);
            }
        }
    }
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0)
        output[row] = accumulator;
}

// Four G64 groups are consumed per loop. Four 8-lane subgroups load their
// scales and one uint32 of packed codes per lane, while the final full-warp
// reduction still produces one output row. This keeps row-level occupancy
// and cuts loop/shuffle/address overhead for long reduction dimensions.
template <typename input_t, int rows_per_block>
__global__ void int4_gemv_packed_f32_vector4_kernel(
    const input_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    float* __restrict__ output,
    int rows,
    int cols,
    int groups)
{
    extern __shared__ float shared_x[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (
        int col = linear_thread;
        col < cols;
        col += 32 * rows_per_block
    )
        shared_x[col] = vq_scalar_to_float(x + col);
    __syncthreads();

    const int row =
        blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= rows)
        return;
    const int packed_cols = cols / 2;
    const uint8_t* packed_row =
        packed + static_cast<long>(row) * packed_cols;
    const __half* scale_row =
        scales + static_cast<long>(row) * groups;
    const int group_in_iteration = lane >> 3;
    const int group_lane = lane & 7;
    float accumulator = 0.f;
    for (int group_base = 0; group_base < groups; group_base += 4) {
        const int group = group_base + group_in_iteration;
        float scale = group_lane == 0
            ? __half2float(scale_row[group])
            : 0.f;
        scale = __shfl_sync(0xffffffffu, scale, 0, 8);
        const uint32_t codes = __ldg(
            reinterpret_cast<const uint32_t*>(
                packed_row + group * 32 + group_lane * 4));
        const int col_begin =
            group * 64 + group_lane * 8;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const uint8_t code =
                static_cast<uint8_t>(codes >> (item * 8));
            const int col = col_begin + item * 2;
            accumulator = __fmaf_rn(
                static_cast<float>((code & 15) - 8) * scale,
                shared_x[col],
                accumulator);
            accumulator = __fmaf_rn(
                static_cast<float>((code >> 4) - 8) * scale,
                shared_x[col + 1],
                accumulator);
        }
    }
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0)
        output[row] = accumulator;
}

template <int rows_per_block>
__global__ void int4_glm_qb_split_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    __nv_bfloat16* __restrict__ nope_output,
    float* __restrict__ rope_output,
    const int heads,
    const int nope_width,
    const int rope_width,
    const int cols,
    const int groups)
{
    extern __shared__ float shared_input[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (
        int col = linear_thread;
        col < cols;
        col += 32 * rows_per_block
    )
        shared_input[col] = input[col];
    __syncthreads();

    const int row =
        blockIdx.x * rows_per_block + threadIdx.y;
    const int head_width = nope_width + rope_width;
    const int rows = heads * head_width;
    if (row >= rows)
        return;
    const uint8_t* packed_row =
        packed + static_cast<long>(row) * (cols / 2);
    const __half* scale_row =
        scales + static_cast<long>(row) * groups;
    float accumulator = 0.f;
    for (int group = 0; group < groups; ++group) {
        float scale = lane == 0
            ? __half2float(scale_row[group])
            : 0.f;
        scale = __shfl_sync(0xffffffffu, scale, 0);
        const uint8_t code =
            __ldg(packed_row + group * 32 + lane);
        const int col = group * 64 + lane * 2;
        accumulator = __fmaf_rn(
            static_cast<float>((code & 15) - 8) * scale,
            shared_input[col],
            accumulator);
        accumulator = __fmaf_rn(
            static_cast<float>((code >> 4) - 8) * scale,
            shared_input[col + 1],
            accumulator);
    }
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0) {
        const int head = row / head_width;
        const int feature = row - head * head_width;
        if (feature < nope_width) {
            nope_output[
                static_cast<long>(head) * nope_width + feature
            ] = __float2bfloat16_rn(accumulator);
        } else {
            rope_output[
                static_cast<long>(head) * rope_width
                + feature - nope_width
            ] = accumulator;
        }
    }
}

template <int rows_per_block>
__global__ void int4_glm_qb_split_vector4_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    __nv_bfloat16* __restrict__ nope_output,
    float* __restrict__ rope_output,
    const int heads,
    const int nope_width,
    const int rope_width,
    const int cols,
    const int groups)
{
    extern __shared__ float shared_input[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (
        int col = linear_thread;
        col < cols;
        col += 32 * rows_per_block
    )
        shared_input[col] = input[col];
    __syncthreads();

    const int row =
        blockIdx.x * rows_per_block + threadIdx.y;
    const int head_width = nope_width + rope_width;
    const int rows = heads * head_width;
    if (row >= rows)
        return;
    const uint8_t* packed_row =
        packed + static_cast<long>(row) * (cols / 2);
    const __half* scale_row =
        scales + static_cast<long>(row) * groups;
    const int group_in_iteration = lane >> 3;
    const int group_lane = lane & 7;
    float accumulator = 0.f;
    for (int group_base = 0; group_base < groups; group_base += 4) {
        const int group = group_base + group_in_iteration;
        float scale = group_lane == 0
            ? __half2float(scale_row[group])
            : 0.f;
        scale = __shfl_sync(0xffffffffu, scale, 0, 8);
        const uint32_t codes = __ldg(
            reinterpret_cast<const uint32_t*>(
                packed_row + group * 32 + group_lane * 4));
        const int col_begin =
            group * 64 + group_lane * 8;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const uint8_t code =
                static_cast<uint8_t>(codes >> (item * 8));
            const int col = col_begin + item * 2;
            accumulator = __fmaf_rn(
                static_cast<float>((code & 15) - 8) * scale,
                shared_input[col],
                accumulator);
            accumulator = __fmaf_rn(
                static_cast<float>((code >> 4) - 8) * scale,
                shared_input[col + 1],
                accumulator);
        }
    }
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0) {
        const int head = row / head_width;
        const int feature = row - head * head_width;
        if (feature < nope_width) {
            nope_output[
                static_cast<long>(head) * nope_width + feature
            ] = __float2bfloat16_rn(accumulator);
        } else {
            rope_output[
                static_cast<long>(head) * rope_width
                + feature - nope_width
            ] = accumulator;
        }
    }
}

std::vector<torch::Tensor> int4_glm_qb_split(
    torch::Tensor input,
    torch::Tensor packed,
    torch::Tensor scales,
    long cols,
    long group_size,
    bool group_vector,
    long heads,
    long nope_width,
    long rope_width,
    c10::optional<torch::Tensor> nope_output_buffer,
    c10::optional<torch::Tensor> rope_output_buffer)
{
    TORCH_CHECK(
        input.is_cuda() && packed.is_cuda() && scales.is_cuda(),
        "GLM Q-B split tensors must be CUDA");
    TORCH_CHECK(
        input.scalar_type() == at::kFloat &&
        packed.scalar_type() == at::kByte &&
        scales.scalar_type() == at::kHalf &&
        input.is_contiguous() &&
        packed.is_contiguous() &&
        scales.is_contiguous() &&
        input.sizes() == torch::IntArrayRef({1, cols}) &&
        packed.dim() == 2 &&
        scales.dim() == 2,
        "GLM Q-B split input layouts do not match");
    TORCH_CHECK(
        group_size == 64 &&
        cols > 0 &&
        cols % group_size == 0 &&
        heads > 0 &&
        nope_width > 0 &&
        rope_width > 0 &&
        packed.size(0) == heads * (nope_width + rope_width) &&
        packed.size(1) * 2 == cols &&
        scales.sizes() == torch::IntArrayRef(
            {packed.size(0), cols / group_size}),
        "GLM Q-B split shapes do not match");
    const int device = input.get_device();
    TORCH_CHECK(
        packed.get_device() == device &&
        scales.get_device() == device,
        "GLM Q-B split tensors must share one device");
    auto nope_output = nope_output_buffer.has_value()
        ? nope_output_buffer.value()
        : torch::empty(
            {heads, 1, nope_width},
            input.options().dtype(at::kBFloat16));
    auto rope_output = rope_output_buffer.has_value()
        ? rope_output_buffer.value()
        : torch::empty(
            {heads, 1, rope_width},
            input.options());
    TORCH_CHECK(
        nope_output.is_cuda() &&
        nope_output.scalar_type() == at::kBFloat16 &&
        nope_output.is_contiguous() &&
        nope_output.sizes() == torch::IntArrayRef(
            {heads, 1, nope_width}) &&
        nope_output.get_device() == device,
        "GLM Q-B no-PE output must be contiguous BF16 [H,1,D]");
    TORCH_CHECK(
        rope_output.is_cuda() &&
        rope_output.scalar_type() == at::kFloat &&
        rope_output.is_contiguous() &&
        rope_output.sizes() == torch::IntArrayRef(
            {heads, 1, rope_width}) &&
        rope_output.get_device() == device,
        "GLM Q-B RoPE output must be contiguous FP32 [H,1,D]");
    constexpr int rows_per_block = 32;
    const int rows = static_cast<int>(packed.size(0));
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto grid = (rows + rows_per_block - 1) / rows_per_block;
    const auto block = dim3(32, rows_per_block);
    const auto shared = static_cast<int>(cols) * sizeof(float);
    if (group_vector) {
        TORCH_CHECK(
            (cols / group_size) % 4 == 0,
            "GLM Q-B vector path requires a multiple of four groups");
        int4_glm_qb_split_vector4_kernel<rows_per_block><<<
            grid,
            block,
            shared,
            stream>>>(
                input.data_ptr<float>(),
                packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(
                    scales.data_ptr<at::Half>()),
                reinterpret_cast<__nv_bfloat16*>(
                    nope_output.data_ptr<at::BFloat16>()),
                rope_output.data_ptr<float>(),
                static_cast<int>(heads),
                static_cast<int>(nope_width),
                static_cast<int>(rope_width),
                static_cast<int>(cols),
                static_cast<int>(cols / group_size));
    } else {
        int4_glm_qb_split_kernel<rows_per_block><<<
            grid,
            block,
            shared,
            stream>>>(
                input.data_ptr<float>(),
                packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(
                    scales.data_ptr<at::Half>()),
                reinterpret_cast<__nv_bfloat16*>(
                    nope_output.data_ptr<at::BFloat16>()),
                rope_output.data_ptr<float>(),
                static_cast<int>(heads),
                static_cast<int>(nope_width),
                static_cast<int>(rope_width),
                static_cast<int>(cols),
                static_cast<int>(cols / group_size));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {nope_output, rope_output};
}

// Decode-only GLM input RMSNorm plus the two projections that consume the
// same normalized hidden row (Q-A and KV-A). Every CTA reproduces the exact
// 256-thread RMS reduction into its local activation tile, then evaluates
// distinct output rows. This removes the global normalized row and avoids
// staging it once for each projection.
template <bool ADD_RESIDUAL, int ROWS_PER_CTA>
__global__ void glm_norm_qkv_int4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual_update,
    const float* __restrict__ norm_weight,
    const uint8_t* __restrict__ q_packed,
    const __half* __restrict__ q_scales,
    const uint8_t* __restrict__ kv_packed,
    const __half* __restrict__ kv_scales,
    float* __restrict__ q_output,
    float* __restrict__ kv_output,
    float* __restrict__ residual_output,
    const int q_rows,
    const int kv_rows,
    const int cols,
    const int groups,
    const float eps)
{
    extern __shared__ float shared_x[];
    __shared__ float reduction[32];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;

    float square_sum = 0.f;
    if (linear_thread < 256) {
        for (
            int col = linear_thread;
            col < cols;
            col += 256
        ) {
            const float value = ADD_RESIDUAL
                ? x[col] + residual_update[col]
                : x[col];
            square_sum += value * value;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            square_sum += __shfl_down_sync(
                0xffffffffu,
                square_sum,
                offset);
        if ((linear_thread & 31) == 0)
            reduction[linear_thread >> 5] = square_sum;
    }
    __syncthreads();
    if (linear_thread < 32) {
        float value = linear_thread < 8
            ? reduction[linear_thread]
            : 0.f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(
                0xffffffffu,
                value,
                offset);
        if (linear_thread == 0)
            reduction[0] = value;
    }
    __syncthreads();
    const float norm_scale = rsqrtf(
        reduction[0] / static_cast<float>(cols) + eps);
    for (
        int col = linear_thread;
        col < cols;
        col += 32 * ROWS_PER_CTA
    ) {
        const float value = ADD_RESIDUAL
            ? x[col] + residual_update[col]
            : x[col];
        shared_x[col] =
            norm_weight[col] * (value * norm_scale);
        if (ADD_RESIDUAL && blockIdx.x == 0)
            residual_output[col] = value;
    }
    __syncthreads();

    const int combined_row =
        blockIdx.x * ROWS_PER_CTA + threadIdx.y;
    if (combined_row >= q_rows + kv_rows)
        return;
    const bool is_q = combined_row < q_rows;
    const int row = is_q ? combined_row : combined_row - q_rows;
    const uint8_t* packed = is_q ? q_packed : kv_packed;
    const __half* scales = is_q ? q_scales : kv_scales;
    float* output = is_q ? q_output : kv_output;
    const int packed_cols = cols / 2;
    const uint8_t* packed_row =
        packed + static_cast<long>(row) * packed_cols;
    const __half* scale_row =
        scales + static_cast<long>(row) * groups;
    float accumulator = 0.f;
    for (int group = 0; group < groups; ++group) {
        float scale = lane == 0
            ? __half2float(scale_row[group])
            : 0.f;
        scale = __shfl_sync(0xffffffffu, scale, 0);
        const uint8_t code =
            __ldg(packed_row + group * 32 + lane);
        const int col = group * 64 + lane * 2;
        accumulator = __fmaf_rn(
            static_cast<float>((code & 15) - 8) * scale,
            shared_x[col],
            accumulator);
        accumulator = __fmaf_rn(
            static_cast<float>((code >> 4) - 8) * scale,
            shared_x[col + 1],
            accumulator);
    }
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0)
        output[row] = accumulator;
}

// Decode-only residual add + post-attention RMSNorm + router projection.
// Eight router rows share one normalized hidden tile per CTA.  CTA 0 also
// materializes the residual and normalized rows needed by the expert MLP.
__global__ void glm_residual_norm_router_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ update,
    const float* __restrict__ norm_weight,
    const float* __restrict__ router_weight,
    float* __restrict__ residual_output,
    float* __restrict__ norm_output,
    float* __restrict__ logits_output,
    const int rows,
    const int cols,
    const float eps)
{
    extern __shared__ float shared_norm[];
    __shared__ float reduction[32];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;

    float square_sum = 0.f;
    for (
        int col = linear_thread;
        col < cols;
        col += 256
    ) {
        const float value = residual[col] + update[col];
        square_sum += value * value;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        square_sum += __shfl_down_sync(
            0xffffffffu,
            square_sum,
            offset);
    if ((linear_thread & 31) == 0)
        reduction[linear_thread >> 5] = square_sum;
    __syncthreads();
    if (linear_thread < 32) {
        float value = linear_thread < 8
            ? reduction[linear_thread]
            : 0.f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(
                0xffffffffu,
                value,
                offset);
        if (linear_thread == 0)
            reduction[0] = value;
    }
    __syncthreads();
    const float norm_scale = rsqrtf(
        reduction[0] / static_cast<float>(cols) + eps);
    for (
        int col = linear_thread;
        col < cols;
        col += 256
    ) {
        const float value = residual[col] + update[col];
        const float normalized =
            norm_weight[col] * (value * norm_scale);
        shared_norm[col] = normalized;
        if (blockIdx.x == 0) {
            residual_output[col] = value;
            norm_output[col] = normalized;
        }
    }
    __syncthreads();

    const int row = blockIdx.x * 8 + threadIdx.y;
    if (row >= rows)
        return;
    const float* weight =
        router_weight + static_cast<long>(row) * cols;
    float accumulator = 0.f;
    for (int col = lane; col < cols; col += 32)
        accumulator = __fmaf_rn(
            shared_norm[col],
            weight[col],
            accumulator);
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0)
        logits_output[row] = accumulator;
}

__global__ void glm_moe_residual_add_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ routed,
    const float* __restrict__ shared,
    float* __restrict__ output,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    ) {
        const float expert_sum = __fadd_rn(
            routed[index],
            shared[index]);
        output[index] = __fadd_rn(
            residual[index],
            expert_sum);
    }
}

__global__ void residual_add3_bf16_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ routed,
    const __nv_bfloat16* __restrict__ shared,
    __nv_bfloat16* __restrict__ output,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    ) {
        const __nv_bfloat16 expert_sum = __float2bfloat16_rn(
            __bfloat162float(routed[index])
            + __bfloat162float(shared[index]));
        output[index] = __float2bfloat16_rn(
            __bfloat162float(residual[index])
            + __bfloat162float(expert_sum));
    }
}

__global__ void glm_ep_reduce_residual_kernel(
    float* __restrict__ primary_partial,
    const float* __restrict__ partial_1,
    const float* __restrict__ partial_2,
    const float* __restrict__ partial_3,
    const float* __restrict__ partial_4,
    const float* __restrict__ partial_5,
    const float* __restrict__ partial_6,
    const float* __restrict__ partial_7,
    const float* __restrict__ partial_8,
    const float* __restrict__ partial_9,
    const float* __restrict__ partial_10,
    const float* __restrict__ partial_11,
    const float* __restrict__ partial_12,
    const float* __restrict__ partial_13,
    const float* __restrict__ partial_14,
    const float* __restrict__ partial_15,
    const float* __restrict__ residual,
    const int contribution_count,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    ) {
        float routed = primary_partial[index];
        if (contribution_count > 1)
            routed = __fadd_rn(routed, partial_1[index]);
        if (contribution_count > 2)
            routed = __fadd_rn(routed, partial_2[index]);
        if (contribution_count > 3)
            routed = __fadd_rn(routed, partial_3[index]);
        if (contribution_count > 4)
            routed = __fadd_rn(routed, partial_4[index]);
        if (contribution_count > 5)
            routed = __fadd_rn(routed, partial_5[index]);
        if (contribution_count > 6)
            routed = __fadd_rn(routed, partial_6[index]);
        if (contribution_count > 7)
            routed = __fadd_rn(routed, partial_7[index]);
        if (contribution_count > 8)
            routed = __fadd_rn(routed, partial_8[index]);
        if (contribution_count > 9)
            routed = __fadd_rn(routed, partial_9[index]);
        if (contribution_count > 10)
            routed = __fadd_rn(routed, partial_10[index]);
        if (contribution_count > 11)
            routed = __fadd_rn(routed, partial_11[index]);
        if (contribution_count > 12)
            routed = __fadd_rn(routed, partial_12[index]);
        if (contribution_count > 13)
            routed = __fadd_rn(routed, partial_13[index]);
        if (contribution_count > 14)
            routed = __fadd_rn(routed, partial_14[index]);
        if (contribution_count > 15)
            routed = __fadd_rn(routed, partial_15[index]);
        primary_partial[index] = __fadd_rn(
            residual[index],
            routed);
    }
}

template <typename output_t>
__global__ void tp_all_rank_reduce_kernel(
    output_t* __restrict__ output,
    const float* __restrict__ partial_0,
    const float* __restrict__ partial_1,
    const float* __restrict__ partial_2,
    const float* __restrict__ partial_3,
    const float* __restrict__ partial_4,
    const float* __restrict__ partial_5,
    const float* __restrict__ partial_6,
    const float* __restrict__ partial_7,
    const float* __restrict__ partial_8,
    const float* __restrict__ partial_9,
    const float* __restrict__ partial_10,
    const float* __restrict__ partial_11,
    const float* __restrict__ partial_12,
    const float* __restrict__ partial_13,
    const float* __restrict__ partial_14,
    const float* __restrict__ partial_15,
    const int contribution_count,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    ) {
        float value = partial_0[index];
        if (contribution_count > 1)
            value = __fadd_rn(value, partial_1[index]);
        if (contribution_count > 2)
            value = __fadd_rn(value, partial_2[index]);
        if (contribution_count > 3)
            value = __fadd_rn(value, partial_3[index]);
        if (contribution_count > 4)
            value = __fadd_rn(value, partial_4[index]);
        if (contribution_count > 5)
            value = __fadd_rn(value, partial_5[index]);
        if (contribution_count > 6)
            value = __fadd_rn(value, partial_6[index]);
        if (contribution_count > 7)
            value = __fadd_rn(value, partial_7[index]);
        if (contribution_count > 8)
            value = __fadd_rn(value, partial_8[index]);
        if (contribution_count > 9)
            value = __fadd_rn(value, partial_9[index]);
        if (contribution_count > 10)
            value = __fadd_rn(value, partial_10[index]);
        if (contribution_count > 11)
            value = __fadd_rn(value, partial_11[index]);
        if (contribution_count > 12)
            value = __fadd_rn(value, partial_12[index]);
        if (contribution_count > 13)
            value = __fadd_rn(value, partial_13[index]);
        if (contribution_count > 14)
            value = __fadd_rn(value, partial_14[index]);
        if (contribution_count > 15)
            value = __fadd_rn(value, partial_15[index]);
        if constexpr (std::is_same_v<output_t, float>)
            output[index] = value;
        else
            output[index] = __float2bfloat16_rn(value);
    }
}

__global__ void tp_moe_finalize_all_rank_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ routed_0,
    const float* __restrict__ routed_1,
    const float* __restrict__ routed_2,
    const float* __restrict__ routed_3,
    const float* __restrict__ routed_4,
    const float* __restrict__ routed_5,
    const float* __restrict__ routed_6,
    const float* __restrict__ routed_7,
    const float* __restrict__ routed_8,
    const float* __restrict__ routed_9,
    const float* __restrict__ routed_10,
    const float* __restrict__ routed_11,
    const float* __restrict__ routed_12,
    const float* __restrict__ routed_13,
    const float* __restrict__ routed_14,
    const float* __restrict__ routed_15,
    const float* __restrict__ shared_0,
    const float* __restrict__ shared_1,
    const float* __restrict__ shared_2,
    const float* __restrict__ shared_3,
    const float* __restrict__ shared_4,
    const float* __restrict__ shared_5,
    const float* __restrict__ shared_6,
    const float* __restrict__ shared_7,
    const float* __restrict__ shared_8,
    const float* __restrict__ shared_9,
    const float* __restrict__ shared_10,
    const float* __restrict__ shared_11,
    const float* __restrict__ shared_12,
    const float* __restrict__ shared_13,
    const float* __restrict__ shared_14,
    const float* __restrict__ shared_15,
    const int contribution_count,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    ) {
        float routed = routed_0[index];
        float shared = shared_0[index];
#define TPQ_ACCUMULATE_MOE_RANK(rank) \
        if (contribution_count > rank) { \
            routed = __fadd_rn(routed, routed_##rank[index]); \
            shared = __fadd_rn(shared, shared_##rank[index]); \
        }
        TPQ_ACCUMULATE_MOE_RANK(1)
        TPQ_ACCUMULATE_MOE_RANK(2)
        TPQ_ACCUMULATE_MOE_RANK(3)
        TPQ_ACCUMULATE_MOE_RANK(4)
        TPQ_ACCUMULATE_MOE_RANK(5)
        TPQ_ACCUMULATE_MOE_RANK(6)
        TPQ_ACCUMULATE_MOE_RANK(7)
        TPQ_ACCUMULATE_MOE_RANK(8)
        TPQ_ACCUMULATE_MOE_RANK(9)
        TPQ_ACCUMULATE_MOE_RANK(10)
        TPQ_ACCUMULATE_MOE_RANK(11)
        TPQ_ACCUMULATE_MOE_RANK(12)
        TPQ_ACCUMULATE_MOE_RANK(13)
        TPQ_ACCUMULATE_MOE_RANK(14)
        TPQ_ACCUMULATE_MOE_RANK(15)
#undef TPQ_ACCUMULATE_MOE_RANK
        // Preserve the exact three-kernel rounding contract: both reductions
        // first materialize BF16, then routed+shared and residual are rounded
        // independently.  Only the intermediate global-memory traffic and
        // launch boundaries disappear.
        const __nv_bfloat16 routed_bf16 =
            __float2bfloat16_rn(routed);
        const __nv_bfloat16 shared_bf16 =
            __float2bfloat16_rn(shared);
        const __nv_bfloat16 expert_sum = __float2bfloat16_rn(
            __bfloat162float(routed_bf16)
            + __bfloat162float(shared_bf16));
        output[index] = __float2bfloat16_rn(
            __bfloat162float(residual[index])
            + __bfloat162float(expert_sum));
    }
}

template <typename input_t>
__global__ void int4_swiglu_packed_f32_kernel(
    const input_t* __restrict__ x,
    const uint8_t* __restrict__ gate_packed,
    const __half* __restrict__ gate_scales,
    const uint8_t* __restrict__ up_packed,
    const __half* __restrict__ up_scales,
    float* __restrict__ output,
    int rows,
    int cols,
    int groups) {
    extern __shared__ float shared_x[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (int col = linear_thread;
         col < cols;
         col += 32 * INT4_ROWS_PER_BLOCK) {
        shared_x[col] = vq_scalar_to_float(x + col);
    }
    __syncthreads();

    const int row =
        blockIdx.x * INT4_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const int packed_cols = cols / 2;
    const int64_t packed_base =
        static_cast<int64_t>(row) * packed_cols;
    const int64_t scale_base =
        static_cast<int64_t>(row) * groups;
    float gate = 0.0f;
    float up = 0.0f;
    for (int group = 0; group < groups; ++group) {
        float gate_scale = lane == 0
            ? __half2float(gate_scales[scale_base + group])
            : 0.0f;
        float up_scale = lane == 0
            ? __half2float(up_scales[scale_base + group])
            : 0.0f;
        gate_scale = __shfl_sync(
            0xffffffffu, gate_scale, 0);
        up_scale = __shfl_sync(
            0xffffffffu, up_scale, 0);
        const int64_t byte_index =
            packed_base + group * 32 + lane;
        const uint8_t gate_q = __ldg(gate_packed + byte_index);
        const uint8_t up_q = __ldg(up_packed + byte_index);
        const int col = group * 64 + lane * 2;
        const float x0 = shared_x[col];
        const float x1 = shared_x[col + 1];
        gate = __fmaf_rn(
            static_cast<float>((gate_q & 15) - 8) * gate_scale,
            x0,
            gate);
        gate = __fmaf_rn(
            static_cast<float>((gate_q >> 4) - 8) * gate_scale,
            x1,
            gate);
        up = __fmaf_rn(
            static_cast<float>((up_q & 15) - 8) * up_scale,
            x0,
            up);
        up = __fmaf_rn(
            static_cast<float>((up_q >> 4) - 8) * up_scale,
            x1,
            up);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0) {
        const float silu = gate / (1.0f + expf(-gate));
        output[row] = silu * up;
    }
}

template <typename input_t>
__global__ void int4_swiglu_packed_f32_vector4_kernel(
    const input_t* __restrict__ x,
    const uint8_t* __restrict__ gate_packed,
    const __half* __restrict__ gate_scales,
    const uint8_t* __restrict__ up_packed,
    const __half* __restrict__ up_scales,
    float* __restrict__ output,
    int rows,
    int cols,
    int groups)
{
    extern __shared__ float shared_x[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (
        int col = linear_thread;
        col < cols;
        col += 32 * INT4_ROWS_PER_BLOCK
    )
        shared_x[col] = vq_scalar_to_float(x + col);
    __syncthreads();

    const int row =
        blockIdx.x * INT4_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows)
        return;
    const int packed_cols = cols / 2;
    const long packed_base =
        static_cast<long>(row) * packed_cols;
    const long scale_base =
        static_cast<long>(row) * groups;
    const int group_in_iteration = lane >> 3;
    const int group_lane = lane & 7;
    float gate = 0.f;
    float up = 0.f;
    for (int group_base = 0; group_base < groups; group_base += 4) {
        const int group = group_base + group_in_iteration;
        float gate_scale = group_lane == 0
            ? __half2float(gate_scales[scale_base + group])
            : 0.f;
        float up_scale = group_lane == 0
            ? __half2float(up_scales[scale_base + group])
            : 0.f;
        gate_scale = __shfl_sync(
            0xffffffffu,
            gate_scale,
            0,
            8);
        up_scale = __shfl_sync(
            0xffffffffu,
            up_scale,
            0,
            8);
        const long byte_index =
            packed_base + group * 32 + group_lane * 4;
        const uint32_t gate_codes = __ldg(
            reinterpret_cast<const uint32_t*>(
                gate_packed + byte_index));
        const uint32_t up_codes = __ldg(
            reinterpret_cast<const uint32_t*>(
                up_packed + byte_index));
        const int col_begin =
            group * 64 + group_lane * 8;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const uint8_t gate_code =
                static_cast<uint8_t>(
                    gate_codes >> (item * 8));
            const uint8_t up_code =
                static_cast<uint8_t>(
                    up_codes >> (item * 8));
            const int col = col_begin + item * 2;
            const float x0 = shared_x[col];
            const float x1 = shared_x[col + 1];
            gate = __fmaf_rn(
                static_cast<float>((gate_code & 15) - 8) *
                    gate_scale,
                x0,
                gate);
            gate = __fmaf_rn(
                static_cast<float>((gate_code >> 4) - 8) *
                    gate_scale,
                x1,
                gate);
            up = __fmaf_rn(
                static_cast<float>((up_code & 15) - 8) *
                    up_scale,
                x0,
                up);
            up = __fmaf_rn(
                static_cast<float>((up_code >> 4) - 8) *
                    up_scale,
                x1,
                up);
        }
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0) {
        const float silu = gate / (1.f + expf(-gate));
        output[row] = silu * up;
    }
}

template <typename input_t, int rows_per_block>
void launch_int4_gemv_packed_f32_rows(
    const input_t* x,
    const uint8_t* packed,
    const __half* scales,
    float* output,
    int rows,
    int cols,
    int groups,
    int device,
    cudaStream_t stream) {
    dim3 block(32, rows_per_block);
    const int blocks =
        (rows + rows_per_block - 1) /
        rows_per_block;
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(float);
    constexpr int tracked_devices = 32;
    static size_t configured_shared_bytes[tracked_devices] = {};
    TORCH_CHECK(
        device >= 0 && device < tracked_devices,
        "INT4 GEMV device index is out of tracked range");
    if (configured_shared_bytes[device] < shared_bytes) {
        int optin_limit = 0;
        const auto query_status = cudaDeviceGetAttribute(
            &optin_limit,
            cudaDevAttrMaxSharedMemoryPerBlockOptin,
            device);
        TORCH_CHECK(
            query_status == cudaSuccess,
            "failed to query opt-in shared memory: ",
            cudaGetErrorString(query_status));
        TORCH_CHECK(
            shared_bytes <= static_cast<size_t>(optin_limit),
            "INT4 GEMV activation row needs ",
            shared_bytes,
            " bytes shared memory, device limit is ",
            optin_limit);
        const auto attr_status = cudaFuncSetAttribute(
            int4_gemv_packed_f32_kernel<
                input_t,
                rows_per_block>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            attr_status == cudaSuccess,
            "failed to configure INT4 GEMV shared memory: ",
            cudaGetErrorString(attr_status));
        configured_shared_bytes[device] = shared_bytes;
    }
    int4_gemv_packed_f32_kernel<input_t, rows_per_block>
        <<<blocks, block, shared_bytes, stream>>>(
            x,
            packed,
            scales,
            output,
            rows,
            cols,
            groups);
}

template <typename input_t, int rows_per_block>
void launch_int4_gemv_packed_f32_vector4_rows(
    const input_t* x,
    const uint8_t* packed,
    const __half* scales,
    float* output,
    int rows,
    int cols,
    int groups,
    int device,
    cudaStream_t stream)
{
    dim3 block(32, rows_per_block);
    const int blocks =
        (rows + rows_per_block - 1) / rows_per_block;
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(float);
    constexpr int tracked_devices = 32;
    static size_t configured_shared_bytes[tracked_devices] = {};
    TORCH_CHECK(
        device >= 0 && device < tracked_devices,
        "INT4 vector GEMV device index is out of tracked range");
    if (configured_shared_bytes[device] < shared_bytes) {
        int optin_limit = 0;
        const auto query_status = cudaDeviceGetAttribute(
            &optin_limit,
            cudaDevAttrMaxSharedMemoryPerBlockOptin,
            device);
        TORCH_CHECK(
            query_status == cudaSuccess &&
            shared_bytes <= static_cast<size_t>(optin_limit),
            "INT4 vector GEMV shared-memory requirement is unsupported");
        const auto attr_status = cudaFuncSetAttribute(
            int4_gemv_packed_f32_vector4_kernel<
                input_t,
                rows_per_block>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            attr_status == cudaSuccess,
            "failed to configure INT4 vector GEMV shared memory: ",
            cudaGetErrorString(attr_status));
        configured_shared_bytes[device] = shared_bytes;
    }
    int4_gemv_packed_f32_vector4_kernel<
        input_t,
        rows_per_block><<<
            blocks,
            block,
            shared_bytes,
            stream>>>(
                x,
                packed,
                scales,
                output,
                rows,
                cols,
                groups);
}

template <typename input_t>
void launch_int4_gemv_packed_f32(
    const input_t* x,
    const uint8_t* packed,
    const __half* scales,
    float* output,
    int rows,
    int cols,
    int groups,
    int device,
    cudaStream_t stream,
    bool group_vector)
{
    if (group_vector && groups % 4 == 0) {
        if (rows <= 2048) {
            launch_int4_gemv_packed_f32_vector4_rows<input_t, 8>(
                x, packed, scales, output, rows, cols, groups,
                device, stream);
        } else if (
            (rows == 6144 && cols == 16384) ||
            (rows <= 6144 && cols <= 2048)
        ) {
            launch_int4_gemv_packed_f32_vector4_rows<input_t, 16>(
                x, packed, scales, output, rows, cols, groups,
                device, stream);
        } else {
            launch_int4_gemv_packed_f32_vector4_rows<input_t, 32>(
                x, packed, scales, output, rows, cols, groups,
                device, stream);
        }
        return;
    }
    if (rows <= 2048) {
        launch_int4_gemv_packed_f32_rows<input_t, 8>(
            x, packed, scales, output, rows, cols, groups,
            device, stream);
    } else if (
        (rows == 6144 && cols == 16384) ||
        (rows <= 6144 && cols <= 2048)
    ) {
        launch_int4_gemv_packed_f32_rows<input_t, 16>(
            x, packed, scales, output, rows, cols, groups,
            device, stream);
    } else {
        launch_int4_gemv_packed_f32_rows<input_t, 32>(
            x, packed, scales, output, rows, cols, groups,
            device, stream);
    }
}

template <typename input_t>
void launch_int4_swiglu_packed_f32(
    const input_t* x,
    const uint8_t* gate_packed,
    const __half* gate_scales,
    const uint8_t* up_packed,
    const __half* up_scales,
    float* output,
    int rows,
    int cols,
    int groups,
    int device,
    cudaStream_t stream,
    bool group_vector) {
    dim3 block(32, INT4_ROWS_PER_BLOCK);
    const int blocks =
        (rows + INT4_ROWS_PER_BLOCK - 1) /
        INT4_ROWS_PER_BLOCK;
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(float);
    constexpr int tracked_devices = 32;
    static size_t configured_shared_bytes[tracked_devices] = {};
    TORCH_CHECK(
        device >= 0 && device < tracked_devices,
        "INT4 SwiGLU device index is out of tracked range");
    if (configured_shared_bytes[device] < shared_bytes) {
        const auto attr_status = cudaFuncSetAttribute(
            int4_swiglu_packed_f32_kernel<input_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            attr_status == cudaSuccess,
            "failed to configure INT4 SwiGLU shared memory: ",
            cudaGetErrorString(attr_status));
        configured_shared_bytes[device] = shared_bytes;
    }
    if (group_vector && groups % 4 == 0) {
        constexpr int vector_tracked_devices = 32;
        static size_t vector_shared_bytes[
            vector_tracked_devices
        ] = {};
        if (vector_shared_bytes[device] < shared_bytes) {
            const auto vector_attr_status = cudaFuncSetAttribute(
                int4_swiglu_packed_f32_vector4_kernel<input_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared_bytes));
            TORCH_CHECK(
                vector_attr_status == cudaSuccess,
                "failed to configure vector INT4 SwiGLU shared memory: ",
                cudaGetErrorString(vector_attr_status));
            vector_shared_bytes[device] = shared_bytes;
        }
        int4_swiglu_packed_f32_vector4_kernel<input_t>
            <<<blocks, block, shared_bytes, stream>>>(
                x,
                gate_packed,
                gate_scales,
                up_packed,
                up_scales,
                output,
                rows,
                cols,
                groups);
    } else {
        int4_swiglu_packed_f32_kernel<input_t>
            <<<blocks, block, shared_bytes, stream>>>(
                x,
                gate_packed,
                gate_scales,
                up_packed,
                up_scales,
                output,
                rows,
                cols,
                groups);
    }
}

torch::Tensor int4_gemv_packed_f32(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor scales,
    long cols,
    long group_size,
    bool group_vector,
    c10::optional<torch::Tensor> output_buffer) {
    TORCH_CHECK(
        x.is_cuda() && packed.is_cuda() && scales.is_cuda(),
        "INT4 GEMV tensors must be CUDA");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat ||
        x.scalar_type() == at::kBFloat16,
        "INT4 GEMV input must be float32 or bfloat16");
    TORCH_CHECK(
        packed.scalar_type() == at::kByte,
        "packed INT4 weights must be uint8");
    TORCH_CHECK(
        scales.scalar_type() == at::kHalf,
        "INT4 scales must be float16");
    TORCH_CHECK(
        x.dim() == 2 && x.size(0) == 1,
        "INT4 GEMV input must be [1,C]");
    TORCH_CHECK(
        packed.dim() == 2 && scales.dim() == 2,
        "INT4 weights and scales must be matrices");
    TORCH_CHECK(
        group_size == 64,
        "direct INT4 GEMV currently requires group size 64");
    TORCH_CHECK(
        cols > 0 && cols % 64 == 0,
        "INT4 columns must be a positive multiple of 64");
    TORCH_CHECK(
        x.size(1) == cols && packed.size(1) * 2 == cols,
        "INT4 GEMV input/weight column mismatch");
    const int rows = static_cast<int>(packed.size(0));
    const int groups = static_cast<int>(cols / group_size);
    TORCH_CHECK(
        scales.size(0) == rows && scales.size(1) == groups,
        "INT4 scale shape mismatch");

    auto xc = x.contiguous();
    auto qc = packed.contiguous();
    auto sc = scales.contiguous();
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {1, rows},
            x.options().dtype(at::kFloat));
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kFloat &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({1, rows}) &&
        output.get_device() == packed.get_device(),
        "INT4 GEMV output buffer must be contiguous float32 [1,R] "
        "on the weight device");
    const int device = packed.get_device();
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kFloat) {
        launch_int4_gemv_packed_f32<float>(
            xc.data_ptr<float>(),
            qc.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()),
            output.data_ptr<float>(),
            rows,
            static_cast<int>(cols),
            groups,
            device,
            stream,
            group_vector);
    } else {
        launch_int4_gemv_packed_f32<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(
                xc.data_ptr<at::BFloat16>()),
            qc.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()),
            output.data_ptr<float>(),
            rows,
            static_cast<int>(cols),
            groups,
            device,
            stream,
            group_vector);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename input_t>
void launch_block_fp8_gemv_f32(
    const input_t* input,
    const uint8_t* weights,
    const float* scales,
    float* output,
    const int rows,
    const int cols,
    const int scale_cols,
    cudaStream_t stream)
{
    constexpr int rows_per_block = 32;
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(__nv_bfloat16);
    if (shared_bytes > 48 * 1024) {
        const auto status = cudaFuncSetAttribute(
            block_fp8_gemv_f32_kernel<input_t, rows_per_block>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            status == cudaSuccess,
            "failed to configure block-FP8 GEMV shared memory: ",
            cudaGetErrorString(status));
    }
    block_fp8_gemv_f32_kernel<
        input_t,
        rows_per_block><<<
            (rows + rows_per_block - 1) / rows_per_block,
            dim3(32, rows_per_block),
            shared_bytes,
            stream>>>(
                input,
                weights,
                scales,
                output,
                rows,
                cols,
                scale_cols);
}

torch::Tensor block_fp8_gemv_f32(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor scales,
    long cols,
    long block_size,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        input.is_cuda() && weights.is_cuda() && scales.is_cuda(),
        "block-FP8 GEMV tensors must be CUDA");
    TORCH_CHECK(
        input.scalar_type() == at::kFloat ||
        input.scalar_type() == at::kBFloat16,
        "block-FP8 GEMV input must be float32 or bfloat16");
    TORCH_CHECK(
        weights.scalar_type() == at::kByte &&
        scales.scalar_type() == at::kFloat,
        "block-FP8 weights/scales must be uint8/float32");
    TORCH_CHECK(
        input.dim() == 2 && input.size(0) == 1 &&
        weights.dim() == 2 && scales.dim() == 2,
        "block-FP8 GEMV expects input [1,C] and matrix weights/scales");
    TORCH_CHECK(
        block_size == 128 && cols > 0 &&
        input.size(1) == cols && weights.size(1) == cols,
        "block-FP8 GEMV currently requires 128x128 blocks");
    const int rows = static_cast<int>(weights.size(0));
    const int scale_rows = (rows + 127) / 128;
    const int scale_cols = (static_cast<int>(cols) + 127) / 128;
    TORCH_CHECK(
        scales.size(0) == scale_rows &&
        scales.size(1) == scale_cols,
        "block-FP8 scale matrix shape mismatch");
    TORCH_CHECK(
        input.get_device() == weights.get_device() &&
        input.get_device() == scales.get_device(),
        "block-FP8 GEMV tensors must share one device");
    auto contiguous_input = input.contiguous();
    auto contiguous_weights = weights.contiguous();
    auto contiguous_scales = scales.contiguous();
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {1, rows},
            input.options().dtype(at::kFloat));
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kFloat &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({1, rows}) &&
        output.get_device() == input.get_device(),
        "block-FP8 output must be contiguous float32 [1,R]");
    auto stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::kFloat) {
        launch_block_fp8_gemv_f32<float>(
            contiguous_input.data_ptr<float>(),
            contiguous_weights.data_ptr<uint8_t>(),
            contiguous_scales.data_ptr<float>(),
            output.data_ptr<float>(),
            rows,
            static_cast<int>(cols),
            scale_cols,
            stream);
    } else {
        launch_block_fp8_gemv_f32<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(
                contiguous_input.data_ptr<at::BFloat16>()),
            contiguous_weights.data_ptr<uint8_t>(),
            contiguous_scales.data_ptr<float>(),
            output.data_ptr<float>(),
            rows,
            static_cast<int>(cols),
            scale_cols,
            stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> glm_norm_qkv_int4(
    torch::Tensor x,
    torch::Tensor norm_weight,
    torch::Tensor q_packed,
    torch::Tensor q_scales,
    torch::Tensor kv_packed,
    torch::Tensor kv_scales,
    long cols,
    long group_size,
    double eps,
    c10::optional<torch::Tensor> residual_update,
    c10::optional<torch::Tensor> q_output_buffer,
    c10::optional<torch::Tensor> kv_output_buffer)
{
    TORCH_CHECK(
        x.is_cuda() && norm_weight.is_cuda() &&
        q_packed.is_cuda() && q_scales.is_cuda() &&
        kv_packed.is_cuda() && kv_scales.is_cuda(),
        "GLM fused Q/K inputs must be CUDA");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat &&
        norm_weight.scalar_type() == at::kFloat &&
        q_packed.scalar_type() == at::kByte &&
        kv_packed.scalar_type() == at::kByte &&
        q_scales.scalar_type() == at::kHalf &&
        kv_scales.scalar_type() == at::kHalf,
        "GLM fused Q/K dtypes do not match");
    TORCH_CHECK(
        x.is_contiguous() && norm_weight.is_contiguous() &&
        q_packed.is_contiguous() && q_scales.is_contiguous() &&
        kv_packed.is_contiguous() && kv_scales.is_contiguous(),
        "GLM fused Q/K tensors must be contiguous");
    TORCH_CHECK(
        x.dim() == 2 && x.size(0) == 1 &&
        norm_weight.dim() == 1 &&
        q_packed.dim() == 2 && kv_packed.dim() == 2 &&
        q_scales.dim() == 2 && kv_scales.dim() == 2 &&
        group_size == 64 && cols > 0 && cols % 64 == 0 &&
        x.size(1) == cols && norm_weight.size(0) == cols &&
        q_packed.size(1) * 2 == cols &&
        kv_packed.size(1) * 2 == cols,
        "GLM fused Q/K shapes do not match");
    const int q_rows = static_cast<int>(q_packed.size(0));
    const int kv_rows = static_cast<int>(kv_packed.size(0));
    const int groups = static_cast<int>(cols / group_size);
    TORCH_CHECK(
        q_rows > 0 && kv_rows > 0 &&
        q_rows % INT4_ROWS_PER_BLOCK == 0 &&
        q_scales.sizes() ==
            torch::IntArrayRef({q_rows, groups}) &&
        kv_scales.sizes() ==
            torch::IntArrayRef({kv_rows, groups}),
        "GLM fused Q/K row/scale shapes do not match");
    const int device = x.get_device();
    TORCH_CHECK(
        norm_weight.get_device() == device &&
        q_packed.get_device() == device &&
        q_scales.get_device() == device &&
        kv_packed.get_device() == device &&
        kv_scales.get_device() == device,
        "GLM fused Q/K tensors must share one device");
    if (residual_update.has_value()) {
        const auto update = residual_update.value();
        TORCH_CHECK(
            update.is_cuda() &&
            update.scalar_type() == at::kFloat &&
            update.is_contiguous() &&
            update.sizes() == x.sizes() &&
            update.get_device() == device,
            "GLM fused residual update must match x");
    }

    auto q_output = q_output_buffer.has_value()
        ? q_output_buffer.value()
        : torch::empty(
            {1, q_rows},
            x.options().dtype(at::kFloat));
    auto kv_output = kv_output_buffer.has_value()
        ? kv_output_buffer.value()
        : torch::empty(
            {1, kv_rows},
            x.options().dtype(at::kFloat));
    TORCH_CHECK(
        q_output.is_cuda() &&
        kv_output.is_cuda() &&
        q_output.scalar_type() == at::kFloat &&
        kv_output.scalar_type() == at::kFloat &&
        q_output.is_contiguous() &&
        kv_output.is_contiguous() &&
        q_output.sizes() ==
            torch::IntArrayRef({1, q_rows}) &&
        kv_output.sizes() ==
            torch::IntArrayRef({1, kv_rows}) &&
        q_output.get_device() == device &&
        kv_output.get_device() == device,
        "GLM fused Q/K output buffers must be contiguous float32 "
        "[1,q_rows]/[1,kv_rows] on the input device");
    auto residual_output = residual_update.has_value()
        ? torch::empty_like(x)
        : torch::Tensor();
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(float);
    constexpr int tracked_devices = 32;
    static size_t configured_shared_bytes[tracked_devices] = {};
    static size_t configured_residual_shared_bytes[
        tracked_devices
    ] = {};
    TORCH_CHECK(
        device >= 0 && device < tracked_devices,
        "GLM fused Q/K device index is out of tracked range");
    if (configured_shared_bytes[device] < shared_bytes) {
        const auto attr_status = cudaFuncSetAttribute(
            glm_norm_qkv_int4_kernel<false, 32>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            attr_status == cudaSuccess,
            "failed to configure GLM fused Q/K shared memory: ",
            cudaGetErrorString(attr_status));
        configured_shared_bytes[device] = shared_bytes;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = q_rows + kv_rows;
    const int blocks =
        (rows + 31) / 32;
    if (residual_update.has_value()) {
        const auto update = residual_update.value();
        if (
            configured_residual_shared_bytes[device] <
            shared_bytes
        ) {
            const auto attr_status = cudaFuncSetAttribute(
                glm_norm_qkv_int4_kernel<true, 32>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared_bytes));
            TORCH_CHECK(
                attr_status == cudaSuccess,
                "failed to configure residual GLM Q/K shared memory: ",
                cudaGetErrorString(attr_status));
            configured_residual_shared_bytes[device] =
                shared_bytes;
        }
        glm_norm_qkv_int4_kernel<true, 32><<<
            blocks,
            dim3(32, 32),
            shared_bytes,
            stream>>>(
                x.data_ptr<float>(),
                update.data_ptr<float>(),
                norm_weight.data_ptr<float>(),
                q_packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(
                    q_scales.data_ptr<at::Half>()),
                kv_packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(
                    kv_scales.data_ptr<at::Half>()),
                q_output.data_ptr<float>(),
                kv_output.data_ptr<float>(),
                residual_output.data_ptr<float>(),
                q_rows,
                kv_rows,
                static_cast<int>(cols),
                groups,
                static_cast<float>(eps));
    } else {
        glm_norm_qkv_int4_kernel<false, 32><<<
            blocks,
            dim3(32, 32),
            shared_bytes,
            stream>>>(
                x.data_ptr<float>(),
                nullptr,
                norm_weight.data_ptr<float>(),
                q_packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(
                    q_scales.data_ptr<at::Half>()),
                kv_packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(
                    kv_scales.data_ptr<at::Half>()),
                q_output.data_ptr<float>(),
                kv_output.data_ptr<float>(),
                nullptr,
                q_rows,
                kv_rows,
                static_cast<int>(cols),
                groups,
                static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (residual_update.has_value())
        return {q_output, kv_output, residual_output};
    return {q_output, kv_output};
}

std::vector<torch::Tensor> glm_residual_norm_router(
    torch::Tensor residual,
    torch::Tensor update,
    torch::Tensor norm_weight,
    torch::Tensor router_weight,
    double eps)
{
    TORCH_CHECK(
        residual.is_cuda() && update.is_cuda() &&
        norm_weight.is_cuda() && router_weight.is_cuda(),
        "GLM residual/router tensors must be CUDA");
    TORCH_CHECK(
        residual.scalar_type() == at::kFloat &&
        update.scalar_type() == at::kFloat &&
        norm_weight.scalar_type() == at::kFloat &&
        router_weight.scalar_type() == at::kFloat,
        "GLM residual/router tensors must be float32");
    TORCH_CHECK(
        residual.is_contiguous() && update.is_contiguous() &&
        norm_weight.is_contiguous() && router_weight.is_contiguous(),
        "GLM residual/router tensors must be contiguous");
    TORCH_CHECK(
        residual.dim() == 2 && residual.size(0) == 1 &&
        update.sizes() == residual.sizes() &&
        norm_weight.dim() == 1 &&
        norm_weight.size(0) == residual.size(1) &&
        router_weight.dim() == 2 &&
        router_weight.size(1) == residual.size(1),
        "GLM residual/router shapes do not match");
    const int device = residual.get_device();
    TORCH_CHECK(
        update.get_device() == device &&
        norm_weight.get_device() == device &&
        router_weight.get_device() == device,
        "GLM residual/router tensors must share one device");
    int rows = static_cast<int>(router_weight.size(0));
    int cols = static_cast<int>(residual.size(1));
    TORCH_CHECK(
        rows > 0 && cols > 0 && cols % 256 == 0,
        "GLM residual/router dimensions are not supported");

    auto residual_output = torch::empty_like(residual);
    auto norm_output = torch::empty_like(residual);
    auto logits_output = torch::empty(
        {1, rows},
        residual.options());
    dim3 block(32, 8);
    const int blocks = (rows + 7) / 8;
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    glm_residual_norm_router_kernel<<<
        blocks,
        block,
        shared_bytes,
        stream>>>(
            residual.data_ptr<float>(),
            update.data_ptr<float>(),
            norm_weight.data_ptr<float>(),
            router_weight.data_ptr<float>(),
            residual_output.data_ptr<float>(),
            norm_output.data_ptr<float>(),
            logits_output.data_ptr<float>(),
            rows,
            cols,
            static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {residual_output, norm_output, logits_output};
}

std::vector<torch::Tensor> glm_residual_norm_router_norm_out(
    torch::Tensor residual,
    torch::Tensor update,
    torch::Tensor norm_weight,
    torch::Tensor router_weight,
    double eps,
    torch::Tensor norm_output,
    c10::optional<torch::Tensor> residual_output_buffer,
    c10::optional<torch::Tensor> logits_output_buffer)
{
    TORCH_CHECK(
        residual.is_cuda() && update.is_cuda() &&
        norm_weight.is_cuda() && router_weight.is_cuda() &&
        norm_output.is_cuda(),
        "GLM residual/router output-buffer tensors must be CUDA");
    TORCH_CHECK(
        residual.scalar_type() == at::kFloat &&
        update.scalar_type() == at::kFloat &&
        norm_weight.scalar_type() == at::kFloat &&
        router_weight.scalar_type() == at::kFloat &&
        norm_output.scalar_type() == at::kFloat,
        "GLM residual/router output-buffer tensors must be float32");
    TORCH_CHECK(
        residual.is_contiguous() && update.is_contiguous() &&
        norm_weight.is_contiguous() && router_weight.is_contiguous() &&
        norm_output.is_contiguous() &&
        residual.dim() == 2 && residual.size(0) == 1 &&
        update.sizes() == residual.sizes() &&
        norm_output.sizes() == residual.sizes() &&
        norm_weight.dim() == 1 &&
        norm_weight.size(0) == residual.size(1) &&
        router_weight.dim() == 2 &&
        router_weight.size(1) == residual.size(1),
        "GLM residual/router output-buffer shapes do not match");
    const int device = residual.get_device();
    TORCH_CHECK(
        update.get_device() == device &&
        norm_weight.get_device() == device &&
        router_weight.get_device() == device &&
        norm_output.get_device() == device,
        "GLM residual/router output-buffer tensors must share one device");
    const int rows = static_cast<int>(router_weight.size(0));
    const int cols = static_cast<int>(residual.size(1));
    TORCH_CHECK(
        rows > 0 && cols > 0 && cols % 256 == 0,
        "GLM residual/router output-buffer dimensions are unsupported");
    auto residual_output = residual_output_buffer.has_value()
        ? residual_output_buffer.value()
        : torch::empty_like(residual);
    auto logits_output = logits_output_buffer.has_value()
        ? logits_output_buffer.value()
        : torch::empty(
            {1, rows},
            residual.options());
    TORCH_CHECK(
        residual_output.is_cuda() &&
        logits_output.is_cuda() &&
        residual_output.scalar_type() == at::kFloat &&
        logits_output.scalar_type() == at::kFloat &&
        residual_output.is_contiguous() &&
        logits_output.is_contiguous() &&
        residual_output.sizes() == residual.sizes() &&
        logits_output.sizes() ==
            torch::IntArrayRef({1, rows}) &&
        residual_output.get_device() == device &&
        logits_output.get_device() == device,
        "GLM residual/router caller outputs must be contiguous float32 "
        "on the input device");
    dim3 block(32, 8);
    const int blocks = (rows + 7) / 8;
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    glm_residual_norm_router_kernel<<<
        blocks,
        block,
        shared_bytes,
        stream>>>(
            residual.data_ptr<float>(),
            update.data_ptr<float>(),
            norm_weight.data_ptr<float>(),
            router_weight.data_ptr<float>(),
            residual_output.data_ptr<float>(),
            norm_output.data_ptr<float>(),
            logits_output.data_ptr<float>(),
            rows,
            cols,
            static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {residual_output, norm_output, logits_output};
}

torch::Tensor residual_add3(
    torch::Tensor residual,
    torch::Tensor routed,
    torch::Tensor shared)
{
    TORCH_CHECK(
        residual.is_cuda() && routed.is_cuda() && shared.is_cuda(),
        "three-way residual tensors must be CUDA");
    TORCH_CHECK(
        (
            residual.scalar_type() == at::kFloat
            || residual.scalar_type() == at::kBFloat16
        )
        && routed.scalar_type() == residual.scalar_type()
        && shared.scalar_type() == residual.scalar_type(),
        "three-way residual tensors must share float32 or bfloat16 dtype");
    TORCH_CHECK(
        residual.is_contiguous() &&
        routed.is_contiguous() &&
        shared.is_contiguous() &&
        routed.sizes() == residual.sizes() &&
        shared.sizes() == residual.sizes(),
        "three-way residual tensors must be contiguous and shape-equal");
    const int device = residual.get_device();
    TORCH_CHECK(
        routed.get_device() == device &&
        shared.get_device() == device,
        "three-way residual tensors must share one device");
    auto output = torch::empty_like(residual);
    const int count = static_cast<int>(residual.numel());
    const int blocks = std::min(32, (count + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (residual.scalar_type() == at::kFloat) {
        glm_moe_residual_add_kernel<<<blocks, 256, 0, stream>>>(
            residual.data_ptr<float>(),
            routed.data_ptr<float>(),
            shared.data_ptr<float>(),
            output.data_ptr<float>(),
            count);
    } else {
        residual_add3_bf16_kernel<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                routed.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                shared.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            count);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor glm_ep_reduce_residual(
    std::vector<torch::Tensor> contributions,
    torch::Tensor residual)
{
    TORCH_CHECK(
        !contributions.empty() && contributions.size() <= 16,
        "GLM TP reduction requires 1 to 16 contributions");
    auto primary_partial = contributions[0];
    TORCH_CHECK(
        primary_partial.is_cuda() &&
        residual.is_cuda(),
        "GLM EP reduction tensors must be CUDA");
    TORCH_CHECK(
        primary_partial.scalar_type() == at::kFloat &&
        residual.scalar_type() == at::kFloat,
        "GLM EP reduction tensors must be float32");
    TORCH_CHECK(
        primary_partial.is_contiguous() &&
        residual.is_contiguous() &&
        primary_partial.numel() == residual.numel(),
        "GLM EP reduction tensors must be contiguous and size-equal");
    const int device = primary_partial.get_device();
    TORCH_CHECK(
        residual.get_device() == device,
        "GLM TP residual must share the primary contribution device");
    for (const auto contribution : contributions) {
        TORCH_CHECK(
            contribution.is_cuda() &&
            contribution.scalar_type() == at::kFloat &&
            contribution.is_contiguous() &&
            contribution.numel() == primary_partial.numel(),
            "GLM TP contributions must be contiguous float32 tensors "
            "with matching size");
        if (contribution.get_device() != device)
            ensure_peer_access(
                device,
                contribution.get_device(),
                "GLM TP contribution reduction");
    }
    const float* contribution_ptrs[16] = {};
    for (size_t index = 1; index < contributions.size(); ++index)
        contribution_ptrs[index] = contributions[index].data_ptr<float>();
    const int count = static_cast<int>(primary_partial.numel());
    const int blocks = std::min(32, (count + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();
    glm_ep_reduce_residual_kernel<<<blocks, 256, 0, stream>>>(
        primary_partial.data_ptr<float>(),
        contribution_ptrs[1],
        contribution_ptrs[2],
        contribution_ptrs[3],
        contribution_ptrs[4],
        contribution_ptrs[5],
        contribution_ptrs[6],
        contribution_ptrs[7],
        contribution_ptrs[8],
        contribution_ptrs[9],
        contribution_ptrs[10],
        contribution_ptrs[11],
        contribution_ptrs[12],
        contribution_ptrs[13],
        contribution_ptrs[14],
        contribution_ptrs[15],
        residual.data_ptr<float>(),
        static_cast<int>(contributions.size()),
        count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return primary_partial.view(residual.sizes());
}

void launch_tp_all_rank_reduce_one(
    const std::vector<torch::Tensor>& contributions,
    torch::Tensor output,
    cudaStream_t stream)
{
    TORCH_CHECK(
        !contributions.empty() && contributions.size() <= 16,
        "TP all-rank reduction requires 1 to 16 contributions");
    TORCH_CHECK(
        output.is_cuda() &&
        output.is_contiguous() &&
        (
            output.scalar_type() == at::kFloat ||
            output.scalar_type() == at::kBFloat16
        ),
        "TP all-rank output must be contiguous CUDA float32/BF16");
    const int target = output.get_device();
    const int count = static_cast<int>(output.numel());
    const float* pointers[16] = {};
    for (size_t index = 0; index < contributions.size(); ++index) {
        const auto contribution = contributions[index];
        TORCH_CHECK(
            contribution.is_cuda() &&
            contribution.scalar_type() == at::kFloat &&
            contribution.is_contiguous() &&
            contribution.numel() == count,
            "TP all-rank contributions must be matching contiguous "
            "CUDA float32 tensors");
        if (contribution.get_device() != target)
            ensure_peer_access(
                target,
                contribution.get_device(),
                "TP all-rank contribution");
        pointers[index] = contribution.data_ptr<float>();
    }
    const int blocks = std::min(32, (count + 255) / 256);
#define TPQ_ALL_RANK_ARGUMENTS \
    pointers[0], pointers[1], pointers[2], pointers[3], \
    pointers[4], pointers[5], pointers[6], pointers[7], \
    pointers[8], pointers[9], pointers[10], pointers[11], \
    pointers[12], pointers[13], pointers[14], pointers[15], \
    static_cast<int>(contributions.size()), count
    if (output.scalar_type() == at::kFloat) {
        tp_all_rank_reduce_kernel<float><<<
            blocks, 256, 0, stream>>>(
                output.data_ptr<float>(),
                TPQ_ALL_RANK_ARGUMENTS);
    } else {
        tp_all_rank_reduce_kernel<__nv_bfloat16><<<
            blocks, 256, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(
                    output.data_ptr<at::BFloat16>()),
                TPQ_ALL_RANK_ARGUMENTS);
    }
#undef TPQ_ALL_RANK_ARGUMENTS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_tp_moe_finalize_one(
    const std::vector<torch::Tensor>& routed_contributions,
    const std::vector<torch::Tensor>& shared_contributions,
    torch::Tensor residual,
    torch::Tensor routed_workspace,
    torch::Tensor shared_workspace,
    torch::Tensor output,
    cudaStream_t stream,
    const bool fused)
{
    TORCH_CHECK(
        residual.is_cuda() &&
        routed_workspace.is_cuda() &&
        shared_workspace.is_cuda() &&
        output.is_cuda() &&
        residual.scalar_type() == at::kBFloat16 &&
        routed_workspace.scalar_type() == at::kBFloat16 &&
        shared_workspace.scalar_type() == at::kBFloat16 &&
        output.scalar_type() == at::kBFloat16 &&
        residual.is_contiguous() &&
        routed_workspace.is_contiguous() &&
        shared_workspace.is_contiguous() &&
        output.is_contiguous() &&
        residual.numel() == output.numel() &&
        routed_workspace.numel() == output.numel() &&
        shared_workspace.numel() == output.numel(),
        "TP MoE finalizer buffers must be matching contiguous BF16 CUDA "
        "tensors");
    const int target = output.get_device();
    TORCH_CHECK(
        residual.get_device() == target &&
        routed_workspace.get_device() == target &&
        shared_workspace.get_device() == target,
        "TP MoE finalizer buffers must share one target device");
    if (fused) {
        TORCH_CHECK(
            !routed_contributions.empty() &&
            routed_contributions.size() == shared_contributions.size() &&
            routed_contributions.size() <= 16,
            "fused TP MoE finalizer requires matching 1 to 16 rank "
            "contributions");
        const float* routed_ptrs[16] = {};
        const float* shared_ptrs[16] = {};
        for (size_t rank = 0; rank < routed_contributions.size(); ++rank) {
            const auto routed = routed_contributions[rank];
            const auto shared = shared_contributions[rank];
            TORCH_CHECK(
                routed.is_cuda() &&
                shared.is_cuda() &&
                routed.scalar_type() == at::kFloat &&
                shared.scalar_type() == at::kFloat &&
                routed.is_contiguous() &&
                shared.is_contiguous() &&
                routed.numel() == output.numel() &&
                shared.numel() == output.numel(),
                "fused TP MoE contributions must be matching contiguous "
                "CUDA float32 tensors");
            if (routed.get_device() != target)
                ensure_peer_access(
                    target,
                    routed.get_device(),
                    "fused TP routed contribution");
            if (shared.get_device() != target)
                ensure_peer_access(
                    target,
                    shared.get_device(),
                    "fused TP shared contribution");
            routed_ptrs[rank] = routed.data_ptr<float>();
            shared_ptrs[rank] = shared.data_ptr<float>();
        }
        const int count = static_cast<int>(output.numel());
        const int blocks = std::min(32, (count + 255) / 256);
#define TPQ_FUSED_MOE_POINTERS(prefix) \
        prefix##_ptrs[0], prefix##_ptrs[1], prefix##_ptrs[2], \
        prefix##_ptrs[3], prefix##_ptrs[4], prefix##_ptrs[5], \
        prefix##_ptrs[6], prefix##_ptrs[7], prefix##_ptrs[8], \
        prefix##_ptrs[9], prefix##_ptrs[10], prefix##_ptrs[11], \
        prefix##_ptrs[12], prefix##_ptrs[13], prefix##_ptrs[14], \
        prefix##_ptrs[15]
        tp_moe_finalize_all_rank_bf16_kernel<<<
            blocks, 256, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(
                    output.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    residual.data_ptr<at::BFloat16>()),
                TPQ_FUSED_MOE_POINTERS(routed),
                TPQ_FUSED_MOE_POINTERS(shared),
                static_cast<int>(routed_contributions.size()),
                count);
#undef TPQ_FUSED_MOE_POINTERS
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    launch_tp_all_rank_reduce_one(
        routed_contributions,
        routed_workspace,
        stream);
    launch_tp_all_rank_reduce_one(
        shared_contributions,
        shared_workspace,
        stream);
    const int count = static_cast<int>(output.numel());
    const int blocks = std::min(32, (count + 255) / 256);
    residual_add3_bf16_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            routed_workspace.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            shared_workspace.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            output.data_ptr<at::BFloat16>()),
        count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> tp_all_rank_reduce(
    std::vector<torch::Tensor> contributions,
    std::vector<torch::Tensor> outputs)
{
    TORCH_CHECK(
        !outputs.empty() && outputs.size() <= 16,
        "TP all-rank reduction requires 1 to 16 outputs");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    for (const auto output : outputs) {
        const int target = output.get_device();
        C10_CUDA_CHECK(cudaSetDevice(target));
        launch_tp_all_rank_reduce_one(
            contributions,
            output,
            at::cuda::getCurrentCUDAStream(target));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
    return outputs;
}

namespace {

bool tp_environment_enabled(const char* name)
{
    const char* setting = std::getenv(name);
    return (
        setting != nullptr &&
        setting[0] == '1' &&
        setting[1] == '\0');
}

inline void graph_dispatch_pause()
{
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || \
    defined(_M_X64)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

int graph_dispatch_spin_iterations()
{
    const char* setting = std::getenv("TPQ_TP_GRAPH_SPIN");
    if (setting == nullptr || setting[0] == '\0')
        return 1 << 20;
    char* end = nullptr;
    const long parsed = std::strtol(setting, &end, 10);
    if (
        end == setting ||
        *end != '\0' ||
        parsed < 0 ||
        parsed > (1 << 26))
        return 1 << 20;
    return static_cast<int>(parsed);
}

using GraphLaunchCallback = cudaError_t (*)(
    const void* context,
    size_t rank,
    cudaEvent_t ready);

struct GraphLaunchTask {
    cudaGraphExec_t graph = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    cudaEvent_t ready = nullptr;
    GraphLaunchCallback callback = nullptr;
    const void* callback_context = nullptr;
    size_t callback_rank = 0;
};

// vLLM assigns one process to every TP rank, so CUDA graph submission happens
// concurrently. TPQ keeps one process to preserve the RAM-expert fallback and
// previously paid cudaSetDevice + three CUDA API calls serially for every
// rank and every layer. A persistent worker per secondary device provides the
// same concurrent host-side submission without process duplication.
class GraphLaunchWorker {
public:
    explicit GraphLaunchWorker(int device)
        : device_(device), thread_(&GraphLaunchWorker::run, this)
    {
    }

    ~GraphLaunchWorker()
    {
        {
            std::lock_guard<std::mutex> lock(sleep_mutex_);
            stopping_.store(true, std::memory_order_release);
        }
        wake_.notify_one();
        if (thread_.joinable())
            thread_.join();
    }

    GraphLaunchWorker(const GraphLaunchWorker&) = delete;
    GraphLaunchWorker& operator=(const GraphLaunchWorker&) = delete;

    uint64_t submit(const GraphLaunchTask& task)
    {
        const uint64_t previous =
            requested_.load(std::memory_order_relaxed);
        while (
            completed_.load(std::memory_order_acquire) != previous
        )
            graph_dispatch_pause();
        task_ = task;
        const uint64_t sequence = previous + 1;
        {
            // The producer update and consumer wait share this mutex. This
            // closes the condition-variable notification window without
            // serializing any CUDA work; the worker only holds the mutex
            // while it is entering or leaving its idle wait.
            std::lock_guard<std::mutex> lock(sleep_mutex_);
            requested_.store(sequence, std::memory_order_release);
        }
        wake_.notify_one();
        return sequence;
    }

    void wait(uint64_t sequence)
    {
        while (
            completed_.load(std::memory_order_acquire) != sequence
        )
            graph_dispatch_pause();
        TORCH_CHECK(
            set_device_status_ == cudaSuccess,
            "parallel graph worker cudaSetDevice failed: ",
            cudaGetErrorString(set_device_status_));
        TORCH_CHECK(
            wait_status_ == cudaSuccess,
            "parallel graph worker cudaStreamWaitEvent failed: ",
            cudaGetErrorString(wait_status_));
        TORCH_CHECK(
            launch_status_ == cudaSuccess,
            "parallel graph worker cudaGraphLaunch failed: ",
            cudaGetErrorString(launch_status_));
        TORCH_CHECK(
            record_status_ == cudaSuccess,
            "parallel graph worker cudaEventRecord failed: ",
            cudaGetErrorString(record_status_));
    }

private:
    void run()
    {
        set_device_status_ = cudaSetDevice(device_);
        uint64_t observed = 0;
        while (!stopping_.load(std::memory_order_acquire)) {
            const uint64_t requested =
                requested_.load(std::memory_order_acquire);
            if (requested != observed) {
                const GraphLaunchTask task = task_;
                if (set_device_status_ == cudaSuccess) {
                    if (task.callback != nullptr) {
                        wait_status_ = task.callback(
                            task.callback_context,
                            task.callback_rank,
                            task.ready);
                        launch_status_ = wait_status_;
                        record_status_ = wait_status_;
                    } else {
                        wait_status_ = cudaStreamWaitEvent(
                            task.stream,
                            task.ready,
                            0);
                        launch_status_ = (
                            wait_status_ == cudaSuccess
                                ? cudaGraphLaunch(task.graph, task.stream)
                                : wait_status_);
                        record_status_ = (
                            launch_status_ == cudaSuccess
                                ? cudaEventRecord(task.done, task.stream)
                                : launch_status_);
                    }
                } else {
                    wait_status_ = set_device_status_;
                    launch_status_ = set_device_status_;
                    record_status_ = set_device_status_;
                }
                observed = requested;
                completed_.store(observed, std::memory_order_release);
                continue;
            }

            // Keep workers hot across adjacent Attention/MoE stages, then
            // sleep when the server is idle so the optimization does not
            // permanently consume one CPU core per rank.
            bool changed = false;
            const int spin_limit = graph_dispatch_spin_iterations();
            for (int spin = 0; spin < spin_limit; ++spin) {
                if (
                    stopping_.load(std::memory_order_relaxed) ||
                    requested_.load(std::memory_order_acquire) != observed
                ) {
                    changed = true;
                    break;
                }
                graph_dispatch_pause();
            }
            if (changed)
                continue;
            std::unique_lock<std::mutex> lock(sleep_mutex_);
            wake_.wait(
                lock,
                [&] {
                    return
                        stopping_.load(std::memory_order_acquire) ||
                        requested_.load(std::memory_order_acquire)
                            != observed;
                });
        }
    }

    int device_;
    GraphLaunchTask task_;
    std::atomic<uint64_t> requested_{0};
    std::atomic<uint64_t> completed_{0};
    std::atomic<bool> stopping_{false};
    std::mutex sleep_mutex_;
    std::condition_variable wake_;
    cudaError_t set_device_status_ = cudaSuccess;
    cudaError_t wait_status_ = cudaSuccess;
    cudaError_t launch_status_ = cudaSuccess;
    cudaError_t record_status_ = cudaSuccess;
    std::thread thread_;
};

constexpr int kMaxGraphDispatchDevices = 16;
std::array<std::unique_ptr<GraphLaunchWorker>, kMaxGraphDispatchDevices>
    graph_launch_workers;
std::array<std::once_flag, kMaxGraphDispatchDevices>
    graph_launch_worker_once;

GraphLaunchWorker& graph_launch_worker(int device)
{
    TORCH_CHECK(
        device >= 0 && device < kMaxGraphDispatchDevices,
        "parallel graph dispatch supports CUDA devices [0, ",
        kMaxGraphDispatchDevices,
        ")");
    std::call_once(
        graph_launch_worker_once[device],
        [device] {
            graph_launch_workers[device] =
                std::make_unique<GraphLaunchWorker>(device);
        });
    return *graph_launch_workers[device];
}

void launch_cuda_graphs_sequential(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    int64_t source_event)
{
    const auto count = devices.size();
    TORCH_CHECK(
        count > 0 &&
        graph_execs.size() == count &&
        streams.size() == count &&
        done_events.size() == count,
        "batched CUDA Graph launch vectors must be non-empty and size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    const auto ready = reinterpret_cast<cudaEvent_t>(
        static_cast<uintptr_t>(source_event));
    for (size_t index = 0; index < count; ++index) {
        C10_CUDA_CHECK(cudaSetDevice(static_cast<int>(devices[index])));
        const auto stream = reinterpret_cast<cudaStream_t>(
            static_cast<uintptr_t>(streams[index]));
        const auto graph_exec = reinterpret_cast<cudaGraphExec_t>(
            static_cast<uintptr_t>(graph_execs[index]));
        const auto done = reinterpret_cast<cudaEvent_t>(
            static_cast<uintptr_t>(done_events[index]));
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, ready, 0));
        C10_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        C10_CUDA_CHECK(cudaEventRecord(done, stream));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
}

void launch_cuda_graphs_parallel(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    int64_t source_event)
{
    const auto count = devices.size();
    TORCH_CHECK(
        count > 1 &&
        count <= kMaxGraphDispatchDevices &&
        graph_execs.size() == count &&
        streams.size() == count &&
        done_events.size() == count,
        "parallel CUDA Graph launch vectors must be size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    const auto ready = reinterpret_cast<cudaEvent_t>(
        static_cast<uintptr_t>(source_event));

    std::array<GraphLaunchWorker*, kMaxGraphDispatchDevices>
        workers{};
    std::array<uint64_t, kMaxGraphDispatchDevices> sequences{};
    for (size_t index = 1; index < count; ++index) {
        const int device = static_cast<int>(devices[index]);
        GraphLaunchWorker& worker = graph_launch_worker(device);
        workers[index] = &worker;
        sequences[index] = worker.submit({
            reinterpret_cast<cudaGraphExec_t>(
                static_cast<uintptr_t>(graph_execs[index])),
            reinterpret_cast<cudaStream_t>(
                static_cast<uintptr_t>(streams[index])),
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(done_events[index])),
            ready,
        });
    }

    const int primary_device = static_cast<int>(devices[0]);
    if (original_device != primary_device)
        C10_CUDA_CHECK(cudaSetDevice(primary_device));
    const auto primary_stream = reinterpret_cast<cudaStream_t>(
        static_cast<uintptr_t>(streams[0]));
    const auto primary_graph = reinterpret_cast<cudaGraphExec_t>(
        static_cast<uintptr_t>(graph_execs[0]));
    const auto primary_done = reinterpret_cast<cudaEvent_t>(
        static_cast<uintptr_t>(done_events[0]));
    C10_CUDA_CHECK(cudaStreamWaitEvent(primary_stream, ready, 0));
    C10_CUDA_CHECK(cudaGraphLaunch(primary_graph, primary_stream));
    C10_CUDA_CHECK(cudaEventRecord(primary_done, primary_stream));

    for (size_t index = 1; index < count; ++index)
        workers[index]->wait(sequences[index]);
    if (original_device != primary_device)
        C10_CUDA_CHECK(cudaSetDevice(original_device));
}

void launch_cuda_graphs_from_events_sequential(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    const std::vector<int64_t>& ready_events)
{
    const auto count = devices.size();
    TORCH_CHECK(
        count > 0 &&
        graph_execs.size() == count &&
        streams.size() == count &&
        done_events.size() == count &&
        ready_events.size() == count,
        "replicated CUDA Graph vectors must be non-empty and size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    for (size_t index = 0; index < count; ++index) {
        C10_CUDA_CHECK(cudaSetDevice(static_cast<int>(devices[index])));
        const auto stream = reinterpret_cast<cudaStream_t>(
            static_cast<uintptr_t>(streams[index]));
        const auto graph_exec = reinterpret_cast<cudaGraphExec_t>(
            static_cast<uintptr_t>(graph_execs[index]));
        const auto done = reinterpret_cast<cudaEvent_t>(
            static_cast<uintptr_t>(done_events[index]));
        const auto ready = reinterpret_cast<cudaEvent_t>(
            static_cast<uintptr_t>(ready_events[index]));
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, ready, 0));
        C10_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        C10_CUDA_CHECK(cudaEventRecord(done, stream));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
}

void launch_cuda_graphs_from_events_parallel(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    const std::vector<int64_t>& ready_events)
{
    const auto count = devices.size();
    TORCH_CHECK(
        count > 1 &&
        count <= kMaxGraphDispatchDevices &&
        graph_execs.size() == count &&
        streams.size() == count &&
        done_events.size() == count &&
        ready_events.size() == count,
        "parallel replicated CUDA Graph vectors must be size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    std::array<GraphLaunchWorker*, kMaxGraphDispatchDevices> workers{};
    std::array<uint64_t, kMaxGraphDispatchDevices> sequences{};
    for (size_t index = 1; index < count; ++index) {
        const int device = static_cast<int>(devices[index]);
        GraphLaunchWorker& worker = graph_launch_worker(device);
        workers[index] = &worker;
        sequences[index] = worker.submit({
            reinterpret_cast<cudaGraphExec_t>(
                static_cast<uintptr_t>(graph_execs[index])),
            reinterpret_cast<cudaStream_t>(
                static_cast<uintptr_t>(streams[index])),
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(done_events[index])),
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(ready_events[index])),
        });
    }
    const int primary_device = static_cast<int>(devices[0]);
    if (original_device != primary_device)
        C10_CUDA_CHECK(cudaSetDevice(primary_device));
    const auto primary_stream = reinterpret_cast<cudaStream_t>(
        static_cast<uintptr_t>(streams[0]));
    C10_CUDA_CHECK(
        cudaStreamWaitEvent(
            primary_stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(ready_events[0])),
            0));
    C10_CUDA_CHECK(
        cudaGraphLaunch(
            reinterpret_cast<cudaGraphExec_t>(
                static_cast<uintptr_t>(graph_execs[0])),
            primary_stream));
    C10_CUDA_CHECK(
        cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(done_events[0])),
            primary_stream));
    for (size_t index = 1; index < count; ++index)
        workers[index]->wait(sequences[index]);
    if (original_device != primary_device)
        C10_CUDA_CHECK(cudaSetDevice(original_device));
}

}  // namespace

void launch_cuda_graphs(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    int64_t source_event)
{
    TORCH_CHECK(
        !devices.empty(),
        "CUDA Graph launch requires at least one TP device");
    int current_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));
    TORCH_CHECK(
        current_device == static_cast<int>(devices[0]),
        "CUDA Graph source event must be recorded on the primary device");
    const auto ready = reinterpret_cast<cudaEvent_t>(
        static_cast<uintptr_t>(source_event));
    C10_CUDA_CHECK(
        cudaEventRecord(
            ready,
            at::cuda::getCurrentCUDAStream(current_device)));
    const char* setting = std::getenv("TPQ_TP_PARALLEL_LAUNCH");
    const bool explicitly_enabled = (
        setting != nullptr &&
        setting[0] == '1' &&
        setting[1] == '\0');
    const bool explicitly_disabled = (
        setting != nullptr &&
        setting[0] == '0' &&
        setting[1] == '\0');
    const bool enabled = (
        devices.size() > 1 &&
        (
            explicitly_enabled ||
            (
                !explicitly_disabled &&
                devices.size() >= 8
            )
        ));
    if (enabled) {
        launch_cuda_graphs_parallel(
            devices,
            graph_execs,
            streams,
            done_events,
            source_event);
    } else {
        launch_cuda_graphs_sequential(
            devices,
            graph_execs,
            streams,
            done_events,
            source_event);
    }
}

void launch_cuda_graphs_from_events(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    const std::vector<int64_t>& ready_events)
{
    const char* setting = std::getenv("TPQ_TP_PARALLEL_LAUNCH");
    const bool explicitly_enabled = (
        setting != nullptr &&
        setting[0] == '1' &&
        setting[1] == '\0');
    const bool explicitly_disabled = (
        setting != nullptr &&
        setting[0] == '0' &&
        setting[1] == '\0');
    const bool enabled = (
        devices.size() > 1 &&
        (
            explicitly_enabled ||
            (
                !explicitly_disabled &&
                devices.size() >= 8
            )
        ));
    if (enabled) {
        launch_cuda_graphs_from_events_parallel(
            devices,
            graph_execs,
            streams,
            done_events,
            ready_events);
    } else {
        launch_cuda_graphs_from_events_sequential(
            devices,
            graph_execs,
            streams,
            done_events,
            ready_events);
    }
}

torch::Tensor launch_cuda_graphs_reduce(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    int64_t source_event,
    std::vector<torch::Tensor> contributions,
    torch::Tensor residual)
{
    launch_cuda_graphs(
        devices,
        graph_execs,
        streams,
        done_events,
        source_event);
    TORCH_CHECK(
        !devices.empty() &&
        contributions.size() >= devices.size() &&
        contributions.size() <= 16,
        "TP graph reduction needs at least one contribution per rank "
        "and supports at most 16");
    const int primary_device = contributions[0].get_device();
    int current_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));
    TORCH_CHECK(
        current_device == primary_device,
        "TP graph reduction must return to the primary device");
    const auto primary_stream = at::cuda::getCurrentCUDAStream();
    for (const auto raw_event : done_events) {
        const auto event = reinterpret_cast<cudaEvent_t>(
            static_cast<uintptr_t>(raw_event));
        C10_CUDA_CHECK(
            cudaStreamWaitEvent(primary_stream, event, 0));
    }
    return glm_ep_reduce_residual(
        contributions,
        residual);
}

std::vector<torch::Tensor> launch_cuda_graphs_reduce_many(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    int64_t source_event,
    std::vector<std::vector<torch::Tensor>> contribution_groups,
    std::vector<torch::Tensor> residuals)
{
    launch_cuda_graphs(
        devices,
        graph_execs,
        streams,
        done_events,
        source_event);
    TORCH_CHECK(
        !devices.empty() &&
        !contribution_groups.empty() &&
        contribution_groups.size() == residuals.size(),
        "TP graph multi-reduction groups/residuals must be non-empty "
        "and size-equal");
    const int primary_device = residuals[0].get_device();
    int current_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));
    TORCH_CHECK(
        current_device == primary_device,
        "TP graph multi-reduction must return to the primary device");
    const auto primary_stream = at::cuda::getCurrentCUDAStream();
    for (const auto raw_event : done_events) {
        const auto event = reinterpret_cast<cudaEvent_t>(
            static_cast<uintptr_t>(raw_event));
        C10_CUDA_CHECK(cudaStreamWaitEvent(primary_stream, event, 0));
    }
    std::vector<torch::Tensor> outputs;
    outputs.reserve(residuals.size());
    for (size_t index = 0; index < residuals.size(); ++index) {
        TORCH_CHECK(
            contribution_groups[index].size() >= devices.size() &&
            contribution_groups[index].size() <= 16,
            "each TP graph multi-reduction needs one contribution per rank");
        TORCH_CHECK(
            residuals[index].get_device() == primary_device,
            "TP graph multi-reduction residuals must share primary device");
        outputs.push_back(
            glm_ep_reduce_residual(
                contribution_groups[index],
                residuals[index]));
    }
    return outputs;
}

std::vector<torch::Tensor> launch_cuda_graphs_reduce_norm_router(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    int64_t source_event,
    std::vector<torch::Tensor> contributions,
    torch::Tensor attention_zero,
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor router_weight,
    double eps,
    torch::Tensor norm_output,
    c10::optional<torch::Tensor> residual_output,
    c10::optional<torch::Tensor> logits_output)
{
    auto attention_update = launch_cuda_graphs_reduce(
        devices,
        graph_execs,
        streams,
        done_events,
        source_event,
        std::move(contributions),
        attention_zero);
    return glm_residual_norm_router_norm_out(
        residual,
        attention_update,
        norm_weight,
        router_weight,
        eps,
        norm_output,
        residual_output,
        logits_output);
}

class TPNoOwnerDecodeLayerPlan;

class TPStageBarrier {
public:
    explicit TPStageBarrier(size_t participants)
        : participants_(participants)
    {
        TORCH_CHECK(
            participants_ > 0,
            "TP stage barrier requires at least one rank");
    }

    void arrive_and_wait() const
    {
        const uint64_t generation =
            generation_.load(std::memory_order_acquire);
        if (
            arrived_.fetch_add(1, std::memory_order_acq_rel) + 1
            == participants_
        ) {
            arrived_.store(0, std::memory_order_release);
            generation_.fetch_add(1, std::memory_order_acq_rel);
            return;
        }
        while (
            generation_.load(std::memory_order_acquire)
            == generation
        )
            graph_dispatch_pause();
    }

private:
    size_t participants_;
    mutable std::atomic<size_t> arrived_{0};
    mutable std::atomic<uint64_t> generation_{0};
};

class TPGraphLaunchBatch {
public:
    TPGraphLaunchBatch(
        std::vector<int64_t> devices,
        std::vector<int64_t> graph_execs,
        std::vector<int64_t> streams,
        std::vector<int64_t> done_events,
        int64_t source_event)
        : devices_(std::move(devices)),
          graph_execs_(std::move(graph_execs)),
          streams_(std::move(streams)),
          done_events_(std::move(done_events)),
          source_event_(source_event),
          collective_event_barrier_enabled_(
              tp_environment_enabled("TPQ_TP_EVENT_BARRIER")),
          fused_moe_finalize_enabled_(
              tp_environment_enabled("TPQ_TP_FUSED_MOE_FINALIZE"))
    {
        validate_handles();
        initialize_collective_events();
    }

    TPGraphLaunchBatch(
        std::vector<int64_t> devices,
        std::vector<std::vector<int64_t>> child_graphs,
        std::vector<int64_t> streams,
        std::vector<int64_t> done_events,
        int64_t source_event)
        : devices_(std::move(devices)),
          streams_(std::move(streams)),
          done_events_(std::move(done_events)),
          source_event_(source_event),
          collective_event_barrier_enabled_(
              tp_environment_enabled("TPQ_TP_EVENT_BARRIER")),
          fused_moe_finalize_enabled_(
              tp_environment_enabled("TPQ_TP_FUSED_MOE_FINALIZE"))
    {
        TORCH_CHECK(
            !devices_.empty() &&
            child_graphs.size() == devices_.size() &&
            streams_.size() == devices_.size() &&
            done_events_.size() == devices_.size() &&
            source_event_ != 0,
            "TP Graph sequence handles must be non-empty and size-equal");
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        owned_graphs_.reserve(devices_.size());
        owned_graph_execs_.reserve(devices_.size());
        graph_execs_.reserve(devices_.size());
        for (size_t rank = 0; rank < devices_.size(); ++rank) {
            TORCH_CHECK(
                !child_graphs[rank].empty(),
                "each TP rank sequence needs at least one child graph");
            C10_CUDA_CHECK(
                cudaSetDevice(static_cast<int>(devices_[rank])));
            cudaGraph_t parent = nullptr;
            C10_CUDA_CHECK(cudaGraphCreate(&parent, 0));
            cudaGraphNode_t previous = nullptr;
            for (const auto raw_child : child_graphs[rank]) {
                TORCH_CHECK(
                    raw_child != 0,
                    "TP Graph sequence child handle must be non-zero");
                cudaGraphNode_t child_node = nullptr;
                C10_CUDA_CHECK(
                    cudaGraphAddChildGraphNode(
                        &child_node,
                        parent,
                        previous == nullptr ? nullptr : &previous,
                        previous == nullptr ? 0 : 1,
                        reinterpret_cast<cudaGraph_t>(
                            static_cast<uintptr_t>(raw_child))));
                previous = child_node;
            }
            cudaGraphExec_t executable = nullptr;
            C10_CUDA_CHECK(
                cudaGraphInstantiateWithFlags(
                    &executable,
                    parent,
                    0));
            owned_graphs_.push_back(parent);
            owned_graph_execs_.push_back(executable);
            graph_execs_.push_back(
                static_cast<int64_t>(
                    reinterpret_cast<uintptr_t>(executable)));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        validate_handles();
        initialize_collective_events();
    }

    TPGraphLaunchBatch(
        std::vector<int64_t> devices,
        std::vector<std::vector<std::vector<int64_t>>> graph_stages,
        std::vector<int64_t> streams,
        std::vector<int64_t> done_events,
        int64_t source_event)
        : devices_(std::move(devices)),
          streams_(std::move(streams)),
          done_events_(std::move(done_events)),
          source_event_(source_event),
          collective_event_barrier_enabled_(
              tp_environment_enabled("TPQ_TP_EVENT_BARRIER")),
          fused_moe_finalize_enabled_(
              tp_environment_enabled("TPQ_TP_FUSED_MOE_FINALIZE"))
    {
        TORCH_CHECK(
            !devices_.empty() &&
            graph_stages.size() == devices_.size() &&
            streams_.size() == devices_.size() &&
            done_events_.size() == devices_.size() &&
            source_event_ != 0,
            "TP Graph DAG handles must be non-empty and size-equal");
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        owned_graphs_.reserve(devices_.size());
        owned_graph_execs_.reserve(devices_.size());
        graph_execs_.reserve(devices_.size());
        for (size_t rank = 0; rank < devices_.size(); ++rank) {
            TORCH_CHECK(
                !graph_stages[rank].empty(),
                "each TP rank DAG needs at least one stage");
            C10_CUDA_CHECK(
                cudaSetDevice(static_cast<int>(devices_[rank])));
            cudaGraph_t parent = nullptr;
            C10_CUDA_CHECK(cudaGraphCreate(&parent, 0));
            std::vector<cudaGraphNode_t> previous;
            for (const auto& stage : graph_stages[rank]) {
                TORCH_CHECK(
                    !stage.empty(),
                    "TP Graph DAG stages must be non-empty");
                std::vector<cudaGraphNode_t> current;
                current.reserve(stage.size());
                for (const auto raw_child : stage) {
                    TORCH_CHECK(
                        raw_child != 0,
                        "TP Graph DAG child handle must be non-zero");
                    cudaGraphNode_t child_node = nullptr;
                    C10_CUDA_CHECK(
                        cudaGraphAddChildGraphNode(
                            &child_node,
                            parent,
                            previous.empty()
                                ? nullptr
                                : previous.data(),
                            previous.size(),
                            reinterpret_cast<cudaGraph_t>(
                                static_cast<uintptr_t>(raw_child))));
                    current.push_back(child_node);
                }
                previous = std::move(current);
            }
            cudaGraphExec_t executable = nullptr;
            C10_CUDA_CHECK(
                cudaGraphInstantiateWithFlags(
                    &executable,
                    parent,
                    0));
            owned_graphs_.push_back(parent);
            owned_graph_execs_.push_back(executable);
            graph_execs_.push_back(
                static_cast<int64_t>(
                    reinterpret_cast<uintptr_t>(executable)));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        validate_handles();
        initialize_collective_events();
    }

    TPGraphLaunchBatch(const TPGraphLaunchBatch&) = delete;
    TPGraphLaunchBatch& operator=(const TPGraphLaunchBatch&) = delete;

    ~TPGraphLaunchBatch()
    {
        int original_device = -1;
        if (cudaGetDevice(&original_device) != cudaSuccess)
            return;
        if (
            collective_ready_event_ != nullptr &&
            cudaSetDevice(static_cast<int>(devices_[0])) == cudaSuccess
        )
            cudaEventDestroy(collective_ready_event_);
        for (size_t index = 0; index < collective_events_.size(); ++index) {
            if (
                cudaSetDevice(static_cast<int>(devices_[index]))
                == cudaSuccess
            )
                cudaEventDestroy(collective_events_[index]);
        }
        for (size_t index = 0; index < owned_graphs_.size(); ++index) {
            if (
                cudaSetDevice(static_cast<int>(devices_[index]))
                != cudaSuccess)
                continue;
            if (owned_graph_execs_[index] != nullptr)
                cudaGraphExecDestroy(owned_graph_execs_[index]);
            if (owned_graphs_[index] != nullptr)
                cudaGraphDestroy(owned_graphs_[index]);
        }
        cudaSetDevice(original_device);
    }

    void launch() const
    {
        launch_cuda_graphs(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            source_event_);
    }

    void launch_from_events(
        std::vector<int64_t> input_events) const
    {
        TORCH_CHECK(
            input_events.size() == devices_.size(),
            "TP graph input events must match graph ranks");
        launch_cuda_graphs_from_events(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            input_events);
    }

    torch::Tensor launch_reduce(
        std::vector<torch::Tensor> contributions,
        torch::Tensor residual) const
    {
        return launch_cuda_graphs_reduce(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            source_event_,
            std::move(contributions),
            residual);
    }

    std::vector<torch::Tensor> launch_reduce_many(
        std::vector<std::vector<torch::Tensor>> contribution_groups,
        std::vector<torch::Tensor> residuals) const
    {
        return launch_cuda_graphs_reduce_many(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            source_event_,
            std::move(contribution_groups),
            std::move(residuals));
    }

    std::vector<torch::Tensor> launch_all_rank(
        std::vector<torch::Tensor> contributions,
        std::vector<torch::Tensor> outputs) const
    {
        launch_cuda_graphs(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            source_event_);
        TORCH_CHECK(
            outputs.size() == devices_.size() &&
            collective_events_.size() == devices_.size(),
            "TP all-rank outputs must match graph rank count");
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        TORCH_CHECK(
            original_device == static_cast<int>(devices_[0]),
            "TP all-rank launch must begin on the primary graph rank");
        const bool event_barrier = collective_event_barrier_enabled();
        if (event_barrier)
            record_collective_ready(done_events_);
        for (size_t rank = 0; rank < devices_.size(); ++rank) {
            const int target = static_cast<int>(devices_[rank]);
            TORCH_CHECK(
                outputs[rank].get_device() == target,
                "TP all-rank output device order must match graph ranks");
            C10_CUDA_CHECK(cudaSetDevice(target));
            const auto stream =
                at::cuda::getCurrentCUDAStream(target);
            if (event_barrier) {
                C10_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        collective_ready_event_,
                        0));
            } else {
                for (const auto raw_event : done_events_) {
                    const auto done = reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(raw_event));
                    C10_CUDA_CHECK(
                        cudaStreamWaitEvent(stream, done, 0));
                }
            }
            launch_tp_all_rank_reduce_one(
                contributions,
                outputs[rank],
                stream);
            C10_CUDA_CHECK(
                cudaEventRecord(
                    collective_events_[rank],
                    stream));
        }
        C10_CUDA_CHECK(
            cudaSetDevice(static_cast<int>(devices_[0])));
        const auto primary_stream =
            at::cuda::getCurrentCUDAStream(
                static_cast<int>(devices_[0]));
        for (const auto event : collective_events_)
            C10_CUDA_CHECK(
                cudaStreamWaitEvent(primary_stream, event, 0));
        return outputs;
    }

    std::vector<torch::Tensor> launch_all_rank_from_events(
        std::vector<int64_t> input_events,
        std::vector<torch::Tensor> contributions,
        std::vector<torch::Tensor> outputs,
        std::vector<int64_t> output_events) const
    {
        TORCH_CHECK(
            input_events.size() == devices_.size() &&
            !outputs.empty() &&
            outputs.size() <= 16 &&
            output_events.size() == outputs.size(),
            "TPHidden inputs must match graph ranks and outputs/events "
            "must form a non-empty rank set");
        launch_cuda_graphs_from_events(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            input_events);
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        const bool event_barrier = collective_event_barrier_enabled();
        if (event_barrier)
            record_collective_ready(done_events_);
        for (size_t rank = 0; rank < outputs.size(); ++rank) {
            const int target = outputs[rank].get_device();
            C10_CUDA_CHECK(cudaSetDevice(target));
            const auto stream =
                at::cuda::getCurrentCUDAStream(target);
            if (event_barrier) {
                C10_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        collective_ready_event_,
                        0));
            } else {
                for (const auto raw_event : done_events_) {
                    const auto done = reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(raw_event));
                    C10_CUDA_CHECK(
                        cudaStreamWaitEvent(stream, done, 0));
                }
            }
            launch_tp_all_rank_reduce_one(
                contributions,
                outputs[rank],
                stream);
            C10_CUDA_CHECK(
                cudaEventRecord(
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(
                            output_events[rank])),
                    stream));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        return outputs;
    }

    std::vector<std::vector<torch::Tensor>>
    launch_all_rank_many_from_events(
        std::vector<int64_t> input_events,
        std::vector<std::vector<torch::Tensor>> contribution_groups,
        std::vector<std::vector<torch::Tensor>> output_groups,
        std::vector<int64_t> output_events) const
    {
        TORCH_CHECK(
            input_events.size() == devices_.size() &&
            !contribution_groups.empty() &&
            contribution_groups.size() == output_groups.size() &&
            output_events.size() == devices_.size(),
            "TPHidden multi-output collective metadata mismatch");
        for (size_t group = 0; group < contribution_groups.size(); ++group) {
            TORCH_CHECK(
                contribution_groups[group].size() == devices_.size() &&
                output_groups[group].size() == devices_.size(),
                "TPHidden multi-output groups must match graph ranks");
        }
        launch_cuda_graphs_from_events(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            input_events);
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        const bool event_barrier = collective_event_barrier_enabled();
        if (event_barrier)
            record_collective_ready(done_events_);
        for (size_t rank = 0; rank < devices_.size(); ++rank) {
            const int target = static_cast<int>(devices_[rank]);
            C10_CUDA_CHECK(cudaSetDevice(target));
            const auto stream =
                at::cuda::getCurrentCUDAStream(target);
            if (event_barrier) {
                C10_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        collective_ready_event_,
                        0));
            } else {
                for (const auto raw_event : done_events_) {
                    const auto done = reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(raw_event));
                    C10_CUDA_CHECK(
                        cudaStreamWaitEvent(stream, done, 0));
                }
            }
            for (
                size_t group = 0;
                group < contribution_groups.size();
                ++group
            ) {
                TORCH_CHECK(
                    output_groups[group][rank].get_device() == target,
                    "TPHidden multi-output device order must match ranks");
                launch_tp_all_rank_reduce_one(
                    contribution_groups[group],
                    output_groups[group][rank],
                    stream);
            }
            C10_CUDA_CHECK(
                cudaEventRecord(
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(
                            output_events[rank])),
                    stream));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        return output_groups;
    }

    std::vector<std::vector<torch::Tensor>>
    reduce_all_rank_many_from_events(
        std::vector<int64_t> input_events,
        std::vector<std::vector<torch::Tensor>> contribution_groups,
        std::vector<std::vector<torch::Tensor>> output_groups,
        std::vector<int64_t> output_events) const
    {
        TORCH_CHECK(
            input_events.size() == devices_.size() &&
            !contribution_groups.empty() &&
            contribution_groups.size() == output_groups.size() &&
            output_events.size() == devices_.size(),
            "TPHidden collective-only metadata mismatch");
        for (size_t group = 0; group < contribution_groups.size(); ++group) {
            TORCH_CHECK(
                contribution_groups[group].size() == devices_.size() &&
                output_groups[group].size() == devices_.size(),
                "TPHidden collective-only groups must match graph ranks");
        }
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        const bool event_barrier = collective_event_barrier_enabled();
        if (event_barrier)
            record_collective_ready(input_events);
        for (size_t rank = 0; rank < devices_.size(); ++rank) {
            const int target = static_cast<int>(devices_[rank]);
            C10_CUDA_CHECK(cudaSetDevice(target));
            const auto stream =
                at::cuda::getCurrentCUDAStream(target);
            if (event_barrier) {
                C10_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        collective_ready_event_,
                        0));
            } else {
                for (const auto raw_event : input_events) {
                    C10_CUDA_CHECK(
                        cudaStreamWaitEvent(
                            stream,
                            reinterpret_cast<cudaEvent_t>(
                                static_cast<uintptr_t>(raw_event)),
                            0));
                }
            }
            for (
                size_t group = 0;
                group < contribution_groups.size();
                ++group
            ) {
                TORCH_CHECK(
                    output_groups[group][rank].get_device() == target,
                    "TPHidden collective-only device order must match ranks");
                launch_tp_all_rank_reduce_one(
                    contribution_groups[group],
                    output_groups[group][rank],
                    stream);
            }
            C10_CUDA_CHECK(
                cudaEventRecord(
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(
                            output_events[rank])),
                    stream));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        return output_groups;
    }

    std::vector<torch::Tensor> launch_moe_all_rank_from_events(
        std::vector<int64_t> input_events,
        std::vector<torch::Tensor> routed_contributions,
        std::vector<torch::Tensor> shared_contributions,
        std::vector<int64_t> shared_events,
        std::vector<torch::Tensor> residuals,
        std::vector<int64_t> residual_events,
        std::vector<torch::Tensor> routed_workspaces,
        std::vector<torch::Tensor> shared_workspaces,
        std::vector<torch::Tensor> outputs,
        std::vector<int64_t> output_events) const
    {
        TORCH_CHECK(
            input_events.size() == devices_.size() &&
            shared_events.size() == devices_.size() &&
            !outputs.empty() &&
            outputs.size() <= 16 &&
            residuals.size() == outputs.size() &&
            residual_events.size() == outputs.size() &&
            routed_workspaces.size() == outputs.size() &&
            shared_workspaces.size() == outputs.size() &&
            output_events.size() == outputs.size(),
            "TP MoE finalizer input/output ranks are inconsistent");
        launch_cuda_graphs_from_events(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            input_events);
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        const bool event_barrier = collective_event_barrier_enabled();
        if (event_barrier)
            record_collective_ready(done_events_, &shared_events);
        for (size_t rank = 0; rank < outputs.size(); ++rank) {
            const int target = outputs[rank].get_device();
            C10_CUDA_CHECK(cudaSetDevice(target));
            const auto stream =
                at::cuda::getCurrentCUDAStream(target);
            if (event_barrier) {
                C10_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        collective_ready_event_,
                        0));
            } else {
                for (const auto raw_event : done_events_) {
                    C10_CUDA_CHECK(
                        cudaStreamWaitEvent(
                            stream,
                            reinterpret_cast<cudaEvent_t>(
                                static_cast<uintptr_t>(raw_event)),
                            0));
                }
                for (const auto raw_event : shared_events) {
                    C10_CUDA_CHECK(
                        cudaStreamWaitEvent(
                            stream,
                            reinterpret_cast<cudaEvent_t>(
                                static_cast<uintptr_t>(raw_event)),
                            0));
                }
            }
            C10_CUDA_CHECK(
                cudaStreamWaitEvent(
                    stream,
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(
                            residual_events[rank])),
                    0));
            launch_tp_moe_finalize_one(
                routed_contributions,
                shared_contributions,
                residuals[rank],
                routed_workspaces[rank],
                shared_workspaces[rank],
                outputs[rank],
                stream,
                fused_moe_finalize_enabled_);
            C10_CUDA_CHECK(
                cudaEventRecord(
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(
                            output_events[rank])),
                    stream));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        return outputs;
    }

    std::vector<torch::Tensor> launch_reduce_norm_router(
        std::vector<torch::Tensor> contributions,
        torch::Tensor attention_zero,
        torch::Tensor residual,
        torch::Tensor norm_weight,
        torch::Tensor router_weight,
        double eps,
        torch::Tensor norm_output,
        c10::optional<torch::Tensor> residual_output,
        c10::optional<torch::Tensor> logits_output) const
    {
        return launch_cuda_graphs_reduce_norm_router(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            source_event_,
            std::move(contributions),
            attention_zero,
            residual,
            norm_weight,
            router_weight,
            eps,
            norm_output,
            residual_output,
            logits_output);
    }

    torch::Tensor launch_moe_layer(
        const TPGraphLaunchBatch& expert_batch,
        std::vector<torch::Tensor> attention_contributions,
        torch::Tensor attention_zero,
        torch::Tensor residual,
        torch::Tensor norm_weight,
        torch::Tensor router_weight,
        double eps,
        torch::Tensor norm_output,
        torch::Tensor residual_output,
        torch::Tensor logits_output,
        torch::Tensor route_bias,
        torch::Tensor route_mask,
        long top_k,
        double routed_scaling,
        torch::Tensor route_weights,
        torch::Tensor route_indices,
        std::vector<torch::Tensor> expert_contributions) const
    {
        auto post = launch_reduce_norm_router(
            std::move(attention_contributions),
            attention_zero,
            residual,
            norm_weight,
            router_weight,
            eps,
            norm_output,
            residual_output,
            logits_output);
        sigmoid_route_out(
            post[2],
            route_bias,
            route_mask,
            top_k,
            routed_scaling,
            route_weights,
            route_indices);
        return expert_batch.launch_reduce(
            std::move(expert_contributions),
            post[0]);
    }

private:
    friend class TPNoOwnerDecodeLayerPlan;

    bool collective_event_barrier_enabled() const
    {
        return collective_event_barrier_enabled_;
    }

    void record_collective_ready(
        const std::vector<int64_t>& events,
        const std::vector<int64_t>* more_events = nullptr) const
    {
        // This event is scheduling metadata only.  It coalesces N identical
        // wait lists into one cross-device wait per output rank; every rank
        // still reduces every contribution into its own TPHidden replica.
        const int primary = static_cast<int>(devices_[0]);
        C10_CUDA_CHECK(cudaSetDevice(primary));
        const auto stream =
            at::cuda::getCurrentCUDAStream(primary);
        for (const auto raw_event : events) {
            C10_CUDA_CHECK(
                cudaStreamWaitEvent(
                    stream,
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(raw_event)),
                    0));
        }
        if (more_events != nullptr) {
            for (const auto raw_event : *more_events) {
                C10_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        reinterpret_cast<cudaEvent_t>(
                            static_cast<uintptr_t>(raw_event)),
                        0));
            }
        }
        C10_CUDA_CHECK(
            cudaEventRecord(
                collective_ready_event_,
                stream));
    }

    void validate_handles() const
    {
        TORCH_CHECK(
            !devices_.empty() &&
            graph_execs_.size() == devices_.size() &&
            streams_.size() == devices_.size() &&
            done_events_.size() == devices_.size() &&
            source_event_ != 0,
            "TP Graph batch handles must be non-empty and size-equal");
    }

    void initialize_collective_events()
    {
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        C10_CUDA_CHECK(
            cudaSetDevice(static_cast<int>(devices_[0])));
        C10_CUDA_CHECK(
            cudaEventCreateWithFlags(
                &collective_ready_event_,
                cudaEventDisableTiming));
        collective_events_.reserve(devices_.size());
        for (const auto raw_device : devices_) {
            C10_CUDA_CHECK(
                cudaSetDevice(static_cast<int>(raw_device)));
            cudaEvent_t event = nullptr;
            C10_CUDA_CHECK(
                cudaEventCreateWithFlags(
                    &event,
                    cudaEventDisableTiming));
            collective_events_.push_back(event);
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
    }

    std::vector<int64_t> devices_;
    std::vector<int64_t> graph_execs_;
    std::vector<int64_t> streams_;
    std::vector<int64_t> done_events_;
    int64_t source_event_;
    bool collective_event_barrier_enabled_;
    bool fused_moe_finalize_enabled_;
    cudaEvent_t collective_ready_event_ = nullptr;
    std::vector<cudaEvent_t> collective_events_;
    std::vector<cudaGraph_t> owned_graphs_;
    std::vector<cudaGraphExec_t> owned_graph_execs_;
};

class TPNoOwnerMoELayerPlan {
public:
    TPNoOwnerMoELayerPlan(
        const TPGraphLaunchBatch& shared_batch,
        const TPGraphLaunchBatch& route_batch,
        const TPGraphLaunchBatch& expert_batch,
        const TPGraphLaunchBatch& final_batch,
        std::vector<int64_t> input_events,
        std::vector<std::vector<torch::Tensor>> route_contribution_groups,
        std::vector<std::vector<torch::Tensor>> route_output_groups,
        std::vector<int64_t> route_output_events,
        std::vector<torch::Tensor> expert_contributions,
        std::vector<torch::Tensor> packed_outputs,
        std::vector<int64_t> packed_output_events,
        std::vector<torch::Tensor> routed_contributions,
        std::vector<torch::Tensor> shared_contributions,
        std::vector<int64_t> shared_events,
        std::vector<torch::Tensor> residuals,
        std::vector<int64_t> residual_events,
        std::vector<torch::Tensor> routed_workspaces,
        std::vector<torch::Tensor> shared_workspaces,
        std::vector<torch::Tensor> outputs,
        std::vector<int64_t> output_events)
        : shared_batch_(&shared_batch),
          route_batch_(&route_batch),
          expert_batch_(&expert_batch),
          final_batch_(&final_batch),
          input_events_(std::move(input_events)),
          route_contribution_groups_(
              std::move(route_contribution_groups)),
          route_output_groups_(std::move(route_output_groups)),
          route_output_events_(std::move(route_output_events)),
          expert_contributions_(std::move(expert_contributions)),
          packed_outputs_(std::move(packed_outputs)),
          packed_output_events_(std::move(packed_output_events)),
          routed_contributions_(std::move(routed_contributions)),
          shared_contributions_(std::move(shared_contributions)),
          shared_events_(std::move(shared_events)),
          residuals_(std::move(residuals)),
          residual_events_(std::move(residual_events)),
          routed_workspaces_(std::move(routed_workspaces)),
          shared_workspaces_(std::move(shared_workspaces)),
          outputs_(std::move(outputs)),
          output_events_(std::move(output_events))
    {
        const auto ranks = input_events_.size();
        TORCH_CHECK(
            ranks > 0 &&
            route_contribution_groups_.size() == 2 &&
            route_output_groups_.size() == 2 &&
            route_output_events_.size() == ranks &&
            expert_contributions_.size() == ranks &&
            packed_outputs_.size() == ranks &&
            packed_output_events_.size() == ranks &&
            routed_contributions_.size() == ranks &&
            shared_contributions_.size() == ranks &&
            shared_events_.size() == ranks &&
            residuals_.size() == ranks &&
            residual_events_.size() == ranks &&
            routed_workspaces_.size() == ranks &&
            shared_workspaces_.size() == ranks &&
            outputs_.size() == ranks &&
            output_events_.size() == ranks,
            "no-owner MoE plan ranks and fixed buffers must match");
        for (size_t group = 0; group < 2; ++group) {
            TORCH_CHECK(
                route_contribution_groups_[group].size() == ranks &&
                route_output_groups_[group].size() == ranks,
                "no-owner MoE route groups must match TP ranks");
        }
    }

    void launch() const
    {
        launch_from_events(input_events_);
    }

    void launch_from_events(
        std::vector<int64_t> input_events) const
    {
        TORCH_CHECK(
            input_events.size() == input_events_.size(),
            "profiled no-owner MoE inputs must match TP ranks");
        // One Python→C++ transition schedules the complete fixed-address
        // no-owner MoE chain.  Every phase is still all-rank: the event
        // boundaries only express true TP collective dependencies.
        shared_batch_->launch_from_events(std::move(input_events));
        route_batch_->reduce_all_rank_many_from_events(
            shared_events_,
            route_contribution_groups_,
            route_output_groups_,
            route_output_events_);
        expert_batch_->launch_all_rank_from_events(
            route_output_events_,
            expert_contributions_,
            packed_outputs_,
            packed_output_events_);
        final_batch_->launch_moe_all_rank_from_events(
            packed_output_events_,
            routed_contributions_,
            shared_contributions_,
            shared_events_,
            residuals_,
            residual_events_,
            routed_workspaces_,
            shared_workspaces_,
            outputs_,
            output_events_);
    }

private:
    friend class TPNoOwnerDecodeLayerPlan;

    // The Python wrapper retains the four owning TPGraphLaunchBatch objects.
    // These pointers therefore only describe immutable scheduling metadata.
    const TPGraphLaunchBatch* shared_batch_;
    const TPGraphLaunchBatch* route_batch_;
    const TPGraphLaunchBatch* expert_batch_;
    const TPGraphLaunchBatch* final_batch_;
    std::vector<int64_t> input_events_;
    std::vector<std::vector<torch::Tensor>> route_contribution_groups_;
    std::vector<std::vector<torch::Tensor>> route_output_groups_;
    std::vector<int64_t> route_output_events_;
    std::vector<torch::Tensor> expert_contributions_;
    std::vector<torch::Tensor> packed_outputs_;
    std::vector<int64_t> packed_output_events_;
    std::vector<torch::Tensor> routed_contributions_;
    std::vector<torch::Tensor> shared_contributions_;
    std::vector<int64_t> shared_events_;
    std::vector<torch::Tensor> residuals_;
    std::vector<int64_t> residual_events_;
    std::vector<torch::Tensor> routed_workspaces_;
    std::vector<torch::Tensor> shared_workspaces_;
    std::vector<torch::Tensor> outputs_;
    std::vector<int64_t> output_events_;
};

class TPNoOwnerDecodeLayerPlan {
public:
    TPNoOwnerDecodeLayerPlan(
        const TPGraphLaunchBatch& attention_batch,
        const TPNoOwnerMoELayerPlan& moe_plan,
        std::vector<torch::Tensor> attention_contributions,
        std::vector<torch::Tensor> attention_outputs,
        std::vector<int64_t> attention_output_events)
        : attention_batch_(&attention_batch),
          moe_plan_(&moe_plan),
          attention_contributions_(
              std::move(attention_contributions)),
          attention_outputs_(std::move(attention_outputs)),
          attention_output_events_(
              std::move(attention_output_events))
    {
        TORCH_CHECK(
            !attention_contributions_.empty() &&
            !attention_outputs_.empty() &&
            attention_outputs_.size()
                == attention_output_events_.size() &&
            attention_outputs_.size()
                == attention_batch_->devices_.size() &&
            moe_plan_->input_events_.size()
                == attention_batch_->devices_.size(),
            "no-owner decode plan attention metadata is incomplete");
        persistent_enabled_ = (
            tp_environment_enabled(
                "TPQ_TP_PERSISTENT_LAYER_PLAN") &&
            moe_plan_->final_batch_
                ->fused_moe_finalize_enabled_ &&
            attention_batch_->devices_.size() > 1 &&
            attention_batch_->devices_.size()
                <= kMaxGraphDispatchDevices);
        if (persistent_enabled_) {
            stage_barrier_ = std::make_unique<TPStageBarrier>(
                attention_batch_->devices_.size());
        }
    }

    void launch_from_events(
        std::vector<int64_t> input_events) const
    {
        TORCH_CHECK(
            input_events.size()
                == attention_batch_->devices_.size(),
            "decode layer input events must match TP ranks");
        if (persistent_enabled_) {
            launch_persistent(std::move(input_events));
            return;
        }
        // One Python→C++ transition now submits the complete routed layer:
        // Attention Column/Head-TP→Row-TP followed by the fixed all-rank
        // MoE plan.  The attention output events are the only dependency
        // between the two all-rank stages; no hidden owner or broadcast is
        // introduced.
        attention_batch_->launch_all_rank_from_events(
            std::move(input_events),
            attention_contributions_,
            attention_outputs_,
            attention_output_events_);
        moe_plan_->launch_from_events(attention_output_events_);
    }

    bool persistent_enabled() const
    {
        return persistent_enabled_;
    }

private:
    static void merge_status(
        cudaError_t& status,
        cudaError_t candidate)
    {
        if (status == cudaSuccess && candidate != cudaSuccess)
            status = candidate;
    }

    static cudaError_t persistent_rank_callback(
        const void* context,
        size_t rank,
        cudaEvent_t ready)
    {
        return static_cast<const TPNoOwnerDecodeLayerPlan*>(
            context)->launch_persistent_rank(rank, ready);
    }

    void publish_stage(
        size_t rank,
        const std::vector<int64_t>& local_events,
        cudaEvent_t ready_event,
        cudaStream_t stream,
        cudaError_t& status) const
    {
        stage_barrier_->arrive_and_wait();
        if (rank == 0) {
            for (const auto raw_event : local_events) {
                merge_status(
                    status,
                    cudaStreamWaitEvent(
                        stream,
                        reinterpret_cast<cudaEvent_t>(
                            static_cast<uintptr_t>(raw_event)),
                        0));
            }
            merge_status(
                status,
                cudaEventRecord(ready_event, stream));
        }
        stage_barrier_->arrive_and_wait();
        merge_status(
            status,
            cudaStreamWaitEvent(stream, ready_event, 0));
    }

    cudaError_t launch_persistent_rank(
        size_t rank,
        cudaEvent_t input_event) const
    {
        cudaError_t status = cudaSetDevice(
            static_cast<int>(
                attention_batch_->devices_[rank]));

        const auto attention_stream =
            reinterpret_cast<cudaStream_t>(
                static_cast<uintptr_t>(
                    attention_batch_->streams_[rank]));
        merge_status(
            status,
            cudaStreamWaitEvent(
                attention_stream,
                input_event,
                0));
        merge_status(
            status,
            cudaGraphLaunch(
                reinterpret_cast<cudaGraphExec_t>(
                    static_cast<uintptr_t>(
                        attention_batch_->graph_execs_[rank])),
                attention_stream));
        merge_status(
            status,
            cudaEventRecord(
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(
                        attention_batch_->done_events_[rank])),
                attention_stream));
        publish_stage(
            rank,
            attention_batch_->done_events_,
            attention_batch_->collective_ready_event_,
            attention_stream,
            status);
        launch_tp_all_rank_reduce_one(
            attention_contributions_,
            attention_outputs_[rank],
            attention_stream);
        merge_status(
            status,
            cudaEventRecord(
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(
                        attention_output_events_[rank])),
                attention_stream));

        const auto shared_batch = moe_plan_->shared_batch_;
        const auto route_batch = moe_plan_->route_batch_;
        const auto shared_stream =
            reinterpret_cast<cudaStream_t>(
                static_cast<uintptr_t>(
                    shared_batch->streams_[rank]));
        merge_status(
            status,
            cudaStreamWaitEvent(
                shared_stream,
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(
                        attention_output_events_[rank])),
                0));
        merge_status(
            status,
            cudaGraphLaunch(
                reinterpret_cast<cudaGraphExec_t>(
                    static_cast<uintptr_t>(
                        shared_batch->graph_execs_[rank])),
                shared_stream));
        const auto shared_event =
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(
                    moe_plan_->shared_events_[rank]));
        merge_status(
            status,
            cudaEventRecord(shared_event, shared_stream));
        if (
            shared_batch->done_events_[rank]
            != moe_plan_->shared_events_[rank]
        ) {
            merge_status(
                status,
                cudaEventRecord(
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(
                            shared_batch->done_events_[rank])),
                    shared_stream));
        }

        const auto route_stream =
            reinterpret_cast<cudaStream_t>(
                static_cast<uintptr_t>(
                    route_batch->streams_[rank]));
        publish_stage(
            rank,
            moe_plan_->shared_events_,
            route_batch->collective_ready_event_,
            route_stream,
            status);
        for (
            size_t group = 0;
            group < moe_plan_->route_contribution_groups_.size();
            ++group
        ) {
            launch_tp_all_rank_reduce_one(
                moe_plan_->route_contribution_groups_[group],
                moe_plan_->route_output_groups_[group][rank],
                route_stream);
        }
        merge_status(
            status,
            cudaEventRecord(
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(
                        moe_plan_->route_output_events_[rank])),
                route_stream));

        const auto expert_batch = moe_plan_->expert_batch_;
        const auto expert_stream =
            reinterpret_cast<cudaStream_t>(
                static_cast<uintptr_t>(
                    expert_batch->streams_[rank]));
        merge_status(
            status,
            cudaStreamWaitEvent(
                expert_stream,
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(
                        moe_plan_->route_output_events_[rank])),
                0));
        merge_status(
            status,
            cudaGraphLaunch(
                reinterpret_cast<cudaGraphExec_t>(
                    static_cast<uintptr_t>(
                        expert_batch->graph_execs_[rank])),
                expert_stream));
        merge_status(
            status,
            cudaEventRecord(
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(
                        expert_batch->done_events_[rank])),
                expert_stream));
        publish_stage(
            rank,
            expert_batch->done_events_,
            expert_batch->collective_ready_event_,
            expert_stream,
            status);
        launch_tp_all_rank_reduce_one(
            moe_plan_->expert_contributions_,
            moe_plan_->packed_outputs_[rank],
            expert_stream);
        merge_status(
            status,
            cudaEventRecord(
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(
                        moe_plan_->packed_output_events_[rank])),
                expert_stream));

        const auto final_batch = moe_plan_->final_batch_;
        const auto final_stream =
            reinterpret_cast<cudaStream_t>(
                static_cast<uintptr_t>(
                    final_batch->streams_[rank]));
        merge_status(
            status,
            cudaStreamWaitEvent(
                final_stream,
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(
                        moe_plan_->packed_output_events_[rank])),
                0));
        merge_status(
            status,
            cudaGraphLaunch(
                reinterpret_cast<cudaGraphExec_t>(
                    static_cast<uintptr_t>(
                        final_batch->graph_execs_[rank])),
                final_stream));
        merge_status(
            status,
            cudaEventRecord(
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(
                        final_batch->done_events_[rank])),
                final_stream));
        publish_stage(
            rank,
            final_batch->done_events_,
            final_batch->collective_ready_event_,
            final_stream,
            status);
        merge_status(
            status,
            cudaStreamWaitEvent(
                final_stream,
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(
                        moe_plan_->residual_events_[rank])),
                0));
        launch_tp_moe_finalize_one(
            moe_plan_->routed_contributions_,
            moe_plan_->shared_contributions_,
            moe_plan_->residuals_[rank],
            moe_plan_->routed_workspaces_[rank],
            moe_plan_->shared_workspaces_[rank],
            moe_plan_->outputs_[rank],
            final_stream,
            true);
        merge_status(
            status,
            cudaEventRecord(
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(
                        moe_plan_->output_events_[rank])),
                final_stream));
        return status;
    }

    void launch_persistent(
        std::vector<int64_t> input_events) const
    {
        const size_t ranks = input_events.size();
        std::array<GraphLaunchWorker*, kMaxGraphDispatchDevices>
            workers{};
        std::array<uint64_t, kMaxGraphDispatchDevices> sequences{};
        for (size_t rank = 1; rank < ranks; ++rank) {
            const int device = static_cast<int>(
                attention_batch_->devices_[rank]);
            GraphLaunchWorker& worker =
                graph_launch_worker(device);
            workers[rank] = &worker;
            GraphLaunchTask task;
            task.ready = reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(input_events[rank]));
            task.callback = &persistent_rank_callback;
            task.callback_context = this;
            task.callback_rank = rank;
            sequences[rank] = worker.submit(task);
        }
        const auto primary_status = launch_persistent_rank(
            0,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(input_events[0])));
        for (size_t rank = 1; rank < ranks; ++rank)
            workers[rank]->wait(sequences[rank]);
        C10_CUDA_CHECK(primary_status);
    }

    const TPGraphLaunchBatch* attention_batch_;
    const TPNoOwnerMoELayerPlan* moe_plan_;
    std::vector<torch::Tensor> attention_contributions_;
    std::vector<torch::Tensor> attention_outputs_;
    std::vector<int64_t> attention_output_events_;
    bool persistent_enabled_ = false;
    mutable std::unique_ptr<TPStageBarrier> stage_barrier_;
};

template <typename output_t>
__global__ void bf16_gemv_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    output_t* __restrict__ output,
    const int rows,
    const int cols)
{
    const int lane = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows)
        return;
    const auto* input2 =
        reinterpret_cast<const __nv_bfloat162*>(input);
    const auto* weight2 =
        reinterpret_cast<const __nv_bfloat162*>(
            weight + static_cast<long>(row) * cols);
    const int pairs = cols >> 1;
    float sum = 0.f;
    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 x = __bfloat1622float2(__ldg(input2 + pair));
        const float2 w = __bfloat1622float2(__ldg(weight2 + pair));
        sum = __fmaf_rn(x.x, w.x, sum);
        sum = __fmaf_rn(x.y, w.y, sum);
    }
    sum = warp_sum_f32(sum);
    if (lane == 0) {
        if constexpr (std::is_same_v<output_t, float>)
            output[row] = sum;
        else
            output[row] = __float2bfloat16_rn(sum);
    }
}

torch::Tensor bf16_gemv_out(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output)
{
    TORCH_CHECK(
        input.is_cuda() &&
        weight.is_cuda() &&
        output.is_cuda() &&
        input.scalar_type() == at::kBFloat16 &&
        weight.scalar_type() == at::kBFloat16 &&
        (
            output.scalar_type() == at::kBFloat16 ||
            output.scalar_type() == at::kFloat
        ),
        "BF16 GEMV requires CUDA BF16 input/weight and BF16/FP32 output");
    TORCH_CHECK(
        input.dim() == 2 &&
        input.size(0) == 1 &&
        weight.dim() == 2 &&
        weight.size(1) == input.size(1) &&
        input.size(1) > 0 &&
        input.size(1) % 2 == 0 &&
        output.dim() == 2 &&
        output.size(0) == 1 &&
        output.size(1) == weight.size(0) &&
        input.is_contiguous() &&
        weight.is_contiguous() &&
        output.is_contiguous() &&
        input.get_device() == weight.get_device() &&
        input.get_device() == output.get_device(),
        "BF16 GEMV tensor shapes/layout/devices are inconsistent");
    constexpr int warps = 8;
    const int rows = static_cast<int>(weight.size(0));
    const int cols = static_cast<int>(weight.size(1));
    const int blocks = (rows + warps - 1) / warps;
    const dim3 threads(32, warps);
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (output.scalar_type() == at::kFloat) {
        bf16_gemv_kernel<float><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                weight.data_ptr<at::BFloat16>()),
            output.data_ptr<float>(),
            rows,
            cols);
    } else {
        bf16_gemv_kernel<__nv_bfloat16><<<
            blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    input.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    weight.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(
                    output.data_ptr<at::BFloat16>()),
                rows,
                cols);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor int4_swiglu_packed_f32(
    torch::Tensor x,
    torch::Tensor gate_packed,
    torch::Tensor gate_scales,
    torch::Tensor up_packed,
    torch::Tensor up_scales,
    long cols,
    long group_size,
    bool group_vector,
    c10::optional<torch::Tensor> output_buffer) {
    TORCH_CHECK(
        x.is_cuda() && gate_packed.is_cuda() &&
        gate_scales.is_cuda() && up_packed.is_cuda() &&
        up_scales.is_cuda(),
        "INT4 SwiGLU tensors must be CUDA");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat ||
        x.scalar_type() == at::kBFloat16,
        "INT4 SwiGLU input must be float32 or bfloat16");
    TORCH_CHECK(
        gate_packed.scalar_type() == at::kByte &&
        up_packed.scalar_type() == at::kByte,
        "INT4 SwiGLU packed weights must be uint8");
    TORCH_CHECK(
        gate_scales.scalar_type() == at::kHalf &&
        up_scales.scalar_type() == at::kHalf,
        "INT4 SwiGLU scales must be float16");
    TORCH_CHECK(
        x.dim() == 2 && x.size(0) == 1,
        "INT4 SwiGLU input must be [1,C]");
    TORCH_CHECK(
        gate_packed.dim() == 2 && up_packed.dim() == 2 &&
        gate_scales.dim() == 2 && up_scales.dim() == 2,
        "INT4 SwiGLU weights and scales must be matrices");
    TORCH_CHECK(
        group_size == 64 && cols > 0 && cols % 64 == 0,
        "INT4 SwiGLU requires positive g64-aligned columns");
    TORCH_CHECK(
        x.size(1) == cols &&
        gate_packed.sizes() == up_packed.sizes() &&
        gate_scales.sizes() == up_scales.sizes() &&
        gate_packed.size(1) * 2 == cols,
        "INT4 SwiGLU input/weight shape mismatch");
    const int rows = static_cast<int>(gate_packed.size(0));
    const int groups = static_cast<int>(cols / group_size);
    TORCH_CHECK(
        gate_scales.size(0) == rows &&
        gate_scales.size(1) == groups,
        "INT4 SwiGLU scale shape mismatch");
    TORCH_CHECK(
        x.get_device() == gate_packed.get_device() &&
        x.get_device() == gate_scales.get_device() &&
        x.get_device() == up_packed.get_device() &&
        x.get_device() == up_scales.get_device(),
        "INT4 SwiGLU tensors must be on one CUDA device");

    auto xc = x.contiguous();
    auto gate_q = gate_packed.contiguous();
    auto gate_s = gate_scales.contiguous();
    auto up_q = up_packed.contiguous();
    auto up_s = up_scales.contiguous();
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {1, rows},
            x.options().dtype(at::kFloat));
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kFloat &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({1, rows}) &&
        output.get_device() == x.get_device(),
        "INT4 SwiGLU output buffer must be contiguous float32 [1,R] "
        "on the input device");
    const int device = x.get_device();
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kFloat) {
        launch_int4_swiglu_packed_f32<float>(
            xc.data_ptr<float>(),
            gate_q.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(
                gate_s.data_ptr<at::Half>()),
            up_q.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(
                up_s.data_ptr<at::Half>()),
            output.data_ptr<float>(),
            rows,
            static_cast<int>(cols),
            groups,
            device,
            stream,
            group_vector);
    } else {
        launch_int4_swiglu_packed_f32<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(
                xc.data_ptr<at::BFloat16>()),
            gate_q.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(
                gate_s.data_ptr<at::Half>()),
            up_q.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(
                up_s.data_ptr<at::Half>()),
            output.data_ptr<float>(),
            rows,
            static_cast<int>(cols),
            groups,
            device,
            stream,
            group_vector);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void flashinfer_mla_batch1_plan_kernel(
    uint8_t* __restrict__ workspace,
    int32_t* __restrict__ kv_indptr,
    int32_t* __restrict__ kv_indices,
    int32_t* __restrict__ kv_len_arr,
    int length,
    int page_size,
    int heads,
    int num_sms,
    int64_t q_indptr_offset,
    int64_t kv_indptr_offset,
    int64_t partial_indptr_offset,
    int64_t merge_start_offset,
    int64_t merge_end_offset,
    int64_t merge_partial_start_offset,
    int64_t merge_partial_end_offset,
    int64_t merge_stride_offset,
    int64_t q_len_offset,
    int64_t kv_len_offset,
    int64_t q_start_offset,
    int64_t kv_start_offset,
    int64_t kv_end_offset,
    int64_t work_indptr_offset) {
    int avg_kv = (length + num_sms - 1) / num_sms;
    int kv_limit;
    if (avg_kv <= 8) {
        kv_limit = 32;
    } else if (avg_kv <= 16) {
        kv_limit = 64;
    } else if (avg_kv <= 32) {
        kv_limit = 128;
    } else if (avg_kv <= 64) {
        kv_limit = 192;
    } else {
        kv_limit = ((avg_kv + 255) / 256) * 256;
    }
    const bool split = length > kv_limit;
    const int num_works = (length + kv_limit - 1) / kv_limit;
    const int quotient = max(length / kv_limit, 1);
    const int row_chunk = (heads + quotient - 1) / quotient;
    const int merge_count = split
        ? (heads + row_chunk - 1) / row_chunk
        : 0;
    const int blocks = (length + page_size - 1) / page_size;

    int32_t* q_indptr_plan = reinterpret_cast<int32_t*>(
        workspace + q_indptr_offset);
    int32_t* kv_indptr_plan = reinterpret_cast<int32_t*>(
        workspace + kv_indptr_offset);
    int32_t* partial_indptr = reinterpret_cast<int32_t*>(
        workspace + partial_indptr_offset);
    int32_t* merge_start = reinterpret_cast<int32_t*>(
        workspace + merge_start_offset);
    int32_t* merge_end = reinterpret_cast<int32_t*>(
        workspace + merge_end_offset);
    int32_t* merge_partial_start = reinterpret_cast<int32_t*>(
        workspace + merge_partial_start_offset);
    int32_t* merge_partial_end = reinterpret_cast<int32_t*>(
        workspace + merge_partial_end_offset);
    int32_t* merge_stride = reinterpret_cast<int32_t*>(
        workspace + merge_stride_offset);
    int32_t* q_len = reinterpret_cast<int32_t*>(
        workspace + q_len_offset);
    int32_t* kv_len = reinterpret_cast<int32_t*>(
        workspace + kv_len_offset);
    int32_t* q_start = reinterpret_cast<int32_t*>(
        workspace + q_start_offset);
    int32_t* kv_start = reinterpret_cast<int32_t*>(
        workspace + kv_start_offset);
    int32_t* kv_end = reinterpret_cast<int32_t*>(
        workspace + kv_end_offset);
    int32_t* work_indptr = reinterpret_cast<int32_t*>(
        workspace + work_indptr_offset);

    for (int i = threadIdx.x; i < num_works; i += blockDim.x) {
        const int start = i * kv_limit;
        q_indptr_plan[i] = 0;
        kv_indptr_plan[i] = 0;
        partial_indptr[i] = split ? i * heads : -1;
        q_len[i] = 1;
        kv_len[i] = length;
        q_start[i] = 0;
        kv_start[i] = start;
        kv_end[i] = min(start + kv_limit, length);
    }
    for (int i = threadIdx.x; i < num_sms; i += blockDim.x) {
        if (i < merge_count) {
            const int start = i * row_chunk;
            merge_start[i] = start;
            merge_end[i] = min(start + row_chunk, heads);
            merge_partial_start[i] = start;
            merge_partial_end[i] = num_works * heads;
            merge_stride[i] = heads;
        } else {
            merge_start[i] = 0;
            merge_end[i] = 0;
            merge_partial_start[i] = 0;
            merge_partial_end[i] = 0;
            merge_stride[i] = 0;
        }
    }
    for (int i = threadIdx.x; i <= num_sms; i += blockDim.x) {
        work_indptr[i] = min(i, num_works);
    }
    for (int i = threadIdx.x; i < blocks; i += blockDim.x) {
        kv_indices[i] = i;
    }
    if (threadIdx.x == 0) {
        kv_indptr[0] = 0;
        kv_indptr[1] = blocks;
        kv_len_arr[0] = length;
    }
}

bool flashinfer_mla_batch1_plan(
    torch::Tensor int_workspace,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    torch::Tensor kv_len_arr,
    long length,
    long page_size,
    long heads,
    std::vector<int64_t> plan_info) {
    TORCH_CHECK(
        int_workspace.is_cuda() &&
        kv_indptr.is_cuda() &&
        kv_indices.is_cuda() &&
        kv_len_arr.is_cuda(),
        "FlashInfer MLA plan buffers must be CUDA");
    TORCH_CHECK(
        int_workspace.scalar_type() == at::kByte &&
        kv_indptr.scalar_type() == at::kInt &&
        kv_indices.scalar_type() == at::kInt &&
        kv_len_arr.scalar_type() == at::kInt,
        "FlashInfer MLA plan buffer dtypes are invalid");
    TORCH_CHECK(
        int_workspace.is_contiguous() &&
        kv_indptr.is_contiguous() &&
        kv_indices.is_contiguous() &&
        kv_len_arr.is_contiguous(),
        "FlashInfer MLA plan buffers must be contiguous");
    TORCH_CHECK(
        kv_indptr.numel() == 2 &&
        kv_len_arr.numel() == 1 &&
        plan_info.size() == 18,
        "FlashInfer MLA batch-1 plan shape mismatch");
    TORCH_CHECK(
        length > 0 &&
        page_size > 0 &&
        heads > 0 &&
        (length + page_size - 1) / page_size <= kv_indices.numel(),
        "FlashInfer MLA batch-1 context length is out of range");
    const int num_sms = static_cast<int>(plan_info[1]);
    TORCH_CHECK(
        plan_info[0] == 1 &&
        num_sms > 0 &&
        plan_info[15] +
            static_cast<int64_t>(num_sms + 1) * sizeof(int32_t) <=
            int_workspace.numel(),
        "Unsupported FlashInfer MLA plan layout");

    auto stream = at::cuda::getCurrentCUDAStream();
    flashinfer_mla_batch1_plan_kernel<<<1, 256, 0, stream>>>(
        int_workspace.data_ptr<uint8_t>(),
        kv_indptr.data_ptr<int32_t>(),
        kv_indices.data_ptr<int32_t>(),
        kv_len_arr.data_ptr<int32_t>(),
        static_cast<int>(length),
        static_cast<int>(page_size),
        static_cast<int>(heads),
        num_sms,
        plan_info[2],
        plan_info[3],
        plan_info[4],
        plan_info[5],
        plan_info[6],
        plan_info[7],
        plan_info[8],
        plan_info[9],
        plan_info[10],
        plan_info[11],
        plan_info[12],
        plan_info[13],
        plan_info[14],
        plan_info[15]);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vq_gemv", &vq_gemv, "VQ grouped GEMV (fused codebook lookup + dot)");
    m.def("kimi_short_conv3", &kimi_short_conv3,
          "Kimi three-way one-token short convolution");
    m.def("kimi_kda_recurrent", &kimi_kda_recurrent,
          "Kimi KDA one-token recurrent update with V-first FP32 state");
    m.def("kimi_gated_rmsnorm", &kimi_gated_rmsnorm,
          "Kimi one-token gated RMSNorm");
    m.def(
          "packed_moe_topk",
          &packed_moe_topk,
          "Packed 8/12/14/16-bit Top-K routed expert MLP");
    m.def(
          "kimi_moe_packed",
          &packed_moe_topk,
          "Compatibility alias for packed_moe_topk");
    m.def("vq_gemv_slots_out", &vq_gemv_slots_out,
          "Stable-slot grouped BF16 VQ GEMV into caller workspace");
    m.def("moe_mlp_slots", &moe_mlp_slots,
          "Stable-slot BF16 VQ MLP (GU + SwiGLU + DN + weighted sum)");
    m.def("moe_mlp_routed_slots", &moe_mlp_routed_slots,
          "Device-routed full-resident VQ MLP partial");
    m.def("moe_mlp_routed_vv", &moe_mlp_routed_vv,
          "Shared-codebook D4/K4096 full-resident VQ MLP partial");
    m.def("moe_mlp_routed_codegemm", &moe_mlp_routed_codegemm,
          "CodeGEMM Psumbook full-resident VQ MLP partial");
    m.def("pack_vq_tensor_shard_codegemm",
          &pack_vq_tensor_shard_codegemm,
          "Pack a tensor-sharded v256/D4 expert for CodeGEMM");
    m.def("unpack_vq_codegemm", &unpack_vq_codegemm,
          "Restore CodeGEMM indices to TPQ row-major layout");
    m.def("expert_dispatch_pack", &expert_dispatch_pack,
          "Pack one full-resident expert dispatch from a peer GPU");
    m.def("tp_peer_copy", &tp_peer_copy,
          "Graph-safe peer copy for TP rank source tensors");
    m.def("tp_attention_peer_dispatch", &tp_attention_peer_dispatch,
          "Graph-safe fused Attention TP peer dispatch");
    m.def("tp_attention_source_pack", &tp_attention_source_pack,
          "Fused Attention TP primary source packing");
    m.def("hc_sinkhorn", &hc_sinkhorn, "HC 4x4 sinkhorn (fused softmax + iterations)");
    m.def("rmsnorm", &rmsnorm, "RMSNorm (fused, f32)");
    m.def(
        "rmsnorm_bf16",
        &rmsnorm_bf16,
        "RMSNorm (fused, BF16 input)");
    m.def(
        "attention_residual_bf16",
        &attention_residual_bf16,
        "Attention residual mixer (fused, BF16)");
    m.def(
        "gated_activation_bf16",
        &gated_activation_bf16,
        "SiLU/SiTU gated activation (fused, BF16)");
    m.def("glm_mla_bmm_decode", &glm_mla_bmm_decode,
          "Direct cuBLAS strided-batched MLA decode GEMM");
    m.def("flashinfer_mla_batch1_plan",
          &flashinfer_mla_batch1_plan,
          "Device-side exact batch-1 FlashInfer MLA scheduler");
    m.def("rope1", &rope1, "RoPE interleaved (decode single-phase, f32)");
    m.def("glm_rope_qk", &glm_rope_qk,
          "GLM MLA Q/K RoPE (fused, HF cat layout, f32)");
    m.def("glm_latent_kv_decode_prepare",
          &glm_latent_kv_decode_prepare,
          "GLM decode RMS/RoPE and BF16 latent KV writes");
    m.def("glm_merge_scores", &glm_merge_scores,
          "GLM latent attention score scale/add (fused, BF16 to f32)");
    m.def(
        "latent_mla_attention_decode",
        &latent_mla_attention_decode,
        "Dynamic-length BF16 latent MLA decode");
    m.def(
        "latent_mla_attention_scores",
        &latent_mla_attention_scores,
        "Dynamic-length BF16 latent MLA score preparation");
    m.def("dsv4_attn_decode", &dsv4_attn_decode, "DSV4 decode attention core (fused, f32)");
    m.def("dsv4_hc_pre", &dsv4_hc_pre, "DSV4 HC pre (fused RMS/GEMV/sinkhorn/reduce)");
    m.def("dsv4_hc_pre_norm", &dsv4_hc_pre_norm,
          "DSV4 BF16 HC pre + RMSNorm (FP32 reductions)");
    m.def("dsv4_hc_post", &dsv4_hc_post,
          "DSV4 BF16 HC post residual mix (FP32 accumulation)");
    m.def("dsv4_route_post", &dsv4_route_post,
          "DSV4 learned-route top-k, gather, normalize and scale");
    m.def("sigmoid_route", &sigmoid_route,
          "Sigmoid corrected Top-K with normalized route weights");
    m.def("sigmoid_route_out", &sigmoid_route_out,
          "Sigmoid route into caller-owned decode buffers");
    m.def(
          "linear_sigmoid_route_out",
          &linear_sigmoid_route_out,
          "FP32 linear projection plus sigmoid Top-K routing");
    m.def("glm_route", &sigmoid_route,
          "Compatibility alias for sigmoid_route");
    m.def("glm_route_out", &sigmoid_route_out,
          "Compatibility alias for sigmoid_route_out");
    m.def("paged_gather_bf16", &paged_gather_bf16,
          "Gather batch-1 BF16 entries from stable paged storage");
    m.def("hadamard_bf16", &hadamard_bf16,
          "Normalized BF16 Walsh-Hadamard transform with FP32 butterflies");
    m.def("int4_embedding_lookup", &int4_embedding_lookup,
          "Packed INT4-G64 single-row embedding lookup");
    m.def("int4_embedding_lookup_device_row",
          &int4_embedding_lookup_device_row,
          "Packed INT4-G64 embedding lookup with a CUDA row index");
    m.def("int4_gemv_packed_f32", &int4_gemv_packed_f32,
          "Shared-input packed INT4-G64 GEMV for float32 decode");
    m.def("block_fp8_gemv_f32", &block_fp8_gemv_f32,
          "Native E4M3 block-scaled GEMV for float32 decode");
    m.def("int4_glm_qb_split", &int4_glm_qb_split,
          "Packed GLM Q-B GEMV into BF16 no-PE and FP32 RoPE outputs");
    m.def("glm_norm_qkv_int4", &glm_norm_qkv_int4,
          "GLM decode RMSNorm plus packed Q-A/KV-A projections");
    m.def("glm_residual_norm_router",
          &glm_residual_norm_router,
          "GLM decode residual add plus RMSNorm and router projection");
    m.def("glm_residual_norm_router_norm_out",
          &glm_residual_norm_router_norm_out,
          "GLM residual/router with caller-owned normalized output");
    m.def("residual_add3",
          &residual_add3,
          "Three-way residual addition for float32 or bfloat16");
    m.def("glm_moe_residual_add",
          &residual_add3,
          "Compatibility alias for three-way residual addition");
    m.def("glm_ep_reduce_residual",
          &glm_ep_reduce_residual,
          "GLM TP routed/shared contribution reduction plus residual");
    m.def("tp_all_rank_reduce",
          &tp_all_rank_reduce,
          "Reduce FP32 TP partials into fixed outputs on every rank");
    m.def("tp_hidden_add_batch",
          &tp_hidden_add_batch,
          "Add fixed BF16 TPHidden replicas in one host call");
    m.def("tp_hidden_rmsnorm_batch",
          &tp_hidden_rmsnorm_batch,
          "RMSNorm fixed BF16 TPHidden replicas in one host call");
    m.def("tp_hidden_residual_mix_batch",
          &tp_hidden_residual_mix_batch,
          "Residual-mix fixed BF16 TPHidden replicas in one host call");
    m.def("launch_cuda_graphs",
          &launch_cuda_graphs,
          "Launch one prepared CUDA Graph per TP rank in one host call");
    m.def("launch_cuda_graphs_reduce",
          &launch_cuda_graphs_reduce,
          "Launch TP graphs, wait ranks and reduce in one host call");
    m.def(
          "launch_cuda_graphs_reduce_norm_router",
          &launch_cuda_graphs_reduce_norm_router,
          "Launch TP Attention graphs then reduce, normalize and route");
    pybind11::class_<TPGraphLaunchBatch>(m, "TPGraphLaunchBatch")
        .def(pybind11::init<
             std::vector<int64_t>,
             std::vector<int64_t>,
             std::vector<int64_t>,
             std::vector<int64_t>,
             int64_t>())
        .def(pybind11::init<
             std::vector<int64_t>,
             std::vector<std::vector<int64_t>>,
             std::vector<int64_t>,
             std::vector<int64_t>,
             int64_t>())
        .def(pybind11::init<
             std::vector<int64_t>,
             std::vector<std::vector<std::vector<int64_t>>>,
             std::vector<int64_t>,
             std::vector<int64_t>,
             int64_t>())
        .def("launch", &TPGraphLaunchBatch::launch)
        .def("launch_reduce", &TPGraphLaunchBatch::launch_reduce)
        .def(
            "launch_reduce_many",
            &TPGraphLaunchBatch::launch_reduce_many)
        .def(
            "launch_all_rank",
            &TPGraphLaunchBatch::launch_all_rank)
        .def(
            "launch_all_rank_from_events",
            &TPGraphLaunchBatch::launch_all_rank_from_events)
        .def(
            "launch_all_rank_many_from_events",
            &TPGraphLaunchBatch::launch_all_rank_many_from_events)
        .def(
            "reduce_all_rank_many_from_events",
            &TPGraphLaunchBatch::reduce_all_rank_many_from_events)
        .def(
            "launch_from_events",
            &TPGraphLaunchBatch::launch_from_events)
        .def(
            "launch_moe_all_rank_from_events",
            &TPGraphLaunchBatch::launch_moe_all_rank_from_events)
        .def(
            "launch_reduce_norm_router",
            &TPGraphLaunchBatch::launch_reduce_norm_router)
        .def(
            "launch_moe_layer",
            &TPGraphLaunchBatch::launch_moe_layer);
    pybind11::class_<TPNoOwnerMoELayerPlan>(
        m,
        "TPNoOwnerMoELayerPlan")
        .def(
            pybind11::init<
                const TPGraphLaunchBatch&,
                const TPGraphLaunchBatch&,
                const TPGraphLaunchBatch&,
                const TPGraphLaunchBatch&,
                std::vector<int64_t>,
                std::vector<std::vector<torch::Tensor>>,
                std::vector<std::vector<torch::Tensor>>,
                std::vector<int64_t>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>>(),
            pybind11::keep_alive<1, 2>(),
            pybind11::keep_alive<1, 3>(),
            pybind11::keep_alive<1, 4>(),
            pybind11::keep_alive<1, 5>())
        .def("launch", &TPNoOwnerMoELayerPlan::launch)
        .def(
            "launch_from_events",
            &TPNoOwnerMoELayerPlan::launch_from_events);
    pybind11::class_<TPNoOwnerDecodeLayerPlan>(
        m,
        "TPNoOwnerDecodeLayerPlan")
        .def(
            pybind11::init<
                const TPGraphLaunchBatch&,
                const TPNoOwnerMoELayerPlan&,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>>(),
            pybind11::keep_alive<1, 2>(),
            pybind11::keep_alive<1, 3>())
        .def(
            "launch_from_events",
            &TPNoOwnerDecodeLayerPlan::launch_from_events)
        .def(
            "persistent_enabled",
            &TPNoOwnerDecodeLayerPlan::persistent_enabled);
    m.def(
          "bf16_gemv_out",
          &bf16_gemv_out,
          "Fixed-output BF16 GEMV with FP32 accumulation");
    m.def("int4_swiglu_packed_f32", &int4_swiglu_packed_f32,
          "Fused gate/up packed INT4-G64 GEMV plus FP32 SwiGLU");
}
