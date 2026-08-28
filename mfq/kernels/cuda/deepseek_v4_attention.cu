#include <cuda_runtime.h>
#include "../../../cpp_runtime/cuda/mfq_tensor_backend.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif
#include <cuda_fp8.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#define MFQ_FATTN_KERNEL_ONLY
#include "mfq_fattn_mma_f16.cuh"

namespace {

constexpr int kHeadDim = 512;
constexpr int kHeads = 64;
constexpr int kHeadsPerTile = 16;
constexpr int kIndexerDim = 128;
constexpr int kIndexerHeads = 64;
constexpr int kIndexerKeysPerTile = 64;
constexpr int kIndexerTopK = 512;

template<typename T>
struct Dsv4TypeTag {
    using type = T;
};

template<typename scalar_t>
__device__ __forceinline__ float dsv4_to_float(scalar_t value);

template<>
__device__ __forceinline__ float dsv4_to_float<half>(half value) {
    return __half2float(value);
}

template<>
__device__ __forceinline__ float dsv4_to_float<float>(float value) {
    return value;
}

__device__ __forceinline__ float dsv4_fp8_e4m3_dequant(
    float value,
    float scale)
{
    const float normalized =
        fminf(448.0f, fmaxf(-448.0f, value / scale));
    const __nv_fp8_e4m3 quantized(normalized);
    return static_cast<float>(quantized) * scale;
}

__device__ __forceinline__ float dsv4_bf16_round(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float dsv4_pow2_ceil(float value) {
    const uint32_t bits = __float_as_uint(value);
    const int exponent = static_cast<int>((bits >> 23) & 0xffu);
    const bool has_mantissa = (bits & 0x7fffffu) != 0;
    return __uint_as_float(
        static_cast<uint32_t>(
            exponent + static_cast<int>(has_mantissa)) << 23);
}

__device__ __forceinline__ float dsv4_fp4_e2m1_dequant(
    float value,
    float scale)
{
    const float normalized =
        fminf(6.0f, fmaxf(-6.0f, value / scale));
#if CUDART_VERSION >= 12080
    const __nv_fp4_e2m1 quantized(normalized);
    return static_cast<float>(quantized) * scale;
#else
    const float magnitude = fabsf(normalized);
    float quantized;
    if (magnitude <= 0.25f) {
        quantized = 0.0f;
    } else if (magnitude < 0.75f) {
        quantized = 0.5f;
    } else if (magnitude <= 1.25f) {
        quantized = 1.0f;
    } else if (magnitude < 1.75f) {
        quantized = 1.5f;
    } else if (magnitude <= 2.5f) {
        quantized = 2.0f;
    } else if (magnitude < 3.5f) {
        quantized = 3.0f;
    } else if (magnitude <= 5.0f) {
        quantized = 4.0f;
    } else {
        quantized = 6.0f;
    }
    return copysignf(quantized * scale, normalized);
#endif
}

__device__ __forceinline__ float dsv4_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// The compressor equations and overlap layout match llama.cpp DeepSeek4 and
// mlx-lm PR 1192: ratio-4 pools previous-A with current-B, while ratio-128
// pools one ordinary window.
template<typename scalar_t, int HEAD_DIM>
__global__ void dsv4_compress_kernel(
    const scalar_t * __restrict__ kv,
    const scalar_t * __restrict__ gate,
    const float * __restrict__ ape,
    const float * __restrict__ norm,
    const scalar_t * __restrict__ prev_kv,
    const scalar_t * __restrict__ prev_gate,
    const int64_t * __restrict__ positions,
    const float * __restrict__ cos,
    const float * __restrict__ sin,
    half * __restrict__ out,
    int B,
    int W,
    int ratio,
    int out_dim,
    int overlap,
    int has_prev,
    int rope_dim,
    int table_stride,
    int quant_mode,
    float eps)
{
    __shared__ float compressed[HEAD_DIM];
    __shared__ float quant_scales[HEAD_DIM / 32];
    constexpr int kWarps = HEAD_DIM / 32;
    __shared__ float warp_sums[kWarps];
    __shared__ float inv_rms;

    const int row = blockIdx.x;
    const int b = row / W;
    const int w = row - b * W;
    const int d = threadIdx.x;
    if (b >= B || d >= HEAD_DIM) return;

    float max_score = -INFINITY;
    const int candidates = overlap ? 2 * ratio : ratio;
    for (int i = 0; i < candidates; ++i) {
        float score = -INFINITY;
        if (!overlap) {
            const int64_t idx =
                (((static_cast<int64_t>(b) * W + w) * ratio + i) * out_dim + d);
            score = dsv4_to_float(gate[idx]) +
                ape[static_cast<int64_t>(i) * out_dim + d];
        } else if (i < ratio) {
            if (w > 0) {
                const int64_t idx =
                    (((static_cast<int64_t>(b) * W + w - 1) * ratio + i) * out_dim + d);
                score = dsv4_to_float(gate[idx]) +
                    ape[static_cast<int64_t>(i) * out_dim + d];
            } else if (has_prev) {
                const int64_t idx =
                    (static_cast<int64_t>(b) * ratio + i) * HEAD_DIM + d;
                score = dsv4_to_float(prev_gate[idx]) +
                    ape[static_cast<int64_t>(i) * out_dim + d];
            }
        } else {
            const int r = i - ratio;
            const int64_t idx =
                (((static_cast<int64_t>(b) * W + w) * ratio + r) * out_dim +
                 HEAD_DIM + d);
            score = dsv4_to_float(gate[idx]) +
                ape[static_cast<int64_t>(r) * out_dim + HEAD_DIM + d];
        }
        max_score = fmaxf(max_score, score);
    }

    float weighted = 0.0f;
    float denominator = 0.0f;
    for (int i = 0; i < candidates; ++i) {
        float value = 0.0f;
        float score = -INFINITY;
        if (!overlap) {
            const int64_t idx =
                (((static_cast<int64_t>(b) * W + w) * ratio + i) * out_dim + d);
            value = dsv4_to_float(kv[idx]);
            score = dsv4_to_float(gate[idx]) +
                ape[static_cast<int64_t>(i) * out_dim + d];
        } else if (i < ratio) {
            if (w > 0) {
                const int64_t idx =
                    (((static_cast<int64_t>(b) * W + w - 1) * ratio + i) * out_dim + d);
                value = dsv4_to_float(kv[idx]);
                score = dsv4_to_float(gate[idx]) +
                    ape[static_cast<int64_t>(i) * out_dim + d];
            } else if (has_prev) {
                const int64_t idx =
                    (static_cast<int64_t>(b) * ratio + i) * HEAD_DIM + d;
                value = dsv4_to_float(prev_kv[idx]);
                score = dsv4_to_float(prev_gate[idx]) +
                    ape[static_cast<int64_t>(i) * out_dim + d];
            }
        } else {
            const int r = i - ratio;
            const int64_t idx =
                (((static_cast<int64_t>(b) * W + w) * ratio + r) * out_dim +
                 HEAD_DIM + d);
            value = dsv4_to_float(kv[idx]);
            score = dsv4_to_float(gate[idx]) +
                ape[static_cast<int64_t>(r) * out_dim + HEAD_DIM + d];
        }
        if (isfinite(score)) {
            const float weight = expf(score - max_score);
            weighted += weight * value;
            denominator += weight;
        }
    }
    const float value = weighted / denominator;
    compressed[d] = value;

    float square_sum = dsv4_warp_sum(value * value);
    const int lane = d & 31;
    const int warp = d >> 5;
    if (lane == 0) warp_sums[warp] = square_sum;
    __syncthreads();
    if (warp == 0) {
        square_sum = lane < kWarps ? warp_sums[lane] : 0.0f;
        square_sum = dsv4_warp_sum(square_sum);
        if (lane == 0) {
            inv_rms = rsqrtf(square_sum / static_cast<float>(HEAD_DIM) + eps);
        }
    }
    __syncthreads();

    compressed[d] = dsv4_bf16_round(
        value * inv_rms * norm[d]);
    __syncthreads();

    const int nope_dim = HEAD_DIM - rope_dim;
    if (d >= nope_dim && ((d - nope_dim) & 1) == 0) {
        const int pair = (d - nope_dim) / 2;
        const int64_t position = positions[row];
        const float c = cos[position * table_stride + pair];
        const float s = sin[position * table_stride + pair];
        const float x0 = compressed[d];
        const float x1 = compressed[d + 1];
        compressed[d] = dsv4_bf16_round(x0 * c - x1 * s);
        compressed[d + 1] = dsv4_bf16_round(x1 * c + x0 * s);
    }
    __syncthreads();

    if (quant_mode == 2) {
#pragma unroll
        for (int half_width = 1;
             half_width < HEAD_DIM;
             half_width <<= 1) {
            const int butterfly = d;
            if (butterfly < HEAD_DIM / 2) {
                const int group = butterfly / half_width;
                const int within = butterfly - group * half_width;
                const int first = group * (2 * half_width) + within;
                const int second = first + half_width;
                const float a = compressed[first];
                const float b = compressed[second];
                compressed[first] = a + b;
                compressed[second] = a - b;
            }
            __syncthreads();
        }
        compressed[d] = dsv4_bf16_round(
            compressed[d] * rsqrtf(static_cast<float>(HEAD_DIM)));
        __syncthreads();
        if ((d & 31) == 0) {
            float amax = 6.0f * 0x1p-126f;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                amax = fmaxf(amax, fabsf(compressed[d + i]));
            }
            quant_scales[d / 32] =
                dsv4_pow2_ceil(amax / 6.0f);
        }
    }
    if (quant_mode == 1 && d < nope_dim && (d & 63) == 0) {
        float amax = 1e-4f;
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            amax = fmaxf(amax, fabsf(compressed[d + i]));
        }
        quant_scales[d / 64] = amax / 448.0f;
    }
    __syncthreads();
    const int64_t out_base = static_cast<int64_t>(row) * HEAD_DIM;
    {
        const float result = quant_mode == 1 && d < nope_dim
            ? dsv4_fp8_e4m3_dequant(
                compressed[d], quant_scales[d / 64])
            : quant_mode == 2
                ? dsv4_fp4_e2m1_dequant(
                    compressed[d], quant_scales[d / 32])
                : compressed[d];
        out[out_base + d] = __float2half_rn(
            dsv4_bf16_round(result));
    }
}

// CUDA-Graph-safe decode update. seq_len stays on the GPU; every invocation
// stores one projected token, and only ratio boundaries emit a pooled row.
template<typename scalar_t, int HEAD_DIM>
__global__ void dsv4_decode_pool_update_kernel(
    const scalar_t * __restrict__ kv_token,
    const scalar_t * __restrict__ gate_token,
    const float * __restrict__ ape,
    const float * __restrict__ norm,
    scalar_t * __restrict__ state_kv,
    scalar_t * __restrict__ state_gate,
    scalar_t * __restrict__ prev_kv,
    scalar_t * __restrict__ prev_gate,
    half * __restrict__ pool,
    const int64_t * __restrict__ seq_len,
    const float * __restrict__ cos,
    const float * __restrict__ sin,
    int B,
    int ratio,
    int out_dim,
    int overlap,
    int pool_capacity,
    int rope_dim,
    int table_stride,
    int quant_mode,
    float eps)
{
    __shared__ float compressed[HEAD_DIM];
    __shared__ float quant_scales[HEAD_DIM / 32];
    constexpr int kWarps = HEAD_DIM / 32;
    __shared__ float warp_sums[kWarps];
    __shared__ float inv_rms;

    const int b = blockIdx.x;
    const int d = threadIdx.x;
    if (b >= B || d >= HEAD_DIM) return;
    const int64_t length = seq_len[b];
    if (length <= 0) return;
    const int64_t position = length - 1;
    const int slot = static_cast<int>(position % ratio);

    for (int feature = d; feature < out_dim; feature += HEAD_DIM) {
        const int64_t token_idx = static_cast<int64_t>(b) * out_dim + feature;
        const int64_t state_idx =
            (static_cast<int64_t>(b) * ratio + slot) * out_dim + feature;
        state_kv[state_idx] = kv_token[token_idx];
        state_gate[state_idx] = gate_token[token_idx];
    }
    __syncthreads();
    if (slot != ratio - 1) return;

    const int pool_index = static_cast<int>(position / ratio);
    if (pool_index >= pool_capacity) return;
    const bool have_previous = pool_index > 0;
    const int candidates = overlap ? 2 * ratio : ratio;
    float max_score = -INFINITY;
    for (int i = 0; i < candidates; ++i) {
        float score = -INFINITY;
        if (!overlap) {
            const int64_t idx =
                (static_cast<int64_t>(b) * ratio + i) * out_dim + d;
            score = dsv4_to_float(state_gate[idx]) +
                ape[static_cast<int64_t>(i) * out_dim + d];
        } else if (i < ratio) {
            if (have_previous) {
                const int64_t idx =
                    (static_cast<int64_t>(b) * ratio + i) * HEAD_DIM + d;
                score = dsv4_to_float(prev_gate[idx]) +
                    ape[static_cast<int64_t>(i) * out_dim + d];
            }
        } else {
            const int r = i - ratio;
            const int64_t idx =
                (static_cast<int64_t>(b) * ratio + r) * out_dim +
                    HEAD_DIM + d;
            score = dsv4_to_float(state_gate[idx]) +
                ape[static_cast<int64_t>(r) * out_dim + HEAD_DIM + d];
        }
        max_score = fmaxf(max_score, score);
    }

    float weighted = 0.0f;
    float denominator = 0.0f;
    for (int i = 0; i < candidates; ++i) {
        float value = 0.0f;
        float score = -INFINITY;
        if (!overlap) {
            const int64_t idx =
                (static_cast<int64_t>(b) * ratio + i) * out_dim + d;
            value = dsv4_to_float(state_kv[idx]);
            score = dsv4_to_float(state_gate[idx]) +
                ape[static_cast<int64_t>(i) * out_dim + d];
        } else if (i < ratio) {
            if (have_previous) {
                const int64_t idx =
                    (static_cast<int64_t>(b) * ratio + i) * HEAD_DIM + d;
                value = dsv4_to_float(prev_kv[idx]);
                score = dsv4_to_float(prev_gate[idx]) +
                    ape[static_cast<int64_t>(i) * out_dim + d];
            }
        } else {
            const int r = i - ratio;
            const int64_t idx =
                (static_cast<int64_t>(b) * ratio + r) * out_dim +
                    HEAD_DIM + d;
            value = dsv4_to_float(state_kv[idx]);
            score = dsv4_to_float(state_gate[idx]) +
                ape[static_cast<int64_t>(r) * out_dim + HEAD_DIM + d];
        }
        if (isfinite(score)) {
            const float weight = expf(score - max_score);
            weighted += weight * value;
            denominator += weight;
        }
    }
    const float value = weighted / denominator;
    compressed[d] = value;

    float square_sum = dsv4_warp_sum(value * value);
    const int lane = d & 31;
    const int warp = d >> 5;
    if (lane == 0) warp_sums[warp] = square_sum;
    __syncthreads();
    if (warp == 0) {
        square_sum = lane < kWarps ? warp_sums[lane] : 0.0f;
        square_sum = dsv4_warp_sum(square_sum);
        if (lane == 0) {
            inv_rms = rsqrtf(
                square_sum / static_cast<float>(HEAD_DIM) + eps);
        }
    }
    __syncthreads();
    compressed[d] = dsv4_bf16_round(
        value * inv_rms * norm[d]);
    __syncthreads();

    const int nope_dim = HEAD_DIM - rope_dim;
    if (d >= nope_dim && ((d - nope_dim) & 1) == 0) {
        const int pair = (d - nope_dim) / 2;
        const float c = cos[
            static_cast<int64_t>(pool_index) * table_stride + pair];
        const float s = sin[
            static_cast<int64_t>(pool_index) * table_stride + pair];
        const float x0 = compressed[d];
        const float x1 = compressed[d + 1];
        compressed[d] = dsv4_bf16_round(x0 * c - x1 * s);
        compressed[d + 1] = dsv4_bf16_round(x1 * c + x0 * s);
    }
    __syncthreads();

    if (quant_mode == 2) {
#pragma unroll
        for (int half_width = 1;
             half_width < HEAD_DIM;
             half_width <<= 1) {
            const int butterfly = d;
            if (butterfly < HEAD_DIM / 2) {
                const int group = butterfly / half_width;
                const int within = butterfly - group * half_width;
                const int first = group * (2 * half_width) + within;
                const int second = first + half_width;
                const float a = compressed[first];
                const float b = compressed[second];
                compressed[first] = a + b;
                compressed[second] = a - b;
            }
            __syncthreads();
        }
        compressed[d] = dsv4_bf16_round(
            compressed[d] * rsqrtf(static_cast<float>(HEAD_DIM)));
        __syncthreads();
        if ((d & 31) == 0) {
            float amax = 6.0f * 0x1p-126f;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                amax = fmaxf(amax, fabsf(compressed[d + i]));
            }
            quant_scales[d / 32] =
                dsv4_pow2_ceil(amax / 6.0f);
        }
    }
    if (quant_mode == 1 && d < nope_dim && (d & 63) == 0) {
        float amax = 1e-4f;
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            amax = fmaxf(amax, fabsf(compressed[d + i]));
        }
        quant_scales[d / 64] = amax / 448.0f;
    }
    __syncthreads();
    const int64_t pool_base =
        (static_cast<int64_t>(b) * pool_capacity + pool_index) * HEAD_DIM;
    {
        const float result = quant_mode == 1 && d < nope_dim
            ? dsv4_fp8_e4m3_dequant(
                compressed[d], quant_scales[d / 64])
            : quant_mode == 2
                ? dsv4_fp4_e2m1_dequant(
                    compressed[d], quant_scales[d / 32])
                : compressed[d];
        pool[pool_base + d] = __float2half_rn(
            dsv4_bf16_round(result));
    }
    __syncthreads();

    if (overlap) {
        for (int r = 0; r < ratio; ++r) {
            const int64_t state_idx =
                (static_cast<int64_t>(b) * ratio + r) * out_dim + d;
            const int64_t prev_idx =
                (static_cast<int64_t>(b) * ratio + r) * HEAD_DIM + d;
            prev_kv[prev_idx] = state_kv[state_idx];
            prev_gate[prev_idx] = state_gate[state_idx];
        }
    }
}

__global__ void dsv4_fp4_sim_kernel(
    const half * __restrict__ input,
    half * __restrict__ output)
{
    __shared__ float reduction[32];
    __shared__ float scale;
    const int lane = threadIdx.x;
    const int64_t offset =
        static_cast<int64_t>(blockIdx.x) * 32 + lane;
    const float value = __half2float(input[offset]);
    reduction[lane] = fabsf(value);
    __syncthreads();
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] = fmaxf(
                reduction[lane], reduction[lane + stride]);
        }
        __syncthreads();
    }
    if (lane == 0) {
        const float amax =
            fmaxf(reduction[0], 6.0f * 0x1p-126f);
        scale = dsv4_pow2_ceil(amax / 6.0f);
    }
    __syncthreads();
    output[offset] = __float2half_rn(
        dsv4_fp4_e2m1_dequant(value, scale));
}

// One block covers 64 index heads by 64 pooled keys. Sixteen warps map to
// the 4x4 WMMA tiles and the head-weighted ReLU reduction remains on chip.
__global__ void dsv4_indexer_scores_kernel(
    const half * __restrict__ q,
    const half * __restrict__ k,
    const half * __restrict__ weights,
    half * __restrict__ out,
    int B,
    int M,
    int K,
    int query_offset,
    int ratio,
    float score_scale)
{
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    __shared__ half q_tile[kIndexerHeads][kIndexerDim];
    __shared__ half k_tile[kIndexerKeysPerTile][kIndexerDim];
    __shared__ float score_tile[kIndexerHeads][kIndexerKeysPerTile];

    const int query = blockIdx.y;
    const int batch = blockIdx.z;
    const int key_base = blockIdx.x * kIndexerKeysPerTile;
    const int tid = threadIdx.y * 32 + threadIdx.x;

    const half * q_row = q +
        (static_cast<int64_t>(batch) * M + query) *
            kIndexerHeads * kIndexerDim;
    for (int i = tid; i < kIndexerHeads * kIndexerDim; i += 512) {
        reinterpret_cast<half *>(q_tile)[i] = q_row[i];
    }
    for (int i = tid; i < kIndexerKeysPerTile * kIndexerDim; i += 512) {
        const int key_local = i / kIndexerDim;
        const int dim = i - key_local * kIndexerDim;
        const int key = key_base + key_local;
        reinterpret_cast<half *>(k_tile)[i] = key < K
            ? k[(static_cast<int64_t>(batch) * K + key) * kIndexerDim + dim]
            : __float2half(0.0f);
    }
    __syncthreads();

    const int warp = threadIdx.y;
    const int head_base = (warp / 4) * 16;
    const int key_local_base = (warp % 4) * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
#pragma unroll
    for (int d = 0; d < kIndexerDim; d += 16) {
        wmma::load_matrix_sync(a, &q_tile[head_base][d], kIndexerDim);
        wmma::load_matrix_sync(b, &k_tile[key_local_base][d], kIndexerDim);
        wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(
        &score_tile[head_base][key_local_base], acc,
        kIndexerKeysPerTile, wmma::mem_row_major);
    __syncthreads();

    if (tid < kIndexerKeysPerTile) {
        const int key = key_base + tid;
        if (key < K) {
            float sum = 0.0f;
#pragma unroll
            for (int head = 0; head < kIndexerHeads; ++head) {
                const float dot = score_tile[head][tid];
                const float weight = __half2float(
                    weights[(static_cast<int64_t>(batch) * M + query) *
                        kIndexerHeads + head]);
                sum += fmaxf(dot, 0.0f) * weight;
            }
            const int visible = min(
                K, (query_offset + query + 1) / ratio);
            out[(static_cast<int64_t>(batch) * M + query) * K + key] =
                key < visible
                    ? __float2half_rn(sum * score_scale)
                    : __float2half(-INFINITY);
        }
    }
#else
    (void)q;
    (void)k;
    (void)weights;
    (void)out;
    (void)B;
    (void)M;
    (void)K;
    (void)query_offset;
    (void)ratio;
    (void)score_scale;
#endif
}

__device__ __forceinline__ unsigned int dsv4_ordered_half(half value) {
    const unsigned int bits = __half_as_ushort(value);
    return (bits & 0x8000u) ? ((~bits) & 0xffffu) : (bits | 0x8000u);
}

template<int TOPK>
__global__ void dsv4_topk_indices_kernel(
    const half * __restrict__ scores,
    int * __restrict__ out,
    int rows,
    int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    const half * row_scores = scores + static_cast<int64_t>(row) * K;
    int * row_out = out + static_cast<int64_t>(row) * TOPK;
    if (K <= TOPK) {
        for (int i = tid; i < TOPK; i += blockDim.x) {
            row_out[i] = i < K ? i : 0;
        }
        return;
    }

    __shared__ unsigned int hist[256];
    __shared__ unsigned int counters[2];
    __shared__ unsigned int state[4];
    hist[tid] = 0;
    if (tid < 2) counters[tid] = 0;
    __syncthreads();

    for (int i = tid; i < K; i += blockDim.x) {
        const unsigned int key = dsv4_ordered_half(row_scores[i]);
        atomicAdd(&hist[key >> 8], 1u);
    }
    __syncthreads();

    if (tid == 0) {
        unsigned int greater = 0;
        unsigned int threshold_hi = 0;
        for (int h = 255; h >= 0; --h) {
            const unsigned int count = hist[h];
            if (greater + count >= TOPK) {
                threshold_hi = static_cast<unsigned int>(h);
                break;
            }
            greater += count;
        }
        state[0] = threshold_hi;
        state[1] = greater;
    }
    __syncthreads();

    hist[tid] = 0;
    __syncthreads();
    const unsigned int threshold_hi = state[0];
    for (int i = tid; i < K; i += blockDim.x) {
        const unsigned int key = dsv4_ordered_half(row_scores[i]);
        if ((key >> 8) == threshold_hi) {
            atomicAdd(&hist[key & 0xffu], 1u);
        }
    }
    __syncthreads();

    if (tid == 0) {
        unsigned int greater = state[1];
        unsigned int threshold_lo = 0;
        for (int l = 255; l >= 0; --l) {
            const unsigned int count = hist[l];
            if (greater + count >= TOPK) {
                threshold_lo = static_cast<unsigned int>(l);
                break;
            }
            greater += count;
        }
        state[2] = (threshold_hi << 8) | threshold_lo;
        counters[0] = 0;
        counters[1] = greater;
    }
    __syncthreads();

    const unsigned int threshold_key = state[2];
    for (int base = 0; base < K; base += blockDim.x) {
        const int i = base + tid;
        if (i < K) {
            const unsigned int key = dsv4_ordered_half(row_scores[i]);
            if (key > threshold_key) {
                const unsigned int pos = atomicAdd(&counters[0], 1u);
                if (pos < TOPK) row_out[pos] = i;
            } else if (key == threshold_key) {
                const unsigned int pos = atomicAdd(&counters[1], 1u);
                if (pos < TOPK) row_out[pos] = i;
            }
        }
        __syncthreads();
    }
}

__global__ void dsv4_prefill_plan_kernel(
    const int * __restrict__ topk,
    int * __restrict__ indices,
    half * __restrict__ mask,
    int B,
    int M,
    int topk_count,
    int selected,
    int query_offset,
    int local_history,
    int pool_len,
    int ratio,
    int window)
{
    const int64_t total = static_cast<int64_t>(B) * M * selected;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int slot = linear % selected;
        const int row = linear / selected;
        const int query = row % M;
        const int raw_len = local_history + M;
        int index = 0;
        bool valid = false;
        if (slot < window) {
            const int local_end = local_history + query + 1;
            const int local_count = min(window, local_end);
            if (slot < local_count) {
                index = local_end - local_count + slot;
                valid = true;
            }
        } else if (slot < window + topk_count) {
            const int pooled = topk[
                static_cast<int64_t>(row) * topk_count + slot - window];
            const int visible = min(
                pool_len, (query_offset + query + 1) / ratio);
            if (pooled >= 0 && pooled < visible) {
                index = raw_len + pooled;
                valid = true;
            }
        }
        indices[linear] = index;
        mask[linear] = valid ? __float2half(0.0f) : __float2half(-INFINITY);
    }
}

__global__ void dsv4_decode_plan_kernel(
    const int * __restrict__ topk,
    const int64_t * __restrict__ seq_len,
    int * __restrict__ indices,
    half * __restrict__ mask,
    int B,
    int topk_count,
    int selected,
    int pool_len,
    int ratio,
    int window)
{
    const int64_t total = static_cast<int64_t>(B) * selected;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int slot = linear % selected;
        const int batch = linear / selected;
        const int64_t length = seq_len[batch];
        int index = 0;
        bool valid = false;
        if (slot < window) {
            const int local_count = static_cast<int>(
                length < window ? length : window);
            if (slot < local_count) {
                const int64_t absolute = length - local_count + slot;
                index = static_cast<int>(absolute % window);
                valid = true;
            }
        } else if (slot < window + topk_count) {
            const int pooled = topk[
                static_cast<int64_t>(batch) * topk_count + slot - window];
            const int64_t pooled_visible = length / ratio;
            const int visible = static_cast<int>(
                pooled_visible < pool_len ? pooled_visible : pool_len);
            if (pooled >= 0 && pooled < visible) {
                index = window + pooled;
                valid = true;
            }
        }
        indices[linear] = index;
        mask[linear] = valid ? __float2half(0.0f) : __float2half(-INFINITY);
    }
}

template<int ncols1, int ncols2>
__launch_bounds__(
    ggml_cuda_fattn_mma_get_nthreads(kHeadDim, kHeadDim, ncols1 * ncols2),
    ggml_cuda_fattn_mma_get_occupancy(kHeadDim, kHeadDim, ncols1 * ncols2))
__global__ void dsv4_sparse_attention_kernel(
    const float * __restrict__ q,
    const half * __restrict__ kv,
    const int * __restrict__ indices,
    const half * __restrict__ mask,
    const float * __restrict__ sinks,
    float * __restrict__ out,
    float2 * __restrict__ meta,
    float scale,
    int B,
    int M,
    int max_seq,
    int selected,
    uint3 ne01)
{
#if defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE)
    constexpr int ncols = ncols1 * ncols2;
    constexpr int nbatch_fa =
        ggml_cuda_fattn_mma_get_nbatch_fa(kHeadDim, kHeadDim, ncols);
    constexpr int nthreads =
        ggml_cuda_fattn_mma_get_nthreads(kHeadDim, kHeadDim, ncols);
    constexpr int nwarps = nthreads / 32;
    constexpr int gqa_ratio = kHeads;
    constexpr int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;

    const int iter_k = (selected + nbatch_fa - 1) / nbatch_fa;
    const int iter_j = (M + ncols1 - 1) / ncols1;
    int kbc = static_cast<int64_t>(blockIdx.x) *
        (iter_k * iter_j * iter_z_gqa * B) / gridDim.x;
    const int kbc_stop = static_cast<int64_t>(blockIdx.x + 1) *
        (iter_k * iter_j * iter_z_gqa * B) / gridDim.x;

    int kb0_start = kbc % iter_k;
    int kb0_stop = min(iter_k, kb0_start + kbc_stop - kbc);
    constexpr bool use_logit_softcap = false;
    constexpr bool V_is_K_view = true;
    constexpr bool use_indirect = true;

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int sequence = kbc / (iter_k * iter_j * iter_z_gqa);
        const int zt_gqa =
            (kbc - iter_k * iter_j * iter_z_gqa * sequence) /
            (iter_k * iter_j);
        const int jt =
            (kbc - iter_k * iter_j * iter_z_gqa * sequence -
             iter_k * iter_j * zt_gqa) / iter_k;
        const int zt_q = zt_gqa * ncols2;
        const float2 * q_f2 = reinterpret_cast<const float2 *>(q) +
            (static_cast<int64_t>(sequence) * kHeads * M +
             static_cast<int64_t>(zt_q) * M) * (kHeadDim / 2);
        const half2 * kv_h2 = reinterpret_cast<const half2 *>(kv) +
            static_cast<int64_t>(sequence) * max_seq * (kHeadDim / 2);
        float2 * dst = reinterpret_cast<float2 *>(out) +
            (static_cast<int64_t>(sequence) * M * kHeads + zt_q) *
                (kHeadDim / 2);
        const int * row_indices = indices +
            (static_cast<int64_t>(sequence) * M + jt) * selected;
        const half * sequence_mask =
            mask + static_cast<int64_t>(sequence) * M * selected;
        constexpr bool is_fixup = false;
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false;
            flash_attn_ext_f16_process_tile<
                kHeadDim, kHeadDim, ncols1, ncols2, nwarps,
                use_logit_softcap, V_is_K_view, needs_fixup, is_fixup,
                use_indirect>(
                q_f2, kv_h2, kv_h2, sequence_mask, sinks + zt_q,
                dst, meta, scale, 1.0f, 0.0f, ne01, kHeads,
                gqa_ratio, selected, kHeadDim / 2,
                M * (kHeadDim / 2), kHeadDim / 2, kHeadDim / 2,
                selected, jt, zt_gqa, kb0_start, kb0_stop, row_indices);
        } else {
            constexpr bool needs_fixup = true;
            flash_attn_ext_f16_process_tile<
                kHeadDim, kHeadDim, ncols1, ncols2, nwarps,
                use_logit_softcap, V_is_K_view, needs_fixup, is_fixup,
                use_indirect>(
                q_f2, kv_h2, kv_h2, sequence_mask, sinks + zt_q,
                dst, meta, scale, 1.0f, 0.0f, ne01, kHeads,
                gqa_ratio, selected, kHeadDim / 2,
                M * (kHeadDim / 2), kHeadDim / 2, kHeadDim / 2,
                selected, jt, zt_gqa, kb0_start, kb0_stop, row_indices);
        }
        kbc += iter_k;
        kbc -= kbc % iter_k;
        kb0_start = 0;
        kb0_stop = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) return;
    const int sequence = kbc / (iter_k * iter_j * iter_z_gqa);
    const int zt_gqa =
        (kbc - iter_k * iter_j * iter_z_gqa * sequence) /
        (iter_k * iter_j);
    const int jt =
        (kbc - iter_k * iter_j * iter_z_gqa * sequence -
         iter_k * iter_j * zt_gqa) / iter_k;
    const int zt_q = zt_gqa * ncols2;
    const float2 * q_f2 = reinterpret_cast<const float2 *>(q) +
        (static_cast<int64_t>(sequence) * kHeads * M +
         static_cast<int64_t>(zt_q) * M) * (kHeadDim / 2);
    const half2 * kv_h2 = reinterpret_cast<const half2 *>(kv) +
        static_cast<int64_t>(sequence) * max_seq * (kHeadDim / 2);
    float2 * dst = reinterpret_cast<float2 *>(out) +
        (static_cast<int64_t>(sequence) * M * kHeads + zt_q) *
            (kHeadDim / 2);
    const int * row_indices = indices +
        (static_cast<int64_t>(sequence) * M + jt) * selected;
    const half * sequence_mask =
        mask + static_cast<int64_t>(sequence) * M * selected;
    constexpr bool needs_fixup = false;
    constexpr bool is_fixup = true;
    flash_attn_ext_f16_process_tile<
        kHeadDim, kHeadDim, ncols1, ncols2, nwarps,
        use_logit_softcap, V_is_K_view, needs_fixup, is_fixup,
        use_indirect>(
        q_f2, kv_h2, kv_h2, sequence_mask, sinks + zt_q,
        dst, meta, scale, 1.0f, 0.0f, ne01, kHeads,
        gqa_ratio, selected, kHeadDim / 2,
        M * (kHeadDim / 2), kHeadDim / 2, kHeadDim / 2,
        selected, jt, zt_gqa, kb0_start, kb0_stop, row_indices);
#else
    (void)q;
    (void)kv;
    (void)indices;
    (void)mask;
    (void)sinks;
    (void)out;
    (void)meta;
    (void)scale;
    (void)B;
    (void)M;
    (void)max_seq;
    (void)selected;
    (void)ne01;
#endif
}

template<int ncols1, int ncols2>
mfq_tensor_backend::Tensor launch_dsv4_sparse_attention(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor mask,
    mfq_tensor_backend::Tensor sinks,
    mfq_tensor_backend::Tensor meta,
    double scale)
{
    constexpr int ncols = ncols1 * ncols2;
    const int B = static_cast<int>(q.size(0));
    const int M = static_cast<int>(q.size(2));
    const int max_seq = static_cast<int>(kv.size(1));
    const int selected = static_cast<int>(indices.size(2));
    int device = 0;
    cudaDeviceProp properties{};
    MFQ_RUNTIME_CHECK(
        cudaGetDevice(&device) == cudaSuccess,
        "dsv4_sparse_attention: cudaGetDevice failed");
    MFQ_RUNTIME_CHECK(
        cudaGetDeviceProperties(&properties, device) == cudaSuccess,
        "dsv4_sparse_attention: cudaGetDeviceProperties failed");
    const int cc = properties.major * 100 + properties.minor * 10;
    const int nthreads =
        ggml_cuda_fattn_mma_get_nthreads(kHeadDim, kHeadDim, ncols, cc);
    const int nwarps = nthreads / 32;
    const int nbatch_fa =
        ggml_cuda_fattn_mma_get_nbatch_fa(kHeadDim, kHeadDim, ncols, cc);
    const int nbatch_k2 =
        ggml_cuda_fattn_mma_get_nbatch_K2(kHeadDim, kHeadDim, ncols, cc);
    const int nbatch_v2 =
        ggml_cuda_fattn_mma_get_nbatch_V2(kHeadDim, kHeadDim, ncols, cc);
    const int nbatch_combine =
        ggml_cuda_fattn_mma_get_nbatch_combine(
            kHeadDim, kHeadDim, ncols, cc);
    const bool q_in_reg =
        ggml_cuda_fattn_mma_get_Q_in_reg(kHeadDim, kHeadDim, ncols, cc);
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const size_t shared_kv = static_cast<size_t>(nbatch_fa) *
        (std::max(nbatch_k2, nbatch_v2) + 4) * sizeof(half2);
    const size_t shared_q =
        static_cast<size_t>(ncols) * (kHeadDim / 2 + 4) * sizeof(half2);
    const size_t shared_mask =
        static_cast<size_t>(ncols1) * (nbatch_fa / 2 + 4) * sizeof(half2);
    const size_t shared_combine =
        static_cast<size_t>(nwarps) * cols_per_warp *
        (nbatch_combine + 4) * sizeof(half2);
    const size_t shmem = std::max(
        shared_combine,
        q_in_reg
            ? std::max(shared_q, shared_kv + shared_mask)
            : shared_q + shared_kv + shared_mask);

    using Kernel = decltype(
        &dsv4_sparse_attention_kernel<ncols1, ncols2>);
    Kernel kernel = dsv4_sparse_attention_kernel<ncols1, ncols2>;
    static bool shmem_set[32] = {};
    MFQ_RUNTIME_CHECK(
        device >= 0 && device < 32,
        "dsv4_sparse_attention: unsupported CUDA device index");
    if (!shmem_set[device]) {
        const cudaError_t status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shmem));
        MFQ_RUNTIME_CHECK(
            status == cudaSuccess,
            "dsv4_sparse_attention: shared-memory attribute failed: ",
            cudaGetErrorString(status));
        shmem_set[device] = true;
    }

    const int iter_z_gqa = kHeads / ncols2;
    const int ntiles_dst = B * M * iter_z_gqa;
    const int ntiles_kv = (selected + nbatch_fa - 1) / nbatch_fa;
    int max_blocks_per_sm = 0;
    cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, kernel, nthreads, shmem);
    MFQ_RUNTIME_CHECK(
        status == cudaSuccess && max_blocks_per_sm > 0,
        "dsv4_sparse_attention: occupancy query failed: ",
        cudaGetErrorString(status));
    const int resident_blocks =
        max_blocks_per_sm * properties.multiProcessorCount;
    const int raw_blocks =
        std::min(resident_blocks, ntiles_dst * ntiles_kv);
    const int rounded_blocks = std::max(
        ntiles_dst, (raw_blocks / ntiles_dst) * ntiles_dst);
    const int blocks_per_tile = rounded_blocks / ntiles_dst;
    const size_t meta_float2 =
        static_cast<size_t>(rounded_blocks) * ncols *
        (2 + kHeadDim / 2);
    MFQ_RUNTIME_CHECK(
        blocks_per_tile == 1 ||
            static_cast<size_t>(meta.numel()) >= 2 * meta_float2,
        "dsv4_sparse_attention: meta workspace too small, need ",
        2 * meta_float2, " float elements");

    auto out = mfq_tensor_backend::empty({B, M, kHeads, kHeadDim}, q.options());
    auto stream = mfq_current_cuda_stream();
    kernel<<<rounded_blocks, dim3(32, nwarps, 1), shmem, stream>>>(
        q.data_ptr<float>(),
        reinterpret_cast<const half *>(kv.data_ptr<mfq_half>()),
        indices.data_ptr<int>(),
        reinterpret_cast<const half *>(mask.data_ptr<mfq_half>()),
        sinks.data_ptr<float>(),
        static_cast<float*>(out.data_ptr()),
        reinterpret_cast<float2 *>(meta.data_ptr<float>()),
        static_cast<float>(scale), B, M, max_seq, selected,
        init_fastdiv_values(M));
    status = cudaGetLastError();
    MFQ_RUNTIME_CHECK(
        status == cudaSuccess,
        "dsv4_sparse_attention launch failed: ",
        cudaGetErrorString(status));

    if (blocks_per_tile > 1) {
        const uint3 fd0 = init_fastdiv_values(M * iter_z_gqa);
        const uint3 fd1 = init_fastdiv_values(M * iter_z_gqa);
        const uint3 fd2 = init_fastdiv_values(M);
        flash_attn_stream_k_fixup_uniform<
            kHeadDim, ncols1, ncols2>
            <<<dim3(ntiles_dst, ncols1, ncols2), kHeadDim, 0, stream>>>(
                static_cast<float*>(out.data_ptr()),
                reinterpret_cast<const float2 *>(meta.data_ptr<float>()),
                M, kHeads, 1, rounded_blocks, kHeads,
                blocks_per_tile, fd0, fd1, fd2);
        status = cudaGetLastError();
        MFQ_RUNTIME_CHECK(
            status == cudaSuccess,
            "dsv4_sparse_attention fixup failed: ",
            cudaGetErrorString(status));
    }
    return out;
}

} // namespace

mfq_tensor_backend::Tensor dsv4_fp4_sim_cuda(mfq_tensor_backend::Tensor input) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
            input.scalar_type() == mfq_tensor_backend::kFloat16 &&
            input.dim() >= 1 && input.size(-1) % 32 == 0,
        "dsv4_fp4_sim: expected contiguous CUDA f16 with last dimension divisible by 32");
    auto output = mfq_tensor_backend::empty_like(input);
    const int64_t groups = input.numel() / 32;
    MFQ_RUNTIME_CHECK(
        groups > 0 && groups <= std::numeric_limits<int>::max(),
        "dsv4_fp4_sim: invalid group count");
    dsv4_fp4_sim_kernel<<<
        static_cast<int>(groups), 32, 0,
        mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half *>(input.data_ptr<mfq_half>()),
        reinterpret_cast<half *>(output.data_ptr<mfq_half>()));
    const cudaError_t status = cudaGetLastError();
    MFQ_RUNTIME_CHECK(
        status == cudaSuccess,
        "dsv4_fp4_sim launch failed: ",
        cudaGetErrorString(status));
    return output;
}

mfq_tensor_backend::Tensor dsv4_compress_cuda(
    mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor gate,
    mfq_tensor_backend::Tensor ape,
    mfq_tensor_backend::Tensor norm,
    mfq_tensor_backend::Tensor prev_kv,
    mfq_tensor_backend::Tensor prev_gate,
    mfq_tensor_backend::Tensor positions,
    mfq_tensor_backend::Tensor cos,
    mfq_tensor_backend::Tensor sin,
    int64_t ratio,
    bool overlap,
    int64_t quant_mode,
    double eps)
{
    const auto scalar_type = kv.scalar_type();
    MFQ_RUNTIME_CHECK(
        kv.is_cuda() && kv.is_contiguous() &&
            (scalar_type == mfq_tensor_backend::kFloat16 ||
             scalar_type == mfq_tensor_backend::kFloat32) &&
            gate.is_cuda() && gate.is_contiguous() &&
            gate.scalar_type() == scalar_type &&
            kv.sizes() == gate.sizes(),
        "dsv4_compress: kv/gate must be matching contiguous CUDA f16/f32");
    const int64_t head_dim = norm.numel();
    MFQ_RUNTIME_CHECK(
        (head_dim == kHeadDim || head_dim == kIndexerDim) &&
            kv.dim() == 4 && kv.size(2) == ratio &&
            ratio > 0 && ratio <= 128 &&
            kv.size(3) == (overlap ? 2 * head_dim : head_dim),
        "dsv4_compress: expected [B,W,ratio,D or 2D], D=128 or 512");
    MFQ_RUNTIME_CHECK(
        ape.is_cuda() && ape.is_contiguous() &&
            ape.scalar_type() == mfq_tensor_backend::kFloat32 &&
            ape.dim() == 2 && ape.size(0) == ratio &&
            ape.size(1) == kv.size(3) &&
            norm.is_cuda() && norm.is_contiguous() &&
            norm.scalar_type() == mfq_tensor_backend::kFloat32 &&
            norm.numel() == head_dim,
        "dsv4_compress: invalid APE or norm");
    MFQ_RUNTIME_CHECK(
        positions.is_cuda() && positions.is_contiguous() &&
            positions.scalar_type() == mfq_tensor_backend::kInt64 &&
            positions.numel() == kv.size(0) * kv.size(1) &&
            cos.is_cuda() && cos.is_contiguous() &&
            cos.scalar_type() == mfq_tensor_backend::kFloat32 &&
            sin.is_cuda() && sin.is_contiguous() &&
            sin.scalar_type() == mfq_tensor_backend::kFloat32 &&
            cos.dim() == 2 && sin.sizes() == cos.sizes() &&
            cos.size(1) >= 32,
        "dsv4_compress: invalid positions or RoPE table");
    const bool has_prev = prev_kv.numel() != 0;
    MFQ_RUNTIME_CHECK(
        !has_prev ||
            (overlap && prev_kv.is_cuda() && prev_kv.is_contiguous() &&
             prev_kv.scalar_type() == scalar_type &&
             prev_gate.is_cuda() && prev_gate.is_contiguous() &&
             prev_gate.scalar_type() == scalar_type &&
             prev_kv.sizes() == prev_gate.sizes() &&
             prev_kv.dim() == 3 && prev_kv.size(0) == kv.size(0) &&
             prev_kv.size(1) == ratio && prev_kv.size(2) == head_dim),
        "dsv4_compress: invalid overlap state");
    MFQ_RUNTIME_CHECK(
        quant_mode == 0 ||
            (quant_mode == 1 && head_dim == kHeadDim) ||
            (quant_mode == 2 && head_dim == kIndexerDim),
        "dsv4_compress: quant_mode must match D=512 FP8 or D=128 FP4 cache");
    auto out = mfq_tensor_backend::empty(
        {kv.size(0), kv.size(1), head_dim},
        kv.options().dtype(mfq_tensor_backend::kFloat16));
    const int blocks =
        static_cast<int>(kv.size(0) * kv.size(1));
    auto launch = [&](auto scalar_tag, auto head_tag) {
        using scalar_t = typename decltype(scalar_tag)::type;
        constexpr int D = decltype(head_tag)::value;
        dsv4_compress_kernel<scalar_t, D><<<
            blocks, D, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const scalar_t *>(kv.data_ptr()),
            reinterpret_cast<const scalar_t *>(gate.data_ptr()),
            ape.data_ptr<float>(), norm.data_ptr<float>(),
            has_prev
                ? reinterpret_cast<const scalar_t *>(prev_kv.data_ptr())
                : nullptr,
            has_prev
                ? reinterpret_cast<const scalar_t *>(prev_gate.data_ptr())
                : nullptr,
            positions.data_ptr<int64_t>(), cos.data_ptr<float>(),
            sin.data_ptr<float>(),
            reinterpret_cast<half *>(out.data_ptr<mfq_half>()),
            static_cast<int>(kv.size(0)), static_cast<int>(kv.size(1)),
            static_cast<int>(ratio), static_cast<int>(kv.size(3)),
            overlap ? 1 : 0, has_prev ? 1 : 0, 64,
            static_cast<int>(cos.size(1)), static_cast<int>(quant_mode),
            static_cast<float>(eps));
    };
    if (scalar_type == mfq_tensor_backend::kFloat32 && head_dim == kHeadDim) {
        launch(Dsv4TypeTag<float>{},
               std::integral_constant<int, kHeadDim>{});
    } else if (scalar_type == mfq_tensor_backend::kFloat32) {
        launch(Dsv4TypeTag<float>{},
               std::integral_constant<int, kIndexerDim>{});
    } else if (head_dim == kHeadDim) {
        launch(Dsv4TypeTag<half>{},
               std::integral_constant<int, kHeadDim>{});
    } else {
        launch(Dsv4TypeTag<half>{},
               std::integral_constant<int, kIndexerDim>{});
    }
    const cudaError_t status = cudaGetLastError();
    MFQ_RUNTIME_CHECK(
        status == cudaSuccess,
        "dsv4_compress launch failed: ", cudaGetErrorString(status));
    return out;
}

mfq_tensor_backend::Tensor dsv4_decode_pool_update_cuda(
    mfq_tensor_backend::Tensor kv_token,
    mfq_tensor_backend::Tensor gate_token,
    mfq_tensor_backend::Tensor ape,
    mfq_tensor_backend::Tensor norm,
    mfq_tensor_backend::Tensor state_kv,
    mfq_tensor_backend::Tensor state_gate,
    mfq_tensor_backend::Tensor prev_kv,
    mfq_tensor_backend::Tensor prev_gate,
    mfq_tensor_backend::Tensor pool,
    mfq_tensor_backend::Tensor seq_len,
    mfq_tensor_backend::Tensor cos,
    mfq_tensor_backend::Tensor sin,
    int64_t ratio,
    bool overlap,
    int64_t quant_mode,
    double eps)
{
    const int64_t head_dim = norm.numel();
    const auto scalar_type = kv_token.scalar_type();
    MFQ_RUNTIME_CHECK(
        head_dim == kHeadDim || head_dim == kIndexerDim,
        "dsv4_decode_pool_update: D must be 128 or 512");
    const int64_t out_dim = overlap ? 2 * head_dim : head_dim;
    MFQ_RUNTIME_CHECK(
        kv_token.is_cuda() && kv_token.is_contiguous() &&
            (scalar_type == mfq_tensor_backend::kFloat16 ||
             scalar_type == mfq_tensor_backend::kFloat32) &&
            gate_token.is_cuda() && gate_token.is_contiguous() &&
            gate_token.scalar_type() == scalar_type &&
            kv_token.sizes() == gate_token.sizes() &&
            kv_token.dim() == 3 && kv_token.size(1) == 1 &&
            kv_token.size(2) == out_dim,
        "dsv4_decode_pool_update: invalid projected token");
    const int64_t B = kv_token.size(0);
    MFQ_RUNTIME_CHECK(
        state_kv.is_cuda() && state_kv.is_contiguous() &&
            state_kv.scalar_type() == scalar_type &&
            state_gate.is_cuda() && state_gate.is_contiguous() &&
            state_gate.scalar_type() == scalar_type &&
            state_kv.sizes() == state_gate.sizes() &&
            state_kv.dim() == 3 && state_kv.size(0) == B &&
            state_kv.size(1) == ratio && state_kv.size(2) == out_dim &&
            ratio > 0 && ratio <= 128,
        "dsv4_decode_pool_update: invalid remainder state");
    MFQ_RUNTIME_CHECK(
        !overlap ||
            (prev_kv.is_cuda() && prev_kv.is_contiguous() &&
             prev_kv.scalar_type() == scalar_type &&
             prev_gate.is_cuda() && prev_gate.is_contiguous() &&
             prev_gate.scalar_type() == scalar_type &&
             prev_kv.sizes() == prev_gate.sizes() &&
             prev_kv.dim() == 3 && prev_kv.size(0) == B &&
             prev_kv.size(1) == ratio && prev_kv.size(2) == head_dim),
        "dsv4_decode_pool_update: invalid overlap history");
    MFQ_RUNTIME_CHECK(
        pool.is_cuda() && pool.is_contiguous() &&
            pool.scalar_type() == mfq_tensor_backend::kFloat16 &&
            pool.dim() == 3 && pool.size(0) == B &&
            pool.size(1) > 0 && pool.size(2) == head_dim &&
            seq_len.is_cuda() && seq_len.is_contiguous() &&
            seq_len.scalar_type() == mfq_tensor_backend::kInt64 &&
            seq_len.numel() == B,
        "dsv4_decode_pool_update: invalid pool or sequence length");
    MFQ_RUNTIME_CHECK(
        ape.is_cuda() && ape.is_contiguous() &&
            ape.scalar_type() == mfq_tensor_backend::kFloat32 &&
            ape.dim() == 2 && ape.size(0) == ratio &&
            ape.size(1) == out_dim &&
            norm.is_cuda() && norm.is_contiguous() &&
            norm.scalar_type() == mfq_tensor_backend::kFloat32 &&
            norm.numel() == head_dim &&
            cos.is_cuda() && cos.is_contiguous() &&
            cos.scalar_type() == mfq_tensor_backend::kFloat32 &&
            sin.is_cuda() && sin.is_contiguous() &&
            sin.scalar_type() == mfq_tensor_backend::kFloat32 &&
            cos.dim() == 2 && sin.sizes() == cos.sizes() &&
            cos.size(1) >= 32,
        "dsv4_decode_pool_update: invalid compressor parameters");
    MFQ_RUNTIME_CHECK(
        quant_mode == 0 ||
            (quant_mode == 1 && head_dim == kHeadDim) ||
            (quant_mode == 2 && head_dim == kIndexerDim),
        "dsv4_decode_pool_update: quant_mode must match D=512 FP8 or D=128 FP4 cache");
    auto launch = [&](auto scalar_tag, auto head_tag) {
        using scalar_t = typename decltype(scalar_tag)::type;
        constexpr int D = decltype(head_tag)::value;
        auto empty_state =
            reinterpret_cast<scalar_t *>(state_kv.data_ptr());
        dsv4_decode_pool_update_kernel<scalar_t, D><<<
            static_cast<int>(B), D, 0,
            mfq_current_cuda_stream()>>>(
            reinterpret_cast<const scalar_t *>(kv_token.data_ptr()),
            reinterpret_cast<const scalar_t *>(gate_token.data_ptr()),
            ape.data_ptr<float>(), norm.data_ptr<float>(),
            empty_state,
            reinterpret_cast<scalar_t *>(state_gate.data_ptr()),
            overlap
                ? reinterpret_cast<scalar_t *>(prev_kv.data_ptr())
                : empty_state,
            overlap
                ? reinterpret_cast<scalar_t *>(prev_gate.data_ptr())
                : empty_state,
            reinterpret_cast<half *>(pool.data_ptr<mfq_half>()),
            seq_len.data_ptr<int64_t>(), cos.data_ptr<float>(),
            sin.data_ptr<float>(), static_cast<int>(B),
            static_cast<int>(ratio), static_cast<int>(out_dim),
            overlap ? 1 : 0, static_cast<int>(pool.size(1)), 64,
            static_cast<int>(cos.size(1)), static_cast<int>(quant_mode),
            static_cast<float>(eps));
    };
    if (scalar_type == mfq_tensor_backend::kFloat32 && head_dim == kHeadDim) {
        launch(Dsv4TypeTag<float>{},
               std::integral_constant<int, kHeadDim>{});
    } else if (scalar_type == mfq_tensor_backend::kFloat32) {
        launch(Dsv4TypeTag<float>{},
               std::integral_constant<int, kIndexerDim>{});
    } else if (head_dim == kHeadDim) {
        launch(Dsv4TypeTag<half>{},
               std::integral_constant<int, kHeadDim>{});
    } else {
        launch(Dsv4TypeTag<half>{},
               std::integral_constant<int, kIndexerDim>{});
    }
    const cudaError_t status = cudaGetLastError();
    MFQ_RUNTIME_CHECK(
        status == cudaSuccess,
        "dsv4_decode_pool_update launch failed: ",
        cudaGetErrorString(status));
    return pool;
}

mfq_tensor_backend::Tensor dsv4_indexer_scores_cuda(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor k,
    mfq_tensor_backend::Tensor weights,
    int64_t query_offset,
    int64_t ratio)
{
    MFQ_RUNTIME_CHECK(
        q.is_cuda() && q.is_contiguous() &&
            q.scalar_type() == mfq_tensor_backend::kFloat16 &&
            k.is_cuda() && k.is_contiguous() &&
            k.scalar_type() == mfq_tensor_backend::kFloat16 &&
            weights.is_cuda() && weights.is_contiguous() &&
            weights.scalar_type() == mfq_tensor_backend::kFloat16,
        "dsv4_indexer_scores: tensors must be contiguous CUDA f16");
    MFQ_RUNTIME_CHECK(
        q.dim() == 4 && q.size(2) == kIndexerHeads &&
            q.size(3) == kIndexerDim &&
            k.dim() == 3 && k.size(0) == q.size(0) &&
            k.size(2) == kIndexerDim &&
            weights.dim() == 3 && weights.size(0) == q.size(0) &&
            weights.size(1) == q.size(1) &&
            weights.size(2) == kIndexerHeads &&
            query_offset >= 0 && ratio > 0,
        "dsv4_indexer_scores: shape mismatch");
    const int B = static_cast<int>(q.size(0));
    const int M = static_cast<int>(q.size(1));
    const int K = static_cast<int>(k.size(1));
    auto out = mfq_tensor_backend::empty({B, M, K}, q.options());
    const dim3 grid(
        (K + kIndexerKeysPerTile - 1) / kIndexerKeysPerTile, M, B);
    dsv4_indexer_scores_kernel<<<
        grid, dim3(32, 16, 1), 0,
        mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half *>(q.data_ptr<mfq_half>()),
        reinterpret_cast<const half *>(k.data_ptr<mfq_half>()),
        reinterpret_cast<const half *>(weights.data_ptr<mfq_half>()),
        reinterpret_cast<half *>(out.data_ptr<mfq_half>()),
        B, M, K, static_cast<int>(query_offset),
        static_cast<int>(ratio),
        1.0f / std::sqrt(
            static_cast<float>(kIndexerDim * kIndexerHeads)));
    const cudaError_t status = cudaGetLastError();
    MFQ_RUNTIME_CHECK(
        status == cudaSuccess,
        "dsv4_indexer_scores launch failed: ",
        cudaGetErrorString(status));
    return out;
}

mfq_tensor_backend::Tensor dsv4_topk512_cuda(mfq_tensor_backend::Tensor scores) {
    MFQ_RUNTIME_CHECK(
        scores.is_cuda() && scores.is_contiguous() &&
            scores.scalar_type() == mfq_tensor_backend::kFloat16 &&
            scores.dim() == 3 && scores.size(2) > 0 &&
            scores.size(0) * scores.size(1) <= INT_MAX &&
            scores.size(2) <= INT_MAX,
        "dsv4_topk512: expected contiguous CUDA f16 [B,M,K]");
    const int rows =
        static_cast<int>(scores.size(0) * scores.size(1));
    const int K = static_cast<int>(scores.size(2));
    auto out = mfq_tensor_backend::empty(
        {scores.size(0), scores.size(1), kIndexerTopK},
        scores.options().dtype(mfq_tensor_backend::kInt32));
    dsv4_topk_indices_kernel<kIndexerTopK><<<
        rows, 256, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half *>(scores.data_ptr<mfq_half>()),
        out.data_ptr<int>(), rows, K);
    const cudaError_t status = cudaGetLastError();
    MFQ_RUNTIME_CHECK(
        status == cudaSuccess,
        "dsv4_topk512 launch failed: ", cudaGetErrorString(status));
    return out;
}

std::vector<mfq_tensor_backend::Tensor> dsv4_build_prefill_plan_cuda(
    mfq_tensor_backend::Tensor topk,
    int64_t query_offset,
    int64_t local_history,
    int64_t pool_len,
    int64_t ratio,
    int64_t window)
{
    MFQ_RUNTIME_CHECK(
        topk.is_cuda() && topk.is_contiguous() &&
            topk.scalar_type() == mfq_tensor_backend::kInt32 &&
            topk.dim() == 3 && topk.size(1) > 0 &&
            query_offset >= 0 && local_history >= 0 &&
            pool_len >= 0 && ratio > 0 && window > 0,
        "dsv4_prefill_plan: invalid input");
    const int B = static_cast<int>(topk.size(0));
    const int M = static_cast<int>(topk.size(1));
    const int topk_count = static_cast<int>(topk.size(2));
    const int selected =
        ((static_cast<int>(window) + topk_count + 31) / 32) * 32;
    auto indices = mfq_tensor_backend::empty(
        {B, M, selected}, topk.options());
    auto mask = mfq_tensor_backend::empty(
        {B, M, selected},
        topk.options().dtype(mfq_tensor_backend::kFloat16));
    const int64_t total =
        static_cast<int64_t>(B) * M * selected;
    const int blocks = std::min<int64_t>(
        65535, std::max<int64_t>(1, (total + 255) / 256));
    dsv4_prefill_plan_kernel<<<
        blocks, 256, 0, mfq_current_cuda_stream()>>>(
        topk.data_ptr<int>(), indices.data_ptr<int>(),
        reinterpret_cast<half *>(mask.data_ptr<mfq_half>()),
        B, M, topk_count, selected, static_cast<int>(query_offset),
        static_cast<int>(local_history), static_cast<int>(pool_len),
        static_cast<int>(ratio), static_cast<int>(window));
    const cudaError_t status = cudaGetLastError();
    MFQ_RUNTIME_CHECK(
        status == cudaSuccess,
        "dsv4_prefill_plan launch failed: ",
        cudaGetErrorString(status));
    return {indices, mask};
}

std::vector<mfq_tensor_backend::Tensor> dsv4_build_decode_plan_cuda(
    mfq_tensor_backend::Tensor topk,
    mfq_tensor_backend::Tensor seq_len,
    int64_t pool_len,
    int64_t ratio,
    int64_t window)
{
    MFQ_RUNTIME_CHECK(
        topk.is_cuda() && topk.is_contiguous() &&
            topk.scalar_type() == mfq_tensor_backend::kInt32 &&
            topk.dim() == 3 && topk.size(1) == 1 &&
            seq_len.is_cuda() && seq_len.is_contiguous() &&
            seq_len.scalar_type() == mfq_tensor_backend::kInt64 &&
            seq_len.numel() == topk.size(0) &&
            pool_len >= 0 && ratio > 0 && window > 0,
        "dsv4_decode_plan: invalid input");
    const int B = static_cast<int>(topk.size(0));
    const int topk_count = static_cast<int>(topk.size(2));
    const int selected =
        ((static_cast<int>(window) + topk_count + 31) / 32) * 32;
    auto indices = mfq_tensor_backend::empty(
        {B, 1, selected}, topk.options());
    auto mask = mfq_tensor_backend::empty(
        {B, 1, selected},
        topk.options().dtype(mfq_tensor_backend::kFloat16));
    const int64_t total = static_cast<int64_t>(B) * selected;
    const int blocks = std::min<int64_t>(
        65535, std::max<int64_t>(1, (total + 255) / 256));
    dsv4_decode_plan_kernel<<<
        blocks, 256, 0, mfq_current_cuda_stream()>>>(
        topk.data_ptr<int>(), seq_len.data_ptr<int64_t>(),
        indices.data_ptr<int>(),
        reinterpret_cast<half *>(mask.data_ptr<mfq_half>()),
        B, topk_count, selected, static_cast<int>(pool_len),
        static_cast<int>(ratio), static_cast<int>(window));
    const cudaError_t status = cudaGetLastError();
    MFQ_RUNTIME_CHECK(
        status == cudaSuccess,
        "dsv4_decode_plan launch failed: ",
        cudaGetErrorString(status));
    return {indices, mask};
}

mfq_tensor_backend::Tensor attention_dsv4_sparse_cuda(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor mask,
    mfq_tensor_backend::Tensor sinks,
    mfq_tensor_backend::Tensor meta,
    double scale)
{
    MFQ_RUNTIME_CHECK(
        q.is_cuda() && q.is_contiguous() &&
            q.scalar_type() == mfq_tensor_backend::kFloat32 &&
            kv.is_cuda() && kv.is_contiguous() &&
            kv.scalar_type() == mfq_tensor_backend::kFloat16 &&
            indices.is_cuda() && indices.is_contiguous() &&
            indices.scalar_type() == mfq_tensor_backend::kInt32 &&
            mask.is_cuda() && mask.is_contiguous() &&
            mask.scalar_type() == mfq_tensor_backend::kFloat16 &&
            sinks.is_cuda() && sinks.is_contiguous() &&
            sinks.scalar_type() == mfq_tensor_backend::kFloat32 &&
            meta.is_cuda() && meta.is_contiguous() &&
            meta.scalar_type() == mfq_tensor_backend::kFloat32,
        "dsv4_sparse_attention: unsupported dtype or placement");
    MFQ_RUNTIME_CHECK(
        q.dim() == 4 && q.size(1) == kHeads &&
            q.size(3) == kHeadDim &&
            kv.dim() == 3 && kv.size(0) == q.size(0) &&
            kv.size(2) == kHeadDim &&
            indices.dim() == 3 && indices.size(0) == q.size(0) &&
            indices.size(1) == q.size(2) &&
            indices.size(2) > 0 && indices.size(2) % 32 == 0 &&
            mask.sizes() == indices.sizes() &&
            sinks.numel() == kHeads,
        "dsv4_sparse_attention: shape mismatch");
    return launch_dsv4_sparse_attention<1, kHeadsPerTile>(
        q, kv, indices, mask, sinks, meta, scale);
}
