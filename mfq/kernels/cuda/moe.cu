// GPU-resident MoE routing and NINT mul_mat_id kernels.
//
// The execution layout follows llama.cpp's CUDA mul_mat_id path:
//   * route ids are token-major [tokens, routes]
//   * large batches are compacted by expert into ids_dst/expert_bounds
//   * results are scattered back to token-major [tokens, routes, out]
//
// MFQ can avoid duplicating the activation during compaction. Shared expert
// inputs use [tokens, K], while routed/down-projection inputs use
// [tokens, routes, K]. ids_dst identifies either source row directly.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "cpp_runtime/moe_cache_transfer.h"
#include "glu.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 4;
constexpr int kRouteTile = 8;
constexpr int kMoeProfileNint4Gs24 = 0;
constexpr int kMoeProfileNint5Gs28 = 1;
constexpr int kMoeProfileNint6Gs24 = 2;
constexpr int kMoeProfileNint8Gs48 = 3;
constexpr int kMoeProfileNint8Gs24 = 4;
constexpr int kMoeProfileNint3Gs24 = 5;
constexpr int kMoeProfileNint2Gs16 = 6;

int g_moe_small_mmq_override = -1;

__global__ void moe_cache_scatter_kernel(
        const std::uint8_t * staging,
        std::int64_t descriptor_offset,
        int transfer_count) {
    const int transfer = static_cast<int>(blockIdx.x);
    if (transfer >= transfer_count) return;
    const auto * descriptors =
        reinterpret_cast<const mfq::MoeCacheScatterDescriptor *>(
            staging + descriptor_offset);
    const auto item = descriptors[transfer];
    auto * destination = reinterpret_cast<std::uint8_t *>(
        static_cast<std::uintptr_t>(item.destination));
    const auto * source = staging + item.source_offset;
    const std::uint64_t nbytes = item.nbytes;
    if ((((reinterpret_cast<std::uintptr_t>(destination) |
            reinterpret_cast<std::uintptr_t>(source) |
            static_cast<std::uintptr_t>(nbytes)) & 15u) == 0u)) {
        auto * output = reinterpret_cast<uint4 *>(destination);
        const auto * input = reinterpret_cast<const uint4 *>(source);
        const std::uint64_t count = nbytes / sizeof(uint4);
        for (std::uint64_t index = threadIdx.x;
             index < count;
             index += blockDim.x) {
            output[index] = input[index];
        }
        return;
    }
    for (std::uint64_t index = threadIdx.x;
         index < nbytes;
         index += blockDim.x) {
        destination[index] = source[index];
    }
}

__global__ void moe_cache_mapped_gather_kernel(
        const mfq::MoeCacheMappedCopyDescriptor * descriptors,
        int transfer_count) {
    const int transfer = static_cast<int>(blockIdx.y);
    if (transfer >= transfer_count) return;
    const auto item = descriptors[transfer];
    auto * destination = reinterpret_cast<std::uint8_t *>(
        static_cast<std::uintptr_t>(item.destination));
    const auto * source = reinterpret_cast<const std::uint8_t *>(
        static_cast<std::uintptr_t>(item.source));
    const std::uint64_t nbytes = item.nbytes;
    const std::uint64_t worker =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t workers =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    if ((((reinterpret_cast<std::uintptr_t>(destination) |
            reinterpret_cast<std::uintptr_t>(source) |
            static_cast<std::uintptr_t>(nbytes)) & 15u) == 0u)) {
        auto * output = reinterpret_cast<uint4 *>(destination);
        const auto * input = reinterpret_cast<const uint4 *>(source);
        const std::uint64_t count = nbytes / sizeof(uint4);
        for (std::uint64_t index = worker;
             index < count;
             index += workers) {
            output[index] = input[index];
        }
        return;
    }
    for (std::uint64_t index = worker;
         index < nbytes;
         index += workers) {
        destination[index] = source[index];
    }
}

bool current_moe_small_mmq() {
    if (g_moe_small_mmq_override >= 0) return g_moe_small_mmq_override != 0;
    static const bool enabled = [] {
        const char * value = std::getenv("MFQ_MOE_SMALL_MMQ");
        return value == nullptr || std::atoi(value) != 0;
    }();
    return enabled;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

template <typename scalar_t>
__device__ __forceinline__ float load_float(const scalar_t * ptr, int index) {
    return static_cast<float>(ptr[index]);
}

template <typename T>
__device__ __forceinline__ const T * ptr_from_i64(int64_t value) {
    return reinterpret_cast<const T *>(static_cast<uintptr_t>(value));
}

template <>
__device__ __forceinline__ float load_float<at::Half>(const at::Half * ptr, int index) {
    return __half2float(*reinterpret_cast<const __half *>(ptr + index));
}

__device__ __forceinline__ float moe_sqrt_softplus(float value) {
    const float softplus = value > 20.0f ? value : log1pf(expf(value));
    return sqrtf(softplus);
}

template <typename scalar_t>
__global__ void __launch_bounds__(128) moe_topk_kernel(
        const scalar_t * __restrict__ logits,
        const float * __restrict__ bias,
        int32_t * __restrict__ ids,
        float * __restrict__ weights,
        int rows,
        int experts,
        int top_k,
        bool use_sigmoid,
        bool use_sqrt_softplus,
        bool normalize,
        bool delayed_softmax,
        float norm_floor,
        float scale) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const scalar_t * row_logits = logits + static_cast<size_t>(row) * experts;
    float softmax_max = -INFINITY;
    if (!use_sigmoid && !use_sqrt_softplus && !delayed_softmax) {
        for (int expert = lane; expert < experts; expert += kWarpSize) {
            float value = load_float(row_logits, expert);
            value = isnan(value) ? -FLT_MAX : value;
            softmax_max = fmaxf(softmax_max, value);
        }
        softmax_max = warp_max(softmax_max);
    }

    float softmax_sum = 1.0f;
    if (!use_sigmoid && !use_sqrt_softplus && !delayed_softmax) {
        softmax_sum = 0.0f;
        for (int expert = lane; expert < experts; expert += kWarpSize) {
            float value = load_float(row_logits, expert);
            value = isnan(value) ? -FLT_MAX : value;
            softmax_sum += expf(value - softmax_max);
        }
        softmax_sum = warp_sum(softmax_sum);
    }

    int selected[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        selected[i] = -1;
    }

    for (int rank = 0; rank < top_k; ++rank) {
        float best_score = -INFINITY;
        float best_weight = -INFINITY;
        int best_expert = INT_MAX;

        for (int expert = lane; expert < experts; expert += kWarpSize) {
            bool already_selected = false;
#pragma unroll
            for (int previous = 0; previous < 16; ++previous) {
                if (previous < rank && selected[previous] == expert) {
                    already_selected = true;
                }
            }
            if (already_selected) {
                continue;
            }

            float raw = load_float(row_logits, expert);
            raw = isnan(raw) ? -FLT_MAX : raw;
            float weight;
            if (delayed_softmax) {
                weight = raw;
            } else if (use_sigmoid) {
                weight = 1.0f / (1.0f + expf(-raw));
            } else if (use_sqrt_softplus) {
                weight = moe_sqrt_softplus(raw);
            } else {
                weight = expf(raw - softmax_max) / softmax_sum;
            }
            const float score = weight + (bias == nullptr ? 0.0f : bias[expert]);
            if (score > best_score || (score == best_score && expert < best_expert)) {
                best_score = score;
                best_weight = weight;
                best_expert = expert;
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_score = __shfl_down_sync(0xffffffffu, best_score, offset);
            const float other_weight = __shfl_down_sync(0xffffffffu, best_weight, offset);
            const int other_expert = __shfl_down_sync(0xffffffffu, best_expert, offset);
            if (other_score > best_score ||
                    (other_score == best_score && other_expert < best_expert)) {
                best_score = other_score;
                best_weight = other_weight;
                best_expert = other_expert;
            }
        }
        best_weight = __shfl_sync(0xffffffffu, best_weight, 0);
        best_expert = __shfl_sync(0xffffffffu, best_expert, 0);
        selected[rank] = best_expert;
        if (lane == 0) {
            ids[static_cast<size_t>(row) * top_k + rank] = best_expert;
            weights[static_cast<size_t>(row) * top_k + rank] = best_weight;
        }
    }

    __syncwarp();
    float value = lane < top_k
        ? weights[static_cast<size_t>(row) * top_k + lane]
        : 0.0f;
    if (delayed_softmax) {
        float selected_max = lane < top_k ? value : -INFINITY;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            selected_max = fmaxf(
                selected_max,
                __shfl_down_sync(0xffffffffu, selected_max, offset));
        }
        selected_max = __shfl_sync(0xffffffffu, selected_max, 0);
        if (lane < top_k) {
            value = expf(value - selected_max);
        } else {
            value = 0.0f;
        }
        const float denom = warp_sum(value);
        if (lane < top_k) {
            value /= denom;
        }
    } else if (normalize) {
        float denom = warp_sum(value);
        denom = fmaxf(denom, norm_floor);
        if (lane < top_k) {
            value /= denom;
        }
    }
    if (lane < top_k) {
        weights[static_cast<size_t>(row) * top_k + lane] = value * scale;
    }
}

__device__ __forceinline__ bool moe_score_before(
        float lhs_score, int lhs_expert, float rhs_score, int rhs_expert) {
    return lhs_score > rhs_score ||
        (lhs_score == rhs_score && lhs_expert < rhs_expert);
}

__global__ void __launch_bounds__(32) moe_topk_1x128_8_kernel(
        const float * __restrict__ logits,
        int32_t * __restrict__ ids,
        float * __restrict__ weights) {
    const int lane = threadIdx.x;
    float values[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        float value = logits[lane + item * 32];
        values[item] = isnan(value) ? -FLT_MAX : value;
    }

    float selected = 0.0f;
#pragma unroll
    for (int rank = 0; rank < 8; ++rank) {
        float best = values[0];
        int best_expert = lane;
#pragma unroll
        for (int item = 1; item < 4; ++item) {
            const int expert = lane + item * 32;
            if (moe_score_before(values[item], expert, best, best_expert)) {
                best = values[item];
                best_expert = expert;
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask);
            const int other_expert = __shfl_xor_sync(0xffffffffu, best_expert, mask);
            if (moe_score_before(other, other_expert, best, best_expert)) {
                best = other;
                best_expert = other_expert;
            }
        }
        if (lane == (best_expert & 31)) {
            values[best_expert >> 5] = -INFINITY;
        }
        if (lane == rank) {
            selected = best;
            ids[rank] = best_expert;
        }
    }

    float selected_max = lane < 8 ? selected : -INFINITY;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        selected_max = fmaxf(
            selected_max,
            __shfl_down_sync(0xffffffffu, selected_max, offset));
    }
    selected_max = __shfl_sync(0xffffffffu, selected_max, 0);
    const float value = lane < 8 ? expf(selected - selected_max) : 0.0f;
    const float denom = warp_sum(value);
    if (lane < 8) weights[lane] = value / denom;
}

__global__ void __launch_bounds__(256) moe_topk_1x256_8_kernel(
        const float * __restrict__ logits,
        int32_t * __restrict__ ids,
        float * __restrict__ weights) {
    __shared__ float scores[256];
    __shared__ int experts[256];
    const int tid = threadIdx.x;
    float score = logits[tid];
    score = isnan(score) ? -FLT_MAX : score;
    scores[tid] = score;
    experts[tid] = tid;
    __syncthreads();

#pragma unroll
    for (int width = 2; width <= 256; width <<= 1) {
#pragma unroll
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            const int other = tid ^ stride;
            if (other > tid) {
                const float lhs_score = scores[tid];
                const int lhs_expert = experts[tid];
                const float rhs_score = scores[other];
                const int rhs_expert = experts[other];
                const bool descending = (tid & width) == 0;
                const bool lhs_before = moe_score_before(
                    lhs_score, lhs_expert, rhs_score, rhs_expert);
                const bool do_swap = descending ? !lhs_before : lhs_before;
                if (do_swap) {
                    scores[tid] = rhs_score;
                    experts[tid] = rhs_expert;
                    scores[other] = lhs_score;
                    experts[other] = lhs_expert;
                }
            }
            __syncthreads();
        }
    }

    if (tid < 32) {
        float value = tid < 8 ? scores[tid] : 0.0f;
        float selected_max = tid < 8 ? value : -INFINITY;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            selected_max = fmaxf(
                selected_max,
                __shfl_down_sync(0xffffffffu, selected_max, offset));
        }
        selected_max = __shfl_sync(0xffffffffu, selected_max, 0);
        value = tid < 8 ? expf(value - selected_max) : 0.0f;
        const float denom = warp_sum(value);
        if (tid < 8) {
            ids[tid] = experts[tid];
            weights[tid] = value / denom;
        }
    }
}

__global__ void __launch_bounds__(256) moe_topk_1x256_6_sqrtsoftplus_kernel(
        const float * __restrict__ logits,
        const float * __restrict__ bias,
        int32_t * __restrict__ ids,
        float * __restrict__ weights,
        float norm_floor,
        float scale) {
    __shared__ float scores[256];
    __shared__ float route_weights[256];
    __shared__ int experts[256];
    const int tid = threadIdx.x;
    float raw = logits[tid];
    raw = isnan(raw) ? -FLT_MAX : raw;
    const float weight = moe_sqrt_softplus(raw);
    route_weights[tid] = weight;
    scores[tid] = weight + (bias == nullptr ? 0.0f : bias[tid]);
    experts[tid] = tid;
    __syncthreads();

#pragma unroll
    for (int width = 2; width <= 256; width <<= 1) {
#pragma unroll
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            const int other = tid ^ stride;
            if (other > tid) {
                const float lhs_score = scores[tid];
                const int lhs_expert = experts[tid];
                const float rhs_score = scores[other];
                const int rhs_expert = experts[other];
                const bool descending = (tid & width) == 0;
                const bool lhs_before = moe_score_before(
                    lhs_score, lhs_expert, rhs_score, rhs_expert);
                const bool do_swap = descending ? !lhs_before : lhs_before;
                if (do_swap) {
                    scores[tid] = rhs_score;
                    experts[tid] = rhs_expert;
                    scores[other] = lhs_score;
                    experts[other] = lhs_expert;
                }
            }
            __syncthreads();
        }
    }

    if (tid < 32) {
        const int expert = tid < 6 ? experts[tid] : 0;
        float value = tid < 6 ? route_weights[expert] : 0.0f;
        float denom = warp_sum(value);
        denom = fmaxf(denom, norm_floor);
        if (tid < 6) {
            ids[tid] = expert;
            weights[tid] = value / denom * scale;
        }
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(128) moe_sqrtsoftplus_weights_kernel(
        const scalar_t * __restrict__ logits,
        const int32_t * __restrict__ ids,
        float * __restrict__ weights,
        int rows,
        int experts,
        int top_k,
        float norm_floor,
        float scale) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }

    float value = 0.0f;
    if (lane < top_k) {
        const int expert = ids[static_cast<size_t>(row) * top_k + lane];
        if (static_cast<unsigned int>(expert) < static_cast<unsigned int>(experts)) {
            float raw = load_float(logits + static_cast<size_t>(row) * experts, expert);
            raw = isnan(raw) ? -FLT_MAX : raw;
            value = moe_sqrt_softplus(raw);
        }
    }
    float denom = warp_sum(value);
    denom = fmaxf(denom, norm_floor);
    if (lane < top_k) {
        weights[static_cast<size_t>(row) * top_k + lane] = value / denom * scale;
    }
}

__global__ void count_experts_kernel(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ counts,
        int pairs,
        int experts) {
    for (int pair = blockIdx.x * blockDim.x + threadIdx.x;
            pair < pairs;
            pair += blockDim.x * gridDim.x) {
        const int expert = ids[pair];
        if (static_cast<unsigned int>(expert) < static_cast<unsigned int>(experts)) {
            atomicAdd(counts + expert, 1);
        }
    }
}

__global__ void scan_expert_counts_kernel(
        const int32_t * __restrict__ counts,
        int32_t * __restrict__ cursors,
        int32_t * __restrict__ expert_bounds,
        int32_t * __restrict__ tile_bounds,
        int experts,
        int tile_m) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int pair_offset = 0;
    int tile_offset = 0;
    expert_bounds[0] = 0;
    tile_bounds[0] = 0;
    for (int expert = 0; expert < experts; ++expert) {
        const int count = counts[expert];
        const int tiles = (count + tile_m - 1) / tile_m;
        cursors[expert] = 0;
        pair_offset += count;
        tile_offset += tiles;
        expert_bounds[expert + 1] = pair_offset;
        tile_bounds[expert + 1] = tile_offset;
    }
}

__global__ void fill_tile_experts_kernel(
        const int32_t * __restrict__ tile_bounds,
        int32_t * __restrict__ tile_experts,
        int experts) {
    for (int expert = blockIdx.x * blockDim.x + threadIdx.x;
            expert < experts;
            expert += blockDim.x * gridDim.x) {
        for (int tile = tile_bounds[expert]; tile < tile_bounds[expert + 1]; ++tile) {
            tile_experts[tile] = expert;
        }
    }
}

__global__ void scatter_routes_kernel(
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ expert_bounds,
        int32_t * __restrict__ cursors,
        int32_t * __restrict__ ids_dst,
        int pairs,
        int experts) {
    for (int pair = blockIdx.x * blockDim.x + threadIdx.x;
            pair < pairs;
            pair += blockDim.x * gridDim.x) {
        const int expert = ids[pair];
        if (static_cast<unsigned int>(expert) >= static_cast<unsigned int>(experts)) {
            continue;
        }
        const int compact = expert_bounds[expert] + atomicAdd(cursors + expert, 1);
        ids_dst[compact] = pair;
    }
}

template <int GS, int BD>
__global__ void quantize_moe_input_kernel(
        const __half * __restrict__ x,
        int8_t * __restrict__ qx,
        float * __restrict__ xscale,
        int rows,
        int k_real,
        int k_pad) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int tid = threadIdx.x;
    const int base = group * GS;
    const bool real = tid < GS && base + tid < k_real;
    const float value = real
        ? __half2float(x[static_cast<size_t>(row) * k_real + base + tid])
        : 0.0f;

    float max_value = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_value = fmaxf(max_value, __shfl_down_sync(0xffffffffu, max_value, offset));
    }
    __shared__ float warp_maxima[2];
    if ((tid & 31) == 0) {
        warp_maxima[tid >> 5] = max_value;
    }
    __syncthreads();
    if (tid < 32) {
        max_value = tid < (BD / 32) ? warp_maxima[tid] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_value = fmaxf(max_value, __shfl_down_sync(0xffffffffu, max_value, offset));
        }
    }
    __shared__ float group_scale;
    if (tid == 0) {
        group_scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
        xscale[static_cast<size_t>(row) * gridDim.y + group] = group_scale;
    }
    __syncthreads();
    if (tid < GS) {
        int quant = 0;
        if (real) {
            quant = static_cast<int>(roundf(value / group_scale));
            quant = max(-127, min(127, quant));
        }
        qx[static_cast<size_t>(row) * k_pad + base + tid] = static_cast<int8_t>(quant);
    }
}

__global__ void quantize_moe_input_24_28_kernel(
        const __half * __restrict__ x,
        int8_t * __restrict__ qx24,
        float * __restrict__ xscale24,
        int8_t * __restrict__ qx28,
        float * __restrict__ xscale28,
        int k_real,
        int groups24,
        int groups28) {
    const int row = blockIdx.x;
    const int combined_group = blockIdx.y;
    const int lane = threadIdx.x;
    const bool use24 = combined_group < groups24;
    const int group = use24 ? combined_group : combined_group - groups24;
    const int gs = use24 ? 24 : 28;
    const int groups = use24 ? groups24 : groups28;
    const int base = group * gs;
    const bool real = lane < gs && base + lane < k_real;
    const float value = real
        ? __half2float(x[static_cast<size_t>(row) * k_real + base + lane])
        : 0.0f;

    float max_value = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_value = fmaxf(max_value, __shfl_down_sync(0xffffffffu, max_value, offset));
    }
    float group_scale = 1.0f;
    if (lane == 0) {
        group_scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
        float * scale_out = use24 ? xscale24 : xscale28;
        scale_out[static_cast<size_t>(row) * groups + group] = group_scale;
    }
    group_scale = __shfl_sync(0xffffffffu, group_scale, 0);
    if (lane < gs) {
        int quant = 0;
        if (real) {
            quant = static_cast<int>(roundf(value / group_scale));
            quant = max(-127, min(127, quant));
        }
        int8_t * q_out = use24 ? qx24 : qx28;
        q_out[static_cast<size_t>(row) * groups * gs + base + lane] =
            static_cast<int8_t>(quant);
    }
}

template <int GS, int BD, bool GELU, bool CLAMPED = false>
__global__ void quantize_moe_glu_input_kernel(
        const __half * __restrict__ gate_up,
        int8_t * __restrict__ qx,
        float * __restrict__ xscale,
        int rows,
        int k_real,
        int k_pad,
        float limit) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int tid = threadIdx.x;
    const int base = group * GS;
    const bool real = tid < GS && base + tid < k_real;
    float value = 0.0f;
    if (real) {
        const size_t row_base = static_cast<size_t>(row) * (2 * k_real);
        float gate = __half2float(gate_up[row_base + base + tid]);
        float up = __half2float(gate_up[row_base + k_real + base + tid]);
        if constexpr (CLAMPED) {
            gate = fminf(gate, limit);
            up = fminf(fmaxf(up, -limit), limit);
        }
        const __half rounded = __float2half_rn(mfq_glu<GELU>(gate, up));
        value = __half2float(rounded);
    }

    float max_value = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_value = fmaxf(max_value, __shfl_down_sync(0xffffffffu, max_value, offset));
    }
    __shared__ float warp_maxima[2];
    if ((tid & 31) == 0) {
        warp_maxima[tid >> 5] = max_value;
    }
    __syncthreads();
    if (tid < 32) {
        max_value = tid < (BD / 32) ? warp_maxima[tid] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_value = fmaxf(max_value, __shfl_down_sync(0xffffffffu, max_value, offset));
        }
    }
    __shared__ float group_scale;
    if (tid == 0) {
        group_scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
        xscale[static_cast<size_t>(row) * gridDim.y + group] = group_scale;
    }
    __syncthreads();
    if (tid < GS) {
        int quant = 0;
        if (real) {
            quant = static_cast<int>(roundf(value / group_scale));
            quant = max(-127, min(127, quant));
        }
        qx[static_cast<size_t>(row) * k_pad + base + tid] = static_cast<int8_t>(quant);
    }
}

template <int GS>
__device__ __forceinline__ void quantize_moe_shared_group(
        const __half * __restrict__ hidden,
        int8_t * __restrict__ qx,
        float * __restrict__ xscale,
        int row,
        int group,
        int groups,
        int k_real,
        int k_pad,
        int lane) {
    const int k = group * GS + lane;
    const bool real = lane < GS && k < k_real;
    const float value = real ? __half2float(hidden[k]) : 0.0f;
    float max_value = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_value = fmaxf(max_value, __shfl_down_sync(0xffffffffu, max_value, offset));
    }
    float group_scale = 1.0f;
    if (lane == 0) {
        group_scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
        xscale[static_cast<size_t>(row) * groups + group] = group_scale;
    }
    group_scale = __shfl_sync(0xffffffffu, group_scale, 0);
    if (lane < GS) {
        int quant = 0;
        if (real) {
            quant = static_cast<int>(roundf(value / group_scale));
            quant = max(-127, min(127, quant));
        }
        qx[static_cast<size_t>(row) * k_pad + group * GS + lane] =
            static_cast<int8_t>(quant);
    }
}

template <bool GELU>
__global__ void quantize_moe_glu_24_28_kernel(
        const __half * __restrict__ gate_up,
        int8_t * __restrict__ qx24,
        float * __restrict__ xscale24,
        int8_t * __restrict__ qx28,
        float * __restrict__ xscale28,
        int k_real,
        int groups24,
        int groups28) {
    extern __shared__ __half hidden[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t row_base = static_cast<size_t>(row) * (2 * k_real);
    for (int k = tid; k < k_real; k += blockDim.x) {
        const float gate = __half2float(gate_up[row_base + k]);
        const float up = __half2float(gate_up[row_base + k_real + k]);
        hidden[k] = __float2half_rn(mfq_glu<GELU>(gate, up));
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int warps = 8;
    for (int group = warp; group < groups24; group += warps) {
        quantize_moe_shared_group<24>(
            hidden, qx24, xscale24, row, group, groups24, k_real, groups24 * 24, lane);
    }
    for (int group = warp; group < groups28; group += warps) {
        quantize_moe_shared_group<28>(
            hidden, qx28, xscale28, row, group, groups28, k_real, groups28 * 28, lane);
    }
}

template <int BITS>
__device__ __forceinline__ uint8_t unpack_one(const uint8_t * values, int index) {
    if constexpr (BITS == 8) {
        return values[index];
    } else {
        constexpr uint32_t mask = (1u << BITS) - 1u;
        const int bit = index * BITS;
        const int byte = bit >> 3;
        const int shift = bit & 7;
        uint32_t word = values[byte];
        if (shift + BITS > 8) {
            word |= static_cast<uint32_t>(values[byte + 1]) << 8;
        }
        return static_cast<uint8_t>((word >> shift) & mask);
    }
}

template <int BITS>
__device__ __forceinline__ int unpack_four(const uint8_t * values, int index) {
    if constexpr (BITS == 4) {
        const uint16_t packed = *reinterpret_cast<const uint16_t *>(values + (index >> 1));
        const uint8_t first = static_cast<uint8_t>(packed);
        const uint8_t second = static_cast<uint8_t>(packed >> 8);
        return static_cast<int>(first & 15u) |
            (static_cast<int>(first >> 4) << 8) |
            (static_cast<int>(second & 15u) << 16) |
            (static_cast<int>(second >> 4) << 24);
    } else if constexpr (BITS == 5) {
        const int bit = index * 5;
        const int byte = bit >> 3;
        const int shift = bit & 7;
        const uint32_t word = static_cast<uint32_t>(values[byte + 0]) |
            (static_cast<uint32_t>(values[byte + 1]) << 8) |
            (static_cast<uint32_t>(values[byte + 2]) << 16);
        const uint32_t unpacked = word >> shift;
        return static_cast<int>(unpacked & 31u) |
            (static_cast<int>((unpacked >> 5) & 31u) << 8) |
            (static_cast<int>((unpacked >> 10) & 31u) << 16) |
            (static_cast<int>((unpacked >> 15) & 31u) << 24);
    } else if constexpr (BITS == 6) {
        const int byte = (index >> 2) * 3;
        const uint8_t b0 = values[byte + 0];
        const uint8_t b1 = values[byte + 1];
        const uint8_t b2 = values[byte + 2];
        const uint8_t q0 = b0 & 63u;
        const uint8_t q1 = (b0 >> 6 | b1 << 2) & 63u;
        const uint8_t q2 = (b1 >> 4 | b2 << 4) & 63u;
        const uint8_t q3 = (b2 >> 2) & 63u;
        return static_cast<int>(q0) |
            (static_cast<int>(q1) << 8) |
            (static_cast<int>(q2) << 16) |
            (static_cast<int>(q3) << 24);
    } else if constexpr (BITS == 8) {
        return *reinterpret_cast<const int *>(values + index);
    } else {
        const uint8_t q0 = unpack_one<BITS>(values, index + 0);
        const uint8_t q1 = unpack_one<BITS>(values, index + 1);
        const uint8_t q2 = unpack_one<BITS>(values, index + 2);
        const uint8_t q3 = unpack_one<BITS>(values, index + 3);
        return static_cast<int>(q0) |
            (static_cast<int>(q1) << 8) |
            (static_cast<int>(q2) << 16) |
            (static_cast<int>(q3) << 24);
    }
}

__device__ __forceinline__ int load_i8x4(const int8_t * values) {
    return *reinterpret_cast<const int *>(values);
}

template <int BITS>
__device__ __forceinline__ int quant_dot(int packed_weight, int packed_x, int x_sum) {
    if constexpr (BITS == 8) {
        return __dp4a(packed_weight ^ static_cast<int>(0x80808080u), packed_x, 0) +
            128 * x_sum;
    } else {
        return __dp4a(packed_weight, packed_x, 0);
    }
}

template <int BITS, int GS, int ITEMS_PER_BLOCK, bool ROUTE_PACKED = false>
__global__ void nint_moe_mmvq_kernel(
        const uint8_t * __restrict__ q_packed,
        const uint8_t * __restrict__ sub_scale,
        const uint8_t * __restrict__ sub_min,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_min,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ expert_local,
        __half * __restrict__ out,
        int tokens,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_pad,
        bool routed_input) {
    constexpr int qbytes = (GS * BITS + 7) / 8;
    constexpr int rows_per_warp = 2;
    constexpr int chunks = (GS + 3) / 4;
    constexpr int groups_per_warp = kWarpSize / chunks;
    const int token = ROUTE_PACKED ? static_cast<int>(blockIdx.z) :
        static_cast<int>(blockIdx.z) * ITEMS_PER_BLOCK + static_cast<int>(threadIdx.y);
    const int route = ROUTE_PACKED ? static_cast<int>(threadIdx.y) :
        static_cast<int>(blockIdx.y);
    const int lane = threadIdx.x;
    const int row0 = blockIdx.x * rows_per_warp;
    if (token >= tokens || route >= routes) {
        return;
    }
    const int pair = token * routes + route;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >= static_cast<unsigned int>(experts)) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (local_expert < 0) {
        return;
    }
    const int source_row = routed_input ? pair : token;

    float acc[rows_per_warp] = {0.0f, 0.0f};
    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int group_offset = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    float neuron_d[rows_per_warp] = {};
    float neuron_m[rows_per_warp] = {};
#pragma unroll
    for (int r = 0; r < rows_per_warp; ++r) {
        const int local_row = row0 + r;
        if (local_row < out_per_expert) {
            const int weight_row = local_expert * out_per_expert + local_row;
            neuron_d[r] = neuron_scale[weight_row];
            neuron_m[r] = neuron_min[weight_row];
        }
    }

    for (int group_base = 0; group_base < groups; group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups || group_offset >= GS) {
            continue;
        }
        const int width = min(4, GS - group_offset);
        const int k = group * GS + group_offset;
        const int8_t * x_ptr = qx + static_cast<size_t>(source_row) * k_pad + k;
        int packed_x = 0;
        int x_sum = 0;
        if (width == 4) {
            packed_x = load_i8x4(x_ptr);
            x_sum = __dp4a(0x01010101, packed_x, 0);
        } else {
            for (int i = 0; i < width; ++i) {
                const int value = static_cast<int>(x_ptr[i]);
                packed_x |= (value & 255) << (8 * i);
                x_sum += value;
            }
        }
        const float activation_scale = xscale[static_cast<size_t>(source_row) * groups + group];

#pragma unroll
        for (int r = 0; r < rows_per_warp; ++r) {
            const int local_row = row0 + r;
            if (local_row >= out_per_expert) {
                continue;
            }
            const int weight_row = local_expert * out_per_expert + local_row;
            const size_t meta = static_cast<size_t>(weight_row) * groups + group;
            const uint8_t * q_group = q_packed + meta * qbytes;
            int packed_weight = 0;
            if (width == 4) {
                packed_weight = unpack_four<BITS>(q_group, group_offset);
            } else {
                for (int i = 0; i < width; ++i) {
                    packed_weight |= static_cast<int>(unpack_one<BITS>(q_group, group_offset + i)) << (8 * i);
                }
            }
            const int dot = quant_dot<BITS>(packed_weight, packed_x, x_sum);
            const float weight_scale = neuron_d[r] * static_cast<float>(sub_scale[meta]);
            const float weight_min = neuron_m[r] * static_cast<float>(sub_min[meta]);
            acc[r] += activation_scale *
                (weight_scale * static_cast<float>(dot) - weight_min * static_cast<float>(x_sum));
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_warp; ++r) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < rows_per_warp; ++r) {
            const int local_row = row0 + r;
            if (local_row < out_per_expert) {
                out[static_cast<size_t>(pair) * out_per_expert + local_row] = __float2half(acc[r]);
            }
        }
    }
}

template <int ITEMS_PER_BLOCK>
__global__ void nint8_zero_moe_mmvq_kernel(
        const uint8_t * __restrict__ q,
        const __half * __restrict__ scale,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ expert_local,
        __half * __restrict__ out,
        int tokens,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_pad,
        bool routed_input) {
    constexpr int gs = 32;
    constexpr int rows_per_warp = 2;
    constexpr int chunks = gs / 4;
    constexpr int groups_per_warp = kWarpSize / chunks;
    const int token =
        static_cast<int>(blockIdx.z) * ITEMS_PER_BLOCK +
        static_cast<int>(threadIdx.y);
    const int route = static_cast<int>(blockIdx.y);
    const int lane = threadIdx.x;
    const int row0 = static_cast<int>(blockIdx.x) * rows_per_warp;
    if (token >= tokens || route >= routes) {
        return;
    }
    const int pair = token * routes + route;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >=
        static_cast<unsigned int>(experts)) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (local_expert < 0) {
        return;
    }
    const int source_row = routed_input ? pair : token;

    float acc[rows_per_warp] = {0.0f, 0.0f};
    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int group_offset = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;

    for (int group_base = 0; group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups) {
            continue;
        }
        const int k = group * gs + group_offset;
        const int packed_x = load_i8x4(
            qx + static_cast<size_t>(source_row) * k_pad + k);
        const float activation_scale =
            xscale[static_cast<size_t>(source_row) * groups + group];

#pragma unroll
        for (int r = 0; r < rows_per_warp; ++r) {
            const int local_row = row0 + r;
            if (local_row >= out_per_expert) {
                continue;
            }
            const int weight_row =
                local_expert * out_per_expert + local_row;
            const size_t meta =
                static_cast<size_t>(weight_row) * groups + group;
            const int packed_weight = load_i8x4(
                reinterpret_cast<const int8_t *>(
                    q + meta * gs + group_offset));
            const int dot = __dp4a(packed_weight, packed_x, 0);
            acc[r] += activation_scale *
                __half2float(scale[meta]) * static_cast<float>(dot);
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_warp; ++r) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < rows_per_warp; ++r) {
            const int local_row = row0 + r;
            if (local_row < out_per_expert) {
                out[static_cast<size_t>(pair) * out_per_expert + local_row] =
                    __float2half(acc[r]);
            }
        }
    }
}

template <int BITS, int GS, int NWARPS>
__global__ void __launch_bounds__(NWARPS * kWarpSize) nint_moe_mmvq_ksplit_kernel(
        const uint8_t * __restrict__ q_packed,
        const uint8_t * __restrict__ sub_scale,
        const uint8_t * __restrict__ sub_min,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_min,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ expert_local,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_pad,
        bool routed_input) {
    constexpr int qbytes = (GS * BITS + 7) / 8;
    constexpr int rows_per_block = 2;
    constexpr int chunks = (GS + 3) / 4;
    constexpr int groups_per_block = (NWARPS * kWarpSize) / chunks;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * kWarpSize + lane;
    const int route = blockIdx.y;
    const int row0 = blockIdx.x * rows_per_block;
    const int expert = ids[route];
    if (static_cast<unsigned int>(expert) >= static_cast<unsigned int>(experts)) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (local_expert < 0) {
        return;
    }
    const int source_row = routed_input ? route : 0;

    float acc[rows_per_block] = {0.0f, 0.0f};
    const int relative_group = tid / chunks;
    const int chunk = tid - relative_group * chunks;
    const int group_offset = chunk * 4;
    const bool active_thread = relative_group < groups_per_block;

    for (int group_base = 0; group_base < groups; group_base += groups_per_block) {
        const int group = group_base + relative_group;
        if (!active_thread || group >= groups) {
            continue;
        }
        const int width = min(4, GS - group_offset);
        const int k = group * GS + group_offset;
        const int8_t * x_ptr = qx + static_cast<size_t>(source_row) * k_pad + k;
        int packed_x = 0;
        int x_sum = 0;
        if (width == 4) {
            packed_x = load_i8x4(x_ptr);
            x_sum = __dp4a(0x01010101, packed_x, 0);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i < width) {
                    const int value = static_cast<int>(x_ptr[i]);
                    packed_x |= (value & 255) << (8 * i);
                    x_sum += value;
                }
            }
        }
        const float activation_scale = xscale[static_cast<size_t>(source_row) * groups + group];

#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            const int local_row = row0 + r;
            if (local_row >= out_per_expert) {
                continue;
            }
            const int weight_row = local_expert * out_per_expert + local_row;
            const size_t meta = static_cast<size_t>(weight_row) * groups + group;
            const uint8_t * q_group = q_packed + meta * qbytes;
            int packed_weight = 0;
            if (width == 4) {
                packed_weight = unpack_four<BITS>(q_group, group_offset);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (i < width) {
                        packed_weight |= static_cast<int>(unpack_one<BITS>(q_group, group_offset + i)) << (8 * i);
                    }
                }
            }
            const int dot = quant_dot<BITS>(packed_weight, packed_x, x_sum);
            const float weight_scale = neuron_scale[weight_row] * static_cast<float>(sub_scale[meta]);
            const float weight_min = neuron_min[weight_row] * static_cast<float>(sub_min[meta]);
            acc[r] += activation_scale *
                (weight_scale * static_cast<float>(dot) - weight_min * static_cast<float>(x_sum));
        }
    }

    __shared__ float partial[NWARPS - 1][rows_per_block][kWarpSize];
    if (warp > 0) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            partial[warp - 1][r][lane] = acc[r];
        }
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }

#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
        for (int w = 0; w < NWARPS - 1; ++w) {
            acc[r] += partial[w][r][lane];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            const int local_row = row0 + r;
            if (local_row < out_per_expert) {
                out[static_cast<size_t>(route) * out_per_expert + local_row] = __float2half(acc[r]);
            }
        }
    }
}

__device__ __forceinline__ int find_expert_tile(
        const int32_t * tile_bounds,
        int experts,
        int tile) {
    int low = 0;
    int high = experts;
    while (low < high) {
        const int middle = (low + high) >> 1;
        if (tile_bounds[middle + 1] <= tile) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

template <int BITS, int GS, int TILE_M>
__global__ void __launch_bounds__(128) nint_moe_grouped_tile_kernel(
        const uint8_t * __restrict__ q_packed,
        const uint8_t * __restrict__ sub_scale,
        const uint8_t * __restrict__ sub_min,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_min,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ expert_local,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_pad,
        int max_tiles,
        bool routed_input) {
    constexpr int qbytes = (GS * BITS + 7) / 8;
    constexpr int chunks = (GS + 3) / 4;
    constexpr int groups_per_warp = kWarpSize / chunks;
    const int row_tiles = (out_per_expert + kRowsPerBlock - 1) / kRowsPerBlock;
    const int linear_block = blockIdx.x;
    const int row_tile = linear_block % row_tiles;
    const int tile = linear_block / row_tiles;
    if (tile >= max_tiles || tile >= tile_bounds[experts]) {
        return;
    }
    const int expert = find_expert_tile(tile_bounds, experts, tile);
    if (expert >= experts) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (local_expert < 0) {
        return;
    }
    const int local_tile = tile - tile_bounds[expert];
    const int first = expert_bounds[expert] + local_tile * TILE_M;
    const int last = min(first + TILE_M, expert_bounds[expert + 1]);
    const int row = row_tile * kRowsPerBlock + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= out_per_expert) {
        return;
    }

    float acc[TILE_M];
#pragma unroll
    for (int item = 0; item < TILE_M; ++item) {
        acc[item] = 0.0f;
    }
    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int group_offset = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    const int weight_row = local_expert * out_per_expert + row;
    const float neuron_d = neuron_scale[weight_row];
    const float neuron_m = neuron_min[weight_row];

    for (int group_base = 0; group_base < groups; group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups || group_offset >= GS) {
            continue;
        }
        const int width = min(4, GS - group_offset);
        const size_t meta = static_cast<size_t>(weight_row) * groups + group;
        const uint8_t * q_group = q_packed + meta * qbytes;
        int packed_weight = 0;
        if (width == 4) {
            packed_weight = unpack_four<BITS>(q_group, group_offset);
        } else {
            for (int i = 0; i < width; ++i) {
                packed_weight |= static_cast<int>(unpack_one<BITS>(q_group, group_offset + i)) << (8 * i);
            }
        }
        const float weight_scale = neuron_d * static_cast<float>(sub_scale[meta]);
        const float weight_min = neuron_m * static_cast<float>(sub_min[meta]);
        const int k = group * GS + group_offset;

#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
            const int compact = first + item;
            if (compact >= last) {
                continue;
            }
            const int pair = ids_dst[compact];
            const int source_row = routed_input ? pair : pair / routes;
            const int8_t * x_ptr = qx + static_cast<size_t>(source_row) * k_pad + k;
            int packed_x = 0;
            int x_sum = 0;
            if (width == 4) {
                packed_x = load_i8x4(x_ptr);
                x_sum = __dp4a(0x01010101, packed_x, 0);
            } else {
                for (int i = 0; i < width; ++i) {
                    const int value = static_cast<int>(x_ptr[i]);
                    packed_x |= (value & 255) << (8 * i);
                    x_sum += value;
                }
            }
            const int dot = quant_dot<BITS>(packed_weight, packed_x, x_sum);
            const float activation_scale = xscale[static_cast<size_t>(source_row) * groups + group];
            acc[item] += activation_scale *
                (weight_scale * static_cast<float>(dot) - weight_min * static_cast<float>(x_sum));
        }
    }

#pragma unroll
    for (int item = 0; item < TILE_M; ++item) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[item] += __shfl_xor_sync(0xffffffffu, acc[item], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
            const int compact = first + item;
            if (compact < last) {
                const int pair = ids_dst[compact];
                out[static_cast<size_t>(pair) * out_per_expert + row] = __float2half(acc[item]);
            }
        }
    }
}

template <int BITS, int GS, int ROWS_PER_WARP = 2>
__device__ __forceinline__ void nint_moe_mmvq_profile(
        const uint8_t * __restrict__ q_packed,
        const uint8_t * __restrict__ sub_scale,
        const uint8_t * __restrict__ sub_min,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_min,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        __half * __restrict__ out,
        int lane,
        int row0,
        int local_expert,
        int source_row,
        int pair,
        int out_per_expert,
        int groups,
        int k_pad) {
    constexpr int qbytes = (GS * BITS + 7) / 8;
    constexpr int chunks = (GS + 3) / 4;
    constexpr int groups_per_warp = kWarpSize / chunks;
    static_assert(GS % 4 == 0, "MoE MMVQ profiles require 4|GS");
    float acc[ROWS_PER_WARP] = {};
    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int group_offset = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    const int group_leader = relative_group * chunks;
    float neuron_d[ROWS_PER_WARP] = {};
    float neuron_m[ROWS_PER_WARP] = {};
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int local_row = row0 + r;
        if (local_row < out_per_expert) {
            const int weight_row = local_expert * out_per_expert + local_row;
            neuron_d[r] = neuron_scale[weight_row];
            neuron_m[r] = neuron_min[weight_row];
        }
    }

    for (int group_base = 0; group_base < groups; group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        const bool valid_group = active_lane && group < groups;
        const int k = group * GS + group_offset;
        const int8_t * x_ptr = qx + static_cast<size_t>(source_row) * k_pad + k;
        const int packed_x = valid_group ? load_i8x4(x_ptr) : 0;
        const int x_sum = valid_group ? __dp4a(0x01010101, packed_x, 0) : 0;
        float activation_scale = valid_group && chunk == 0
            ? xscale[static_cast<size_t>(source_row) * groups + group]
            : 0.0f;
        activation_scale = __shfl_sync(
            0xffffffffu, activation_scale, group_leader);

#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int local_row = row0 + r;
            const bool valid_row = valid_group && local_row < out_per_expert;
            int dot = 0;
            int ss = 0;
            int sm = 0;
            if (valid_row) {
                const int weight_row = local_expert * out_per_expert + local_row;
                const size_t meta = static_cast<size_t>(weight_row) * groups + group;
                const uint8_t * q_group = q_packed + meta * qbytes;
                const int packed_weight = unpack_four<BITS>(q_group, group_offset);
                dot = quant_dot<BITS>(packed_weight, packed_x, x_sum);
                if (chunk == 0) {
                    ss = static_cast<int>(sub_scale[meta]);
                    sm = static_cast<int>(sub_min[meta]);
                }
            }
            ss = __shfl_sync(0xffffffffu, ss, group_leader);
            sm = __shfl_sync(0xffffffffu, sm, group_leader);
            if (valid_row) {
                const float weight_scale = neuron_d[r] * static_cast<float>(ss);
                const float weight_min = neuron_m[r] * static_cast<float>(sm);
                acc[r] += activation_scale *
                    (weight_scale * static_cast<float>(dot) -
                     weight_min * static_cast<float>(x_sum));
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int local_row = row0 + r;
            if (local_row < out_per_expert) {
                out[static_cast<size_t>(pair) * out_per_expert + local_row] = __float2half(acc[r]);
            }
        }
    }
}

template <int BITS, int GS, int NWARPS>
__device__ __forceinline__ void nint_moe_mmvq_group_profile(
        const uint8_t * __restrict__ q_packed,
        const uint8_t * __restrict__ sub_scale,
        const uint8_t * __restrict__ sub_min,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_min,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        __half * __restrict__ out,
        float * __restrict__ partial,
        int warp,
        int lane,
        int row,
        int local_expert,
        int source_row,
        int pair,
        int out_per_expert,
        int groups) {
    static_assert(GS % 4 == 0, "MoE group MMVQ profiles require 4|GS");
    constexpr int qbytes = (GS * BITS + 7) / 8;
    constexpr int chunks = GS / 4;
    const int tid = warp * kWarpSize + lane;
    const int weight_row = local_expert * out_per_expert + row;
    const float neuron_d = neuron_scale[weight_row];
    const float neuron_m = neuron_min[weight_row];
    const int k_pad = groups * GS;
    const int8_t * xrow = qx + static_cast<size_t>(source_row) * k_pad;
    const float * xsrow = xscale + static_cast<size_t>(source_row) * groups;
    float acc = 0.0f;

    for (int group = tid; group < groups; group += NWARPS * kWarpSize) {
        const size_t meta = static_cast<size_t>(weight_row) * groups + group;
        const uint8_t * q_group = q_packed + meta * qbytes;
        const float activation_scale = xsrow[group];
        const float weight_scale = neuron_d * static_cast<float>(sub_scale[meta]);
        const float weight_min = neuron_m * static_cast<float>(sub_min[meta]);
#pragma unroll
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int offset = chunk * 4;
            const int packed_x = load_i8x4(xrow + group * GS + offset);
            const int x_sum = __dp4a(0x01010101, packed_x, 0);
            const int packed_weight = unpack_four<BITS>(q_group, offset);
            const int dot = quant_dot<BITS>(packed_weight, packed_x, x_sum);
            acc += activation_scale *
                (weight_scale * static_cast<float>(dot) -
                 weight_min * static_cast<float>(x_sum));
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, offset);
        }
        if (lane == 0) {
            out[static_cast<size_t>(pair) * out_per_expert + row] = __float2half(acc);
        }
    }
}

template <int BITS, int GS>
__device__ __forceinline__ void nint_moe_mmvq_groupwarp_profile(
        const uint8_t * __restrict__ q_packed,
        const uint8_t * __restrict__ sub_scale,
        const uint8_t * __restrict__ sub_min,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_min,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        __half * __restrict__ out,
        int lane,
        int row,
        int local_expert,
        int source_row,
        int pair,
        int out_per_expert,
        int groups) {
    static_assert(GS % 4 == 0, "MoE group-warp profiles require 4|GS");
    constexpr int qbytes = (GS * BITS + 7) / 8;
    constexpr int chunks = GS / 4;
    const int weight_row = local_expert * out_per_expert + row;
    const float neuron_d = neuron_scale[weight_row];
    const float neuron_m = neuron_min[weight_row];
    const int k_pad = groups * GS;
    const int8_t * xrow = qx + static_cast<size_t>(source_row) * k_pad;
    const float * xsrow = xscale + static_cast<size_t>(source_row) * groups;
    float acc = 0.0f;

    for (int group = lane; group < groups; group += kWarpSize) {
        const size_t meta = static_cast<size_t>(weight_row) * groups + group;
        const uint8_t * q_group = q_packed + meta * qbytes;
        const float activation_scale = xsrow[group];
        const float weight_scale = neuron_d * static_cast<float>(sub_scale[meta]);
        const float weight_min = neuron_m * static_cast<float>(sub_min[meta]);
#pragma unroll
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int offset = chunk * 4;
            const int packed_x = load_i8x4(xrow + group * GS + offset);
            const int x_sum = __dp4a(0x01010101, packed_x, 0);
            const int packed_weight = unpack_four<BITS>(q_group, offset);
            const int dot = quant_dot<BITS>(packed_weight, packed_x, x_sum);
            acc += activation_scale *
                (weight_scale * static_cast<float>(dot) -
                 weight_min * static_cast<float>(x_sum));
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        out[static_cast<size_t>(pair) * out_per_expert + row] = __float2half(acc);
    }
}

template <int BITS, int GS, int NWARPS, bool GELU>
__device__ __forceinline__ void nint_moe_mmvq_glu_group_profile(
        const uint8_t * __restrict__ q_packed,
        const uint8_t * __restrict__ sub_scale,
        const uint8_t * __restrict__ sub_min,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_min,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        __half * __restrict__ out,
        float * __restrict__ partial_gate,
        float * __restrict__ partial_up,
        int warp,
        int lane,
        int hidden_row,
        int local_expert,
        int source_row,
        int output_row,
        int hidden_width,
        int groups) {
    static_assert(GS % 4 == 0, "MoE group GLU profiles require 4|GS");
    constexpr int qbytes = (GS * BITS + 7) / 8;
    constexpr int chunks = GS / 4;
    const int tid = warp * kWarpSize + lane;
    const int out_per_expert = 2 * hidden_width;
    const int gate_weight_row = local_expert * out_per_expert + hidden_row;
    const int up_weight_row = gate_weight_row + hidden_width;
    const float gate_neuron_d = neuron_scale[gate_weight_row];
    const float gate_neuron_m = neuron_min[gate_weight_row];
    const float up_neuron_d = neuron_scale[up_weight_row];
    const float up_neuron_m = neuron_min[up_weight_row];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int group = tid; group < groups; group += NWARPS * kWarpSize) {
        const size_t gate_meta = static_cast<size_t>(gate_weight_row) * groups + group;
        const size_t up_meta = static_cast<size_t>(up_weight_row) * groups + group;
        const uint8_t * gate_group = q_packed + gate_meta * qbytes;
        const uint8_t * up_group = q_packed + up_meta * qbytes;
        const float activation_scale = xscale[static_cast<size_t>(source_row) * groups + group];
        const float gate_scale = gate_neuron_d * static_cast<float>(sub_scale[gate_meta]);
        const float gate_min = gate_neuron_m * static_cast<float>(sub_min[gate_meta]);
        const float up_scale = up_neuron_d * static_cast<float>(sub_scale[up_meta]);
        const float up_min = up_neuron_m * static_cast<float>(sub_min[up_meta]);
#pragma unroll
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int offset = chunk * 4;
            const int packed_x = load_i8x4(
                qx + static_cast<size_t>(source_row) * groups * GS + group * GS + offset);
            const int x_sum = __dp4a(0x01010101, packed_x, 0);
            const int gate_weight = unpack_four<BITS>(gate_group, offset);
            const int up_weight = unpack_four<BITS>(up_group, offset);
            const int gate_dot = quant_dot<BITS>(gate_weight, packed_x, x_sum);
            const int up_dot = quant_dot<BITS>(up_weight, packed_x, x_sum);
            gate_acc += activation_scale *
                (gate_scale * static_cast<float>(gate_dot) -
                 gate_min * static_cast<float>(x_sum));
            up_acc += activation_scale *
                (up_scale * static_cast<float>(up_dot) -
                 up_min * static_cast<float>(x_sum));
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_xor_sync(0xffffffffu, up_acc, offset);
    }
    if (lane == 0) {
        partial_gate[warp] = gate_acc;
        partial_up[warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        gate_acc = lane < NWARPS ? partial_gate[lane] : 0.0f;
        up_acc = lane < NWARPS ? partial_up[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffffu, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffffu, up_acc, offset);
        }
        if (lane == 0) {
            const float gate = __half2float(__float2half_rn(gate_acc));
            const float up = __half2float(__float2half_rn(up_acc));
            out[static_cast<size_t>(output_row) * hidden_width + hidden_row] =
                __float2half_rn(mfq_glu<GELU>(gate, up));
        }
    }
}

template <int TOKENS_PER_BLOCK, int PROFILE_MASK>
__global__ void __launch_bounds__(256) nint_moe_hetero_mmvq_kernel(
        const int64_t * __restrict__ weight_ptrs,
        const int32_t * __restrict__ pool_params,
        const int64_t * __restrict__ activation_ptrs,
        const int32_t * __restrict__ expert_pool,
        const int32_t * __restrict__ expert_local,
        const int32_t * __restrict__ ids,
        __half * __restrict__ out,
        int tokens,
        int routes,
        int experts,
        int out_per_expert,
        bool routed_input) {
    constexpr int rows_per_warp = 2;
    const int token = blockIdx.z * TOKENS_PER_BLOCK + threadIdx.y;
    const int route = blockIdx.y;
    const int lane = threadIdx.x;
    const int row0 = blockIdx.x * rows_per_warp;
    if (token >= tokens) {
        return;
    }
    const int pair = token * routes + route;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >= static_cast<unsigned int>(experts)) {
        return;
    }
    const int pool = expert_pool[expert];
    const int local_expert = expert_local[expert];
    if (pool < 0 || local_expert < 0) {
        return;
    }
    const int64_t * weights = weight_ptrs + static_cast<size_t>(pool) * 5;
    const int64_t * activations = activation_ptrs + static_cast<size_t>(pool) * 2;
    const int32_t * params = pool_params + static_cast<size_t>(pool) * 2;
    const uint8_t * q_packed = ptr_from_i64<uint8_t>(weights[0]);
    const uint8_t * sub_scale = ptr_from_i64<uint8_t>(weights[1]);
    const uint8_t * sub_min = ptr_from_i64<uint8_t>(weights[2]);
    const float * neuron_scale = ptr_from_i64<float>(weights[3]);
    const float * neuron_min = ptr_from_i64<float>(weights[4]);
    const int8_t * qx = ptr_from_i64<int8_t>(activations[0]);
    const float * xscale = ptr_from_i64<float>(activations[1]);
    const int profile = params[0];
    const int groups = params[1];
    const int source_row = routed_input ? pair : token;

#define MFQ_MOE_MMVQ_PROFILE(BITS_VALUE, GS_VALUE) \
    nint_moe_mmvq_profile<BITS_VALUE, GS_VALUE>( \
        q_packed, sub_scale, sub_min, neuron_scale, neuron_min, qx, xscale, out, \
        lane, row0, local_expert, source_row, pair, out_per_expert, groups, groups * GS_VALUE)
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint2Gs16)) != 0) {
        if (profile == kMoeProfileNint2Gs16) { MFQ_MOE_MMVQ_PROFILE(2, 16); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint4Gs24)) != 0) {
        if (profile == kMoeProfileNint4Gs24) { MFQ_MOE_MMVQ_PROFILE(4, 24); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint5Gs28)) != 0) {
        if (profile == kMoeProfileNint5Gs28) { MFQ_MOE_MMVQ_PROFILE(5, 28); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint6Gs24)) != 0) {
        if (profile == kMoeProfileNint6Gs24) { MFQ_MOE_MMVQ_PROFILE(6, 24); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint8Gs48)) != 0) {
        if (profile == kMoeProfileNint8Gs48) { MFQ_MOE_MMVQ_PROFILE(8, 48); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint8Gs24)) != 0) {
        if (profile == kMoeProfileNint8Gs24) { MFQ_MOE_MMVQ_PROFILE(8, 24); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint3Gs24)) != 0) {
        if (profile == kMoeProfileNint3Gs24) { MFQ_MOE_MMVQ_PROFILE(3, 24); return; }
    }
#undef MFQ_MOE_MMVQ_PROFILE
}

template <int PROFILE_MASK, int NWARPS>
__global__ void __launch_bounds__(NWARPS * kWarpSize)
nint_moe_hetero_mmvq_group_kernel(
        const int64_t * __restrict__ weight_ptrs,
        const int32_t * __restrict__ pool_params,
        const int64_t * __restrict__ activation_ptrs,
        const int32_t * __restrict__ expert_pool,
        const int32_t * __restrict__ expert_local,
        const int32_t * __restrict__ ids,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        bool routed_input) {
    __shared__ float partial[NWARPS];
    const int route = blockIdx.y;
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int expert = ids[route];
    if (static_cast<unsigned int>(expert) >= static_cast<unsigned int>(experts)) return;
    const int pool = expert_pool[expert];
    const int local_expert = expert_local[expert];
    if (pool < 0 || local_expert < 0) return;
    const int64_t * weights = weight_ptrs + static_cast<size_t>(pool) * 5;
    const int64_t * activations = activation_ptrs + static_cast<size_t>(pool) * 2;
    const int32_t * params = pool_params + static_cast<size_t>(pool) * 2;
    const uint8_t * q_packed = ptr_from_i64<uint8_t>(weights[0]);
    const uint8_t * sub_scale = ptr_from_i64<uint8_t>(weights[1]);
    const uint8_t * sub_min = ptr_from_i64<uint8_t>(weights[2]);
    const float * neuron_scale = ptr_from_i64<float>(weights[3]);
    const float * neuron_min = ptr_from_i64<float>(weights[4]);
    const int8_t * qx = ptr_from_i64<int8_t>(activations[0]);
    const float * xscale = ptr_from_i64<float>(activations[1]);
    const int profile = params[0];
    const int groups = params[1];
    const int source_row = routed_input ? route : 0;

#define MFQ_MOE_GROUP_PROFILE(BITS_VALUE, GS_VALUE) \
    nint_moe_mmvq_group_profile<BITS_VALUE, GS_VALUE, NWARPS>( \
        q_packed, sub_scale, sub_min, neuron_scale, neuron_min, qx, xscale, out, partial, \
        warp, lane, row, local_expert, source_row, route, out_per_expert, groups)
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint2Gs16)) != 0) {
        if (profile == kMoeProfileNint2Gs16) { MFQ_MOE_GROUP_PROFILE(2, 16); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint4Gs24)) != 0) {
        if (profile == kMoeProfileNint4Gs24) { MFQ_MOE_GROUP_PROFILE(4, 24); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint5Gs28)) != 0) {
        if (profile == kMoeProfileNint5Gs28) { MFQ_MOE_GROUP_PROFILE(5, 28); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint6Gs24)) != 0) {
        if (profile == kMoeProfileNint6Gs24) { MFQ_MOE_GROUP_PROFILE(6, 24); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint8Gs48)) != 0) {
        if (profile == kMoeProfileNint8Gs48) { MFQ_MOE_GROUP_PROFILE(8, 48); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint8Gs24)) != 0) {
        if (profile == kMoeProfileNint8Gs24) { MFQ_MOE_GROUP_PROFILE(8, 24); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint3Gs24)) != 0) {
        if (profile == kMoeProfileNint3Gs24) { MFQ_MOE_GROUP_PROFILE(3, 24); return; }
    }
#undef MFQ_MOE_GROUP_PROFILE
}

template <int PROFILE_MASK, int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
nint_moe_hetero_mmvq_groupwarp_kernel(
        const int64_t * __restrict__ weight_ptrs,
        const int32_t * __restrict__ pool_params,
        const int64_t * __restrict__ activation_ptrs,
        const int32_t * __restrict__ expert_pool,
        const int32_t * __restrict__ expert_local,
        const int32_t * __restrict__ ids,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        bool routed_input) {
    const int route = blockIdx.y;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= out_per_expert) return;
    const int expert = ids[route];
    if (static_cast<unsigned int>(expert) >= static_cast<unsigned int>(experts)) return;
    const int pool = expert_pool[expert];
    const int local_expert = expert_local[expert];
    if (pool < 0 || local_expert < 0) return;
    const int64_t * weights = weight_ptrs + static_cast<size_t>(pool) * 5;
    const int64_t * activations = activation_ptrs + static_cast<size_t>(pool) * 2;
    const int32_t * params = pool_params + static_cast<size_t>(pool) * 2;
    const uint8_t * q_packed = ptr_from_i64<uint8_t>(weights[0]);
    const uint8_t * sub_scale = ptr_from_i64<uint8_t>(weights[1]);
    const uint8_t * sub_min = ptr_from_i64<uint8_t>(weights[2]);
    const float * neuron_scale = ptr_from_i64<float>(weights[3]);
    const float * neuron_min = ptr_from_i64<float>(weights[4]);
    const int8_t * qx = ptr_from_i64<int8_t>(activations[0]);
    const float * xscale = ptr_from_i64<float>(activations[1]);
    const int profile = params[0];
    const int groups = params[1];
    const int source_row = routed_input ? route : 0;

#define MFQ_MOE_GROUPWARP_PROFILE(BITS_VALUE, GS_VALUE) \
    nint_moe_mmvq_groupwarp_profile<BITS_VALUE, GS_VALUE>( \
        q_packed, sub_scale, sub_min, neuron_scale, neuron_min, qx, xscale, out, \
        lane, row, local_expert, source_row, route, out_per_expert, groups)
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint2Gs16)) != 0) {
        if (profile == kMoeProfileNint2Gs16) { MFQ_MOE_GROUPWARP_PROFILE(2, 16); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint4Gs24)) != 0) {
        if (profile == kMoeProfileNint4Gs24) { MFQ_MOE_GROUPWARP_PROFILE(4, 24); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint5Gs28)) != 0) {
        if (profile == kMoeProfileNint5Gs28) { MFQ_MOE_GROUPWARP_PROFILE(5, 28); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint6Gs24)) != 0) {
        if (profile == kMoeProfileNint6Gs24) { MFQ_MOE_GROUPWARP_PROFILE(6, 24); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint8Gs48)) != 0) {
        if (profile == kMoeProfileNint8Gs48) { MFQ_MOE_GROUPWARP_PROFILE(8, 48); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint8Gs24)) != 0) {
        if (profile == kMoeProfileNint8Gs24) { MFQ_MOE_GROUPWARP_PROFILE(8, 24); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint3Gs24)) != 0) {
        if (profile == kMoeProfileNint3Gs24) { MFQ_MOE_GROUPWARP_PROFILE(3, 24); return; }
    }
#undef MFQ_MOE_GROUPWARP_PROFILE
}

template <int PROFILE_MASK, int NWARPS, bool GELU>
__global__ void __launch_bounds__(NWARPS * kWarpSize)
nint_moe_hetero_mmvq_glu_group_kernel(
        const int64_t * __restrict__ weight_ptrs,
        const int32_t * __restrict__ pool_params,
        const int64_t * __restrict__ activation_ptrs,
        const int32_t * __restrict__ expert_pool,
        const int32_t * __restrict__ expert_local,
        const int32_t * __restrict__ ids,
        __half * __restrict__ out,
        int routes,
        int experts,
        int hidden_width) {
    __shared__ float partial_gate[NWARPS];
    __shared__ float partial_up[NWARPS];
    const int token = blockIdx.z;
    const int route = blockIdx.y;
    const int output_row = token * routes + route;
    const int hidden_row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int expert = ids[output_row];
    if (static_cast<unsigned int>(expert) >= static_cast<unsigned int>(experts)) return;
    const int pool = expert_pool[expert];
    const int local_expert = expert_local[expert];
    if (pool < 0 || local_expert < 0) return;
    const int64_t * weights = weight_ptrs + static_cast<size_t>(pool) * 5;
    const int64_t * activations = activation_ptrs + static_cast<size_t>(pool) * 2;
    const int32_t * params = pool_params + static_cast<size_t>(pool) * 2;
    const uint8_t * q_packed = ptr_from_i64<uint8_t>(weights[0]);
    const uint8_t * sub_scale = ptr_from_i64<uint8_t>(weights[1]);
    const uint8_t * sub_min = ptr_from_i64<uint8_t>(weights[2]);
    const float * neuron_scale = ptr_from_i64<float>(weights[3]);
    const float * neuron_min = ptr_from_i64<float>(weights[4]);
    const int8_t * qx = ptr_from_i64<int8_t>(activations[0]);
    const float * xscale = ptr_from_i64<float>(activations[1]);
    const int profile = params[0];
    const int groups = params[1];

#define MFQ_MOE_GLU_GROUP_PROFILE(BITS_VALUE, GS_VALUE) \
    nint_moe_mmvq_glu_group_profile<BITS_VALUE, GS_VALUE, NWARPS, GELU>( \
        q_packed, sub_scale, sub_min, neuron_scale, neuron_min, qx, xscale, out, \
        partial_gate, partial_up, warp, lane, hidden_row, local_expert, token, output_row, \
        hidden_width, groups)
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint2Gs16)) != 0) {
        if (profile == kMoeProfileNint2Gs16) { MFQ_MOE_GLU_GROUP_PROFILE(2, 16); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint4Gs24)) != 0) {
        if (profile == kMoeProfileNint4Gs24) { MFQ_MOE_GLU_GROUP_PROFILE(4, 24); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint5Gs28)) != 0) {
        if (profile == kMoeProfileNint5Gs28) { MFQ_MOE_GLU_GROUP_PROFILE(5, 28); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint6Gs24)) != 0) {
        if (profile == kMoeProfileNint6Gs24) { MFQ_MOE_GLU_GROUP_PROFILE(6, 24); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint8Gs48)) != 0) {
        if (profile == kMoeProfileNint8Gs48) { MFQ_MOE_GLU_GROUP_PROFILE(8, 48); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint8Gs24)) != 0) {
        if (profile == kMoeProfileNint8Gs24) { MFQ_MOE_GLU_GROUP_PROFILE(8, 24); return; }
    }
    if constexpr ((PROFILE_MASK & (1 << kMoeProfileNint3Gs24)) != 0) {
        if (profile == kMoeProfileNint3Gs24) { MFQ_MOE_GLU_GROUP_PROFILE(3, 24); return; }
    }
#undef MFQ_MOE_GLU_GROUP_PROFILE
}

template <int BITS, int GS, int TILE_M>
__device__ __forceinline__ void nint_moe_grouped_tile_profile(
        const uint8_t * __restrict__ q_packed,
        const uint8_t * __restrict__ sub_scale,
        const uint8_t * __restrict__ sub_min,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_min,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids_dst,
        __half * __restrict__ out,
        int lane,
        int row,
        int first,
        int last,
        int local_expert,
        int routes,
        int out_per_expert,
        int groups,
        bool routed_input) {
    constexpr int qbytes = (GS * BITS + 7) / 8;
    constexpr int chunks = (GS + 3) / 4;
    constexpr int groups_per_warp = kWarpSize / chunks;
    float acc[TILE_M];
#pragma unroll
    for (int item = 0; item < TILE_M; ++item) {
        acc[item] = 0.0f;
    }
    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int group_offset = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    const int weight_row = local_expert * out_per_expert + row;
    const float neuron_d = neuron_scale[weight_row];
    const float neuron_m = neuron_min[weight_row];
    const int k_pad = groups * GS;

    for (int group_base = 0; group_base < groups; group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups || group_offset >= GS) {
            continue;
        }
        const int width = min(4, GS - group_offset);
        const size_t meta = static_cast<size_t>(weight_row) * groups + group;
        const uint8_t * q_group = q_packed + meta * qbytes;
        int packed_weight = 0;
        if (width == 4) {
            packed_weight = unpack_four<BITS>(q_group, group_offset);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i < width) {
                    packed_weight |= static_cast<int>(
                        unpack_one<BITS>(q_group, group_offset + i)) << (8 * i);
                }
            }
        }
        const float weight_scale = neuron_d * static_cast<float>(sub_scale[meta]);
        const float weight_min = neuron_m * static_cast<float>(sub_min[meta]);
        const int k = group * GS + group_offset;

#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
            const int compact = first + item;
            if (compact >= last) {
                continue;
            }
            const int pair = ids_dst[compact];
            const int source_row = routed_input ? pair : pair / routes;
            const int8_t * x_ptr = qx + static_cast<size_t>(source_row) * k_pad + k;
            int packed_x = 0;
            int x_sum = 0;
            if (width == 4) {
                packed_x = load_i8x4(x_ptr);
                x_sum = __dp4a(0x01010101, packed_x, 0);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (i < width) {
                        const int value = static_cast<int>(x_ptr[i]);
                        packed_x |= (value & 255) << (8 * i);
                        x_sum += value;
                    }
                }
            }
            const int dot = quant_dot<BITS>(packed_weight, packed_x, x_sum);
            const float activation_scale = xscale[static_cast<size_t>(source_row) * groups + group];
            acc[item] += activation_scale *
                (weight_scale * static_cast<float>(dot) - weight_min * static_cast<float>(x_sum));
        }
    }

#pragma unroll
    for (int item = 0; item < TILE_M; ++item) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[item] += __shfl_xor_sync(0xffffffffu, acc[item], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
            const int compact = first + item;
            if (compact < last) {
                const int pair = ids_dst[compact];
                out[static_cast<size_t>(pair) * out_per_expert + row] = __float2half(acc[item]);
            }
        }
    }
}

template <int TILE_M>
__device__ __forceinline__ void nint8_zero_moe_grouped_tile_profile(
        const uint8_t * __restrict__ q,
        const __half * __restrict__ scale,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids_dst,
        __half * __restrict__ out,
        int lane,
        int row,
        int first,
        int last,
        int local_expert,
        int routes,
        int out_per_expert,
        int groups,
        bool routed_input) {
    constexpr int gs = 32;
    constexpr int chunks = gs / 4;
    constexpr int groups_per_warp = kWarpSize / chunks;
    float acc[TILE_M];
#pragma unroll
    for (int item = 0; item < TILE_M; ++item) {
        acc[item] = 0.0f;
    }
    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int group_offset = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    const int weight_row = local_expert * out_per_expert + row;
    const int k_pad = groups * gs;

    for (int group_base = 0; group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups) {
            continue;
        }
        const size_t meta =
            static_cast<size_t>(weight_row) * groups + group;
        const int packed_weight = load_i8x4(
            reinterpret_cast<const int8_t *>(
                q + meta * gs + group_offset));
        const float weight_scale = __half2float(scale[meta]);
        const int k = group * gs + group_offset;

#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
            const int compact = first + item;
            if (compact >= last) {
                continue;
            }
            const int pair = ids_dst[compact];
            const int source_row = routed_input ? pair : pair / routes;
            const int packed_x = load_i8x4(
                qx + static_cast<size_t>(source_row) * k_pad + k);
            const int dot = __dp4a(packed_weight, packed_x, 0);
            const float activation_scale =
                xscale[static_cast<size_t>(source_row) * groups + group];
            acc[item] +=
                activation_scale * weight_scale * static_cast<float>(dot);
        }
    }

#pragma unroll
    for (int item = 0; item < TILE_M; ++item) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[item] += __shfl_xor_sync(0xffffffffu, acc[item], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
            const int compact = first + item;
            if (compact < last) {
                const int pair = ids_dst[compact];
                out[static_cast<size_t>(pair) * out_per_expert + row] =
                    __float2half(acc[item]);
            }
        }
    }
}

template <int TILE_M>
__global__ void __launch_bounds__(128) nint8_zero_moe_grouped_tile_kernel(
        const uint8_t * __restrict__ q,
        const __half * __restrict__ scale,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ expert_local,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int max_tiles,
        bool routed_input) {
    const int row_tiles =
        (out_per_expert + kRowsPerBlock - 1) / kRowsPerBlock;
    const int linear_block = static_cast<int>(blockIdx.x);
    const int row_tile = linear_block % row_tiles;
    const int tile = linear_block / row_tiles;
    if (tile >= max_tiles || tile >= tile_bounds[experts]) {
        return;
    }
    const int expert = find_expert_tile(tile_bounds, experts, tile);
    if (expert >= experts) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (local_expert < 0) {
        return;
    }
    const int local_tile = tile - tile_bounds[expert];
    const int first = expert_bounds[expert] + local_tile * TILE_M;
    const int last = min(first + TILE_M, expert_bounds[expert + 1]);
    const int row = row_tile * kRowsPerBlock + threadIdx.y;
    if (row >= out_per_expert) {
        return;
    }
    nint8_zero_moe_grouped_tile_profile<TILE_M>(
        q, scale, qx, xscale, ids_dst, out, threadIdx.x, row, first, last,
        local_expert, routes, out_per_expert, groups, routed_input);
}

template <int TILE_M>
__global__ void __launch_bounds__(128)
nint8_zero_moe_grouped_tile_persistent_kernel(
        const uint8_t * __restrict__ q,
        const __half * __restrict__ scale,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        const int32_t * __restrict__ expert_local,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        bool routed_input) {
    const int row_tiles =
        (out_per_expert + kRowsPerBlock - 1) / kRowsPerBlock;
    const int total_tiles = tile_bounds[experts];
    const int64_t total_tasks =
        static_cast<int64_t>(total_tiles) * row_tiles;
    for (int64_t task = blockIdx.x; task < total_tasks;
         task += gridDim.x) {
        const int row_tile = static_cast<int>(task % row_tiles);
        const int tile = static_cast<int>(task / row_tiles);
        const int expert = tile_experts[tile];
        const int local_expert = expert_local[expert];
        if (local_expert < 0) {
            continue;
        }
        const int local_tile = tile - tile_bounds[expert];
        const int first = expert_bounds[expert] + local_tile * TILE_M;
        const int last = min(first + TILE_M, expert_bounds[expert + 1]);
        const int row = row_tile * kRowsPerBlock + threadIdx.y;
        if (row >= out_per_expert) {
            continue;
        }
        nint8_zero_moe_grouped_tile_profile<TILE_M>(
            q, scale, qx, xscale, ids_dst, out, threadIdx.x, row, first,
            last, local_expert, routes, out_per_expert, groups,
            routed_input);
    }
}

template <int BITS, int GS, int TILE_M>
__global__ void __launch_bounds__(128) nint_moe_grouped_tile_persistent_kernel(
        const uint8_t * __restrict__ q_packed,
        const uint8_t * __restrict__ sub_scale,
        const uint8_t * __restrict__ sub_min,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_min,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        const int32_t * __restrict__ expert_local,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        bool routed_input) {
    const int row_tiles = (out_per_expert + kRowsPerBlock - 1) / kRowsPerBlock;
    const int total_tiles = tile_bounds[experts];
    const int64_t total_tasks = static_cast<int64_t>(total_tiles) * row_tiles;
    for (int64_t task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        const int row_tile = static_cast<int>(task % row_tiles);
        const int tile = static_cast<int>(task / row_tiles);
        const int expert = tile_experts[tile];
        const int local_expert = expert_local[expert];
        if (local_expert < 0) {
            continue;
        }
        const int local_tile = tile - tile_bounds[expert];
        const int first = expert_bounds[expert] + local_tile * TILE_M;
        const int last = min(first + TILE_M, expert_bounds[expert + 1]);
        const int row = row_tile * kRowsPerBlock + threadIdx.y;
        if (row >= out_per_expert) {
            continue;
        }
        nint_moe_grouped_tile_profile<BITS, GS, TILE_M>(
            q_packed, sub_scale, sub_min, neuron_scale, neuron_min, qx, xscale,
            ids_dst, out, threadIdx.x, row, first, last, local_expert, routes,
            out_per_expert, groups, routed_input);
    }
}

template <int TILE_M>
__global__ void __launch_bounds__(128) nint_moe_hetero_grouped_tile_kernel(
        const int64_t * __restrict__ weight_ptrs,
        const int32_t * __restrict__ pool_params,
        const int64_t * __restrict__ activation_ptrs,
        const int32_t * __restrict__ expert_pool,
        const int32_t * __restrict__ expert_local,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        bool routed_input) {
    const int row_tiles = (out_per_expert + kRowsPerBlock - 1) / kRowsPerBlock;
    const int total_tiles = tile_bounds[experts];
    const int64_t total_tasks = static_cast<int64_t>(total_tiles) * row_tiles;
    for (int64_t task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        const int row_tile = static_cast<int>(task % row_tiles);
        const int tile = static_cast<int>(task / row_tiles);
        const int expert = tile_experts[tile];
        const int pool = expert_pool[expert];
        const int local_expert = expert_local[expert];
        const int local_tile = tile - tile_bounds[expert];
        const int first = expert_bounds[expert] + local_tile * TILE_M;
        const int last = min(first + TILE_M, expert_bounds[expert + 1]);
        const int row = row_tile * kRowsPerBlock + threadIdx.y;
        const int lane = threadIdx.x;
        if (row >= out_per_expert || pool < 0 || local_expert < 0) {
            continue;
        }

        const int64_t * weights = weight_ptrs + static_cast<size_t>(pool) * 5;
        const int64_t * activations = activation_ptrs + static_cast<size_t>(pool) * 2;
        const int32_t * params = pool_params + static_cast<size_t>(pool) * 2;
        const uint8_t * q_packed = ptr_from_i64<uint8_t>(weights[0]);
        const uint8_t * sub_scale = ptr_from_i64<uint8_t>(weights[1]);
        const uint8_t * sub_min = ptr_from_i64<uint8_t>(weights[2]);
        const float * neuron_scale = ptr_from_i64<float>(weights[3]);
        const float * neuron_min = ptr_from_i64<float>(weights[4]);
        const int8_t * qx = ptr_from_i64<int8_t>(activations[0]);
        const float * xscale = ptr_from_i64<float>(activations[1]);
        const int profile = params[0];
        const int groups = params[1];

#define MFQ_MOE_TILE_PROFILE(BITS_VALUE, GS_VALUE) \
        nint_moe_grouped_tile_profile<BITS_VALUE, GS_VALUE, TILE_M>( \
            q_packed, sub_scale, sub_min, neuron_scale, neuron_min, qx, xscale, ids_dst, out, \
            lane, row, first, last, local_expert, routes, out_per_expert, groups, routed_input)
        switch (profile) {
            case kMoeProfileNint2Gs16: MFQ_MOE_TILE_PROFILE(2, 16); break;
            case kMoeProfileNint3Gs24: MFQ_MOE_TILE_PROFILE(3, 24); break;
            case kMoeProfileNint4Gs24: MFQ_MOE_TILE_PROFILE(4, 24); break;
            case kMoeProfileNint5Gs28: MFQ_MOE_TILE_PROFILE(5, 28); break;
            case kMoeProfileNint6Gs24: MFQ_MOE_TILE_PROFILE(6, 24); break;
            case kMoeProfileNint8Gs48: MFQ_MOE_TILE_PROFILE(8, 48); break;
            case kMoeProfileNint8Gs24: MFQ_MOE_TILE_PROFILE(8, 24); break;
        }
#undef MFQ_MOE_TILE_PROFILE
    }
}

static __device__ __forceinline__ int moe_mma168_i(int l) {
    return ((l / 2) * 8) + (threadIdx.x / 4);
}

static __device__ __forceinline__ int moe_mma168_j(int l) {
    return ((threadIdx.x % 4) * 2) + (l % 2);
}

static __device__ __forceinline__ void moe_load_mma_a_m16n8k32(
        int (&a)[4], const int * ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "l"(ptr));
}

static __device__ __forceinline__ void moe_load_mma_b_m16n8k32(
        int (&b)[2], const int * ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(b[0]), "=r"(b[1]) : "l"(ptr));
}

static __device__ __forceinline__ void moe_mma_m16n8k32_s8(
        int (&d)[4], const int (&a)[4], const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int BITS, int GS>
__global__ void __launch_bounds__(256) nint_moe_group32_mmq_kernel(
        const uint8_t * __restrict__ q_packed,
        const uint8_t * __restrict__ sub_scale,
        const uint8_t * __restrict__ sub_min,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_min,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        const int32_t * __restrict__ expert_local,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_pad,
        bool routed_input) {
    constexpr int BM = 16;
    constexpr int BN = 64;
    constexpr int NW = 8;
    constexpr int ROWS_PER_WARP = 8;
    constexpr int CHUNK_GROUPS = 8;
    constexpr int GROUP_KPACK = 8;
    constexpr int KSTRIDE = CHUNK_GROUPS * GROUP_KPACK + 4;
    constexpr int QBYTES = (GS * BITS + 7) / 8;
    constexpr int QUARTETS = GS / 4;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * 32 + lane;
    const int ntiles = (out_per_expert + BN - 1) / BN;
    const int fine_tile = blockIdx.x / ntiles;
    const int ntile = blockIdx.x - fine_tile * ntiles;
    const int total_fine_tiles = tile_bounds[experts];
    if (fine_tile >= total_fine_tiles) {
        return;
    }
    const int expert = tile_experts[fine_tile];
    const int local_fine_tile = fine_tile - tile_bounds[expert];
    if ((local_fine_tile & 1) != 0) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (local_expert < 0) {
        return;
    }
    const int first = expert_bounds[expert] + local_fine_tile * kRouteTile;
    const int last = min(first + BM, expert_bounds[expert + 1]);
    const int n0 = ntile * BN;

    __shared__ int As[BM][KSTRIDE];
    __shared__ int Bs[NW][ROWS_PER_WARP][KSTRIDE];
    __shared__ float Wd[CHUNK_GROUPS][BN];
    __shared__ float Wm[CHUNK_GROUPS][BN];
    __shared__ float Axs[CHUNK_GROUPS][BM];
    __shared__ int Axsum[CHUNK_GROUPS][BM];

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nchunks = (groups + CHUNK_GROUPS - 1) / CHUNK_GROUPS;
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        const int gbase = chunk * CHUNK_GROUPS;
        const int eff = min(CHUNK_GROUPS, groups - gbase);

        constexpr int A_WORDS = BM * CHUNK_GROUPS * GROUP_KPACK;
        for (int index = tid; index < A_WORDS; index += NW * 32) {
            const int ml = index / (CHUNK_GROUPS * GROUP_KPACK);
            const int rem = index - ml * CHUNK_GROUPS * GROUP_KPACK;
            const int gl = rem / GROUP_KPACK;
            const int quartet = rem - gl * GROUP_KPACK;
            int packed = 0;
            const int compact = first + ml;
            if (compact < last && gl < eff && quartet < QUARTETS) {
                const int pair = ids_dst[compact];
                const int source_row = routed_input ? pair : pair / routes;
                const int k = (gbase + gl) * GS + quartet * 4;
                packed = load_i8x4(qx + static_cast<size_t>(source_row) * k_pad + k);
            }
            As[ml][gl * GROUP_KPACK + quartet] = packed;
        }
        __syncthreads();

        constexpr int W_TASKS = BN * CHUNK_GROUPS;
        for (int task = tid; task < W_TASKS; task += NW * 32) {
            const int nn = task / CHUNK_GROUPS;
            const int gl = task - nn * CHUNK_GROUPS;
            const int gn = n0 + nn;
            const int group = gbase + gl;
            const bool valid = gn < out_per_expert && gl < eff;
            float d = 0.0f;
            float m = 0.0f;
            const uint8_t * qg = nullptr;
            if (valid) {
                const int weight_row = local_expert * out_per_expert + gn;
                const size_t meta = static_cast<size_t>(weight_row) * groups + group;
                d = neuron_scale[weight_row] * static_cast<float>(sub_scale[meta]);
                m = neuron_min[weight_row] * static_cast<float>(sub_min[meta]);
                qg = q_packed + meta * QBYTES;
                if constexpr (BITS == 8) {
                    m -= 128.0f * d;
                }
            }
#pragma unroll
            for (int quartet = 0; quartet < GROUP_KPACK; ++quartet) {
                int packed = 0;
                if (valid && quartet < QUARTETS) {
                    packed = unpack_four<BITS>(qg, quartet * 4);
                    if constexpr (BITS == 8) {
                        packed ^= static_cast<int>(0x80808080u);
                    }
                }
                Bs[nn / ROWS_PER_WARP][nn % ROWS_PER_WARP]
                  [gl * GROUP_KPACK + quartet] = packed;
            }
            Wd[gl][nn] = d;
            Wm[gl][nn] = m;
        }

        constexpr int A_META = BM * CHUNK_GROUPS;
        for (int index = tid; index < A_META; index += NW * 32) {
            const int ml = index / CHUNK_GROUPS;
            const int gl = index - ml * CHUNK_GROUPS;
            const int compact = first + ml;
            float scale = 0.0f;
            int sum = 0;
            if (compact < last && gl < eff) {
                const int pair = ids_dst[compact];
                const int source_row = routed_input ? pair : pair / routes;
                scale = xscale[static_cast<size_t>(source_row) * groups + gbase + gl];
#pragma unroll
                for (int quartet = 0; quartet < GROUP_KPACK; ++quartet) {
                    sum = __dp4a(0x01010101, As[ml][gl * GROUP_KPACK + quartet], sum);
                }
            }
            Axs[gl][ml] = scale;
            Axsum[gl][ml] = sum;
        }
        __syncthreads();

#pragma unroll
        for (int gl = 0; gl < CHUNK_GROUPS; ++gl) {
            int ar[4];
            int br[2];
            moe_load_mma_a_m16n8k32(
                ar, &As[lane % 16][gl * GROUP_KPACK + (lane / 16) * 4]);
            moe_load_mma_b_m16n8k32(
                br, &Bs[warp][lane % 8][gl * GROUP_KPACK + (((lane / 8) * 4) & 7)]);
            int dot[4] = {0, 0, 0, 0};
            moe_mma_m16n8k32_s8(dot, ar, br);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int ml = moe_mma168_i(l);
                const int nn = warp * ROWS_PER_WARP + moe_mma168_j(l);
                if (gl < eff && first + ml < last && n0 + nn < out_per_expert) {
                    const float xs = Axs[gl][ml];
                    acc[l] = fmaf(xs * Wd[gl][nn], static_cast<float>(dot[l]), acc[l]);
                    acc[l] = fmaf(-xs * Wm[gl][nn], static_cast<float>(Axsum[gl][ml]), acc[l]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int ml = moe_mma168_i(l);
        const int gn = n0 + warp * ROWS_PER_WARP + moe_mma168_j(l);
        const int compact = first + ml;
        if (compact < last && gn < out_per_expert) {
            const int pair = ids_dst[compact];
            out[static_cast<size_t>(pair) * out_per_expert + gn] = __float2half(acc[l]);
        }
    }
}

constexpr int kMoeMmaBn = 64;
constexpr int kMoeMmaMaxBkStride = 120;

template <int BITS, int GS, int GROUPS_PER_CHUNK, int BM>
__device__ __forceinline__ void nint_moe_mma_profile(
        const uint8_t * __restrict__ q_packed,
        const uint8_t * __restrict__ sub_scale,
        const uint8_t * __restrict__ sub_min,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_min,
        const __half * __restrict__ x,
        const int32_t * __restrict__ ids_dst,
        __half * __restrict__ out,
        __half (*W_s)[kMoeMmaMaxBkStride],
        __half (*X_s)[kMoeMmaMaxBkStride],
        float (*C_s)[16][16],
        int first,
        int last,
        int n0,
        int local_expert,
        int routes,
        int out_per_expert,
        int weight_out_stride,
        int weight_row_offset,
        int groups,
        int k_real,
        bool routed_input) {
    constexpr int BK = GS * GROUPS_PER_CHUNK;
    constexpr int MTILES = BM / 16;
    constexpr int NFRAGS = kMoeMmaBn / 16;
    constexpr int ACCS_PER_WARP = (MTILES + 1) / 2;
    constexpr int QBYTES = (GS * BITS + 7) / 8;
    static_assert(BK % 16 == 0 && BK <= kMoeMmaMaxBkStride);
    static_assert(BM == 16 || BM == 32 || BM == 64);

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int warp_m0 = warp / NFRAGS;
    const int warp_n = warp % NFRAGS;

    using FragA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                                         __half, nvcuda::wmma::row_major>;
    using FragB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                                         __half, nvcuda::wmma::col_major>;
    using FragC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>;
    FragC acc[ACCS_PER_WARP];
#pragma unroll
    for (int a = 0; a < ACCS_PER_WARP; ++a) {
        nvcuda::wmma::fill_fragment(acc[a], 0.0f);
    }

    const int chunks = (groups + GROUPS_PER_CHUNK - 1) / GROUPS_PER_CHUNK;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int gbase = chunk * GROUPS_PER_CHUNK;
        const int kb = gbase * GS;
        const int weight_tasks = kMoeMmaBn * GROUPS_PER_CHUNK;
        for (int task = tid; task < weight_tasks; task += 256) {
            const int nn = task / GROUPS_PER_CHUNK;
            const int gl = task - nn * GROUPS_PER_CHUNK;
            const int gn = n0 + nn;
            const int group = gbase + gl;
            const bool valid = gn < out_per_expert && group < groups;
            float d = 0.0f;
            float m = 0.0f;
            const uint8_t * qg = nullptr;
            if (valid) {
                const int weight_row =
                    local_expert * weight_out_stride + weight_row_offset + gn;
                const size_t meta = static_cast<size_t>(weight_row) * groups + group;
                d = neuron_scale[weight_row] * static_cast<float>(sub_scale[meta]);
                m = neuron_min[weight_row] * static_cast<float>(sub_min[meta]);
                qg = q_packed + meta * QBYTES;
            }
#pragma unroll
            for (int i = 0; i < GS; ++i) {
                const float qv = valid ? static_cast<float>(unpack_one<BITS>(qg, i)) : 0.0f;
                W_s[nn][gl * GS + i] = valid ? __float2half_rn(d * qv - m)
                                               : __float2half_rn(0.0f);
            }
        }

        constexpr int XPAIRS = BM * (BK / 2);
        for (int index = tid; index < XPAIRS; index += 256) {
            const int mm = index / (BK / 2);
            const int pair_k = index - mm * (BK / 2);
            const int compact = first + mm;
            const int k = kb + pair_k * 2;
            __half2 value = __float2half2_rn(0.0f);
            if (compact < last) {
                const int pair = ids_dst[compact];
                const int source_row = routed_input ? pair : pair / routes;
                const __half * row = x + static_cast<size_t>(source_row) * k_real;
                if (k + 1 < k_real) {
                    value = *reinterpret_cast<const __half2 *>(row + k);
                } else if (k < k_real) {
                    value = __halves2half2(row[k], __float2half_rn(0.0f));
                }
            }
            *reinterpret_cast<__half2 *>(&X_s[mm][pair_k * 2]) = value;
        }
        __syncthreads();

        const bool warp_active = warp_m0 < MTILES;
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            if (warp_active) {
                FragB bfrag;
                nvcuda::wmma::load_matrix_sync(
                    bfrag, &W_s[warp_n * 16][ks], kMoeMmaMaxBkStride);
#pragma unroll
                for (int a = 0; a < ACCS_PER_WARP; ++a) {
                    const int mi = warp_m0 + a * 2;
                    if (mi < MTILES) {
                        FragA afrag;
                        nvcuda::wmma::load_matrix_sync(
                            afrag, &X_s[mi * 16][ks], kMoeMmaMaxBkStride);
                        nvcuda::wmma::mma_sync(acc[a], afrag, bfrag, acc[a]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < ACCS_PER_WARP; ++a) {
        const int mi = warp_m0 + a * 2;
        const bool owns = mi < MTILES;
        if (owns) {
            nvcuda::wmma::store_matrix_sync(
                &C_s[warp][0][0], acc[a], 16, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        if (owns) {
            const int compact0 = first + mi * 16;
            const int gn0 = n0 + warp_n * 16;
#pragma unroll
            for (int element = lane; element < 16 * 16; element += 32) {
                const int r = element / 16;
                const int c = element - r * 16;
                const int compact = compact0 + r;
                const int gn = gn0 + c;
                if (compact < last && gn < out_per_expert) {
                    const int pair = ids_dst[compact];
                    out[static_cast<size_t>(pair) * out_per_expert + gn] =
                        __float2half_rn(C_s[warp][r][c]);
                }
            }
        }
        __syncthreads();
    }
}

template <int GROUPS_PER_CHUNK, int BM>
__device__ __forceinline__ void nint8_zero_moe_mma_profile(
        const uint8_t * __restrict__ q,
        const __half * __restrict__ scale,
        const __half * __restrict__ x,
        const int32_t * __restrict__ ids_dst,
        __half * __restrict__ out,
        __half (*W_s)[kMoeMmaMaxBkStride],
        __half (*X_s)[kMoeMmaMaxBkStride],
        float (*C_s)[16][16],
        int first,
        int last,
        int n0,
        int local_expert,
        int routes,
        int out_per_expert,
        int weight_out_stride,
        int weight_row_offset,
        int groups,
        int k_real,
        bool routed_input) {
    constexpr int GS = 32;
    constexpr int BK = GS * GROUPS_PER_CHUNK;
    constexpr int MTILES = BM / 16;
    constexpr int NFRAGS = kMoeMmaBn / 16;
    constexpr int ACCS_PER_WARP = (MTILES + 1) / 2;
    static_assert(BK % 16 == 0 && BK <= kMoeMmaMaxBkStride);
    static_assert(BM == 16 || BM == 32 || BM == 64);

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int warp_m0 = warp / NFRAGS;
    const int warp_n = warp % NFRAGS;

    using FragA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                                         __half, nvcuda::wmma::row_major>;
    using FragB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                                         __half, nvcuda::wmma::col_major>;
    using FragC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>;
    FragC acc[ACCS_PER_WARP];
#pragma unroll
    for (int a = 0; a < ACCS_PER_WARP; ++a) {
        nvcuda::wmma::fill_fragment(acc[a], 0.0f);
    }

    const int chunks = (groups + GROUPS_PER_CHUNK - 1) / GROUPS_PER_CHUNK;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int gbase = chunk * GROUPS_PER_CHUNK;
        const int kb = gbase * GS;
        const int weight_tasks = kMoeMmaBn * GROUPS_PER_CHUNK;
        for (int task = tid; task < weight_tasks; task += 256) {
            const int nn = task / GROUPS_PER_CHUNK;
            const int gl = task - nn * GROUPS_PER_CHUNK;
            const int gn = n0 + nn;
            const int group = gbase + gl;
            const bool valid = gn < out_per_expert && group < groups;
            const int8_t * qg = nullptr;
            float d = 0.0f;
            if (valid) {
                const int weight_row =
                    local_expert * weight_out_stride + weight_row_offset + gn;
                const size_t meta = static_cast<size_t>(weight_row) * groups + group;
                qg = reinterpret_cast<const int8_t *>(q + meta * GS);
                d = __half2float(scale[meta]);
            }
#pragma unroll
            for (int i = 0; i < GS; ++i) {
                const float value = valid
                    ? d * static_cast<float>(qg[i])
                    : 0.0f;
                W_s[nn][gl * GS + i] = __float2half_rn(value);
            }
        }

        constexpr int XPAIRS = BM * (BK / 2);
        for (int index = tid; index < XPAIRS; index += 256) {
            const int mm = index / (BK / 2);
            const int pair_k = index - mm * (BK / 2);
            const int compact = first + mm;
            const int k = kb + pair_k * 2;
            __half2 value = __float2half2_rn(0.0f);
            if (compact < last) {
                const int pair = ids_dst[compact];
                const int source_row = routed_input ? pair : pair / routes;
                const __half * row = x + static_cast<size_t>(source_row) * k_real;
                if (k + 1 < k_real) {
                    value = *reinterpret_cast<const __half2 *>(row + k);
                } else if (k < k_real) {
                    value = __halves2half2(row[k], __float2half_rn(0.0f));
                }
            }
            *reinterpret_cast<__half2 *>(&X_s[mm][pair_k * 2]) = value;
        }
        __syncthreads();

        const bool warp_active = warp_m0 < MTILES;
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            if (warp_active) {
                FragB bfrag;
                nvcuda::wmma::load_matrix_sync(
                    bfrag, &W_s[warp_n * 16][ks], kMoeMmaMaxBkStride);
#pragma unroll
                for (int a = 0; a < ACCS_PER_WARP; ++a) {
                    const int mi = warp_m0 + a * 2;
                    if (mi < MTILES) {
                        FragA afrag;
                        nvcuda::wmma::load_matrix_sync(
                            afrag, &X_s[mi * 16][ks], kMoeMmaMaxBkStride);
                        nvcuda::wmma::mma_sync(acc[a], afrag, bfrag, acc[a]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < ACCS_PER_WARP; ++a) {
        const int mi = warp_m0 + a * 2;
        const bool owns = mi < MTILES;
        if (owns) {
            nvcuda::wmma::store_matrix_sync(
                &C_s[warp][0][0], acc[a], 16, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        if (owns) {
            const int compact0 = first + mi * 16;
            const int gn0 = n0 + warp_n * 16;
#pragma unroll
            for (int element = lane; element < 16 * 16; element += 32) {
                const int r = element / 16;
                const int c = element - r * 16;
                const int compact = compact0 + r;
                const int gn = gn0 + c;
                if (compact < last && gn < out_per_expert) {
                    const int pair = ids_dst[compact];
                    out[static_cast<size_t>(pair) * out_per_expert + gn] =
                        __float2half_rn(C_s[warp][r][c]);
                }
            }
        }
        __syncthreads();
    }
}

template <int BM>
__global__ void __launch_bounds__(256, 1) nint8_zero_moe_mma_kernel(
        const uint8_t * __restrict__ q,
        const __half * __restrict__ scale,
        const int32_t * __restrict__ expert_local,
        const __half * __restrict__ x,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_real,
        bool routed_input) {
    constexpr int fine_tiles_per_mma = BM / kRouteTile;
    constexpr int groups_per_chunk = 3;
    __shared__ __half W_s[kMoeMmaBn][kMoeMmaMaxBkStride];
    __shared__ __half X_s[BM][kMoeMmaMaxBkStride];
    __shared__ float C_s[8][16][16];

    const int ntiles_n = (out_per_expert + kMoeMmaBn - 1) / kMoeMmaBn;
    const int total_fine_tiles = tile_bounds[experts];
    const int64_t total_tasks = static_cast<int64_t>(total_fine_tiles) * ntiles_n;
    for (int64_t task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        const int fine_tile = static_cast<int>(task / ntiles_n);
        const int ntile =
            static_cast<int>(task - static_cast<int64_t>(fine_tile) * ntiles_n);
        const int expert = tile_experts[fine_tile];
        const int local_fine_tile = fine_tile - tile_bounds[expert];
        if (local_fine_tile % fine_tiles_per_mma != 0) {
            continue;
        }
        const int local_expert = expert_local[expert];
        if (local_expert < 0) {
            continue;
        }
        const int first = expert_bounds[expert] + local_fine_tile * kRouteTile;
        const int last = min(first + BM, expert_bounds[expert + 1]);
        const int n0 = ntile * kMoeMmaBn;
        nint8_zero_moe_mma_profile<groups_per_chunk, BM>(
            q, scale, x, ids_dst, out, W_s, X_s, C_s, first, last, n0,
            local_expert, routes, out_per_expert, out_per_expert, 0,
            groups, k_real, routed_input);
    }
}

template <int BM>
__global__ void __launch_bounds__(256, 1) nint_moe_hetero_mma_kernel(
        const int64_t * __restrict__ weight_ptrs,
        const int32_t * __restrict__ pool_params,
        const int32_t * __restrict__ expert_pool,
        const int32_t * __restrict__ expert_local,
        const __half * __restrict__ x,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        int weight_out_stride,
        int weight_row_offset,
        int k_real,
        bool routed_input) {
    constexpr int fine_tiles_per_mma = BM / kRouteTile;
    __shared__ __half W_s[kMoeMmaBn][kMoeMmaMaxBkStride];
    __shared__ __half X_s[BM][kMoeMmaMaxBkStride];
    __shared__ float C_s[8][16][16];

    const int ntiles_n = (out_per_expert + kMoeMmaBn - 1) / kMoeMmaBn;
    const int total_fine_tiles = tile_bounds[experts];
    const int64_t total_tasks = static_cast<int64_t>(total_fine_tiles) * ntiles_n;
    for (int64_t task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        const int fine_tile = static_cast<int>(task / ntiles_n);
        const int ntile = static_cast<int>(task - static_cast<int64_t>(fine_tile) * ntiles_n);
        const int expert = tile_experts[fine_tile];
        const int local_fine_tile = fine_tile - tile_bounds[expert];
        if (local_fine_tile % fine_tiles_per_mma != 0) {
            continue;
        }
        const int pool = expert_pool[expert];
        const int local_expert = expert_local[expert];
        if (pool < 0 || local_expert < 0) {
            continue;
        }
        const int first = expert_bounds[expert] + local_fine_tile * kRouteTile;
        const int last = min(first + BM, expert_bounds[expert + 1]);
        const int n0 = ntile * kMoeMmaBn;
        const int64_t * weights = weight_ptrs + static_cast<size_t>(pool) * 5;
        const int32_t * params = pool_params + static_cast<size_t>(pool) * 2;
        const uint8_t * q_packed = ptr_from_i64<uint8_t>(weights[0]);
        const uint8_t * sub_scale = ptr_from_i64<uint8_t>(weights[1]);
        const uint8_t * sub_min = ptr_from_i64<uint8_t>(weights[2]);
        const float * neuron_scale = ptr_from_i64<float>(weights[3]);
        const float * neuron_min = ptr_from_i64<float>(weights[4]);
        const int profile = params[0];
        const int groups = params[1];

#define MFQ_MOE_MMA_PROFILE(BITS_VALUE, GS_VALUE, GROUPS_VALUE) \
        nint_moe_mma_profile<BITS_VALUE, GS_VALUE, GROUPS_VALUE, BM>( \
            q_packed, sub_scale, sub_min, neuron_scale, neuron_min, x, ids_dst, out, \
            W_s, X_s, C_s, first, last, n0, local_expert, routes, out_per_expert, \
            weight_out_stride, weight_row_offset, groups, k_real, routed_input)
        switch (profile) {
            case kMoeProfileNint2Gs16: MFQ_MOE_MMA_PROFILE(2, 16, 4); break;
            case kMoeProfileNint3Gs24: MFQ_MOE_MMA_PROFILE(3, 24, 4); break;
            case kMoeProfileNint4Gs24: MFQ_MOE_MMA_PROFILE(4, 24, 4); break;
            case kMoeProfileNint5Gs28: MFQ_MOE_MMA_PROFILE(5, 28, 4); break;
            case kMoeProfileNint6Gs24: MFQ_MOE_MMA_PROFILE(6, 24, 4); break;
            case kMoeProfileNint8Gs48: MFQ_MOE_MMA_PROFILE(8, 48, 2); break;
            case kMoeProfileNint8Gs24: MFQ_MOE_MMA_PROFILE(8, 24, 4); break;
        }
#undef MFQ_MOE_MMA_PROFILE
    }
}

__global__ void moe_weighted_reduce_kernel(
        const __half * __restrict__ pair_output,
        const float * __restrict__ weights,
        __half * __restrict__ output,
        int tokens,
        int routes,
        int width) {
    const size_t total = static_cast<size_t>(tokens) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            index < total;
            index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int token = static_cast<int>(index / width);
        const int column = static_cast<int>(index - static_cast<size_t>(token) * width);
        float value = 0.0f;
        for (int route = 0; route < routes; ++route) {
            const size_t pair = static_cast<size_t>(token) * routes + route;
            value += weights[pair] * __half2float(pair_output[pair * width + column]);
        }
        output[index] = __float2half(value);
    }
}

template <bool GELU>
__global__ void moe_glu_split_kernel(
        const __half * __restrict__ gate_up,
        __half * __restrict__ output,
        int rows,
        int width) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(rows) * width;
    if (index >= total) {
        return;
    }
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index - static_cast<int64_t>(row) * width);
    const int64_t base = static_cast<int64_t>(row) * (2 * width);
    const float gate = __half2float(gate_up[base + column]);
    const float up = __half2float(gate_up[base + width + column]);
    output[index] = __float2half_rn(mfq_glu<GELU>(gate, up));
}

__global__ void moe_add_shared_gate_kernel(
        const __half * __restrict__ routed,
        const __half * __restrict__ shared,
        const float * __restrict__ gate_logits,
        __half * __restrict__ output,
        int rows,
        int width) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(rows) * width;
    if (index >= total) {
        return;
    }
    const int row = static_cast<int>(index / width);
    const float gate = 1.0f / (1.0f + expf(-gate_logits[row]));
    const float value = __half2float(routed[index]) + gate * __half2float(shared[index]);
    output[index] = __float2half_rn(value);
}

__global__ void moe_weighted_reduce_shared_gate_kernel(
        const __half * __restrict__ pair_output,
        const float * __restrict__ weights,
        const __half * __restrict__ shared,
        const float * __restrict__ gate_logits,
        __half * __restrict__ output,
        int tokens,
        int routes,
        int width) {
    const size_t total = static_cast<size_t>(tokens) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            index < total;
            index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int token = static_cast<int>(index / width);
        const int column = static_cast<int>(index - static_cast<size_t>(token) * width);
        float routed = 0.0f;
        for (int route = 0; route < routes; ++route) {
            const size_t pair = static_cast<size_t>(token) * routes + route;
            routed += weights[pair] * __half2float(pair_output[pair * width + column]);
        }
        routed = __half2float(__float2half_rn(routed));
        const float gate = 1.0f / (1.0f + expf(-gate_logits[token]));
        output[index] = __float2half_rn(
            routed + gate * __half2float(shared[index]));
    }
}

void check_same_device(const torch::Tensor & reference, const torch::Tensor & tensor, const char * name) {
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on ", reference.device());
}

void check_nint_weight(
        const torch::Tensor & q_packed,
        const torch::Tensor & sub_scale,
        const torch::Tensor & sub_min,
        const torch::Tensor & neuron_scale,
        const torch::Tensor & neuron_min,
        int local_experts,
        int out_per_expert,
        int gs,
        int bits) {
    TORCH_CHECK(q_packed.is_cuda() && q_packed.is_contiguous() && q_packed.scalar_type() == torch::kUInt8,
        "q_packed must be contiguous CUDA uint8");
    TORCH_CHECK(q_packed.dim() == 3, "q_packed must have [experts*out, groups, qbytes] shape");
    const int rows = local_experts * out_per_expert;
    const int groups = static_cast<int>(q_packed.size(1));
    const int qbytes = (gs * bits + 7) / 8;
    TORCH_CHECK(q_packed.size(0) == rows && q_packed.size(2) == qbytes,
        "q_packed shape does not match experts/out/gs/bits");
    TORCH_CHECK(sub_scale.is_cuda() && sub_scale.is_contiguous() && sub_scale.scalar_type() == torch::kUInt8 &&
        sub_scale.size(0) == rows && sub_scale.size(1) == groups,
        "sub_scale must have contiguous CUDA uint8 [experts*out, groups] shape");
    TORCH_CHECK(sub_min.is_cuda() && sub_min.is_contiguous() && sub_min.scalar_type() == torch::kUInt8 &&
        sub_min.sizes() == sub_scale.sizes(), "sub_min shape mismatch");
    TORCH_CHECK(neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == torch::kFloat32 && neuron_scale.numel() == rows,
        "neuron_scale shape mismatch");
    TORCH_CHECK(neuron_min.is_cuda() && neuron_min.is_contiguous() &&
        neuron_min.scalar_type() == torch::kFloat32 && neuron_min.numel() == rows,
        "neuron_min shape mismatch");
    check_same_device(q_packed, sub_scale, "sub_scale");
    check_same_device(q_packed, sub_min, "sub_min");
    check_same_device(q_packed, neuron_scale, "neuron_scale");
    check_same_device(q_packed, neuron_min, "neuron_min");
}

void build_expert_map(
        const torch::Tensor & ids,
        int experts,
        int tile_m,
        torch::Tensor & counts,
        torch::Tensor & cursors,
        torch::Tensor & ids_dst,
        torch::Tensor & expert_bounds,
        torch::Tensor & tile_bounds,
        torch::Tensor & tile_experts,
        cudaStream_t stream) {
    const int pairs = static_cast<int>(ids.numel());
    TORCH_CHECK(counts.is_cuda() && counts.is_contiguous() && counts.scalar_type() == torch::kInt32 &&
        counts.numel() >= experts, "counts workspace is too small");
    TORCH_CHECK(cursors.is_cuda() && cursors.is_contiguous() && cursors.scalar_type() == torch::kInt32 &&
        cursors.numel() >= experts, "cursors workspace is too small");
    TORCH_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() && ids_dst.scalar_type() == torch::kInt32 &&
        ids_dst.numel() >= pairs, "ids_dst workspace is too small");
    TORCH_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
        expert_bounds.scalar_type() == torch::kInt32 && expert_bounds.numel() >= experts + 1,
        "expert_bounds workspace is too small");
    TORCH_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
        tile_bounds.scalar_type() == torch::kInt32 && tile_bounds.numel() >= experts + 1,
        "tile_bounds workspace is too small");
    TORCH_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
        tile_experts.scalar_type() == torch::kInt32 && tile_experts.numel() >= pairs,
        "tile_experts workspace is too small");
    check_same_device(ids, counts, "counts");
    check_same_device(ids, cursors, "cursors");
    check_same_device(ids, ids_dst, "ids_dst");
    check_same_device(ids, expert_bounds, "expert_bounds");
    check_same_device(ids, tile_bounds, "tile_bounds");
    check_same_device(ids, tile_experts, "tile_experts");

    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int32_t>(), 0, experts * sizeof(int32_t), stream));
    const int block = 256;
    const int grid = std::min((pairs + block - 1) / block, 65535);
    count_experts_kernel<<<grid, block, 0, stream>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), pairs, experts);
    scan_expert_counts_kernel<<<1, 1, 0, stream>>>(
        counts.data_ptr<int32_t>(), cursors.data_ptr<int32_t>(),
        expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(), experts, tile_m);
    fill_tile_experts_kernel<<<(experts + 255) / 256, 256, 0, stream>>>(
        tile_bounds.data_ptr<int32_t>(), tile_experts.data_ptr<int32_t>(), experts);
    scatter_routes_kernel<<<grid, block, 0, stream>>>(
        ids.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(), cursors.data_ptr<int32_t>(),
        ids_dst.data_ptr<int32_t>(), pairs, experts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int GS>
void launch_quantize(
        const torch::Tensor & x,
        torch::Tensor & qx,
        torch::Tensor & xscale,
        int rows,
        int k_real,
        int k_pad,
        int groups,
        cudaStream_t stream) {
    constexpr int block = ((GS + 31) / 32) * 32;
    quantize_moe_input_kernel<GS, block><<<dim3(rows, groups), block, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), rows, k_real, k_pad);
}

template <int GS, bool GELU, bool CLAMPED = false>
void launch_quantize_glu(
        const torch::Tensor & gate_up,
        torch::Tensor & qx,
        torch::Tensor & xscale,
        int rows,
        int k_real,
        int k_pad,
        int groups,
        cudaStream_t stream,
        float limit = 0.0f) {
    constexpr int block = ((GS + 31) / 32) * 32;
    quantize_moe_glu_input_kernel<GS, block, GELU, CLAMPED><<<
        dim3(rows, groups), block, 0, stream>>>(
        reinterpret_cast<const __half *>(gate_up.data_ptr<at::Half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
        rows, k_real, k_pad, limit);
}

template <int BITS, int GS>
void launch_grouped_matmul(
        const torch::Tensor & q_packed,
        const torch::Tensor & sub_scale,
        const torch::Tensor & sub_min,
        const torch::Tensor & neuron_scale,
        const torch::Tensor & neuron_min,
        const torch::Tensor & qx,
        const torch::Tensor & xscale,
        const torch::Tensor & ids,
        const torch::Tensor & expert_local,
        const torch::Tensor & ids_dst,
        const torch::Tensor & expert_bounds,
        const torch::Tensor & tile_bounds,
        const torch::Tensor & tile_experts,
        torch::Tensor & out,
        int tokens,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_pad,
        bool routed_input,
        cudaStream_t stream) {
    static const bool disable_token_warp = [] {
        const char * value = std::getenv("MFQ_DISABLE_MOE_TOKEN_WARP");
        return value != nullptr && std::atoi(value) != 0;
    }();
    static const int token_warps = [] {
        const char * value = std::getenv("MFQ_MOE_TOKEN_WARPS");
        if (value == nullptr) return 16;
        const int parsed = std::atoi(value);
        return parsed == 8 || parsed == 16 || parsed == 32 ? parsed : 16;
    }();
    if constexpr (GS == 24 || GS == 28) {
        if (current_moe_small_mmq() && tokens >= 16 && tokens <= 128) {
            const int fine_tiles = (tokens * routes + kRouteTile - 1) / kRouteTile + experts;
            const int ntiles = (out_per_expert + 63) / 64;
            nint_moe_group32_mmq_kernel<BITS, GS><<<
                fine_tiles * ntiles, dim3(32, 8), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(),
                tile_bounds.data_ptr<int32_t>(), tile_experts.data_ptr<int32_t>(),
                expert_local.data_ptr<int32_t>(), reinterpret_cast<__half *>(out.data_ptr<at::Half>()),
                routes, experts, out_per_expert, groups, k_pad, routed_input);
            return;
        }
    }
    if (!disable_token_warp && tokens > 8 && tokens <= 64) {
        const int row_blocks = (out_per_expert + 1) / 2;
        if (token_warps == 8) {
            nint_moe_mmvq_kernel<BITS, GS, 8><<<
                dim3(row_blocks, routes, (tokens + 7) / 8), dim3(32, 8), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
                reinterpret_cast<__half *>(out.data_ptr<at::Half>()), tokens, routes, experts,
                out_per_expert, groups, k_pad, routed_input);
        } else if (token_warps == 16) {
            nint_moe_mmvq_kernel<BITS, GS, 16><<<
                dim3(row_blocks, routes, (tokens + 15) / 16), dim3(32, 16), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
                reinterpret_cast<__half *>(out.data_ptr<at::Half>()), tokens, routes, experts,
                out_per_expert, groups, k_pad, routed_input);
        } else {
            nint_moe_mmvq_kernel<BITS, GS, 32><<<
                dim3(row_blocks, routes, (tokens + 31) / 32), dim3(32, 32), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
                reinterpret_cast<__half *>(out.data_ptr<at::Half>()), tokens, routes, experts,
                out_per_expert, groups, k_pad, routed_input);
        }
        return;
    }
    if (tokens <= 8) {
        const int row_blocks = (out_per_expert + 1) / 2;
        if (tokens == 1) {
            constexpr bool route_packed_profile =
                (BITS == 3 && GS == 24) || (BITS == 4 && GS == 24) || (BITS == 5 && GS == 28) ||
                (BITS == 6 && GS == 24) || (BITS == 8 && (GS == 24 || GS == 48));
            static const int ksplit_warps = [] {
                const char * value = std::getenv("MFQ_MOE_KSPLIT_WARPS");
                if (value == nullptr) return 0;
                const int parsed = std::atoi(value);
                return parsed == 2 || parsed == 4 ? parsed : 0;
            }();
            static const int ksplit_min_groups = [] {
                const char * value = std::getenv("MFQ_MOE_KSPLIT_MIN_GROUPS");
                return value == nullptr ? 48 : std::max(1, std::atoi(value));
            }();
            static const bool disable_route_packed = [] {
                const char * value = std::getenv("MFQ_DISABLE_MOE_ROUTE_PACKED");
                return value != nullptr && std::atoi(value) != 0;
            }();
            if constexpr (route_packed_profile) {
                if (groups >= ksplit_min_groups && ksplit_warps == 4) {
                    nint_moe_mmvq_ksplit_kernel<BITS, GS, 4><<<
                        dim3(row_blocks, routes, 1), dim3(32, 4), 0, stream>>>(
                        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                        neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                        xscale.data_ptr<float>(), ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
                        reinterpret_cast<__half *>(out.data_ptr<at::Half>()), routes, experts,
                        out_per_expert, groups, k_pad, routed_input);
                    return;
                }
                if (groups >= ksplit_min_groups && ksplit_warps == 2) {
                    nint_moe_mmvq_ksplit_kernel<BITS, GS, 2><<<
                        dim3(row_blocks, routes, 1), dim3(32, 2), 0, stream>>>(
                        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                        neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                        xscale.data_ptr<float>(), ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
                        reinterpret_cast<__half *>(out.data_ptr<at::Half>()), routes, experts,
                        out_per_expert, groups, k_pad, routed_input);
                    return;
                }
                if (!disable_route_packed && routes <= 8) {
                    nint_moe_mmvq_kernel<BITS, GS, 8, true><<<
                        dim3(row_blocks, 1, 1), dim3(32, 8), 0, stream>>>(
                        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                        neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                        xscale.data_ptr<float>(), ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
                        reinterpret_cast<__half *>(out.data_ptr<at::Half>()), tokens, routes, experts,
                        out_per_expert, groups, k_pad, routed_input);
                    return;
                }
            }
            nint_moe_mmvq_kernel<BITS, GS, 1><<<dim3(row_blocks, routes, 1), dim3(32, 1), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
                reinterpret_cast<__half *>(out.data_ptr<at::Half>()), tokens, routes, experts,
                out_per_expert, groups, k_pad, routed_input);
        } else if (tokens <= 2) {
            nint_moe_mmvq_kernel<BITS, GS, 2><<<dim3(row_blocks, routes, 1), dim3(32, 2), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
                reinterpret_cast<__half *>(out.data_ptr<at::Half>()), tokens, routes, experts,
                out_per_expert, groups, k_pad, routed_input);
        } else if (tokens <= 4) {
            nint_moe_mmvq_kernel<BITS, GS, 4><<<dim3(row_blocks, routes, 1), dim3(32, 4), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
                reinterpret_cast<__half *>(out.data_ptr<at::Half>()), tokens, routes, experts,
                out_per_expert, groups, k_pad, routed_input);
        } else {
            nint_moe_mmvq_kernel<BITS, GS, 8><<<dim3(row_blocks, routes, 1), dim3(32, 8), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
                reinterpret_cast<__half *>(out.data_ptr<at::Half>()), tokens, routes, experts,
                out_per_expert, groups, k_pad, routed_input);
        }
        return;
    }

    const int pairs = tokens * routes;
    const int row_tiles = (out_per_expert + kRowsPerBlock - 1) / kRowsPerBlock;
    static const bool disable_persistent = [] {
        const char * value = std::getenv("MFQ_DISABLE_MOE_PERSISTENT");
        return value != nullptr && std::atoi(value) != 0;
    }();
    if (!disable_persistent && tokens <= 32) {
        const int64_t max_tasks = static_cast<int64_t>(pairs) * row_tiles;
        constexpr int persistent_blocks = 4096;
        const int blocks = static_cast<int>(std::min<int64_t>(max_tasks, persistent_blocks));
        nint_moe_grouped_tile_persistent_kernel<BITS, GS, kRouteTile><<<
            blocks, dim3(32, 4), 0, stream>>>(
            q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
            xscale.data_ptr<float>(), ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(),
            tile_bounds.data_ptr<int32_t>(), tile_experts.data_ptr<int32_t>(),
            expert_local.data_ptr<int32_t>(), reinterpret_cast<__half *>(out.data_ptr<at::Half>()),
            routes, experts, out_per_expert, groups, routed_input);
        return;
    }

    const int max_tiles = (pairs + kRouteTile - 1) / kRouteTile + experts;
    const int64_t blocks = static_cast<int64_t>(row_tiles) * max_tiles;
    TORCH_CHECK(blocks <= INT_MAX, "grouped NINT launch grid is too large");
    nint_moe_grouped_tile_kernel<BITS, GS, kRouteTile><<<static_cast<int>(blocks), dim3(32, 4), 0, stream>>>(
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
        neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(),
        tile_bounds.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
        reinterpret_cast<__half *>(out.data_ptr<at::Half>()),
        routes, experts, out_per_expert, groups, k_pad, max_tiles, routed_input);
}

void launch_nint8_zero_moe_mma(
        const torch::Tensor & q,
        const torch::Tensor & scale,
        const torch::Tensor & expert_local,
        const torch::Tensor & x,
        const torch::Tensor & ids_dst,
        const torch::Tensor & expert_bounds,
        const torch::Tensor & tile_bounds,
        const torch::Tensor & tile_experts,
        torch::Tensor & out,
        int tokens,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_real,
        bool routed_input,
        cudaStream_t stream) {
    const int ntiles_n =
        (out_per_expert + kMoeMmaBn - 1) / kMoeMmaBn;
    const int pairs = tokens * routes;
    const int64_t max_fine_tiles =
        (pairs + kRouteTile - 1) / kRouteTile + experts;
    const int64_t max_tasks = max_fine_tiles * ntiles_n;
    static const int block_cap = [] {
        const char * value = std::getenv("MFQ_MOE_PREFILL_MMA_BLOCKS");
        return value == nullptr ? 4096 : std::max(1, std::atoi(value));
    }();
    static const int forced_bm = [] {
        const char * value = std::getenv("MFQ_MOE_PREFILL_MMA_BM");
        if (value == nullptr) return 0;
        const int parsed = std::atoi(value);
        return parsed == 16 || parsed == 32 || parsed == 64 ? parsed : 0;
    }();
    const int blocks = static_cast<int>(
        std::max<int64_t>(1, std::min<int64_t>(block_cap, max_tasks)));
    const dim3 threads(32, 8);
    const int bm = forced_bm != 0 ? forced_bm :
        (tokens <= 128 ? 64 : (tokens <= 512 ? 32 : 64));
#define MFQ_Q8_ZERO_MOE_MMA(BM_VALUE) \
    nint8_zero_moe_mma_kernel<BM_VALUE><<<blocks, threads, 0, stream>>>( \
        q.data_ptr<uint8_t>(), \
        reinterpret_cast<const __half *>(scale.data_ptr<at::Half>()), \
        expert_local.data_ptr<int32_t>(), \
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), \
        ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(), \
        tile_bounds.data_ptr<int32_t>(), tile_experts.data_ptr<int32_t>(), \
        reinterpret_cast<__half *>(out.data_ptr<at::Half>()), routes, experts, \
        out_per_expert, groups, k_real, routed_input)
    if (bm == 16) {
        MFQ_Q8_ZERO_MOE_MMA(16);
    } else if (bm == 32) {
        MFQ_Q8_ZERO_MOE_MMA(32);
    } else {
        MFQ_Q8_ZERO_MOE_MMA(64);
    }
#undef MFQ_Q8_ZERO_MOE_MMA
}

void launch_nint8_zero_grouped_matmul(
        const torch::Tensor & q,
        const torch::Tensor & scale,
        const torch::Tensor & qx,
        const torch::Tensor & xscale,
        const torch::Tensor & ids,
        const torch::Tensor & expert_local,
        const torch::Tensor & ids_dst,
        const torch::Tensor & expert_bounds,
        const torch::Tensor & tile_bounds,
        const torch::Tensor & tile_experts,
        torch::Tensor & out,
        int tokens,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_pad,
        bool routed_input,
        cudaStream_t stream) {
    static const bool disable_token_warp = [] {
        const char * value = std::getenv("MFQ_DISABLE_MOE_TOKEN_WARP");
        return value != nullptr && std::atoi(value) != 0;
    }();
    static const int token_warps = [] {
        const char * value = std::getenv("MFQ_MOE_TOKEN_WARPS");
        if (value == nullptr) return 16;
        const int parsed = std::atoi(value);
        return parsed == 8 || parsed == 16 || parsed == 32 ? parsed : 16;
    }();
    const auto launch_direct = [&](int items) {
        const int row_blocks = (out_per_expert + 1) / 2;
#define MFQ_Q8_ZERO_MOE_DIRECT(ITEMS) \
        nint8_zero_moe_mmvq_kernel<ITEMS><<< \
            dim3(row_blocks, routes, (tokens + ITEMS - 1) / ITEMS), \
            dim3(32, ITEMS), 0, stream>>>( \
                q.data_ptr<uint8_t>(), \
                reinterpret_cast<const __half *>(scale.data_ptr<at::Half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), \
                ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(), \
                reinterpret_cast<__half *>(out.data_ptr<at::Half>()), \
                tokens, routes, experts, out_per_expert, groups, k_pad, \
                routed_input)
        switch (items) {
            case 1: MFQ_Q8_ZERO_MOE_DIRECT(1); break;
            case 2: MFQ_Q8_ZERO_MOE_DIRECT(2); break;
            case 4: MFQ_Q8_ZERO_MOE_DIRECT(4); break;
            case 8: MFQ_Q8_ZERO_MOE_DIRECT(8); break;
            case 16: MFQ_Q8_ZERO_MOE_DIRECT(16); break;
            case 32: MFQ_Q8_ZERO_MOE_DIRECT(32); break;
        }
#undef MFQ_Q8_ZERO_MOE_DIRECT
    };
    if (!disable_token_warp && tokens > 8 && tokens <= 64) {
        launch_direct(token_warps);
        return;
    }
    if (tokens <= 8) {
        launch_direct(tokens == 1 ? 1 : tokens <= 2 ? 2 : tokens <= 4 ? 4 : 8);
        return;
    }

    const int pairs = tokens * routes;
    const int row_tiles =
        (out_per_expert + kRowsPerBlock - 1) / kRowsPerBlock;
    static const bool disable_persistent = [] {
        const char * value = std::getenv("MFQ_DISABLE_MOE_PERSISTENT");
        return value != nullptr && std::atoi(value) != 0;
    }();
    if (!disable_persistent && tokens <= 32) {
        const int64_t max_tasks =
            static_cast<int64_t>(pairs) * row_tiles;
        constexpr int persistent_blocks = 4096;
        const int blocks = static_cast<int>(
            std::min<int64_t>(max_tasks, persistent_blocks));
        nint8_zero_moe_grouped_tile_persistent_kernel<kRouteTile><<<
            blocks, dim3(32, 4), 0, stream>>>(
                q.data_ptr<uint8_t>(),
                reinterpret_cast<const __half *>(scale.data_ptr<at::Half>()),
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                ids_dst.data_ptr<int32_t>(),
                expert_bounds.data_ptr<int32_t>(),
                tile_bounds.data_ptr<int32_t>(),
                tile_experts.data_ptr<int32_t>(),
                expert_local.data_ptr<int32_t>(),
                reinterpret_cast<__half *>(out.data_ptr<at::Half>()),
                routes, experts, out_per_expert, groups, routed_input);
        return;
    }

    const int max_tiles =
        (pairs + kRouteTile - 1) / kRouteTile + experts;
    const int64_t blocks =
        static_cast<int64_t>(row_tiles) * max_tiles;
    TORCH_CHECK(
        blocks <= INT_MAX, "grouped NINT8-0 launch grid is too large");
    nint8_zero_moe_grouped_tile_kernel<kRouteTile><<<
        static_cast<int>(blocks), dim3(32, 4), 0, stream>>>(
            q.data_ptr<uint8_t>(),
            reinterpret_cast<const __half *>(scale.data_ptr<at::Half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            ids_dst.data_ptr<int32_t>(),
            expert_bounds.data_ptr<int32_t>(),
            tile_bounds.data_ptr<int32_t>(),
            expert_local.data_ptr<int32_t>(),
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()),
            routes, experts, out_per_expert, groups, max_tiles,
            routed_input);
}

} // namespace

void mfq::moe_cache_scatter_cuda(
        const std::uint8_t * staging,
        std::int64_t descriptor_offset,
        int transfer_count,
        cudaStream_t stream) {
    TORCH_CHECK(staging != nullptr, "MoE cache staging pointer is null");
    TORCH_CHECK(descriptor_offset >= 0,
        "MoE cache descriptor offset must be non-negative");
    TORCH_CHECK(transfer_count > 0,
        "MoE cache scatter requires at least one transfer");
    moe_cache_scatter_kernel<<<transfer_count, 256, 0, stream>>>(
        staging, descriptor_offset, transfer_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mfq::moe_cache_mapped_gather_cuda(
        const MoeCacheMappedCopyDescriptor * descriptors,
        int transfer_count,
        int blocks_per_transfer,
        cudaStream_t stream) {
    TORCH_CHECK(descriptors != nullptr,
        "MoE cache mapped-copy descriptor pointer is null");
    TORCH_CHECK(transfer_count > 0,
        "MoE cache mapped gather requires at least one transfer");
    TORCH_CHECK(blocks_per_transfer >= 1 && blocks_per_transfer <= 128,
        "MoE cache mapped gather blocks must be in [1, 128]");
    const dim3 grid(
        static_cast<unsigned int>(blocks_per_transfer),
        static_cast<unsigned int>(transfer_count));
    moe_cache_mapped_gather_kernel<<<grid, 256, 0, stream>>>(
        descriptors, transfer_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nint_moe_set_small_mmq_cuda(int64_t mode) {
    TORCH_CHECK(mode >= -1 && mode <= 1, "small-M MoE MMQ mode must be in [-1,1]");
    g_moe_small_mmq_override = static_cast<int>(mode);
}

std::vector<torch::Tensor> moe_topk_cuda(
        torch::Tensor logits,
        int64_t top_k,
        bool use_sigmoid,
        bool use_sqrt_softplus,
        bool normalize,
        bool delayed_softmax,
        c10::optional<torch::Tensor> bias,
        double norm_floor,
        double scale) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous() && logits.dim() == 2,
        "logits must be contiguous CUDA [tokens, experts]");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32 || logits.scalar_type() == torch::kFloat16,
        "logits must be float16 or float32");
    const int rows = static_cast<int>(logits.size(0));
    const int experts = static_cast<int>(logits.size(1));
    TORCH_CHECK(rows > 0 && experts > 0, "logits dimensions must be nonzero");
    TORCH_CHECK(top_k >= 1 && top_k <= 16 && top_k <= experts,
        "top_k must be in [1, min(16, experts)]");
    TORCH_CHECK(!(normalize && delayed_softmax),
        "selected-weight normalization and delayed softmax are mutually exclusive");
    TORCH_CHECK(!(use_sigmoid && delayed_softmax),
        "sigmoid routing and delayed softmax are mutually exclusive");
    TORCH_CHECK(!(use_sqrt_softplus && delayed_softmax),
        "sqrt-softplus routing and delayed softmax are mutually exclusive");
    TORCH_CHECK(!(use_sigmoid && use_sqrt_softplus),
        "sigmoid and sqrt-softplus routing are mutually exclusive");
    const float * bias_ptr = nullptr;
    if (bias.has_value()) {
        auto & value = bias.value();
        TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.scalar_type() == torch::kFloat32 &&
            value.numel() == experts, "bias must be contiguous CUDA float32 [experts]");
        check_same_device(logits, value, "bias");
        bias_ptr = value.data_ptr<float>();
    }

    auto ids = torch::empty({rows, top_k}, logits.options().dtype(torch::kInt32));
    auto weights = torch::empty({rows, top_k}, logits.options().dtype(torch::kFloat32));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    static const bool disable_topk_sort = [] {
        const char * value = std::getenv("MFQ_DISABLE_MOE_TOPK_SORT");
        return value != nullptr && std::atoi(value) != 0;
    }();
    if (!disable_topk_sort && rows == 1 && experts == 128 && top_k == 8 &&
            logits.scalar_type() == torch::kFloat32 && bias_ptr == nullptr &&
            !use_sigmoid && !normalize && delayed_softmax && scale == 1.0) {
        moe_topk_1x128_8_kernel<<<1, 32, 0, stream>>>(
            logits.data_ptr<float>(), ids.data_ptr<int32_t>(), weights.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {ids, weights};
    }
    if (!disable_topk_sort && rows == 1 && experts == 256 && top_k == 8 &&
            logits.scalar_type() == torch::kFloat32 && bias_ptr == nullptr &&
            !use_sigmoid && !use_sqrt_softplus && !normalize && delayed_softmax && scale == 1.0) {
        moe_topk_1x256_8_kernel<<<1, 256, 0, stream>>>(
            logits.data_ptr<float>(), ids.data_ptr<int32_t>(), weights.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {ids, weights};
    }
    if (!disable_topk_sort && rows == 1 && experts == 256 && top_k == 6 &&
            logits.scalar_type() == torch::kFloat32 &&
            !use_sigmoid && use_sqrt_softplus && normalize && !delayed_softmax) {
        moe_topk_1x256_6_sqrtsoftplus_kernel<<<1, 256, 0, stream>>>(
            logits.data_ptr<float>(), bias_ptr, ids.data_ptr<int32_t>(), weights.data_ptr<float>(),
            static_cast<float>(norm_floor), static_cast<float>(scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {ids, weights};
    }
    const dim3 block(32, 4);
    const int grid = (rows + 3) / 4;
    if (logits.scalar_type() == torch::kFloat16) {
        moe_topk_kernel<at::Half><<<grid, block, 0, stream>>>(
            logits.data_ptr<at::Half>(), bias_ptr, ids.data_ptr<int32_t>(), weights.data_ptr<float>(),
            rows, experts, static_cast<int>(top_k), use_sigmoid, use_sqrt_softplus,
            normalize, delayed_softmax,
            static_cast<float>(norm_floor), static_cast<float>(scale));
    } else {
        moe_topk_kernel<float><<<grid, block, 0, stream>>>(
            logits.data_ptr<float>(), bias_ptr, ids.data_ptr<int32_t>(), weights.data_ptr<float>(),
            rows, experts, static_cast<int>(top_k), use_sigmoid, use_sqrt_softplus,
            normalize, delayed_softmax,
            static_cast<float>(norm_floor), static_cast<float>(scale));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {ids, weights};
}

torch::Tensor moe_sqrtsoftplus_weights_cuda(
        torch::Tensor logits,
        torch::Tensor ids,
        double norm_floor,
        double scale) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous() && logits.dim() == 2,
        "logits must be contiguous CUDA [tokens, experts]");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32 || logits.scalar_type() == torch::kFloat16,
        "logits must be float16 or float32");
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.dim() == 2 &&
        ids.scalar_type() == torch::kInt32, "ids must be contiguous CUDA int32 [tokens, top_k]");
    check_same_device(logits, ids, "ids");
    TORCH_CHECK(ids.size(0) == logits.size(0), "ids and logits must have the same token count");
    const int rows = static_cast<int>(logits.size(0));
    const int experts = static_cast<int>(logits.size(1));
    const int top_k = static_cast<int>(ids.size(1));
    TORCH_CHECK(rows > 0 && experts > 0, "logits dimensions must be nonzero");
    TORCH_CHECK(top_k >= 1 && top_k <= 16 && top_k <= experts,
        "top_k must be in [1, min(16, experts)]");

    auto weights = torch::empty(ids.sizes(), logits.options().dtype(torch::kFloat32));
    const dim3 block(32, 4);
    const int grid = (rows + 3) / 4;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (logits.scalar_type() == torch::kFloat16) {
        moe_sqrtsoftplus_weights_kernel<at::Half><<<grid, block, 0, stream>>>(
            logits.data_ptr<at::Half>(), ids.data_ptr<int32_t>(), weights.data_ptr<float>(),
            rows, experts, top_k, static_cast<float>(norm_floor), static_cast<float>(scale));
    } else {
        moe_sqrtsoftplus_weights_kernel<float><<<grid, block, 0, stream>>>(
            logits.data_ptr<float>(), ids.data_ptr<int32_t>(), weights.data_ptr<float>(),
            rows, experts, top_k, static_cast<float>(norm_floor), static_cast<float>(scale));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return weights;
}

std::vector<torch::Tensor> moe_build_expert_map_cuda(
        torch::Tensor ids,
        int64_t n_experts,
        int64_t tile_m) {
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32 && ids.dim() == 2,
        "ids must be contiguous CUDA int32 [tokens, routes]");
    TORCH_CHECK(n_experts > 0 && n_experts <= 4096, "n_experts must be in [1, 4096]");
    TORCH_CHECK(tile_m > 0 && tile_m <= 1024, "tile_m must be in [1, 1024]");
    const int experts = static_cast<int>(n_experts);
    const int pairs = static_cast<int>(ids.numel());
    auto options = ids.options();
    auto counts = torch::empty({experts}, options);
    auto cursors = torch::empty({experts}, options);
    auto ids_dst = torch::empty({pairs}, options);
    auto expert_bounds = torch::empty({experts + 1}, options);
    auto tile_bounds = torch::empty({experts + 1}, options);
    auto tile_experts = torch::empty({pairs}, options);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    build_expert_map(ids, experts, static_cast<int>(tile_m), counts, cursors, ids_dst,
        expert_bounds, tile_bounds, tile_experts, stream);
    return {ids_dst, expert_bounds, tile_bounds, tile_experts, counts};
}

void nint_moe_quantize_input_ws_cuda(
        torch::Tensor x,
        int64_t gs,
        torch::Tensor qx,
        torch::Tensor xscale) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat16 &&
        (x.dim() == 2 || x.dim() == 3),
        "x must be contiguous CUDA float16 [T,K] or [T,R,K]");
    TORCH_CHECK(gs == 16 || gs == 24 || gs == 28 || gs == 48,
        "heterogeneous NINT MoE quantization supports gs in {16,24,28,48}");
    const int k_real = static_cast<int>(x.size(-1));
    const int rows = static_cast<int>(x.numel() / k_real);
    TORCH_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == torch::kInt8 &&
        qx.dim() == 2 && qx.size(0) >= rows,
        "qx must be contiguous CUDA int8 [rows,K_pad]");
    TORCH_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
        xscale.scalar_type() == torch::kFloat32 && xscale.dim() == 2 && xscale.size(0) >= rows,
        "xscale must be contiguous CUDA float32 [rows,groups]");
    check_same_device(x, qx, "qx");
    check_same_device(x, xscale, "xscale");
    const int groups = static_cast<int>(xscale.size(1));
    const int k_pad = groups * static_cast<int>(gs);
    TORCH_CHECK(qx.size(1) >= k_pad && k_real <= k_pad,
        "heterogeneous NINT MoE quantization workspace is too small");
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto flat = x.reshape({rows, k_real});
    switch (static_cast<int>(gs)) {
        case 16: launch_quantize<16>(flat, qx, xscale, rows, k_real, k_pad, groups, stream); break;
        case 24: launch_quantize<24>(flat, qx, xscale, rows, k_real, k_pad, groups, stream); break;
        case 28: launch_quantize<28>(flat, qx, xscale, rows, k_real, k_pad, groups, stream); break;
        case 48: launch_quantize<48>(flat, qx, xscale, rows, k_real, k_pad, groups, stream); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nint_moe_quantize_24_28_ws_cuda(
        torch::Tensor x,
        torch::Tensor qx24,
        torch::Tensor xscale24,
        torch::Tensor qx28,
        torch::Tensor xscale28) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat16 &&
        (x.dim() == 2 || x.dim() == 3),
        "x must be contiguous CUDA float16 [T,K] or [T,R,K]");
    const int k_real = static_cast<int>(x.size(-1));
    const int rows = static_cast<int>(x.numel() / k_real);
    TORCH_CHECK(qx24.is_cuda() && qx24.is_contiguous() &&
        qx24.scalar_type() == torch::kInt8 && qx24.dim() == 2 && qx24.size(0) >= rows,
        "qx24 must be contiguous CUDA int8 [rows,k_pad24]");
    TORCH_CHECK(xscale24.is_cuda() && xscale24.is_contiguous() &&
        xscale24.scalar_type() == torch::kFloat32 && xscale24.dim() == 2 &&
        xscale24.size(0) >= rows,
        "xscale24 must be contiguous CUDA float32 [rows,groups24]");
    TORCH_CHECK(qx28.is_cuda() && qx28.is_contiguous() &&
        qx28.scalar_type() == torch::kInt8 && qx28.dim() == 2 && qx28.size(0) >= rows,
        "qx28 must be contiguous CUDA int8 [rows,k_pad28]");
    TORCH_CHECK(xscale28.is_cuda() && xscale28.is_contiguous() &&
        xscale28.scalar_type() == torch::kFloat32 && xscale28.dim() == 2 &&
        xscale28.size(0) >= rows,
        "xscale28 must be contiguous CUDA float32 [rows,groups28]");
    const int groups24 = static_cast<int>(xscale24.size(1));
    const int groups28 = static_cast<int>(xscale28.size(1));
    TORCH_CHECK(qx24.size(1) >= groups24 * 24 && k_real <= groups24 * 24,
        "gs24 activation workspace is too small");
    TORCH_CHECK(qx28.size(1) >= groups28 * 28 && k_real <= groups28 * 28,
        "gs28 activation workspace is too small");
    check_same_device(x, qx24, "qx24");
    check_same_device(x, xscale24, "xscale24");
    check_same_device(x, qx28, "qx28");
    check_same_device(x, xscale28, "xscale28");
    quantize_moe_input_24_28_kernel<<<
        dim3(rows, groups24 + groups28), 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
        qx24.data_ptr<int8_t>(), xscale24.data_ptr<float>(),
        qx28.data_ptr<int8_t>(), xscale28.data_ptr<float>(),
        k_real, groups24, groups28);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nint_moe_quantize_swiglu_input_ws_cuda(
        torch::Tensor gate_up,
        int64_t gs,
        torch::Tensor qx,
        torch::Tensor xscale) {
    TORCH_CHECK(gate_up.is_cuda() && gate_up.is_contiguous() &&
        gate_up.scalar_type() == torch::kFloat16 && gate_up.dim() == 3 &&
        gate_up.size(2) > 0 && gate_up.size(2) % 2 == 0,
        "gate_up must be contiguous CUDA float16 [T,R,2*K]");
    TORCH_CHECK(gs == 16 || gs == 24 || gs == 28 || gs == 48,
        "heterogeneous NINT MoE SwiGLU quantization supports gs in {16,24,28,48}");
    const int k_real = static_cast<int>(gate_up.size(2) / 2);
    const int rows = static_cast<int>(gate_up.size(0) * gate_up.size(1));
    TORCH_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == torch::kInt8 &&
        qx.dim() == 2 && qx.size(0) >= rows,
        "qx must be contiguous CUDA int8 [rows,k_pad]");
    TORCH_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
        xscale.scalar_type() == torch::kFloat32 && xscale.dim() == 2 && xscale.size(0) >= rows,
        "xscale must be contiguous CUDA float32 [rows,groups]");
    check_same_device(gate_up, qx, "qx");
    check_same_device(gate_up, xscale, "xscale");
    const int groups = static_cast<int>(xscale.size(1));
    const int k_pad = groups * static_cast<int>(gs);
    TORCH_CHECK(qx.size(1) >= k_pad && k_real <= k_pad,
        "heterogeneous NINT MoE SwiGLU quantization workspace is too small");
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (static_cast<int>(gs)) {
        case 16: launch_quantize_glu<16, false>(gate_up, qx, xscale, rows, k_real, k_pad, groups, stream); break;
        case 24: launch_quantize_glu<24, false>(gate_up, qx, xscale, rows, k_real, k_pad, groups, stream); break;
        case 28: launch_quantize_glu<28, false>(gate_up, qx, xscale, rows, k_real, k_pad, groups, stream); break;
        case 48: launch_quantize_glu<48, false>(gate_up, qx, xscale, rows, k_real, k_pad, groups, stream); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nint_moe_quantize_swiglu_clamped_input_ws_cuda(
        torch::Tensor gate_up,
        int64_t gs,
        double limit,
        torch::Tensor qx,
        torch::Tensor xscale) {
    TORCH_CHECK(gate_up.is_cuda() && gate_up.is_contiguous() &&
        gate_up.scalar_type() == torch::kFloat16 && gate_up.dim() == 3 &&
        gate_up.size(2) > 0 && gate_up.size(2) % 2 == 0,
        "gate_up must be contiguous CUDA float16 [T,R,2*K]");
    TORCH_CHECK(gs == 16 || gs == 24 || gs == 28 || gs == 48,
        "clamped SwiGLU MoE quantization supports gs in {16,24,28,48}");
    TORCH_CHECK(std::isfinite(limit) && limit > 0.0,
        "clamped SwiGLU limit must be finite and positive");
    const int k_real = static_cast<int>(gate_up.size(2) / 2);
    const int rows = static_cast<int>(gate_up.size(0) * gate_up.size(1));
    TORCH_CHECK(qx.is_cuda() && qx.is_contiguous() &&
        qx.scalar_type() == torch::kInt8 && qx.dim() == 2 &&
        qx.size(0) >= rows,
        "qx must be contiguous CUDA int8 [rows,k_pad]");
    TORCH_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
        xscale.scalar_type() == torch::kFloat32 && xscale.dim() == 2 &&
        xscale.size(0) >= rows,
        "xscale must be contiguous CUDA float32 [rows,groups]");
    check_same_device(gate_up, qx, "qx");
    check_same_device(gate_up, xscale, "xscale");
    const int groups = static_cast<int>(xscale.size(1));
    const int k_pad = groups * static_cast<int>(gs);
    TORCH_CHECK(qx.size(1) >= k_pad && k_real <= k_pad,
        "clamped SwiGLU MoE quantization workspace is too small");
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float limit_f = static_cast<float>(limit);
    switch (static_cast<int>(gs)) {
        case 16:
            launch_quantize_glu<16, false, true>(
                gate_up, qx, xscale, rows, k_real, k_pad, groups, stream, limit_f);
            break;
        case 24:
            launch_quantize_glu<24, false, true>(
                gate_up, qx, xscale, rows, k_real, k_pad, groups, stream, limit_f);
            break;
        case 28:
            launch_quantize_glu<28, false, true>(
                gate_up, qx, xscale, rows, k_real, k_pad, groups, stream, limit_f);
            break;
        case 48:
            launch_quantize_glu<48, false, true>(
                gate_up, qx, xscale, rows, k_real, k_pad, groups, stream, limit_f);
            break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nint_moe_quantize_geglu_input_ws_cuda(
        torch::Tensor gate_up,
        int64_t gs,
        torch::Tensor qx,
        torch::Tensor xscale) {
    TORCH_CHECK(gate_up.is_cuda() && gate_up.is_contiguous() &&
        gate_up.scalar_type() == torch::kFloat16 && gate_up.dim() == 3 &&
        gate_up.size(2) > 0 && gate_up.size(2) % 2 == 0,
        "gate_up must be contiguous CUDA float16 [T,R,2*K]");
    TORCH_CHECK(gs == 16 || gs == 24 || gs == 28 || gs == 48,
        "heterogeneous NINT MoE GeGLU quantization supports gs in {16,24,28,48}");
    const int k_real = static_cast<int>(gate_up.size(2) / 2);
    const int rows = static_cast<int>(gate_up.size(0) * gate_up.size(1));
    TORCH_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == torch::kInt8 &&
        qx.dim() == 2 && qx.size(0) >= rows,
        "qx must be contiguous CUDA int8 [rows,k_pad]");
    TORCH_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
        xscale.scalar_type() == torch::kFloat32 && xscale.dim() == 2 && xscale.size(0) >= rows,
        "xscale must be contiguous CUDA float32 [rows,groups]");
    check_same_device(gate_up, qx, "qx");
    check_same_device(gate_up, xscale, "xscale");
    const int groups = static_cast<int>(xscale.size(1));
    const int k_pad = groups * static_cast<int>(gs);
    TORCH_CHECK(qx.size(1) >= k_pad && k_real <= k_pad,
        "heterogeneous NINT MoE GeGLU quantization workspace is too small");
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (static_cast<int>(gs)) {
        case 16: launch_quantize_glu<16, true>(gate_up, qx, xscale, rows, k_real, k_pad, groups, stream); break;
        case 24: launch_quantize_glu<24, true>(gate_up, qx, xscale, rows, k_real, k_pad, groups, stream); break;
        case 28: launch_quantize_glu<28, true>(gate_up, qx, xscale, rows, k_real, k_pad, groups, stream); break;
        case 48: launch_quantize_glu<48, true>(gate_up, qx, xscale, rows, k_real, k_pad, groups, stream); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool GELU>
static void nint_moe_quantize_glu_24_28_ws_cuda_impl(
        torch::Tensor gate_up,
        torch::Tensor qx24,
        torch::Tensor xscale24,
        torch::Tensor qx28,
        torch::Tensor xscale28) {
    TORCH_CHECK(gate_up.is_cuda() && gate_up.is_contiguous() &&
        gate_up.scalar_type() == torch::kFloat16 && gate_up.dim() == 3 &&
        gate_up.size(2) > 0 && gate_up.size(2) % 2 == 0,
        "gate_up must be contiguous CUDA float16 [T,R,2*K]");
    const int k_real = static_cast<int>(gate_up.size(2) / 2);
    const int rows = static_cast<int>(gate_up.size(0) * gate_up.size(1));
    TORCH_CHECK(k_real <= 4096,
        "fused gs24/gs28 SwiGLU quantization supports K <= 4096");
    TORCH_CHECK(qx24.is_cuda() && qx24.is_contiguous() &&
        qx24.scalar_type() == torch::kInt8 && qx24.dim() == 2 && qx24.size(0) >= rows,
        "qx24 must be contiguous CUDA int8 [rows,k_pad24]");
    TORCH_CHECK(xscale24.is_cuda() && xscale24.is_contiguous() &&
        xscale24.scalar_type() == torch::kFloat32 && xscale24.dim() == 2 &&
        xscale24.size(0) >= rows,
        "xscale24 must be contiguous CUDA float32 [rows,groups24]");
    TORCH_CHECK(qx28.is_cuda() && qx28.is_contiguous() &&
        qx28.scalar_type() == torch::kInt8 && qx28.dim() == 2 && qx28.size(0) >= rows,
        "qx28 must be contiguous CUDA int8 [rows,k_pad28]");
    TORCH_CHECK(xscale28.is_cuda() && xscale28.is_contiguous() &&
        xscale28.scalar_type() == torch::kFloat32 && xscale28.dim() == 2 &&
        xscale28.size(0) >= rows,
        "xscale28 must be contiguous CUDA float32 [rows,groups28]");
    const int groups24 = static_cast<int>(xscale24.size(1));
    const int groups28 = static_cast<int>(xscale28.size(1));
    TORCH_CHECK(qx24.size(1) >= groups24 * 24 && k_real <= groups24 * 24,
        "gs24 activation workspace is too small");
    TORCH_CHECK(qx28.size(1) >= groups28 * 28 && k_real <= groups28 * 28,
        "gs28 activation workspace is too small");
    check_same_device(gate_up, qx24, "qx24");
    check_same_device(gate_up, xscale24, "xscale24");
    check_same_device(gate_up, qx28, "qx28");
    check_same_device(gate_up, xscale28, "xscale28");
    constexpr int block = 256;
    const size_t shared_bytes = static_cast<size_t>(k_real) * sizeof(__half);
    quantize_moe_glu_24_28_kernel<GELU><<<
        rows, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half *>(gate_up.data_ptr<at::Half>()),
        qx24.data_ptr<int8_t>(), xscale24.data_ptr<float>(),
        qx28.data_ptr<int8_t>(), xscale28.data_ptr<float>(),
        k_real, groups24, groups28);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nint_moe_quantize_swiglu_24_28_ws_cuda(
        torch::Tensor gate_up,
        torch::Tensor qx24,
        torch::Tensor xscale24,
        torch::Tensor qx28,
        torch::Tensor xscale28) {
    nint_moe_quantize_glu_24_28_ws_cuda_impl<false>(
        gate_up, qx24, xscale24, qx28, xscale28);
}

void nint_moe_quantize_geglu_24_28_ws_cuda(
        torch::Tensor gate_up,
        torch::Tensor qx24,
        torch::Tensor xscale24,
        torch::Tensor qx28,
        torch::Tensor xscale28) {
    nint_moe_quantize_glu_24_28_ws_cuda_impl<true>(
        gate_up, qx24, xscale24, qx28, xscale28);
}

torch::Tensor nint_moe_grouped_matmul_hetero_qx_cuda(
        torch::Tensor weight_ptrs,
        torch::Tensor pool_params,
        torch::Tensor activation_ptrs,
        torch::Tensor expert_pool,
        torch::Tensor expert_local,
        torch::Tensor ids,
        int64_t profile_mask,
        int64_t n_experts,
        int64_t out_per_expert,
        int64_t input_width,
        bool routed_input,
        torch::Tensor out,
        torch::Tensor ids_dst,
        torch::Tensor expert_bounds,
        torch::Tensor tile_bounds,
        torch::Tensor tile_experts) {
    TORCH_CHECK(n_experts > 0 && n_experts <= 4096, "n_experts must be in [1,4096]");
    TORCH_CHECK(out_per_expert > 0 && out_per_expert <= INT_MAX,
        "out_per_expert must be positive");
    TORCH_CHECK(input_width > 0 && input_width <= INT_MAX, "input_width must be positive");
    TORCH_CHECK(profile_mask > 0 && profile_mask < 128,
        "profile_mask must select at least one supported heterogeneous profile");
    TORCH_CHECK(weight_ptrs.is_cuda() && weight_ptrs.is_contiguous() &&
        weight_ptrs.scalar_type() == torch::kInt64 && weight_ptrs.dim() == 2 &&
        weight_ptrs.size(0) > 0 && weight_ptrs.size(1) == 5,
        "weight_ptrs must be contiguous CUDA int64 [pools,5]");
    const int pools = static_cast<int>(weight_ptrs.size(0));
    TORCH_CHECK(pool_params.is_cuda() && pool_params.is_contiguous() &&
        pool_params.scalar_type() == torch::kInt32 && pool_params.dim() == 2 &&
        pool_params.size(0) == pools && pool_params.size(1) == 2,
        "pool_params must be contiguous CUDA int32 [pools,2]");
    TORCH_CHECK(activation_ptrs.is_cuda() && activation_ptrs.is_contiguous() &&
        activation_ptrs.scalar_type() == torch::kInt64 && activation_ptrs.dim() == 2 &&
        activation_ptrs.size(0) == pools && activation_ptrs.size(1) == 2,
        "activation_ptrs must be contiguous CUDA int64 [pools,2]");
    const int experts = static_cast<int>(n_experts);
    TORCH_CHECK(expert_pool.is_cuda() && expert_pool.is_contiguous() &&
        expert_pool.scalar_type() == torch::kInt32 && expert_pool.numel() == experts,
        "expert_pool must be contiguous CUDA int32 [experts]");
    TORCH_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() &&
        expert_local.scalar_type() == torch::kInt32 && expert_local.numel() == experts,
        "expert_local must be contiguous CUDA int32 [experts]");
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32 &&
        ids.dim() == 2, "ids must be contiguous CUDA int32 [tokens,routes]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    TORCH_CHECK(tokens > 0 && routes > 0, "ids dimensions must be nonzero");
    const int output_width = static_cast<int>(out_per_expert);
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kFloat16 &&
        out.dim() == 3 && out.size(0) == tokens && out.size(1) == routes &&
        out.size(2) == output_width,
        "out must be contiguous CUDA float16 [tokens,routes,out_per_expert]");

    check_same_device(weight_ptrs, pool_params, "pool_params");
    check_same_device(weight_ptrs, activation_ptrs, "activation_ptrs");
    check_same_device(weight_ptrs, expert_pool, "expert_pool");
    check_same_device(weight_ptrs, expert_local, "expert_local");
    check_same_device(weight_ptrs, ids, "ids");
    check_same_device(weight_ptrs, out, "out");
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(tokens <= 8, "heterogeneous MoE launch only supports up to eight tokens");
    const int row_blocks = (output_width + 1) / 2;
#define MFQ_MOE_HETERO_GROUP4(PROFILE_MASK_VALUE) \
        nint_moe_hetero_mmvq_group_kernel<PROFILE_MASK_VALUE, 4><<< \
            dim3(output_width, routes, 1), dim3(32, 4), 0, stream>>>( \
            weight_ptrs.data_ptr<int64_t>(), pool_params.data_ptr<int32_t>(), \
            activation_ptrs.data_ptr<int64_t>(), expert_pool.data_ptr<int32_t>(), \
            expert_local.data_ptr<int32_t>(), ids.data_ptr<int32_t>(), \
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()), routes, experts, \
            output_width, routed_input)
#define MFQ_MOE_HETERO_GROUPWARP4(PROFILE_MASK_VALUE) \
        nint_moe_hetero_mmvq_groupwarp_kernel<PROFILE_MASK_VALUE, 4><<< \
            dim3((output_width + 3) / 4, routes, 1), dim3(32, 4), 0, stream>>>( \
            weight_ptrs.data_ptr<int64_t>(), pool_params.data_ptr<int32_t>(), \
            activation_ptrs.data_ptr<int64_t>(), expert_pool.data_ptr<int32_t>(), \
            expert_local.data_ptr<int32_t>(), ids.data_ptr<int32_t>(), \
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()), routes, experts, \
            output_width, routed_input)
#define MFQ_MOE_HETERO_LEGACY(TOKEN_TILE, PROFILE_MASK_VALUE) \
        nint_moe_hetero_mmvq_kernel<TOKEN_TILE, PROFILE_MASK_VALUE><<< \
            dim3(row_blocks, routes, 1), dim3(32, TOKEN_TILE), 0, stream>>>( \
            weight_ptrs.data_ptr<int64_t>(), pool_params.data_ptr<int32_t>(), \
            activation_ptrs.data_ptr<int64_t>(), expert_pool.data_ptr<int32_t>(), \
            expert_local.data_ptr<int32_t>(), ids.data_ptr<int32_t>(), \
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()), tokens, routes, experts, \
            output_width, routed_input)
    if (tokens == 1) {
        TORCH_CHECK(routes <= 8, "M=1 heterogeneous MoE supports at most eight routes");
#define MFQ_MOE_HETERO_DISPATCH(PROFILE_MASK_VALUE) \
        do { \
            if (input_width >= 1024) { \
                MFQ_MOE_HETERO_GROUP4(PROFILE_MASK_VALUE); \
            } else { \
                MFQ_MOE_HETERO_GROUPWARP4(PROFILE_MASK_VALUE); \
            } \
        } while (0)
        switch (static_cast<int>(profile_mask)) {
            case 3: MFQ_MOE_HETERO_DISPATCH(3); break;
            case 7: MFQ_MOE_HETERO_DISPATCH(7); break;
            default: MFQ_MOE_HETERO_DISPATCH(127); break;
        }
#undef MFQ_MOE_HETERO_DISPATCH
    } else {
#define MFQ_MOE_HETERO_TOKEN_DISPATCH(PROFILE_MASK_VALUE) \
        do { \
            if (tokens <= 2) { MFQ_MOE_HETERO_LEGACY(2, PROFILE_MASK_VALUE); } \
            else if (tokens <= 4) { MFQ_MOE_HETERO_LEGACY(4, PROFILE_MASK_VALUE); } \
            else { MFQ_MOE_HETERO_LEGACY(8, PROFILE_MASK_VALUE); } \
        } while (0)
        switch (static_cast<int>(profile_mask)) {
            case 3: MFQ_MOE_HETERO_TOKEN_DISPATCH(3); break;
            case 7: MFQ_MOE_HETERO_TOKEN_DISPATCH(7); break;
            default: MFQ_MOE_HETERO_TOKEN_DISPATCH(127); break;
        }
#undef MFQ_MOE_HETERO_TOKEN_DISPATCH
    }
#undef MFQ_MOE_HETERO_LEGACY
#undef MFQ_MOE_HETERO_GROUPWARP4
#undef MFQ_MOE_HETERO_GROUP4
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nint_moe_grouped_matmul_hetero_glu_qx_cuda(
        torch::Tensor weight_ptrs,
        torch::Tensor pool_params,
        torch::Tensor activation_ptrs,
        torch::Tensor expert_pool,
        torch::Tensor expert_local,
        torch::Tensor ids,
        int64_t profile_mask,
        int64_t n_experts,
        int64_t hidden_width,
        bool gelu,
        torch::Tensor out) {
    TORCH_CHECK(profile_mask > 0 && profile_mask < 128,
        "profile_mask must select a supported heterogeneous profile");
    TORCH_CHECK(n_experts > 0 && n_experts <= 4096, "n_experts must be in [1,4096]");
    TORCH_CHECK(hidden_width > 0 && hidden_width <= INT_MAX, "hidden_width must be positive");
    TORCH_CHECK(weight_ptrs.is_cuda() && weight_ptrs.is_contiguous() &&
        weight_ptrs.scalar_type() == torch::kInt64 && weight_ptrs.dim() == 2 &&
        weight_ptrs.size(1) == 5, "weight_ptrs must be CUDA int64 [pools,5]");
    const int pools = static_cast<int>(weight_ptrs.size(0));
    TORCH_CHECK(pool_params.is_cuda() && pool_params.is_contiguous() &&
        pool_params.scalar_type() == torch::kInt32 && pool_params.sizes() ==
            torch::IntArrayRef({pools, 2}), "pool_params must be CUDA int32 [pools,2]");
    TORCH_CHECK(activation_ptrs.is_cuda() && activation_ptrs.is_contiguous() &&
        activation_ptrs.scalar_type() == torch::kInt64 && activation_ptrs.sizes() ==
            torch::IntArrayRef({pools, 2}), "activation_ptrs must be CUDA int64 [pools,2]");
    TORCH_CHECK(expert_pool.is_cuda() && expert_pool.is_contiguous() &&
        expert_pool.scalar_type() == torch::kInt32 && expert_pool.numel() == n_experts,
        "expert_pool must be CUDA int32 [experts]");
    TORCH_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() &&
        expert_local.scalar_type() == torch::kInt32 && expert_local.numel() == n_experts,
        "expert_local must be CUDA int32 [experts]");
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32 &&
        ids.dim() == 2 && ids.size(0) > 0 && ids.size(0) <= 4 &&
        ids.size(1) > 0 && ids.size(1) <= 8,
        "ids must be CUDA int32 [tokens,routes] with at most four tokens and eight routes");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kFloat16 &&
        out.dim() == 3 && out.size(0) == tokens && out.size(1) == routes &&
        out.size(2) == hidden_width,
        "out must be CUDA float16 [tokens,routes,hidden_width]");
    check_same_device(weight_ptrs, pool_params, "pool_params");
    check_same_device(weight_ptrs, activation_ptrs, "activation_ptrs");
    check_same_device(weight_ptrs, expert_pool, "expert_pool");
    check_same_device(weight_ptrs, expert_local, "expert_local");
    check_same_device(weight_ptrs, ids, "ids");
    check_same_device(weight_ptrs, out, "out");

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define MFQ_MOE_GLU_GROUP4(PROFILE_MASK_VALUE) \
    do { \
        if (gelu) { \
            nint_moe_hetero_mmvq_glu_group_kernel<PROFILE_MASK_VALUE, 4, true><<< \
                dim3(static_cast<int>(hidden_width), routes, tokens), dim3(32, 4), 0, stream>>>( \
                weight_ptrs.data_ptr<int64_t>(), pool_params.data_ptr<int32_t>(), \
                activation_ptrs.data_ptr<int64_t>(), expert_pool.data_ptr<int32_t>(), \
                expert_local.data_ptr<int32_t>(), ids.data_ptr<int32_t>(), \
                reinterpret_cast<__half *>(out.data_ptr<at::Half>()), routes, \
                static_cast<int>(n_experts), static_cast<int>(hidden_width)); \
        } else { \
            nint_moe_hetero_mmvq_glu_group_kernel<PROFILE_MASK_VALUE, 4, false><<< \
                dim3(static_cast<int>(hidden_width), routes, tokens), dim3(32, 4), 0, stream>>>( \
                weight_ptrs.data_ptr<int64_t>(), pool_params.data_ptr<int32_t>(), \
                activation_ptrs.data_ptr<int64_t>(), expert_pool.data_ptr<int32_t>(), \
                expert_local.data_ptr<int32_t>(), ids.data_ptr<int32_t>(), \
                reinterpret_cast<__half *>(out.data_ptr<at::Half>()), routes, \
                static_cast<int>(n_experts), static_cast<int>(hidden_width)); \
        } \
    } while (0)
#define MFQ_MOE_GLU_PROFILE_DISPATCH(PROFILE_MASK_VALUE) \
    do { \
        MFQ_MOE_GLU_GROUP4(PROFILE_MASK_VALUE); \
    } while (0)
    switch (static_cast<int>(profile_mask)) {
        case 3: MFQ_MOE_GLU_PROFILE_DISPATCH(3); break;
        case 7: MFQ_MOE_GLU_PROFILE_DISPATCH(7); break;
        default: MFQ_MOE_GLU_PROFILE_DISPATCH(127); break;
    }
#undef MFQ_MOE_GLU_PROFILE_DISPATCH
#undef MFQ_MOE_GLU_GROUP4
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

static torch::Tensor nint_moe_grouped_matmul_hetero_f16_impl(
        torch::Tensor weight_ptrs,
        torch::Tensor pool_params,
        torch::Tensor expert_pool,
        torch::Tensor expert_local,
        torch::Tensor x,
        torch::Tensor ids,
        int64_t n_experts,
        int64_t out_per_expert,
        int64_t input_width,
        bool routed_input,
        torch::Tensor out,
        torch::Tensor ids_dst,
        torch::Tensor expert_bounds,
        torch::Tensor tile_bounds,
        torch::Tensor tile_experts,
        int64_t weight_out_stride,
        int64_t weight_row_offset) {
    TORCH_CHECK(n_experts > 0 && n_experts <= 4096, "n_experts must be in [1,4096]");
    TORCH_CHECK(out_per_expert > 0 && out_per_expert <= INT_MAX,
        "out_per_expert must be positive");
    TORCH_CHECK(weight_out_stride > 0 && weight_out_stride <= INT_MAX,
        "weight_out_stride must be positive");
    TORCH_CHECK(weight_row_offset >= 0 && weight_row_offset <= INT_MAX &&
        weight_row_offset + out_per_expert <= weight_out_stride,
        "weight row slice must fit within weight_out_stride");
    TORCH_CHECK(input_width > 0 && input_width <= INT_MAX, "input_width must be positive");
    TORCH_CHECK(weight_ptrs.is_cuda() && weight_ptrs.is_contiguous() &&
        weight_ptrs.scalar_type() == torch::kInt64 && weight_ptrs.dim() == 2 &&
        weight_ptrs.size(0) > 0 && weight_ptrs.size(1) == 5,
        "weight_ptrs must be contiguous CUDA int64 [pools,5]");
    const int pools = static_cast<int>(weight_ptrs.size(0));
    TORCH_CHECK(pool_params.is_cuda() && pool_params.is_contiguous() &&
        pool_params.scalar_type() == torch::kInt32 && pool_params.dim() == 2 &&
        pool_params.size(0) == pools && pool_params.size(1) == 2,
        "pool_params must be contiguous CUDA int32 [pools,2]");
    const int experts = static_cast<int>(n_experts);
    TORCH_CHECK(expert_pool.is_cuda() && expert_pool.is_contiguous() &&
        expert_pool.scalar_type() == torch::kInt32 && expert_pool.numel() == experts,
        "expert_pool must be contiguous CUDA int32 [experts]");
    TORCH_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() &&
        expert_local.scalar_type() == torch::kInt32 && expert_local.numel() == experts,
        "expert_local must be contiguous CUDA int32 [experts]");
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32 &&
        ids.dim() == 2, "ids must be contiguous CUDA int32 [tokens,routes]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    TORCH_CHECK(tokens > 8 && routes > 0, "grouped MMA requires more than eight tokens");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat16 &&
        (x.dim() == 2 || x.dim() == 3) && x.size(-1) == input_width,
        "x must be contiguous CUDA float16 with exact input width");
    TORCH_CHECK(routed_input == (x.dim() == 3), "routed_input does not match x rank");
    TORCH_CHECK((!routed_input && x.size(0) == tokens) ||
        (routed_input && x.size(0) == tokens && x.size(1) == routes),
        "x leading dimensions do not match ids");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kFloat16 &&
        out.dim() == 3 && out.size(0) == tokens && out.size(1) == routes &&
        out.size(2) == out_per_expert,
        "out must be contiguous CUDA float16 [tokens,routes,out_per_expert]");
    TORCH_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
        ids_dst.scalar_type() == torch::kInt32 && ids_dst.numel() >= pairs,
        "ids_dst workspace is too small");
    TORCH_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
        expert_bounds.scalar_type() == torch::kInt32 && expert_bounds.numel() >= experts + 1,
        "expert_bounds workspace is too small");
    TORCH_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
        tile_bounds.scalar_type() == torch::kInt32 && tile_bounds.numel() >= experts + 1,
        "tile_bounds workspace is too small");
    TORCH_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
        tile_experts.scalar_type() == torch::kInt32 && tile_experts.numel() >= pairs,
        "tile_experts workspace is too small");
    check_same_device(weight_ptrs, pool_params, "pool_params");
    check_same_device(weight_ptrs, expert_pool, "expert_pool");
    check_same_device(weight_ptrs, expert_local, "expert_local");
    check_same_device(weight_ptrs, x, "x");
    check_same_device(weight_ptrs, ids, "ids");
    check_same_device(weight_ptrs, out, "out");
    check_same_device(weight_ptrs, ids_dst, "ids_dst");
    check_same_device(weight_ptrs, expert_bounds, "expert_bounds");
    check_same_device(weight_ptrs, tile_bounds, "tile_bounds");
    check_same_device(weight_ptrs, tile_experts, "tile_experts");

    const int output_width = static_cast<int>(out_per_expert);
    const int ntiles_n = (output_width + kMoeMmaBn - 1) / kMoeMmaBn;
    const int64_t max_fine_tiles = (pairs + kRouteTile - 1) / kRouteTile + experts;
    const int64_t max_tasks = max_fine_tiles * ntiles_n;
    static const int block_cap = [] {
        const char * value = std::getenv("MFQ_MOE_PREFILL_MMA_BLOCKS");
        return value == nullptr ? 4096 : std::max(1, std::atoi(value));
    }();
    static const int forced_bm = [] {
        const char * value = std::getenv("MFQ_MOE_PREFILL_MMA_BM");
        if (value == nullptr) return 0;
        const int parsed = std::atoi(value);
        return parsed == 16 || parsed == 32 || parsed == 64 ? parsed : 0;
    }();
    const int blocks = static_cast<int>(
        std::max<int64_t>(1, std::min<int64_t>(block_cap, max_tasks)));
    const dim3 threads(32, 8);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int bm = forced_bm != 0 ? forced_bm :
        (tokens <= 128 ? 64 : (tokens <= 512 ? 32 : 64));
    if (bm == 16) {
        nint_moe_hetero_mma_kernel<16><<<blocks, threads, 0, stream>>>(
            weight_ptrs.data_ptr<int64_t>(), pool_params.data_ptr<int32_t>(),
            expert_pool.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
            reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
            ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(),
            tile_bounds.data_ptr<int32_t>(), tile_experts.data_ptr<int32_t>(),
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()), routes, experts,
            output_width, static_cast<int>(weight_out_stride),
            static_cast<int>(weight_row_offset),
            static_cast<int>(input_width), routed_input);
    } else if (bm == 32) {
        nint_moe_hetero_mma_kernel<32><<<blocks, threads, 0, stream>>>(
            weight_ptrs.data_ptr<int64_t>(), pool_params.data_ptr<int32_t>(),
            expert_pool.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
            reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
            ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(),
            tile_bounds.data_ptr<int32_t>(), tile_experts.data_ptr<int32_t>(),
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()), routes, experts,
            output_width, static_cast<int>(weight_out_stride),
            static_cast<int>(weight_row_offset),
            static_cast<int>(input_width), routed_input);
    } else {
        nint_moe_hetero_mma_kernel<64><<<blocks, threads, 0, stream>>>(
            weight_ptrs.data_ptr<int64_t>(), pool_params.data_ptr<int32_t>(),
            expert_pool.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
            reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
            ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(),
            tile_bounds.data_ptr<int32_t>(), tile_experts.data_ptr<int32_t>(),
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()), routes, experts,
            output_width, static_cast<int>(weight_out_stride),
            static_cast<int>(weight_row_offset),
            static_cast<int>(input_width), routed_input);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nint_moe_grouped_matmul_hetero_f16_cuda(
        torch::Tensor weight_ptrs,
        torch::Tensor pool_params,
        torch::Tensor expert_pool,
        torch::Tensor expert_local,
        torch::Tensor x,
        torch::Tensor ids,
        int64_t n_experts,
        int64_t out_per_expert,
        int64_t input_width,
        bool routed_input,
        torch::Tensor out,
        torch::Tensor ids_dst,
        torch::Tensor expert_bounds,
        torch::Tensor tile_bounds,
        torch::Tensor tile_experts) {
    return nint_moe_grouped_matmul_hetero_f16_impl(
        weight_ptrs, pool_params, expert_pool, expert_local, x, ids,
        n_experts, out_per_expert, input_width, routed_input, out,
        ids_dst, expert_bounds, tile_bounds, tile_experts,
        out_per_expert, 0);
}

torch::Tensor nint_moe_grouped_matmul_hetero_f16_slice_cuda(
        torch::Tensor weight_ptrs,
        torch::Tensor pool_params,
        torch::Tensor expert_pool,
        torch::Tensor expert_local,
        torch::Tensor x,
        torch::Tensor ids,
        int64_t n_experts,
        int64_t out_per_expert,
        int64_t input_width,
        bool routed_input,
        torch::Tensor out,
        torch::Tensor ids_dst,
        torch::Tensor expert_bounds,
        torch::Tensor tile_bounds,
        torch::Tensor tile_experts,
        int64_t weight_out_stride,
        int64_t weight_row_offset) {
    return nint_moe_grouped_matmul_hetero_f16_impl(
        weight_ptrs, pool_params, expert_pool, expert_local, x, ids,
        n_experts, out_per_expert, input_width, routed_input, out,
        ids_dst, expert_bounds, tile_bounds, tile_experts,
        weight_out_stride, weight_row_offset);
}

torch::Tensor nint_moe_grouped_matmul_pool_ws_cuda(
        torch::Tensor q_packed,
        torch::Tensor sub_scale,
        torch::Tensor sub_min,
        torch::Tensor neuron_scale,
        torch::Tensor neuron_min,
        torch::Tensor x,
        torch::Tensor ids,
        torch::Tensor expert_local,
        int64_t n_experts,
        int64_t n_local_experts,
        int64_t out_per_expert,
        int64_t gs,
        int64_t bits,
        bool route_map_ready,
        bool input_quantized,
        torch::Tensor out,
        torch::Tensor qx,
        torch::Tensor xscale,
        torch::Tensor counts,
        torch::Tensor cursors,
        torch::Tensor ids_dst,
        torch::Tensor expert_bounds,
        torch::Tensor tile_bounds,
        torch::Tensor tile_experts) {
    TORCH_CHECK(n_experts > 0 && n_experts <= 4096, "n_experts must be in [1, 4096]");
    TORCH_CHECK(n_local_experts > 0 && n_local_experts <= n_experts,
        "n_local_experts must be in [1, n_experts]");
    TORCH_CHECK(out_per_expert > 0 && out_per_expert <= INT_MAX, "out_per_expert must be positive");
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8,
        "NINT MoE supports bits in {2,3,4,5,6,8}");
    TORCH_CHECK(gs == 16 || gs == 24 || gs == 26 || gs == 28 || gs == 32 || gs == 48,
        "NINT MoE supports gs in {16,24,26,28,32,48}");
    const int experts = static_cast<int>(n_experts);
    const int local_experts = static_cast<int>(n_local_experts);
    const int output_width = static_cast<int>(out_per_expert);
    check_nint_weight(q_packed, sub_scale, sub_min, neuron_scale, neuron_min,
        local_experts, output_width, static_cast<int>(gs), static_cast<int>(bits));

    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32 && ids.dim() == 2,
        "ids must be contiguous CUDA int32 [tokens, routes]");
    TORCH_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() &&
        expert_local.scalar_type() == torch::kInt32 && expert_local.dim() == 1 &&
        expert_local.numel() == experts,
        "expert_local must be contiguous CUDA int32 [n_experts]");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat16 &&
        (x.dim() == 2 || x.dim() == 3), "x must be contiguous CUDA float16 [T,K] or [T,R,K]");
    check_same_device(q_packed, x, "x");
    check_same_device(q_packed, ids, "ids");
    check_same_device(q_packed, expert_local, "expert_local");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    TORCH_CHECK(tokens > 0 && routes > 0, "ids dimensions must be nonzero");
    const bool routed_input = x.dim() == 3;
    if (routed_input) {
        TORCH_CHECK(x.size(0) == tokens && x.size(1) == routes,
            "routed x must have [tokens, routes, K] leading dimensions");
    } else {
        TORCH_CHECK(x.size(0) == tokens, "shared x must have one row per token");
    }
    const int input_rows = routed_input ? tokens * routes : tokens;
    const int groups = static_cast<int>(q_packed.size(1));
    const int k_pad = groups * static_cast<int>(gs);
    const int k_real = input_quantized
        ? k_pad
        : static_cast<int>(x.size(-1));
    TORCH_CHECK(k_real <= k_pad, "x K exceeds packed NINT K");
    TORCH_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == torch::kInt8 &&
        qx.dim() == 2 && qx.size(0) >= input_rows && qx.size(1) >= k_pad,
        "qx workspace is too small");
    TORCH_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == torch::kFloat32 &&
        xscale.dim() == 2 && xscale.size(0) >= input_rows && xscale.size(1) >= groups,
        "xscale workspace is too small");
    check_same_device(q_packed, qx, "qx");
    check_same_device(q_packed, xscale, "xscale");

    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kFloat16 &&
        out.dim() == 3 && out.size(0) == tokens && out.size(1) == routes &&
        out.size(2) == output_width,
        "out must be contiguous CUDA float16 [tokens, routes, out_per_expert]");
    check_same_device(q_packed, out, "out");
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (tokens > 8 && !route_map_ready) {
        build_expert_map(ids, experts, kRouteTile, counts, cursors, ids_dst,
            expert_bounds, tile_bounds, tile_experts, stream);
    }
    if (tokens > 8) {
        TORCH_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
            tile_experts.scalar_type() == torch::kInt32 && tile_experts.numel() >= tokens * routes,
            "tile_experts workspace is too small");
        check_same_device(ids, tile_experts, "tile_experts");
    }

#define MFQ_MOE_LAUNCH(BITS_VALUE, GS_VALUE) \
    do { \
        if (!input_quantized) { \
            launch_quantize<GS_VALUE>(x.reshape({input_rows, k_real}), qx, xscale, input_rows, k_real, k_pad, groups, stream); \
        } \
        launch_grouped_matmul<BITS_VALUE, GS_VALUE>( \
            q_packed, sub_scale, sub_min, neuron_scale, neuron_min, qx, xscale, ids, expert_local, ids_dst, \
            expert_bounds, tile_bounds, tile_experts, out, tokens, routes, experts, output_width, groups, k_pad, \
            routed_input, stream); \
    } while (0)

#define MFQ_MOE_SWITCH_GS(BITS_VALUE) \
    switch (static_cast<int>(gs)) { \
        case 16: MFQ_MOE_LAUNCH(BITS_VALUE, 16); break; \
        case 24: MFQ_MOE_LAUNCH(BITS_VALUE, 24); break; \
        case 26: MFQ_MOE_LAUNCH(BITS_VALUE, 26); break; \
        case 28: MFQ_MOE_LAUNCH(BITS_VALUE, 28); break; \
        case 32: MFQ_MOE_LAUNCH(BITS_VALUE, 32); break; \
        case 48: MFQ_MOE_LAUNCH(BITS_VALUE, 48); break; \
    }

    switch (static_cast<int>(bits)) {
        case 2: MFQ_MOE_SWITCH_GS(2); break;
        case 3: MFQ_MOE_SWITCH_GS(3); break;
        case 4: MFQ_MOE_SWITCH_GS(4); break;
        case 5: MFQ_MOE_SWITCH_GS(5); break;
        case 6: MFQ_MOE_SWITCH_GS(6); break;
        case 8: MFQ_MOE_SWITCH_GS(8); break;
    }
#undef MFQ_MOE_SWITCH_GS
#undef MFQ_MOE_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nint8_zero_moe_grouped_matmul_pool_ws_cuda(
        torch::Tensor q,
        torch::Tensor scale,
        torch::Tensor x,
        torch::Tensor ids,
        torch::Tensor expert_local,
        int64_t n_experts,
        int64_t n_local_experts,
        int64_t out_per_expert,
        bool route_map_ready,
        bool input_quantized,
        bool use_f16_mma,
        torch::Tensor out,
        torch::Tensor qx,
        torch::Tensor xscale,
        torch::Tensor counts,
        torch::Tensor cursors,
        torch::Tensor ids_dst,
        torch::Tensor expert_bounds,
        torch::Tensor tile_bounds,
        torch::Tensor tile_experts) {
    TORCH_CHECK(
        n_experts > 0 && n_experts <= 4096,
        "n_experts must be in [1, 4096]");
    TORCH_CHECK(
        n_local_experts > 0 && n_local_experts <= n_experts,
        "n_local_experts must be in [1, n_experts]");
    TORCH_CHECK(
        out_per_expert > 0 && out_per_expert <= INT_MAX,
        "out_per_expert must be positive");
    const int experts = static_cast<int>(n_experts);
    const int local_experts = static_cast<int>(n_local_experts);
    const int output_width = static_cast<int>(out_per_expert);
    TORCH_CHECK(
        q.is_cuda() && q.is_contiguous() &&
        q.scalar_type() == torch::kUInt8 && q.dim() == 3 &&
        q.size(0) == static_cast<int64_t>(local_experts) * output_width &&
        q.size(2) == 32,
        "NINT8-0 MoE q must be contiguous CUDA uint8 "
        "[local_experts*out,groups,32]");
    TORCH_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type() == torch::kFloat16 && scale.dim() == 2 &&
        scale.size(0) == q.size(0) && scale.size(1) == q.size(1),
        "NINT8-0 MoE scale must be contiguous CUDA f16 "
        "[local_experts*out,groups]");
    check_same_device(q, scale, "scale");

    TORCH_CHECK(
        ids.is_cuda() && ids.is_contiguous() &&
        ids.scalar_type() == torch::kInt32 && ids.dim() == 2,
        "ids must be contiguous CUDA int32 [tokens, routes]");
    TORCH_CHECK(
        expert_local.is_cuda() && expert_local.is_contiguous() &&
        expert_local.scalar_type() == torch::kInt32 &&
        expert_local.dim() == 1 && expert_local.numel() == experts,
        "expert_local must be contiguous CUDA int32 [n_experts]");
    TORCH_CHECK(
        x.is_cuda() && x.is_contiguous() &&
        x.scalar_type() == torch::kFloat16 &&
        (x.dim() == 2 || x.dim() == 3),
        "x must be contiguous CUDA float16 [T,K] or [T,R,K]");
    check_same_device(q, x, "x");
    check_same_device(q, ids, "ids");
    check_same_device(q, expert_local, "expert_local");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    TORCH_CHECK(tokens > 0 && routes > 0, "ids dimensions must be nonzero");
    const bool routed_input = x.dim() == 3;
    if (routed_input) {
        TORCH_CHECK(
            x.size(0) == tokens && x.size(1) == routes,
            "routed x must have [tokens, routes, K] leading dimensions");
    } else {
        TORCH_CHECK(
            x.size(0) == tokens,
            "shared x must have one row per token");
    }
    const int input_rows = routed_input ? tokens * routes : tokens;
    const int groups = static_cast<int>(q.size(1));
    const int k_pad = groups * 32;
    const int input_width = static_cast<int>(x.size(-1));
    const int k_real =
        input_quantized ? k_pad : static_cast<int>(x.size(-1));
    TORCH_CHECK(input_width <= k_pad, "x K exceeds packed NINT8-0 K");
    TORCH_CHECK(
        qx.is_cuda() && qx.is_contiguous() &&
        qx.scalar_type() == torch::kInt8 && qx.dim() == 2 &&
        qx.size(0) >= input_rows && qx.size(1) >= k_pad,
        "qx workspace is too small");
    TORCH_CHECK(
        xscale.is_cuda() && xscale.is_contiguous() &&
        xscale.scalar_type() == torch::kFloat32 && xscale.dim() == 2 &&
        xscale.size(0) >= input_rows && xscale.size(1) >= groups,
        "xscale workspace is too small");
    check_same_device(q, qx, "qx");
    check_same_device(q, xscale, "xscale");
    TORCH_CHECK(
        out.is_cuda() && out.is_contiguous() &&
        out.scalar_type() == torch::kFloat16 && out.dim() == 3 &&
        out.size(0) == tokens && out.size(1) == routes &&
        out.size(2) == output_width,
        "out must be contiguous CUDA float16 "
        "[tokens, routes, out_per_expert]");
    check_same_device(q, out, "out");

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (tokens > 8 && !route_map_ready) {
        build_expert_map(
            ids, experts, kRouteTile, counts, cursors, ids_dst,
            expert_bounds, tile_bounds, tile_experts, stream);
    }
    if (tokens > 8) {
        TORCH_CHECK(
            tile_experts.is_cuda() && tile_experts.is_contiguous() &&
            tile_experts.scalar_type() == torch::kInt32 &&
            tile_experts.numel() >= tokens * routes,
            "tile_experts workspace is too small");
        check_same_device(ids, tile_experts, "tile_experts");
    }
    if (use_f16_mma) {
        TORCH_CHECK(tokens > 8, "NINT8-0 grouped MMA requires more than eight tokens");
        launch_nint8_zero_moe_mma(
            q, scale, expert_local, x, ids_dst, expert_bounds, tile_bounds,
            tile_experts, out, tokens, routes, experts, output_width, groups,
            input_width, routed_input, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    if (!input_quantized) {
        launch_quantize<32>(
            x.reshape({input_rows, k_real}), qx, xscale,
            input_rows, k_real, k_pad, groups, stream);
    }
    launch_nint8_zero_grouped_matmul(
        q, scale, qx, xscale, ids, expert_local, ids_dst,
        expert_bounds, tile_bounds, tile_experts, out, tokens, routes,
        experts, output_width, groups, k_pad, routed_input, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor moe_weighted_reduce_cuda(torch::Tensor pair_output, torch::Tensor weights) {
    TORCH_CHECK(pair_output.is_cuda() && pair_output.is_contiguous() &&
        pair_output.scalar_type() == torch::kFloat16 && pair_output.dim() == 3,
        "pair_output must be contiguous CUDA float16 [tokens, routes, width]");
    TORCH_CHECK(weights.is_cuda() && weights.is_contiguous() &&
        weights.scalar_type() == torch::kFloat32 && weights.dim() == 2,
        "weights must be contiguous CUDA float32 [tokens, routes]");
    TORCH_CHECK(pair_output.size(0) == weights.size(0) && pair_output.size(1) == weights.size(1),
        "pair_output and weights leading dimensions must match");
    check_same_device(pair_output, weights, "weights");
    const int tokens = static_cast<int>(pair_output.size(0));
    const int routes = static_cast<int>(pair_output.size(1));
    const int width = static_cast<int>(pair_output.size(2));
    auto output = torch::empty({tokens, width}, pair_output.options());
    const int block = 256;
    const int64_t total = static_cast<int64_t>(tokens) * width;
    const int grid = static_cast<int>((total + block - 1) / block);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    moe_weighted_reduce_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half *>(pair_output.data_ptr<at::Half>()),
        weights.data_ptr<float>(), reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
        tokens, routes, width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor moe_swiglu_split_cuda(torch::Tensor gate_up) {
    TORCH_CHECK(gate_up.is_cuda() && gate_up.is_contiguous() &&
        gate_up.scalar_type() == torch::kFloat16 && gate_up.dim() == 3,
        "gate_up must be contiguous CUDA float16 [tokens, routes, 2 * width]");
    TORCH_CHECK(gate_up.size(2) > 0 && gate_up.size(2) % 2 == 0,
        "gate_up width must be positive and even");
    const int tokens = static_cast<int>(gate_up.size(0));
    const int routes = static_cast<int>(gate_up.size(1));
    const int width = static_cast<int>(gate_up.size(2) / 2);
    TORCH_CHECK(tokens > 0 && routes > 0, "gate_up leading dimensions must be nonzero");
    auto output = torch::empty({tokens, routes, width}, gate_up.options());
    const int rows = tokens * routes;
    const int64_t total = static_cast<int64_t>(rows) * width;
    constexpr int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    moe_glu_split_kernel<false><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half *>(gate_up.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()), rows, width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor moe_geglu_split_cuda(torch::Tensor gate_up) {
    TORCH_CHECK(gate_up.is_cuda() && gate_up.is_contiguous() &&
        gate_up.scalar_type() == torch::kFloat16 && gate_up.dim() >= 2,
        "gate_up must be contiguous CUDA float16 [..., 2 * width]");
    TORCH_CHECK(gate_up.size(-1) > 0 && gate_up.size(-1) % 2 == 0,
        "gate_up width must be positive and even");
    const int width = static_cast<int>(gate_up.size(-1) / 2);
    const int64_t rows64 = gate_up.numel() / (2 * width);
    TORCH_CHECK(rows64 > 0 && rows64 <= INT_MAX, "gate_up row count is unsupported");
    auto shape = gate_up.sizes().vec();
    shape.back() = width;
    auto output = torch::empty(shape, gate_up.options());
    constexpr int block = 256;
    const int64_t total = rows64 * width;
    const int grid = static_cast<int>((total + block - 1) / block);
    moe_glu_split_kernel<true><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half *>(gate_up.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
        static_cast<int>(rows64), width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void moe_apply_expert_scale_kernel(
        float * __restrict__ weights,
        const int32_t * __restrict__ ids,
        const float * __restrict__ scales,
        int total) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total;
         index += blockDim.x * gridDim.x) {
        weights[index] *= scales[ids[index]];
    }
}

torch::Tensor moe_apply_expert_scale_cuda(
        torch::Tensor weights, torch::Tensor ids, torch::Tensor scales) {
    TORCH_CHECK(weights.is_cuda() && weights.is_contiguous() &&
        weights.scalar_type() == torch::kFloat32 && weights.dim() == 2,
        "weights must be contiguous CUDA float32 [tokens, routes]");
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() &&
        ids.scalar_type() == torch::kInt32 && ids.sizes() == weights.sizes(),
        "ids must be contiguous CUDA int32 with the same shape as weights");
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() &&
        scales.scalar_type() == torch::kFloat32 && scales.dim() == 1,
        "scales must be contiguous CUDA float32 [experts]");
    check_same_device(weights, ids, "ids");
    check_same_device(weights, scales, "scales");
    const int64_t total64 = weights.numel();
    TORCH_CHECK(total64 <= INT_MAX, "expert scale tensor is too large");
    constexpr int block = 256;
    const int grid = static_cast<int>((total64 + block - 1) / block);
    moe_apply_expert_scale_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        weights.data_ptr<float>(), ids.data_ptr<int32_t>(), scales.data_ptr<float>(),
        static_cast<int>(total64));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return weights;
}

torch::Tensor moe_add_shared_gate_cuda(
        torch::Tensor routed,
        torch::Tensor shared,
        torch::Tensor gate_logits) {
    TORCH_CHECK(routed.is_cuda() && routed.is_contiguous() &&
        routed.scalar_type() == torch::kFloat16 && routed.dim() == 2,
        "routed must be contiguous CUDA float16 [tokens, width]");
    TORCH_CHECK(shared.is_cuda() && shared.is_contiguous() &&
        shared.scalar_type() == torch::kFloat16 && shared.sizes() == routed.sizes(),
        "shared must match routed as contiguous CUDA float16");
    TORCH_CHECK(gate_logits.is_cuda() && gate_logits.is_contiguous() &&
        gate_logits.scalar_type() == torch::kFloat32 && gate_logits.dim() == 2 &&
        gate_logits.size(0) == routed.size(0) && gate_logits.size(1) == 1,
        "gate_logits must be contiguous CUDA float32 [tokens, 1]");
    check_same_device(routed, shared, "shared");
    check_same_device(routed, gate_logits, "gate_logits");
    const int rows = static_cast<int>(routed.size(0));
    const int width = static_cast<int>(routed.size(1));
    auto output = torch::empty_like(routed);
    const int64_t total = static_cast<int64_t>(rows) * width;
    constexpr int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    moe_add_shared_gate_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half *>(routed.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(shared.data_ptr<at::Half>()),
        gate_logits.data_ptr<float>(),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()), rows, width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor moe_weighted_reduce_shared_gate_cuda(
        torch::Tensor pair_output,
        torch::Tensor weights,
        torch::Tensor shared,
        torch::Tensor gate_logits) {
    TORCH_CHECK(pair_output.is_cuda() && pair_output.is_contiguous() &&
        pair_output.scalar_type() == torch::kFloat16 && pair_output.dim() == 3,
        "pair_output must be contiguous CUDA float16 [tokens, routes, width]");
    TORCH_CHECK(weights.is_cuda() && weights.is_contiguous() &&
        weights.scalar_type() == torch::kFloat32 && weights.dim() == 2 &&
        pair_output.size(0) == weights.size(0) && pair_output.size(1) == weights.size(1),
        "weights must match pair_output as contiguous CUDA float32 [tokens, routes]");
    TORCH_CHECK(shared.is_cuda() && shared.is_contiguous() &&
        shared.scalar_type() == torch::kFloat16 && shared.dim() == 2 &&
        shared.size(0) == pair_output.size(0) && shared.size(1) == pair_output.size(2),
        "shared must be contiguous CUDA float16 [tokens, width]");
    TORCH_CHECK(gate_logits.is_cuda() && gate_logits.is_contiguous() &&
        gate_logits.scalar_type() == torch::kFloat32 && gate_logits.dim() == 2 &&
        gate_logits.size(0) == pair_output.size(0) && gate_logits.size(1) == 1,
        "gate_logits must be contiguous CUDA float32 [tokens, 1]");
    check_same_device(pair_output, weights, "weights");
    check_same_device(pair_output, shared, "shared");
    check_same_device(pair_output, gate_logits, "gate_logits");
    const int tokens = static_cast<int>(pair_output.size(0));
    const int routes = static_cast<int>(pair_output.size(1));
    const int width = static_cast<int>(pair_output.size(2));
    auto output = torch::empty({tokens, width}, pair_output.options());
    constexpr int block = 256;
    const int64_t total = static_cast<int64_t>(tokens) * width;
    const int grid = static_cast<int>((total + block - 1) / block);
    moe_weighted_reduce_shared_gate_kernel<<<
        grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half *>(pair_output.data_ptr<at::Half>()),
        weights.data_ptr<float>(),
        reinterpret_cast<const __half *>(shared.data_ptr<at::Half>()),
        gate_logits.data_ptr<float>(),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
        tokens, routes, width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
