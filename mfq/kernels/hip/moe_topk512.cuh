#pragma once
#include <hip/hip_math_constants.h>
// 512-expert, top-10, unbiased softmax routing. Keep MFQ's original
// per-lane summation and shuffle-down normalization order.
template <typename scalar_t, bool Sorted, bool Redux=false>
__global__ void __launch_bounds__(32) topk512_specialized_kernel(
        const scalar_t* __restrict__ logits, int32_t* __restrict__ ids,
        float* __restrict__ weights, float floor, float scale) {
    const int lane = threadIdx.x;
    logits += size_t(blockIdx.x) * 512;
    ids += size_t(blockIdx.x) * 10;
    weights += size_t(blockIdx.x) * 10;
    float values[16];
    float maximum = -INFINITY;
#pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        float raw = load_float(logits, lane + slot * 32);
        values[slot] = isnan(raw) ? -FLT_MAX : raw;
        maximum = fmaxf(maximum, values[slot]);
    }
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        values[slot] = expf(values[slot] - maximum);
        sum += values[slot];
    }
    sum = warp_sum(sum);
    int expert_ids[16];
#pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        values[slot] /= sum;
        expert_ids[slot] = lane + slot * 32;
        if constexpr (Sorted) {
            if (isnan(values[slot])) { values[slot] = -INFINITY; expert_ids[slot] = INT_MAX; }
        }
    }
    if constexpr (Sorted) {
        // Fully unrolled register sorting network: no dynamically indexed
        // register array, shared-memory traffic, or block barriers.
#pragma unroll
        for (int width = 2; width <= 16; width <<= 1) {
#pragma unroll
            for (int stride = width >> 1; stride; stride >>= 1) {
#pragma unroll
                for (int slot = 0; slot < 16; ++slot) {
                    const int peer = slot ^ stride;
                    if (peer > slot) {
                        const bool first = values[slot] > values[peer] ||
                            (values[slot] == values[peer] && expert_ids[slot] < expert_ids[peer]);
                        if (((slot & width) == 0) ? !first : first) {
                            const float v = values[slot]; values[slot] = values[peer]; values[peer] = v;
                            const int e = expert_ids[slot]; expert_ids[slot] = expert_ids[peer]; expert_ids[peer] = e;
                        }
                    }
                }
            }
        }
    }
    float selected = 0.0f;
#pragma unroll 1
    for (int rank = 0; rank < 10; ++rank) {
        float best = -INFINITY;
        int expert = INT_MAX;
        if constexpr (Sorted) { best = values[0]; expert = expert_ids[0]; }
        else {
#pragma unroll
            for (int slot = 0; slot < 16; ++slot) {
                const int id = lane + slot * 32;
                if (values[slot] > best || (values[slot] == best && id < expert)) {
                    best = values[slot]; expert = id;
                }
            }
        }
#if __CUDA_ARCH__ >= 800
        if constexpr (Redux) {
            // Softmax weights are nonnegative; -infinity is the only
            // negative sentinel. Signed float-flip preserves their order.
            const int bits = __float_as_int(best);
            const int ordered = bits < 0 ? bits ^ 0x7fffffff : bits;
            const int winner = __reduce_max_sync(0xffffffffu, ordered);
            expert = __reduce_min_sync(0xffffffffu, ordered == winner ? expert : INT_MAX);
            best = __int_as_float(winner < 0 ? winner ^ 0x7fffffff : winner);
        } else
#endif
        {
#pragma unroll
            for (int offset = 16; offset; offset >>= 1) {
                const float other = __shfl_down_sync(0xffffffffu, best, offset);
                const int other_id = __shfl_down_sync(0xffffffffu, expert, offset);
                if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
            }
            best = __shfl_sync(0xffffffffu, best, 0);
            expert = __shfl_sync(0xffffffffu, expert, 0);
        }
        if (lane == rank) { selected = best; ids[rank] = expert; }
        if constexpr (Sorted) {
            if (expert != INT_MAX && lane == (expert & 31)) {
#pragma unroll
                for (int slot = 0; slot < 15; ++slot) {
                    values[slot] = values[slot + 1]; expert_ids[slot] = expert_ids[slot + 1];
                }
                values[15] = -INFINITY; expert_ids[15] = INT_MAX;
            }
        } else {
#pragma unroll
            for (int slot = 0; slot < 16; ++slot) {
                if (expert == lane + slot * 32) values[slot] = -INFINITY;
            }
        }
    }
    const float denominator = fmaxf(warp_sum(selected), floor);
    if (lane < 10) weights[lane] = (selected / denominator) * scale;
}
