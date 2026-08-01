// Block reduction primitives (mirrors llama.cpp reduce_rows.cuh).

#pragma once

#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, o);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    }
    return v;
}

// NW = blockDim.x / 32 (compile-time).
template <int NW>
__device__ __forceinline__ float block_sum(float v)
{
    __shared__ float sm[NW];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) {
        sm[wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        v = (lane < NW) ? sm[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) {
            sm[0] = v;
        }
    }
    __syncthreads();
    return sm[0];
}

template <int NW>
__device__ __forceinline__ float block_max(float v)
{
    __shared__ float sm[NW];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) {
        sm[wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        v = (lane < NW) ? sm[lane] : 0.0f;
        v = warp_max(v);
        if (lane == 0) {
            sm[0] = v;
        }
    }
    __syncthreads();
    return sm[0];
}
