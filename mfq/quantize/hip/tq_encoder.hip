// Copyright 2026 MFQ contributors. SPDX-License-Identifier: Apache-2.0
// Independent suffix Viterbi for 256-element, 16-state-bit TQ rings.
// A free pass supplies the midpoint boundary; a constrained pass closes it.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_math_constants.h>
#include <algorithm>
#include "../../kernels/hip/mfq_hip_compat.h"
#include <cstdint>
#ifdef _WIN32
#define MFQ_TQ_EXPORT __declspec(dllexport)
#else
#define MFQ_TQ_EXPORT
#endif

namespace {
template<int K> constexpr int edges = 1 << (16 - K);
template<int K> constexpr int threads = edges<K> < 1024 ? edges<K> : 1024;

__device__ uint64_t minimum(uint64_t key, uint64_t* shared) {
    for (int d = 16; d; d /= 2) {
        const uint64_t other = __shfl_down_sync(0xffffffff, key, d);
        key = key < other ? key : other;
    }
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = key;
    __syncthreads();
    if (threadIdx.x == 0)
        for (int j = 1; j < blockDim.x / 32; ++j)
            shared[0] = shared[0] < shared[j] ? shared[0] : shared[j];
    __syncthreads();
    return shared[0];
}

template<int K>
__global__ __launch_bounds__(threads<K>, 1) void encode_rings(
    const float* x, const float* metric, const half* table, float* values,
    uint16_t* states, uint8_t* history, float* global_cost, int* boundaries) {
    constexpr int E = edges<K>, T = threads<K>, N = E / T;
    const int tid = threadIdx.x;
    const int64_t ring = blockIdx.x;
    extern __shared__ float shared_cost[];
    // Low-bit cost tables use global memory to leave room for block metadata.
    volatile float* cost = K <= 2 ? global_cost + ring * E : shared_cost;
    __shared__ uint64_t minima[32];
    __shared__ int boundary;
    x += ring * 256; metric += ring * 256;
    values += ring * 256; states += ring * 256; history += ring * 256 * E;
    for (int pass = 0; pass < 2; ++pass) {
        for (int e = tid; e < E; e += T)
            cost[e] = pass ? (e == boundary ? 0.f : INFINITY) : 0.f;
        __syncthreads();
        for (int t = 0; t < 256; ++t) {
            const int at = pass ? t : (t + 128) % 256;
            float next[N]; uint8_t pick[N];
            const float target = x[at], weight = metric[at];
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                const int e = tid + i * T;
                float best = INFINITY; int chosen = 0;
                #pragma unroll
                for (int j = 0; j < (1 << K); ++j) {
                    const int state = e + j * E;
                    const float delta = (__half2float(table[state]) - target) * weight;
                    const float candidate = __fmaf_rn(delta, delta, cost[state >> K]);
                    if (candidate < best) { best = candidate; chosen = j; }
                }
                next[i] = best; pick[i] = uint8_t(chosen);
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                const int e = tid + i * T;
                cost[e] = next[i]; history[int64_t(t) * E + e] = pick[i];
            }
            __syncthreads();
        }
        if (pass == 0) {
            uint64_t best = UINT64_MAX;
            for (int e = tid; e < E; e += T) {
                const uint64_t key = (uint64_t(__float_as_uint(cost[e])) << 32) | uint32_t(e);
                best = best < key ? best : key;
            }
            const uint64_t chosen = minimum(best, minima);
            if (tid == 0) {
                int e = int(uint32_t(chosen));
                for (int t = 255; t >= 128; --t)
                    e = (e + int(history[int64_t(t) * E + e]) * E) >> K;
                boundary = e; boundaries[ring] = e;
            }
            __syncthreads();
        } else if (tid == 0) {
            int e = boundary;
            for (int t = 255; t >= 0; --t) {
                const int state = e + int(history[int64_t(t) * E + e]) * E;
                states[t] = uint16_t(state); values[t] = __half2float(table[state]);
                e = state >> K;
            }
        }
    }
}

template<int K> int capacity(int count) {
    constexpr int bytes = K <= 2 ? 0 : edges<K> * int(sizeof(float));
    auto kernel = encode_rings<K>;
    hipError_t error = hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (error != hipSuccess) return -int(error);
    int resident = 0, device = 0;
    error = hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, threads<K>, bytes);
    if (error != hipSuccess) return -int(error);
    error = hipGetDevice(&device);
    if (error != hipSuccess) return -int(error);
    hipDeviceProp_t prop;
    error = hipGetDeviceProperties(&prop, device);
    if (error != hipSuccess) return -int(error);
    return resident ? std::min(count, resident * prop.multiProcessorCount) : -int(hipErrorInvalidConfiguration);
}

template<int K> int launch(const float* x, const float* metric, const half* table,
    float* values, uint16_t* states, uint8_t* history, float* cost, int* boundaries,
    int count, int wave, hipStream_t stream) {
    constexpr int bytes = K <= 2 ? 0 : edges<K> * int(sizeof(float));
    for (int first = 0; first < count; first += wave) {
        const int n = std::min(wave, count - first);
        const int64_t offset = int64_t(first) * 256;
        encode_rings<K><<<n, threads<K>, bytes, stream>>>(x + offset, metric + offset,
            table, values + offset, states + offset, history, cost, boundaries + first);
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess) return int(error);
    }
    return 0;
}
} // namespace

extern "C" MFQ_TQ_EXPORT int mfq_tq_encoder_wave(int count, int bits) {
    if (count <= 0) return -int(hipErrorInvalidValue);
    #define C(K) case K: return capacity<K>(count)
    switch (bits) { C(1); C(2); C(3); C(4); C(5); C(6); C(7); C(8); }
    #undef C
    return -int(hipErrorInvalidValue);
}

extern "C" MFQ_TQ_EXPORT int mfq_tq_encode(const float* x, const float* metric,
    const half* table, float* values, uint16_t* states, uint8_t* history, float* cost,
    int* boundaries, int count, int bits, int wave, hipStream_t stream) {
    if (count <= 0 || wave <= 0 || !x || !metric || !table || !values ||
        !states || !history || !cost || !boundaries) return int(hipErrorInvalidValue);
    #define C(K) case K: return launch<K>(x,metric,table,values,states,history,cost,boundaries,count,wave,stream)
    switch (bits) { C(1); C(2); C(3); C(4); C(5); C(6); C(7); C(8); }
    #undef C
    return int(hipErrorInvalidValue);
}
