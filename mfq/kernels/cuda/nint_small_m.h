#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

struct NintSmallMProjection {
    const uint8_t* q_packed;
    const uint8_t* sub_scale;
    const uint8_t* sub_min;
    const float* neuron_scale;
    const float* neuron_min;
    void* out;
    int n;
    const int32_t* xsum = nullptr;
};

using Nint4Gs24Projection = NintSmallMProjection;

void launch_nint23_group4_small_m(
    NintSmallMProjection weight, const int8_t* qx, const float* xs,
    int bits, int m, int ng, int kpad, cudaStream_t stream);

void launch_nint5_gs28_small_m(
    NintSmallMProjection weight, const int8_t* qx, const float* xs,
    int m, int ng, int kpad, cudaStream_t stream);

void launch_nint4_gs24_small_m_reuse(
    Nint4Gs24Projection weight, const int8_t* qx, const float* xs,
    int m, int ng, int kpad, cudaStream_t stream);

void launch_nint6_gs24_small_m_reuse(
    NintSmallMProjection weight, const int8_t* qx, const float* xs,
    int m, int ng, int kpad, cudaStream_t stream);

void launch_nint8_gs48_small_m(
    NintSmallMProjection weight, const int8_t* qx, const float* xs,
    int m, int ng, int kpad, cudaStream_t stream);
