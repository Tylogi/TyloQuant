#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

struct Nint4Gs24Projection {
    const uint8_t* q_packed;
    const uint8_t* sub_scale;
    const uint8_t* sub_min;
    const float* neuron_scale;
    const float* neuron_min;
    void* out;
    int n;
    const int32_t* xsum = nullptr;
};

void launch_nint4_gs24_small_m_reuse(
    Nint4Gs24Projection weight, const int8_t* qx, const float* xs,
    int m, int ng, int kpad, cudaStream_t stream);
