#pragma once

#include <cuda_runtime.h>

template <bool GELU>
__device__ __forceinline__ float mfq_glu(float gate, float up) {
    if constexpr (GELU) {
        constexpr float kGeluA = 0.044715f;
        constexpr float kSqrt2OverPi = 0.79788456080286535587989211986876f;
        const float gelu = 0.5f * gate *
            (1.0f + tanhf(kSqrt2OverPi * gate * (1.0f + kGeluA * gate * gate)));
        return up * gelu;
    } else {
        return up * gate / (1.0f + expf(-gate));
    }
}

__device__ __forceinline__ float mfq_glu_runtime(float gate, float up, int activation) {
    return activation == 1 ? mfq_glu<true>(gate, up) : mfq_glu<false>(gate, up);
}
