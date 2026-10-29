#pragma once
#include <hip/hip_runtime.h>

// E4M3FN uses bias 7 and a 448 finite maximum. MI300's native FNUZ
// conversion uses bias 8 and a 240 maximum, so it cannot simulate this cache.
__device__ __forceinline__ float mfq_round_e4m3fn(float value) {
    const float bounded=fminf(448.f,fmaxf(-448.f,value));
    const float magnitude=fabsf(bounded);
    const int exponent=int((__float_as_uint(magnitude)>>23)&255);
    const float step=__uint_as_float(unsigned(max(exponent-3,118))<<23);
    return copysignf(nearbyintf(magnitude/step)*step,bounded);
}

__device__ __forceinline__ float mfq_decode_e4m3fn(unsigned char bits) {
    const unsigned magnitude=bits&127u, exponent=magnitude>>3, mantissa=magnitude&7u;
    const float value=magnitude==127u ? __uint_as_float(0x7fc00000u) :
        exponent ? __uint_as_float(((exponent+120u)<<23)|(mantissa<<20)) :
        float(mantissa)*0x1p-9f;
    return copysignf(value,(bits&128u)?-1.f:1.f);
}
