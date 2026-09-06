#include "mfq_tensor_backend.h"
#include "nint_small_m.h"

__device__ __forceinline__ int small_m_unpack_int4(uint32_t packed)
{
    return ((int)(packed & 0x000f)) |
           ((int)(packed & 0x00f0) << 4) |
           ((int)(packed & 0x0f00) << 8) |
           ((int)(packed & 0xf000) << 12);
}

template <int MROWS>
__global__ void __launch_bounds__(128) nint4_gs24_small_m_reuse_kernel(
    Nint4Gs24Projection weight,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    int ng, int kpad)
{
    constexpr int NWARPS = 4;
    const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (row >= weight.n) return;
    const auto* qrow = weight.q_packed + (size_t)row * ng * 12;
    const auto* ssrow = weight.sub_scale + (size_t)row * ng;
    const auto* smrow = weight.sub_min + (size_t)row * ng;
    float pd[MROWS] = {}, pm[MROWS] = {};
    for (int g = warp * 32 + lane; g < ng; g += NWARPS * 32) {
        const auto* qwords = reinterpret_cast<const uint32_t*>(qrow + g * 12);
        const uint32_t qw0 = qwords[0], qw1 = qwords[1], qw2 = qwords[2];
        int dsum[MROWS] = {}, msum[MROWS] = {};
        #pragma unroll
        for (int chunk = 0; chunk < 6; ++chunk) {
            const uint32_t qw = chunk < 2 ? qw0 : (chunk < 4 ? qw1 : qw2);
            const int qv = small_m_unpack_int4(qw >> ((chunk & 1) * 16));
            #pragma unroll
            for (int m = 0; m < MROWS; ++m) {
                const int xv = *reinterpret_cast<const int*>(qx + (size_t)m * kpad + g * 24 + chunk * 4);
                dsum[m] = __dp4a(qv, xv, dsum[m]);
                msum[m] = __dp4a(0x01010101, xv, msum[m]);
            }
        }
        const float ss = ssrow[g], sm = smrow[g];
        #pragma unroll
        for (int m = 0; m < MROWS; ++m) {
            const float xs = xscale[(size_t)m * ng + g];
            pd[m] += xs * ss * (float)dsum[m];
            pm[m] += xs * sm * (float)msum[m];
        }
    }
    __shared__ float partial_d[MROWS][NWARPS];
    __shared__ float partial_m[MROWS][NWARPS];
    #pragma unroll
    for (int m = 0; m < MROWS; ++m) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            pd[m] += __shfl_xor_sync(0xffffffff, pd[m], offset);
            pm[m] += __shfl_xor_sync(0xffffffff, pm[m], offset);
        }
        if (lane == 0) {
            partial_d[m][warp] = pd[m];
            partial_m[m][warp] = pm[m];
        }
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int m = 0; m < MROWS; ++m) {
            float d = lane < NWARPS ? partial_d[m][lane] : 0.f;
            float b = lane < NWARPS ? partial_m[m][lane] : 0.f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                d += __shfl_xor_sync(0xffffffff, d, offset);
                b += __shfl_xor_sync(0xffffffff, b, offset);
            }
            if (lane == 0) reinterpret_cast<__half*>(weight.out)[(size_t)m * weight.n + row] =
                __float2half(weight.neuron_scale[row] * d - weight.neuron_min[row] * b);
        }
    }
}

void launch_nint4_gs24_small_m_reuse(
    Nint4Gs24Projection weight, const int8_t* qx, const float* xs,
    int m, int ng, int kpad, cudaStream_t stream)
{
#define MFQ_NINT4_SMALL_M_CASE(M) \
    case M: nint4_gs24_small_m_reuse_kernel<M><<<weight.n, 128, 0, stream>>>(weight, qx, xs, ng, kpad); break
    switch (m) {
        MFQ_NINT4_SMALL_M_CASE(2);
        MFQ_NINT4_SMALL_M_CASE(3);
        MFQ_NINT4_SMALL_M_CASE(4);
        MFQ_NINT4_SMALL_M_CASE(5);
        MFQ_NINT4_SMALL_M_CASE(6);
        default: MFQ_RUNTIME_CHECK(false, "NINT4 small-M reuse requires M2-6");
    }
#undef MFQ_NINT4_SMALL_M_CASE
}
