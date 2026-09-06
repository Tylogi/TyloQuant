#include "mfq_tensor_backend.h"
#include "nint_small_m.h"
#include "reduce.cuh"
#include <climits>

__device__ __forceinline__ int small_m_unpack_int4(uint32_t packed)
{
    return ((int)(packed & 0x000f)) |
           ((int)(packed & 0x00f0) << 4) |
           ((int)(packed & 0x0f00) << 8) |
           ((int)(packed & 0xf000) << 12);
}

template <int MROWS, bool FLOAT_OUTPUT = false>
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
            if (lane == 0) {
                const __half rounded = __float2half(weight.neuron_scale[row] * d - weight.neuron_min[row] * b);
                if constexpr (FLOAT_OUTPUT)
                    reinterpret_cast<float*>(weight.out)[(size_t)m * weight.n + row] = __half2float(rounded);
                else reinterpret_cast<__half*>(weight.out)[(size_t)m * weight.n + row] = rounded;
            }
        }
    }
}

// One warp owns an output row while retaining the original four-part sum tree.
template <int MROWS, bool FLOAT_OUTPUT>
__global__ void __launch_bounds__(128) nint4_gs24_small_m_warp_rows_kernel(
    Nint4Gs24Projection weight, const int8_t* __restrict__ qx,
    const float* __restrict__ xscale, int ng, int kpad)
{
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= weight.n) return;
    const auto* qrow = weight.q_packed + (size_t)row * ng * 12;
    const auto* ssrow = weight.sub_scale + (size_t)row * ng;
    const auto* smrow = weight.sub_min + (size_t)row * ng;
    float partial_d[4][MROWS], partial_m[4][MROWS];
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
        float pd[MROWS] = {}, pm[MROWS] = {};
        for (int g = part * 32 + lane; g < ng; g += 128) {
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
        #pragma unroll
        for (int m = 0; m < MROWS; ++m) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                pd[m] += __shfl_xor_sync(0xffffffff, pd[m], offset);
                pm[m] += __shfl_xor_sync(0xffffffff, pm[m], offset);
            }
            partial_d[part][m] = pd[m];
            partial_m[part][m] = pm[m];
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < MROWS; ++m) {
            const float d = (partial_d[0][m] + partial_d[2][m]) + (partial_d[1][m] + partial_d[3][m]);
            const float b = (partial_m[0][m] + partial_m[2][m]) + (partial_m[1][m] + partial_m[3][m]);
            const __half rounded = __float2half(weight.neuron_scale[row] * d - weight.neuron_min[row] * b);
            if constexpr (FLOAT_OUTPUT)
                reinterpret_cast<float*>(weight.out)[(size_t)m * weight.n + row] = __half2float(rounded);
            else reinterpret_cast<__half*>(weight.out)[(size_t)m * weight.n + row] = rounded;
        }
    }
}

template <bool FLOAT_OUTPUT>
static void launch_small_m(
    Nint4Gs24Projection weight, const int8_t* qx, const float* xs,
    int m, int ng, int kpad, cudaStream_t stream)
{
    const char* warp_rows_env = std::getenv("MFQ_NINT4_SMALL_M_WARP_ROWS");
    const bool warp_rows = warp_rows_env != nullptr && warp_rows_env[0] == '1';
#define MFQ_NINT4_SMALL_M_CASE(M) \
    case M: \
        if (warp_rows) nint4_gs24_small_m_warp_rows_kernel<M, FLOAT_OUTPUT> \
            <<<weight.n / 4 + (weight.n % 4 != 0), 128, 0, stream>>>(weight, qx, xs, ng, kpad); \
        else nint4_gs24_small_m_reuse_kernel<M, FLOAT_OUTPUT> \
            <<<weight.n, 128, 0, stream>>>(weight, qx, xs, ng, kpad); \
        break
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

void launch_nint4_gs24_small_m_reuse(
    Nint4Gs24Projection weight, const int8_t* qx, const float* xs,
    int m, int ng, int kpad, cudaStream_t stream)
{
    launch_small_m<false>(weight, qx, xs, m, ng, kpad, stream);
}

__global__ void small_m_quantize_f32_half_rn_kernel(
    const float* __restrict__ input, int8_t* __restrict__ qx,
    float* __restrict__ scales, int k, int kpad)
{
    const int m = blockIdx.x, g = blockIdx.y, lane = threadIdx.x;
    const bool valid = lane < 24 && g * 24 + lane < k;
    const float x = valid ? __half2float(__float2half_rn(input[(size_t)m * k + g * 24 + lane])) : 0.f;
    const float amax = block_max<1>(fabsf(x));
    const float scale = amax > 0.f ? amax / 127.f : 1.f;
    int code = 0;
    if (valid) code = (int)fminf(fmaxf(roundf(x / scale), -127.f), 127.f);
    if (lane == 0) scales[(size_t)m * gridDim.y + g] = scale;
    if (lane < 24) qx[(size_t)m * kpad + g * 24 + lane] = (int8_t)code;
}

mfq_tensor_backend::Tensor nint4_gs24_small_m_f32_ws_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor s, mfq_tensor_backend::Tensor sm,
    mfq_tensor_backend::Tensor ns, mfq_tensor_backend::Tensor nm, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xs)
{
    using namespace mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == kFloat32 && x.dim() == 2,
        "NINT4 small-M F32 boundary requires contiguous CUDA F32 [M,K]");
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == kUInt8 &&
        q.dim() == 3 && q.size(2) == 12 && q.size(0) > 0 && q.size(1) > 0,
        "NINT4 small-M F32 weights require packed GS24 [N,ng,12]");
    const int64_t rows = x.size(0), columns = q.size(0), groups = q.size(1);
    MFQ_RUNTIME_CHECK(rows >= 2 && rows <= 6 && x.size(1) > 0 &&
        groups <= INT_MAX / 24 && columns <= INT_MAX && x.size(1) <= groups * 24,
        "NINT4 small-M F32 geometry exceeds supported bounds");
    for (const auto* value : {&q, &s, &sm, &ns, &nm, &qx, &xs}) {
        MFQ_RUNTIME_CHECK(value->is_cuda() && value->device() == x.device() && value->is_contiguous(),
            "NINT4 small-M F32 tensors must be contiguous on one CUDA device");
    }
    for (const auto* value : {&s, &sm}) MFQ_RUNTIME_CHECK(value->scalar_type() == kUInt8 &&
        value->dim() == 2 && value->size(0) == columns && value->size(1) == groups,
        "NINT4 small-M F32 submetadata shape or dtype mismatch");
    for (const auto* value : {&ns, &nm}) MFQ_RUNTIME_CHECK(value->scalar_type() == kFloat32 &&
        value->dim() == 1 && value->size(0) == columns,
        "NINT4 small-M F32 neuron metadata shape or dtype mismatch");
    MFQ_RUNTIME_CHECK(qx.scalar_type() == kInt8 && qx.dim() == 2 &&
        qx.size(0) >= rows && qx.size(1) == groups * 24 &&
        xs.scalar_type() == kFloat32 && xs.dim() == 2 && xs.size(0) >= rows && xs.size(1) == groups,
        "NINT4 small-M F32 workspace shape or dtype mismatch");
    MfqCudaGuard guard(x.device());
    auto output = mfq_tensor_backend::empty({rows, columns}, x.options());
    auto stream = mfq_current_cuda_stream();
    small_m_quantize_f32_half_rn_kernel<<<dim3(rows, groups), 32, 0, stream>>>(
        x.data_ptr<float>(), qx.data_ptr<int8_t>(), xs.data_ptr<float>(), (int)x.size(1), (int)(groups * 24));
    Nint4Gs24Projection weight{q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), sm.data_ptr<uint8_t>(),
        ns.data_ptr<float>(), nm.data_ptr<float>(), output.data_ptr<float>(), (int)columns};
    launch_small_m<true>(weight, qx.data_ptr<int8_t>(), xs.data_ptr<float>(),
        (int)rows, (int)groups, (int)(groups * 24), stream);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
