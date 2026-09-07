#include "mfq_tensor_backend.h"
#include "nint_small_m.h"
#include "reduce.cuh"
#include <climits>

template <int BITS, int GS, int MROWS>
__global__ void __launch_bounds__(128) nint_group4_small_m_kernel(
    NintSmallMProjection weight, const int8_t* __restrict__ qx,
    const float* __restrict__ xs, int ng, int kpad)
{
    constexpr int CHUNKS = GS / 4, GPW = 32 / CHUNKS;
    constexpr int QBYTES = (GS * BITS + 7) / 8;
    static_assert((BITS == 2 && GS == 16) || (BITS == 3 && GS == 24) || (BITS == 5 && GS == 28));
    const int row = blockIdx.x * 4 + threadIdx.y, lane = threadIdx.x;
    if (row >= weight.n) return;
    const int relg = lane / CHUNKS, chunk = lane % CHUNKS;
    const auto* qrow = weight.q_packed + (size_t)row * ng * QBYTES;
    const auto* ssrow = weight.sub_scale + (size_t)row * ng;
    const auto* smrow = weight.sub_min + (size_t)row * ng;
    const float ns = weight.neuron_scale[row], nm = weight.neuron_min[row];
    float acc[MROWS] = {};
    for (int gb = 0; gb < ng; gb += GPW) {
        const int g = gb + relg;
        if (relg >= GPW || g >= ng) continue;
        const auto* packed = qrow + (size_t)g * QBYTES;
        int qv;
        if constexpr (BITS == 2) {
            const unsigned v = packed[chunk];
            qv = int((v & 3u) | ((v & 12u) << 6) | ((v & 48u) << 12) | ((v & 192u) << 18));
        } else if constexpr (BITS == 3) {
            // Four 3-bit values occupy 12 bits. Both nibble alignments fit
            // exactly two bytes, including the last chunk of a GS24 group.
            const int byte = chunk * 3 / 2;
            const unsigned v = (unsigned(packed[byte]) | (unsigned(packed[byte + 1]) << 8))
                >> ((chunk & 1) * 4);
            qv = int((v & 7u) | ((v & 56u) << 5) | ((v & 448u) << 10) | ((v & 3584u) << 15));
        } else {
            // Four 5-bit values fit three bytes, including GS28's final chunk.
            const int byte = chunk * 5 / 2;
            const unsigned v = (unsigned(packed[byte]) | (unsigned(packed[byte + 1]) << 8)
                | (unsigned(packed[byte + 2]) << 16)) >> ((chunk & 1) * 4);
            qv = int((v & 31u) | ((v & 992u) << 3) | ((v & 31744u) << 6)
                | ((v & 1015808u) << 9));
        }
        const float ss = ssrow[g], sm = smrow[g];
        #pragma unroll
        for (int m = 0; m < MROWS; ++m) {
            // CUDA storage, Kpad and g*GS+chunk*4 are all 4-byte aligned.
            const int xv = *reinterpret_cast<const int*>(qx + (size_t)m * kpad + g * GS + chunk * 4);
            const int di = __dp4a(qv, xv, 0), mi = __dp4a(0x01010101, xv, 0);
            // Keep the original group4 lane assignment, expression and sum tree.
            acc[m] += xs[(size_t)m * ng + g] * (ns * ss * float(di) - nm * sm * float(mi));
        }
    }
    #pragma unroll
    for (int m = 0; m < MROWS; ++m) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
        if (lane == 0)
            reinterpret_cast<__half*>(weight.out)[(size_t)m * weight.n + row] = __float2half(acc[m]);
    }
}

void launch_nint23_group4_small_m(
    NintSmallMProjection weight, const int8_t* qx, const float* xs,
    int bits, int m, int ng, int kpad, cudaStream_t stream)
{
#define MFQ_NINT23_CASE(M) \
    case M: \
        if (bits == 2) nint_group4_small_m_kernel<2, 16, M> \
            <<<dim3((weight.n + 3) / 4), dim3(32, 4), 0, stream>>>(weight, qx, xs, ng, kpad); \
        else nint_group4_small_m_kernel<3, 24, M> \
            <<<dim3((weight.n + 3) / 4), dim3(32, 4), 0, stream>>>(weight, qx, xs, ng, kpad); \
        break
    MFQ_RUNTIME_CHECK(bits == 2 || bits == 3, "NINT23 small-M bits must be 2 or 3");
    switch (m) {
        MFQ_NINT23_CASE(2);
        MFQ_NINT23_CASE(3);
        MFQ_NINT23_CASE(4);
        MFQ_NINT23_CASE(5);
        MFQ_NINT23_CASE(6);
        default: MFQ_RUNTIME_CHECK(false, "NINT23 small-M requires M2-6");
    }
#undef MFQ_NINT23_CASE
}

void launch_nint5_gs28_small_m(
    NintSmallMProjection weight, const int8_t* qx, const float* xs,
    int m, int ng, int kpad, cudaStream_t stream)
{
#define MFQ_NINT5_CASE(M) \
    case M: nint_group4_small_m_kernel<5, 28, M> \
        <<<dim3((weight.n + 3) / 4), dim3(32, 4), 0, stream>>>(weight, qx, xs, ng, kpad); break
    switch (m) {
        MFQ_NINT5_CASE(2);
        MFQ_NINT5_CASE(3);
        MFQ_NINT5_CASE(4);
        MFQ_NINT5_CASE(5);
        MFQ_NINT5_CASE(6);
        default: MFQ_RUNTIME_CHECK(false, "NINT5 GS28 small-M requires M2-6");
    }
#undef MFQ_NINT5_CASE
}

template <int MROWS, bool SPLIT_M = false>
__global__ void __launch_bounds__(128) nint8_gs48_small_m_kernel(
    NintSmallMProjection weight, const int8_t* __restrict__ qx,
    const float* __restrict__ xs, int ng, int kpad)
{
    const int row = blockIdx.x * 4 + threadIdx.y, lane = threadIdx.x;
    if (row >= weight.n) return;
    const auto* qrow = weight.q_packed + (size_t)row * ng * 48;
    const auto* ssrow = weight.sub_scale + (size_t)row * ng;
    const auto* smrow = weight.sub_min + (size_t)row * ng;
    const float ns = weight.neuron_scale[row], nm = weight.neuron_min[row];
    // Splitting the independent token dimension increases the grid without
    // changing any output's K order or its warp reduction.
    constexpr int LOCAL_M = SPLIT_M ? 1 : MROWS;
    const int m_base = SPLIT_M ? blockIdx.y : 0;
    float acc[LOCAL_M] = {};
    for (int base = lane * 4; base < kpad; base += 128) {
        const uint32_t qv = *reinterpret_cast<const uint32_t*>(qrow + base);
        const int g = base / 48;
        const float de = ns * float(ssrow[g]), me = nm * float(smrow[g]);
        #pragma unroll
        for (int m = 0; m < LOCAL_M; ++m) {
            const int xv = *reinterpret_cast<const int*>(qx + (size_t)(m_base + m) * kpad + base);
            int di;
            // The full unsigned-weight dot is exactly the old signed dot plus
            // 128*xsum. Its four-term integer range is exact in FP32.
            asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(di) : "r"(qv), "r"(xv), "r"(0));
            const int sumi = __dp4a(0x01010101, xv, 0);
            acc[m] += xs[(size_t)(m_base + m) * ng + g] * (de * float(di) - me * float(sumi));
        }
    }
    #pragma unroll
    for (int m = 0; m < LOCAL_M; ++m) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
        if (lane == 0)
            reinterpret_cast<__half*>(weight.out)[(size_t)(m_base + m) * weight.n + row] = __float2half(acc[m]);
    }
}

void launch_nint8_gs48_small_m(
    NintSmallMProjection weight, const int8_t* qx, const float* xs,
    int m, int ng, int kpad, cudaStream_t stream)
{
    const char* split_env = std::getenv("MFQ_NINT8_GS48_SMALL_M_SPLIT_M");
    if (split_env != nullptr && split_env[0] == '1') {
        MFQ_RUNTIME_CHECK(m >= 2 && m <= 6, "NINT8 GS48 split-M requires M2-6");
        nint8_gs48_small_m_kernel<1, true>
            <<<dim3((weight.n + 3) / 4, m), dim3(32, 4), 0, stream>>>(weight, qx, xs, ng, kpad);
        return;
    }
#define MFQ_NINT8_CASE(M) \
    case M: nint8_gs48_small_m_kernel<M> \
        <<<dim3((weight.n + 3) / 4), dim3(32, 4), 0, stream>>>(weight, qx, xs, ng, kpad); break
    switch (m) {
        MFQ_NINT8_CASE(2);
        MFQ_NINT8_CASE(3);
        MFQ_NINT8_CASE(4);
        MFQ_NINT8_CASE(5);
        MFQ_NINT8_CASE(6);
        default: MFQ_RUNTIME_CHECK(false, "NINT8 GS48 small-M requires M2-6");
    }
#undef MFQ_NINT8_CASE
}

__device__ __forceinline__ int small_m_unpack_int4(uint32_t packed, unsigned selector)
{
    // Interleave low/high nibbles into signed dp4a's positive byte lanes.
    return (int)(__byte_perm(packed, packed >> 4, selector) & 0x0f0f0f0fu);
}

template <int MROWS, bool FLOAT_OUTPUT = false, bool PRECOMPUTED_SUM = false, int BITS = 4>
__global__ void __launch_bounds__(128) nint_gs24_small_m_reuse_kernel(
    Nint4Gs24Projection weight,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    int ng, int kpad)
{
    static_assert(BITS == 4 || BITS == 6);
    constexpr int NWARPS = 4, QBYTES = 24 * BITS / 8;
    const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (row >= weight.n) return;
    const auto* qrow = weight.q_packed + (size_t)row * ng * QBYTES;
    const auto* ssrow = weight.sub_scale + (size_t)row * ng;
    const auto* smrow = weight.sub_min + (size_t)row * ng;
    float pd[MROWS] = {}, pm[MROWS] = {};
    for (int g = warp * 32 + lane; g < ng; g += NWARPS * 32) {
        const auto* qgroup = qrow + g * QBYTES;
        uint32_t qw0 = 0, qw1 = 0, qw2 = 0;
        uint16_t q6[9];
        if constexpr (BITS == 4) {
            const auto* qwords = reinterpret_cast<const uint32_t*>(qgroup);
            qw0 = qwords[0]; qw1 = qwords[1]; qw2 = qwords[2];
        } else {
            const auto* qwords = reinterpret_cast<const uint16_t*>(qgroup);
            #pragma unroll
            for (int i = 0; i < 9; ++i) q6[i] = qwords[i];
        }
        int dsum[MROWS] = {}, msum[MROWS] = {};
        #pragma unroll
        for (int chunk = 0; chunk < 6; ++chunk) {
            int qv;
            if constexpr (BITS == 4) {
                const uint32_t qw = chunk < 2 ? qw0 : (chunk < 4 ? qw1 : qw2);
                qv = small_m_unpack_int4(qw, (chunk & 1) ? 0x7362u : 0x5140u);
            } else {
                const uint32_t w0 = q6[(chunk / 2) * 3], w1 = q6[(chunk / 2) * 3 + 1];
                const uint32_t w2 = q6[(chunk / 2) * 3 + 2];
                const uint32_t packed = (chunk & 1) ? ((w1 >> 8) | (w2 << 8))
                    : (w0 | ((w1 & 255u) << 16));
                qv = int((packed & 63u) | ((packed & 4032u) << 2)
                    | ((packed & 258048u) << 4) | ((packed & 16515072u) << 6));
            }
            #pragma unroll
            for (int m = 0; m < MROWS; ++m) {
                const int xv = *reinterpret_cast<const int*>(qx + (size_t)m * kpad + g * 24 + chunk * 4);
                dsum[m] = __dp4a(qv, xv, dsum[m]);
                if constexpr (!PRECOMPUTED_SUM)
                    msum[m] = __dp4a(0x01010101, xv, msum[m]);
            }
        }
        const float ss = ssrow[g], sm = smrow[g];
        #pragma unroll
        for (int m = 0; m < MROWS; ++m) {
            const float xs = xscale[(size_t)m * ng + g];
            pd[m] += xs * ss * (float)dsum[m];
            // The exact GS24 integer sum is shared by every output row.
            const int sum = PRECOMPUTED_SUM ? weight.xsum[(size_t)m * ng + g] : msum[m];
            pm[m] += xs * sm * (float)sum;
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

template <bool FLOAT_OUTPUT, int BITS = 4>
static void launch_small_m(
    Nint4Gs24Projection weight, const int8_t* qx, const float* xs,
    int m, int ng, int kpad, cudaStream_t stream)
{
#define MFQ_NINT4_SMALL_M_CASE(M) \
    case M: \
        if (weight.xsum != nullptr) nint_gs24_small_m_reuse_kernel<M, FLOAT_OUTPUT, true, BITS> \
            <<<weight.n, 128, 0, stream>>>(weight, qx, xs, ng, kpad); \
        else nint_gs24_small_m_reuse_kernel<M, FLOAT_OUTPUT, false, BITS> \
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

void launch_nint6_gs24_small_m_reuse(
    NintSmallMProjection weight, const int8_t* qx, const float* xs,
    int m, int ng, int kpad, cudaStream_t stream)
{
    MFQ_RUNTIME_CHECK(weight.xsum == nullptr, "NINT6 small-M keeps the original integer reduction");
#define MFQ_NINT6_CASE(M) \
    case M: nint_gs24_small_m_reuse_kernel<M, false, false, 6> \
        <<<weight.n, 128, 0, stream>>>(weight, qx, xs, ng, kpad); break
    switch (m) {
        MFQ_NINT6_CASE(2);
        MFQ_NINT6_CASE(3);
        MFQ_NINT6_CASE(4);
        MFQ_NINT6_CASE(5);
        MFQ_NINT6_CASE(6);
        default: MFQ_RUNTIME_CHECK(false, "NINT6 GS24 small-M requires M2-6");
    }
#undef MFQ_NINT6_CASE
}

__global__ void small_m_quantize_f32_half_rn_kernel(
    const float* __restrict__ input, int8_t* __restrict__ qx,
    float* __restrict__ scales, int32_t* __restrict__ sums, int k, int kpad)
{
    const int m = blockIdx.x, g = blockIdx.y, lane = threadIdx.x;
    const bool valid = lane < 24 && g * 24 + lane < k;
    const float x = valid ? __half2float(__float2half_rn(input[(size_t)m * k + g * 24 + lane])) : 0.f;
    const float amax = block_max<1>(fabsf(x));
    const float scale = amax > 0.f ? amax / 127.f : 1.f;
    int code = 0;
    if (valid) code = (int)fminf(fmaxf(roundf(x / scale), -127.f), 127.f);
    if (sums != nullptr) {
        const int sum = (int)block_sum<1>((float)code);
        if (lane == 0) sums[(size_t)m * gridDim.y + g] = sum;
    }
    if (lane == 0) scales[(size_t)m * gridDim.y + g] = scale;
    if (lane < 24) qx[(size_t)m * kpad + g * 24 + lane] = (int8_t)code;
}

template <int BITS>
static mfq_tensor_backend::Tensor nint_gs24_small_m_f32_ws_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor s, mfq_tensor_backend::Tensor sm,
    mfq_tensor_backend::Tensor ns, mfq_tensor_backend::Tensor nm, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xs, mfq_tensor_backend::Tensor xm)
{
    using namespace mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == kFloat32 && x.dim() == 2,
        "NINT GS24 small-M F32 boundary requires contiguous CUDA F32 [M,K]");
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == kUInt8 &&
        q.dim() == 3 && q.size(2) == 24 * BITS / 8 && q.size(0) > 0 && q.size(1) > 0,
        "NINT GS24 small-M F32 weights require canonical packed GS24 groups");
    const int64_t rows = x.size(0), columns = q.size(0), groups = q.size(1);
    MFQ_RUNTIME_CHECK(rows >= 2 && rows <= 6 && x.size(1) > 0 &&
        groups <= INT_MAX / 24 && columns <= INT_MAX && x.size(1) <= groups * 24,
        "NINT GS24 small-M F32 geometry exceeds supported bounds");
    for (const auto* value : {&q, &s, &sm, &ns, &nm, &qx, &xs, &xm}) {
        MFQ_RUNTIME_CHECK(value->is_cuda() && value->device() == x.device() && value->is_contiguous(),
            "NINT GS24 small-M F32 tensors must be contiguous on one CUDA device");
    }
    for (const auto* value : {&s, &sm}) MFQ_RUNTIME_CHECK(value->scalar_type() == kUInt8 &&
        value->dim() == 2 && value->size(0) == columns && value->size(1) == groups,
        "NINT GS24 small-M F32 submetadata shape or dtype mismatch");
    for (const auto* value : {&ns, &nm}) MFQ_RUNTIME_CHECK(value->scalar_type() == kFloat32 &&
        value->dim() == 1 && value->size(0) == columns,
        "NINT GS24 small-M F32 neuron metadata shape or dtype mismatch");
    MFQ_RUNTIME_CHECK(qx.scalar_type() == kInt8 && qx.dim() == 2 &&
        qx.size(0) >= rows && qx.size(1) == groups * 24 &&
        xs.scalar_type() == kFloat32 && xs.dim() == 2 && xs.size(0) >= rows && xs.size(1) == groups &&
        xm.scalar_type() == kInt32 && xm.dim() == 2 && xm.size(0) >= rows && xm.size(1) == groups,
        "NINT GS24 small-M F32 workspace shape or dtype mismatch");
    MfqCudaGuard guard(x.device());
    auto output = mfq_tensor_backend::empty({rows, columns}, x.options());
    auto stream = mfq_current_cuda_stream();
    const char* sum_env = std::getenv("MFQ_NINT4_SMALL_M_XSUM");
    auto* sums = BITS == 4 && sum_env != nullptr && sum_env[0] == '1' ? xm.data_ptr<int32_t>() : nullptr;
    small_m_quantize_f32_half_rn_kernel<<<dim3(rows, groups), 32, 0, stream>>>(
        x.data_ptr<float>(), qx.data_ptr<int8_t>(), xs.data_ptr<float>(), sums, (int)x.size(1), (int)(groups * 24));
    Nint4Gs24Projection weight{q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), sm.data_ptr<uint8_t>(),
        ns.data_ptr<float>(), nm.data_ptr<float>(), output.data_ptr<float>(), (int)columns, sums};
    launch_small_m<true, BITS>(weight, qx.data_ptr<int8_t>(), xs.data_ptr<float>(),
        (int)rows, (int)groups, (int)(groups * 24), stream);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

mfq_tensor_backend::Tensor nint4_gs24_small_m_f32_ws_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor s, mfq_tensor_backend::Tensor sm,
    mfq_tensor_backend::Tensor ns, mfq_tensor_backend::Tensor nm, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xs, mfq_tensor_backend::Tensor xm)
{
    return nint_gs24_small_m_f32_ws_cuda<4>(q, s, sm, ns, nm, x, qx, xs, xm);
}

mfq_tensor_backend::Tensor nint6_gs24_small_m_f32_ws_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor s, mfq_tensor_backend::Tensor sm,
    mfq_tensor_backend::Tensor ns, mfq_tensor_backend::Tensor nm, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xs, mfq_tensor_backend::Tensor xm)
{
    return nint_gs24_small_m_f32_ws_cuda<6>(q, s, sm, ns, nm, x, qx, xs, xm);
}
