// NINT INT-fused-GEMM (dequant-during-GEMM, no Wq materialization).
//
//   y[m,n] = sum_k w[n,k] * x[m,k], with w stored as two-level nint INT, dequantized per
//   group inside the GEMM:
//     w[n,k] = d_eff[n,g] * q[n,g,i] - m_eff[n,g],   g = k/gs, i = k%gs
//     d_eff  = neuron_scale[n] * sub_scale[n,g]
//     m_eff  = neuron_min[n]   * sub_min[n,g]
//
// Tile: each block computes a [BN neurons x BM rows] output tile; X tile (fp16) and q tile
// (u8) live in shared memory, reloaded once per group, dequant fused with the MAC. fp32
// accumulate, fp16 output. Matches llama.cpp mul_mat_q's "compressed weights resident,
// activations streamed" memory strategy: no full fp16 Wq materialization.
//
// Tensor layout (matches mfq.kernels.torch_backend.to_gpu):
//   q[N, ng, gs] uint8 ; sub_scale/sub_min[N, ng] uint8 ; neuron_scale/min[N] f32
//   x[M, neuron_len] fp16 (glue zero-pads to neuron_len).

#include <cuda_runtime.h>
#include "../../../cpp_runtime/cuda/mfq_tensor_backend.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <vector>

#include "reduce.cuh"
#include "glu.cuh"
using namespace nvcuda;

#define MFQ_CUBLAS_CHECK(expr) MFQ_RUNTIME_CHECK((expr) == CUBLAS_STATUS_SUCCESS, "cuBLAS call failed: ", #expr)

__global__ void nint8_one_quantize_reconstruct_kernel(
    const __half* __restrict__ x,
    int8_t* __restrict__ q,
    __half* __restrict__ d_out,
    __half* __restrict__ s_out,
    __half* __restrict__ reconstructed,
    int M,
    int K,
    int groups)
{
    const int group_index = (int)blockIdx.x;
    if (group_index >= M * groups) {
        return;
    }
    const int lane = (int)threadIdx.x;
    const int row = group_index / groups;
    const int group = group_index - row * groups;
    const int k = group * 32 + lane;
    const float value = k < K
        ? __half2float(x[(size_t)row * K + k])
        : 0.0f;
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(
            amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    const float scale = amax / 127.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    int code = (int)roundf(value * inverse);
    code = max(-127, min(127, code));
    q[(size_t)group_index * 32 + lane] = (int8_t)code;

    int sum = code;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    const __half stored_scale = __float2half_rn(scale);
    if (lane == 0) {
        d_out[group_index] = stored_scale;
        s_out[group_index] = __float2half_rn((float)sum * scale);
    }
    if (k < K) {
        reconstructed[(size_t)row * K + k] =
            __float2half_rn((float)code * __half2float(stored_scale));
    }
}

std::vector<mfq_tensor_backend::Tensor> nint8_one_quantize_reconstruct_cuda(
    mfq_tensor_backend::Tensor x)
{
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
        x.is_contiguous() && x.dim() == 2,
        "NINT8-1 input must be CUDA contiguous fp16 rank-2");
    const int M = (int)x.size(0);
    const int K = (int)x.size(1);
    MFQ_RUNTIME_CHECK(M > 0 && K > 0, "NINT8-1 input must be non-empty");
    const int groups = (K + 31) / 32;
    auto q = mfq_tensor_backend::empty(
        {M, groups, 32}, x.options().dtype(mfq_tensor_backend::kInt8));
    auto d = mfq_tensor_backend::empty({M, groups}, x.options());
    auto s = mfq_tensor_backend::empty({M, groups}, x.options());
    auto reconstructed = mfq_tensor_backend::empty_like(x);
    cudaStream_t stream = mfq_current_cuda_stream();
    nint8_one_quantize_reconstruct_kernel<<<
        M * groups, 32, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
        q.data_ptr<int8_t>(),
        reinterpret_cast<__half*>(d.data_ptr<mfq_half>()),
        reinterpret_cast<__half*>(s.data_ptr<mfq_half>()),
        reinterpret_cast<__half*>(
            reconstructed.data_ptr<mfq_half>()),
        M, K, groups);
    MFQ_RUNTIME_CHECK(
        cudaGetLastError() == cudaSuccess,
        "NINT8-1 quantize/reconstruct kernel launch failed");
    return {q, d, s, reconstructed};
}

template <int BITS>
__device__ __forceinline__ uint8_t unpack_qbits_one_dequant(const uint8_t* p, int lane)
{
    if constexpr (BITS == 8) {
        return p[lane];
    } else {
        constexpr uint32_t MASK = (1u << BITS) - 1u;
        int bit = lane * BITS;
        int byte = bit >> 3;
        int shift = bit & 7;
        uint32_t word = (uint32_t)p[byte];
        if (shift + BITS > 8) {
            word |= ((uint32_t)p[byte + 1] << 8);
        }
        return (uint8_t)((word >> shift) & MASK);
    }
}

template <int BITS, int GS>
__global__ void dequant_full_packed_compact_bits_kernel_early(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    __half* __restrict__ w,
    int N, int ng, int neuron_len)
{
    constexpr int QBYTES = (GS * BITS + 7) / 8;
    size_t total = (size_t)N * neuron_len;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        int k = (int)(idx % neuron_len);
        int n = (int)(idx / neuron_len);
        int g = k / GS;
        int gi = k - g * GS;
        size_t meta_idx = (size_t)n * ng + g;
        uint8_t qv = unpack_qbits_one_dequant<BITS>(q_packed + meta_idx * QBYTES, gi);
        float de = neuron_scale[n] * (float)sub_scale[meta_idx];
        float me = neuron_min[n] * (float)sub_min[meta_idx];
        w[idx] = __float2half(de * (float)qv - me);
    }
}

__device__ __forceinline__ uint8_t unpack_q5_gs28_exec_one(const uint8_t* p, int lane)
{
    const uint8_t low = (p[lane >> 1] >> ((lane & 1) * 4)) & 0x0f;
    const int chunk = lane >> 3;
    const int within = lane & 7;
    const int qh_byte = within >> 1;
    const int qh_bit = chunk + ((within & 1) ? 4 : 0);
    const uint8_t high = (p[14 + qh_byte] >> qh_bit) & 1u;
    return low | (high << 4);
}

__global__ void dequant_full_q5_gs28_exec_kernel(
    const uint8_t* __restrict__ q_packed,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    __half* __restrict__ w,
    int N, int ng, int neuron_len)
{
    constexpr int GS = 28;
    const size_t total = (size_t)N * neuron_len;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int k = (int)(idx % neuron_len);
        const int n = (int)(idx / neuron_len);
        const int g = k / GS;
        const int gi = k - g * GS;
        const uint8_t* qg = q_packed + ((size_t)n * ng + g) * 20;
        const uint8_t qv = unpack_q5_gs28_exec_one(qg, gi);
        const float de = neuron_scale[n] * (float)qg[18];
        const float me = neuron_min[n] * (float)qg[19];
        w[idx] = __float2half(de * (float)qv - me);
    }
}

__global__ void repack_q5_gs28_exec_kernel(
    const uint8_t* __restrict__ src,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    uint8_t* __restrict__ dst,
    size_t groups)
{
    constexpr int GS = 28;
    constexpr int SRC_QBYTES = 18;
    constexpr int DST_QBYTES = 20;
    for (size_t group = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         group < groups;
         group += (size_t)gridDim.x * blockDim.x) {
        const uint8_t* in = src + group * SRC_QBYTES;
        uint8_t* out = dst + group * DST_QBYTES;
        uint32_t qh = 0;
#pragma unroll
        for (int i = 0; i < GS; i += 2) {
            const uint8_t q0 = unpack_qbits_one_dequant<5>(in, i);
            const uint8_t q1 = unpack_qbits_one_dequant<5>(in, i + 1);
            out[i >> 1] = (q0 & 0x0f) | ((q1 & 0x0f) << 4);

            const int chunk = i >> 3;
            const int pair = (i & 7) >> 1;
            qh |= (uint32_t)(q0 >> 4) << (8 * pair + chunk);
            qh |= (uint32_t)(q1 >> 4) << (8 * pair + 4 + chunk);
        }
        out[14] = (uint8_t)qh;
        out[15] = (uint8_t)(qh >> 8);
        out[16] = (uint8_t)(qh >> 16);
        out[17] = (uint8_t)(qh >> 24);
        out[18] = sub_scale[group];
        out[19] = sub_min[group];
    }
}

template <int GS>
__global__ void dequant_wq_packed_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, GS/2]
    const float* __restrict__ d_eff,         // [N, ng]
    __half* __restrict__ wq,                 // [N, neuron_len]
    int N, int ng, int neuron_len)
{
    bool aligned_pairs = (neuron_len & 1) == 0;
    int qbytes = GS / 2;
    int total = N * ng * qbytes;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int b = idx % qbytes;
        int t = idx / qbytes;
        int g = t % ng;
        int n = t / ng;
        int k = g * GS + b * 2;
        if (k >= neuron_len) {
            continue;
        }
        uint8_t qv = q_packed[(size_t)n * ng * qbytes + g * qbytes + b];
        float de = d_eff[(size_t)n * ng + g];
        __half lo = __float2half(de * (float)(qv & 0x0f));
        if (k + 1 < neuron_len) {
            __half hi = __float2half(de * (float)(qv >> 4));
            if (aligned_pairs) {
                *reinterpret_cast<__half2*>(wq + (size_t)n * neuron_len + k) = __halves2half2(lo, hi);
            } else {
                wq[(size_t)n * neuron_len + k] = lo;
                wq[(size_t)n * neuron_len + k + 1] = hi;
            }
        } else {
            wq[(size_t)n * neuron_len + k] = lo;
        }
    }
}

mfq_tensor_backend::Tensor nint_dequant_wq_packed_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor d_eff, int64_t neuron_len, int64_t gs)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(d_eff.is_cuda() && d_eff.scalar_type() == mfq_tensor_backend::kFloat32 && d_eff.is_contiguous(),
                "d_eff must be cuda contiguous f32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    MFQ_RUNTIME_CHECK((int)d_eff.size(0) == N && (int)d_eff.size(1) == ng, "d_eff shape mismatch");
    MFQ_RUNTIME_CHECK(neuron_len <= (int64_t)ng * gs, "neuron_len must be <= ng * gs");
    auto wq = mfq_tensor_backend::empty({N, neuron_len}, d_eff.options().dtype(mfq_tensor_backend::kHalf));
    cudaStream_t stream = mfq_current_cuda_stream();
    int total = N * ng * ((int)gs / 2);
    int block = 256;
    int grid = std::min((total + block - 1) / block, 65535);

#define DQWQLAUNCH(GSVAL)                                                           \
    dequant_wq_packed_kernel<GSVAL><<<grid, block, 0, stream>>>(                   \
        q_packed.data_ptr<uint8_t>(), d_eff.data_ptr<float>(),                     \
        reinterpret_cast<__half*>(wq.data_ptr<mfq_half>()), N, ng, (int)neuron_len)

    switch ((int)gs) {
        case 16: DQWQLAUNCH(16); break;
        case 24: DQWQLAUNCH(24); break;
        case 32: DQWQLAUNCH(32); break;
        case 48: DQWQLAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_dequant_wq_packed: gs must be in {16,24,32,48}, got ", gs);
    }
#undef DQWQLAUNCH
    return wq;
}

mfq_tensor_backend::Tensor nint_cublas_gemm_nt_f32acc_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor w)
{
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kHalf && x.is_contiguous(),
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(w.is_cuda() && w.scalar_type() == mfq_tensor_backend::kHalf && w.is_contiguous(),
                "w must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(x.dim() == 2 && w.dim() == 2, "x and w must be rank-2");
    int M = (int)x.size(0);
    int K = (int)x.size(1);
    int N = (int)w.size(0);
    MFQ_RUNTIME_CHECK((int)w.size(1) == K, "w K mismatch");

    auto y = mfq_tensor_backend::empty({M, N}, x.options());
    cublasHandle_t handle = mfq_current_cublas_handle();
    cudaStream_t stream = mfq_current_cuda_stream();
    MFQ_CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    MFQ_CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        w.data_ptr<mfq_half>(), CUDA_R_16F, K,
        x.data_ptr<mfq_half>(), CUDA_R_16F, K,
        &beta,
        y.data_ptr<mfq_half>(), CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return y;
}

mfq_tensor_backend::Tensor nint_cublas_gemm_nt_f16acc_cuda(mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor w)
{
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kHalf && x.is_contiguous(),
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(w.is_cuda() && w.scalar_type() == mfq_tensor_backend::kHalf && w.is_contiguous(),
                "w must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(x.dim() == 2 && w.dim() == 2, "x and w must be rank-2");
    int M = (int)x.size(0);
    int K = (int)x.size(1);
    int N = (int)w.size(0);
    MFQ_RUNTIME_CHECK((int)w.size(1) == K, "w K mismatch");

    auto y = mfq_tensor_backend::empty({M, N}, x.options());
    cublasHandle_t handle = mfq_current_cublas_handle();
    cudaStream_t stream = mfq_current_cuda_stream();
    MFQ_CUBLAS_CHECK(cublasSetStream(handle, stream));

    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    MFQ_CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        w.data_ptr<mfq_half>(), CUDA_R_16F, K,
        x.data_ptr<mfq_half>(), CUDA_R_16F, K,
        &beta,
        y.data_ptr<mfq_half>(), CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return y;
}

template <int GS>
__global__ void dequant_full_packed_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, GS/2]
    const float* __restrict__ d_eff,         // [N, ng]
    const float* __restrict__ m_eff,         // [N, ng]
    __half* __restrict__ w,                  // [N, neuron_len]
    int N, int ng, int neuron_len)
{
    bool aligned_pairs = (neuron_len & 1) == 0;
    int qbytes = GS / 2;
    int total = N * ng * qbytes;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int b = idx % qbytes;
        int t = idx / qbytes;
        int g = t % ng;
        int n = t / ng;
        int k = g * GS + b * 2;
        if (k >= neuron_len) {
            continue;
        }
        uint8_t qv = q_packed[(size_t)n * ng * qbytes + g * qbytes + b];
        float de = d_eff[(size_t)n * ng + g];
        float me = m_eff[(size_t)n * ng + g];
        __half lo = __float2half(de * (float)(qv & 0x0f) - me);
        if (k + 1 < neuron_len) {
            __half hi = __float2half(de * (float)(qv >> 4) - me);
            if (aligned_pairs) {
                *reinterpret_cast<__half2*>(w + (size_t)n * neuron_len + k) = __halves2half2(lo, hi);
            } else {
                w[(size_t)n * neuron_len + k] = lo;
                w[(size_t)n * neuron_len + k + 1] = hi;
            }
        } else {
            w[(size_t)n * neuron_len + k] = lo;
        }
    }
}

mfq_tensor_backend::Tensor nint_dequant_full_packed_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor d_eff, mfq_tensor_backend::Tensor m_eff, int64_t neuron_len, int64_t gs)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(d_eff.is_cuda() && d_eff.scalar_type() == mfq_tensor_backend::kFloat32 && d_eff.is_contiguous(),
                "d_eff must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(m_eff.is_cuda() && m_eff.scalar_type() == mfq_tensor_backend::kFloat32 && m_eff.is_contiguous(),
                "m_eff must be cuda contiguous f32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    MFQ_RUNTIME_CHECK((int)d_eff.size(0) == N && (int)d_eff.size(1) == ng, "d_eff shape mismatch");
    MFQ_RUNTIME_CHECK((int)m_eff.size(0) == N && (int)m_eff.size(1) == ng, "m_eff shape mismatch");
    MFQ_RUNTIME_CHECK(neuron_len <= (int64_t)ng * gs, "neuron_len must be <= ng * gs");
    auto w = mfq_tensor_backend::empty({N, neuron_len}, d_eff.options().dtype(mfq_tensor_backend::kHalf));
    cudaStream_t stream = mfq_current_cuda_stream();
    int total = N * ng * ((int)gs / 2);
    int block = 256;
    int grid = std::min((total + block - 1) / block, 65535);

#define DQFULLLAUNCH(GSVAL)                                                        \
    dequant_full_packed_kernel<GSVAL><<<grid, block, 0, stream>>>(                \
        q_packed.data_ptr<uint8_t>(), d_eff.data_ptr<float>(),                    \
        m_eff.data_ptr<float>(), reinterpret_cast<__half*>(w.data_ptr<mfq_half>()), \
        N, ng, (int)neuron_len)

    switch ((int)gs) {
        case 16: DQFULLLAUNCH(16); break;
        case 24: DQFULLLAUNCH(24); break;
        case 32: DQFULLLAUNCH(32); break;
        case 48: DQFULLLAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_dequant_full_packed: gs must be in {16,24,32,48}, got ", gs);
    }
#undef DQFULLLAUNCH
    return w;
}

template <int GS>
__global__ void dequant_full_packed_compact_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, GS/2]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    __half* __restrict__ w,                  // [N, neuron_len]
    int N, int ng, int neuron_len)
{
    bool aligned_pairs = (neuron_len & 1) == 0;
    int qbytes = GS / 2;
    int total = N * ng * qbytes;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int b = idx % qbytes;
        int t = idx / qbytes;
        int g = t % ng;
        int n = t / ng;
        int k = g * GS + b * 2;
        if (k >= neuron_len) {
            continue;
        }
        size_t meta_idx = (size_t)n * ng + g;
        uint8_t qv = q_packed[meta_idx * qbytes + b];
        float de = neuron_scale[n] * (float)sub_scale[meta_idx];
        float me = neuron_min[n] * (float)sub_min[meta_idx];
        __half lo = __float2half(de * (float)(qv & 0x0f) - me);
        if (k + 1 < neuron_len) {
            __half hi = __float2half(de * (float)(qv >> 4) - me);
            if (aligned_pairs) {
                *reinterpret_cast<__half2*>(w + (size_t)n * neuron_len + k) = __halves2half2(lo, hi);
            } else {
                w[(size_t)n * neuron_len + k] = lo;
                w[(size_t)n * neuron_len + k + 1] = hi;
            }
        } else {
            w[(size_t)n * neuron_len + k] = lo;
        }
    }
}

__global__ void dequant_full_packed_compact_gs24_vec2_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    __half* __restrict__ w,
    int N, int ng, int neuron_len)
{
    constexpr int GS = 24;
    constexpr int QBYTES = 12;
    constexpr int TASKS_PER_GROUP = 6;
    int total_tasks = N * ng * TASKS_PER_GROUP;
    for (int task = blockIdx.x * blockDim.x + threadIdx.x;
         task < total_tasks;
         task += blockDim.x * gridDim.x) {
        int group = task / TASKS_PER_GROUP;
        int slot = task - group * TASKS_PER_GROUP;
        int n = group / ng;
        int g = group - n * ng;
        int k0 = g * GS + slot * 4;
        float de = neuron_scale[n] * (float)sub_scale[group];
        float me = neuron_min[n] * (float)sub_min[group];
        uint16_t qw = *reinterpret_cast<const uint16_t*>(
            q_packed + (size_t)group * QBYTES + slot * 2);
        #pragma unroll
        for (int b = 0; b < 2; ++b) {
            int k = k0 + b * 2;
            if (k < neuron_len) {
                uint8_t qv = (uint8_t)(qw >> (b * 8));
                __half lo = __float2half(de * (float)(qv & 0x0f) - me);
                if (k + 1 < neuron_len) {
                    __half hi = __float2half(de * (float)(qv >> 4) - me);
                    *reinterpret_cast<__half2*>(w + (size_t)n * neuron_len + k) = __halves2half2(lo, hi);
                } else {
                    w[(size_t)n * neuron_len + k] = lo;
                }
            }
        }
    }
}

__global__ void dequant_full_packed_compact_int6_gs24_vec4_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    __half* __restrict__ w,
    int N, int ng, int neuron_len)
{
    constexpr int GS = 24;
    constexpr int QBYTES = 18;
    constexpr int TASKS_PER_GROUP = 6;
    int total_tasks = N * ng * TASKS_PER_GROUP;
    for (int task = blockIdx.x * blockDim.x + threadIdx.x;
         task < total_tasks;
         task += blockDim.x * gridDim.x) {
        int group = task / TASKS_PER_GROUP;
        int chunk = task - group * TASKS_PER_GROUP;
        int n = group / ng;
        int g = group - n * ng;
        int k0 = g * GS + chunk * 4;
        const uint8_t* p = q_packed + (size_t)group * QBYTES + chunk * 3;
        uint32_t packed = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
        float de = neuron_scale[n] * (float)sub_scale[group];
        float me = neuron_min[n] * (float)sub_min[group];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int k = k0 + j;
            if (k < neuron_len) {
                uint8_t qv = (uint8_t)((packed >> (j * 6)) & 0x3f);
                w[(size_t)n * neuron_len + k] = __float2half(de * (float)qv - me);
            }
        }
    }
}

mfq_tensor_backend::Tensor nint_dequant_full_packed_compact_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, int64_t neuron_len, int64_t gs)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    MFQ_RUNTIME_CHECK((int)sub_scale.size(0) == N && (int)sub_scale.size(1) == ng, "sub_scale shape mismatch");
    MFQ_RUNTIME_CHECK(sub_min.sizes() == sub_scale.sizes(), "sub_min shape mismatch");
    MFQ_RUNTIME_CHECK((int)neuron_scale.size(0) == N && (int)neuron_min.size(0) == N, "neuron metadata shape mismatch");
    MFQ_RUNTIME_CHECK(neuron_len <= (int64_t)ng * gs, "neuron_len must be <= ng * gs");
    auto w = mfq_tensor_backend::empty({N, neuron_len}, neuron_scale.options().dtype(mfq_tensor_backend::kHalf));
    cudaStream_t stream = mfq_current_cuda_stream();
    int total = N * ng * ((int)gs / 2);
    int block = 256;
    int grid = std::min((total + block - 1) / block, 65535);

#define DQFULLCOMPACTLAUNCH(GSVAL)                                                 \
    dequant_full_packed_compact_kernel<GSVAL><<<grid, block, 0, stream>>>(        \
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),              \
        sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),              \
        neuron_min.data_ptr<float>(), reinterpret_cast<__half*>(w.data_ptr<mfq_half>()), \
        N, ng, (int)neuron_len)

    switch ((int)gs) {
        case 16: DQFULLCOMPACTLAUNCH(16); break;
        case 24: {
            const char* vec_env = std::getenv("MFQ_NINT4_GS24_DQ_VEC2");
            if (vec_env == nullptr || vec_env[0] != '0') {
                int vec_total = N * ng * 6;
                int vec_grid = std::min((vec_total + block - 1) / block, 65535);
                dequant_full_packed_compact_gs24_vec2_kernel<<<vec_grid, block, 0, stream>>>(
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),
                    sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),
                    neuron_min.data_ptr<float>(), reinterpret_cast<__half*>(w.data_ptr<mfq_half>()),
                    N, ng, (int)neuron_len);
            } else {
                DQFULLCOMPACTLAUNCH(24);
            }
            break;
        }
        case 32: DQFULLCOMPACTLAUNCH(32); break;
        case 48: DQFULLCOMPACTLAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_dequant_full_packed_compact: gs must be in {16,24,32,48}, got ", gs);
    }
#undef DQFULLCOMPACTLAUNCH
    return w;
}

mfq_tensor_backend::Tensor nint_dequant_full_packed_compact_bits_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, int64_t neuron_len, int64_t gs, int64_t bits)
{
    MFQ_RUNTIME_CHECK(bits == 2 || bits == 3 || bits == 5 || bits == 6 || bits == 8,
                "packed-bits full dequant supports bits in {2,3,5,6,8}, got ", bits);
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == (((int)gs * (int)bits + 7) / 8),
                "q_packed last dim must equal ceil(gs*bits/8)");
    MFQ_RUNTIME_CHECK((int)sub_scale.size(0) == N && (int)sub_scale.size(1) == ng, "sub_scale shape mismatch");
    MFQ_RUNTIME_CHECK(sub_min.sizes() == sub_scale.sizes(), "sub_min shape mismatch");
    MFQ_RUNTIME_CHECK((int)neuron_scale.size(0) == N && (int)neuron_min.size(0) == N, "neuron metadata shape mismatch");
    MFQ_RUNTIME_CHECK(neuron_len <= (int64_t)ng * gs, "neuron_len must be <= ng * gs");
    auto w = mfq_tensor_backend::empty({N, neuron_len}, neuron_scale.options().dtype(mfq_tensor_backend::kHalf));
    cudaStream_t stream = mfq_current_cuda_stream();
    size_t total = (size_t)N * (size_t)neuron_len;
    int block = 256;
    int grid = (int)std::min<size_t>((total + block - 1) / block, 65535);

#define DQFULLBITSLAUNCH(BITSVAL, GSVAL)                                               \
    dequant_full_packed_compact_bits_kernel_early<BITSVAL, GSVAL><<<grid, block, 0, stream>>>( \
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),                    \
        sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                    \
        neuron_min.data_ptr<float>(), reinterpret_cast<__half*>(w.data_ptr<mfq_half>()), \
        N, ng, (int)neuron_len)

#define DQFULLBITS_GS_SWITCH(BITSVAL)                                                   \
    switch ((int)gs) {                                                                  \
        case 16: DQFULLBITSLAUNCH(BITSVAL, 16); break;                                  \
        case 20: DQFULLBITSLAUNCH(BITSVAL, 20); break;                                  \
        case 22: DQFULLBITSLAUNCH(BITSVAL, 22); break;                                  \
        case 24: DQFULLBITSLAUNCH(BITSVAL, 24); break;                                  \
        case 26: DQFULLBITSLAUNCH(BITSVAL, 26); break;                                  \
        case 28: DQFULLBITSLAUNCH(BITSVAL, 28); break;                                  \
        case 30: DQFULLBITSLAUNCH(BITSVAL, 30); break;                                  \
        case 32: DQFULLBITSLAUNCH(BITSVAL, 32); break;                                  \
        case 34: DQFULLBITSLAUNCH(BITSVAL, 34); break;                                  \
        case 36: DQFULLBITSLAUNCH(BITSVAL, 36); break;                                  \
        case 40: DQFULLBITSLAUNCH(BITSVAL, 40); break;                                  \
        case 48: DQFULLBITSLAUNCH(BITSVAL, 48); break;                                  \
        case 64: DQFULLBITSLAUNCH(BITSVAL, 64); break;                                  \
        default: MFQ_RUNTIME_CHECK(false, "packed-bits full dequant unsupported gs ", gs);     \
    }

    if (bits == 2) {
        DQFULLBITS_GS_SWITCH(2);
    } else if (bits == 3) {
        DQFULLBITS_GS_SWITCH(3);
    } else if (bits == 5) {
        DQFULLBITS_GS_SWITCH(5);
    } else if (bits == 6) {
        const char* vec_env = std::getenv("MFQ_NINT6_GS24_DQ_VEC4");
        if (gs == 24 && (vec_env == nullptr || vec_env[0] != '0')) {
            int vec_total = N * ng * 6;
            int vec_grid = std::min((vec_total + block - 1) / block, 65535);
            dequant_full_packed_compact_int6_gs24_vec4_kernel<<<vec_grid, block, 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),
                neuron_min.data_ptr<float>(), reinterpret_cast<__half*>(w.data_ptr<mfq_half>()),
                N, ng, (int)neuron_len);
        } else {
            DQFULLBITS_GS_SWITCH(6);
        }
    } else {
        DQFULLBITS_GS_SWITCH(8);
    }
#undef DQFULLBITS_GS_SWITCH
#undef DQFULLBITSLAUNCH
    return w;
}

mfq_tensor_backend::Tensor nint5_gs28_q5_repack_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(q_packed.dim() == 3 && q_packed.size(2) == 18,
                "NINT5 gs28 q_packed must have shape [N,ng,18]");
    const int N = (int)q_packed.size(0);
    const int ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous() &&
                sub_scale.dim() == 2 && sub_scale.size(0) == N && sub_scale.size(1) == ng,
                "sub_scale must be cuda contiguous uint8 [N,ng]");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous() &&
                sub_min.sizes() == sub_scale.sizes(), "sub_min shape mismatch");
    auto out = mfq_tensor_backend::empty({N, ng, 20}, q_packed.options());
    const size_t groups = (size_t)N * (size_t)ng;
    constexpr int block = 256;
    const int grid = (int)std::min<size_t>((groups + block - 1) / block, 65535);
    cudaStream_t stream = mfq_current_cuda_stream();
    repack_q5_gs28_exec_kernel<<<grid, block, 0, stream>>>(
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
        out.data_ptr<uint8_t>(), groups);
    return out;
}

mfq_tensor_backend::Tensor nint5_gs28_q5_dequant_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    int64_t neuron_len)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(q_packed.dim() == 3 && q_packed.size(2) == 20,
                "NINT5 gs28 Q5 execution tensor must have shape [N,ng,20]");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    const int N = (int)q_packed.size(0);
    const int ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK(neuron_scale.numel() == N && neuron_min.numel() == N, "neuron metadata shape mismatch");
    MFQ_RUNTIME_CHECK(neuron_len <= (int64_t)ng * 28, "neuron_len must be <= ng*28");
    auto w = mfq_tensor_backend::empty({N, neuron_len}, neuron_scale.options().dtype(mfq_tensor_backend::kHalf));
    const size_t total = (size_t)N * (size_t)neuron_len;
    constexpr int block = 256;
    const int grid = (int)std::min<size_t>((total + block - 1) / block, 65535);
    cudaStream_t stream = mfq_current_cuda_stream();
    dequant_full_q5_gs28_exec_kernel<<<grid, block, 0, stream>>>(
        q_packed.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(),
        reinterpret_cast<__half*>(w.data_ptr<mfq_half>()), N, ng, (int)neuron_len);
    return w;
}

template <int QBPT>
__global__ void dequant_full_packed_gs24_qbpt_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, 12]
    const float* __restrict__ d_eff,         // [N, ng]
    const float* __restrict__ m_eff,         // [N, ng]
    __half* __restrict__ w,                  // [N, neuron_len]
    int N, int ng, int neuron_len)
{
    constexpr int GS = 24;
    constexpr int QBYTES = GS / 2;
    constexpr int JOBS_PER_GROUP = (QBYTES + QBPT - 1) / QBPT;
    int total = N * ng * JOBS_PER_GROUP;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int jb = idx % JOBS_PER_GROUP;
        int t = idx / JOBS_PER_GROUP;
        int g = t % ng;
        int n = t / ng;
        int b0 = jb * QBPT;
        float de = d_eff[(size_t)n * ng + g];
        float me = m_eff[(size_t)n * ng + g];
        #pragma unroll
        for (int r = 0; r < QBPT; ++r) {
            int b = b0 + r;
            if (b >= QBYTES) {
                continue;
            }
            int k = g * GS + b * 2;
            if (k >= neuron_len) {
                continue;
            }
            uint8_t qv = q_packed[(size_t)n * ng * QBYTES + g * QBYTES + b];
            __half lo = __float2half(de * (float)(qv & 0x0f) - me);
            if (k + 1 < neuron_len) {
                __half hi = __float2half(de * (float)(qv >> 4) - me);
                *reinterpret_cast<__half2*>(w + (size_t)n * neuron_len + k) = __halves2half2(lo, hi);
            } else {
                w[(size_t)n * neuron_len + k] = lo;
            }
        }
    }
}

mfq_tensor_backend::Tensor nint_dequant_full_packed_gs24_x2_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor d_eff, mfq_tensor_backend::Tensor m_eff, int64_t neuron_len)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(d_eff.is_cuda() && d_eff.scalar_type() == mfq_tensor_backend::kFloat32 && d_eff.is_contiguous(),
                "d_eff must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(m_eff.is_cuda() && m_eff.scalar_type() == mfq_tensor_backend::kFloat32 && m_eff.is_contiguous(),
                "m_eff must be cuda contiguous f32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == 12, "q_packed last dim must be 12 for gs24");
    MFQ_RUNTIME_CHECK((int)d_eff.size(0) == N && (int)d_eff.size(1) == ng, "d_eff shape mismatch");
    MFQ_RUNTIME_CHECK((int)m_eff.size(0) == N && (int)m_eff.size(1) == ng, "m_eff shape mismatch");
    MFQ_RUNTIME_CHECK(neuron_len <= (int64_t)ng * 24, "neuron_len must be <= ng * 24");
    auto w = mfq_tensor_backend::empty({N, neuron_len}, d_eff.options().dtype(mfq_tensor_backend::kHalf));
    cudaStream_t stream = mfq_current_cuda_stream();
    constexpr int JOBS_PER_GROUP = 6;
    int total = N * ng * JOBS_PER_GROUP;
    int block = 256;
    int grid = std::min((total + block - 1) / block, 65535);
    dequant_full_packed_gs24_qbpt_kernel<2><<<grid, block, 0, stream>>>(
        q_packed.data_ptr<uint8_t>(), d_eff.data_ptr<float>(), m_eff.data_ptr<float>(),
        reinterpret_cast<__half*>(w.data_ptr<mfq_half>()), N, ng, (int)neuron_len);
    return w;
}

__global__ void dequant_full_packed_gs24_x2h2_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, 12]
    const __half2* __restrict__ eff_pair,    // [N, ng], low=d_eff, high=m_eff
    __half* __restrict__ w,                  // [N, neuron_len]
    int N, int ng, int neuron_len)
{
    constexpr int GS = 24;
    constexpr int QBYTES = GS / 2;
    constexpr int JOBS_PER_GROUP = 6;
    int total = N * ng * JOBS_PER_GROUP;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int jb = idx % JOBS_PER_GROUP;
        int t = idx / JOBS_PER_GROUP;
        int g = t % ng;
        int n = t / ng;
        int b = jb * 2;
        int k = g * GS + b * 2;
        if (k >= neuron_len) {
            continue;
        }
        uint8_t q0 = q_packed[(size_t)n * ng * QBYTES + g * QBYTES + b];
        uint8_t q1 = q_packed[(size_t)n * ng * QBYTES + g * QBYTES + b + 1];
        __half2 dm = eff_pair[(size_t)n * ng + g];
        __half d = __low2half(dm);
        __half m = __high2half(dm);
        __half2 dd = __halves2half2(d, d);
        __half2 mm = __halves2half2(__hneg(m), __hneg(m));
        __half2 r0 = __hfma2(__halves2half2(__float2half((float)(q0 & 0x0f)), __float2half((float)(q0 >> 4))), dd, mm);
        __half2 r1 = __hfma2(__halves2half2(__float2half((float)(q1 & 0x0f)), __float2half((float)(q1 >> 4))), dd, mm);
        size_t out = (size_t)n * neuron_len + k;
        if (k + 3 < neuron_len) {
            *reinterpret_cast<__half2*>(w + out) = r0;
            *reinterpret_cast<__half2*>(w + out + 2) = r1;
        } else {
            w[out] = __low2half(r0);
            if (k + 1 < neuron_len) {
                w[out + 1] = __high2half(r0);
            }
            if (k + 2 < neuron_len) {
                w[out + 2] = __low2half(r1);
            }
            if (k + 3 < neuron_len) {
                w[out + 3] = __high2half(r1);
            }
        }
    }
}

mfq_tensor_backend::Tensor nint_dequant_full_packed_gs24_x2h2_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor eff_pair, int64_t neuron_len)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(eff_pair.is_cuda() && eff_pair.scalar_type() == mfq_tensor_backend::kHalf && eff_pair.is_contiguous(),
                "eff_pair must be cuda contiguous fp16");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == 12, "q_packed last dim must be 12 for gs24");
    MFQ_RUNTIME_CHECK(eff_pair.dim() == 3 && (int)eff_pair.size(0) == N && (int)eff_pair.size(1) == ng && (int)eff_pair.size(2) == 2,
                "eff_pair shape mismatch");
    MFQ_RUNTIME_CHECK(neuron_len <= (int64_t)ng * 24, "neuron_len must be <= ng * 24");
    auto w = mfq_tensor_backend::empty({N, neuron_len}, eff_pair.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    int total = N * ng * 6;
    int block = 256;
    int grid = std::min((total + block - 1) / block, 65535);
    dequant_full_packed_gs24_x2h2_kernel<<<grid, block, 0, stream>>>(
        q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half2*>(eff_pair.data_ptr<mfq_half>()),
        reinterpret_cast<__half*>(w.data_ptr<mfq_half>()), N, ng, (int)neuron_len);
    return w;
}

template <int GS>
__global__ void dequant_full_packed_h2_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, GS/2]
    const __half2* __restrict__ eff_pair,    // [N, ng], low=d_eff, high=m_eff
    __half* __restrict__ w,                  // [N, neuron_len]
    int N, int ng, int neuron_len)
{
    bool aligned_pairs = (neuron_len & 1) == 0;
    int qbytes = GS / 2;
    int total = N * ng * qbytes;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int b = idx % qbytes;
        int t = idx / qbytes;
        int g = t % ng;
        int n = t / ng;
        int k = g * GS + b * 2;
        if (k >= neuron_len) {
            continue;
        }
        uint8_t qv = q_packed[(size_t)n * ng * qbytes + g * qbytes + b];
        __half2 dm = eff_pair[(size_t)n * ng + g];
        __half d = __low2half(dm);
        __half m = __high2half(dm);
        __half2 qh = __halves2half2(__float2half((float)(qv & 0x0f)), __float2half((float)(qv >> 4)));
        __half2 res = __hfma2(qh, __halves2half2(d, d), __halves2half2(__hneg(m), __hneg(m)));
        if (k + 1 < neuron_len) {
            if (aligned_pairs) {
                *reinterpret_cast<__half2*>(w + (size_t)n * neuron_len + k) = res;
            } else {
                w[(size_t)n * neuron_len + k] = __low2half(res);
                w[(size_t)n * neuron_len + k + 1] = __high2half(res);
            }
        } else {
            w[(size_t)n * neuron_len + k] = __low2half(res);
        }
    }
}

mfq_tensor_backend::Tensor nint_dequant_full_packed_h2_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor eff_pair, int64_t neuron_len, int64_t gs)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(eff_pair.is_cuda() && eff_pair.scalar_type() == mfq_tensor_backend::kHalf && eff_pair.is_contiguous(),
                "eff_pair must be cuda contiguous fp16");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    MFQ_RUNTIME_CHECK(eff_pair.dim() == 3 && (int)eff_pair.size(0) == N && (int)eff_pair.size(1) == ng && (int)eff_pair.size(2) == 2,
                "eff_pair shape mismatch");
    MFQ_RUNTIME_CHECK(neuron_len <= (int64_t)ng * gs, "neuron_len must be <= ng * gs");
    auto w = mfq_tensor_backend::empty({N, neuron_len}, eff_pair.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    int total = N * ng * ((int)gs / 2);
    int block = 256;
    int grid = std::min((total + block - 1) / block, 65535);

#define DQFULLH2LAUNCH(GSVAL)                                                      \
    dequant_full_packed_h2_kernel<GSVAL><<<grid, block, 0, stream>>>(             \
        q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half2*>(eff_pair.data_ptr<mfq_half>()), \
        reinterpret_cast<__half*>(w.data_ptr<mfq_half>()), N, ng, (int)neuron_len)

    switch ((int)gs) {
        case 16: DQFULLH2LAUNCH(16); break;
        case 24: DQFULLH2LAUNCH(24); break;
        case 32: DQFULLH2LAUNCH(32); break;
        case 48: DQFULLH2LAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_dequant_full_packed_h2: gs must be in {16,24,32,48}, got ", gs);
    }
#undef DQFULLH2LAUNCH
    return w;
}

// ---------------------------------------------------------------------------
// INT-GEMV (decode / small batch). Mirrors llama.cpp mmvq.cu + vecdotq.cuh:
// pre-quantize x to per-group int8 (+ group scale), then warp-per-output-row dot
// via __dp4a. nint stores q byte-per-value (0..15), so 4 q values pack into one int
// with no nibble unpacking -> dp4a(q_int, qx_int) directly.
//
//   dot[row,m] = neuron_scale[row]*sum_d - neuron_min[row]*sum_m
//   sum_d      = sum_g xscale[m,g]*sub_scale[row,g]*(sum q.qx)_g
//   sum_m      = sum_g xscale[m,g]*sub_min[row,g]  *(sum qx)_g
// ---------------------------------------------------------------------------

template <int GS, int BD, bool WITH_SUM=false, bool INPUT_BF16=false>
__global__ void quantize_x_kernel(
    const void* __restrict__ x,      // [M, K_real], FP16 or BF16
    int8_t* __restrict__ qx,         // [M, K_pad]
    float* __restrict__ xscale,      // [M, ng]
    int32_t* __restrict__ xsum,      // [M, ng]
    int M, int K_real, int K_pad)
{
    int m = blockIdx.x;
    int g = blockIdx.y;
    int tid = threadIdx.x;
    int ng = gridDim.y;
    int base = g * GS;
    bool real = (tid < GS) && (base + tid < K_real);
    float xv = 0.0f;
    if (tid < GS && real) {
        const size_t index = (size_t)m * K_real + base + tid;
        if constexpr (INPUT_BF16) {
            const float source = __bfloat162float(
                reinterpret_cast<const __nv_bfloat16*>(x)[index]);
            xv = __half2float(__float2half(source));
        } else {
            xv = __half2float(
                reinterpret_cast<const __half*>(x)[index]);
        }
    }

    float amax = block_max<BD / 32>(fabsf(xv));
    float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    int qi = 0;
    if (tid < GS) {
        if (real) {
            float q = roundf(xv / scale);
            q = fminf(fmaxf(q, -127.0f), 127.0f);
            qi = (int)q;
        }
    }
    int sumi = 0;
    if constexpr (WITH_SUM) {
        sumi = (int)block_sum<BD / 32>((float)qi);
    }
    if (tid == 0) {
        xscale[(size_t)m * ng + g] = scale;
        if constexpr (WITH_SUM) {
            xsum[(size_t)m * ng + g] = sumi;
        }
    }
    if (tid < GS) {
        qx[(size_t)m * K_pad + base + tid] = real ? (int8_t)qi : (int8_t)0;
    }
}

// Direct NINT group -> K32 activation layout. Padding exists only in the
// transient MMQ workspace; the stored MFQ group remains compact.
template <int GS>
__global__ void __launch_bounds__(256) quantize_x_group32_layout_kernel(
    const __half* __restrict__ x, // [M, K_real]
    int32_t* __restrict__ qx_mmq, // [ceil(ng/8), M_pad, 68]
    float* __restrict__ xscale,   // [M, ng]
    int32_t* __restrict__ xsum,   // [M, ng]
    int M, int K_real, int ng, int M_pad)
{
    constexpr int CHUNK_GROUPS = 8;
    constexpr int GROUP_KPACK = 8;
    constexpr int KSTRIDE = CHUNK_GROUPS * GROUP_KPACK + 4;
    constexpr int GROUPS_PER_BLOCK = 32;

    const int m = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int g_block = blockIdx.y * GROUPS_PER_BLOCK;

    #pragma unroll
    for (int round = 0; round < GROUPS_PER_BLOCK / 8; ++round) {
        const int g = g_block + round * 8 + warp;
        const bool valid_group = g < ng;
        const int k = g * GS + lane;
        const bool valid_value = valid_group && lane < GS && k < K_real;
        const float xv = valid_value ? __half2float(x[(size_t)m * K_real + k]) : 0.0f;

        float amax = fabsf(xv);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        int qi = 0;
        if (valid_value) {
            qi = (int)fminf(fmaxf(roundf(xv / scale), -127.0f), 127.0f);
        }

        int sumi = qi;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sumi += __shfl_xor_sync(0xffffffff, sumi, offset);
        }
        if (lane == 0 && valid_group) {
            xscale[(size_t)m * ng + g] = scale;
            xsum[(size_t)m * ng + g] = sumi;
        }

        int packed = 0;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int src_lane = lane < GS / 4 ? lane * 4 + u : 0;
            const int qv = __shfl_sync(0xffffffff, qi, src_lane);
            packed |= (qv & 0xff) << (8 * u);
        }
        if (lane < GROUP_KPACK && valid_group) {
            const int chunk = g / CHUNK_GROUPS;
            const int gl = g % CHUNK_GROUPS;
            const size_t base = ((size_t)chunk * M_pad + m) * KSTRIDE;
            qx_mmq[base + gl * GROUP_KPACK + lane] = lane < GS / 4 ? packed : 0;
        }
    }
}

// NINT2 uses gs16. Pack two independently scaled groups into the two halves
// of one K32 activation tile. The weight tile masks the opposite half for each
// group, preserving separate dot products while halving activation traffic.
__global__ void __launch_bounds__(256) quantize_x_gs16_pair32_layout_kernel(
    const __half* __restrict__ x, // [M, K_real]
    int32_t* __restrict__ qx_mmq, // [ceil(ng/8), M_pad, 36]
    float* __restrict__ xscale,   // [M, ng]
    int32_t* __restrict__ xsum,   // [M, ng]
    int M, int K_real, int ng, int M_pad)
{
    constexpr int GS = 16;
    constexpr int CHUNK_GROUPS = 8;
    constexpr int PAIRS_PER_CHUNK = CHUNK_GROUPS / 2;
    constexpr int GROUP_KPACK = 8;
    constexpr int KSTRIDE = PAIRS_PER_CHUNK * GROUP_KPACK + 4;
    constexpr int GROUPS_PER_BLOCK = 64;

    const int m = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int half = lane >> 4;
    const int local = lane & 15;
    const int g_block = blockIdx.y * GROUPS_PER_BLOCK;

    #pragma unroll
    for (int round = 0; round < GROUPS_PER_BLOCK / 16; ++round) {
        const int g_even = g_block + round * 16 + warp * 2;
        const int g = g_even + half;
        const bool valid_group = g < ng;
        const int k = g * GS + local;
        const bool valid_value = valid_group && k < K_real;
        const float xv = valid_value
            ? __half2float(x[(size_t)m * K_real + k])
            : 0.0f;

        float amax = fabsf(xv);
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            amax = fmaxf(
                amax, __shfl_xor_sync(0xffffffff, amax, offset, 16));
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        int qi = valid_value
            ? (int)fminf(fmaxf(roundf(xv / scale), -127.0f), 127.0f)
            : 0;

        int sumi = qi;
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            sumi += __shfl_xor_sync(0xffffffff, sumi, offset, 16);
        }
        if (local == 0 && valid_group) {
            xscale[(size_t)m * ng + g] = scale;
            xsum[(size_t)m * ng + g] = sumi;
        }

        int packed = 0;
        const int word = local < 4 ? local : 0;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int src_lane = half * 16 + word * 4 + u;
            const int qv = __shfl_sync(0xffffffff, qi, src_lane);
            packed |= (qv & 0xff) << (8 * u);
        }
        if (local < 4 && valid_group) {
            const int chunk = g_even / CHUNK_GROUPS;
            const int pair_local = (g_even % CHUNK_GROUPS) / 2;
            const size_t base = ((size_t)chunk * M_pad + m) * KSTRIDE;
            qx_mmq[base + pair_local * GROUP_KPACK + half * 4 + local] =
                packed;
        }
    }
}

template <int GS, int BD, int MODE>
__global__ void quantize_x_gate_kernel(
    const __half* __restrict__ x,     // [M, K_real]
    const __half* __restrict__ gate,  // [M, K_real]
    int8_t* __restrict__ qx,          // [M, K_pad]
    float* __restrict__ xscale,       // [M, ng]
    int32_t* __restrict__ xsum,       // [M, ng]
    int M, int K_real, int K_pad)
{
    int m = blockIdx.x;
    int g = blockIdx.y;
    int tid = threadIdx.x;
    int ng = gridDim.y;
    int base = g * GS;
    bool real = (tid < GS) && (base + tid < K_real);
    float xv = 0.0f;
    if (real) {
        size_t off = (size_t)m * K_real + base + tid;
        float gv = __half2float(gate[off]);
        float sig = 1.0f / (1.0f + expf(-gv));
        float mult = MODE == 1 ? sig : gv * sig;
        xv = __half2float(x[off]) * mult;
    }

    float amax = block_max<BD / 32>(fabsf(xv));
    float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    int qi = 0;
    if (tid < GS && real) {
        float q = roundf(xv / scale);
        q = fminf(fmaxf(q, -127.0f), 127.0f);
        qi = (int)q;
    }
    int sumi = (int)block_sum<BD / 32>((float)qi);
    if (tid == 0) {
        xscale[(size_t)m * ng + g] = scale;
        xsum[(size_t)m * ng + g] = sumi;
    }
    if (tid < GS) {
        qx[(size_t)m * K_pad + base + tid] = real ? (int8_t)qi : (int8_t)0;
    }
}

__device__ __forceinline__ int unpack_int4x4(const uint8_t* p)
{
    const uint8_t a = p[0];
    const uint8_t b = p[1];
    return ((int)(a & 0x0f)) |
           ((int)(a >> 4) << 8) |
           ((int)(b & 0x0f) << 16) |
           ((int)(b >> 4) << 24);
}

__device__ __forceinline__ int unpack_int4x4_u16(uint32_t packed)
{
    return ((int)(packed & 0x000f)) |
           ((int)(packed & 0x00f0) << 4) |
           ((int)(packed & 0x0f00) << 8) |
           ((int)(packed & 0xf000) << 12);
}

// Unpack 4 consecutive 6-bit values from a 3-byte little-endian bitstream into
// one int (one value per byte lane) for __dp4a. Bit offset of element 0 must be
// byte-aligned, i.e. the caller passes p = base of a 4-element group slice.
__device__ __forceinline__ int unpack_int6x4(const uint8_t* p)
{
    uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
    int q0 = (w      ) & 63;
    int q1 = (w >> 6 ) & 63;
    int q2 = (w >> 12) & 63;
    int q3 = (w >> 18) & 63;
    return q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
}

__device__ __forceinline__ int unpack_int6x4_u24(uint32_t w)
{
    int q0 = (w      ) & 63;
    int q1 = (w >> 6 ) & 63;
    int q2 = (w >> 12) & 63;
    int q3 = (w >> 18) & 63;
    return q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
}

template <int BITS>
__device__ __forceinline__ uint8_t unpack_qbits_one(const uint8_t* p, int lane)
{
    if constexpr (BITS == 8) {
        return p[lane];
    } else {
        constexpr uint32_t MASK = (1u << BITS) - 1u;
        int bit = lane * BITS;
        int byte = bit >> 3;
        int shift = bit & 7;
        uint32_t word = (uint32_t)p[byte];
        if (shift + BITS > 8) {
            word |= ((uint32_t)p[byte + 1] << 8);
        }
        return (uint8_t)((word >> shift) & MASK);
    }
}

template <int BITS>
__device__ __forceinline__ int unpack_qbits4(const uint8_t* p, int lane)
{
    if constexpr (BITS == 2) {
        const uint8_t packed = p[lane >> 2];
        return ((int)(packed & 0x03)) |
               ((int)((packed >> 2) & 0x03) << 8) |
               ((int)((packed >> 4) & 0x03) << 16) |
               ((int)((packed >> 6) & 0x03) << 24);
    } else if constexpr (BITS == 5) {
        int base = (lane >> 3) * 5;
        if ((lane & 4) == 0) {
            uint8_t b0 = p[base + 0];
            uint8_t b1 = p[base + 1];
            uint8_t b2 = p[base + 2];
            uint8_t q0 =  b0        & 31;
            uint8_t q1 = (b0 >> 5 | b1 << 3) & 31;
            uint8_t q2 = (b1 >> 2) & 31;
            uint8_t q3 = (b1 >> 7 | b2 << 1) & 31;
            return (int)q0 | ((int)q1 << 8) | ((int)q2 << 16) | ((int)q3 << 24);
        } else {
            uint8_t b2 = p[base + 2];
            uint8_t b3 = p[base + 3];
            uint8_t b4 = p[base + 4];
            uint8_t q4 = (b2 >> 4 | b3 << 4) & 31;
            uint8_t q5 = (b3 >> 1) & 31;
            uint8_t q6 = (b3 >> 6 | b4 << 2) & 31;
            uint8_t q7 = (b4 >> 3) & 31;
            return (int)q4 | ((int)q5 << 8) | ((int)q6 << 16) | ((int)q7 << 24);
        }
    } else if constexpr (BITS == 6) {
        int base = (lane >> 2) * 3;
        uint8_t b0 = p[base + 0];
        uint8_t b1 = p[base + 1];
        uint8_t b2 = p[base + 2];
        uint8_t q0 =  b0        & 63;
        uint8_t q1 = (b0 >> 6 | b1 << 2) & 63;
        uint8_t q2 = (b1 >> 4 | b2 << 4) & 63;
        uint8_t q3 = (b2 >> 2) & 63;
        return (int)q0 | ((int)q1 << 8) | ((int)q2 << 16) | ((int)q3 << 24);
    }
    uint8_t q0 = unpack_qbits_one<BITS>(p, lane + 0);
    uint8_t q1 = unpack_qbits_one<BITS>(p, lane + 1);
    uint8_t q2 = unpack_qbits_one<BITS>(p, lane + 2);
    uint8_t q3 = unpack_qbits_one<BITS>(p, lane + 3);
    return (int)q0 | ((int)q1 << 8) | ((int)q2 << 16) | ((int)q3 << 24);
}

__device__ __forceinline__ int load_i8x4_unaligned(const int8_t* p)
{
    const uint8_t* u = reinterpret_cast<const uint8_t*>(p);
    return (int)u[0] | ((int)u[1] << 8) | ((int)u[2] << 16) | ((int)u[3] << 24);
}

template <int BITS>
__device__ __forceinline__ int dp4a_qbits_xs8(int qv, int xv, int xsum)
{
    if constexpr (BITS == 8) {
        return __dp4a(qv ^ (int)0x80808080u, xv, 0) + 128 * xsum;
    }
    return __dp4a(qv, xv, 0);
}

template <int BITS, int GS, int MAX_M>
__global__ void __launch_bounds__(128) gemv_packed_bits_batch_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, ceil(GS*BITS/8)]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int WPB = 4;
    constexpr int QBYTES = (GS * BITS + 7) / 8;
    int row = blockIdx.x * WPB + threadIdx.y;
    int m_base = blockIdx.y * MAX_M;
    int lane = threadIdx.x;
    if (row >= N || m_base >= M) {
        return;
    }

    const uint8_t* qrow = q_packed + (size_t)row * ng * QBYTES;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min + (size_t)row * ng;
    float ns = neuron_scale[row];
    float nm = neuron_min[row];

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc[m] = 0.0f;
    }

    for (int k = lane; k < K_pad; k += 32) {
        int g = k / GS;
        int gi = k - g * GS;
        uint8_t qv = unpack_qbits_one<BITS>(qrow + (size_t)g * QBYTES, gi);
        uint8_t ss = ssrow[g];
        uint8_t sm = smrow[g];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            int gm = m_base + m;
            if (gm < M) {
                int qi = (int)qx[(size_t)gm * K_pad + k];
                float xs = xscale[(size_t)gm * ng + g];
                acc[m] += xs * ((ns * (float)ss * (float)qv - nm * (float)sm) * (float)qi);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], o);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            int gm = m_base + m;
            if (gm < M) {
                out[(size_t)gm * N + row] = __float2half(acc[m]);
            }
        }
    }
}

template <int BITS, int GS, int MAX_M, int MSPLIT=1>
__global__ void __launch_bounds__(128 * MSPLIT) gemv_packed_bits_group4_batch_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, ceil(GS*BITS/8)]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int WPB = 4;
    constexpr int QBYTES = (GS * BITS + 7) / 8;
    constexpr int CHUNKS = (GS + 3) / 4;
    constexpr int GPW = 32 / CHUNKS;
    constexpr int MPW = (MAX_M + MSPLIT - 1) / MSPLIT;
    int warp = threadIdx.y;
    int row = blockIdx.x * WPB + warp / MSPLIT;
    int msplit = warp % MSPLIT;
    int lane = threadIdx.x;
    if (row >= N) {
        return;
    }

    const uint8_t* qrow = q_packed + (size_t)row * ng * QBYTES;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min + (size_t)row * ng;
    float ns = neuron_scale[row];
    float nm = neuron_min[row];

    float acc[MPW];
    #pragma unroll
    for (int m = 0; m < MPW; ++m) {
        acc[m] = 0.0f;
    }

    int relg = lane / CHUNKS;
    int ci = lane - relg * CHUNKS;
    int off = ci * 4;
    bool active_lane = relg < GPW;
    bool full4 = active_lane && ((off + 3) < GS);
    bool tail = active_lane && (off < GS) && !full4;
    for (int gb = 0; gb < ng; gb += GPW) {
        int g = gb + relg;
        if (!active_lane || g >= ng) {
            continue;
        }
        const uint8_t* qg = qrow + (size_t)g * QBYTES;
        uint8_t ss = ssrow[g];
        uint8_t sm = smrow[g];
        if (full4) {
            int qv;
            if constexpr (BITS == 6 && GS == 24) {
                qv = unpack_int6x4(qg + ci * 3);
            } else {
                qv = unpack_qbits4<BITS>(qg, off);
            }
            int k = g * GS + off;
            #pragma unroll
            for (int m = 0; m < MPW; ++m) {
                int gm = msplit * MPW + m;
                if (gm < M) {
                    const int8_t* qxrow = qx + (size_t)gm * K_pad;
                    int xv;
                    if constexpr (BITS == 6 && GS == 24) {
                        xv = *reinterpret_cast<const int*>(qxrow + k);
                    } else {
                        xv = load_i8x4_unaligned(qxrow + k);
                    }
                    int di = __dp4a(qv, xv, 0);
                    int mi = __dp4a(0x01010101, xv, 0);
                    float xs = xscale[(size_t)gm * ng + g];
                    acc[m] += xs * (ns * (float)ss * (float)di - nm * (float)sm * (float)mi);
                }
            }
        } else if (tail) {
            #pragma unroll
            for (int m = 0; m < MPW; ++m) {
                int gm = msplit * MPW + m;
                if (gm < M) {
                    const int8_t* qxrow = qx + (size_t)gm * K_pad;
                    int di = 0;
                    int mi = 0;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        int gi = off + r;
                        if (gi < GS) {
                            int qi = (int)qxrow[(size_t)g * GS + gi];
                            int qv = (int)unpack_qbits_one<BITS>(qg, gi);
                            di += qv * qi;
                            mi += qi;
                        }
                    }
                    float xs = xscale[(size_t)gm * ng + g];
                    acc[m] += xs * (ns * (float)ss * (float)di - nm * (float)sm * (float)mi);
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MPW; ++m) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], o);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < MPW; ++m) {
            int gm = msplit * MPW + m;
            if (gm < M) {
                out[(size_t)gm * N + row] = __float2half(acc[m]);
            }
        }
    }
}

template <int BITS, int GS>
__global__ void __launch_bounds__(256) gemv_packed_bits_swiglu_pair_kernel(
    const uint8_t* __restrict__ q_packed,    // [2*N, ng, ceil(GS*BITS/8)], rows [gate, up]
    const uint8_t* __restrict__ sub_scale,   // [2*N, ng]
    const uint8_t* __restrict__ sub_min,     // [2*N, ng]
    const float* __restrict__ neuron_scale,  // [2*N]
    const float* __restrict__ neuron_min,    // [2*N]
    const int8_t* __restrict__ qx,           // [1, K_pad]
    const float* __restrict__ xscale,        // [1, ng]
    __half* __restrict__ out,                // [1, N]
    int N, int ng, int K_pad, int activation)
{
    constexpr int PPB = 4;
    constexpr int QBYTES = (GS * BITS + 7) / 8;
    constexpr int CHUNKS = (GS + 3) / 4;
    constexpr int GPW = 32 / CHUNKS;
    int pair = blockIdx.x * PPB + (threadIdx.y >> 1);
    int half_id = threadIdx.y & 1;
    int lane = threadIdx.x;
    int row = pair + (half_id ? N : 0);

    __shared__ float vals[PPB * 2];
    float acc = 0.0f;
    if (pair < N) {
        const uint8_t* qrow = q_packed + (size_t)row * ng * QBYTES;
        const uint8_t* ssrow = sub_scale + (size_t)row * ng;
        const uint8_t* smrow = sub_min + (size_t)row * ng;
        float ns = neuron_scale[row];
        float nm = neuron_min[row];

        int relg = lane / CHUNKS;
        int ci = lane - relg * CHUNKS;
        int off = ci * 4;
        bool active_lane = relg < GPW;
        bool full4 = active_lane && ((off + 3) < GS);
        bool tail = active_lane && (off < GS) && !full4;
        for (int gb = 0; gb < ng; gb += GPW) {
            int g = gb + relg;
            if (!active_lane || g >= ng) {
                continue;
            }
            const uint8_t* qg = qrow + (size_t)g * QBYTES;
            uint8_t ss = ssrow[g];
            uint8_t sm = smrow[g];
            float xs = xscale[g];
            if (full4) {
                int qv = unpack_qbits4<BITS>(qg, off);
                int k = g * GS + off;
                int xv = load_i8x4_unaligned(qx + k);
                int mi = __dp4a(0x01010101, xv, 0);
                int di = dp4a_qbits_xs8<BITS>(qv, xv, mi);
                acc += xs * (ns * (float)ss * (float)di - nm * (float)sm * (float)mi);
            } else if (tail) {
                int di = 0;
                int mi = 0;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int gi = off + r;
                    if (gi < GS) {
                        int qi = (int)qx[(size_t)g * GS + gi];
                        di += (int)unpack_qbits_one<BITS>(qg, gi) * qi;
                        mi += qi;
                    }
                }
                acc += xs * (ns * (float)ss * (float)di - nm * (float)sm * (float)mi);
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, o);
        }
    }
    if (lane == 0) {
        vals[threadIdx.y] = acc;
    }
    __syncthreads();
    if (lane == 0 && threadIdx.y < PPB) {
        int p = blockIdx.x * PPB + threadIdx.y;
        if (p < N) {
            float gate = vals[threadIdx.y * 2 + 0];
            float up   = vals[threadIdx.y * 2 + 1];
            gate = __half2float(__float2half(gate));
            up   = __half2float(__float2half(up));
            out[p] = __float2half(mfq_glu_runtime(gate, up, activation));
        }
    }
}

template <int BITS, int GS, int WPB>
__global__ void __launch_bounds__(WPB * 32) gemv_packed_bits_glu_combined_pair_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    __half* __restrict__ out,
    int N, int ng, int K_pad, int activation)
{
    constexpr int QBYTES = (GS * BITS + 7) / 8;
    constexpr int CHUNKS = (GS + 3) / 4;
    constexpr int GPW = 32 / CHUNKS;
    const int pair = blockIdx.x * WPB + threadIdx.y;
    const int lane = threadIdx.x;
    if (pair >= N) {
        return;
    }

    const int up_row = pair + N;
    const uint8_t* qgate = q_packed + (size_t)pair * ng * QBYTES;
    const uint8_t* qup = q_packed + (size_t)up_row * ng * QBYTES;
    const uint8_t* ssg = sub_scale + (size_t)pair * ng;
    const uint8_t* smg = sub_min + (size_t)pair * ng;
    const uint8_t* ssu = sub_scale + (size_t)up_row * ng;
    const uint8_t* smu = sub_min + (size_t)up_row * ng;
    const float nsg = neuron_scale[pair];
    const float nmg = neuron_min[pair];
    const float nsu = neuron_scale[up_row];
    const float nmu = neuron_min[up_row];

    const int relg = lane / CHUNKS;
    const int ci = lane - relg * CHUNKS;
    const int off = ci * 4;
    const bool active_lane = relg < GPW;
    const bool full4 = active_lane && off + 3 < GS;
    const bool tail = active_lane && off < GS && !full4;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int gb = 0; gb < ng; gb += GPW) {
        const int g = gb + relg;
        if (!active_lane || g >= ng) {
            continue;
        }
        const uint8_t* qg = qgate + (size_t)g * QBYTES;
        const uint8_t* qu = qup + (size_t)g * QBYTES;
        const float xs = xscale[g];
        if (full4) {
            const int qvg = unpack_qbits4<BITS>(qg, off);
            const int qvu = unpack_qbits4<BITS>(qu, off);
            const int xv = load_i8x4_unaligned(qx + (size_t)g * GS + off);
            const int mi = __dp4a(0x01010101, xv, 0);
            const int dig = dp4a_qbits_xs8<BITS>(qvg, xv, mi);
            const int diu = dp4a_qbits_xs8<BITS>(qvu, xv, mi);
            gate_acc += xs * (nsg * (float)ssg[g] * (float)dig -
                              nmg * (float)smg[g] * (float)mi);
            up_acc += xs * (nsu * (float)ssu[g] * (float)diu -
                            nmu * (float)smu[g] * (float)mi);
        } else if (tail) {
            int dig = 0;
            int diu = 0;
            int mi = 0;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int gi = off + r;
                if (gi < GS) {
                    const int xv = (int)qx[(size_t)g * GS + gi];
                    dig += (int)unpack_qbits_one<BITS>(qg, gi) * xv;
                    diu += (int)unpack_qbits_one<BITS>(qu, gi) * xv;
                    mi += xv;
                }
            }
            gate_acc += xs * (nsg * (float)ssg[g] * (float)dig -
                              nmg * (float)smg[g] * (float)mi);
            up_acc += xs * (nsu * (float)ssu[g] * (float)diu -
                            nmu * (float)smu[g] * (float)mi);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
        up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
    }
    if (lane == 0) {
        const float gate = __half2float(__float2half(gate_acc));
        const float up = __half2float(__float2half(up_acc));
        out[pair] = __float2half(mfq_glu_runtime(gate, up, activation));
    }
}

template <int MTILE>
__global__ void __launch_bounds__(128) gemv_nint6_gs26_batch_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, 20]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int GS = 26;
    constexpr int QBYTES = 20;
    constexpr int WPB = 4;
    constexpr int CHUNKS = 7;
    constexpr int GPW = 4;

    int row = blockIdx.x * WPB + threadIdx.y;
    int m0 = blockIdx.y * MTILE;
    int lane = threadIdx.x;
    if (row >= N) {
        return;
    }

    const uint8_t* qrow = q_packed + (size_t)row * ng * QBYTES;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min + (size_t)row * ng;
    float ns = neuron_scale[row];
    float nm = neuron_min[row];

    int relg = lane / CHUNKS;
    int ci = lane - relg * CHUNKS;
    int off = ci * 4;
    bool active_lane = relg < GPW;
    bool full4 = active_lane && (ci < 6);
    bool tail2 = active_lane && (ci == 6);

    float acc[MTILE];
    #pragma unroll
    for (int mi = 0; mi < MTILE; ++mi) {
        acc[mi] = 0.0f;
    }

    for (int gb = 0; gb < ng; gb += GPW) {
        int g = gb + relg;
        if (!active_lane || g >= ng) {
            continue;
        }
        const uint8_t* qg = qrow + (size_t)g * QBYTES;
        uint8_t ss = ssrow[g];
        uint8_t sm = smrow[g];
        int k = g * GS + off;
        if (full4) {
            int qv = unpack_qbits4<6>(qg, off);
            #pragma unroll
            for (int mi = 0; mi < MTILE; ++mi) {
                int m = m0 + mi;
                if (m < M) {
                    const int8_t* qxrow = qx + (size_t)m * K_pad;
                    int xv = load_i8x4_unaligned(qxrow + k);
                    int di = __dp4a(qv, xv, 0);
                    int smi = __dp4a(0x01010101, xv, 0);
                    float xs = xscale[(size_t)m * ng + g];
                    acc[mi] += xs * (ns * (float)ss * (float)di - nm * (float)sm * (float)smi);
                }
            }
        } else if (tail2) {
            int q24 = (int)unpack_qbits_one<6>(qg, 24);
            int q25 = (int)unpack_qbits_one<6>(qg, 25);
            #pragma unroll
            for (int mi = 0; mi < MTILE; ++mi) {
                int m = m0 + mi;
                if (m < M) {
                    const int8_t* qxrow = qx + (size_t)m * K_pad;
                    int x24 = (int)qxrow[(size_t)g * GS + 24];
                    int x25 = (int)qxrow[(size_t)g * GS + 25];
                    int di = q24 * x24 + q25 * x25;
                    int smi = x24 + x25;
                    float xs = xscale[(size_t)m * ng + g];
                    acc[mi] += xs * (ns * (float)ss * (float)di - nm * (float)sm * (float)smi);
                }
            }
        }
    }

    #pragma unroll
    for (int mi = 0; mi < MTILE; ++mi) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], o);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int mi = 0; mi < MTILE; ++mi) {
            int m = m0 + mi;
            if (m < M) {
                out[(size_t)m * N + row] = __float2half(acc[mi]);
            }
        }
    }
}

template <int MTILE>
__global__ void __launch_bounds__(128) gemv_nint5_gs28_batch_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, 18]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int GS = 28;
    constexpr int QBYTES = 18;
    constexpr int WPB = 4;
    constexpr int CHUNKS = 7;
    constexpr int GPW = 4;

    int row = blockIdx.x * WPB + threadIdx.y;
    int m0 = blockIdx.y * MTILE;
    int lane = threadIdx.x;
    if (row >= N) {
        return;
    }

    const uint8_t* qrow = q_packed + (size_t)row * ng * QBYTES;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min + (size_t)row * ng;
    float ns = neuron_scale[row];
    float nm = neuron_min[row];

    int relg = lane / CHUNKS;
    int ci = lane - relg * CHUNKS;
    int off = ci * 4;
    bool active_lane = relg < GPW;

    float acc[MTILE];
    #pragma unroll
    for (int mi = 0; mi < MTILE; ++mi) {
        acc[mi] = 0.0f;
    }

    for (int gb = 0; gb < ng; gb += GPW) {
        int g = gb + relg;
        if (!active_lane || g >= ng) {
            continue;
        }
        const uint8_t* qg = qrow + (size_t)g * QBYTES;
        int qv = unpack_qbits4<5>(qg, off);
        int k = g * GS + off;
        uint8_t ss = ssrow[g];
        uint8_t sm = smrow[g];
        #pragma unroll
        for (int mi = 0; mi < MTILE; ++mi) {
            int m = m0 + mi;
            if (m < M) {
                const int8_t* qxrow = qx + (size_t)m * K_pad;
                int xv = load_i8x4_unaligned(qxrow + k);
                int di = __dp4a(qv, xv, 0);
                int smi = __dp4a(0x01010101, xv, 0);
                float xs = xscale[(size_t)m * ng + g];
                acc[mi] += xs * (ns * (float)ss * (float)di - nm * (float)sm * (float)smi);
            }
        }
    }

    #pragma unroll
    for (int mi = 0; mi < MTILE; ++mi) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], o);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int mi = 0; mi < MTILE; ++mi) {
            int m = m0 + mi;
            if (m < M) {
                out[(size_t)m * N + row] = __float2half(acc[mi]);
            }
        }
    }
}

template <int GS, int BD, int MODE>
__global__ void quantize_x_gate_nosum_kernel(
    const __half* __restrict__ x,     // [M, K_real]
    const __half* __restrict__ gate,  // [M, K_real]
    int8_t* __restrict__ qx,          // [M, K_pad]
    float* __restrict__ xscale,       // [M, ng]
    int M, int K_real, int K_pad)
{
    int m = blockIdx.x;
    int g = blockIdx.y;
    int tid = threadIdx.x;
    int ng = gridDim.y;
    int base = g * GS;
    bool real = (tid < GS) && (base + tid < K_real);
    float xv = 0.0f;
    if (real) {
        size_t off = (size_t)m * K_real + base + tid;
        float gv = __half2float(gate[off]);
        float sig = 1.0f / (1.0f + expf(-gv));
        float mult = MODE == 1 ? sig : gv * sig;
        xv = __half2float(x[off]) * mult;
    }

    float amax = block_max<BD / 32>(fabsf(xv));
    float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    int qi = 0;
    if (tid < GS && real) {
        float q = roundf(xv / scale);
        q = fminf(fmaxf(q, -127.0f), 127.0f);
        qi = (int)q;
    }
    if (tid == 0) {
        xscale[(size_t)m * ng + g] = scale;
    }
    if (tid < GS) {
        qx[(size_t)m * K_pad + base + tid] = real ? (int8_t)qi : (int8_t)0;
    }
}

template <int WPB>
__global__ void __launch_bounds__(512) gemv_nint6_gs26_argmax_stage1_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, 20]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [1, K_pad]
    const float* __restrict__ xscale,        // [1, ng]
    float* __restrict__ block_vals,
    int* __restrict__ block_idxs,
    int N, int ng, int K_pad)
{
    constexpr int GS = 26;
    constexpr int QBYTES = 20;
    constexpr int CHUNKS = 7;
    constexpr int GPW = 4;

    int row = blockIdx.x * WPB + threadIdx.y;
    int lane = threadIdx.x;
    int relg = lane / CHUNKS;
    int ci = lane - relg * CHUNKS;
    int off = ci * 4;
    bool active_lane = relg < GPW;
    bool full4 = active_lane && (ci < 6);
    bool tail2 = active_lane && (ci == 6);
    float acc = 0.0f;

    if (row < N) {
        const uint8_t* qrow = q_packed + (size_t)row * ng * QBYTES;
        const uint8_t* ssrow = sub_scale + (size_t)row * ng;
        const uint8_t* smrow = sub_min + (size_t)row * ng;
        float ns = neuron_scale[row];
        float nm = neuron_min[row];
        for (int gb = 0; gb < ng; gb += GPW) {
            int g = gb + relg;
            if (!active_lane || g >= ng) {
                continue;
            }
            const uint8_t* qg = qrow + (size_t)g * QBYTES;
            uint8_t ss = ssrow[g];
            uint8_t sm = smrow[g];
            int k = g * GS + off;
            if (full4) {
                int qv = unpack_qbits4<6>(qg, off);
                int xv = load_i8x4_unaligned(qx + k);
                int di = __dp4a(qv, xv, 0);
                int smi = __dp4a(0x01010101, xv, 0);
                float xs = xscale[g];
                acc += xs * (ns * (float)ss * (float)di - nm * (float)sm * (float)smi);
            } else if (tail2) {
                int q24 = (int)unpack_qbits_one<6>(qg, 24);
                int q25 = (int)unpack_qbits_one<6>(qg, 25);
                int x24 = (int)qx[(size_t)g * GS + 24];
                int x25 = (int)qx[(size_t)g * GS + 25];
                int di = q24 * x24 + q25 * x25;
                int smi = x24 + x25;
                float xs = xscale[g];
                acc += xs * (ns * (float)ss * (float)di - nm * (float)sm * (float)smi);
            }
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, o);
    }

    __shared__ float vals[WPB];
    __shared__ int idxs[WPB];
    if (lane == 0) {
        vals[threadIdx.y] = row < N ? __half2float(__float2half(acc)) : -FLT_MAX;
        idxs[threadIdx.y] = row < N ? row : INT_MAX;
    }
    __syncthreads();
    if (threadIdx.y == 0 && lane == 0) {
        float best = vals[0];
        int best_i = idxs[0];
        #pragma unroll
        for (int i = 1; i < WPB; ++i) {
            float v = vals[i];
            int idx = idxs[i];
            if (v > best || (v == best && idx < best_i)) {
                best = v;
                best_i = idx;
            }
        }
        block_vals[blockIdx.x] = best;
        block_idxs[blockIdx.x] = best_i;
    }
}

__device__ __forceinline__ int2 unpack_int5x8(const uint8_t* p)
{
    const uint64_t w = (uint64_t)p[0] |
                       ((uint64_t)p[1] << 8) |
                       ((uint64_t)p[2] << 16) |
                       ((uint64_t)p[3] << 24) |
                       ((uint64_t)p[4] << 32);
    const int lo = (int)((w      ) & 31u) |
                   (int)((w >>  5) & 31u) << 8 |
                   (int)((w >> 10) & 31u) << 16 |
                   (int)((w >> 15) & 31u) << 24;
    const int hi = (int)((w >> 20) & 31u) |
                   (int)((w >> 25) & 31u) << 8 |
                   (int)((w >> 30) & 31u) << 16 |
                   (int)((w >> 35) & 31u) << 24;
    return make_int2(lo, hi);
}

__device__ __forceinline__ float dot_nint5_gs28_q5_exec(
    const uint8_t* __restrict__ q_packed,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    int row, int lane, int m, int ng, int K_pad,
    int group_start, int group_stride)
{
    constexpr int GS = 28;
    constexpr int CHUNKS = 4;
    const int relg = lane / CHUNKS;
    const int ci = lane - relg * CHUNKS;
    const int off = ci * 8;
    const uint8_t* qrow = q_packed + (size_t)row * ng * 20;
    const int8_t* qxrow = qx + (size_t)m * K_pad;
    const float* xsrow = xscale + (size_t)m * ng;
    const float ns = neuron_scale[row];
    const float nm = neuron_min[row];
    float acc = 0.0f;

    for (int gb = group_start; gb < ng; gb += group_stride) {
        const int g = gb + relg;
        if (g >= ng) {
            continue;
        }
        const uint8_t* qg = qrow + (size_t)g * 20;
        const uint32_t ql = *reinterpret_cast<const uint32_t*>(qg + ci * 4);
        const uint32_t qh =
            (uint32_t)*reinterpret_cast<const uint16_t*>(qg + 14) |
            ((uint32_t)*reinterpret_cast<const uint16_t*>(qg + 16) << 16);
        const int qeven = (int)((ql & 0x0f0f0f0f) | (((qh >> ci) << 4) & 0x10101010));
        const int qodd = (int)(((ql >> 4) & 0x0f0f0f0f) | (((qh >> (ci + 4)) << 4) & 0x10101010));

        const int8_t* xg = qxrow + (size_t)g * GS + off;
        const int xlo = *reinterpret_cast<const int*>(xg);
        const int xhi = ci < 3 ? *reinterpret_cast<const int*>(xg + 4) : 0;
        const int xeven = __byte_perm(xlo, xhi, 0x6420);
        const int xodd = __byte_perm(xlo, xhi, 0x7531);
        const int di = __dp4a(qodd, xodd, __dp4a(qeven, xeven, 0));
        const int smi = __dp4a(0x01010101, xodd, __dp4a(0x01010101, xeven, 0));
        const uint16_t affine = *reinterpret_cast<const uint16_t*>(qg + 18);
        const float xs = xsrow[g];
        acc += xs * (ns * (float)(affine & 0xffu) * (float)di -
                     nm * (float)(affine >> 8) * (float)smi);
    }
    return acc;
}

template <int WPB>
__global__ void __launch_bounds__(WPB * 32, 1) gemv_nint5_gs28_q5_exec_kernel(
    const uint8_t* __restrict__ q_packed,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    __half* __restrict__ out,
    int M, int N, int ng, int K_pad)
{
    const int row = blockIdx.x * WPB + threadIdx.y;
    const int lane = threadIdx.x;
    const int m = blockIdx.y;
    float acc = row < N
        ? dot_nint5_gs28_q5_exec(
              q_packed, neuron_scale, neuron_min,
              qx, xscale, row, lane, m, ng, K_pad, 0, 8)
        : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0 && row < N) {
        out[(size_t)m * N + row] = __float2half(acc);
    }
}

template <int WPB>
__global__ void __launch_bounds__(WPB * 32, 1) gemv_nint5_gs28_q5_exec_argmax_stage1_kernel(
    const uint8_t* __restrict__ q_packed,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    float* __restrict__ block_vals,
    int* __restrict__ block_idxs,
    int N, int ng, int K_pad)
{
    const int row = blockIdx.x * WPB + threadIdx.y;
    const int lane = threadIdx.x;
    float acc = row < N
        ? dot_nint5_gs28_q5_exec(
              q_packed, neuron_scale, neuron_min,
              qx, xscale, row, lane, 0, ng, K_pad, 0, 8)
        : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    __shared__ float vals[WPB];
    __shared__ int idxs[WPB];
    if (lane == 0) {
        vals[threadIdx.y] = row < N ? __half2float(__float2half(acc)) : -FLT_MAX;
        idxs[threadIdx.y] = row < N ? row : INT_MAX;
    }
    __syncthreads();
    if (threadIdx.y == 0 && lane == 0) {
        float best = vals[0];
        int best_i = idxs[0];
#pragma unroll
        for (int i = 1; i < WPB; ++i) {
            const float v = vals[i];
            const int idx = idxs[i];
            if (v > best || (v == best && idx < best_i)) {
                best = v;
                best_i = idx;
            }
        }
        block_vals[blockIdx.x] = best;
        block_idxs[blockIdx.x] = best_i;
    }
}

template <int WPB>
__global__ void __launch_bounds__(WPB * 32, 1) gemv_nint5_gs28_argmax_stage1_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, 18]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [1, K_pad]
    const float* __restrict__ xscale,        // [1, ng]
    float* __restrict__ block_vals,
    int* __restrict__ block_idxs,
    int N, int ng, int K_pad)
{
    constexpr int GS = 28;
    constexpr int QBYTES = 18;
    constexpr int CHUNKS = 4;
    constexpr int GPW = 8;

    const int row = blockIdx.x * WPB + threadIdx.y;
    const int lane = threadIdx.x;
    const int relg = lane / CHUNKS;
    const int ci = lane - relg * CHUNKS;
    const int off = ci * 8;
    float acc = 0.0f;

    if (row < N) {
        const uint8_t* qrow = q_packed + (size_t)row * ng * QBYTES;
        const uint8_t* ssrow = sub_scale + (size_t)row * ng;
        const uint8_t* smrow = sub_min + (size_t)row * ng;
        const float ns = neuron_scale[row];
        const float nm = neuron_min[row];
        for (int gb = 0; gb < ng; gb += GPW) {
            const int g = gb + relg;
            if (g >= ng) {
                continue;
            }
            const uint8_t* qg = qrow + (size_t)g * QBYTES;
            const int8_t* xg = qx + (size_t)g * GS;
            int di;
            int smi;
            if (ci < 3) {
                const int2 qv = unpack_int5x8(qg + ci * 5);
                const int2 xv = make_int2(
                    load_i8x4_unaligned(xg + off),
                    load_i8x4_unaligned(xg + off + 4));
                di = __dp4a(qv.y, xv.y, __dp4a(qv.x, xv.x, 0));
                smi = __dp4a(0x01010101, xv.y, __dp4a(0x01010101, xv.x, 0));
            } else {
                const int qv = unpack_qbits4<5>(qg, 24);
                const int xv = load_i8x4_unaligned(xg + 24);
                di = __dp4a(qv, xv, 0);
                smi = __dp4a(0x01010101, xv, 0);
            }
            const float xs = xscale[g];
            acc += xs * (ns * (float)ssrow[g] * (float)di -
                         nm * (float)smrow[g] * (float)smi);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    __shared__ float vals[WPB];
    __shared__ int idxs[WPB];
    if (lane == 0) {
        vals[threadIdx.y] = row < N ? __half2float(__float2half(acc)) : -FLT_MAX;
        idxs[threadIdx.y] = row < N ? row : INT_MAX;
    }
    __syncthreads();
    if (threadIdx.y == 0 && lane == 0) {
        float best = vals[0];
        int best_i = idxs[0];
#pragma unroll
        for (int i = 1; i < WPB; ++i) {
            const float v = vals[i];
            const int idx = idxs[i];
            if (v > best || (v == best && idx < best_i)) {
                best = v;
                best_i = idx;
            }
        }
        block_vals[blockIdx.x] = best;
        block_idxs[blockIdx.x] = best_i;
    }
}

template <int WPB>
__global__ void __launch_bounds__(512) gemv_nint6_gs24_argmax_stage1_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, 18]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [1, K_pad]
    const float* __restrict__ xscale,        // [1, ng]
    float* __restrict__ block_vals,
    int* __restrict__ block_idxs,
    int N, int ng, int K_pad)
{
    constexpr int GS = 24;
    constexpr int QBYTES = 18;
    int row = blockIdx.x * WPB + threadIdx.y;
    int lane = threadIdx.x;
    float pd = 0.0f;
    float pm = 0.0f;

    if (row < N) {
        const uint8_t* qrow = q_packed + (size_t)row * ng * QBYTES;
        const uint8_t* ssrow = sub_scale + (size_t)row * ng;
        const uint8_t* smrow = sub_min + (size_t)row * ng;
        constexpr int STRIDE = 32 * 4;
        for (int base = lane * 4; base < K_pad; base += STRIDE) {
            int qv = unpack_int6x4(qrow + base * 6 / 8);
            int xv = *reinterpret_cast<const int*>(qx + base);
            int g = base / GS;
            float xs = xscale[g];
            pd += xs * (float)ssrow[g] * (float)__dp4a(qv, xv, 0);
            pm += xs * (float)smrow[g] * (float)__dp4a(0x01010101, xv, 0);
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        pd += __shfl_xor_sync(0xffffffff, pd, o);
        pm += __shfl_xor_sync(0xffffffff, pm, o);
    }

    __shared__ float vals[WPB];
    __shared__ int idxs[WPB];
    if (lane == 0) {
        float v = row < N ? neuron_scale[row] * pd - neuron_min[row] * pm : -FLT_MAX;
        vals[threadIdx.y] = row < N ? __half2float(__float2half(v)) : -FLT_MAX;
        idxs[threadIdx.y] = row < N ? row : INT_MAX;
    }
    __syncthreads();
    if (threadIdx.y == 0 && lane == 0) {
        float best = vals[0];
        int best_i = idxs[0];
        #pragma unroll
        for (int i = 1; i < WPB; ++i) {
            float v = vals[i];
            int idx = idxs[i];
            if (v > best || (v == best && idx < best_i)) {
                best = v;
                best_i = idx;
            }
        }
        block_vals[blockIdx.x] = best;
        block_idxs[blockIdx.x] = best_i;
    }
}

__global__ void nint_argmax_reduce_kernel(
    const float* __restrict__ vals,
    const int* __restrict__ idxs,
    int64_t* __restrict__ out,
    int nb)
{
    __shared__ float warp_vals[8];
    __shared__ int warp_idxs[8];
    float best = -FLT_MAX;
    int best_i = INT_MAX;
    for (int i = threadIdx.x; i < nb; i += blockDim.x) {
        float v = vals[i];
        int idx = idxs[i];
        if (v > best || (v == best && idx < best_i)) {
            best = v;
            best_i = idx;
        }
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float v = __shfl_down_sync(0xffffffff, best, offset);
        int idx = __shfl_down_sync(0xffffffff, best_i, offset);
        if (lane + offset < 32 && (v > best || (v == best && idx < best_i))) {
            best = v;
            best_i = idx;
        }
    }
    if (lane == 0) {
        warp_vals[warp] = best;
        warp_idxs[warp] = best_i;
    }
    __syncthreads();
    if (warp == 0) {
        best = lane < 8 ? warp_vals[lane] : -FLT_MAX;
        best_i = lane < 8 ? warp_idxs[lane] : INT_MAX;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float v = __shfl_down_sync(0xffffffff, best, offset);
            int idx = __shfl_down_sync(0xffffffff, best_i, offset);
            if (lane + offset < 32 && (v > best || (v == best && idx < best_i))) {
                best = v;
                best_i = idx;
            }
        }
        if (lane == 0) {
            out[0] = (int64_t)best_i;
        }
    }
}

__global__ void linear_out_head_rinv_kernel(
    const float* __restrict__ y,      // [H, D]
    float* __restrict__ rinv,         // [H]
    int H, int D, float eps)
{
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= H) {
        return;
    }
    float ssq = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = y[(size_t)h * D + d];
        ssq += v * v;
    }
    ssq = block_sum<4>(ssq);
    if (tid == 0) {
        rinv[h] = rsqrtf(ssq / (float)D + eps);
    }
}

template <int GS, int BD>
__global__ void linear_out_norm_gate_quant_kernel(
    const float* __restrict__ y,       // [H, D]
    const __half* __restrict__ z,      // [K_real]
    const float* __restrict__ w,       // [D]
    const float* __restrict__ rinv,    // [H]
    int8_t* __restrict__ qx,           // [K_pad]
    float* __restrict__ xscale,        // [ng]
    int H, int D, int K_real, int K_pad)
{
    int g = blockIdx.x;
    int tid = threadIdx.x;
    int k = g * GS + tid;
    float xv = 0.0f;
    if (tid < GS && k < K_real) {
        int h = k / D;
        int d = k - h * D;
        float yn = y[(size_t)h * D + d] * rinv[h] * w[d];
        yn = __half2float(__float2half(yn));
        float zg = __half2float(z[k]);
        float sig = 1.0f / (1.0f + expf(-zg));
        xv = yn * (zg * sig);
    }
    float amax = block_max<BD / 32>(fabsf(xv));
    float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    int qi = 0;
    if (tid < GS) {
        if (k < K_real) {
            float q = roundf(xv / scale);
            q = fminf(fmaxf(q, -127.0f), 127.0f);
            qi = (int)q;
        }
        qx[k] = (int8_t)qi;
    }
    if (tid == 0) {
        xscale[g] = scale;
    }
}

template <int GS, int MAX_M>
__global__ void __launch_bounds__(128) gemv_packed_u8_batch_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, GS]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    const int32_t* __restrict__ xsum,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int WPB = 4;
    int row = blockIdx.x * WPB + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= N) {
        return;
    }

    const uint8_t* qrow = q_packed + (size_t)row * ng * GS;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min + (size_t)row * ng;
    float ns = neuron_scale[row];
    float nm = neuron_min[row];

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc[m] = 0.0f;
    }

    constexpr int STRIDE = 32 * 4;
    for (int base = lane * 4; base < K_pad; base += STRIDE) {
        int qv = *reinterpret_cast<const int*>(qrow + base);
        int qvc = qv ^ (int)0x80808080u;  // signed bytes are q-128
        int g = base / GS;
        uint8_t ss = ssrow[g];
        uint8_t sm = smrow[g];
        float de = ns * (float)ss;
        float me = nm * (float)sm;
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const int8_t* qxrow = qx + (size_t)m * K_pad;
                int xv = *reinterpret_cast<const int*>(qxrow + base);
                int di = __dp4a(qvc, xv, 0);
                int sumi = __dp4a(0x01010101, xv, 0);
                float xs = xscale[(size_t)m * ng + g];
                acc[m] += xs * (de * ((float)di + 128.0f * (float)sumi) - me * (float)sumi);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], o);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                out[(size_t)m * N + row] = __float2half(acc[m]);
            }
        }
    }
}

template <int GS>
__global__ void __launch_bounds__(128) gemv_packed_u8_groupwise_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, GS]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [B * groups, K_pad]
    const float* __restrict__ xscale,        // [B * groups, ng]
    __half* __restrict__ out,                // [B, N]
    int B, int groups, int rows_per_group, int N, int ng, int K_pad)
{
    constexpr int WPB = 4;
    const int output_index = blockIdx.x * WPB + threadIdx.y;
    const int lane = threadIdx.x;
    const int total_outputs = B * N;
    if (output_index >= total_outputs) {
        return;
    }

    const int batch = output_index / N;
    const int row = output_index - batch * N;
    const int group = row / rows_per_group;
    const int input_row = batch * groups + group;
    const uint8_t* qrow = q_packed + (size_t)row * ng * GS;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min + (size_t)row * ng;
    const int8_t* qxrow = qx + (size_t)input_row * K_pad;
    const float* xsrow = xscale + (size_t)input_row * ng;
    const float ns = neuron_scale[row];
    const float nm = neuron_min[row];
    float acc = 0.0f;

    constexpr int STRIDE = 32 * 4;
    for (int base = lane * 4; base < K_pad; base += STRIDE) {
        const int qv = *reinterpret_cast<const int*>(qrow + base);
        const int qvc = qv ^ (int)0x80808080u;
        const int xv = *reinterpret_cast<const int*>(qxrow + base);
        const int di = __dp4a(qvc, xv, 0);
        const int sumi = __dp4a(0x01010101, xv, 0);
        const int g = base / GS;
        const float de = ns * (float)ssrow[g];
        const float me = nm * (float)smrow[g];
        const float xs = xsrow[g];
        acc += xs * (de * ((float)di + 128.0f * (float)sumi) - me * (float)sumi);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        out[output_index] = __float2half(acc);
    }
}

template <int GS, bool USE_XSUM=false>
__global__ void __launch_bounds__(128) gemv_kernel(
    const uint8_t* __restrict__ q,           // [N, ng, GS]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    const int32_t* __restrict__ xsum,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int WPB = 4;   // warps per block
    int row = blockIdx.x * WPB + threadIdx.y;
    int m = blockIdx.y;
    int lane = threadIdx.x;
    if (row >= N || m >= M) {
        return;
    }

    const uint8_t* qrow  = q + (size_t)row * ng * GS;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min   + (size_t)row * ng;
    const int8_t*  qxrow = qx + (size_t)m * K_pad;
    const float*   xsrow = xscale + (size_t)m * ng;
    const int32_t* xsumrow = xsum + (size_t)m * ng;

    float pd = 0.0f, pm = 0.0f;
    constexpr int STRIDE = 32 * 4;   // one 128-element chunk per warp per step
    // K_pad and base are both multiples of 4, so base < K_pad implies base+3 < K_pad.
    for (int base = lane * 4; base < K_pad; base += STRIDE) {
        int qv = *reinterpret_cast<const int*>(qrow + base);
        int xv = *reinterpret_cast<const int*>(qxrow + base);
        int di = __dp4a(qv, xv, 0);             // sum q.qx over 4 elems
        int g = base / GS;                       // GS multiple of 4 -> 4 elems share group g
        float xs = xsrow[g];
        pd += xs * (float)ssrow[g] * (float)di;
        if constexpr (USE_XSUM) {
            if ((base % GS) == 0) {
                pm += xs * (float)smrow[g] * (float)xsumrow[g];
            }
        } else {
            int mi = __dp4a(0x01010101, xv, 0);
            pm += xs * (float)smrow[g] * (float)mi;
        }
    }
    // warp reduce
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        pd += __shfl_xor_sync(0xffffffff, pd, o);
        pm += __shfl_xor_sync(0xffffffff, pm, o);
    }
    if (lane == 0) {
        out[(size_t)m * N + row] = __float2half(neuron_scale[row] * pd - neuron_min[row] * pm);
    }
}

template <int GS, bool USE_XSUM=false>
__global__ void __launch_bounds__(256) gemv_packed_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, GS/2]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    const int32_t* __restrict__ xsum,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int WPB = 8;
    int row = blockIdx.x * WPB + threadIdx.y;
    int m = blockIdx.y;
    int lane = threadIdx.x;
    if (row >= N || m >= M) {
        return;
    }

    const uint8_t* qrow  = q_packed + (size_t)row * ng * (GS / 2);
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min   + (size_t)row * ng;
    const int8_t*  qxrow = qx + (size_t)m * K_pad;
    const float*   xsrow = xscale + (size_t)m * ng;
    const int32_t* xsumrow = xsum + (size_t)m * ng;

    float pd = 0.0f, pm = 0.0f;
    constexpr int STRIDE = 32 * 4;
    for (int base = lane * 4; base < K_pad; base += STRIDE) {
        int qv = unpack_int4x4(qrow + base / 2);
        int xv = *reinterpret_cast<const int*>(qxrow + base);
        int di = __dp4a(qv, xv, 0);
        int g = base / GS;
        float xs = xsrow[g];
        pd += xs * (float)ssrow[g] * (float)di;
        if constexpr (USE_XSUM) {
            if ((base % GS) == 0) {
                pm += xs * (float)smrow[g] * (float)xsumrow[g];
            }
        } else {
            int mi = __dp4a(0x01010101, xv, 0);
            pm += xs * (float)smrow[g] * (float)mi;
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        pd += __shfl_xor_sync(0xffffffff, pd, o);
        pm += __shfl_xor_sync(0xffffffff, pm, o);
    }
    if (lane == 0) {
        out[(size_t)m * N + row] = __float2half(neuron_scale[row] * pd - neuron_min[row] * pm);
    }
}

template <int GS, bool USE_XSUM=false, int NWARPS=4>
__global__ void __launch_bounds__(256) gemv_packed_multiwarp_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    const int32_t* __restrict__ xsum,
    __half* __restrict__ out,
    int M, int N, int ng, int K_pad)
{
    int row = blockIdx.x;
    int m = blockIdx.y;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    if (row >= N || m >= M) {
        return;
    }

    const uint8_t* qrow  = q_packed + (size_t)row * ng * (GS / 2);
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min   + (size_t)row * ng;
    const int8_t*  qxrow = qx + (size_t)m * K_pad;
    const float*   xsrow = xscale + (size_t)m * ng;
    const int32_t* xsumrow = xsum + (size_t)m * ng;

    float pd = 0.0f;
    float pm = 0.0f;
    constexpr int STRIDE = NWARPS * 32 * 4;
    for (int base = (warp * 32 + lane) * 4; base < K_pad; base += STRIDE) {
        int qv = unpack_int4x4(qrow + base / 2);
        int xv = *reinterpret_cast<const int*>(qxrow + base);
        int g = base / GS;
        float xs = xsrow[g];
        pd += xs * (float)ssrow[g] * (float)__dp4a(qv, xv, 0);
        if constexpr (USE_XSUM) {
            if ((base % GS) == 0) {
                pm += xs * (float)smrow[g] * (float)xsumrow[g];
            }
        } else {
            pm += xs * (float)smrow[g] * (float)__dp4a(0x01010101, xv, 0);
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        pd += __shfl_xor_sync(0xffffffff, pd, o);
        pm += __shfl_xor_sync(0xffffffff, pm, o);
    }
    __shared__ float partial_d[NWARPS];
    __shared__ float partial_m[NWARPS];
    if (lane == 0) {
        partial_d[warp] = pd;
        partial_m[warp] = pm;
    }
    __syncthreads();

    if (warp == 0) {
        pd = lane < NWARPS ? partial_d[lane] : 0.0f;
        pm = lane < NWARPS ? partial_m[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            pd += __shfl_xor_sync(0xffffffff, pd, o);
            pm += __shfl_xor_sync(0xffffffff, pm, o);
        }
        if (lane == 0) {
            out[(size_t)m * N + row] = __float2half(neuron_scale[row] * pd - neuron_min[row] * pm);
        }
    }
}

template <int GS, int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32, 1) gemv_packed_u8_m1_row_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    __half* __restrict__ out,
    int N, int ng, int K_pad)
{
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    const uint8_t* qrow = q_packed + (size_t)row * ng * GS;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min + (size_t)row * ng;
    const float ns = neuron_scale[row];
    const float nm = neuron_min[row];
    float acc = 0.0f;

    constexpr int STRIDE = NWARPS * 32 * 4;
    for (int base = (warp * 32 + lane) * 4; base < K_pad; base += STRIDE) {
        const int qv = *reinterpret_cast<const int*>(qrow + base);
        const int qvc = qv ^ (int)0x80808080u;
        const int xv = *reinterpret_cast<const int*>(qx + base);
        const int di = __dp4a(qvc, xv, 0);
        const int sumi = __dp4a(0x01010101, xv, 0);
        const int g = base / GS;
        const float de = ns * (float)ssrow[g];
        const float me = nm * (float)smrow[g];
        const float xs = xscale[g];
        acc += xs * (de * ((float)di + 128.0f * (float)sumi) - me * (float)sumi);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[NWARPS];
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) out[row] = __float2half(acc);
    }
}

template <int NWARPS=4, bool VEC_LOAD=false, bool OUTPUT_BF16=false>
__global__ void __launch_bounds__(256) gemv_packed_gs24_group_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    const int32_t*,
    void* __restrict__ out,
    int M, int N, int ng, int K_pad)
{
    constexpr int GS = 24;
    int row = blockIdx.x;
    int m = blockIdx.y;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    if (row >= N || m >= M) return;

    const uint8_t* qrow = q_packed + (size_t)row * ng * (GS / 2);
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min + (size_t)row * ng;
    const int8_t* qxrow = qx + (size_t)m * K_pad;
    const float* xsrow = xscale + (size_t)m * ng;
    float pd = 0.0f;
    float pm = 0.0f;

    for (int g = warp * 32 + lane; g < ng; g += NWARPS * 32) {
        int dsum = 0;
        int msum = 0;
        const uint8_t* qgroup = qrow + g * (GS / 2);
        uint32_t qw0 = 0;
        uint32_t qw1 = 0;
        uint32_t qw2 = 0;
        if constexpr (VEC_LOAD) {
            const uint32_t* qwords = reinterpret_cast<const uint32_t*>(qgroup);
            qw0 = qwords[0];
            qw1 = qwords[1];
            qw2 = qwords[2];
        }
        #pragma unroll
        for (int chunk = 0; chunk < 6; ++chunk) {
            int base = g * GS + chunk * 4;
            int xv = *reinterpret_cast<const int*>(qxrow + base);
            int qv;
            if constexpr (VEC_LOAD) {
                uint32_t qw = chunk < 2 ? qw0 : (chunk < 4 ? qw1 : qw2);
                qv = unpack_int4x4_u16(qw >> ((chunk & 1) * 16));
            } else {
                qv = unpack_int4x4(qgroup + chunk * 2);
            }
            dsum = __dp4a(qv, xv, dsum);
            msum = __dp4a(0x01010101, xv, msum);
        }
        float xs = xsrow[g];
        pd += xs * (float)ssrow[g] * (float)dsum;
        pm += xs * (float)smrow[g] * (float)msum;
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        pd += __shfl_xor_sync(0xffffffff, pd, o);
        pm += __shfl_xor_sync(0xffffffff, pm, o);
    }
    __shared__ float partial_d[NWARPS];
    __shared__ float partial_m[NWARPS];
    if (lane == 0) {
        partial_d[warp] = pd;
        partial_m[warp] = pm;
    }
    __syncthreads();
    if (warp == 0) {
        pd = lane < NWARPS ? partial_d[lane] : 0.0f;
        pm = lane < NWARPS ? partial_m[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            pd += __shfl_xor_sync(0xffffffff, pd, o);
            pm += __shfl_xor_sync(0xffffffff, pm, o);
        }
        if (lane == 0) {
            const size_t output_index = (size_t)m * N + row;
            const __half rounded = __float2half(
                neuron_scale[row] * pd - neuron_min[row] * pm);
            if constexpr (OUTPUT_BF16) {
                reinterpret_cast<__nv_bfloat16*>(out)[output_index] =
                    __float2bfloat16(__half2float(rounded));
            } else {
                reinterpret_cast<__half*>(out)[output_index] = rounded;
            }
        }
    }
}

struct Nint4Gs24Projection {
    const uint8_t* q_packed;
    const uint8_t* sub_scale;
    const uint8_t* sub_min;
    const float* neuron_scale;
    const float* neuron_min;
    void* out;
    int n;
};

template <int NPROJ, bool OUTPUT_BF16>
__global__ void __launch_bounds__(128) gemv_packed_gs24_multi_group_kernel(
    Nint4Gs24Projection first,
    Nint4Gs24Projection second,
    Nint4Gs24Projection third,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    int ng)
{
    constexpr int GS = 24;
    constexpr int NWARPS = 4;
    int row = blockIdx.x;
    Nint4Gs24Projection projection = first;
    if (row >= first.n) {
        row -= first.n;
        projection = second;
        if constexpr (NPROJ == 3) {
            if (row >= second.n) {
                row -= second.n;
                projection = third;
            }
        }
    }
    if (row >= projection.n) return;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const uint8_t* qrow =
        projection.q_packed + (size_t)row * ng * (GS / 2);
    const uint8_t* ssrow = projection.sub_scale + (size_t)row * ng;
    const uint8_t* smrow = projection.sub_min + (size_t)row * ng;
    float pd = 0.0f;
    float pm = 0.0f;

    for (int g = warp * 32 + lane; g < ng; g += NWARPS * 32) {
        int dsum = 0;
        int msum = 0;
        const uint8_t* qgroup = qrow + g * (GS / 2);
        const uint32_t* qwords =
            reinterpret_cast<const uint32_t*>(qgroup);
        const uint32_t qw0 = qwords[0];
        const uint32_t qw1 = qwords[1];
        const uint32_t qw2 = qwords[2];
        #pragma unroll
        for (int chunk = 0; chunk < 6; ++chunk) {
            const int base = g * GS + chunk * 4;
            const int xv = *reinterpret_cast<const int*>(qx + base);
            const uint32_t qw =
                chunk < 2 ? qw0 : (chunk < 4 ? qw1 : qw2);
            const int qv =
                unpack_int4x4_u16(qw >> ((chunk & 1) * 16));
            dsum = __dp4a(qv, xv, dsum);
            msum = __dp4a(0x01010101, xv, msum);
        }
        const float xs = xscale[g];
        pd += xs * (float)ssrow[g] * (float)dsum;
        pm += xs * (float)smrow[g] * (float)msum;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        pd += __shfl_xor_sync(0xffffffff, pd, offset);
        pm += __shfl_xor_sync(0xffffffff, pm, offset);
    }
    __shared__ float partial_d[NWARPS];
    __shared__ float partial_m[NWARPS];
    if (lane == 0) {
        partial_d[warp] = pd;
        partial_m[warp] = pm;
    }
    __syncthreads();
    if (warp == 0) {
        pd = lane < NWARPS ? partial_d[lane] : 0.0f;
        pm = lane < NWARPS ? partial_m[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            pd += __shfl_xor_sync(0xffffffff, pd, offset);
            pm += __shfl_xor_sync(0xffffffff, pm, offset);
        }
        if (lane == 0) {
            const __half rounded = __float2half(
                projection.neuron_scale[row] * pd -
                projection.neuron_min[row] * pm);
            if constexpr (OUTPUT_BF16) {
                reinterpret_cast<__nv_bfloat16*>(projection.out)[row] =
                    __float2bfloat16(__half2float(rounded));
            } else {
                reinterpret_cast<__half*>(projection.out)[row] = rounded;
            }
        }
    }
}

// NINT6 (6-bit) packed GEMV, structurally identical to gemv_packed_kernel<GS>
// but with 6-bit unpack. Valid only when 4 | GS (so a lane's 4 consecutive
// elements never cross a group boundary -> single sub-scale per lane). The
// recommended NINT6 profile is (6,24,7); 4|24 holds.
template <int GS, bool USE_XSUM=false>
__global__ void __launch_bounds__(256) gemv_packed_int6_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, (GS*6+7)/8]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    const int32_t* __restrict__ xsum,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int WPB = 8;
    constexpr int QBYTES = (GS * 6 + 7) / 8;
    int row = blockIdx.x * WPB + threadIdx.y;
    int m = blockIdx.y;
    int lane = threadIdx.x;
    if (row >= N || m >= M) {
        return;
    }

    const uint8_t* qrow  = q_packed + (size_t)row * ng * QBYTES;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min   + (size_t)row * ng;
    const int8_t*  qxrow = qx + (size_t)m * K_pad;
    const float*   xsrow = xscale + (size_t)m * ng;
    const int32_t* xsumrow = xsum + (size_t)m * ng;

    float pd = 0.0f, pm = 0.0f;
    constexpr int STRIDE = 32 * 4;
    for (int base = lane * 4; base < K_pad; base += STRIDE) {
        int qv = unpack_int6x4(qrow + base * 6 / 8);
        int xv = *reinterpret_cast<const int*>(qxrow + base);
        int di = __dp4a(qv, xv, 0);
        int g = base / GS;
        float xs = xsrow[g];
        pd += xs * (float)ssrow[g] * (float)di;
        if constexpr (USE_XSUM) {
            if ((base % GS) == 0) {
                pm += xs * (float)smrow[g] * (float)xsumrow[g];
            }
        } else {
            int mi = __dp4a(0x01010101, xv, 0);
            pm += xs * (float)smrow[g] * (float)mi;
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        pd += __shfl_xor_sync(0xffffffff, pd, o);
        pm += __shfl_xor_sync(0xffffffff, pm, o);
    }
    if (lane == 0) {
        out[(size_t)m * N + row] = __float2half(neuron_scale[row] * pd - neuron_min[row] * pm);
    }
}

template <int GS, bool USE_XSUM=false, int NWARPS=4>
__global__ void __launch_bounds__(256) gemv_packed_int6_multiwarp_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    const int32_t* __restrict__ xsum,
    __half* __restrict__ out,
    int M, int N, int ng, int K_pad)
{
    constexpr int QBYTES = (GS * 6 + 7) / 8;
    int row = blockIdx.x;
    int m = blockIdx.y;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    if (row >= N || m >= M) {
        return;
    }

    const uint8_t* qrow  = q_packed + (size_t)row * ng * QBYTES;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min   + (size_t)row * ng;
    const int8_t*  qxrow = qx + (size_t)m * K_pad;
    const float*   xsrow = xscale + (size_t)m * ng;
    const int32_t* xsumrow = xsum + (size_t)m * ng;

    float pd = 0.0f;
    float pm = 0.0f;
    constexpr int STRIDE = NWARPS * 32 * 4;
    for (int base = (warp * 32 + lane) * 4; base < K_pad; base += STRIDE) {
        int qv = unpack_int6x4(qrow + base * 6 / 8);
        int xv = *reinterpret_cast<const int*>(qxrow + base);
        int g = base / GS;
        float xs = xsrow[g];
        pd += xs * (float)ssrow[g] * (float)__dp4a(qv, xv, 0);
        if constexpr (USE_XSUM) {
            if ((base % GS) == 0) {
                pm += xs * (float)smrow[g] * (float)xsumrow[g];
            }
        } else {
            pm += xs * (float)smrow[g] * (float)__dp4a(0x01010101, xv, 0);
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        pd += __shfl_xor_sync(0xffffffff, pd, o);
        pm += __shfl_xor_sync(0xffffffff, pm, o);
    }
    __shared__ float partial_d[NWARPS];
    __shared__ float partial_m[NWARPS];
    if (lane == 0) {
        partial_d[warp] = pd;
        partial_m[warp] = pm;
    }
    __syncthreads();

    if (warp == 0) {
        pd = lane < NWARPS ? partial_d[lane] : 0.0f;
        pm = lane < NWARPS ? partial_m[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            pd += __shfl_xor_sync(0xffffffff, pd, o);
            pm += __shfl_xor_sync(0xffffffff, pm, o);
        }
        if (lane == 0) {
            out[(size_t)m * N + row] = __float2half(neuron_scale[row] * pd - neuron_min[row] * pm);
        }
    }
}

template <int NWARPS=4, bool U16_LOAD=false>
__global__ void __launch_bounds__(256) gemv_packed_int6_gs24_group_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    const int32_t*,
    __half* __restrict__ out,
    int M, int N, int ng, int K_pad)
{
    constexpr int GS = 24;
    constexpr int QBYTES = 18;
    int row = blockIdx.x;
    int m = blockIdx.y;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    if (row >= N || m >= M) return;

    const uint8_t* qrow = q_packed + (size_t)row * ng * QBYTES;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min + (size_t)row * ng;
    const int8_t* qxrow = qx + (size_t)m * K_pad;
    const float* xsrow = xscale + (size_t)m * ng;
    float pd = 0.0f;
    float pm = 0.0f;

    for (int g = warp * 32 + lane; g < ng; g += NWARPS * 32) {
        int dsum = 0;
        int msum = 0;
        const uint8_t* qgroup = qrow + g * QBYTES;
        if constexpr (U16_LOAD) {
            #pragma unroll
            for (int chunk = 0; chunk < 3; ++chunk) {
                const uint16_t* qw = reinterpret_cast<const uint16_t*>(qgroup + chunk * 6);
                uint32_t w0 = qw[0];
                uint32_t w1 = qw[1];
                uint32_t w2 = qw[2];
                int qv0 = unpack_int6x4_u24(w0 | ((w1 & 0xff) << 16));
                int qv1 = unpack_int6x4_u24((w1 >> 8) | (w2 << 8));
                int base = g * GS + chunk * 8;
                int xv0 = *reinterpret_cast<const int*>(qxrow + base);
                int xv1 = *reinterpret_cast<const int*>(qxrow + base + 4);
                dsum = __dp4a(qv0, xv0, dsum);
                dsum = __dp4a(qv1, xv1, dsum);
                msum = __dp4a(0x01010101, xv0, msum);
                msum = __dp4a(0x01010101, xv1, msum);
            }
        } else {
            #pragma unroll
            for (int chunk = 0; chunk < 6; ++chunk) {
                int base = g * GS + chunk * 4;
                int xv = *reinterpret_cast<const int*>(qxrow + base);
                dsum = __dp4a(unpack_int6x4(qgroup + chunk * 3), xv, dsum);
                msum = __dp4a(0x01010101, xv, msum);
            }
        }
        float xs = xsrow[g];
        pd += xs * (float)ssrow[g] * (float)dsum;
        pm += xs * (float)smrow[g] * (float)msum;
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        pd += __shfl_xor_sync(0xffffffff, pd, o);
        pm += __shfl_xor_sync(0xffffffff, pm, o);
    }
    __shared__ float partial_d[NWARPS];
    __shared__ float partial_m[NWARPS];
    if (lane == 0) {
        partial_d[warp] = pd;
        partial_m[warp] = pm;
    }
    __syncthreads();
    if (warp == 0) {
        pd = lane < NWARPS ? partial_d[lane] : 0.0f;
        pm = lane < NWARPS ? partial_m[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            pd += __shfl_xor_sync(0xffffffff, pd, o);
            pm += __shfl_xor_sync(0xffffffff, pm, o);
        }
        if (lane == 0) {
            out[(size_t)m * N + row] = __float2half(neuron_scale[row] * pd - neuron_min[row] * pm);
        }
    }
}

template <int GS>
__global__ void __launch_bounds__(256) gemv_packed_swiglu_pair_kernel(
    const uint8_t* __restrict__ q_packed,    // [2*N, ng, GS/2], rows [gate, up]
    const uint8_t* __restrict__ sub_scale,   // [2*N, ng]
    const uint8_t* __restrict__ sub_min,     // [2*N, ng]
    const float* __restrict__ neuron_scale,  // [2*N]
    const float* __restrict__ neuron_min,    // [2*N]
    const int8_t* __restrict__ qx,           // [1, K_pad]
    const float* __restrict__ xscale,        // [1, ng]
    __half* __restrict__ out,                // [1, N]
    int N, int ng, int K_pad, int activation)
{
    constexpr int PPB = 4;
    int pair = blockIdx.x * PPB + (threadIdx.y >> 1);
    int half_id = threadIdx.y & 1;
    int lane = threadIdx.x;
    int row = pair + (half_id ? N : 0);

    __shared__ float vals[PPB * 2];
    float acc = 0.0f;
    if (pair < N) {
        const uint8_t* qrow  = q_packed + (size_t)row * ng * (GS / 2);
        const uint8_t* ssrow = sub_scale + (size_t)row * ng;
        const uint8_t* smrow = sub_min   + (size_t)row * ng;
        float pd = 0.0f, pm = 0.0f;
        constexpr int STRIDE = 32 * 4;
        for (int base = lane * 4; base < K_pad; base += STRIDE) {
            int qv = unpack_int4x4(qrow + base / 2);
            int xv = *reinterpret_cast<const int*>(qx + base);
            int di = __dp4a(qv, xv, 0);
            int mi = __dp4a(0x01010101, xv, 0);
            int g = base / GS;
            float xs = xscale[g];
            pd += xs * (float)ssrow[g] * (float)di;
            pm += xs * (float)smrow[g] * (float)mi;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            pd += __shfl_xor_sync(0xffffffff, pd, o);
            pm += __shfl_xor_sync(0xffffffff, pm, o);
        }
        if (lane == 0) {
            acc = neuron_scale[row] * pd - neuron_min[row] * pm;
        }
    }
    if (lane == 0) {
        vals[threadIdx.y] = acc;
    }
    __syncthreads();
    if (lane == 0 && threadIdx.y < PPB) {
        int p = blockIdx.x * PPB + threadIdx.y;
        if (p < N) {
            float gate = vals[threadIdx.y * 2 + 0];
            float up   = vals[threadIdx.y * 2 + 1];
            gate = __half2float(__float2half(gate));
            up   = __half2float(__float2half(up));
            out[p] = __float2half(mfq_glu_runtime(gate, up, activation));
        }
    }
}

template <int GS, int NWARPS=4>
__global__ void __launch_bounds__(256) gemv_packed_swiglu_multiwarp_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    __half* __restrict__ out,
    int N, int ng, int K_pad, int activation)
{
    int pair = blockIdx.x;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    if (pair >= N) {
        return;
    }

    int up_row = pair + N;
    const uint8_t* qgate = q_packed + (size_t)pair * ng * (GS / 2);
    const uint8_t* qup   = q_packed + (size_t)up_row * ng * (GS / 2);
    const uint8_t* ssg   = sub_scale + (size_t)pair * ng;
    const uint8_t* smg   = sub_min + (size_t)pair * ng;
    const uint8_t* ssu   = sub_scale + (size_t)up_row * ng;
    const uint8_t* smu   = sub_min + (size_t)up_row * ng;

    float pdg = 0.0f;
    float pmg = 0.0f;
    float pdu = 0.0f;
    float pmu = 0.0f;
    constexpr int STRIDE = NWARPS * 32 * 4;
    for (int base = (warp * 32 + lane) * 4; base < K_pad; base += STRIDE) {
        int xv = *reinterpret_cast<const int*>(qx + base);
        int g = base / GS;
        float xs = xscale[g];
        int mi = __dp4a(0x01010101, xv, 0);
        pdg += xs * (float)ssg[g] * (float)__dp4a(unpack_int4x4(qgate + base / 2), xv, 0);
        pmg += xs * (float)smg[g] * (float)mi;
        pdu += xs * (float)ssu[g] * (float)__dp4a(unpack_int4x4(qup + base / 2), xv, 0);
        pmu += xs * (float)smu[g] * (float)mi;
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        pdg += __shfl_xor_sync(0xffffffff, pdg, o);
        pmg += __shfl_xor_sync(0xffffffff, pmg, o);
        pdu += __shfl_xor_sync(0xffffffff, pdu, o);
        pmu += __shfl_xor_sync(0xffffffff, pmu, o);
    }
    __shared__ float partial[4][NWARPS];
    if (lane == 0) {
        partial[0][warp] = pdg;
        partial[1][warp] = pmg;
        partial[2][warp] = pdu;
        partial[3][warp] = pmu;
    }
    __syncthreads();

    if (warp == 0) {
        pdg = lane < NWARPS ? partial[0][lane] : 0.0f;
        pmg = lane < NWARPS ? partial[1][lane] : 0.0f;
        pdu = lane < NWARPS ? partial[2][lane] : 0.0f;
        pmu = lane < NWARPS ? partial[3][lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            pdg += __shfl_xor_sync(0xffffffff, pdg, o);
            pmg += __shfl_xor_sync(0xffffffff, pmg, o);
            pdu += __shfl_xor_sync(0xffffffff, pdu, o);
            pmu += __shfl_xor_sync(0xffffffff, pmu, o);
        }
        if (lane == 0) {
            float gate = neuron_scale[pair] * pdg - neuron_min[pair] * pmg;
            float up = neuron_scale[up_row] * pdu - neuron_min[up_row] * pmu;
            gate = __half2float(__float2half(gate));
            up = __half2float(__float2half(up));
            out[pair] = __float2half(mfq_glu_runtime(gate, up, activation));
        }
    }
}

template <int NWARPS=4, bool VEC_LOAD=false>
__global__ void __launch_bounds__(256) gemv_packed_swiglu_gs24_group_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    __half* __restrict__ out,
    int N, int ng, int K_pad, int activation)
{
    constexpr int GS = 24;
    int pair = blockIdx.x;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    if (pair >= N) return;

    int up_row = pair + N;
    const uint8_t* qgate = q_packed + (size_t)pair * ng * (GS / 2);
    const uint8_t* qup = q_packed + (size_t)up_row * ng * (GS / 2);
    const uint8_t* ssg = sub_scale + (size_t)pair * ng;
    const uint8_t* smg = sub_min + (size_t)pair * ng;
    const uint8_t* ssu = sub_scale + (size_t)up_row * ng;
    const uint8_t* smu = sub_min + (size_t)up_row * ng;
    float pdg = 0.0f;
    float pmg = 0.0f;
    float pdu = 0.0f;
    float pmu = 0.0f;

    for (int g = warp * 32 + lane; g < ng; g += NWARPS * 32) {
        int dgsum = 0;
        int dusum = 0;
        int msum = 0;
        const uint8_t* qggroup = qgate + g * (GS / 2);
        const uint8_t* qugroup = qup + g * (GS / 2);
        uint32_t qgw0 = 0, qgw1 = 0, qgw2 = 0;
        uint32_t quw0 = 0, quw1 = 0, quw2 = 0;
        if constexpr (VEC_LOAD) {
            const uint32_t* qgwords = reinterpret_cast<const uint32_t*>(qggroup);
            const uint32_t* quwords = reinterpret_cast<const uint32_t*>(qugroup);
            qgw0 = qgwords[0]; qgw1 = qgwords[1]; qgw2 = qgwords[2];
            quw0 = quwords[0]; quw1 = quwords[1]; quw2 = quwords[2];
        }
        #pragma unroll
        for (int chunk = 0; chunk < 6; ++chunk) {
            int base = g * GS + chunk * 4;
            int xv = *reinterpret_cast<const int*>(qx + base);
            int qvg;
            int qvu;
            if constexpr (VEC_LOAD) {
                uint32_t qgw = chunk < 2 ? qgw0 : (chunk < 4 ? qgw1 : qgw2);
                uint32_t quw = chunk < 2 ? quw0 : (chunk < 4 ? quw1 : quw2);
                qvg = unpack_int4x4_u16(qgw >> ((chunk & 1) * 16));
                qvu = unpack_int4x4_u16(quw >> ((chunk & 1) * 16));
            } else {
                qvg = unpack_int4x4(qggroup + chunk * 2);
                qvu = unpack_int4x4(qugroup + chunk * 2);
            }
            dgsum = __dp4a(qvg, xv, dgsum);
            dusum = __dp4a(qvu, xv, dusum);
            msum = __dp4a(0x01010101, xv, msum);
        }
        float xs = xscale[g];
        pdg += xs * (float)ssg[g] * (float)dgsum;
        pmg += xs * (float)smg[g] * (float)msum;
        pdu += xs * (float)ssu[g] * (float)dusum;
        pmu += xs * (float)smu[g] * (float)msum;
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        pdg += __shfl_xor_sync(0xffffffff, pdg, o);
        pmg += __shfl_xor_sync(0xffffffff, pmg, o);
        pdu += __shfl_xor_sync(0xffffffff, pdu, o);
        pmu += __shfl_xor_sync(0xffffffff, pmu, o);
    }
    __shared__ float partial[4][NWARPS];
    if (lane == 0) {
        partial[0][warp] = pdg;
        partial[1][warp] = pmg;
        partial[2][warp] = pdu;
        partial[3][warp] = pmu;
    }
    __syncthreads();
    if (warp == 0) {
        pdg = lane < NWARPS ? partial[0][lane] : 0.0f;
        pmg = lane < NWARPS ? partial[1][lane] : 0.0f;
        pdu = lane < NWARPS ? partial[2][lane] : 0.0f;
        pmu = lane < NWARPS ? partial[3][lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            pdg += __shfl_xor_sync(0xffffffff, pdg, o);
            pmg += __shfl_xor_sync(0xffffffff, pmg, o);
            pdu += __shfl_xor_sync(0xffffffff, pdu, o);
            pmu += __shfl_xor_sync(0xffffffff, pmu, o);
        }
        if (lane == 0) {
            float gate = neuron_scale[pair] * pdg - neuron_min[pair] * pmg;
            float up = neuron_scale[up_row] * pdu - neuron_min[up_row] * pmu;
            gate = __half2float(__float2half(gate));
            up = __half2float(__float2half(up));
            out[pair] = __float2half(mfq_glu_runtime(gate, up, activation));
        }
    }
}

template <int GU_GS, int DOWN_GS>
__global__ void __launch_bounds__(1024) ffn_gate_up_swiglu_quant_int4_kernel(
    const uint8_t* __restrict__ q_packed,    // [2*N, gu_ng, GU_GS/2], rows [gate, up]
    const uint8_t* __restrict__ sub_scale,   // [2*N, gu_ng]
    const uint8_t* __restrict__ sub_min,     // [2*N, gu_ng]
    const float* __restrict__ neuron_scale,  // [2*N]
    const float* __restrict__ neuron_min,    // [2*N]
    const int8_t* __restrict__ in_qx,        // [1, gu_K_pad]
    const float* __restrict__ in_xscale,     // [1, gu_ng]
    int8_t* __restrict__ out_qx,             // [1, down_K_pad]
    float* __restrict__ out_xscale,          // [1, down_ng]
    int32_t* __restrict__ out_xsum,          // [1, down_ng]
    int N, int gu_ng, int gu_K_pad, int down_ng, int down_K_pad, int activation)
{
    int gout = blockIdx.x;
    int elem = threadIdx.y;
    int lane = threadIdx.x;
    int row = gout * DOWN_GS + elem;

    __shared__ float vals[DOWN_GS];
    __shared__ int qvals[DOWN_GS];
    __shared__ float scale_sh;

    float v = 0.0f;
    if (row < N) {
        const int up_row = row + N;
        const uint8_t* qgate = q_packed + (size_t)row * gu_ng * (GU_GS / 2);
        const uint8_t* qup   = q_packed + (size_t)up_row * gu_ng * (GU_GS / 2);
        const uint8_t* ssg   = sub_scale + (size_t)row * gu_ng;
        const uint8_t* smg   = sub_min   + (size_t)row * gu_ng;
        const uint8_t* ssu   = sub_scale + (size_t)up_row * gu_ng;
        const uint8_t* smu   = sub_min   + (size_t)up_row * gu_ng;
        float nsg = neuron_scale[row];
        float nmg = neuron_min[row];
        float nsu = neuron_scale[up_row];
        float nmu = neuron_min[up_row];

        float pdg = 0.0f, pmg = 0.0f, pdu = 0.0f, pmu = 0.0f;
        constexpr int STRIDE = 32 * 4;
        for (int base = lane * 4; base < gu_K_pad; base += STRIDE) {
            int qvg = unpack_int4x4(qgate + base / 2);
            int qvu = unpack_int4x4(qup + base / 2);
            int xv = *reinterpret_cast<const int*>(in_qx + base);
            int dig = __dp4a(qvg, xv, 0);
            int diu = __dp4a(qvu, xv, 0);
            int mi = __dp4a(0x01010101, xv, 0);
            int g = base / GU_GS;
            float xs = in_xscale[g];
            pdg += xs * (float)ssg[g] * (float)dig;
            pmg += xs * (float)smg[g] * (float)mi;
            pdu += xs * (float)ssu[g] * (float)diu;
            pmu += xs * (float)smu[g] * (float)mi;
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            pdg += __shfl_xor_sync(0xffffffff, pdg, o);
            pmg += __shfl_xor_sync(0xffffffff, pmg, o);
            pdu += __shfl_xor_sync(0xffffffff, pdu, o);
            pmu += __shfl_xor_sync(0xffffffff, pmu, o);
        }
        if (lane == 0) {
            float gate = nsg * pdg - nmg * pmg;
            float up   = nsu * pdu - nmu * pmu;
            gate = __half2float(__float2half(gate));
            up   = __half2float(__float2half(up));
            v = mfq_glu_runtime(gate, up, activation);
        }
    }

    if (lane == 0) {
        vals[elem] = v;
    }
    __syncthreads();

    if (elem == 0) {
        float amax = (lane < DOWN_GS) ? fabsf(vals[lane]) : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        }
        if (lane == 0) {
            float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
            scale_sh = scale;
            out_xscale[gout] = scale;
        }
    }
    __syncthreads();

    if (lane == 0) {
        int qi = 0;
        if (row < N) {
            float q = roundf(vals[elem] / scale_sh);
            q = fminf(fmaxf(q, -127.0f), 127.0f);
            qi = (int)q;
        }
        qvals[elem] = qi;
        int off = gout * DOWN_GS + elem;
        if (off < down_K_pad) {
            out_qx[off] = (int8_t)qi;
        }
    }
    __syncthreads();

    if (elem == 0) {
        int sum = (lane < DOWN_GS) ? qvals[lane] : 0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, o);
        }
        if (lane == 0) {
            out_xsum[gout] = sum;
        }
    }
}

template <int BITS, int GU_GS, int DOWN_GS>
__global__ void __launch_bounds__(1024) ffn_gate_up_swiglu_quant_bits_kernel(
    const uint8_t* __restrict__ q_packed,    // [2*N, gu_ng, ceil(GU_GS*BITS/8)]
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int8_t* __restrict__ in_qx,
    const float* __restrict__ in_xscale,
    int8_t* __restrict__ out_qx,
    float* __restrict__ out_xscale,
    int32_t* __restrict__ out_xsum,
    int N, int gu_ng, int gu_K_pad, int down_ng, int down_K_pad, int activation)
{
    constexpr int QBYTES = (GU_GS * BITS + 7) / 8;
    constexpr int CHUNKS = (GU_GS + 3) / 4;
    constexpr int GPW = 32 / CHUNKS;
    int gout = blockIdx.x;
    int elem = threadIdx.y;
    int lane = threadIdx.x;
    int row = gout * DOWN_GS + elem;

    __shared__ float vals[DOWN_GS];
    __shared__ int qvals[DOWN_GS];
    __shared__ float scale_sh;

    float v = 0.0f;
    if (row < N) {
        const int up_row = row + N;
        const uint8_t* qgate = q_packed + (size_t)row * gu_ng * QBYTES;
        const uint8_t* qup   = q_packed + (size_t)up_row * gu_ng * QBYTES;
        const uint8_t* ssg   = sub_scale + (size_t)row * gu_ng;
        const uint8_t* smg   = sub_min   + (size_t)row * gu_ng;
        const uint8_t* ssu   = sub_scale + (size_t)up_row * gu_ng;
        const uint8_t* smu   = sub_min   + (size_t)up_row * gu_ng;
        float nsg = neuron_scale[row];
        float nmg = neuron_min[row];
        float nsu = neuron_scale[up_row];
        float nmu = neuron_min[up_row];

        float pdg = 0.0f, pmg = 0.0f, pdu = 0.0f, pmu = 0.0f;
        int relg = lane / CHUNKS;
        int ci = lane - relg * CHUNKS;
        int off = ci * 4;
        bool active_lane = relg < GPW;
        bool full4 = active_lane && ((off + 3) < GU_GS);
        bool tail = active_lane && (off < GU_GS) && !full4;
        for (int gb = 0; gb < gu_ng; gb += GPW) {
            int g = gb + relg;
            if (!active_lane || g >= gu_ng) {
                continue;
            }
            const uint8_t* qg = qgate + (size_t)g * QBYTES;
            const uint8_t* qu = qup + (size_t)g * QBYTES;
            float xs = in_xscale[g];
            if (full4) {
                int qvg = unpack_qbits4<BITS>(qg, off);
                int qvu = unpack_qbits4<BITS>(qu, off);
                int k = g * GU_GS + off;
                int xv = load_i8x4_unaligned(in_qx + k);
                int mi = __dp4a(0x01010101, xv, 0);
                int dig = dp4a_qbits_xs8<BITS>(qvg, xv, mi);
                int diu = dp4a_qbits_xs8<BITS>(qvu, xv, mi);
                pdg += xs * (float)ssg[g] * (float)dig;
                pmg += xs * (float)smg[g] * (float)mi;
                pdu += xs * (float)ssu[g] * (float)diu;
                pmu += xs * (float)smu[g] * (float)mi;
            } else if (tail) {
                int dig = 0, diu = 0, mi = 0;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int gi = off + r;
                    if (gi < GU_GS) {
                        int xq = (int)in_qx[(size_t)g * GU_GS + gi];
                        dig += (int)unpack_qbits_one<BITS>(qg, gi) * xq;
                        diu += (int)unpack_qbits_one<BITS>(qu, gi) * xq;
                        mi += xq;
                    }
                }
                pdg += xs * (float)ssg[g] * (float)dig;
                pmg += xs * (float)smg[g] * (float)mi;
                pdu += xs * (float)ssu[g] * (float)diu;
                pmu += xs * (float)smu[g] * (float)mi;
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            pdg += __shfl_xor_sync(0xffffffff, pdg, o);
            pmg += __shfl_xor_sync(0xffffffff, pmg, o);
            pdu += __shfl_xor_sync(0xffffffff, pdu, o);
            pmu += __shfl_xor_sync(0xffffffff, pmu, o);
        }
        if (lane == 0) {
            float gate = nsg * pdg - nmg * pmg;
            float up   = nsu * pdu - nmu * pmu;
            gate = __half2float(__float2half(gate));
            up   = __half2float(__float2half(up));
            v = mfq_glu_runtime(gate, up, activation);
        }
    }

    if (lane == 0) {
        vals[elem] = v;
    }
    __syncthreads();

    if (elem == 0) {
        float amax = (lane < DOWN_GS) ? fabsf(vals[lane]) : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        }
        if (lane == 0) {
            float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
            scale_sh = scale;
            out_xscale[gout] = scale;
        }
    }
    __syncthreads();

    if (lane == 0) {
        int qi = 0;
        if (row < N) {
            float q = roundf(vals[elem] / scale_sh);
            q = fminf(fmaxf(q, -127.0f), 127.0f);
            qi = (int)q;
        }
        qvals[elem] = qi;
        int off = gout * DOWN_GS + elem;
        if (off < down_K_pad) {
            out_qx[off] = (int8_t)qi;
        }
    }
    __syncthreads();

    if (elem == 0) {
        int sum = (lane < DOWN_GS) ? qvals[lane] : 0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, o);
        }
        if (lane == 0) {
            out_xsum[gout] = sum;
        }
    }
}

template <int GS, int MAX_M>
__global__ void __launch_bounds__(128) gemv_packed_batch_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, GS/2]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    const int32_t* __restrict__ xsum,        // unused, kept for workspace ABI
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int WPB = 4;
    int row = blockIdx.x * WPB + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= N) {
        return;
    }

    const uint8_t* qrow  = q_packed + (size_t)row * ng * (GS / 2);
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min   + (size_t)row * ng;

    float pd[MAX_M], pm[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        pd[m] = 0.0f;
        pm[m] = 0.0f;
    }

    constexpr int STRIDE = 32 * 4;
    for (int base = lane * 4; base < K_pad; base += STRIDE) {
        int qv = unpack_int4x4(qrow + base / 2);
        int g = base / GS;
        uint8_t ss = ssrow[g];
        uint8_t sm = smrow[g];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const int8_t* qxrow = qx + (size_t)m * K_pad;
                int xv = *reinterpret_cast<const int*>(qxrow + base);
                int di = __dp4a(qv, xv, 0);
                int mi = __dp4a(0x01010101, xv, 0);
                float xs = xscale[(size_t)m * ng + g];
                pd[m] += xs * (float)ss * (float)di;
                pm[m] += xs * (float)sm * (float)mi;
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            pd[m] += __shfl_xor_sync(0xffffffff, pd[m], o);
            pm[m] += __shfl_xor_sync(0xffffffff, pm[m], o);
        }
    }
    if (lane == 0) {
        float ns = neuron_scale[row];
        float nm = neuron_min[row];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                out[(size_t)m * N + row] = __float2half(ns * pd[m] - nm * pm[m]);
            }
        }
    }
}

template <int MAX_M, int MSPLIT=1>
__global__ void __launch_bounds__(128 * MSPLIT) gemv_packed_gs24_batch_group_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, 12]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    const int32_t* __restrict__ xsum,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int GS = 24;
    constexpr int QBYTES = 12;
    constexpr int WPB = 4;
    constexpr int MPW = (MAX_M + MSPLIT - 1) / MSPLIT;
    int warp = threadIdx.y;
    int row = blockIdx.x * WPB + warp / MSPLIT;
    int msplit = warp % MSPLIT;
    int lane = threadIdx.x;
    int m0 = blockIdx.y * MAX_M;
    if (row >= N) return;

    const uint8_t* qrow = q_packed + (size_t)row * ng * QBYTES;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min + (size_t)row * ng;
    float acc[MPW];
    float ns = neuron_scale[row];
    float nm = neuron_min[row];
    #pragma unroll
    for (int m = 0; m < MPW; ++m) {
        acc[m] = 0.0f;
    }

    for (int g = lane; g < ng; g += 32) {
        const uint8_t* qgroup = qrow + (size_t)g * QBYTES;
        const uint32_t* qwords = reinterpret_cast<const uint32_t*>(qgroup);
        uint32_t qw0 = qwords[0];
        uint32_t qw1 = qwords[1];
        uint32_t qw2 = qwords[2];
        int dot[MPW];
        #pragma unroll
        for (int m = 0; m < MPW; ++m) dot[m] = 0;

        #pragma unroll
        for (int chunk = 0; chunk < 6; ++chunk) {
            uint32_t qw = chunk < 2 ? qw0 : (chunk < 4 ? qw1 : qw2);
            int qv = unpack_int4x4_u16(qw >> ((chunk & 1) * 16));
            #pragma unroll
            for (int m = 0; m < MPW; ++m) {
                int gm = m0 + msplit * MPW + m;
                if (gm < M) {
                    const int8_t* xgroup = qx + (size_t)gm * K_pad + (size_t)g * GS;
                    int xv = *reinterpret_cast<const int*>(xgroup + chunk * 4);
                    dot[m] = __dp4a(qv, xv, dot[m]);
                }
            }
        }

        float ss = (float)ssrow[g];
        float sm = (float)smrow[g];
        #pragma unroll
        for (int m = 0; m < MPW; ++m) {
            int gm = m0 + msplit * MPW + m;
            if (gm < M) {
                float xs = xscale[(size_t)gm * ng + g];
                acc[m] += xs * (ns * ss * (float)dot[m] -
                                nm * sm * (float)xsum[(size_t)gm * ng + g]);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MPW; ++m) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], o);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < MPW; ++m) {
            int gm = m0 + msplit * MPW + m;
            if (gm < M) out[(size_t)gm * N + row] = __float2half(acc[m]);
        }
    }
}

template <int MAX_M, int MSPLIT=1>
__global__ void __launch_bounds__(128 * MSPLIT) gemv_packed_int6_gs24_batch_group_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, 18]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    const int32_t* __restrict__ xsum,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int GS = 24;
    constexpr int QBYTES = 18;
    constexpr int WPB = 4;
    constexpr int MPW = (MAX_M + MSPLIT - 1) / MSPLIT;
    int warp = threadIdx.y;
    int row = blockIdx.x * WPB + warp / MSPLIT;
    int msplit = warp % MSPLIT;
    int lane = threadIdx.x;
    int m0 = blockIdx.y * MAX_M;
    if (row >= N) return;

    const uint8_t* qrow = q_packed + (size_t)row * ng * QBYTES;
    const uint8_t* ssrow = sub_scale + (size_t)row * ng;
    const uint8_t* smrow = sub_min + (size_t)row * ng;
    float pd[MPW], pm[MPW];
    #pragma unroll
    for (int m = 0; m < MPW; ++m) {
        pd[m] = 0.0f;
        pm[m] = 0.0f;
    }

    for (int g = lane; g < ng; g += 32) {
        const uint8_t* qgroup = qrow + (size_t)g * QBYTES;
        int dot[MPW];
        #pragma unroll
        for (int m = 0; m < MPW; ++m) dot[m] = 0;
        #pragma unroll
        for (int chunk = 0; chunk < 6; ++chunk) {
            int qv = unpack_int6x4(qgroup + chunk * 3);
            #pragma unroll
            for (int m = 0; m < MPW; ++m) {
                int gm = m0 + msplit * MPW + m;
                if (gm < M) {
                    const int8_t* xgroup = qx + (size_t)gm * K_pad + (size_t)g * GS;
                    int xv = *reinterpret_cast<const int*>(xgroup + chunk * 4);
                    dot[m] = __dp4a(qv, xv, dot[m]);
                }
            }
        }

        float ss = (float)ssrow[g];
        float sm = (float)smrow[g];
        #pragma unroll
        for (int m = 0; m < MPW; ++m) {
            int gm = m0 + msplit * MPW + m;
            if (gm < M) {
                float xs = xscale[(size_t)gm * ng + g];
                pd[m] += xs * ss * (float)dot[m];
                pm[m] += xs * sm * (float)xsum[(size_t)gm * ng + g];
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MPW; ++m) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            pd[m] += __shfl_xor_sync(0xffffffff, pd[m], o);
            pm[m] += __shfl_xor_sync(0xffffffff, pm[m], o);
        }
    }
    if (lane == 0) {
        float ns = neuron_scale[row];
        float nm = neuron_min[row];
        #pragma unroll
        for (int m = 0; m < MPW; ++m) {
            int gm = m0 + msplit * MPW + m;
            if (gm < M) out[(size_t)gm * N + row] = __float2half(ns * pd[m] - nm * pm[m]);
        }
    }
}

template <int GS, int MAX_M>
__global__ void __launch_bounds__(128) gemv_packed_batch_eff_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, GS/2]
    const __half* __restrict__ d_eff,        // [N, ng]
    const __half* __restrict__ m_eff,        // [N, ng]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int WPB = 4;
    int row = blockIdx.x * WPB + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= N) {
        return;
    }

    const uint8_t* qrow = q_packed + (size_t)row * ng * (GS / 2);
    const __half* derow = d_eff + (size_t)row * ng;
    const __half* merow = m_eff + (size_t)row * ng;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc[m] = 0.0f;
    }

    constexpr int STRIDE = 32 * 4;
    for (int base = lane * 4; base < K_pad; base += STRIDE) {
        int qv = unpack_int4x4(qrow + base / 2);
        int g = base / GS;
        float de = __half2float(derow[g]);
        float me = __half2float(merow[g]);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const int8_t* qxrow = qx + (size_t)m * K_pad;
                int xv = *reinterpret_cast<const int*>(qxrow + base);
                int di = __dp4a(qv, xv, 0);
                int mi = __dp4a(0x01010101, xv, 0);
                float xs = xscale[(size_t)m * ng + g];
                acc[m] += xs * (de * (float)di - me * (float)mi);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], o);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                out[(size_t)m * N + row] = __float2half(acc[m]);
            }
        }
    }
}

template <int GS, int MAX_M>
__global__ void __launch_bounds__(128) gemv_packed_batch_eff2_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, GS/2]
    const __half2* __restrict__ eff_pair,    // [N, ng], low=d_eff high=m_eff
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int WPB = 4;
    int row = blockIdx.x * WPB + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= N) {
        return;
    }

    const uint8_t* qrow = q_packed + (size_t)row * ng * (GS / 2);
    const __half2* eprow = eff_pair + (size_t)row * ng;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc[m] = 0.0f;
    }

    constexpr int STRIDE = 32 * 4;
    for (int base = lane * 4; base < K_pad; base += STRIDE) {
        int qv = unpack_int4x4(qrow + base / 2);
        int g = base / GS;
        __half2 ep = eprow[g];
        float de = __low2float(ep);
        float me = __high2float(ep);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const int8_t* qxrow = qx + (size_t)m * K_pad;
                int xv = *reinterpret_cast<const int*>(qxrow + base);
                int di = __dp4a(qv, xv, 0);
                int mi = __dp4a(0x01010101, xv, 0);
                float xs = xscale[(size_t)m * ng + g];
                acc[m] += xs * (de * (float)di - me * (float)mi);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], o);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                out[(size_t)m * N + row] = __float2half(acc[m]);
            }
        }
    }
}

mfq_tensor_backend::Tensor nint_gemv_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs)
{
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.scalar_type() == mfq_tensor_backend::kUInt8 && q.is_contiguous(),
                "q must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    int N = (int)q.size(0), ng = (int)q.size(1);
    int M = (int)x.size(0), K_real = (int)x.size(1);
    int K_pad = ng * (int)gs;
    auto qx = mfq_tensor_backend::empty({M, K_pad}, x.options().dtype(mfq_tensor_backend::kInt8));
    auto xscale = mfq_tensor_backend::empty({M, ng}, x.options().dtype(mfq_tensor_backend::kFloat32));
    auto xsum = mfq_tensor_backend::empty({M, ng}, x.options().dtype(mfq_tensor_backend::kInt32));
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

#define QLAUNCH(GSVAL)                                                                  \
    do {                                                                                \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                    \
        quantize_x_kernel<GSVAL, BD><<<dim3(M, ng), BD, 0, stream>>>(                   \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                    \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),  \
            M, K_real, K_pad);                                                          \
        gemv_kernel<GSVAL><<<dim3((N + 3) / 4, M), dim3(32, 4), 0, stream>>>(           \
            q.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),                       \
            sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                \
            neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                        \
            xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                         \
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);      \
    } while (0)

    switch ((int)gs) {
        case 16: QLAUNCH(16); break;
        case 24: QLAUNCH(24); break;
        case 32: QLAUNCH(32); break;
        case 48: QLAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_gemv: gs must be in {16,24,32,48}, got ", gs);
    }
#undef QLAUNCH
    return out;
}

static bool nint_use_multiwarp_gemv()
{
    const char* env = std::getenv("MFQ_NINT_GEMV_MULTI_WARP");
    if (env != nullptr) {
        return env[0] != '0';
    }
    const char* q4_env = std::getenv("MFQ_NINT4_GEMV_MULTI_WARP");
    return q4_env != nullptr && q4_env[0] != '0';
}

static int nint_multiwarp_count(const char* name, int default_warps)
{
    const char* all_env = std::getenv("MFQ_NINT_GEMV_MULTI_WARP");
    if (all_env != nullptr && all_env[0] == '0') {
        return 1;
    }
    const char* env = std::getenv(name);
    if (env == nullptr) {
        return default_warps;
    }
    if (env[0] == '8') return 8;
    if (env[0] == '4') return 4;
    if (env[0] == '2') return 2;
    return 1;
}

static bool nint_gs24_group_enabled(const char* name, bool default_enabled)
{
    const char* env = std::getenv(name);
    return env == nullptr ? default_enabled : env[0] != '0';
}

mfq_tensor_backend::Tensor nint_gemv_packed_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    int M = (int)x.size(0), K_real = (int)x.size(1);
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)xsum.size(0) >= M && (int)xsum.size(1) >= ng, "xsum workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
#define QPWSLAUNCH(GSVAL)                                                               \
    do {                                                                                \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                    \
        quantize_x_kernel<GSVAL, BD><<<dim3(M, ng), BD, 0, stream>>>(                   \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                    \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),  \
            M, K_real, K_pad);                                                          \
        if (GSVAL == 24 && M == 1 && nint_gs24_group_enabled("MFQ_NINT4_GS24_VEC_LOAD", true)) { \
            gemv_packed_gs24_group_kernel<4, true><<<dim3(N, M), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                     \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);  \
        } else if (GSVAL == 24 && M == 1 && nint_gs24_group_enabled("MFQ_NINT4_GS24_GROUP", true)) { \
            gemv_packed_gs24_group_kernel<4><<<dim3(N, M), dim3(32, 4), 0, stream>>>(   \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                     \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);  \
        } else if (M == 1 && nint_use_multiwarp_gemv()) {                               \
            gemv_packed_multiwarp_kernel<GSVAL><<<dim3(N, M), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                     \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);  \
        } else {                                                                        \
            gemv_packed_kernel<GSVAL><<<dim3((N + 7) / 8, M), dim3(32, 8), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                     \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);  \
        }                                                                               \
    } while (0)

    switch ((int)gs) {
        case 16: QPWSLAUNCH(16); break;
        case 24: QPWSLAUNCH(24); break;
        case 32: QPWSLAUNCH(32); break;
        case 48: QPWSLAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_gemv_packed_ws: gs must be in {16,24,32,48}, got ", gs);
    }
#undef QPWSLAUNCH
    return out;
}

mfq_tensor_backend::Tensor nint4_gs24_gemv_bf16_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor sub_min, mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(
        q_packed.is_cuda() && q_packed.is_contiguous() &&
        q_packed.scalar_type() == mfq_tensor_backend::kUInt8 &&
        q_packed.dim() == 3 && q_packed.size(2) == 12,
        "direct BF16 NINT4 output requires CUDA packed gs24 weights");
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() &&
        (x.scalar_type() == mfq_tensor_backend::kFloat16 ||
         x.scalar_type() == mfq_tensor_backend::kBFloat16) &&
        x.dim() == 2 && x.size(0) == 1,
        "direct BF16 NINT4 output requires CUDA contiguous FP16 or BF16 M=1 input");
    MFQ_RUNTIME_CHECK(
        qx.is_cuda() && qx.is_contiguous() &&
        qx.scalar_type() == mfq_tensor_backend::kInt8 &&
        xscale.is_cuda() && xscale.is_contiguous() &&
        xscale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        xsum.is_cuda() && xsum.is_contiguous() &&
        xsum.scalar_type() == mfq_tensor_backend::kInt32,
        "direct BF16 NINT4 output workspace mismatch");
    const int N = (int)q_packed.size(0);
    const int ng = (int)q_packed.size(1);
    const int K_pad = ng * 24;
    MFQ_RUNTIME_CHECK(
        x.size(1) <= K_pad && qx.size(0) >= 1 &&
        qx.size(1) >= K_pad && xscale.size(0) >= 1 &&
        xscale.size(1) >= ng && xsum.size(0) >= 1 &&
        xsum.size(1) >= ng,
        "direct BF16 NINT4 output workspace is too small");
    auto out = mfq_tensor_backend::empty(
        {1, N}, x.options().dtype(mfq_tensor_backend::kBFloat16));
    cudaStream_t stream = mfq_current_cuda_stream();
    if (x.scalar_type() == mfq_tensor_backend::kBFloat16) {
        quantize_x_kernel<24, 32, false, true>
            <<<dim3(1, ng), 32, 0, stream>>>(
                x.data_ptr(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
                1, (int)x.size(1), K_pad);
    } else {
        quantize_x_kernel<24, 32>
            <<<dim3(1, ng), 32, 0, stream>>>(
                x.data_ptr(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
                1, (int)x.size(1), K_pad);
    }
    gemv_packed_gs24_group_kernel<4, true, true>
        <<<dim3(N, 1), dim3(32, 4), 0, stream>>>(
            q_packed.data_ptr<uint8_t>(),
            sub_scale.data_ptr<uint8_t>(),
            sub_min.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(),
            neuron_min.data_ptr<float>(),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            xsum.data_ptr<int32_t>(), out.data_ptr(),
            1, N, ng, K_pad);
    return out;
}

// NINT6 packed GEMV with caller workspace. Only valid for 4|gs (the kernel
// assumes a lane's 4 consecutive elements stay within one group). Covers the
// NINT6 catalog values 20/24/28/32/36/40/48/64.
mfq_tensor_backend::Tensor nint_gemv_packed_int6_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    int qbytes = ((int)gs * 6 + 7) / 8;
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == qbytes, "q_packed last dim must equal (gs*6+7)/8");
    MFQ_RUNTIME_CHECK((int)gs % 4 == 0, "nint_gemv_packed_int6_ws requires 4|gs, got ", gs);
    int M = (int)x.size(0), K_real = (int)x.size(1);
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)xsum.size(0) >= M && (int)xsum.size(1) >= ng, "xsum workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    int nwarps6 = nint_multiwarp_count("MFQ_NINT6_GEMV_WARPS", 4);
    bool group6 = nint_gs24_group_enabled("MFQ_NINT6_GS24_GROUP", false);
    bool u16_group6 = nint_gs24_group_enabled("MFQ_NINT6_GS24_U16_GROUP", true);
    bool batch_group6 = nint_gs24_group_enabled("MFQ_NINT6_GS24_BATCH_GROUP", true);
    const char* split6_env = std::getenv("MFQ_NINT6_GS24_BATCH_SPLIT");
    int batch_group6_split = split6_env != nullptr ? std::atoi(split6_env) : (M >= 4 ? 2 : 1);
#define QPWS6BATCH(MVAL)                                                                \
    do {                                                                                 \
        if (batch_group6_split >= 2) {                                                   \
            gemv_packed_int6_gs24_batch_group_kernel<MVAL, 2><<<dim3((N + 3) / 4, (M + (MVAL) - 1) / (MVAL)), dim3(32, 8), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),             \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),             \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                     \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                      \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);   \
        } else {                                                                         \
            gemv_packed_int6_gs24_batch_group_kernel<MVAL, 1><<<dim3((N + 3) / 4, (M + (MVAL) - 1) / (MVAL)), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),             \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),             \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                     \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                      \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);   \
        }                                                                                \
    } while (0)
#define QPWS6LAUNCH(GSVAL)                                                                \
    do {                                                                                  \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                      \
        if (GSVAL == 24 && M > 1 && batch_group6) {                                      \
            quantize_x_kernel<GSVAL, BD, true><<<dim3(M, ng), BD, 0, stream>>>(           \
                reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                  \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), \
                M, K_real, K_pad);                                                        \
        } else {                                                                          \
            quantize_x_kernel<GSVAL, BD><<<dim3(M, ng), BD, 0, stream>>>(                 \
                reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                  \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), \
                M, K_real, K_pad);                                                        \
        }                                                                                 \
        if (GSVAL == 24 && M > 1 && batch_group6) {                                      \
            if (M == 2) { QPWS6BATCH(2); }                                               \
            else if (M == 3) { QPWS6BATCH(3); }                                          \
            else if (M == 4) { QPWS6BATCH(4); }                                          \
            else if (M == 5) { QPWS6BATCH(5); }                                          \
            else { QPWS6BATCH(8); }                                                      \
        } else if (GSVAL == 24 && M == 1 && u16_group6 && nwarps6 == 8) {                \
            gemv_packed_int6_gs24_group_kernel<8, true><<<dim3(N, M), dim3(32, 8), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),              \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),              \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                      \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                       \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);    \
        } else if (GSVAL == 24 && M == 1 && u16_group6 && nwarps6 == 4) {                 \
            gemv_packed_int6_gs24_group_kernel<4, true><<<dim3(N, M), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),              \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),              \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                      \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                       \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);    \
        } else if (GSVAL == 24 && M == 1 && u16_group6 && nwarps6 == 2) {                 \
            gemv_packed_int6_gs24_group_kernel<2, true><<<dim3(N, M), dim3(32, 2), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),              \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),              \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                      \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                       \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);    \
        } else if (GSVAL == 24 && M == 1 && group6 && nwarps6 == 8) {                     \
            gemv_packed_int6_gs24_group_kernel<8><<<dim3(N, M), dim3(32, 8), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),              \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),              \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                      \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                       \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);    \
        } else if (GSVAL == 24 && M == 1 && group6 && nwarps6 == 4) {                     \
            gemv_packed_int6_gs24_group_kernel<4><<<dim3(N, M), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),              \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),              \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                      \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                       \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);    \
        } else if (GSVAL == 24 && M == 1 && group6 && nwarps6 == 2) {                     \
            gemv_packed_int6_gs24_group_kernel<2><<<dim3(N, M), dim3(32, 2), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),              \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),              \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                      \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                       \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);    \
        } else if (M == 1 && nwarps6 == 8) {                                              \
            gemv_packed_int6_multiwarp_kernel<GSVAL, false, 8><<<dim3(N, M), dim3(32, 8), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),              \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),              \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                      \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                       \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);    \
        } else if (M == 1 && nwarps6 == 4) {                                              \
            gemv_packed_int6_multiwarp_kernel<GSVAL><<<dim3(N, M), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),              \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),              \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                      \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                       \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);    \
        } else if (M == 1 && nwarps6 == 2) {                                              \
            gemv_packed_int6_multiwarp_kernel<GSVAL, false, 2><<<dim3(N, M), dim3(32, 2), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),              \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),              \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                      \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                       \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);    \
        } else {                                                                          \
            gemv_packed_int6_kernel<GSVAL><<<dim3((N + 7) / 8, M), dim3(32, 8), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),              \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),              \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                      \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                       \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);    \
        }                                                                                 \
    } while (0)

    switch ((int)gs) {
        case 20: QPWS6LAUNCH(20); break;
        case 24: QPWS6LAUNCH(24); break;
        case 28: QPWS6LAUNCH(28); break;
        case 32: QPWS6LAUNCH(32); break;
        case 36: QPWS6LAUNCH(36); break;
        case 40: QPWS6LAUNCH(40); break;
        case 48: QPWS6LAUNCH(48); break;
        case 64: QPWS6LAUNCH(64); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_gemv_packed_int6_ws: 4|gs not in catalog, got ", gs);
    }
#undef QPWS6LAUNCH
#undef QPWS6BATCH
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_qx_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, int64_t gs,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    int M = (int)qx.size(0);
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)xsum.size(0) >= M && (int)xsum.size(1) >= ng, "xsum workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, qx.options().dtype(mfq_tensor_backend::kFloat16));
    cudaStream_t stream = mfq_current_cuda_stream();

#define QXLAUNCH(GSVAL)                                                                \
    do {                                                                               \
        if (GSVAL == 24 && M == 1 && nint_gs24_group_enabled("MFQ_NINT4_GS24_VEC_LOAD", true)) { \
            gemv_packed_gs24_group_kernel<4, true><<<dim3(N, M), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),           \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),           \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                   \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                    \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (GSVAL == 24 && M == 1 && nint_gs24_group_enabled("MFQ_NINT4_GS24_GROUP", true)) { \
            gemv_packed_gs24_group_kernel<4><<<dim3(N, M), dim3(32, 4), 0, stream>>>(  \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),           \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),           \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                   \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                    \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 1 && nint_use_multiwarp_gemv()) {                              \
            gemv_packed_multiwarp_kernel<GSVAL><<<dim3(N, M), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),           \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),           \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                   \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                    \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else {                                                                       \
            gemv_packed_kernel<GSVAL><<<dim3((N + 7) / 8, M), dim3(32, 8), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),           \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),           \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                   \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                    \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        }                                                                              \
    } while (0)

    switch ((int)gs) {
        case 16: QXLAUNCH(16); break;
        case 24: QXLAUNCH(24); break;
        case 32: QXLAUNCH(32); break;
        case 48: QXLAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_gemv_packed_qx_ws: gs must be in {16,24,32,48}, got ", gs);
    }
#undef QXLAUNCH
    return out;
}

void nint4_gs24_quantize_input_ws_cuda(
    mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() &&
        (x.scalar_type() == mfq_tensor_backend::kHalf ||
         x.scalar_type() == mfq_tensor_backend::kBFloat16),
        "x must be CUDA contiguous FP16 or BF16");
    MFQ_RUNTIME_CHECK(
        qx.is_cuda() && qx.is_contiguous() &&
        qx.scalar_type() == mfq_tensor_backend::kInt8,
        "qx workspace must be CUDA contiguous int8");
    MFQ_RUNTIME_CHECK(
        xscale.is_cuda() && xscale.is_contiguous() &&
        xscale.scalar_type() == mfq_tensor_backend::kFloat32,
        "xscale workspace must be CUDA contiguous FP32");
    MFQ_RUNTIME_CHECK(
        xsum.is_cuda() && xsum.is_contiguous() &&
        xsum.scalar_type() == mfq_tensor_backend::kInt32,
        "xsum workspace must be CUDA contiguous int32");
    MFQ_RUNTIME_CHECK(
        x.dim() == 2 && x.size(0) == 1,
        "shared NINT4 activation quantization requires M=1");
    const int ng = (int)xscale.size(1);
    const int K_pad = ng * 24;
    MFQ_RUNTIME_CHECK(
        qx.size(0) == 1 && qx.size(1) >= K_pad,
        "qx workspace shape mismatch");
    MFQ_RUNTIME_CHECK(
        xscale.size(0) == 1 && xsum.size(0) == 1 &&
        xsum.size(1) >= ng,
        "activation scale workspace shape mismatch");
    MFQ_RUNTIME_CHECK(
        x.size(1) <= K_pad,
        "activation width exceeds the padded NINT4 width");
    cudaStream_t stream = mfq_current_cuda_stream();
    if (x.scalar_type() == mfq_tensor_backend::kBFloat16) {
        quantize_x_kernel<24, 32, false, true>
            <<<dim3(1, ng), 32, 0, stream>>>(
                x.data_ptr(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
                1, (int)x.size(1), K_pad);
    } else {
        quantize_x_kernel<24, 32>
            <<<dim3(1, ng), 32, 0, stream>>>(
                x.data_ptr(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
                1, (int)x.size(1), K_pad);
    }
}

std::vector<mfq_tensor_backend::Tensor> nint4_gs24_gemv_multi_qx_ws_cuda(
    const std::vector<mfq_tensor_backend::Tensor> & q_packed,
    const std::vector<mfq_tensor_backend::Tensor> & sub_scale,
    const std::vector<mfq_tensor_backend::Tensor> & sub_min,
    const std::vector<mfq_tensor_backend::Tensor> & neuron_scale,
    const std::vector<mfq_tensor_backend::Tensor> & neuron_min,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum,
    bool output_bf16)
{
    const size_t count = q_packed.size();
    MFQ_RUNTIME_CHECK(
        count == 2 || count == 3,
        "multi-projection NINT4 GEMV requires two or three projections");
    MFQ_RUNTIME_CHECK(
        sub_scale.size() == count && sub_min.size() == count &&
        neuron_scale.size() == count && neuron_min.size() == count,
        "multi-projection NINT4 metadata count mismatch");
    MFQ_RUNTIME_CHECK(
        qx.is_cuda() && qx.is_contiguous() &&
        qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.size(0) == 1,
        "shared qx must be CUDA contiguous int8 with M=1");
    MFQ_RUNTIME_CHECK(
        xscale.is_cuda() && xscale.is_contiguous() &&
        xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.size(0) == 1,
        "shared xscale must be CUDA contiguous FP32 with M=1");
    MFQ_RUNTIME_CHECK(
        xsum.is_cuda() && xsum.is_contiguous() &&
        xsum.scalar_type() == mfq_tensor_backend::kInt32 && xsum.size(0) == 1,
        "shared xsum must be CUDA contiguous int32 with M=1");
    const int ng = (int)xscale.size(1);
    MFQ_RUNTIME_CHECK(
        qx.size(1) >= (int64_t)ng * 24 && xsum.size(1) >= ng,
        "shared NINT4 workspace is too small");

    std::vector<mfq_tensor_backend::Tensor> outputs;
    outputs.reserve(count);
    std::vector<Nint4Gs24Projection> projections;
    projections.reserve(count);
    int total_rows = 0;
    for (size_t index = 0; index < count; ++index) {
        MFQ_RUNTIME_CHECK(
            q_packed[index].is_cuda() &&
            q_packed[index].is_contiguous() &&
            q_packed[index].scalar_type() == mfq_tensor_backend::kUInt8 &&
            q_packed[index].dim() == 3 &&
            q_packed[index].size(1) == ng &&
            q_packed[index].size(2) == 12,
            "multi-projection NINT4 packed-weight shape mismatch");
        MFQ_RUNTIME_CHECK(
            sub_scale[index].is_cuda() &&
            sub_scale[index].is_contiguous() &&
            sub_scale[index].scalar_type() == mfq_tensor_backend::kUInt8 &&
            sub_min[index].is_cuda() &&
            sub_min[index].is_contiguous() &&
            sub_min[index].scalar_type() == mfq_tensor_backend::kUInt8,
            "multi-projection NINT4 sub-scale metadata mismatch");
        MFQ_RUNTIME_CHECK(
            neuron_scale[index].is_cuda() &&
            neuron_scale[index].is_contiguous() &&
            neuron_scale[index].scalar_type() == mfq_tensor_backend::kFloat32 &&
            neuron_min[index].is_cuda() &&
            neuron_min[index].is_contiguous() &&
            neuron_min[index].scalar_type() == mfq_tensor_backend::kFloat32,
            "multi-projection NINT4 neuron metadata mismatch");
        const int rows = (int)q_packed[index].size(0);
        outputs.push_back(mfq_tensor_backend::empty(
            {1, rows}, qx.options().dtype(
                output_bf16 ? mfq_tensor_backend::kBFloat16 : mfq_tensor_backend::kFloat16)));
        projections.push_back({
            q_packed[index].data_ptr<uint8_t>(),
            sub_scale[index].data_ptr<uint8_t>(),
            sub_min[index].data_ptr<uint8_t>(),
            neuron_scale[index].data_ptr<float>(),
            neuron_min[index].data_ptr<float>(),
            outputs.back().data_ptr(),
            rows,
        });
        total_rows += rows;
    }
    Nint4Gs24Projection third = projections.back();
    cudaStream_t stream = mfq_current_cuda_stream();
    if (count == 2) {
        if (output_bf16) {
            gemv_packed_gs24_multi_group_kernel<2, true>
                <<<total_rows, dim3(32, 4), 0, stream>>>(
                    projections[0], projections[1], third,
                    qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), ng);
        } else {
            gemv_packed_gs24_multi_group_kernel<2, false>
                <<<total_rows, dim3(32, 4), 0, stream>>>(
                    projections[0], projections[1], third,
                    qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), ng);
        }
    } else {
        if (output_bf16) {
            gemv_packed_gs24_multi_group_kernel<3, true>
                <<<total_rows, dim3(32, 4), 0, stream>>>(
                    projections[0], projections[1], projections[2],
                    qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), ng);
        } else {
            gemv_packed_gs24_multi_group_kernel<3, false>
                <<<total_rows, dim3(32, 4), 0, stream>>>(
                    projections[0], projections[1], projections[2],
                    qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), ng);
        }
    }
    return outputs;
}

mfq_tensor_backend::Tensor nint_gemv_packed_gate_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate,
    int64_t gs, int64_t mode, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(mode == 1 || mode == 2, "gate mode must be 1(sigmoid) or 2(silu)");
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(gate.is_cuda() && gate.is_contiguous() && gate.scalar_type() == mfq_tensor_backend::kHalf,
                "gate must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(x.sizes() == gate.sizes(), "x and gate must have the same shape");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    int M = (int)x.size(0), K_real = (int)x.size(1);
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)xsum.size(0) >= M && (int)xsum.size(1) >= ng, "xsum workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

#define QPGATEWSLAUNCH(GSVAL)                                                          \
    do {                                                                                \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                    \
        if (mode == 1) {                                                                \
            quantize_x_gate_kernel<GSVAL, BD, 1><<<dim3(M, ng), BD, 0, stream>>>(       \
                reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                \
                reinterpret_cast<const __half*>(gate.data_ptr<mfq_half>()),             \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), \
                M, K_real, K_pad);                                                      \
        } else {                                                                        \
            quantize_x_gate_kernel<GSVAL, BD, 2><<<dim3(M, ng), BD, 0, stream>>>(       \
                reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                \
                reinterpret_cast<const __half*>(gate.data_ptr<mfq_half>()),             \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), \
                M, K_real, K_pad);                                                      \
        }                                                                               \
        if (GSVAL == 24 && M == 1 && nint_gs24_group_enabled("MFQ_NINT4_GS24_VEC_LOAD", true)) { \
            gemv_packed_gs24_group_kernel<4, true><<<dim3(N, M), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                     \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);  \
        } else if (GSVAL == 24 && M == 1 && nint_gs24_group_enabled("MFQ_NINT4_GS24_GROUP", true)) { \
            gemv_packed_gs24_group_kernel<4><<<dim3(N, M), dim3(32, 4), 0, stream>>>(   \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                     \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);  \
        } else if (M == 1 && nint_use_multiwarp_gemv()) {                               \
            gemv_packed_multiwarp_kernel<GSVAL><<<dim3(N, M), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                     \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);  \
        } else {                                                                        \
            gemv_packed_kernel<GSVAL><<<dim3((N + 7) / 8, M), dim3(32, 8), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                     \
                reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);  \
        }                                                                               \
    } while (0)

    switch ((int)gs) {
        case 16: QPGATEWSLAUNCH(16); break;
        case 24: QPGATEWSLAUNCH(24); break;
        case 32: QPGATEWSLAUNCH(32); break;
        case 48: QPGATEWSLAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_gemv_packed_gate_ws: gs must be in {16,24,32,48}, got ", gs);
    }
#undef QPGATEWSLAUNCH
    return out;
}

static mfq_tensor_backend::Tensor nint_gemv_packed_glu_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum,
    int activation)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    int N2 = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((N2 & 1) == 0, "swiglu packed GEMV expects concatenated [gate, up] rows");
    int N = N2 / 2;
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    MFQ_RUNTIME_CHECK((int)sub_scale.size(0) == N2 && (int)sub_scale.size(1) == ng, "sub_scale shape mismatch");
    MFQ_RUNTIME_CHECK(sub_min.sizes() == sub_scale.sizes(), "sub_min shape mismatch");
    MFQ_RUNTIME_CHECK((int)neuron_scale.size(0) == N2 && (int)neuron_min.size(0) == N2, "neuron metadata shape mismatch");
    MFQ_RUNTIME_CHECK(x.dim() == 2 && x.size(0) == 1, "swiglu packed GEMV fast path supports only M=1");
    int K_real = (int)x.size(1);
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= 1 && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= 1 && (int)xscale.size(1) >= ng, "xscale workspace too small");
    auto out = mfq_tensor_backend::empty({1, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    int swiglu_nwarps = nint_multiwarp_count("MFQ_NINT_SWIGLU_WARPS", 4);
    bool swiglu_group = nint_gs24_group_enabled("MFQ_NINT_SWIGLU_GS24_GROUP", true);
    bool swiglu_vec_load = nint_gs24_group_enabled("MFQ_NINT_SWIGLU_GS24_VEC_LOAD", true);

#define QPSWIGLULAUNCH(GSVAL)                                                          \
    do {                                                                                \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                    \
        quantize_x_kernel<GSVAL, BD><<<dim3(1, ng), BD, 0, stream>>>(                   \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                    \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),  \
            1, K_real, K_pad);                                                          \
        if (GSVAL == 24 && swiglu_vec_load && swiglu_nwarps == 8) {                    \
            gemv_packed_swiglu_gs24_group_kernel<8, true><<<N, dim3(32, 8), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                N, ng, K_pad, activation);                                              \
        } else if (GSVAL == 24 && swiglu_vec_load && swiglu_nwarps == 4) {             \
            gemv_packed_swiglu_gs24_group_kernel<4, true><<<N, dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                N, ng, K_pad, activation);                                              \
        } else if (GSVAL == 24 && swiglu_vec_load && swiglu_nwarps == 2) {             \
            gemv_packed_swiglu_gs24_group_kernel<2, true><<<N, dim3(32, 2), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                N, ng, K_pad, activation);                                              \
        } else if (GSVAL == 24 && swiglu_group && swiglu_nwarps == 8) {                \
            gemv_packed_swiglu_gs24_group_kernel<8><<<N, dim3(32, 8), 0, stream>>>(    \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                N, ng, K_pad, activation);                                              \
        } else if (GSVAL == 24 && swiglu_group && swiglu_nwarps == 4) {                \
            gemv_packed_swiglu_gs24_group_kernel<4><<<N, dim3(32, 4), 0, stream>>>(    \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                N, ng, K_pad, activation);                                              \
        } else if (GSVAL == 24 && swiglu_group && swiglu_nwarps == 2) {                \
            gemv_packed_swiglu_gs24_group_kernel<2><<<N, dim3(32, 2), 0, stream>>>(    \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                N, ng, K_pad, activation);                                              \
        } else if (swiglu_nwarps == 8) {                                               \
            gemv_packed_swiglu_multiwarp_kernel<GSVAL, 8><<<N, dim3(32, 8), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                N, ng, K_pad, activation);                                              \
        } else if (swiglu_nwarps == 4) {                                               \
            gemv_packed_swiglu_multiwarp_kernel<GSVAL><<<N, dim3(32, 4), 0, stream>>>(  \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                N, ng, K_pad, activation);                                              \
        } else if (swiglu_nwarps == 2) {                                               \
            gemv_packed_swiglu_multiwarp_kernel<GSVAL, 2><<<N, dim3(32, 2), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                N, ng, K_pad, activation);                                              \
        } else {                                                                        \
            gemv_packed_swiglu_pair_kernel<GSVAL><<<dim3((N + 3) / 4), dim3(32, 8), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                N, ng, K_pad, activation);                                              \
        }                                                                               \
    } while (0)

    switch ((int)gs) {
        case 16: QPSWIGLULAUNCH(16); break;
        case 24: QPSWIGLULAUNCH(24); break;
        case 32: QPSWIGLULAUNCH(32); break;
        case 48: QPSWIGLULAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_gemv_packed_swiglu_ws: gs must be in {16,24,32,48}, got ", gs);
    }
#undef QPSWIGLULAUNCH
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_batch_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    int M = (int)x.size(0), K_real = (int)x.size(1);
    bool group24_batch = nint_gs24_group_enabled("MFQ_NINT4_GS24_BATCH_GROUP", true);
    const char* split24_env = std::getenv("MFQ_NINT4_GS24_BATCH_SPLIT");
    int group24_msplit = split24_env != nullptr ? std::atoi(split24_env) : 2;
    MFQ_RUNTIME_CHECK(M >= 1 && (M <= 8 || ((int)gs == 24 && group24_batch && M <= 16)),
                "batched packed GEMV supports M in [1, 8], or M <= 16 for gs24 group mode");
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

#define QPBATCHLAUNCH(GSVAL, MVAL)                                                     \
    do {                                                                                \
        if ((GSVAL) == 24 && group24_batch) {                                           \
            if (group24_msplit >= 4 && (MVAL) >= 4) {                                   \
                gemv_packed_gs24_batch_group_kernel<MVAL, 4><<<dim3((N + 3) / 4, (M + (MVAL) - 1) / (MVAL)), dim3(32, 16), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            } else if (group24_msplit >= 2 && (MVAL) >= 2) {                            \
                gemv_packed_gs24_batch_group_kernel<MVAL, 2><<<dim3((N + 3) / 4, (M + (MVAL) - 1) / (MVAL)), dim3(32, 8), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            } else {                                                                    \
                gemv_packed_gs24_batch_group_kernel<MVAL, 1><<<dim3((N + 3) / 4, (M + (MVAL) - 1) / (MVAL)), dim3(32, 4), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            }                                                                           \
        } else {                                                                        \
            gemv_packed_batch_kernel<GSVAL, MVAL><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        }                                                                               \
    } while (0)

#define QPBWSLAUNCH(GSVAL)                                                              \
    do {                                                                                \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                    \
        if ((GSVAL) == 24 && group24_batch) {                                           \
            quantize_x_kernel<GSVAL, BD, true><<<dim3(M, ng), BD, 0, stream>>>(         \
                reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), \
                M, K_real, K_pad);                                                      \
        } else {                                                                        \
            quantize_x_kernel<GSVAL, BD><<<dim3(M, ng), BD, 0, stream>>>(               \
                reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), \
                M, K_real, K_pad);                                                      \
        }                                                                               \
        if (M == 2) {                                                                   \
            QPBATCHLAUNCH(GSVAL, 2);                                                    \
        } else if (M == 3) {                                                            \
            QPBATCHLAUNCH(GSVAL, 3);                                                    \
        } else if (M == 4) {                                                            \
            QPBATCHLAUNCH(GSVAL, 4);                                                    \
        } else if (M == 5) {                                                            \
            QPBATCHLAUNCH(GSVAL, 5);                                                    \
        } else {                                                                        \
            QPBATCHLAUNCH(GSVAL, 8);                                                    \
        }                                                                               \
    } while (0)

    switch ((int)gs) {
        case 16: QPBWSLAUNCH(16); break;
        case 24: QPBWSLAUNCH(24); break;
        case 32: QPBWSLAUNCH(32); break;
        case 48: QPBWSLAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_gemv_packed_batch_ws: gs must be in {16,24,32,48}, got ", gs);
    }
#undef QPBWSLAUNCH
#undef QPBATCHLAUNCH
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_swiglu_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    return nint_gemv_packed_glu_ws_cuda(
        q_packed, sub_scale, sub_min, neuron_scale, neuron_min, x,
        gs, qx, xscale, xsum, 0);
}

mfq_tensor_backend::Tensor nint_gemv_packed_geglu_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    return nint_gemv_packed_glu_ws_cuda(
        q_packed, sub_scale, sub_min, neuron_scale, neuron_min, x,
        gs, qx, xscale, xsum, 1);
}

mfq_tensor_backend::Tensor nint_gemv_packed_bits_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, int64_t bits, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(bits == 2 || bits == 3 || bits == 5 || bits == 6 || bits == 8,
                "packed-bits GEMV supports bits in {2,3,5,6,8}, got ", bits);
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == (((int)gs * (int)bits + 7) / 8),
                "q_packed last dim must equal ceil(gs*bits/8)");
    MFQ_RUNTIME_CHECK((int)sub_scale.size(0) == N && (int)sub_scale.size(1) == ng, "sub_scale shape mismatch");
    MFQ_RUNTIME_CHECK(sub_min.sizes() == sub_scale.sizes(), "sub_min shape mismatch");
    MFQ_RUNTIME_CHECK((int)neuron_scale.size(0) == N && (int)neuron_min.size(0) == N, "neuron metadata shape mismatch");
    int M = (int)x.size(0), K_real = (int)x.size(1);
    MFQ_RUNTIME_CHECK(M >= 1 && (M <= 8 || (bits == 3 && M <= 64)),
                "packed-bits GEMV supports M in [1,8], or NINT3 M in [1,64]");
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

#define QPBITSLAUNCH(BITSVAL, GSVAL)                                                   \
    do {                                                                                \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                    \
        quantize_x_kernel<GSVAL, BD><<<dim3(M, ng), BD, 0, stream>>>(                   \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                    \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),  \
            M, K_real, K_pad);                                                          \
        if (M == 1) {                                                                   \
            gemv_packed_bits_batch_kernel<BITSVAL, GSVAL, 1><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 2) {                                                            \
            gemv_packed_bits_batch_kernel<BITSVAL, GSVAL, 2><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 3) {                                                            \
            gemv_packed_bits_batch_kernel<BITSVAL, GSVAL, 3><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 4) {                                                            \
            gemv_packed_bits_batch_kernel<BITSVAL, GSVAL, 4><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 5) {                                                            \
            gemv_packed_bits_batch_kernel<BITSVAL, GSVAL, 5><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else {                                                                        \
            gemv_packed_bits_batch_kernel<BITSVAL, GSVAL, 8><<<dim3((N + 3) / 4, (M + 7) / 8), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        }                                                                               \
    } while (0)

#define QPBITSFASTLAUNCH(BITSVAL, GSVAL)                                                \
    do {                                                                                \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                    \
        quantize_x_kernel<GSVAL, BD><<<dim3(M, ng), BD, 0, stream>>>(                   \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                    \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),  \
            M, K_real, K_pad);                                                          \
        if ((BITSVAL) == 5 && (GSVAL) == 28 && M > 1) {                                 \
            const char* mtile_env = std::getenv("MFQ_NINT5_GS28_MTILE");                \
            if (mtile_env != nullptr && mtile_env[0] == '2') {                          \
                gemv_nint5_gs28_batch_kernel<2><<<dim3((N + 3) / 4, (M + 1) / 2), dim3(32, 4), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            } else if (mtile_env != nullptr && mtile_env[0] == '4') {                   \
                gemv_nint5_gs28_batch_kernel<4><<<dim3((N + 3) / 4, (M + 3) / 4), dim3(32, 4), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            } else if (mtile_env != nullptr && mtile_env[0] == '8') {                   \
                gemv_nint5_gs28_batch_kernel<8><<<dim3((N + 3) / 4, (M + 7) / 8), dim3(32, 4), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            } else if (M <= 4 || M == 8) {                                              \
                gemv_nint5_gs28_batch_kernel<4><<<dim3((N + 3) / 4, (M + 3) / 4), dim3(32, 4), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            } else {                                                                    \
                gemv_nint5_gs28_batch_kernel<8><<<dim3((N + 3) / 4, (M + 7) / 8), dim3(32, 4), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            }                                                                           \
        } else if ((BITSVAL) == 6 && (GSVAL) == 26) {                                   \
            const char* mtile_env = std::getenv("MFQ_NINT6_GS26_MTILE");                \
            if (mtile_env != nullptr && mtile_env[0] == '2') {                          \
                gemv_nint6_gs26_batch_kernel<2><<<dim3((N + 3) / 4, (M + 1) / 2), dim3(32, 4), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            } else if (mtile_env != nullptr && mtile_env[0] == '4') {                   \
                gemv_nint6_gs26_batch_kernel<4><<<dim3((N + 3) / 4, (M + 3) / 4), dim3(32, 4), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            } else if (mtile_env != nullptr && mtile_env[0] == '8') {                   \
                gemv_nint6_gs26_batch_kernel<8><<<dim3((N + 3) / 4, (M + 7) / 8), dim3(32, 4), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            } else if (M <= 2) {                                                        \
                gemv_nint6_gs26_batch_kernel<2><<<dim3((N + 3) / 4, (M + 1) / 2), dim3(32, 4), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            } else if (M <= 4) {                                                        \
                gemv_nint6_gs26_batch_kernel<4><<<dim3((N + 3) / 4, (M + 3) / 4), dim3(32, 4), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            } else {                                                                    \
                gemv_nint6_gs26_batch_kernel<8><<<dim3((N + 3) / 4, (M + 7) / 8), dim3(32, 4), 0, stream>>>( \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                    neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
            }                                                                           \
        } else if (M == 1) {                                                            \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 1><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 2) {                                                            \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 2><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 3) {                                                            \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 3><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 4) {                                                            \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 4><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 5) {                                                            \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 5><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else {                                                                        \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 8><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        }                                                                               \
    } while (0)

#define QPBITS_GS_SWITCH(BITSVAL)                                                       \
    switch ((int)gs) {                                                                  \
        case 16: QPBITSLAUNCH(BITSVAL, 16); break;                                      \
        case 20: QPBITSLAUNCH(BITSVAL, 20); break;                                      \
        case 22: QPBITSLAUNCH(BITSVAL, 22); break;                                      \
        case 24: QPBITSLAUNCH(BITSVAL, 24); break;                                      \
        case 26: QPBITSLAUNCH(BITSVAL, 26); break;                                      \
        case 28: QPBITSLAUNCH(BITSVAL, 28); break;                                      \
        case 30: QPBITSLAUNCH(BITSVAL, 30); break;                                      \
        case 32: QPBITSLAUNCH(BITSVAL, 32); break;                                      \
        case 34: QPBITSLAUNCH(BITSVAL, 34); break;                                      \
        case 36: QPBITSLAUNCH(BITSVAL, 36); break;                                      \
        case 40: QPBITSLAUNCH(BITSVAL, 40); break;                                      \
        case 48: QPBITSLAUNCH(BITSVAL, 48); break;                                      \
        case 64: QPBITSLAUNCH(BITSVAL, 64); break;                                      \
        default: MFQ_RUNTIME_CHECK(false, "packed-bits GEMV unsupported gs ", gs);             \
    }

#define QPBITS_FAST_GS_SWITCH(BITSVAL)                                                  \
    switch ((int)gs) {                                                                  \
        case 16: QPBITSFASTLAUNCH(BITSVAL, 16); break;                                  \
        case 20: QPBITSFASTLAUNCH(BITSVAL, 20); break;                                  \
        case 22: QPBITSFASTLAUNCH(BITSVAL, 22); break;                                  \
        case 24: QPBITSFASTLAUNCH(BITSVAL, 24); break;                                  \
        case 26: QPBITSFASTLAUNCH(BITSVAL, 26); break;                                  \
        case 28: QPBITSFASTLAUNCH(BITSVAL, 28); break;                                  \
        case 30: QPBITSFASTLAUNCH(BITSVAL, 30); break;                                  \
        case 32: QPBITSFASTLAUNCH(BITSVAL, 32); break;                                  \
        case 34: QPBITSFASTLAUNCH(BITSVAL, 34); break;                                  \
        case 36: QPBITSFASTLAUNCH(BITSVAL, 36); break;                                  \
        case 40: QPBITSFASTLAUNCH(BITSVAL, 40); break;                                  \
        case 48: QPBITSFASTLAUNCH(BITSVAL, 48); break;                                  \
        case 64: QPBITSFASTLAUNCH(BITSVAL, 64); break;                                  \
        default: MFQ_RUNTIME_CHECK(false, "packed-bits fast GEMV unsupported gs ", gs);        \
    }

    const char* generic_env = std::getenv("MFQ_NINT_BITS_GEMV_GENERIC");
    const char* fast_env = std::getenv("MFQ_NINT_BITS_GEMV_FAST");
    bool force_generic = (generic_env != nullptr && generic_env[0] == '1');
    bool force_fast = (fast_env != nullptr && fast_env[0] == '1');
    if (bits == 2) {
        if (!force_generic) { QPBITS_FAST_GS_SWITCH(2); }
        else { QPBITS_GS_SWITCH(2); }
    } else if (bits == 3) {
        if (M <= 8 && !force_generic) { QPBITS_FAST_GS_SWITCH(3); }
        else { QPBITS_GS_SWITCH(3); }
    } else if (bits == 5) {
        if (!force_generic) { QPBITS_FAST_GS_SWITCH(5); }
        else { QPBITS_GS_SWITCH(5); }
    } else if (bits == 6) {
        if ((force_fast || M == 1 || (int)gs == 26) && !force_generic) { QPBITS_FAST_GS_SWITCH(6); }
        else { QPBITS_GS_SWITCH(6); }
    } else {
        QPBITS_GS_SWITCH(8);
    }
#undef QPBITS_FAST_GS_SWITCH
#undef QPBITS_GS_SWITCH
#undef QPBITSFASTLAUNCH
#undef QPBITSLAUNCH
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_bits_qx_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, int64_t gs, int64_t bits,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    (void)xsum;
    MFQ_RUNTIME_CHECK(bits == 2 || bits == 3 || bits == 5 || bits == 6 || bits == 8,
                "packed-bits qx GEMV supports bits in {2,3,5,6,8}, got ", bits);
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == (((int)gs * (int)bits + 7) / 8),
                "q_packed last dim must equal ceil(gs*bits/8)");
    MFQ_RUNTIME_CHECK((int)sub_scale.size(0) == N && (int)sub_scale.size(1) == ng, "sub_scale shape mismatch");
    MFQ_RUNTIME_CHECK(sub_min.sizes() == sub_scale.sizes(), "sub_min shape mismatch");
    MFQ_RUNTIME_CHECK((int)neuron_scale.size(0) == N && (int)neuron_min.size(0) == N, "neuron metadata shape mismatch");
    int M = (int)qx.size(0);
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 8, "packed-bits qx GEMV supports M in [1, 8]");
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, qx.options().dtype(mfq_tensor_backend::kFloat16));
    cudaStream_t stream = mfq_current_cuda_stream();

#define QPBITSQXLAUNCH(BITSVAL, GSVAL)                                                 \
    do {                                                                                \
        if ((BITSVAL) == 5 && (GSVAL) == 28) {                                          \
            gemv_nint5_gs28_batch_kernel<1><<<dim3((N + 3) / 4, M), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                M, N, ng, K_pad);                                                       \
        } else if ((BITSVAL) == 6 && (GSVAL) == 26) {                                   \
            gemv_nint6_gs26_batch_kernel<1><<<dim3((N + 3) / 4, M), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                M, N, ng, K_pad);                                                       \
        } else {                                                                        \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 1><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                    \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                M, N, ng, K_pad);                                                       \
        }                                                                               \
    } while (0)

#define QPBITSQX_GS_SWITCH(BITSVAL)                                                     \
    switch ((int)gs) {                                                                  \
        case 16: QPBITSQXLAUNCH(BITSVAL, 16); break;                                    \
        case 20: QPBITSQXLAUNCH(BITSVAL, 20); break;                                    \
        case 22: QPBITSQXLAUNCH(BITSVAL, 22); break;                                    \
        case 24: QPBITSQXLAUNCH(BITSVAL, 24); break;                                    \
        case 26: QPBITSQXLAUNCH(BITSVAL, 26); break;                                    \
        case 28: QPBITSQXLAUNCH(BITSVAL, 28); break;                                    \
        case 30: QPBITSQXLAUNCH(BITSVAL, 30); break;                                    \
        case 32: QPBITSQXLAUNCH(BITSVAL, 32); break;                                    \
        case 34: QPBITSQXLAUNCH(BITSVAL, 34); break;                                    \
        case 36: QPBITSQXLAUNCH(BITSVAL, 36); break;                                    \
        case 40: QPBITSQXLAUNCH(BITSVAL, 40); break;                                    \
        case 48: QPBITSQXLAUNCH(BITSVAL, 48); break;                                    \
        case 64: QPBITSQXLAUNCH(BITSVAL, 64); break;                                    \
        default: MFQ_RUNTIME_CHECK(false, "packed-bits qx GEMV unsupported gs ", gs);         \
    }

    if (bits == 2) {
        QPBITSQX_GS_SWITCH(2);
    } else if (bits == 3) {
        QPBITSQX_GS_SWITCH(3);
    } else if (bits == 5) {
        QPBITSQX_GS_SWITCH(5);
    } else if (bits == 6) {
        QPBITSQX_GS_SWITCH(6);
    } else {
        QPBITSQX_GS_SWITCH(8);
    }
#undef QPBITSQX_GS_SWITCH
#undef QPBITSQXLAUNCH
    return out;
}

static void nint_ffn_gate_up_glu_quant_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gu_gs, int64_t gu_bits, int64_t down_gs,
    mfq_tensor_backend::Tensor gu_qx, mfq_tensor_backend::Tensor gu_xscale, mfq_tensor_backend::Tensor gu_xsum,
    mfq_tensor_backend::Tensor down_qx, mfq_tensor_backend::Tensor down_xscale, mfq_tensor_backend::Tensor down_xsum,
    int activation)
{
    MFQ_RUNTIME_CHECK(gu_bits == 2 || gu_bits == 3 || gu_bits == 4 || gu_bits == 5 || gu_bits == 6 || gu_bits == 8,
                "FFN gate/up quant fusion supports bits in {2,3,4,5,6,8}, got ", gu_bits);
    MFQ_RUNTIME_CHECK(down_gs >= 1 && down_gs <= 32,
                "FFN gate/up quant fusion requires down_gs <= 32, got ", down_gs);
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(x.dim() == 2 && x.size(0) == 1, "FFN gate/up quant fusion supports only decode M=1");
    MFQ_RUNTIME_CHECK(gu_qx.is_cuda() && gu_qx.is_contiguous() && gu_qx.scalar_type() == mfq_tensor_backend::kInt8,
                "gu_qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(gu_xscale.is_cuda() && gu_xscale.is_contiguous() && gu_xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "gu_xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(gu_xsum.is_cuda() && gu_xsum.is_contiguous() && gu_xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "gu_xsum workspace must be cuda contiguous int32");
    MFQ_RUNTIME_CHECK(down_qx.is_cuda() && down_qx.is_contiguous() && down_qx.scalar_type() == mfq_tensor_backend::kInt8,
                "down_qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(down_xscale.is_cuda() && down_xscale.is_contiguous() && down_xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "down_xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(down_xsum.is_cuda() && down_xsum.is_contiguous() && down_xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "down_xsum workspace must be cuda contiguous int32");

    int N2 = (int)q_packed.size(0), gu_ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((N2 & 1) == 0, "FFN gate/up quant fusion expects concatenated [gate, up] rows");
    int N = N2 / 2;
    if (gu_bits == 4) {
        MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gu_gs, "q_packed last dim must equal gu_gs/2");
    } else {
        MFQ_RUNTIME_CHECK((int)q_packed.size(2) == (((int)gu_gs * (int)gu_bits + 7) / 8),
                    "q_packed last dim must equal ceil(gu_gs*gu_bits/8)");
    }
    MFQ_RUNTIME_CHECK((int)sub_scale.size(0) == N2 && (int)sub_scale.size(1) == gu_ng, "sub_scale shape mismatch");
    MFQ_RUNTIME_CHECK(sub_min.sizes() == sub_scale.sizes(), "sub_min shape mismatch");
    MFQ_RUNTIME_CHECK((int)neuron_scale.size(0) == N2 && (int)neuron_min.size(0) == N2, "neuron metadata shape mismatch");
    int K_real = (int)x.size(1);
    int gu_K_pad = gu_ng * (int)gu_gs;
    int down_ng = (int)down_xscale.size(1);
    int down_K_pad = down_ng * (int)down_gs;
    MFQ_RUNTIME_CHECK((int)gu_qx.size(0) >= 1 && (int)gu_qx.size(1) >= gu_K_pad, "gu_qx workspace too small");
    MFQ_RUNTIME_CHECK((int)gu_xscale.size(0) >= 1 && (int)gu_xscale.size(1) >= gu_ng, "gu_xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)gu_xsum.size(0) >= 1 && (int)gu_xsum.size(1) >= gu_ng, "gu_xsum workspace too small");
    MFQ_RUNTIME_CHECK((int)down_qx.size(0) >= 1 && (int)down_qx.size(1) >= down_K_pad, "down_qx workspace too small");
    MFQ_RUNTIME_CHECK((int)down_xsum.size(0) >= 1 && (int)down_xsum.size(1) >= down_ng, "down_xsum workspace too small");
    MFQ_RUNTIME_CHECK(N <= down_K_pad, "down workspace cannot hold intermediate activations");
    cudaStream_t stream = mfq_current_cuda_stream();

#define FFN_QUANT_LAUNCH(GU_GSVAL, DOWN_GSVAL)                                         \
    do {                                                                                \
        constexpr int BD = ((GU_GSVAL + 31) / 32) * 32;                                 \
        quantize_x_kernel<GU_GSVAL, BD><<<dim3(1, gu_ng), BD, 0, stream>>>(             \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                    \
            gu_qx.data_ptr<int8_t>(), gu_xscale.data_ptr<float>(), gu_xsum.data_ptr<int32_t>(), \
            1, K_real, gu_K_pad);                                                       \
        if (gu_bits == 2) {                                                             \
            ffn_gate_up_swiglu_quant_bits_kernel<2, GU_GSVAL, DOWN_GSVAL><<<dim3(down_ng), dim3(32, DOWN_GSVAL), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), gu_qx.data_ptr<int8_t>(),                 \
                gu_xscale.data_ptr<float>(), down_qx.data_ptr<int8_t>(),                \
                down_xscale.data_ptr<float>(), down_xsum.data_ptr<int32_t>(),           \
                N, gu_ng, gu_K_pad, down_ng, down_K_pad, activation);                   \
        } else if (gu_bits == 3) {                                                      \
            ffn_gate_up_swiglu_quant_bits_kernel<3, GU_GSVAL, DOWN_GSVAL><<<dim3(down_ng), dim3(32, DOWN_GSVAL), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), gu_qx.data_ptr<int8_t>(),                 \
                gu_xscale.data_ptr<float>(), down_qx.data_ptr<int8_t>(),                \
                down_xscale.data_ptr<float>(), down_xsum.data_ptr<int32_t>(),           \
                N, gu_ng, gu_K_pad, down_ng, down_K_pad, activation);                   \
        } else if (gu_bits == 4) {                                                      \
            ffn_gate_up_swiglu_quant_int4_kernel<GU_GSVAL, DOWN_GSVAL><<<dim3(down_ng), dim3(32, DOWN_GSVAL), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), gu_qx.data_ptr<int8_t>(),                 \
                gu_xscale.data_ptr<float>(), down_qx.data_ptr<int8_t>(),                \
                down_xscale.data_ptr<float>(), down_xsum.data_ptr<int32_t>(),           \
                N, gu_ng, gu_K_pad, down_ng, down_K_pad, activation);                   \
        } else if (gu_bits == 5) {                                                      \
            ffn_gate_up_swiglu_quant_bits_kernel<5, GU_GSVAL, DOWN_GSVAL><<<dim3(down_ng), dim3(32, DOWN_GSVAL), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), gu_qx.data_ptr<int8_t>(),                 \
                gu_xscale.data_ptr<float>(), down_qx.data_ptr<int8_t>(),                \
                down_xscale.data_ptr<float>(), down_xsum.data_ptr<int32_t>(),           \
                N, gu_ng, gu_K_pad, down_ng, down_K_pad, activation);                   \
        } else if (gu_bits == 6) {                                                      \
            ffn_gate_up_swiglu_quant_bits_kernel<6, GU_GSVAL, DOWN_GSVAL><<<dim3(down_ng), dim3(32, DOWN_GSVAL), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), gu_qx.data_ptr<int8_t>(),                 \
                gu_xscale.data_ptr<float>(), down_qx.data_ptr<int8_t>(),                \
                down_xscale.data_ptr<float>(), down_xsum.data_ptr<int32_t>(),           \
                N, gu_ng, gu_K_pad, down_ng, down_K_pad, activation);                   \
        } else {                                                                        \
            ffn_gate_up_swiglu_quant_bits_kernel<8, GU_GSVAL, DOWN_GSVAL><<<dim3(down_ng), dim3(32, DOWN_GSVAL), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),            \
                sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),            \
                neuron_min.data_ptr<float>(), gu_qx.data_ptr<int8_t>(),                 \
                gu_xscale.data_ptr<float>(), down_qx.data_ptr<int8_t>(),                \
                down_xscale.data_ptr<float>(), down_xsum.data_ptr<int32_t>(),           \
                N, gu_ng, gu_K_pad, down_ng, down_K_pad, activation);                   \
        }                                                                               \
    } while (0)

#define FFN_DOWN_GS_SWITCH(GU_GSVAL)                                                    \
    switch ((int)down_gs) {                                                             \
        case 16: FFN_QUANT_LAUNCH(GU_GSVAL, 16); break;                                 \
        case 20: FFN_QUANT_LAUNCH(GU_GSVAL, 20); break;                                 \
        case 22: FFN_QUANT_LAUNCH(GU_GSVAL, 22); break;                                 \
        case 24: FFN_QUANT_LAUNCH(GU_GSVAL, 24); break;                                 \
        case 26: FFN_QUANT_LAUNCH(GU_GSVAL, 26); break;                                 \
        case 28: FFN_QUANT_LAUNCH(GU_GSVAL, 28); break;                                 \
        case 30: FFN_QUANT_LAUNCH(GU_GSVAL, 30); break;                                 \
        case 32: FFN_QUANT_LAUNCH(GU_GSVAL, 32); break;                                 \
        default: MFQ_RUNTIME_CHECK(false, "FFN gate/up quant fusion unsupported down_gs ", down_gs); \
    }

    switch ((int)gu_gs) {
        case 16: FFN_DOWN_GS_SWITCH(16); break;
        case 20: FFN_DOWN_GS_SWITCH(20); break;
        case 22: FFN_DOWN_GS_SWITCH(22); break;
        case 24: FFN_DOWN_GS_SWITCH(24); break;
        case 26: FFN_DOWN_GS_SWITCH(26); break;
        case 28: FFN_DOWN_GS_SWITCH(28); break;
        case 30: FFN_DOWN_GS_SWITCH(30); break;
        case 32: FFN_DOWN_GS_SWITCH(32); break;
        case 34: FFN_DOWN_GS_SWITCH(34); break;
        case 36: FFN_DOWN_GS_SWITCH(36); break;
        case 40: FFN_DOWN_GS_SWITCH(40); break;
        case 48: FFN_DOWN_GS_SWITCH(48); break;
        case 64: FFN_DOWN_GS_SWITCH(64); break;
        default: MFQ_RUNTIME_CHECK(false, "FFN gate/up quant fusion unsupported gu_gs ", gu_gs);
    }
#undef FFN_DOWN_GS_SWITCH
#undef FFN_QUANT_LAUNCH
}

void nint_ffn_gate_up_swiglu_quant_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gu_gs, int64_t gu_bits, int64_t down_gs,
    mfq_tensor_backend::Tensor gu_qx, mfq_tensor_backend::Tensor gu_xscale, mfq_tensor_backend::Tensor gu_xsum,
    mfq_tensor_backend::Tensor down_qx, mfq_tensor_backend::Tensor down_xscale, mfq_tensor_backend::Tensor down_xsum)
{
    nint_ffn_gate_up_glu_quant_ws_cuda(
        q_packed, sub_scale, sub_min, neuron_scale, neuron_min, x,
        gu_gs, gu_bits, down_gs, gu_qx, gu_xscale, gu_xsum,
        down_qx, down_xscale, down_xsum, 0);
}

void nint_ffn_gate_up_geglu_quant_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gu_gs, int64_t gu_bits, int64_t down_gs,
    mfq_tensor_backend::Tensor gu_qx, mfq_tensor_backend::Tensor gu_xscale, mfq_tensor_backend::Tensor gu_xsum,
    mfq_tensor_backend::Tensor down_qx, mfq_tensor_backend::Tensor down_xscale, mfq_tensor_backend::Tensor down_xsum)
{
    nint_ffn_gate_up_glu_quant_ws_cuda(
        q_packed, sub_scale, sub_min, neuron_scale, neuron_min, x,
        gu_gs, gu_bits, down_gs, gu_qx, gu_xscale, gu_xsum,
        down_qx, down_xscale, down_xsum, 1);
}

static mfq_tensor_backend::Tensor nint_gemv_packed_bits_glu_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, int64_t bits, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum,
    int activation)
{
    MFQ_RUNTIME_CHECK(bits == 2 || bits == 3 || bits == 5 || bits == 6 || bits == 8,
                "packed-bits GLU GEMV supports bits in {2,3,5,6,8}, got ", bits);
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    int N2 = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((N2 & 1) == 0, "packed-bits swiglu GEMV expects concatenated [gate, up] rows");
    int N = N2 / 2;
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == (((int)gs * (int)bits + 7) / 8),
                "q_packed last dim must equal ceil(gs*bits/8)");
    MFQ_RUNTIME_CHECK((int)sub_scale.size(0) == N2 && (int)sub_scale.size(1) == ng, "sub_scale shape mismatch");
    MFQ_RUNTIME_CHECK(sub_min.sizes() == sub_scale.sizes(), "sub_min shape mismatch");
    MFQ_RUNTIME_CHECK((int)neuron_scale.size(0) == N2 && (int)neuron_min.size(0) == N2, "neuron metadata shape mismatch");
    MFQ_RUNTIME_CHECK(x.dim() == 2 && x.size(0) == 1, "packed-bits swiglu GEMV fast path supports only M=1");
    int K_real = (int)x.size(1);
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= 1 && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= 1 && (int)xscale.size(1) >= ng, "xscale workspace too small");
    auto out = mfq_tensor_backend::empty({1, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

    const char* combined_env = std::getenv("MFQ_NINT_GLU_COMBINED");
    if (combined_env == nullptr && bits == 8 && gs == 48) {
        combined_env = std::getenv("MFQ_NINT8_GS48_GLU_COMBINED");
    }
    const bool use_combined = combined_env == nullptr || combined_env[0] != '0';

#define QPBITSSWIGLULAUNCH(BITSVAL, GSVAL)                                             \
    do {                                                                                \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                    \
        quantize_x_kernel<GSVAL, BD><<<dim3(1, ng), BD, 0, stream>>>(                   \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                    \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),  \
            1, K_real, K_pad);                                                          \
        if (use_combined) {                                                             \
            gemv_packed_bits_glu_combined_pair_kernel<BITSVAL, GSVAL, 16>               \
                <<<dim3((N + 15) / 16), dim3(32, 16), 0, stream>>>(                    \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),        \
                    sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),        \
                    neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                    N, ng, K_pad, activation);                                          \
        } else {                                                                        \
            gemv_packed_bits_swiglu_pair_kernel<BITSVAL, GSVAL>                         \
                <<<dim3((N + 3) / 4), dim3(32, 8), 0, stream>>>(                        \
                    q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),        \
                    sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),        \
                    neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                \
                    xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
                    N, ng, K_pad, activation);                                          \
        }                                                                               \
    } while (0)

#define QPBITSSWIGLU_GS_SWITCH(BITSVAL)                                                 \
    switch ((int)gs) {                                                                  \
        case 16: QPBITSSWIGLULAUNCH(BITSVAL, 16); break;                                \
        case 20: QPBITSSWIGLULAUNCH(BITSVAL, 20); break;                                \
        case 22: QPBITSSWIGLULAUNCH(BITSVAL, 22); break;                                \
        case 24: QPBITSSWIGLULAUNCH(BITSVAL, 24); break;                                \
        case 26: QPBITSSWIGLULAUNCH(BITSVAL, 26); break;                                \
        case 28: QPBITSSWIGLULAUNCH(BITSVAL, 28); break;                                \
        case 30: QPBITSSWIGLULAUNCH(BITSVAL, 30); break;                                \
        case 32: QPBITSSWIGLULAUNCH(BITSVAL, 32); break;                                \
        case 34: QPBITSSWIGLULAUNCH(BITSVAL, 34); break;                                \
        case 36: QPBITSSWIGLULAUNCH(BITSVAL, 36); break;                                \
        case 40: QPBITSSWIGLULAUNCH(BITSVAL, 40); break;                                \
        case 48: QPBITSSWIGLULAUNCH(BITSVAL, 48); break;                                \
        case 64: QPBITSSWIGLULAUNCH(BITSVAL, 64); break;                                \
        default: MFQ_RUNTIME_CHECK(false, "packed-bits swiglu GEMV unsupported gs ", gs);      \
    }

    if (bits == 2) {
        QPBITSSWIGLU_GS_SWITCH(2);
    } else if (bits == 3) {
        QPBITSSWIGLU_GS_SWITCH(3);
    } else if (bits == 5) {
        QPBITSSWIGLU_GS_SWITCH(5);
    } else if (bits == 6) {
        QPBITSSWIGLU_GS_SWITCH(6);
    } else {
        QPBITSSWIGLU_GS_SWITCH(8);
    }
#undef QPBITSSWIGLU_GS_SWITCH
#undef QPBITSSWIGLULAUNCH
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_bits_swiglu_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, int64_t bits, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    return nint_gemv_packed_bits_glu_ws_cuda(
        q_packed, sub_scale, sub_min, neuron_scale, neuron_min, x,
        gs, bits, qx, xscale, xsum, 0);
}

mfq_tensor_backend::Tensor nint_gemv_packed_bits_geglu_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, int64_t bits, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    return nint_gemv_packed_bits_glu_ws_cuda(
        q_packed, sub_scale, sub_min, neuron_scale, neuron_min, x,
        gs, bits, qx, xscale, xsum, 1);
}

mfq_tensor_backend::Tensor nint5_gs28_q5_gemv_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(q_packed.dim() == 3 && q_packed.size(2) == 20,
                "NINT5 gs28 Q5 execution tensor must have shape [N,ng,20]");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kHalf && x.is_contiguous() && x.dim() == 2,
                "x must be cuda contiguous fp16 [M,K]");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.is_contiguous(),
                "qx must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.is_contiguous(),
                "xscale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.scalar_type() == mfq_tensor_backend::kInt32 && xsum.is_contiguous(),
                "xsum must be cuda contiguous int32");
    const int N = (int)q_packed.size(0);
    const int ng = (int)q_packed.size(1);
    const int M = (int)x.size(0);
    const int K_real = (int)x.size(1);
    const int K_pad = ng * 28;
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 8, "Q5 execution GEMV supports M in [1,8]");
    MFQ_RUNTIME_CHECK(neuron_scale.numel() == N && neuron_min.numel() == N, "neuron metadata shape mismatch");
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    quantize_x_kernel<28, 32><<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
        M, K_real, K_pad);

    const char* wpb_env = std::getenv("MFQ_NINT5_Q5_WPB");
    int wpb = 8;
    if (wpb_env != nullptr) {
        if (wpb_env[0] == '4') wpb = 4;
        else if (wpb_env[0] == '1' && wpb_env[1] == '6') wpb = 16;
    }
    if (wpb == 4) {
        gemv_nint5_gs28_q5_exec_kernel<4><<<dim3((N + 3) / 4, M), dim3(32, 4), 0, stream>>>(
            q_packed.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
            xscale.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);
    } else if (wpb == 16) {
        gemv_nint5_gs28_q5_exec_kernel<16><<<dim3((N + 15) / 16, M), dim3(32, 16), 0, stream>>>(
            q_packed.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
            xscale.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);
    } else {
        gemv_nint5_gs28_q5_exec_kernel<8><<<dim3((N + 7) / 8, M), dim3(32, 8), 0, stream>>>(
            q_packed.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
            xscale.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);
    }
    return out;
}

mfq_tensor_backend::Tensor nint5_gs28_q5_argmax_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum,
    mfq_tensor_backend::Tensor block_vals, mfq_tensor_backend::Tensor block_idxs)
{
    MFQ_RUNTIME_CHECK(x.dim() == 2 && x.size(0) == 1, "Q5 execution argmax expects x [1,K]");
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous() &&
                q_packed.dim() == 3 && q_packed.size(2) == 20,
                "q_packed must be cuda contiguous uint8 [N,ng,20]");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kHalf && x.is_contiguous(),
                "x must be cuda contiguous fp16");
    const int N = (int)q_packed.size(0);
    const int ng = (int)q_packed.size(1);
    const int K_real = (int)x.size(1);
    const int K_pad = ng * 28;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= 1 && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= 1 && (int)xscale.size(1) >= ng, "xscale workspace too small");
    cudaStream_t stream = mfq_current_cuda_stream();
    quantize_x_kernel<28, 32><<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
        1, K_real, K_pad);

    const char* wpb_env = std::getenv("MFQ_NINT5_Q5_WPB");
    int wpb = 8;
    if (wpb_env != nullptr) {
        if (wpb_env[0] == '4') wpb = 4;
        else if (wpb_env[0] == '1' && wpb_env[1] == '6') wpb = 16;
    }
    const int nb = (N + wpb - 1) / wpb;
    MFQ_RUNTIME_CHECK(block_vals.is_cuda() && block_vals.scalar_type() == mfq_tensor_backend::kFloat32 &&
                block_vals.is_contiguous() && block_vals.numel() >= nb, "block_vals workspace too small");
    MFQ_RUNTIME_CHECK(block_idxs.is_cuda() && block_idxs.scalar_type() == mfq_tensor_backend::kInt32 &&
                block_idxs.is_contiguous() && block_idxs.numel() >= nb, "block_idxs workspace too small");
    auto out = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions().device(x.device()).dtype(mfq_tensor_backend::kInt64));
    if (wpb == 4) {
        gemv_nint5_gs28_q5_exec_argmax_stage1_kernel<4><<<nb, dim3(32, 4), 0, stream>>>(
            q_packed.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
            xscale.data_ptr<float>(),
            block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), N, ng, K_pad);
    } else if (wpb == 16) {
        gemv_nint5_gs28_q5_exec_argmax_stage1_kernel<16><<<nb, dim3(32, 16), 0, stream>>>(
            q_packed.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
            xscale.data_ptr<float>(),
            block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), N, ng, K_pad);
    } else {
        gemv_nint5_gs28_q5_exec_argmax_stage1_kernel<8><<<nb, dim3(32, 8), 0, stream>>>(
            q_packed.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
            xscale.data_ptr<float>(),
            block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), N, ng, K_pad);
    }
    nint_argmax_reduce_kernel<<<1, 256, 0, stream>>>(
        block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), out.data_ptr<int64_t>(), nb);
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_bits_argmax_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, int64_t bits, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum,
    mfq_tensor_backend::Tensor block_vals, mfq_tensor_backend::Tensor block_idxs)
{
    const bool nint5_gs28 = bits == 5 && gs == 28;
    const bool nint6 = bits == 6 && (gs == 24 || gs == 26);
    MFQ_RUNTIME_CHECK(nint5_gs28 || nint6,
                "argmax fast path supports NINT5 gs28 and NINT6 gs24/gs26");
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(x.dim() == 2 && x.size(0) == 1, "argmax fast path expects x [1,K]");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    int K_real = (int)x.size(1);
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == ((int)gs * (int)bits + 7) / 8,
                "argmax q_packed last dim mismatch");
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= 1 && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= 1 && (int)xscale.size(1) >= ng, "xscale workspace too small");

    cudaStream_t stream = mfq_current_cuda_stream();
    if (nint5_gs28) {
        quantize_x_kernel<28, 32><<<dim3(1, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
            1, K_real, K_pad);
    } else if (gs == 24) {
        quantize_x_kernel<24, 32><<<dim3(1, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
            1, K_real, K_pad);
    } else {
        quantize_x_kernel<26, 32><<<dim3(1, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
            1, K_real, K_pad);
    }

    const char* wpb_env = std::getenv("MFQ_LM_HEAD_ARGMAX_WPB");
    int wpb = nint5_gs28 || gs == 24 ? 8 : 16;
    if (wpb_env != nullptr) {
        if (wpb_env[0] == '4') {
            wpb = 4;
        } else if (wpb_env[0] == '8') {
            wpb = 8;
        } else if (wpb_env[0] == '1' && wpb_env[1] == '6') {
            wpb = 16;
        }
    }
    int nb = (N + wpb - 1) / wpb;
    MFQ_RUNTIME_CHECK(block_vals.is_cuda() && block_vals.is_contiguous() && block_vals.scalar_type() == mfq_tensor_backend::kFloat32 &&
                block_vals.numel() >= nb, "block_vals workspace too small");
    MFQ_RUNTIME_CHECK(block_idxs.is_cuda() && block_idxs.is_contiguous() && block_idxs.scalar_type() == mfq_tensor_backend::kInt32 &&
                block_idxs.numel() >= nb, "block_idxs workspace too small");
    auto out = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions().device(x.device()).dtype(mfq_tensor_backend::kInt64));
    if (nint5_gs28) {
        if (wpb == 4) {
            gemv_nint5_gs28_argmax_stage1_kernel<4><<<nb, dim3(32, 4), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), N, ng, K_pad);
        } else if (wpb == 16) {
            gemv_nint5_gs28_argmax_stage1_kernel<16><<<nb, dim3(32, 16), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), N, ng, K_pad);
        } else {
            gemv_nint5_gs28_argmax_stage1_kernel<8><<<nb, dim3(32, 8), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), N, ng, K_pad);
        }
    } else if (gs == 24) {
        if (wpb == 4) {
            gemv_nint6_gs24_argmax_stage1_kernel<4><<<nb, dim3(32, 4), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), N, ng, K_pad);
        } else if (wpb == 16) {
            gemv_nint6_gs24_argmax_stage1_kernel<16><<<nb, dim3(32, 16), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), N, ng, K_pad);
        } else {
            gemv_nint6_gs24_argmax_stage1_kernel<8><<<nb, dim3(32, 8), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), N, ng, K_pad);
        }
    } else {
        if (wpb == 4) {
            gemv_nint6_gs26_argmax_stage1_kernel<4><<<nb, dim3(32, 4), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), N, ng, K_pad);
        } else if (wpb == 16) {
            gemv_nint6_gs26_argmax_stage1_kernel<16><<<nb, dim3(32, 16), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), N, ng, K_pad);
        } else {
            gemv_nint6_gs26_argmax_stage1_kernel<8><<<nb, dim3(32, 8), 0, stream>>>(
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(), block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), N, ng, K_pad);
        }
    }
    nint_argmax_reduce_kernel<<<1, 256, 0, stream>>>(
        block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), out.data_ptr<int64_t>(), nb);
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_bits_m1_out_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, int64_t bits, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum,
    mfq_tensor_backend::Tensor out)
{
    MFQ_RUNTIME_CHECK(bits == 6 && gs == 26, "m1 out fast path supports only NINT6 gs26");
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(x.dim() == 2 && x.size(0) == 1, "m1 out fast path expects x [1,K]");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    int K_real = (int)x.size(1);
    int K_pad = ng * 26;
    MFQ_RUNTIME_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == mfq_tensor_backend::kHalf &&
                out.dim() == 2 && out.size(0) == 1 && out.size(1) == N, "out must be [1,N] fp16");
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == 20, "NINT6 gs26 q_packed last dim must be 20");
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= 1 && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= 1 && (int)xscale.size(1) >= ng, "xscale workspace too small");

    cudaStream_t stream = mfq_current_cuda_stream();
    quantize_x_kernel<26, 32><<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
        1, K_real, K_pad);
    gemv_nint6_gs26_batch_kernel<2><<<dim3((N + 3) / 4, 1), dim3(32, 4), 0, stream>>>(
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
        neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), 1, N, ng, K_pad);
    return out;
}

template <int MAX_M>
__global__ void __launch_bounds__(128) nint8_zero_gemv_batch_kernel(
    const int8_t* __restrict__ q,
    const __half* __restrict__ scale,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    __half* __restrict__ out,
    int M,
    int N,
    int ng,
    int K_pad)
{
    constexpr int warps_per_block = 4;
    const int row = blockIdx.x * warps_per_block + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    float acc[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    constexpr int stride = 32 * 4;
    for (int base = lane * 4; base < K_pad; base += stride) {
        const int group = base / 32;
        const int offset = base - group * 32;
        const int packed_weight = *reinterpret_cast<const int*>(
            q + (static_cast<size_t>(row) * ng + group) * 32 + offset);
        const float weight_scale = __half2float(
            scale[static_cast<size_t>(row) * ng + group]);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const int packed_x = *reinterpret_cast<const int*>(
                    qx + static_cast<size_t>(m) * K_pad + base);
                const int dot = __dp4a(packed_weight, packed_x, 0);
                acc[m] += weight_scale *
                    xscale[static_cast<size_t>(m) * ng + group] *
                    static_cast<float>(dot);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                out[static_cast<size_t>(m) * N + row] =
                    __float2half(acc[m]);
            }
        }
    }
}

template <int MMQ_X>
__global__ void __launch_bounds__(256) nint8_zero_mmq_kernel(
    const int8_t* __restrict__ q,
    const __half* __restrict__ scale,
    const int8_t* __restrict__ qx,
    const float* __restrict__ xscale,
    __half* __restrict__ out,
    int M,
    int N,
    int ng,
    int K_pad)
{
    constexpr int warps = 8;
    constexpr int warp_size = 32;
    constexpr int tile_n = 64;
    constexpr int groups_per_chunk = 8;
    constexpr int k_chunk = 32 * groups_per_chunk;
    constexpr int outputs_per_warp = MMQ_X / warps;
    constexpr int rows_per_lane = tile_n / warp_size;

    const int n0 = blockIdx.x * tile_n;
    const int m0 = blockIdx.y * MMQ_X;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * warp_size + lane;

    __shared__ int8_t weight_q[tile_n][k_chunk];
    __shared__ __half weight_scale[tile_n][groups_per_chunk];
    __shared__ int8_t activation_q[MMQ_X][k_chunk];
    __shared__ float activation_scale[MMQ_X][groups_per_chunk];

    float sum[outputs_per_warp][rows_per_lane];
#pragma unroll
    for (int j = 0; j < outputs_per_warp; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_lane; ++i) sum[j][i] = 0.0f;
    }

    const int chunks = (ng + groups_per_chunk - 1) / groups_per_chunk;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int group_base = chunk * groups_per_chunk;
        const int active_groups = min(groups_per_chunk, ng - group_base);

        for (int index = tid; index < tile_n * k_chunk;
             index += warps * warp_size) {
            const int row = index / k_chunk;
            const int column = index % k_chunk;
            const int local_group = column / 32;
            const bool valid = local_group < active_groups && n0 + row < N;
            weight_q[row][column] = valid
                ? q[(static_cast<size_t>(n0 + row) * ng +
                     group_base + local_group) * 32 + column % 32]
                : static_cast<int8_t>(0);
        }
        for (int index = tid;
             index < tile_n * groups_per_chunk;
             index += warps * warp_size) {
            const int row = index / groups_per_chunk;
            const int local_group = index % groups_per_chunk;
            const bool valid =
                local_group < active_groups && n0 + row < N;
            weight_scale[row][local_group] = valid
                ? scale[static_cast<size_t>(n0 + row) * ng +
                        group_base + local_group]
                : __float2half(0.0f);
        }
        for (int index = tid; index < MMQ_X * k_chunk;
             index += warps * warp_size) {
            const int row = index / k_chunk;
            const int column = index % k_chunk;
            const int local_group = column / 32;
            const bool valid =
                local_group < active_groups && m0 + row < M;
            activation_q[row][column] = valid
                ? qx[static_cast<size_t>(m0 + row) * K_pad +
                     (group_base + local_group) * 32 + column % 32]
                : static_cast<int8_t>(0);
        }
        for (int index = tid;
             index < MMQ_X * groups_per_chunk;
             index += warps * warp_size) {
            const int row = index / groups_per_chunk;
            const int local_group = index % groups_per_chunk;
            const bool valid =
                local_group < active_groups && m0 + row < M;
            activation_scale[row][local_group] = valid
                ? xscale[static_cast<size_t>(m0 + row) * ng +
                         group_base + local_group]
                : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < outputs_per_warp; ++j) {
            const int local_m = warp + j * warps;
            const int m = m0 + local_m;
#pragma unroll
            for (int i = 0; i < rows_per_lane; ++i) {
                const int local_n = lane + i * warp_size;
                const int n = n0 + local_n;
                if (m < M && n < N) {
                    float partial = 0.0f;
#pragma unroll
                    for (int local_group = 0;
                         local_group < groups_per_chunk;
                         ++local_group) {
                        if (group_base + local_group >= ng) break;
                        int dot = 0;
#pragma unroll
                        for (int part = 0; part < 8; ++part) {
                            const int packed_weight =
                                *reinterpret_cast<const int*>(
                                    weight_q[local_n] +
                                    local_group * 32 + part * 4);
                            const int packed_x =
                                *reinterpret_cast<const int*>(
                                    activation_q[local_m] +
                                    local_group * 32 + part * 4);
                            dot = __dp4a(packed_weight, packed_x, dot);
                        }
                        partial +=
                            __half2float(
                                weight_scale[local_n][local_group]) *
                            activation_scale[local_m][local_group] *
                            static_cast<float>(dot);
                    }
                    sum[j][i] += partial;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < outputs_per_warp; ++j) {
        const int m = m0 + warp + j * warps;
#pragma unroll
        for (int i = 0; i < rows_per_lane; ++i) {
            const int n = n0 + lane + i * warp_size;
            if (m < M && n < N) {
                out[static_cast<size_t>(m) * N + n] =
                    __float2half(sum[j][i]);
            }
        }
    }
}

__global__ void nint8_zero_dequant_kernel(
    const int8_t* __restrict__ q,
    const __half* __restrict__ scale,
    __half* __restrict__ out,
    int N,
    int ng,
    int neuron_len)
{
    const size_t total = static_cast<size_t>(N) * neuron_len;
    for (size_t index =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / neuron_len);
        const int column = static_cast<int>(index % neuron_len);
        const int group = column / 32;
        const int lane = column % 32;
        const size_t block = static_cast<size_t>(row) * ng + group;
        out[index] = __float2half(
            __half2float(scale[block]) *
            static_cast<float>(q[block * 32 + lane]));
    }
}

mfq_tensor_backend::Tensor nint8_zero_gemv_ws_cuda(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor scale,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale)
{
    MFQ_RUNTIME_CHECK(
        q.is_cuda() && q.is_contiguous() &&
            q.scalar_type() == mfq_tensor_backend::kUInt8 && q.dim() == 3 &&
            q.size(2) == 32,
        "NINT8-0 q must be contiguous CUDA uint8 [N,ng,32]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
            scale.scalar_type() == mfq_tensor_backend::kFloat16 && scale.dim() == 2 &&
            scale.size(0) == q.size(0) && scale.size(1) == q.size(1),
        "NINT8-0 scale must be contiguous CUDA f16 [N,ng]");
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == mfq_tensor_backend::kFloat16 && x.dim() == 2,
        "NINT8-0 x must be contiguous CUDA f16 [M,K]");
    const int M = static_cast<int>(x.size(0));
    const int N = static_cast<int>(q.size(0));
    const int ng = static_cast<int>(q.size(1));
    const int K_real = static_cast<int>(x.size(1));
    const int K_pad = ng * 32;
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 8 && K_real <= K_pad,
        "NINT8-0 GEMV expects M in [1,8] and K <= packed K");
    MFQ_RUNTIME_CHECK(
        qx.is_cuda() && qx.is_contiguous() &&
            qx.scalar_type() == mfq_tensor_backend::kInt8 &&
            qx.size(0) >= M && qx.size(1) >= K_pad,
        "NINT8-0 qx workspace is too small");
    MFQ_RUNTIME_CHECK(
        xscale.is_cuda() && xscale.is_contiguous() &&
            xscale.scalar_type() == mfq_tensor_backend::kFloat32 &&
            xscale.size(0) >= M && xscale.size(1) >= ng,
        "NINT8-0 xscale workspace is too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    quantize_x_kernel<32, 32><<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), nullptr,
        M, K_real, K_pad);
#define NINT80_GEMV_CASE(M_VALUE) \
    case M_VALUE: \
        nint8_zero_gemv_batch_kernel<M_VALUE><<< \
            dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
            reinterpret_cast<const int8_t*>(q.data_ptr<uint8_t>()), \
            reinterpret_cast<const __half*>(scale.data_ptr<mfq_half>()), \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), \
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
            M, N, ng, K_pad); \
        break
    switch (M) {
        NINT80_GEMV_CASE(1);
        NINT80_GEMV_CASE(2);
        NINT80_GEMV_CASE(3);
        NINT80_GEMV_CASE(4);
        NINT80_GEMV_CASE(5);
        NINT80_GEMV_CASE(6);
        NINT80_GEMV_CASE(7);
        NINT80_GEMV_CASE(8);
    }
#undef NINT80_GEMV_CASE
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

mfq_tensor_backend::Tensor nint8_zero_mmq_ws_cuda(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor scale,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale)
{
    MFQ_RUNTIME_CHECK(
        q.is_cuda() && q.is_contiguous() &&
            q.scalar_type() == mfq_tensor_backend::kUInt8 && q.dim() == 3 &&
            q.size(2) == 32,
        "NINT8-0 q must be contiguous CUDA uint8 [N,ng,32]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
            scale.scalar_type() == mfq_tensor_backend::kFloat16 && scale.dim() == 2 &&
            scale.size(0) == q.size(0) && scale.size(1) == q.size(1),
        "NINT8-0 scale must be contiguous CUDA f16 [N,ng]");
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == mfq_tensor_backend::kFloat16 && x.dim() == 2,
        "NINT8-0 x must be contiguous CUDA f16 [M,K]");
    const int M = static_cast<int>(x.size(0));
    const int N = static_cast<int>(q.size(0));
    const int ng = static_cast<int>(q.size(1));
    const int K_real = static_cast<int>(x.size(1));
    const int K_pad = ng * 32;
    MFQ_RUNTIME_CHECK(M >= 9 && M <= 64 && K_real <= K_pad,
        "NINT8-0 MMQ expects M in [9,64] and K <= packed K");
    MFQ_RUNTIME_CHECK(
        qx.is_cuda() && qx.is_contiguous() &&
            qx.scalar_type() == mfq_tensor_backend::kInt8 &&
            qx.size(0) >= M && qx.size(1) >= K_pad,
        "NINT8-0 qx workspace is too small");
    MFQ_RUNTIME_CHECK(
        xscale.is_cuda() && xscale.is_contiguous() &&
            xscale.scalar_type() == mfq_tensor_backend::kFloat32 &&
            xscale.size(0) >= M && xscale.size(1) >= ng,
        "NINT8-0 xscale workspace is too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    quantize_x_kernel<32, 32><<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), nullptr,
        M, K_real, K_pad);
    constexpr int tile_n = 64;
    if (M <= 16) {
        nint8_zero_mmq_kernel<16><<<
            dim3((N + tile_n - 1) / tile_n, 1),
            dim3(32, 8), 0, stream>>>(
            reinterpret_cast<const int8_t*>(q.data_ptr<uint8_t>()),
            reinterpret_cast<const __half*>(scale.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()),
            M, N, ng, K_pad);
    } else {
        nint8_zero_mmq_kernel<32><<<
            dim3((N + tile_n - 1) / tile_n, (M + 31) / 32),
            dim3(32, 8), 0, stream>>>(
            reinterpret_cast<const int8_t*>(q.data_ptr<uint8_t>()),
            reinterpret_cast<const __half*>(scale.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()),
            M, N, ng, K_pad);
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

mfq_tensor_backend::Tensor nint8_zero_dequant_cuda(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor scale,
    int64_t neuron_len)
{
    MFQ_RUNTIME_CHECK(
        q.is_cuda() && q.is_contiguous() &&
            q.scalar_type() == mfq_tensor_backend::kUInt8 && q.dim() == 3 &&
            q.size(2) == 32,
        "NINT8-0 q must be contiguous CUDA uint8 [N,ng,32]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
            scale.scalar_type() == mfq_tensor_backend::kFloat16 && scale.dim() == 2 &&
            scale.size(0) == q.size(0) && scale.size(1) == q.size(1),
        "NINT8-0 scale must be contiguous CUDA f16 [N,ng]");
    MFQ_RUNTIME_CHECK(
        neuron_len > 0 && neuron_len <= q.size(1) * 32,
        "NINT8-0 neuron_len is invalid");
    const int N = static_cast<int>(q.size(0));
    const int ng = static_cast<int>(q.size(1));
    auto out = mfq_tensor_backend::empty(
        {N, neuron_len}, q.options().dtype(mfq_tensor_backend::kFloat16));
    constexpr int threads = 256;
    const size_t total = static_cast<size_t>(N) * neuron_len;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    blocks = std::min(blocks, 4096);
    nint8_zero_dequant_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const int8_t*>(q.data_ptr<uint8_t>()),
        reinterpret_cast<const __half*>(scale.data_ptr<mfq_half>()),
        reinterpret_cast<__half*>(out.data_ptr<mfq_half>()),
        N, ng, static_cast<int>(neuron_len));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_u8_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == (int)gs, "NINT8 q_packed last dim must equal gs");
    int M = (int)x.size(0), K_real = (int)x.size(1);
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 8, "NINT8 GEMV supports M in [1, 8]");
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)xsum.size(0) >= M && (int)xsum.size(1) >= ng, "xsum workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

    if (M == 1 && gs == 48 && N <= 64) {
        quantize_x_kernel<48, 64, true><<<dim3(1, ng), 64, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
            1, K_real, K_pad);
        gemv_packed_u8_m1_row_kernel<48, 1><<<N, dim3(32, 1), 0, stream>>>(           \
            q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),              \
            sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),              \
            neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                      \
            xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
            N, ng, K_pad);
        return out;
    }

#define QPU8LAUNCH(GSVAL)                                                               \
    do {                                                                                \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                    \
        quantize_x_kernel<GSVAL, BD, true><<<dim3(M, ng), BD, 0, stream>>>(             \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                    \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),  \
            M, K_real, K_pad);                                                          \
        if (M == 1) {                                                                   \
            gemv_packed_u8_batch_kernel<GSVAL, 1><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 2) {                                                            \
            gemv_packed_u8_batch_kernel<GSVAL, 2><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 3) {                                                            \
            gemv_packed_u8_batch_kernel<GSVAL, 3><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 4) {                                                            \
            gemv_packed_u8_batch_kernel<GSVAL, 4><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 5) {                                                            \
            gemv_packed_u8_batch_kernel<GSVAL, 5><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else {                                                                        \
            gemv_packed_u8_batch_kernel<GSVAL, 8><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        }                                                                               \
    } while (0)

    switch ((int)gs) {
        case 16: QPU8LAUNCH(16); break;
        case 24: QPU8LAUNCH(24); break;
        case 32: QPU8LAUNCH(32); break;
        case 48: QPU8LAUNCH(48); break;
        case 64: QPU8LAUNCH(64); break;
        default: MFQ_RUNTIME_CHECK(false, "NINT8 GEMV unsupported gs ", gs);
    }
#undef QPU8LAUNCH
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_u8_groupwise_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, int64_t groups, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale,
    mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(
        q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 &&
            q_packed.is_contiguous(),
        "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(
        sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
            sub_scale.is_contiguous(),
        "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(
        sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 &&
            sub_min.is_contiguous(),
        "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
            neuron_scale.is_contiguous(),
        "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(
        neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 &&
            neuron_min.is_contiguous(),
        "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf &&
            x.dim() == 3,
        "groupwise input must be contiguous cuda fp16 [B, groups, K]");
    MFQ_RUNTIME_CHECK(gs == 48, "groupwise NINT8 GEMV currently requires gs48");
    MFQ_RUNTIME_CHECK(groups > 0 && x.size(1) == groups, "groupwise input group mismatch");

    const int B = (int)x.size(0);
    const int input_groups = (int)x.size(1);
    const int K_real = (int)x.size(2);
    const int N = (int)q_packed.size(0);
    const int ng = (int)q_packed.size(1);
    const int K_pad = ng * (int)gs;
    const int input_rows = B * input_groups;
    MFQ_RUNTIME_CHECK(B > 0 && N % input_groups == 0, "groupwise output rows must divide groups");
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == 48, "NINT8 gs48 q_packed layout mismatch");
    MFQ_RUNTIME_CHECK(
        (int)qx.size(0) >= input_rows && (int)qx.size(1) >= K_pad,
        "groupwise qx workspace too small");
    MFQ_RUNTIME_CHECK(
        (int)xscale.size(0) >= input_rows && (int)xscale.size(1) >= ng,
        "groupwise xscale workspace too small");
    MFQ_RUNTIME_CHECK(
        (int)xsum.size(0) >= input_rows && (int)xsum.size(1) >= ng,
        "groupwise xsum workspace too small");

    auto out = mfq_tensor_backend::empty({B, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    quantize_x_kernel<48, 64, true><<<dim3(input_rows, ng), 64, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
        input_rows, K_real, K_pad);
    constexpr int kWarpsPerBlock = 4;
    const int total_outputs = B * N;
    gemv_packed_u8_groupwise_kernel<48><<<
        (total_outputs + kWarpsPerBlock - 1) / kWarpsPerBlock,
        dim3(32, kWarpsPerBlock), 0, stream>>>(
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),
        sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),
        neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()),
        B, input_groups, N / input_groups, N, ng, K_pad);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_bits_linear_out_norm_gate_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor y, mfq_tensor_backend::Tensor z, mfq_tensor_backend::Tensor norm_weight,
    int64_t gs, int64_t bits, int64_t dv, double eps,
    mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum, mfq_tensor_backend::Tensor rinv)
{
    MFQ_RUNTIME_CHECK(bits == 5 && gs == 28, "linear_out_norm_gate fast path supports only NINT5 gs28");
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(y.is_cuda() && y.scalar_type() == mfq_tensor_backend::kFloat32 && y.is_contiguous(),
                "y must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(z.is_cuda() && z.scalar_type() == mfq_tensor_backend::kHalf && z.is_contiguous(),
                "z must be cuda contiguous f16");
    MFQ_RUNTIME_CHECK(norm_weight.is_cuda() && norm_weight.scalar_type() == mfq_tensor_backend::kFloat32 && norm_weight.is_contiguous(),
                "norm_weight must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == mfq_tensor_backend::kInt8 && qx.is_contiguous(),
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == mfq_tensor_backend::kFloat32 && xscale.is_contiguous(),
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.scalar_type() == mfq_tensor_backend::kInt32 && xsum.is_contiguous(),
                "xsum workspace must be cuda contiguous int32");
    MFQ_RUNTIME_CHECK(rinv.is_cuda() && rinv.scalar_type() == mfq_tensor_backend::kFloat32 && rinv.is_contiguous(),
                "rinv workspace must be cuda contiguous f32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    int K_real = (int)z.numel();
    int D = (int)dv;
    MFQ_RUNTIME_CHECK(D > 0 && K_real % D == 0, "linear_out_norm_gate: invalid dv");
    int H = K_real / D;
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)y.numel() == K_real, "linear_out_norm_gate: y/z size mismatch");
    MFQ_RUNTIME_CHECK((int)norm_weight.numel() == D, "linear_out_norm_gate: norm weight size mismatch");
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == 18, "linear_out_norm_gate: q_packed last dim must be 18");
    MFQ_RUNTIME_CHECK((int)qx.numel() >= K_pad, "linear_out_norm_gate: qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.numel() >= ng, "linear_out_norm_gate: xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)rinv.numel() >= H, "linear_out_norm_gate: rinv workspace too small");

    auto out = mfq_tensor_backend::empty({1, N}, z.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    linear_out_head_rinv_kernel<<<H, 128, 0, stream>>>(
        y.data_ptr<float>(), rinv.data_ptr<float>(), H, D, (float)eps);
    linear_out_norm_gate_quant_kernel<28, 32><<<ng, 32, 0, stream>>>(
        y.data_ptr<float>(), reinterpret_cast<const __half*>(z.data_ptr<mfq_half>()),
        norm_weight.data_ptr<float>(), rinv.data_ptr<float>(), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), H, D, K_real, K_pad);
    gemv_packed_bits_group4_batch_kernel<5, 28, 1><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>(
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(),
        neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), 1, N, ng, K_pad);
    (void)xsum;
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_bits_gate_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate,
    int64_t gs, int64_t bits, int64_t mode, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(mode == 1 || mode == 2, "gate mode must be 1(sigmoid) or 2(silu)");
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(x.sizes() == gate.sizes(), "x and gate must have the same shape");
    MFQ_RUNTIME_CHECK(gate.is_cuda() && gate.is_contiguous() && gate.scalar_type() == mfq_tensor_backend::kHalf,
                "gate must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");

    int N = (int)q_packed.size(0);
    int M = (int)x.size(0), K_real = (int)x.size(1);
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 8, "packed-bits gated GEMV supports M in [1, 8]");
    int ng = (int)q_packed.size(1);
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == (((int)gs * (int)bits + 7) / 8),
                "q_packed last dim must equal ceil(gs*bits/8)");
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)xsum.size(0) >= M && (int)xsum.size(1) >= ng, "xsum workspace too small");
    cudaStream_t stream = mfq_current_cuda_stream();
    auto out = mfq_tensor_backend::empty({M, N}, x.options());

    // NINT8 uses a byte-specialized GEMV with an affine correction from xsum.
    // The generic packed-bits kernel does not implement that execution layout.
    if (bits == 8) {
#define QPU8_GATE_GEMV(GSVAL, MVAL)                                               \
        gemv_packed_u8_batch_kernel<GSVAL, MVAL><<<                               \
            dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>(                         \
            q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),          \
            sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),          \
            neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                  \
            xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                   \
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad)
#define QPU8_GATE_LAUNCH(GSVAL)                                                   \
        do {                                                                      \
            constexpr int BD = ((GSVAL + 31) / 32) * 32;                         \
            if (mode == 1) {                                                      \
                quantize_x_gate_kernel<GSVAL, BD, 1><<<dim3(M, ng), BD, 0, stream>>>( \
                    reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),      \
                    reinterpret_cast<const __half*>(gate.data_ptr<mfq_half>()),   \
                    qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),              \
                    xsum.data_ptr<int32_t>(), M, K_real, K_pad);                  \
            } else {                                                              \
                quantize_x_gate_kernel<GSVAL, BD, 2><<<dim3(M, ng), BD, 0, stream>>>( \
                    reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),      \
                    reinterpret_cast<const __half*>(gate.data_ptr<mfq_half>()),   \
                    qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),              \
                    xsum.data_ptr<int32_t>(), M, K_real, K_pad);                  \
            }                                                                     \
            if (M == 1)      { QPU8_GATE_GEMV(GSVAL, 1); }                       \
            else if (M == 2) { QPU8_GATE_GEMV(GSVAL, 2); }                       \
            else if (M == 3) { QPU8_GATE_GEMV(GSVAL, 3); }                       \
            else if (M == 4) { QPU8_GATE_GEMV(GSVAL, 4); }                       \
            else if (M == 5) { QPU8_GATE_GEMV(GSVAL, 5); }                       \
            else             { QPU8_GATE_GEMV(GSVAL, 8); }                       \
        } while (0)
        switch ((int)gs) {
            case 16: QPU8_GATE_LAUNCH(16); break;
            case 24: QPU8_GATE_LAUNCH(24); break;
            case 32: QPU8_GATE_LAUNCH(32); break;
            case 48: QPU8_GATE_LAUNCH(48); break;
            case 64: QPU8_GATE_LAUNCH(64); break;
            default: MFQ_RUNTIME_CHECK(false, "NINT8 gated GEMV unsupported gs ", gs);
        }
#undef QPU8_GATE_LAUNCH
#undef QPU8_GATE_GEMV
        return out;
    }

    const char* gate6_env = std::getenv("MFQ_NINT6_GATE_GS24_BATCH");
    bool fast_gate6 = bits == 6 && gs == 24 &&
        (gate6_env == nullptr || gate6_env[0] != '0');
#define QPBITS_GATE_QUANT(GSVAL)                                                        \
    do {                                                                                \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                    \
        if (mode == 1) {                                                                \
            quantize_x_gate_nosum_kernel<GSVAL, BD, 1><<<dim3(M, ng), BD, 0, stream>>>( \
                reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                \
                reinterpret_cast<const __half*>(gate.data_ptr<mfq_half>()),             \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                        \
                M, K_real, K_pad);                                                      \
        } else {                                                                        \
            quantize_x_gate_nosum_kernel<GSVAL, BD, 2><<<dim3(M, ng), BD, 0, stream>>>( \
                reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                \
                reinterpret_cast<const __half*>(gate.data_ptr<mfq_half>()),             \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                        \
                M, K_real, K_pad);                                                      \
        }                                                                               \
    } while (0)

    switch ((int)gs) {
        case 16: QPBITS_GATE_QUANT(16); break;
        case 20: QPBITS_GATE_QUANT(20); break;
        case 22: QPBITS_GATE_QUANT(22); break;
        case 24: QPBITS_GATE_QUANT(24); break;
        case 26: QPBITS_GATE_QUANT(26); break;
        case 28: QPBITS_GATE_QUANT(28); break;
        case 30: QPBITS_GATE_QUANT(30); break;
        case 32: QPBITS_GATE_QUANT(32); break;
        case 34: QPBITS_GATE_QUANT(34); break;
        case 36: QPBITS_GATE_QUANT(36); break;
        case 40: QPBITS_GATE_QUANT(40); break;
        case 48: QPBITS_GATE_QUANT(48); break;
        case 64: QPBITS_GATE_QUANT(64); break;
        default: MFQ_RUNTIME_CHECK(false, "packed-bits gated GEMV unsupported gs ", gs);
    }
#undef QPBITS_GATE_QUANT

    if (fast_gate6 && M > 1) {
#define QPFASTGATE6(MVAL, SPLITVAL)                                                      \
        gemv_packed_bits_group4_batch_kernel<6, 24, MVAL, SPLITVAL><<<                 \
            dim3((N + 3) / 4), dim3(32, 4 * (SPLITVAL)), 0, stream>>>(                 \
            q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),                \
            sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                \
            neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                        \
            xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), \
            M, N, ng, K_pad)
        if (M == 2)      { QPFASTGATE6(2, 2); }
        else if (M == 3) { QPFASTGATE6(3, 2); }
        else if (M == 4) { QPFASTGATE6(4, 2); }
        else if (M == 5) { QPFASTGATE6(5, 2); }
        else             { QPFASTGATE6(8, 2); }
#undef QPFASTGATE6
        return out;
    }

#define QPBITS_GATE_GEMV(BITSVAL, GSVAL)                                                \
    do {                                                                                \
        if ((BITSVAL) == 6 && (GSVAL) == 26) {                                          \
            gemv_nint6_gs26_batch_kernel<2><<<dim3((N + 3) / 4, (M + 1) / 2), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 1) {                                                            \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 1><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 2) {                                                            \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 2><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 3) {                                                            \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 3><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 4) {                                                            \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 4><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 5) {                                                            \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 5><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else {                                                                        \
            gemv_packed_bits_group4_batch_kernel<BITSVAL, GSVAL, 8><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(), sub_min.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(), \
                xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        }                                                                               \
    } while (0)

#define QPBITS_GATE_GS_SWITCH(BITSVAL)                                                  \
    switch ((int)gs) {                                                                  \
        case 16: QPBITS_GATE_GEMV(BITSVAL, 16); break;                                  \
        case 20: QPBITS_GATE_GEMV(BITSVAL, 20); break;                                  \
        case 22: QPBITS_GATE_GEMV(BITSVAL, 22); break;                                  \
        case 24: QPBITS_GATE_GEMV(BITSVAL, 24); break;                                  \
        case 26: QPBITS_GATE_GEMV(BITSVAL, 26); break;                                  \
        case 28: QPBITS_GATE_GEMV(BITSVAL, 28); break;                                  \
        case 30: QPBITS_GATE_GEMV(BITSVAL, 30); break;                                  \
        case 32: QPBITS_GATE_GEMV(BITSVAL, 32); break;                                  \
        case 34: QPBITS_GATE_GEMV(BITSVAL, 34); break;                                  \
        case 36: QPBITS_GATE_GEMV(BITSVAL, 36); break;                                  \
        case 40: QPBITS_GATE_GEMV(BITSVAL, 40); break;                                  \
        case 48: QPBITS_GATE_GEMV(BITSVAL, 48); break;                                  \
        case 64: QPBITS_GATE_GEMV(BITSVAL, 64); break;                                  \
        default: MFQ_RUNTIME_CHECK(false, "packed-bits gated GEMV unsupported gs ", gs);       \
    }

    if (bits == 2) {
        QPBITS_GATE_GS_SWITCH(2);
    } else if (bits == 3) {
        QPBITS_GATE_GS_SWITCH(3);
    } else if (bits == 5) {
        QPBITS_GATE_GS_SWITCH(5);
    } else if (bits == 6) {
        QPBITS_GATE_GS_SWITCH(6);
    } else if (bits == 8) {
        QPBITS_GATE_GS_SWITCH(8);
    } else {
        MFQ_RUNTIME_CHECK(false, "packed-bits gated GEMV supports bits in {2,3,5,6,8}, got ", bits);
    }
#undef QPBITS_GATE_GS_SWITCH
#undef QPBITS_GATE_GEMV
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_batch_eff_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor d_eff, mfq_tensor_backend::Tensor m_eff, mfq_tensor_backend::Tensor x,
    int64_t gs, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(d_eff.is_cuda() && d_eff.scalar_type() == mfq_tensor_backend::kHalf && d_eff.is_contiguous(),
                "d_eff must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(m_eff.is_cuda() && m_eff.scalar_type() == mfq_tensor_backend::kHalf && m_eff.is_contiguous(),
                "m_eff must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    MFQ_RUNTIME_CHECK((int)d_eff.size(0) == N && (int)d_eff.size(1) == ng, "d_eff shape mismatch");
    MFQ_RUNTIME_CHECK(m_eff.sizes() == d_eff.sizes(), "m_eff shape mismatch");
    int M = (int)x.size(0), K_real = (int)x.size(1);
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 8, "fused-metadata batched GEMV supports M in [1, 8]");
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

#define QPBEWSLAUNCH(GSVAL)                                                              \
    do {                                                                                 \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                     \
        quantize_x_kernel<GSVAL, BD><<<dim3(M, ng), BD, 0, stream>>>(                    \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                     \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),   \
            M, K_real, K_pad);                                                           \
        if (M == 2) {                                                                    \
            gemv_packed_batch_eff_kernel<GSVAL, 2><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(d_eff.data_ptr<mfq_half>()), reinterpret_cast<const __half*>(m_eff.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 3) {                                                             \
            gemv_packed_batch_eff_kernel<GSVAL, 3><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(d_eff.data_ptr<mfq_half>()), reinterpret_cast<const __half*>(m_eff.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 4) {                                                             \
            gemv_packed_batch_eff_kernel<GSVAL, 4><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(d_eff.data_ptr<mfq_half>()), reinterpret_cast<const __half*>(m_eff.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 5) {                                                             \
            gemv_packed_batch_eff_kernel<GSVAL, 5><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(d_eff.data_ptr<mfq_half>()), reinterpret_cast<const __half*>(m_eff.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else {                                                                         \
            gemv_packed_batch_eff_kernel<GSVAL, 8><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(d_eff.data_ptr<mfq_half>()), reinterpret_cast<const __half*>(m_eff.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        }                                                                                \
    } while (0)

    switch ((int)gs) {
        case 16: QPBEWSLAUNCH(16); break;
        case 24: QPBEWSLAUNCH(24); break;
        case 32: QPBEWSLAUNCH(32); break;
        case 48: QPBEWSLAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_gemv_packed_batch_eff_ws: gs must be in {16,24,32,48}, got ", gs);
    }
#undef QPBEWSLAUNCH
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_batch_eff2_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor eff_pair, mfq_tensor_backend::Tensor x,
    int64_t gs, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(eff_pair.is_cuda() && eff_pair.scalar_type() == mfq_tensor_backend::kHalf && eff_pair.is_contiguous(),
                "eff_pair must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    MFQ_RUNTIME_CHECK((int)eff_pair.size(0) == N && (int)eff_pair.size(1) == ng && (int)eff_pair.size(2) == 2,
                "eff_pair shape must be [N, ng, 2]");
    int M = (int)x.size(0), K_real = (int)x.size(1);
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 8, "half2 fused-metadata batched GEMV supports M in [1, 8]");
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

#define QPBE2WSLAUNCH(GSVAL)                                                             \
    do {                                                                                 \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                     \
        quantize_x_kernel<GSVAL, BD><<<dim3(M, ng), BD, 0, stream>>>(                    \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                     \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),   \
            M, K_real, K_pad);                                                           \
        if (M == 2) {                                                                    \
            gemv_packed_batch_eff2_kernel<GSVAL, 2><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half2*>(eff_pair.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 3) {                                                             \
            gemv_packed_batch_eff2_kernel<GSVAL, 3><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half2*>(eff_pair.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 4) {                                                             \
            gemv_packed_batch_eff2_kernel<GSVAL, 4><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half2*>(eff_pair.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 5) {                                                             \
            gemv_packed_batch_eff2_kernel<GSVAL, 5><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half2*>(eff_pair.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else {                                                                         \
            gemv_packed_batch_eff2_kernel<GSVAL, 8><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half2*>(eff_pair.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        }                                                                                \
    } while (0)

    switch ((int)gs) {
        case 16: QPBE2WSLAUNCH(16); break;
        case 24: QPBE2WSLAUNCH(24); break;
        case 32: QPBE2WSLAUNCH(32); break;
        case 48: QPBE2WSLAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_gemv_packed_batch_eff2_ws: gs must be in {16,24,32,48}, got ", gs);
    }
#undef QPBE2WSLAUNCH
    return out;
}

mfq_tensor_backend::Tensor nint_gemv_packed_batch_eff2_gate_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor eff_pair, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor gate,
    int64_t gs, int64_t mode, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(mode == 1 || mode == 2, "gate mode must be 1(sigmoid) or 2(silu)");
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(eff_pair.is_cuda() && eff_pair.scalar_type() == mfq_tensor_backend::kHalf && eff_pair.is_contiguous(),
                "eff_pair must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(gate.is_cuda() && gate.is_contiguous() && gate.scalar_type() == mfq_tensor_backend::kHalf,
                "gate must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(x.sizes() == gate.sizes(), "x and gate must have the same shape");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    MFQ_RUNTIME_CHECK((int)eff_pair.size(0) == N && (int)eff_pair.size(1) == ng && (int)eff_pair.size(2) == 2,
                "eff_pair shape must be [N, ng, 2]");
    int M = (int)x.size(0), K_real = (int)x.size(1);
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 8, "half2 gated batched GEMV supports M in [1, 8]");
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)xsum.size(0) >= M && (int)xsum.size(1) >= ng, "xsum workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

#define QPBE2GATEWSLAUNCH(GSVAL)                                                       \
    do {                                                                                \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                    \
        if (mode == 1) {                                                                \
            quantize_x_gate_kernel<GSVAL, BD, 1><<<dim3(M, ng), BD, 0, stream>>>(       \
                reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                \
                reinterpret_cast<const __half*>(gate.data_ptr<mfq_half>()),             \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), \
                M, K_real, K_pad);                                                      \
        } else {                                                                        \
            quantize_x_gate_kernel<GSVAL, BD, 2><<<dim3(M, ng), BD, 0, stream>>>(       \
                reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                \
                reinterpret_cast<const __half*>(gate.data_ptr<mfq_half>()),             \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(), \
                M, K_real, K_pad);                                                      \
        }                                                                               \
        if (M == 2) {                                                                   \
            gemv_packed_batch_eff2_kernel<GSVAL, 2><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half2*>(eff_pair.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 3) {                                                            \
            gemv_packed_batch_eff2_kernel<GSVAL, 3><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half2*>(eff_pair.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 4) {                                                            \
            gemv_packed_batch_eff2_kernel<GSVAL, 4><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half2*>(eff_pair.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else if (M == 5) {                                                            \
            gemv_packed_batch_eff2_kernel<GSVAL, 5><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half2*>(eff_pair.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        } else {                                                                        \
            gemv_packed_batch_eff2_kernel<GSVAL, 8><<<dim3((N + 3) / 4), dim3(32, 4), 0, stream>>>( \
                q_packed.data_ptr<uint8_t>(), reinterpret_cast<const __half2*>(eff_pair.data_ptr<mfq_half>()), \
                qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad); \
        }                                                                               \
    } while (0)

    switch ((int)gs) {
        case 16: QPBE2GATEWSLAUNCH(16); break;
        case 24: QPBE2GATEWSLAUNCH(24); break;
        case 32: QPBE2GATEWSLAUNCH(32); break;
        case 48: QPBE2GATEWSLAUNCH(48); break;
        default: MFQ_RUNTIME_CHECK(false, "nint_gemv_packed_batch_eff2_gate_ws: gs must be in {16,24,32,48}, got ", gs);
    }
#undef QPBE2GATEWSLAUNCH
    return out;
}

// ---------------------------------------------------------------------------
// Tiled dp4a MMQ (faithful port of llama.cpp mmq.cu structure).
// Block computes a [MMQ_Y neurons x MMQ_X batch] output tile. Per K-chunk (GPK
// groups): weight tile wq[MMQ_Y][KC] + scales and activation tile axq[MMQ_X][KC]
// + xscale are cooperatively loaded into shared memory ONCE, then reused across
// the other axis. Warp/lanes tile the output: threadIdx.y (warp) -> batch col j
// (each warp does MMQ_X/NWARPS cols), threadIdx.x (lane) -> neuron row i (each
// lane does MMQ_Y/32 rows). So each thread accumulates several outputs in
// registers from shared-mem reads (no global re-read, no full Wq materialize),
// matching llama.cpp's warp-subtiled vecdot. Dot is __dp4a(int8,int8).
// ---------------------------------------------------------------------------

template <
    int GS, int GPK, int MMQ_X,
    bool PACKED=false, bool PACKED_VEC_LOAD=false, bool USE_XSUM=false, bool ACT_MMQ=false,
    bool WEIGHT_MMQ=false>
__global__ void __launch_bounds__(256) mmq_kernel(
    const uint8_t* __restrict__ q,          // row-major or MMQ-exec packed weights
    const uint8_t* __restrict__ sub_scale,  // row-major or MMQ-exec scales
    const uint8_t* __restrict__ sub_min,    // row-major or MMQ-exec mins
    const float* __restrict__ neuron_scale, // [N]
    const float* __restrict__ neuron_min,   // [N]
    const int8_t* __restrict__ qx,          // [M, K_pad]
    const float* __restrict__ xscale,       // [M, ng]
    const int32_t* __restrict__ xsum,       // [M, ng]
    __half* __restrict__ out,               // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int NWARPS = 8;
    constexpr int WS = 32;
    constexpr int MMQ_Y = 64;            // neurons per block
    constexpr int KC = GS * GPK;         // K elements per shared-mem chunk
    constexpr int NJ = MMQ_X / NWARPS;   // batch cols per thread
    constexpr int NI = MMQ_Y / WS;       // neuron rows per thread

    int n0 = blockIdx.x * MMQ_Y;
    int m0 = blockIdx.y * MMQ_X;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int tid = warp * WS + lane;
    int total = NWARPS * WS;

    __shared__ uint8_t wq [MMQ_Y][KC];
    __shared__ uint8_t wss[MMQ_Y][GPK];
    __shared__ uint8_t wsm[MMQ_Y][GPK];
    __shared__ int8_t  axq[MMQ_X][KC];
    __shared__ float   axs[MMQ_X][GPK];
    __shared__ int32_t axsum[MMQ_X][GPK];

    float sum_d[NJ][NI], sum_m[NJ][NI];
    #pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
        #pragma unroll
        for (int ii = 0; ii < NI; ++ii) {
            sum_d[jj][ii] = 0.0f;
            sum_m[jj][ii] = 0.0f;
        }
    }

    // this thread's neuron rows -> neuron_scale/min
    float ns[NI], nm[NI];
    #pragma unroll
    for (int ii = 0; ii < NI; ++ii) {
        int n = n0 + lane + ii * WS;
        ns[ii] = (n < N) ? neuron_scale[n] : 0.0f;
        nm[ii] = (n < N) ? neuron_min[n]   : 0.0f;
    }

    int nchunks = (ng + GPK - 1) / GPK;
    for (int c = 0; c < nchunks; ++c) {
        int gbase = c * GPK;
        int eff = min((int)GPK, ng - gbase);   // valid groups this chunk (tail)

        // load weight q tile [MMQ_Y][KC]
        if constexpr (PACKED && PACKED_VEC_LOAD) {
            for (int idx = tid; idx < MMQ_Y * (KC / 4); idx += total) {
                int r = idx / (KC / 4), col4 = (idx % (KC / 4)) * 4;
                int gl = col4 / GS;
                bool ok = (gl < eff) && (n0 + r < N);
                int qv = 0;
                if (ok) {
                    if constexpr (WEIGHT_MMQ) {
                        size_t off = (((size_t)blockIdx.x * nchunks + c) * MMQ_Y + r) * (KC / 2) + (col4 / 2);
                        qv = unpack_int4x4(q + off);
                    } else {
                        qv = unpack_int4x4(q + (size_t)(n0 + r) * ng * (GS / 2) + (gbase + gl) * (GS / 2) + (col4 % GS) / 2);
                    }
                }
                *reinterpret_cast<int*>(wq[r] + col4) = qv;
            }
        } else if constexpr (PACKED) {
            for (int idx = tid; idx < MMQ_Y * (KC / 2); idx += total) {
                int r = idx / (KC / 2), col2 = idx % (KC / 2);
                int col = col2 * 2, gl = col / GS;
                bool ok = (gl < eff) && (n0 + r < N);
                uint8_t packed = 0;
                if (ok) {
                    if constexpr (WEIGHT_MMQ) {
                        size_t off = (((size_t)blockIdx.x * nchunks + c) * MMQ_Y + r) * (KC / 2) + col2;
                        packed = q[off];
                    } else {
                        packed = q[(size_t)(n0 + r) * ng * (GS / 2) + (gbase + gl) * (GS / 2) + (col % GS) / 2];
                    }
                }
                wq[r][col + 0] = packed & 0x0f;
                wq[r][col + 1] = packed >> 4;
            }
        } else {
            for (int idx = tid; idx < MMQ_Y * KC; idx += total) {
                int r = idx / KC, col = idx % KC, gl = col / GS;
                bool ok = (gl < eff) && (n0 + r < N);
                if constexpr (WEIGHT_MMQ) {
                    size_t off = (((size_t)blockIdx.x * nchunks + c) * MMQ_Y + r) * KC + col;
                    wq[r][col] = ok ? q[off] : (uint8_t)0;
                } else {
                    wq[r][col] = ok ? q[(size_t)(n0 + r) * ng * GS + (gbase + gl) * GS + (col % GS)]
                                    : (uint8_t)0;
                }
            }
        }
        // load weight sub_scale/sub_min [MMQ_Y][GPK]
        for (int idx = tid; idx < MMQ_Y * GPK; idx += total) {
            int r = idx / GPK, gl = idx % GPK;
            bool ok = (gl < eff) && (n0 + r < N);
            if constexpr (WEIGHT_MMQ) {
                size_t off = (((size_t)blockIdx.x * nchunks + c) * MMQ_Y + r) * GPK + gl;
                wss[r][gl] = ok ? sub_scale[off] : (uint8_t)0;
                wsm[r][gl] = ok ? sub_min[off]   : (uint8_t)0;
            } else {
                wss[r][gl] = ok ? sub_scale[(size_t)(n0 + r) * ng + gbase + gl] : (uint8_t)0;
                wsm[r][gl] = ok ? sub_min  [(size_t)(n0 + r) * ng + gbase + gl] : (uint8_t)0;
            }
        }
        // load activation qx tile [MMQ_X][KC]
        for (int idx = tid; idx < MMQ_X * KC; idx += total) {
            int r = idx / KC, col = idx % KC, gl = col / GS;
            bool ok = (gl < eff) && (m0 + r < M);
            if constexpr (ACT_MMQ) {
                size_t off = (((size_t)blockIdx.y * nchunks + c) * MMQ_X + r) * KC + col;
                axq[r][col] = ok ? qx[off] : (int8_t)0;
            } else {
                axq[r][col] = ok ? qx[(size_t)(m0 + r) * K_pad + (gbase + gl) * GS + (col % GS)]
                                 : (int8_t)0;
            }
        }
        // load activation xscale [MMQ_X][GPK]
        for (int idx = tid; idx < MMQ_X * GPK; idx += total) {
            int r = idx / GPK, gl = idx % GPK;
            bool ok = (gl < eff) && (m0 + r < M);
            if constexpr (ACT_MMQ) {
                size_t off = (((size_t)blockIdx.y * nchunks + c) * MMQ_X + r) * GPK + gl;
                axs[r][gl] = ok ? xscale[off] : 0.0f;
                if constexpr (USE_XSUM) {
                    axsum[r][gl] = ok ? xsum[off] : 0;
                }
            } else {
                axs[r][gl] = ok ? xscale[(size_t)(m0 + r) * ng + gbase + gl] : 0.0f;
                if constexpr (USE_XSUM) {
                    axsum[r][gl] = ok ? xsum[(size_t)(m0 + r) * ng + gbase + gl] : 0;
                }
            }
        }
        __syncthreads();

        // accumulate this thread's NJ*NI outputs over the chunk's groups
        #pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            int j = warp + jj * NWARPS;          // batch col in tile
            int m = m0 + j;
            #pragma unroll
            for (int ii = 0; ii < NI; ++ii) {
                int i = lane + ii * WS;          // neuron row in tile
                int n = n0 + i;
                if (m < M && n < N) {
                    float pd = 0.0f, pm = 0.0f;
                    #pragma unroll
                    for (int gl = 0; gl < GPK; ++gl) {
                        int g = gbase + gl;
                        if (g >= ng) {
                            break;
                        }
                        float xs = axs[j][gl];
                        float ss = (float)wss[i][gl];
                        float sm = (float)wsm[i][gl];
                        int di = 0, mi = 0;
                        #pragma unroll
                        for (int t = 0; t < GS / 4; ++t) {
                            int qv = *reinterpret_cast<const int*>(wq[i] + gl * GS + t * 4);
                            int xv = *reinterpret_cast<const int*>(axq[j] + gl * GS + t * 4);
                            di = __dp4a(qv, xv, di);
                            if constexpr (!USE_XSUM) {
                                mi = __dp4a(0x01010101, xv, mi);
                            }
                        }
                        pd += xs * ss * (float)di;
                        if constexpr (USE_XSUM) {
                            pm += xs * sm * (float)axsum[j][gl];
                        } else {
                            pm += xs * sm * (float)mi;
                        }
                    }
                    sum_d[jj][ii] += pd;
                    sum_m[jj][ii] += pm;
                }
            }
        }
        __syncthreads();
    }

    // write outputs
    #pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
        int m = m0 + warp + jj * NWARPS;
        #pragma unroll
        for (int ii = 0; ii < NI; ++ii) {
            int n = n0 + lane + ii * WS;
            if (m < M && n < N) {
                out[(size_t)m * N + n] = __float2half(ns[ii] * sum_d[jj][ii] - nm[ii] * sum_m[jj][ii]);
            }
        }
    }
}

template <int MMQ_X>
__global__ void __launch_bounds__(256) mmq24_packed_ws_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, 12]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    const int32_t* __restrict__ xsum,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int GS = 24;
    constexpr int GPK = 10;
    constexpr int CHUNKS = 6;
    constexpr int QBYTES = 12;
    constexpr int NWARPS = 8;
    constexpr int WS = 32;
    constexpr int MMQ_Y = 64;
    constexpr int NJ = MMQ_X / NWARPS;
    constexpr int NI = MMQ_Y / WS;

    int n0 = blockIdx.x * MMQ_Y;
    int m0 = blockIdx.y * MMQ_X;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int tid = warp * WS + lane;
    int total = NWARPS * WS;

    __shared__ int     wqi[MMQ_Y][GPK * CHUNKS + 1];
    __shared__ uint8_t wss[MMQ_Y][GPK + 1];
    __shared__ uint8_t wsm[MMQ_Y][GPK + 1];
    __shared__ int     axqi[MMQ_X][GPK][CHUNKS];
    __shared__ float   axs[MMQ_X][GPK];
    __shared__ int32_t axsum[MMQ_X][GPK];

    float sum_d[NJ][NI], sum_m[NJ][NI];
    #pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
        #pragma unroll
        for (int ii = 0; ii < NI; ++ii) {
            sum_d[jj][ii] = 0.0f;
            sum_m[jj][ii] = 0.0f;
        }
    }

    float ns[NI], nm[NI];
    #pragma unroll
    for (int ii = 0; ii < NI; ++ii) {
        int n = n0 + lane + ii * WS;
        ns[ii] = (n < N) ? neuron_scale[n] : 0.0f;
        nm[ii] = (n < N) ? neuron_min[n] : 0.0f;
    }

    int nchunks = (ng + GPK - 1) / GPK;
    for (int c = 0; c < nchunks; ++c) {
        int gbase = c * GPK;
        int eff = min((int)GPK, ng - gbase);

        for (int idx = tid; idx < MMQ_Y * GPK * CHUNKS; idx += total) {
            int chunk = idx % CHUNKS;
            int gl = (idx / CHUNKS) % GPK;
            int r = idx / (GPK * CHUNKS);
            bool ok = (gl < eff) && (n0 + r < N);
            int qv = 0;
            if (ok) {
                qv = unpack_int4x4(q_packed + (size_t)(n0 + r) * ng * QBYTES
                                    + (size_t)(gbase + gl) * QBYTES + chunk * 2);
            }
            wqi[r][gl * CHUNKS + chunk] = qv;
        }

        for (int idx = tid; idx < MMQ_Y * GPK; idx += total) {
            int r = idx / GPK;
            int gl = idx % GPK;
            bool ok = (gl < eff) && (n0 + r < N);
            wss[r][gl] = ok ? sub_scale[(size_t)(n0 + r) * ng + gbase + gl] : (uint8_t)0;
            wsm[r][gl] = ok ? sub_min  [(size_t)(n0 + r) * ng + gbase + gl] : (uint8_t)0;
        }

        for (int idx = tid; idx < MMQ_X * GPK * CHUNKS; idx += total) {
            int chunk = idx % CHUNKS;
            int gl = (idx / CHUNKS) % GPK;
            int r = idx / (GPK * CHUNKS);
            bool ok = (gl < eff) && (m0 + r < M);
            int xv = 0;
            if (ok) {
                xv = *reinterpret_cast<const int*>(qx + (size_t)(m0 + r) * K_pad
                                                   + (size_t)(gbase + gl) * GS + chunk * 4);
            }
            axqi[r][gl][chunk] = xv;
        }

        for (int idx = tid; idx < MMQ_X * GPK; idx += total) {
            int r = idx / GPK;
            int gl = idx % GPK;
            bool ok = (gl < eff) && (m0 + r < M);
            axs[r][gl] = ok ? xscale[(size_t)(m0 + r) * ng + gbase + gl] : 0.0f;
            axsum[r][gl] = ok ? xsum[(size_t)(m0 + r) * ng + gbase + gl] : 0;
        }
        __syncthreads();

        #pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            int j = warp + jj * NWARPS;
            int m = m0 + j;
            #pragma unroll
            for (int ii = 0; ii < NI; ++ii) {
                int i = lane + ii * WS;
                int n = n0 + i;
                if (m < M && n < N) {
                    float pd = 0.0f;
                    float pm = 0.0f;
                    #pragma unroll
                    for (int gl = 0; gl < GPK; ++gl) {
                        int g = gbase + gl;
                        if (g >= ng) {
                            break;
                        }
                        int di = 0;
                        di = __dp4a(wqi[i][gl * CHUNKS + 0], axqi[j][gl][0], di);
                        di = __dp4a(wqi[i][gl * CHUNKS + 1], axqi[j][gl][1], di);
                        di = __dp4a(wqi[i][gl * CHUNKS + 2], axqi[j][gl][2], di);
                        di = __dp4a(wqi[i][gl * CHUNKS + 3], axqi[j][gl][3], di);
                        di = __dp4a(wqi[i][gl * CHUNKS + 4], axqi[j][gl][4], di);
                        di = __dp4a(wqi[i][gl * CHUNKS + 5], axqi[j][gl][5], di);
                        float xs = axs[j][gl];
                        pd += xs * (float)wss[i][gl] * (float)di;
                        pm += xs * (float)wsm[i][gl] * (float)axsum[j][gl];
                    }
                    sum_d[jj][ii] += pd;
                    sum_m[jj][ii] += pm;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
        int m = m0 + warp + jj * NWARPS;
        #pragma unroll
        for (int ii = 0; ii < NI; ++ii) {
            int n = n0 + lane + ii * WS;
            if (m < M && n < N) {
                out[(size_t)m * N + n] = __float2half(ns[ii] * sum_d[jj][ii] - nm[ii] * sum_m[jj][ii]);
            }
        }
    }
}

template <int MMQ_X, int NWARPS>
__global__ void __launch_bounds__(256) mmq24_small_packed_ws_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, 12]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    const int32_t* __restrict__ xsum,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int GS = 24;
    constexpr int GPK = 10;
    constexpr int CHUNKS = 6;
    constexpr int QBYTES = 12;
    constexpr int WS = 32;
    constexpr int MMQ_Y = 32;
    constexpr int NJ = MMQ_X / NWARPS;

    int n0 = blockIdx.x * MMQ_Y;
    int m0 = blockIdx.y * MMQ_X;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int tid = warp * WS + lane;
    int total = NWARPS * WS;

    __shared__ int     wqi[MMQ_Y][GPK * CHUNKS + 1];
    __shared__ uint8_t wss[MMQ_Y][GPK + 1];
    __shared__ uint8_t wsm[MMQ_Y][GPK + 1];
    __shared__ int     axqi[MMQ_X][GPK][CHUNKS];
    __shared__ float   axs[MMQ_X][GPK];
    __shared__ int32_t axsum[MMQ_X][GPK];

    float sum_d[NJ], sum_m[NJ];
    #pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
        sum_d[jj] = 0.0f;
        sum_m[jj] = 0.0f;
    }

    int n = n0 + lane;
    float ns = (n < N) ? neuron_scale[n] : 0.0f;
    float nm = (n < N) ? neuron_min[n] : 0.0f;

    int nchunks = (ng + GPK - 1) / GPK;
    for (int c = 0; c < nchunks; ++c) {
        int gbase = c * GPK;
        int eff = min((int)GPK, ng - gbase);

        for (int idx = tid; idx < MMQ_Y * GPK * CHUNKS; idx += total) {
            int chunk = idx % CHUNKS;
            int gl = (idx / CHUNKS) % GPK;
            int r = idx / (GPK * CHUNKS);
            bool ok = (gl < eff) && (n0 + r < N);
            int qv = 0;
            if (ok) {
                qv = unpack_int4x4(q_packed + (size_t)(n0 + r) * ng * QBYTES
                                    + (size_t)(gbase + gl) * QBYTES + chunk * 2);
            }
            wqi[r][gl * CHUNKS + chunk] = qv;
        }

        for (int idx = tid; idx < MMQ_Y * GPK; idx += total) {
            int r = idx / GPK;
            int gl = idx % GPK;
            bool ok = (gl < eff) && (n0 + r < N);
            wss[r][gl] = ok ? sub_scale[(size_t)(n0 + r) * ng + gbase + gl] : (uint8_t)0;
            wsm[r][gl] = ok ? sub_min  [(size_t)(n0 + r) * ng + gbase + gl] : (uint8_t)0;
        }

        for (int idx = tid; idx < MMQ_X * GPK * CHUNKS; idx += total) {
            int chunk = idx % CHUNKS;
            int gl = (idx / CHUNKS) % GPK;
            int r = idx / (GPK * CHUNKS);
            bool ok = (gl < eff) && (m0 + r < M);
            int xv = 0;
            if (ok) {
                xv = *reinterpret_cast<const int*>(qx + (size_t)(m0 + r) * K_pad
                                                   + (size_t)(gbase + gl) * GS + chunk * 4);
            }
            axqi[r][gl][chunk] = xv;
        }

        for (int idx = tid; idx < MMQ_X * GPK; idx += total) {
            int r = idx / GPK;
            int gl = idx % GPK;
            bool ok = (gl < eff) && (m0 + r < M);
            axs[r][gl] = ok ? xscale[(size_t)(m0 + r) * ng + gbase + gl] : 0.0f;
            axsum[r][gl] = ok ? xsum[(size_t)(m0 + r) * ng + gbase + gl] : 0;
        }
        __syncthreads();

        if (n < N) {
            #pragma unroll
            for (int jj = 0; jj < NJ; ++jj) {
                int j = warp + jj * NWARPS;
                int m = m0 + j;
                if (m < M) {
                    float pd = 0.0f;
                    float pm = 0.0f;
                    #pragma unroll
                    for (int gl = 0; gl < GPK; ++gl) {
                        int g = gbase + gl;
                        if (g >= ng) {
                            break;
                        }
                        int di = 0;
                        di = __dp4a(wqi[lane][gl * CHUNKS + 0], axqi[j][gl][0], di);
                        di = __dp4a(wqi[lane][gl * CHUNKS + 1], axqi[j][gl][1], di);
                        di = __dp4a(wqi[lane][gl * CHUNKS + 2], axqi[j][gl][2], di);
                        di = __dp4a(wqi[lane][gl * CHUNKS + 3], axqi[j][gl][3], di);
                        di = __dp4a(wqi[lane][gl * CHUNKS + 4], axqi[j][gl][4], di);
                        di = __dp4a(wqi[lane][gl * CHUNKS + 5], axqi[j][gl][5], di);
                        float xs = axs[j][gl];
                        pd += xs * (float)wss[lane][gl] * (float)di;
                        pm += xs * (float)wsm[lane][gl] * (float)axsum[j][gl];
                    }
                    sum_d[jj] += pd;
                    sum_m[jj] += pm;
                }
            }
        }
        __syncthreads();
    }

    if (n < N) {
        #pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            int m = m0 + warp + jj * NWARPS;
            if (m < M) {
                out[(size_t)m * N + n] = __float2half(ns * sum_d[jj] - nm * sum_m[jj]);
            }
        }
    }
}

template <int GS, int GPK, int MMQ_X>
__global__ void __launch_bounds__(256) mmq_u8_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, GS]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const int8_t* __restrict__ qx,           // [M, K_pad]
    const float* __restrict__ xscale,        // [M, ng]
    const int32_t* __restrict__ xsum,        // [M, ng]
    __half* __restrict__ out,                // [M, N]
    int M, int N, int ng, int K_pad)
{
    constexpr int NWARPS = 8;
    constexpr int WS = 32;
    constexpr int MMQ_Y = 64;
    constexpr int KC = GS * GPK;
    constexpr int NJ = MMQ_X / NWARPS;
    constexpr int NI = MMQ_Y / WS;

    int n0 = blockIdx.x * MMQ_Y;
    int m0 = blockIdx.y * MMQ_X;
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int tid = warp * WS + lane;
    int total = NWARPS * WS;

    __shared__ uint8_t wq [MMQ_Y][KC];
    __shared__ uint8_t wss[MMQ_Y][GPK];
    __shared__ uint8_t wsm[MMQ_Y][GPK];
    __shared__ int8_t  axq[MMQ_X][KC];
    __shared__ float   axs[MMQ_X][GPK];
    __shared__ int32_t axsum[MMQ_X][GPK];

    float sum_d[NJ][NI], sum_m[NJ][NI];
    #pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
        #pragma unroll
        for (int ii = 0; ii < NI; ++ii) {
            sum_d[jj][ii] = 0.0f;
            sum_m[jj][ii] = 0.0f;
        }
    }

    float ns[NI], nm[NI];
    #pragma unroll
    for (int ii = 0; ii < NI; ++ii) {
        int n = n0 + lane + ii * WS;
        ns[ii] = (n < N) ? neuron_scale[n] : 0.0f;
        nm[ii] = (n < N) ? neuron_min[n] : 0.0f;
    }

    int nchunks = (ng + GPK - 1) / GPK;
    for (int c = 0; c < nchunks; ++c) {
        int gbase = c * GPK;
        int eff = min((int)GPK, ng - gbase);

        for (int idx = tid; idx < MMQ_Y * KC; idx += total) {
            int r = idx / KC;
            int col = idx % KC;
            int gl = col / GS;
            bool ok = (gl < eff) && (n0 + r < N);
            wq[r][col] = ok ? q_packed[(size_t)(n0 + r) * ng * GS + (gbase + gl) * GS + (col % GS)]
                             : (uint8_t)128;
        }
        for (int idx = tid; idx < MMQ_Y * GPK; idx += total) {
            int r = idx / GPK;
            int gl = idx % GPK;
            bool ok = (gl < eff) && (n0 + r < N);
            wss[r][gl] = ok ? sub_scale[(size_t)(n0 + r) * ng + gbase + gl] : (uint8_t)0;
            wsm[r][gl] = ok ? sub_min  [(size_t)(n0 + r) * ng + gbase + gl] : (uint8_t)0;
        }
        for (int idx = tid; idx < MMQ_X * KC; idx += total) {
            int r = idx / KC;
            int col = idx % KC;
            int gl = col / GS;
            bool ok = (gl < eff) && (m0 + r < M);
            axq[r][col] = ok ? qx[(size_t)(m0 + r) * K_pad + (gbase + gl) * GS + (col % GS)]
                              : (int8_t)0;
        }
        for (int idx = tid; idx < MMQ_X * GPK; idx += total) {
            int r = idx / GPK;
            int gl = idx % GPK;
            bool ok = (gl < eff) && (m0 + r < M);
            axs[r][gl] = ok ? xscale[(size_t)(m0 + r) * ng + gbase + gl] : 0.0f;
            axsum[r][gl] = ok ? xsum[(size_t)(m0 + r) * ng + gbase + gl] : 0;
        }
        __syncthreads();

        #pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            int j = warp + jj * NWARPS;
            int m = m0 + j;
            #pragma unroll
            for (int ii = 0; ii < NI; ++ii) {
                int i = lane + ii * WS;
                int n = n0 + i;
                if (m < M && n < N) {
                    float pd = 0.0f;
                    float pm = 0.0f;
                    #pragma unroll
                    for (int gl = 0; gl < GPK; ++gl) {
                        int g = gbase + gl;
                        if (g >= ng) {
                            break;
                        }
                        int di = 0;
                        #pragma unroll
                        for (int t = 0; t < GS / 4; ++t) {
                            int qv = *reinterpret_cast<const int*>(wq[i] + gl * GS + t * 4);
                            int xv = *reinterpret_cast<const int*>(axq[j] + gl * GS + t * 4);
                            di = __dp4a(qv ^ (int)0x80808080u, xv, di);
                        }
                        float xs = axs[j][gl];
                        float xsg = (float)axsum[j][gl];
                        pd += xs * (float)wss[i][gl] * ((float)di + 128.0f * xsg);
                        pm += xs * (float)wsm[i][gl] * xsg;
                    }
                    sum_d[jj][ii] += pd;
                    sum_m[jj][ii] += pm;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
        int m = m0 + warp + jj * NWARPS;
        #pragma unroll
        for (int ii = 0; ii < NI; ++ii) {
            int n = n0 + lane + ii * WS;
            if (m < M && n < N) {
                out[(size_t)m * N + n] = __float2half(ns[ii] * sum_d[jj][ii] - nm[ii] * sum_m[jj][ii]);
            }
        }
    }
}

mfq_tensor_backend::Tensor nint_mmq_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs)
{
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.scalar_type() == mfq_tensor_backend::kUInt8 && q.is_contiguous(),
                "q must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    int N = (int)q.size(0), ng = (int)q.size(1);
    int M = (int)x.size(0), K_real = (int)x.size(1);
    int K_pad = ng * (int)gs;
    auto qx = mfq_tensor_backend::empty({M, K_pad}, x.options().dtype(mfq_tensor_backend::kInt8));
    auto xscale = mfq_tensor_backend::empty({M, ng}, x.options().dtype(mfq_tensor_backend::kFloat32));
    auto xsum = mfq_tensor_backend::empty({M, ng}, x.options().dtype(mfq_tensor_backend::kInt32));
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    constexpr int MMQ_Y = 64;
    // pick batch tile to match M (avoid wasted batch slots, like llama.cpp mmq_x dispatch)
    int mmq_x = (M <= 8) ? 8 : (M <= 16 ? 16 : 32);

#define MLAUNCH(GSVAL, GPKVAL, MMQXVAL)                                                     \
    do {                                                                                    \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                        \
        quantize_x_kernel<GSVAL, BD, true><<<dim3(M, ng), BD, 0, stream>>>(                 \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                        \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),      \
            M, K_real, K_pad);                                                              \
        mmq_kernel<GSVAL, GPKVAL, MMQXVAL, false, false, true><<<dim3((N + MMQ_Y - 1) / MMQ_Y, \
                    (M + MMQXVAL - 1) / MMQXVAL), dim3(32, 8), 0, stream>>>(                \
            q.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),                           \
            sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                    \
            neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                            \
            xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                             \
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);          \
    } while (0)

#define MLAUNCH_GS(GSVAL, GPKVAL)                                       \
    if (mmq_x == 8)  { MLAUNCH(GSVAL, GPKVAL, 8);  }                    \
    else if (mmq_x == 16) { MLAUNCH(GSVAL, GPKVAL, 16); }              \
    else                 { MLAUNCH(GSVAL, GPKVAL, 32); }

    switch ((int)gs) {
        case 16: MLAUNCH_GS(16, 16); break;
        case 24: MLAUNCH_GS(24, 10); break;
        case 32: MLAUNCH_GS(32, 8);  break;
        case 48: MLAUNCH_GS(48, 5);  break;
        default: MFQ_RUNTIME_CHECK(false, "nint_mmq: gs must be in {16,24,32,48}, got ", gs);
    }
#undef MLAUNCH_GS
#undef MLAUNCH
    return out;
}

mfq_tensor_backend::Tensor nint_mmq_packed_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) * 2 == gs, "q_packed last dim must equal gs/2");
    int M = (int)x.size(0), K_real = (int)x.size(1);
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)xsum.size(0) >= M && (int)xsum.size(1) >= ng, "xsum workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    constexpr int MMQ_Y = 64;
    int mmq_x = (M <= 8) ? 8 : (M <= 16 ? 16 : 32);

#define MPWSLAUNCH(GSVAL, GPKVAL, MMQXVAL)                                                 \
    do {                                                                                    \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                        \
        quantize_x_kernel<GSVAL, BD, true><<<dim3(M, ng), BD, 0, stream>>>(                 \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                        \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),      \
            M, K_real, K_pad);                                                              \
        mmq_kernel<GSVAL, GPKVAL, MMQXVAL, true, (MMQXVAL == 32), true><<<dim3((N + MMQ_Y - 1) / MMQ_Y, \
                    (M + MMQXVAL - 1) / MMQXVAL), dim3(32, 8), 0, stream>>>(                \
            q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),                    \
            sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                    \
            neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                            \
            xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                             \
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);          \
    } while (0)

#define MPWSLAUNCH_SPLIT24(MMQXVAL)                                                         \
    do {                                                                                    \
        quantize_x_kernel<24, 32, true><<<dim3(M, ng), 32, 0, stream>>>(                    \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                        \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),      \
            M, K_real, K_pad);                                                              \
        mmq24_packed_ws_kernel<MMQXVAL><<<                                                   \
                dim3((N + MMQ_Y - 1) / MMQ_Y, (M + MMQXVAL - 1) / MMQXVAL),                 \
                dim3(32, 8), 0, stream>>>(                                                  \
            q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),                    \
            sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                    \
            neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                            \
            xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                             \
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);          \
    } while (0)

#define MPWSLAUNCH_SMALL24(MMQXVAL, WARPSVAL)                                               \
    do {                                                                                    \
        quantize_x_kernel<24, 32, true><<<dim3(M, ng), 32, 0, stream>>>(                    \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                        \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),      \
            M, K_real, K_pad);                                                              \
        mmq24_small_packed_ws_kernel<MMQXVAL, WARPSVAL><<<                                  \
                dim3((N + 31) / 32, (M + (MMQXVAL) - 1) / (MMQXVAL)), dim3(32, WARPSVAL), 0, stream>>>( \
            q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),                    \
            sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                    \
            neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                            \
            xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                             \
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);          \
    } while (0)

#define MPWSLAUNCH_GS(GSVAL, GPKVAL)                                   \
    if (mmq_x == 8)  { MPWSLAUNCH(GSVAL, GPKVAL, 8);  }                 \
    else if (mmq_x == 16) { MPWSLAUNCH(GSVAL, GPKVAL, 16); }            \
    else                 { MPWSLAUNCH(GSVAL, GPKVAL, 32); }

    if ((int)gs == 24) {
        const char* m8_env = std::getenv("MFQ_NINT_GS24_M8");
        bool use_m8 = (M == 8) && (m8_env == nullptr || m8_env[0] != '0');
        if (use_m8) {
            const char* m8_warps_env = std::getenv("MFQ_NINT_GS24_M8_WARPS");
            if (m8_warps_env != nullptr && m8_warps_env[0] == '4') { MPWSLAUNCH_SMALL24(8, 4); }
            else { MPWSLAUNCH_SMALL24(8, 8); }
            return out;
        }
        const char* split_env = std::getenv("MFQ_NINT_GS24_MMQ_SPLIT");
        bool use_split24 = (mmq_x != 8);
        if (split_env != nullptr) {
            use_split24 = (split_env[0] == '1');
        }
        if (use_split24) {
            const char* m16_env = std::getenv("MFQ_NINT_GS24_M16");
            bool use_m16 = (M == 16) && (m16_env == nullptr || m16_env[0] != '0');
            const char* m16_warps_env = std::getenv("MFQ_NINT_GS24_M16_WARPS");
            if (use_m16 && m16_warps_env != nullptr && m16_warps_env[0] == '4') { MPWSLAUNCH_SMALL24(16, 4); }
            else if (use_m16) { MPWSLAUNCH_SMALL24(16, 8); }
            else if (mmq_x == 8) { MPWSLAUNCH_SPLIT24(8); }
            else if (mmq_x == 16) { MPWSLAUNCH_SPLIT24(16); }
            else { MPWSLAUNCH_SPLIT24(32); }
            return out;
        }
    }

    switch ((int)gs) {
        case 16: MPWSLAUNCH_GS(16, 16); break;
        case 24: MPWSLAUNCH_GS(24, 10); break;
        case 32: MPWSLAUNCH_GS(32, 8);  break;
        case 48: MPWSLAUNCH_GS(48, 5);  break;
        default: MFQ_RUNTIME_CHECK(false, "nint_mmq_packed_ws: gs must be in {16,24,32,48}, got ", gs);
    }
#undef MPWSLAUNCH_GS
#undef MPWSLAUNCH_SMALL24
#undef MPWSLAUNCH_SPLIT24
#undef MPWSLAUNCH
    return out;
}

mfq_tensor_backend::Tensor nint_mmq_packed_u8_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t gs, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    int N = (int)q_packed.size(0), ng = (int)q_packed.size(1);
    MFQ_RUNTIME_CHECK((int)q_packed.size(2) == (int)gs, "NINT8 q_packed last dim must equal gs");
    int M = (int)x.size(0), K_real = (int)x.size(1);
    MFQ_RUNTIME_CHECK(M >= 1 && M <= 4096, "NINT8 MMQ supports M in [1, 4096]");
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)xsum.size(0) >= M && (int)xsum.size(1) >= ng, "xsum workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    constexpr int MMQ_Y = 64;
    int mmq_x = (M <= 8) ? 8 : (M <= 16 ? 16 : 32);

#define MPU8LAUNCH(GSVAL, GPKVAL, MMQXVAL)                                                \
    do {                                                                                    \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                        \
        quantize_x_kernel<GSVAL, BD, true><<<dim3(M, ng), BD, 0, stream>>>(                 \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                        \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),      \
            M, K_real, K_pad);                                                              \
        mmq_u8_kernel<GSVAL, GPKVAL, MMQXVAL><<<                                            \
                dim3((N + MMQ_Y - 1) / MMQ_Y, (M + MMQXVAL - 1) / MMQXVAL),                 \
                dim3(32, 8), 0, stream>>>(                                                  \
            q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),                    \
            sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                    \
            neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                            \
            xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                             \
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);          \
    } while (0)

#define MPU8LAUNCH_GS(GSVAL, GPKVAL)                                  \
    if (mmq_x == 8) { MPU8LAUNCH(GSVAL, GPKVAL, 8); }                  \
    else if (mmq_x == 16) { MPU8LAUNCH(GSVAL, GPKVAL, 16); }           \
    else { MPU8LAUNCH(GSVAL, GPKVAL, 32); }

    switch ((int)gs) {
        case 16: MPU8LAUNCH_GS(16, 16); break;
        case 24: MPU8LAUNCH_GS(24, 10); break;
        case 32: MPU8LAUNCH_GS(32, 8);  break;
        case 48: MPU8LAUNCH_GS(48, 5);  break;
        case 64: MPU8LAUNCH_GS(64, 4);  break;
        default: MFQ_RUNTIME_CHECK(false, "NINT8 MMQ unsupported gs ", gs);
    }
#undef MPU8LAUNCH_GS
#undef MPU8LAUNCH
    return out;
}

mfq_tensor_backend::Tensor nint_mmq_packed_exec_ws_cuda(
    mfq_tensor_backend::Tensor q_mmq_packed, mfq_tensor_backend::Tensor sub_scale_mmq, mfq_tensor_backend::Tensor sub_min_mmq,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    int64_t ng_in, int64_t gs, mfq_tensor_backend::Tensor qx, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum)
{
    MFQ_RUNTIME_CHECK(q_mmq_packed.is_cuda() && q_mmq_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_mmq_packed.is_contiguous(),
                "q_mmq_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale_mmq.is_cuda() && sub_scale_mmq.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale_mmq.is_contiguous(),
                "sub_scale_mmq must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min_mmq.is_cuda() && sub_min_mmq.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min_mmq.is_contiguous(),
                "sub_min_mmq must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kHalf,
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() && qx.scalar_type() == mfq_tensor_backend::kInt8,
                "qx workspace must be cuda contiguous int8");
    MFQ_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() && xscale.scalar_type() == mfq_tensor_backend::kFloat32,
                "xscale workspace must be cuda contiguous f32");
    MFQ_RUNTIME_CHECK(xsum.is_cuda() && xsum.is_contiguous() && xsum.scalar_type() == mfq_tensor_backend::kInt32,
                "xsum workspace must be cuda contiguous int32");
    MFQ_RUNTIME_CHECK(q_mmq_packed.dim() == 4 && sub_scale_mmq.dim() == 4 && sub_min_mmq.dim() == 4,
                "MMQ execution weights must be rank-4");
    int N = (int)neuron_scale.size(0);
    int ng = (int)ng_in;
    int M = (int)x.size(0), K_real = (int)x.size(1);
    int K_pad = ng * (int)gs;
    MFQ_RUNTIME_CHECK((int)neuron_min.size(0) == N, "neuron_min size mismatch");
    MFQ_RUNTIME_CHECK((int)qx.size(0) >= M && (int)qx.size(1) >= K_pad, "qx workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)xsum.size(0) >= M && (int)xsum.size(1) >= ng, "xsum workspace too small");
    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    constexpr int MMQ_Y = 64;
    int mmq_x = (M <= 8) ? 8 : (M <= 16 ? 16 : 32);

#define MPEXECWSLAUNCH(GSVAL, GPKVAL, MMQXVAL)                                            \
    do {                                                                                    \
        constexpr int BD = ((GSVAL + 31) / 32) * 32;                                        \
        int nchunks = (ng + GPKVAL - 1) / GPKVAL;                                          \
        MFQ_RUNTIME_CHECK((int)q_mmq_packed.size(0) >= (N + MMQ_Y - 1) / MMQ_Y,                  \
                    "q_mmq_packed neuron tiles too small");                                \
        MFQ_RUNTIME_CHECK((int)q_mmq_packed.size(1) >= nchunks && (int)q_mmq_packed.size(2) == MMQ_Y && \
                    (int)q_mmq_packed.size(3) == (GSVAL * GPKVAL) / 2,                     \
                    "q_mmq_packed shape mismatch");                                       \
        MFQ_RUNTIME_CHECK((int)sub_scale_mmq.size(0) >= (N + MMQ_Y - 1) / MMQ_Y &&               \
                    (int)sub_scale_mmq.size(1) >= nchunks &&                               \
                    (int)sub_scale_mmq.size(2) == MMQ_Y &&                                 \
                    (int)sub_scale_mmq.size(3) == GPKVAL,                                  \
                    "sub_scale_mmq shape mismatch");                                      \
        MFQ_RUNTIME_CHECK(sub_min_mmq.sizes() == sub_scale_mmq.sizes(), "sub_min_mmq shape mismatch"); \
        quantize_x_kernel<GSVAL, BD, true><<<dim3(M, ng), BD, 0, stream>>>(                 \
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                        \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),      \
            M, K_real, K_pad);                                                              \
        mmq_kernel<GSVAL, GPKVAL, MMQXVAL, true, false, true, false, true><<<               \
                    dim3((N + MMQ_Y - 1) / MMQ_Y, (M + MMQXVAL - 1) / MMQXVAL),             \
                    dim3(32, 8), 0, stream>>>(                                              \
            q_mmq_packed.data_ptr<uint8_t>(), sub_scale_mmq.data_ptr<uint8_t>(),            \
            sub_min_mmq.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                \
            neuron_min.data_ptr<float>(), qx.data_ptr<int8_t>(),                            \
            xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                             \
            reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), M, N, ng, K_pad);          \
    } while (0)

#define MPEXECWSLAUNCH_GS(GSVAL, GPKVAL)                            \
    if (mmq_x == 8)  { MPEXECWSLAUNCH(GSVAL, GPKVAL, 8);  }          \
    else if (mmq_x == 16) { MPEXECWSLAUNCH(GSVAL, GPKVAL, 16); }     \
    else                 { MPEXECWSLAUNCH(GSVAL, GPKVAL, 32); }

    switch ((int)gs) {
        case 16: MPEXECWSLAUNCH_GS(16, 16); break;
        case 24: MPEXECWSLAUNCH_GS(24, 10); break;
        case 32: MPEXECWSLAUNCH_GS(32, 8);  break;
        case 48: MPEXECWSLAUNCH_GS(48, 5);  break;
        default: MFQ_RUNTIME_CHECK(false, "nint_mmq_packed_exec_ws: gs must be in {16,24,32,48}, got ", gs);
    }
#undef MPEXECWSLAUNCH_GS
#undef MPEXECWSLAUNCH
    return out;
}

// Integer Tensor Core fragment helpers used by the group32 MMQ kernel.

static __device__ __forceinline__ int mma168_i(const int l)
{
    return ((l / 2) * 8) + (threadIdx.x / 4);
}

static __device__ __forceinline__ int mma168_j(const int l)
{
    return ((threadIdx.x % 4) * 2) + (l % 2);
}

static __device__ __forceinline__ void load_mma_a_m16n8k32(int (&a)[4], const int* ptr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "l"(ptr));
}

static __device__ __forceinline__ void load_mma_b_m16n8k32(int (&b)[2], const int* ptr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(b[0]), "=r"(b[1])
        : "l"(ptr));
}

static __device__ __forceinline__ void mma_m16n8k32_s8(int (&d)[4], const int (&a)[4], const int (&b)[2])
{
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// NINT integer MMQ with one transient K32 tile per group. Padding values are
// shared-memory zeros, so compact gs16/gs24 weights each need one
// m16n8k32 instruction per group.
template <int BITS, int GS, int MTILES, int NFRAGS, bool NEED_CHECK>
__global__ void __launch_bounds__(256) mmq_mma_group32_kernel(
    const uint8_t* __restrict__ q_packed,
    const uint8_t* __restrict__ sub_scale,
    const uint8_t* __restrict__ sub_min,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_min,
    const int32_t* __restrict__ qx_mmq,
    const float* __restrict__ xscale,
    const int32_t* __restrict__ xsum,
    __half* __restrict__ out,
    float* __restrict__ partial,
    int M, int N, int ng, int M_pad)
{
    constexpr int CHUNK_GROUPS = 8;
    constexpr int GROUP_KPACK = 8;
    constexpr int A_GROUPS = GS == 16 ? CHUNK_GROUPS / 2 : CHUNK_GROUPS;
    constexpr int A_KSTRIDE = A_GROUPS * GROUP_KPACK + 4;
    constexpr int B_KSTRIDE = CHUNK_GROUPS * GROUP_KPACK + 4;
    constexpr int TM = 16 * MTILES;
    constexpr int TN = 8;
    constexpr int NW = 8;
    constexpr int ROWS_PER_WARP = TN * NFRAGS;
    constexpr int BN = NW * ROWS_PER_WARP;
    constexpr int QBYTES = (GS * BITS + 7) / 8;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * TM;
    const int n0 = blockIdx.x * BN;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned char* smem = smem_raw;
    int (*As)[A_KSTRIDE] = reinterpret_cast<int (*)[A_KSTRIDE]>(smem);
    smem += sizeof(int) * TM * A_KSTRIDE;
    int (*Bs)[ROWS_PER_WARP][B_KSTRIDE] =
        reinterpret_cast<int (*)[ROWS_PER_WARP][B_KSTRIDE]>(smem);
    smem += sizeof(int) * NW * ROWS_PER_WARP * B_KSTRIDE;
    __half2 (*Wdm)[NW * ROWS_PER_WARP] =
        reinterpret_cast<__half2 (*)[NW * ROWS_PER_WARP]>(smem);
    smem += sizeof(__half2) * CHUNK_GROUPS * NW * ROWS_PER_WARP;
    float (*Axs)[TM] = reinterpret_cast<float (*)[TM]>(smem);
    smem += sizeof(float) * CHUNK_GROUPS * TM;
    int32_t (*Axsum)[TM] = reinterpret_cast<int32_t (*)[TM]>(smem);

    float acc[NFRAGS][MTILES][4];
    #pragma unroll
    for (int nf = 0; nf < NFRAGS; ++nf) {
        #pragma unroll
        for (int mt = 0; mt < MTILES; ++mt) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                acc[nf][mt][l] = 0.0f;
            }
        }
    }

    const int nchunks = (ng + CHUNK_GROUPS - 1) / CHUNK_GROUPS;
    const int chunk_begin = (nchunks * (int)blockIdx.z) / (int)gridDim.z;
    const int chunk_end = (nchunks * ((int)blockIdx.z + 1)) / (int)gridDim.z;
    for (int chunk = chunk_begin; chunk < chunk_end; ++chunk) {
        const int gbase = chunk * CHUNK_GROUPS;
        const int eff = min(CHUNK_GROUPS, ng - gbase);

        constexpr int A_VECS = TM * A_GROUPS * GROUP_KPACK / 4;
        for (int idx = tid; idx < A_VECS; idx += NW * 32) {
            const int ml = idx / (A_GROUPS * GROUP_KPACK / 4);
            const int kv = idx - ml * (A_GROUPS * GROUP_KPACK / 4);
            const int gm = m0 + ml;
            int4 value = make_int4(0, 0, 0, 0);
            if (!NEED_CHECK || gm < M) {
                const int32_t* src = qx_mmq +
                    ((size_t)chunk * M_pad + gm) * A_KSTRIDE + kv * 4;
                value = *reinterpret_cast<const int4*>(src);
            }
            *reinterpret_cast<int4*>(&As[ml][kv * 4]) = value;
        }

        constexpr int W_TASKS = ROWS_PER_WARP * CHUNK_GROUPS;
        for (int task = lane; task < W_TASKS; task += 32) {
            const int jl = task / CHUNK_GROUPS;
            const int gl = task - jl * CHUNK_GROUPS;
            const int gn = n0 + warp * ROWS_PER_WARP + jl;
            const int g = gbase + gl;
            const bool valid = (!NEED_CHECK || gn < N) && gl < eff;
            const uint8_t* qg = valid
                ? q_packed + ((size_t)gn * ng + g) * QBYTES
                : nullptr;

            if constexpr (BITS == 2 && GS == 16) {
                #pragma unroll
                for (int quartet = 0; quartet < GROUP_KPACK; ++quartet) {
                    Bs[warp][jl][gl * GROUP_KPACK + quartet] = 0;
                }
                #pragma unroll
                for (int quartet = 0; quartet < GS / 4; ++quartet) {
                    Bs[warp][jl][gl * GROUP_KPACK + (gl & 1) * 4 + quartet] =
                        valid ? unpack_qbits4<2>(qg, quartet * 4) : 0;
                }
            } else if constexpr (BITS == 3) {
                #pragma unroll
                for (int quartet = 0; quartet < GS / 4; ++quartet) {
                    Bs[warp][jl][gl * GROUP_KPACK + quartet] =
                        valid ? unpack_qbits4<BITS>(qg, quartet * 4) : 0;
                }
            } else if constexpr (BITS == 4) {
                #pragma unroll
                for (int word = 0; word < 3; ++word) {
                    const uint32_t qw = valid
                        ? *reinterpret_cast<const uint32_t*>(qg + word * 4)
                        : 0;
                    Bs[warp][jl][gl * GROUP_KPACK + word * 2 + 0] = unpack_int4x4_u16(qw);
                    Bs[warp][jl][gl * GROUP_KPACK + word * 2 + 1] = unpack_int4x4_u16(qw >> 16);
                }
            } else {
                #pragma unroll
                for (int quartet = 0; quartet < GS / 4; ++quartet) {
                    Bs[warp][jl][gl * GROUP_KPACK + quartet] =
                        valid ? unpack_int6x4(qg + quartet * 3) : 0;
                }
            }
            if constexpr (!(BITS == 2 && GS == 16)) {
                #pragma unroll
                for (int quartet = GS / 4; quartet < GROUP_KPACK; ++quartet) {
                    Bs[warp][jl][gl * GROUP_KPACK + quartet] = 0;
                }
            }

            float de = 0.0f;
            float me = 0.0f;
            if (valid) {
                const size_t meta = (size_t)gn * ng + g;
                de = neuron_scale[gn] * (float)sub_scale[meta];
                me = neuron_min[gn] * (float)sub_min[meta];
            }
            Wdm[gl][warp * ROWS_PER_WARP + jl] = __floats2half2_rn(de, me);
        }

        for (int idx = tid; idx < TM * CHUNK_GROUPS; idx += NW * 32) {
            const int ml = idx / CHUNK_GROUPS;
            const int gl = idx - ml * CHUNK_GROUPS;
            const int gm = m0 + ml;
            const bool valid = (!NEED_CHECK || gm < M) && gl < eff;
            Axs[gl][ml] = valid ? xscale[(size_t)gm * ng + gbase + gl] : 0.0f;
            Axsum[gl][ml] = valid ? xsum[(size_t)gm * ng + gbase + gl] : 0;
        }
        __syncthreads();

        #pragma unroll
        for (int gl = 0; gl < CHUNK_GROUPS; ++gl) {
            constexpr int A_PAIR_DIVISOR = GS == 16 ? 2 : 1;
            const int ag = gl / A_PAIR_DIVISOR;
            float activation_scale[MTILES][2];
            int32_t activation_sum[MTILES][2];
            #pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                #pragma unroll
                for (int row = 0; row < 2; ++row) {
                    const int ml = mt * 16 + row * 8 + lane / 4;
                    activation_scale[mt][row] = Axs[gl][ml];
                    activation_sum[mt][row] = Axsum[gl][ml];
                }
            }

            int ar[MTILES][4];
            #pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                load_mma_a_m16n8k32(
                    ar[mt], &As[mt * 16 + lane % 16][ag * GROUP_KPACK + (lane / 16) * 4]);
            }

            float2 weight_dm[NFRAGS][2];
            #pragma unroll
            for (int nf = 0; nf < NFRAGS; ++nf) {
                #pragma unroll
                for (int column = 0; column < 2; ++column) {
                    const int jl = nf * TN + (lane % 4) * 2 + column;
                    weight_dm[nf][column] = __half22float2(
                        Wdm[gl][warp * ROWS_PER_WARP + jl]);
                }
            }

            #pragma unroll
            for (int nf = 0; nf < NFRAGS; ++nf) {
                int br[2];
                load_mma_b_m16n8k32(
                    br, &Bs[warp][nf * TN + lane % 8]
                                  [gl * GROUP_KPACK + (((lane / 8) * 4) & 7)]);
                #pragma unroll
                for (int mt = 0; mt < MTILES; ++mt) {
                    int cd[4] = {0, 0, 0, 0};
                    mma_m16n8k32_s8(cd, ar[mt], br);
                    #pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const int ml = mt * 16 + mma168_i(l);
                        const int jl = nf * TN + mma168_j(l);
                        const int gn = n0 + warp * ROWS_PER_WARP + jl;
                        const bool valid_mn = !NEED_CHECK || (m0 + ml < M && gn < N);
                        if (valid_mn && gl < eff) {
                            const int row = l / 2;
                            const int column = l % 2;
                            const float xs = activation_scale[mt][row];
                            const float2 dm = weight_dm[nf][column];
                            acc[nf][mt][l] = fmaf(
                                xs * dm.x, (float)cd[l], acc[nf][mt][l]);
                            acc[nf][mt][l] = fmaf(
                                -xs * dm.y,
                                (float)activation_sum[mt][row],
                                acc[nf][mt][l]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nf = 0; nf < NFRAGS; ++nf) {
        #pragma unroll
        for (int mt = 0; mt < MTILES; ++mt) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int gm = m0 + mt * 16 + mma168_i(l);
                const int gn = n0 + warp * ROWS_PER_WARP + nf * TN + mma168_j(l);
                if (!NEED_CHECK || (gm < M && gn < N)) {
                    if (gridDim.z > 1) {
                        partial[((size_t)blockIdx.z * M + gm) * N + gn] = acc[nf][mt][l];
                    } else {
                        out[(size_t)gm * N + gn] = __float2half(acc[nf][mt][l]);
                    }
                }
            }
        }
    }
}

template <int GS, int MTILES, int NFRAGS>
static constexpr size_t group32_mmq_shared_bytes()
{
    constexpr int chunk_groups = 8;
    constexpr int group_kpack = 8;
    constexpr int a_groups = GS == 16 ? chunk_groups / 2 : chunk_groups;
    constexpr int a_kstride = a_groups * group_kpack + 4;
    constexpr int b_kstride = chunk_groups * group_kpack + 4;
    constexpr int tm = 16 * MTILES;
    constexpr int nw = 8;
    constexpr int rows_per_warp = 8 * NFRAGS;
    return sizeof(int) * tm * a_kstride +
        sizeof(int) * nw * rows_per_warp * b_kstride +
        sizeof(__half2) * chunk_groups * nw * rows_per_warp +
        sizeof(float) * chunk_groups * tm +
        sizeof(int32_t) * chunk_groups * tm;
}

template <bool NEED_CHECK>
static void configure_nint6_group32_128x128()
{
    constexpr int smem_bytes =
        (int)group32_mmq_shared_bytes<24, 8, 2>();
    static std::once_flag configured;
    std::call_once(configured, [=]() {
        MFQ_RUNTIME_CHECK(
            cudaFuncSetAttribute(
                mmq_mma_group32_kernel<6, 24, 8, 2, NEED_CHECK>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes) == cudaSuccess,
            "failed to configure NINT6 128x128 MMQ shared memory");
    });
}

__global__ void reduce_mma24_splitk_kernel(
    const float* __restrict__ partial, __half* __restrict__ out,
    int split_k, int total)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (s < split_k) {
                sum += partial[(size_t)s * total + idx];
            }
        }
        out[idx] = __float2half(sum);
    }
}

mfq_tensor_backend::Tensor nint_mmq_gs24_group32_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor qx_mmq, mfq_tensor_backend::Tensor xscale, mfq_tensor_backend::Tensor xsum,
    int64_t split_k, mfq_tensor_backend::Tensor partial)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kHalf && x.is_contiguous(),
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(qx_mmq.is_cuda() && qx_mmq.scalar_type() == mfq_tensor_backend::kInt32 && qx_mmq.is_contiguous(),
                "qx_mmq must be cuda contiguous int32");
    MFQ_RUNTIME_CHECK(split_k == 1 || split_k == 2,
                "nint_mmq_gs24_group32_ws split_k must be 1 or 2");

    const int N = (int)q_packed.size(0);
    const int ng = (int)q_packed.size(1);
    const int M = (int)x.size(0);
    const int K_real = (int)x.size(1);
    const int M_pad = ((M + 15) / 16) * 16;
    const int nchunks = (ng + 7) / 8;
    const int qbytes = (int)q_packed.size(2);
    const bool bits2 = qbytes == 4;
    const bool bits3 = qbytes == 9;
    const bool bits6 = qbytes == 18;
    const int gs = bits2 ? 16 : 24;
    const int kstride = bits2 ? 36 : 68;
    MFQ_RUNTIME_CHECK(q_packed.dim() == 3 &&
                    (bits2 || bits3 || qbytes == 12 || bits6),
                "q_packed must contain NINT2 gs16 or NINT3/NINT4/NINT6 gs24 groups");
    MFQ_RUNTIME_CHECK(M >= 9, "nint_mmq_gs24_group32_ws requires M>=9");
    MFQ_RUNTIME_CHECK(K_real <= ng * gs, "x K exceeds packed weight K");
    MFQ_RUNTIME_CHECK(qx_mmq.numel() >= (int64_t)nchunks * M_pad * kstride,
                "qx_mmq workspace too small");
    MFQ_RUNTIME_CHECK((int)xscale.size(0) >= M && (int)xscale.size(1) >= ng, "xscale workspace too small");
    MFQ_RUNTIME_CHECK((int)xsum.size(0) >= M && (int)xsum.size(1) >= ng, "xsum workspace too small");
    if (split_k > 1) {
        MFQ_RUNTIME_CHECK(partial.is_cuda() && partial.scalar_type() == mfq_tensor_backend::kFloat32 && partial.is_contiguous(),
                    "partial must be cuda contiguous float32 for split-K");
        MFQ_RUNTIME_CHECK(partial.numel() >= split_k * (int64_t)M * N, "partial workspace too small");
    }

    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    if (bits2) {
        quantize_x_gs16_pair32_layout_kernel<<<
            dim3(M, (ng + 63) / 64), dim3(32, 8), 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
            qx_mmq.data_ptr<int32_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
            M, K_real, ng, M_pad);
    } else {
        quantize_x_group32_layout_kernel<24><<<
            dim3(M, (ng + 31) / 32), dim3(32, 8), 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),
            qx_mmq.data_ptr<int32_t>(), xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),
            M, K_real, ng, M_pad);
    }

    float* partial_ptr = split_k > 1 ? partial.data_ptr<float>() : nullptr;
#define GROUP32_MMQ24_LAUNCH_EX(BITSVAL, GSVAL, MTILESVAL, NFRAGSVAL, CHECKVAL)        \
    mmq_mma_group32_kernel<BITSVAL, GSVAL, MTILESVAL, NFRAGSVAL, CHECKVAL><<<           \
        dim3((N + 64 * NFRAGSVAL - 1) / (64 * NFRAGSVAL),                              \
             (M + 16 * MTILESVAL - 1) / (16 * MTILESVAL),                              \
             (unsigned)split_k), dim3(32, 8),                                          \
        group32_mmq_shared_bytes<GSVAL, MTILESVAL, NFRAGSVAL>(), stream>>>(             \
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),                   \
        sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                   \
        neuron_min.data_ptr<float>(), qx_mmq.data_ptr<int32_t>(),                      \
        xscale.data_ptr<float>(), xsum.data_ptr<int32_t>(),                            \
        reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), partial_ptr,              \
        M, N, ng, M_pad)

#define GROUP32_MMQ24_LAUNCH(BITSVAL, GSVAL, MTILESVAL, CHECKVAL)                      \
    GROUP32_MMQ24_LAUNCH_EX(BITSVAL, GSVAL, MTILESVAL, 1, CHECKVAL)

#define GROUP32_MMQ24_NINT6_128_LAUNCH(CHECKVAL)                                       \
    do {                                                                                \
        configure_nint6_group32_128x128<CHECKVAL>();                                   \
        GROUP32_MMQ24_LAUNCH_EX(6, 24, 8, 2, CHECKVAL);                                \
    } while (0)

#define GROUP32_MMQ24_DISPATCH(BITSVAL, GSVAL)                                          \
    do {                                                                                \
        if (M <= 16) {                                                                  \
            if (M == 16 && N % 64 == 0) GROUP32_MMQ24_LAUNCH(BITSVAL, GSVAL, 1, false); \
            else GROUP32_MMQ24_LAUNCH(BITSVAL, GSVAL, 1, true);                        \
        } else if (M <= 32) {                                                           \
            if (M == 32 && N % 64 == 0) GROUP32_MMQ24_LAUNCH(BITSVAL, GSVAL, 2, false); \
            else GROUP32_MMQ24_LAUNCH(BITSVAL, GSVAL, 2, true);                        \
        } else {                                                                        \
            if (M % 64 == 0 && N % 64 == 0) GROUP32_MMQ24_LAUNCH(BITSVAL, GSVAL, 4, false); \
            else GROUP32_MMQ24_LAUNCH(BITSVAL, GSVAL, 4, true);                        \
        }                                                                               \
    } while (0)

#define GROUP32_MMQ24_NINT6_DISPATCH()                                                  \
    do {                                                                                \
        if (M >= 128) {                                                                 \
            if (M % 128 == 0 && N % 128 == 0) GROUP32_MMQ24_NINT6_128_LAUNCH(false);   \
            else GROUP32_MMQ24_NINT6_128_LAUNCH(true);                                 \
        } else {                                                                        \
            GROUP32_MMQ24_DISPATCH(6, 24);                                              \
        }                                                                               \
    } while (0)
    if (bits2) GROUP32_MMQ24_DISPATCH(2, 16);
    else if (bits3) GROUP32_MMQ24_DISPATCH(3, 24);
    else if (bits6) GROUP32_MMQ24_NINT6_DISPATCH();
    else GROUP32_MMQ24_DISPATCH(4, 24);
#undef GROUP32_MMQ24_NINT6_DISPATCH
#undef GROUP32_MMQ24_NINT6_128_LAUNCH
#undef GROUP32_MMQ24_DISPATCH
#undef GROUP32_MMQ24_LAUNCH
#undef GROUP32_MMQ24_LAUNCH_EX

    if (split_k > 1) {
        const int total = M * N;
        constexpr int block = 256;
        const int grid = std::min((total + block - 1) / block, 65535);
        reduce_mma24_splitk_kernel<<<grid, block, 0, stream>>>(
            partial_ptr, reinterpret_cast<__half*>(out.data_ptr<mfq_half>()),
            (int)split_k, total);
    }
    return out;
}
// Exact packed-NINT MMQ. Four quantization groups form one K=(4*GS)
// chunk. A block dequantizes one 64-row weight tile directly to fp16 shared
// memory and reuses it for 16..128 activation rows before the next chunk.
// This avoids both the full fp16 weight tensor and the activation quantization
// required by the int8-MMA experiment above.
template <
    int BITS,
    int GS,
    int MTILES,
    bool WRITE_PARTIAL,
    int GROUPS_PER_CHUNK_VALUE = 4>
__global__ void __launch_bounds__(256) mmq_f16_packed_kernel(
    const uint8_t* __restrict__ q_packed,    // [N, ng, ceil(GS*BITS/8)]
    const uint8_t* __restrict__ sub_scale,   // [N, ng]
    const uint8_t* __restrict__ sub_min,     // [N, ng]
    const float* __restrict__ neuron_scale,  // [N]
    const float* __restrict__ neuron_min,    // [N]
    const __half* __restrict__ x,            // [M, K_real]
    __half* __restrict__ out,                // [M, N]
    float* __restrict__ partial,              // [split_k, M, N]
    int M, int N, int ng, int K_real)
{
    constexpr int GROUPS_PER_CHUNK = GROUPS_PER_CHUNK_VALUE;
    constexpr int QBYTES = (GS * BITS + 7) / 8;
    constexpr int BK = GS * GROUPS_PER_CHUNK;
    constexpr int BK_STRIDE = BK + 8;
    constexpr int BM = 16 * MTILES;
    constexpr int BN = 64;
    constexpr int NW = 8;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int N_FRAGS = BN / WMMA_N;
    constexpr int ACCS_PER_WARP = (MTILES + 1) / 2;

    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int tid = warp * 32 + lane;
    int m0 = blockIdx.y * BM;
    int n0 = blockIdx.x * BN;
    int warp_m0 = warp / N_FRAGS;
    int warp_n = warp % N_FRAGS;

    __shared__ __half W_s[BN][BK_STRIDE];
    __shared__ __half X_s[BM][BK_STRIDE];
    __shared__ float C_s[N_FRAGS][WMMA_M][WMMA_N];

    using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                 __half, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                 __half, wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
    FragC acc[ACCS_PER_WARP];
    #pragma unroll
    for (int a = 0; a < ACCS_PER_WARP; ++a) {
        wmma::fill_fragment(acc[a], 0.0f);
    }

    int nchunks = (ng + GROUPS_PER_CHUNK - 1) / GROUPS_PER_CHUNK;
    int chunk_begin = WRITE_PARTIAL ? (nchunks * (int)blockIdx.z) / (int)gridDim.z : 0;
    int chunk_end = WRITE_PARTIAL ? (nchunks * ((int)blockIdx.z + 1)) / (int)gridDim.z : nchunks;
    for (int chunk = chunk_begin; chunk < chunk_end; ++chunk) {
        int gbase = chunk * GROUPS_PER_CHUNK;
        int kb = gbase * GS;

        // Exactly 256 row/group tasks: every thread reads one pair of scales
        // and expands one complete packed group.
        if (tid < BN * GROUPS_PER_CHUNK) {
            int nn = tid / GROUPS_PER_CHUNK;
            int gl = tid % GROUPS_PER_CHUNK;
            int gn = n0 + nn;
            int g = gbase + gl;
            bool valid_w = gn < N && g < ng;
            float de = 0.0f;
            float me = 0.0f;
            const uint8_t* qg = nullptr;
            if (valid_w) {
                size_t meta = (size_t)gn * ng + g;
                de = neuron_scale[gn] * (float)sub_scale[meta];
                me = neuron_min[gn] * (float)sub_min[meta];
                qg = q_packed + meta * QBYTES;
            }
            if constexpr (BITS == 3) {
            #pragma unroll
            for (int chunk8 = 0; chunk8 < GS / 8; ++chunk8) {
                const int byte = chunk8 * 3;
                const uint8_t b0 = valid_w ? qg[byte] : 0;
                const uint8_t b1 = valid_w ? qg[byte + 1] : 0;
                const uint8_t b2 = valid_w ? qg[byte + 2] : 0;
                const uint8_t values[8] = {
                    (uint8_t)(b0 & 7),
                    (uint8_t)((b0 >> 3) & 7),
                    (uint8_t)(((b0 >> 6) | (b1 << 2)) & 7),
                    (uint8_t)((b1 >> 1) & 7),
                    (uint8_t)((b1 >> 4) & 7),
                    (uint8_t)(((b1 >> 7) | (b2 << 1)) & 7),
                    (uint8_t)((b2 >> 2) & 7),
                    (uint8_t)((b2 >> 5) & 7),
                };
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const __half lo = valid_w
                        ? __float2half(de * (float)values[pair * 2] - me)
                        : __float2half(0.0f);
                    const __half hi = valid_w
                        ? __float2half(de * (float)values[pair * 2 + 1] - me)
                        : __float2half(0.0f);
                    *reinterpret_cast<__half2*>(
                        &W_s[nn][gl * GS + chunk8 * 8 + pair * 2]) =
                        __halves2half2(lo, hi);
                }
            }
            } else if constexpr (BITS == 4) {
            #pragma unroll
            for (int word = 0; word < QBYTES / 4; ++word) {
                uint32_t qw = valid_w
                    ? *reinterpret_cast<const uint32_t*>(qg + word * 4)
                    : 0;
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    uint8_t qv = (uint8_t)(qw >> (byte * 8));
                    __half lo = valid_w ? __float2half(de * (float)(qv & 0x0f) - me)
                                        : __float2half(0.0f);
                    __half hi = valid_w ? __float2half(de * (float)(qv >> 4) - me)
                                        : __float2half(0.0f);
                    int b = word * 4 + byte;
                    *reinterpret_cast<__half2*>(&W_s[nn][gl * GS + b * 2]) =
                        __halves2half2(lo, hi);
                }
            }
            } else if constexpr (BITS == 6) {
            #pragma unroll
            for (int quartet = 0; quartet < GS / 4; ++quartet) {
                int q4 = valid_w ? unpack_int6x4(qg + quartet * 3) : 0;
                #pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    int shift = pair * 16;
                    uint8_t q0 = (uint8_t)(q4 >> shift);
                    uint8_t q1 = (uint8_t)(q4 >> (shift + 8));
                    __half lo = valid_w ? __float2half(de * (float)q0 - me)
                                        : __float2half(0.0f);
                    __half hi = valid_w ? __float2half(de * (float)q1 - me)
                                        : __float2half(0.0f);
                    *reinterpret_cast<__half2*>(&W_s[nn][gl * GS + quartet * 4 + pair * 2]) =
                        __halves2half2(lo, hi);
                }
            }
            } else {
            #pragma unroll
            for (int pair = 0; pair < GS / 2; ++pair) {
                const int index = pair * 2;
                const uint8_t q0 = valid_w ? unpack_qbits_one<BITS>(qg, index) : 0;
                const uint8_t q1 = valid_w ? unpack_qbits_one<BITS>(qg, index + 1) : 0;
                const __half lo = valid_w ? __float2half(de * (float)q0 - me)
                                          : __float2half(0.0f);
                const __half hi = valid_w ? __float2half(de * (float)q1 - me)
                                          : __float2half(0.0f);
                *reinterpret_cast<__half2*>(&W_s[nn][gl * GS + index]) =
                    __halves2half2(lo, hi);
            }
            }
        }

        // Copy the activation tile as half2. K_real can end inside the final
        // quantization group; missing values are zero without materializing pad.
        constexpr int X_PAIRS = BM * (BK / 2);
        for (int idx = tid; idx < X_PAIRS; idx += NW * 32) {
            int mm = idx / (BK / 2);
            int pair = idx - mm * (BK / 2);
            int gm = m0 + mm;
            int k = kb + pair * 2;
            __half2 xv = __float2half2_rn(0.0f);
            if (gm < M && k + 1 < K_real && (K_real & 1) == 0) {
                xv = *reinterpret_cast<const __half2*>(x + (size_t)gm * K_real + k);
            } else if (gm < M && k < K_real) {
                __half x0 = x[(size_t)gm * K_real + k];
                __half x1 = k + 1 < K_real ? x[(size_t)gm * K_real + k + 1]
                                           : __float2half(0.0f);
                xv = __halves2half2(x0, x1);
            }
            *reinterpret_cast<__half2*>(&X_s[mm][pair * 2]) = xv;
        }
        __syncthreads();

        bool warp_active = warp_m0 < MTILES;
        #pragma unroll
        for (int ks = 0; ks < BK; ks += WMMA_K) {
            if (warp_active) {
                FragB bfrag;
                wmma::load_matrix_sync(bfrag, &W_s[warp_n * WMMA_N][ks], BK_STRIDE);
                #pragma unroll
                for (int a = 0; a < ACCS_PER_WARP; ++a) {
                    int mi = warp_m0 + a * 2;
                    if (mi < MTILES) {
                        FragA afrag;
                        wmma::load_matrix_sync(afrag, &X_s[mi * WMMA_M][ks], BK_STRIDE);
                        wmma::mma_sync(acc[a], afrag, bfrag, acc[a]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // The two warp rows reuse four 16x16 float buffers. The extra block-wide
    // barriers happen once per output tile and save 4 KiB of shared memory,
    // enough to raise residency for the M32 and M64 kernels.
    #pragma unroll
    for (int a = 0; a < ACCS_PER_WARP; ++a) {
        #pragma unroll
        for (int warp_row = 0; warp_row < 2; ++warp_row) {
            int mi = warp_m0 + a * 2;
            bool owns = warp_m0 == warp_row && mi < MTILES;
            if (owns) {
                wmma::store_matrix_sync(&C_s[warp_n][0][0], acc[a], WMMA_N,
                                        wmma::mem_row_major);
            }
            __syncthreads();
            if (owns) {
                int gm0 = m0 + mi * WMMA_M;
                int gn0 = n0 + warp_n * WMMA_N;
                #pragma unroll
                for (int e = lane; e < WMMA_M * WMMA_N; e += 32) {
                    int r = e / WMMA_N;
                    int c = e - r * WMMA_N;
                    int gm = gm0 + r;
                    int gn = gn0 + c;
                    if (gm < M && gn < N) {
                        if constexpr (WRITE_PARTIAL) {
                            partial[((size_t)blockIdx.z * M + gm) * N + gn] = C_s[warp_n][r][c];
                        } else {
                            out[(size_t)gm * N + gn] = __float2half(C_s[warp_n][r][c]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

mfq_tensor_backend::Tensor nint_mmq_gs24_f16_nint3_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous float32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous float32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kHalf && x.is_contiguous(),
                "x must be cuda contiguous fp16");
    const int N = (int)q_packed.size(0);
    const int ng = (int)q_packed.size(1);
    const int M = (int)x.size(0);
    const int K_real = (int)x.size(1);
    const int qbytes = q_packed.dim() == 3 ? (int)q_packed.size(2) : 0;
    const bool bits2 = qbytes == 4;
    const bool bits3 = qbytes == 9;
    const int gs = bits2 ? 16 : 24;
    MFQ_RUNTIME_CHECK(bits2 || bits3,
                "f16 packed MMQ requires NINT2 gs16 or NINT3 gs24");
    MFQ_RUNTIME_CHECK(M >= 9, "f16 packed MMQ requires M>=9");
    MFQ_RUNTIME_CHECK(K_real <= ng * gs, "x K exceeds packed weight K");
    MFQ_RUNTIME_CHECK(sub_scale.sizes() == q_packed.sizes().slice(0, 2), "sub_scale shape mismatch");
    MFQ_RUNTIME_CHECK(sub_min.sizes() == q_packed.sizes().slice(0, 2), "sub_min shape mismatch");
    MFQ_RUNTIME_CHECK(neuron_scale.numel() == N && neuron_min.numel() == N, "neuron metadata shape mismatch");

    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

#define F16_MMQ_NINT23_LAUNCH(BITSVAL, GSVAL, MTILESVAL)                                 \
    mmq_f16_packed_kernel<BITSVAL, GSVAL, MTILESVAL, false><<<                           \
        dim3((N + 63) / 64, (M + 16 * MTILESVAL - 1) / (16 * MTILESVAL)),                \
        dim3(32, 8), 0, stream>>>(                                                        \
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),                     \
        sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                     \
        neuron_min.data_ptr<float>(),                                                     \
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                         \
        reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), nullptr, M, N, ng, K_real)

    if (bits2) {
        if (M <= 16) F16_MMQ_NINT23_LAUNCH(2, 16, 1);
        else if (M <= 48) F16_MMQ_NINT23_LAUNCH(2, 16, 2);
        else if (M <= 96) F16_MMQ_NINT23_LAUNCH(2, 16, 4);
        else F16_MMQ_NINT23_LAUNCH(2, 16, 8);
    } else {
        if (M <= 16) F16_MMQ_NINT23_LAUNCH(3, 24, 1);
        else if (M <= 48) F16_MMQ_NINT23_LAUNCH(3, 24, 2);
        else if (M <= 96) F16_MMQ_NINT23_LAUNCH(3, 24, 4);
        else F16_MMQ_NINT23_LAUNCH(3, 24, 8);
    }
#undef F16_MMQ_NINT23_LAUNCH
    return out;
}

mfq_tensor_backend::Tensor nint_mmq_gs24_f16_nint4_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous float32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous float32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kHalf && x.is_contiguous(),
                "x must be cuda contiguous fp16");
    int N = (int)q_packed.size(0);
    int ng = (int)q_packed.size(1);
    int M = (int)x.size(0);
    int K_real = (int)x.size(1);
    MFQ_RUNTIME_CHECK(q_packed.dim() == 3 && (int)q_packed.size(2) == 12,
                "nint_mmq_gs24_f16_nint4 requires packed NINT4 gs24");
    MFQ_RUNTIME_CHECK(M >= 16 && M <= 32, "nint_mmq_gs24_f16_nint4 supports M=16..32");
    MFQ_RUNTIME_CHECK(K_real <= ng * 24, "x K exceeds packed weight K");
    MFQ_RUNTIME_CHECK(sub_scale.sizes() == q_packed.sizes().slice(0, 2), "sub_scale shape mismatch");
    MFQ_RUNTIME_CHECK(sub_min.sizes() == q_packed.sizes().slice(0, 2), "sub_min shape mismatch");
    MFQ_RUNTIME_CHECK(neuron_scale.numel() == N && neuron_min.numel() == N, "neuron metadata shape mismatch");

    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

#define F16_MMQ24_NINT4_LAUNCH(MTILESVAL)                                                 \
    mmq_f16_packed_kernel<4, 24, MTILESVAL, false><<<                                   \
        dim3((N + 63) / 64, (M + 16 * MTILESVAL - 1) / (16 * MTILESVAL)),                \
        dim3(32, 8), 0, stream>>>(                                                        \
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),                    \
        sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                    \
        neuron_min.data_ptr<float>(),                                                    \
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),                        \
        reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), nullptr, M, N, ng, K_real)

    if (M == 16) F16_MMQ24_NINT4_LAUNCH(1);
    else F16_MMQ24_NINT4_LAUNCH(2);
#undef F16_MMQ24_NINT4_LAUNCH
    return out;
}

__global__ void reduce_f16_mmq_split4_kernel(
    const float* __restrict__ partial, __half* __restrict__ out, int total)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x) {
        float sum = partial[idx] + partial[(size_t)total + idx]
                  + partial[(size_t)2 * total + idx] + partial[(size_t)3 * total + idx];
        out[idx] = __float2half(sum);
    }
}

mfq_tensor_backend::Tensor nint_mmq_gs24_f16_nint6_split4_ws_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min, mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor partial)
{
    MFQ_RUNTIME_CHECK(q_packed.is_cuda() && q_packed.scalar_type() == mfq_tensor_backend::kUInt8 && q_packed.is_contiguous(),
                "q_packed must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_scale.is_cuda() && sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 && sub_scale.is_contiguous(),
                "sub_scale must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(sub_min.is_cuda() && sub_min.scalar_type() == mfq_tensor_backend::kUInt8 && sub_min.is_contiguous(),
                "sub_min must be cuda contiguous uint8");
    MFQ_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_scale.is_contiguous(),
                "neuron_scale must be cuda contiguous float32");
    MFQ_RUNTIME_CHECK(neuron_min.is_cuda() && neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 && neuron_min.is_contiguous(),
                "neuron_min must be cuda contiguous float32");
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kHalf && x.is_contiguous(),
                "x must be cuda contiguous fp16");
    MFQ_RUNTIME_CHECK(partial.is_cuda() && partial.scalar_type() == mfq_tensor_backend::kFloat32 && partial.is_contiguous(),
                "partial workspace must be cuda contiguous float32");
    int N = (int)q_packed.size(0);
    int ng = (int)q_packed.size(1);
    int M = (int)x.size(0);
    int K_real = (int)x.size(1);
    MFQ_RUNTIME_CHECK(q_packed.dim() == 3 && (int)q_packed.size(2) == 18,
                "nint_mmq_gs24_f16_nint6_split4 requires packed NINT6 gs24");
    MFQ_RUNTIME_CHECK(M >= 16 && M <= 32, "nint_mmq_gs24_f16_nint6_split4 supports M=16..32");
    MFQ_RUNTIME_CHECK(K_real <= ng * 24, "x K exceeds packed weight K");
    MFQ_RUNTIME_CHECK(sub_scale.sizes() == q_packed.sizes().slice(0, 2), "sub_scale shape mismatch");
    MFQ_RUNTIME_CHECK(sub_min.sizes() == q_packed.sizes().slice(0, 2), "sub_min shape mismatch");
    MFQ_RUNTIME_CHECK(neuron_scale.numel() == N && neuron_min.numel() == N, "neuron metadata shape mismatch");
    MFQ_RUNTIME_CHECK(partial.numel() >= 4LL * M * N, "partial workspace too small");

    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

#define F16_MMQ24_SPLIT_LAUNCH(MTILESVAL)                                                 \
    mmq_f16_packed_kernel<6, 24, MTILESVAL, true><<<                                    \
        dim3((N + 63) / 64, (M + 16 * MTILESVAL - 1) / (16 * MTILESVAL), 4),             \
        dim3(32, 8), 0, stream>>>(                                                        \
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),                     \
        sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),                     \
        neuron_min.data_ptr<float>(),                                                     \
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()), nullptr,                \
        partial.data_ptr<float>(), M, N, ng, K_real)

    if (M <= 16) F16_MMQ24_SPLIT_LAUNCH(1);
    else F16_MMQ24_SPLIT_LAUNCH(2);
#undef F16_MMQ24_SPLIT_LAUNCH

    int total = M * N;
    int block = 256;
    int grid = std::min((total + block - 1) / block, 65535);
    reduce_f16_mmq_split4_kernel<<<grid, block, 0, stream>>>(
        partial.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), total);
    return out;
}

mfq_tensor_backend::Tensor nint_mmq_f16_packed_cuda(
    mfq_tensor_backend::Tensor q_packed,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor x,
    int64_t gs,
    int64_t bits)
{
    MFQ_RUNTIME_CHECK(
        q_packed.is_cuda() &&
        q_packed.scalar_type() == mfq_tensor_backend::kUInt8 &&
        q_packed.is_contiguous() && q_packed.dim() == 3,
        "common FP16 packed MMQ q_packed must be CUDA contiguous uint8 rank-3");
    MFQ_RUNTIME_CHECK(
        sub_scale.is_cuda() &&
        sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
        sub_scale.is_contiguous(),
        "common FP16 packed MMQ sub_scale must be CUDA contiguous uint8");
    MFQ_RUNTIME_CHECK(
        sub_min.is_cuda() &&
        sub_min.scalar_type() == mfq_tensor_backend::kUInt8 &&
        sub_min.is_contiguous(),
        "common FP16 packed MMQ sub_min must be CUDA contiguous uint8");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() &&
        neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_scale.is_contiguous(),
        "common FP16 packed MMQ neuron_scale must be CUDA contiguous float32");
    MFQ_RUNTIME_CHECK(
        neuron_min.is_cuda() &&
        neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_min.is_contiguous(),
        "common FP16 packed MMQ neuron_min must be CUDA contiguous float32");
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
        x.is_contiguous() && x.dim() == 2,
        "common FP16 packed MMQ x must be CUDA contiguous fp16 rank-2");
    const int N = (int)q_packed.size(0);
    const int ng = (int)q_packed.size(1);
    const int M = (int)x.size(0);
    const int K_real = (int)x.size(1);
    MFQ_RUNTIME_CHECK(M >= 16, "common FP16 packed MMQ requires M >= 16");
    MFQ_RUNTIME_CHECK(
        bits == 2 || bits == 3 || bits == 4 ||
        bits == 5 || bits == 6 || bits == 8,
        "common FP16 packed MMQ unsupported NINT bits");
    MFQ_RUNTIME_CHECK(
        (bits == 2 && gs == 16) ||
        ((bits == 3 || bits == 4 || bits == 6) && gs == 24) ||
        (bits == 5 && gs == 28) ||
        (bits == 8 && (gs == 24 || gs == 48)),
        "common FP16 packed MMQ unsupported NINT profile");
    MFQ_RUNTIME_CHECK(
        q_packed.size(2) == (gs * bits + 7) / 8,
        "common FP16 packed MMQ q_packed width mismatch");
    MFQ_RUNTIME_CHECK(
        sub_scale.sizes() == q_packed.sizes().slice(0, 2) &&
        sub_min.sizes() == q_packed.sizes().slice(0, 2),
        "common FP16 packed MMQ group metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        neuron_scale.numel() == N && neuron_min.numel() == N,
        "common FP16 packed MMQ neuron metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        K_real <= ng * gs,
        "common FP16 packed MMQ activation width exceeds packed weights");

    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();

#define COMMON_F16_MMQ_LAUNCH(BITS_VALUE, GS_VALUE, GPC_VALUE, MTILES_VALUE) \
    mmq_f16_packed_kernel<                                                    \
        BITS_VALUE, GS_VALUE, MTILES_VALUE, false, GPC_VALUE><<<              \
        dim3((N + 63) / 64,                                                   \
             (M + 16 * MTILES_VALUE - 1) / (16 * MTILES_VALUE)),             \
        dim3(32, 8), 0, stream>>>(                                            \
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),          \
        sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),          \
        neuron_min.data_ptr<float>(),                                         \
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),              \
        reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), nullptr,         \
        M, N, ng, K_real)

#define COMMON_F16_MMQ_DISPATCH(BITS_VALUE, GS_VALUE, GPC_VALUE)       \
    do {                                                               \
        if (M <= 16) {                                                  \
            COMMON_F16_MMQ_LAUNCH(BITS_VALUE, GS_VALUE, GPC_VALUE, 1); \
        } else if (M <= 32) {                                           \
            COMMON_F16_MMQ_LAUNCH(BITS_VALUE, GS_VALUE, GPC_VALUE, 2); \
        } else if (M <= 64) {                                           \
            COMMON_F16_MMQ_LAUNCH(BITS_VALUE, GS_VALUE, GPC_VALUE, 4); \
        } else {                                                        \
            COMMON_F16_MMQ_LAUNCH(BITS_VALUE, GS_VALUE, GPC_VALUE, 8); \
        }                                                               \
    } while (0)

    if (bits == 2) {
        COMMON_F16_MMQ_DISPATCH(2, 16, 4);
    } else if (bits == 3) {
        COMMON_F16_MMQ_DISPATCH(3, 24, 4);
    } else if (bits == 4) {
        COMMON_F16_MMQ_DISPATCH(4, 24, 4);
    } else if (bits == 5) {
        if (M <= 16) {
            COMMON_F16_MMQ_LAUNCH(5, 28, 4, 1);
        } else if (M <= 32) {
            COMMON_F16_MMQ_LAUNCH(5, 28, 4, 2);
        } else {
            COMMON_F16_MMQ_LAUNCH(5, 28, 4, 4);
        }
    } else if (bits == 6) {
        COMMON_F16_MMQ_DISPATCH(6, 24, 4);
    } else if (gs == 24) {
        COMMON_F16_MMQ_DISPATCH(8, 24, 4);
    } else {
        COMMON_F16_MMQ_DISPATCH(8, 48, 2);
    }
#undef COMMON_F16_MMQ_DISPATCH
#undef COMMON_F16_MMQ_LAUNCH

    MFQ_RUNTIME_CHECK(
        cudaGetLastError() == cudaSuccess,
        "common FP16 packed MMQ kernel launch failed");
    return out;
}

mfq_tensor_backend::Tensor nint_mmq_f32_packed_cuda(
    mfq_tensor_backend::Tensor q_packed,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor x,
    int64_t gs,
    int64_t bits)
{
    MFQ_RUNTIME_CHECK(
        q_packed.is_cuda() &&
        q_packed.scalar_type() == mfq_tensor_backend::kUInt8 &&
        q_packed.is_contiguous() && q_packed.dim() == 3,
        "common FP32-output packed MMQ q_packed must be CUDA contiguous uint8 rank-3");
    MFQ_RUNTIME_CHECK(
        sub_scale.is_cuda() &&
        sub_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
        sub_scale.is_contiguous(),
        "common FP32-output packed MMQ sub_scale must be CUDA contiguous uint8");
    MFQ_RUNTIME_CHECK(
        sub_min.is_cuda() &&
        sub_min.scalar_type() == mfq_tensor_backend::kUInt8 &&
        sub_min.is_contiguous(),
        "common FP32-output packed MMQ sub_min must be CUDA contiguous uint8");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() &&
        neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_scale.is_contiguous(),
        "common FP32-output packed MMQ neuron_scale must be CUDA contiguous float32");
    MFQ_RUNTIME_CHECK(
        neuron_min.is_cuda() &&
        neuron_min.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_min.is_contiguous(),
        "common FP32-output packed MMQ neuron_min must be CUDA contiguous float32");
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
        x.is_contiguous() && x.dim() == 2,
        "common FP32-output packed MMQ x must be CUDA contiguous fp16 rank-2");
    const int N = (int)q_packed.size(0);
    const int ng = (int)q_packed.size(1);
    const int M = (int)x.size(0);
    const int K_real = (int)x.size(1);
    MFQ_RUNTIME_CHECK(M >= 16, "common FP32-output packed MMQ requires M >= 16");
    MFQ_RUNTIME_CHECK(
        bits == 2 || bits == 3 || bits == 4 ||
        bits == 5 || bits == 6 || bits == 8,
        "common FP32-output packed MMQ unsupported NINT bits");
    MFQ_RUNTIME_CHECK(
        (bits == 2 && gs == 16) ||
        ((bits == 3 || bits == 4 || bits == 6) && gs == 24) ||
        (bits == 5 && gs == 28) ||
        (bits == 8 && (gs == 24 || gs == 48)),
        "common FP32-output packed MMQ unsupported NINT profile");
    MFQ_RUNTIME_CHECK(
        q_packed.size(2) == (gs * bits + 7) / 8,
        "common FP32-output packed MMQ q_packed width mismatch");
    MFQ_RUNTIME_CHECK(
        sub_scale.sizes() == q_packed.sizes().slice(0, 2) &&
        sub_min.sizes() == q_packed.sizes().slice(0, 2),
        "common FP32-output packed MMQ group metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        neuron_scale.numel() == N && neuron_min.numel() == N,
        "common FP32-output packed MMQ neuron metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        K_real <= ng * gs,
        "common FP32-output packed MMQ activation width exceeds packed weights");

    auto out = mfq_tensor_backend::empty(
        {M, N}, x.options().dtype(mfq_tensor_backend::kFloat32));
    cudaStream_t stream = mfq_current_cuda_stream();

#define COMMON_F32_MMQ_LAUNCH(BITS_VALUE, GS_VALUE, GPC_VALUE, MTILES_VALUE) \
    mmq_f16_packed_kernel<                                                    \
        BITS_VALUE, GS_VALUE, MTILES_VALUE, true, GPC_VALUE><<<               \
        dim3((N + 63) / 64,                                                   \
             (M + 16 * MTILES_VALUE - 1) / (16 * MTILES_VALUE), 1),          \
        dim3(32, 8), 0, stream>>>(                                            \
        q_packed.data_ptr<uint8_t>(), sub_scale.data_ptr<uint8_t>(),          \
        sub_min.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),          \
        neuron_min.data_ptr<float>(),                                         \
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),              \
        nullptr, out.data_ptr<float>(), M, N, ng, K_real)

#define COMMON_F32_MMQ_DISPATCH(BITS_VALUE, GS_VALUE, GPC_VALUE)       \
    do {                                                               \
        if (M <= 16) {                                                  \
            COMMON_F32_MMQ_LAUNCH(BITS_VALUE, GS_VALUE, GPC_VALUE, 1); \
        } else if (M <= 32) {                                           \
            COMMON_F32_MMQ_LAUNCH(BITS_VALUE, GS_VALUE, GPC_VALUE, 2); \
        } else if (M <= 64) {                                           \
            COMMON_F32_MMQ_LAUNCH(BITS_VALUE, GS_VALUE, GPC_VALUE, 4); \
        } else {                                                        \
            COMMON_F32_MMQ_LAUNCH(BITS_VALUE, GS_VALUE, GPC_VALUE, 8); \
        }                                                               \
    } while (0)

    if (bits == 2) {
        COMMON_F32_MMQ_DISPATCH(2, 16, 4);
    } else if (bits == 3) {
        COMMON_F32_MMQ_DISPATCH(3, 24, 4);
    } else if (bits == 4) {
        COMMON_F32_MMQ_DISPATCH(4, 24, 4);
    } else if (bits == 5) {
        if (M <= 16) {
            COMMON_F32_MMQ_LAUNCH(5, 28, 4, 1);
        } else if (M <= 32) {
            COMMON_F32_MMQ_LAUNCH(5, 28, 4, 2);
        } else {
            COMMON_F32_MMQ_LAUNCH(5, 28, 4, 4);
        }
    } else if (bits == 6) {
        COMMON_F32_MMQ_DISPATCH(6, 24, 4);
    } else if (gs == 24) {
        COMMON_F32_MMQ_DISPATCH(8, 24, 4);
    } else {
        COMMON_F32_MMQ_DISPATCH(8, 48, 2);
    }
#undef COMMON_F32_MMQ_DISPATCH
#undef COMMON_F32_MMQ_LAUNCH

    MFQ_RUNTIME_CHECK(
        cudaGetLastError() == cudaSuccess,
        "common FP32-output packed MMQ kernel launch failed");
    return out;
}

template <int MTILES, bool WRITE_F32>
__global__ void __launch_bounds__(256) nint8_zero_mmq_f16_packed_kernel(
    const uint8_t* __restrict__ q,
    const __half* __restrict__ scale,
    const __half* __restrict__ x,
    __half* __restrict__ out,
    float* __restrict__ out_f32,
    int M,
    int N,
    int groups,
    int K_real)
{
    constexpr int GROUPS_PER_CHUNK = 3;
    constexpr int GS = 32;
    constexpr int BK = GROUPS_PER_CHUNK * GS;
    constexpr int BK_STRIDE = BK + 8;
    constexpr int BM = 16 * MTILES;
    constexpr int BN = 64;
    constexpr int N_FRAGS = BN / 16;
    constexpr int ACCS_PER_WARP = (MTILES + 1) / 2;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int warp_m0 = warp / N_FRAGS;
    const int warp_n = warp % N_FRAGS;

    __shared__ __half W_s[BN][BK_STRIDE];
    __shared__ __half X_s[BM][BK_STRIDE];
    __shared__ float C_s[N_FRAGS][16][16];

    using FragA = wmma::fragment<
        wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
    using FragB = wmma::fragment<
        wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;
    using FragC = wmma::fragment<
        wmma::accumulator, 16, 16, 16, float>;
    FragC acc[ACCS_PER_WARP];
#pragma unroll
    for (int index = 0; index < ACCS_PER_WARP; ++index) {
        wmma::fill_fragment(acc[index], 0.0f);
    }

    const int chunks =
        (groups + GROUPS_PER_CHUNK - 1) / GROUPS_PER_CHUNK;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int gbase = chunk * GROUPS_PER_CHUNK;
        const int kb = gbase * GS;
        if (tid < BN * GROUPS_PER_CHUNK) {
            const int nn = tid / GROUPS_PER_CHUNK;
            const int gl = tid - nn * GROUPS_PER_CHUNK;
            const int gn = n0 + nn;
            const int group = gbase + gl;
            const bool valid = gn < N && group < groups;
            const size_t meta =
                (size_t)gn * groups + group;
            const int8_t* qg = valid
                ? reinterpret_cast<const int8_t*>(q + meta * GS)
                : nullptr;
            const float d = valid ? __half2float(scale[meta]) : 0.0f;
#pragma unroll
            for (int pair = 0; pair < GS / 2; ++pair) {
                const int index = pair * 2;
                const __half lo = valid
                    ? __float2half_rn(d * (float)qg[index])
                    : __float2half_rn(0.0f);
                const __half hi = valid
                    ? __float2half_rn(d * (float)qg[index + 1])
                    : __float2half_rn(0.0f);
                *reinterpret_cast<__half2*>(
                    &W_s[nn][gl * GS + index]) =
                    __halves2half2(lo, hi);
            }
        }

        constexpr int X_PAIRS = BM * (BK / 2);
        for (int index = tid; index < X_PAIRS; index += 256) {
            const int mm = index / (BK / 2);
            const int pair = index - mm * (BK / 2);
            const int gm = m0 + mm;
            const int k = kb + pair * 2;
            __half2 value = __float2half2_rn(0.0f);
            if (gm < M && k + 1 < K_real && (K_real & 1) == 0) {
                value = *reinterpret_cast<const __half2*>(
                    x + (size_t)gm * K_real + k);
            } else if (gm < M && k < K_real) {
                const __half lo = x[(size_t)gm * K_real + k];
                const __half hi = k + 1 < K_real
                    ? x[(size_t)gm * K_real + k + 1]
                    : __float2half_rn(0.0f);
                value = __halves2half2(lo, hi);
            }
            *reinterpret_cast<__half2*>(
                &X_s[mm][pair * 2]) = value;
        }
        __syncthreads();

        const bool warp_active = warp_m0 < MTILES;
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            if (warp_active) {
                FragB bfrag;
                wmma::load_matrix_sync(
                    bfrag, &W_s[warp_n * 16][ks], BK_STRIDE);
#pragma unroll
                for (int a = 0; a < ACCS_PER_WARP; ++a) {
                    const int mi = warp_m0 + a * 2;
                    if (mi < MTILES) {
                        FragA afrag;
                        wmma::load_matrix_sync(
                            afrag, &X_s[mi * 16][ks], BK_STRIDE);
                        wmma::mma_sync(
                            acc[a], afrag, bfrag, acc[a]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < ACCS_PER_WARP; ++a) {
#pragma unroll
        for (int warp_row = 0; warp_row < 2; ++warp_row) {
            const int mi = warp_m0 + a * 2;
            const bool owns = warp_m0 == warp_row && mi < MTILES;
            if (owns) {
                wmma::store_matrix_sync(
                    &C_s[warp_n][0][0], acc[a], 16,
                    wmma::mem_row_major);
            }
            __syncthreads();
            if (owns) {
                const int gm0 = m0 + mi * 16;
                const int gn0 = n0 + warp_n * 16;
                for (int element = lane;
                     element < 16 * 16;
                     element += 32) {
                    const int row = element / 16;
                    const int column = element - row * 16;
                    const int gm = gm0 + row;
                    const int gn = gn0 + column;
                    if (gm < M && gn < N) {
                        if constexpr (WRITE_F32) {
                            out_f32[(size_t)gm * N + gn] =
                                C_s[warp_n][row][column];
                        } else {
                            out[(size_t)gm * N + gn] =
                                __float2half_rn(C_s[warp_n][row][column]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

mfq_tensor_backend::Tensor nint8_zero_mmq_f16_packed_cuda(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor scale,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len)
{
    MFQ_RUNTIME_CHECK(
        q.is_cuda() && q.scalar_type() == mfq_tensor_backend::kUInt8 &&
        q.is_contiguous() && q.dim() == 3 && q.size(2) == 32,
        "NINT8-0 common FP16 MMQ q must be CUDA contiguous uint8 [N,G,32]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.scalar_type() == mfq_tensor_backend::kFloat16 &&
        scale.is_contiguous() && scale.sizes() == q.sizes().slice(0, 2),
        "NINT8-0 common FP16 MMQ scale shape mismatch");
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
        x.is_contiguous() && x.dim() == 2,
        "NINT8-0 common FP16 MMQ x must be CUDA contiguous fp16 rank-2");
    const int M = (int)x.size(0);
    const int K = (int)neuron_len;
    const int N = (int)q.size(0);
    const int groups = (int)q.size(1);
    MFQ_RUNTIME_CHECK(M >= 16, "NINT8-0 common FP16 MMQ requires M >= 16");
    MFQ_RUNTIME_CHECK(
        K > 0 && K <= groups * 32 && x.size(1) == K,
        "NINT8-0 common FP16 MMQ neuron_len mismatch");

    auto out = mfq_tensor_backend::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
#define NINT8_ZERO_F16_MMQ_LAUNCH(MTILES_VALUE)                        \
    nint8_zero_mmq_f16_packed_kernel<MTILES_VALUE, false><<<           \
        dim3((N + 63) / 64,                                            \
             (M + 16 * MTILES_VALUE - 1) / (16 * MTILES_VALUE)),      \
        dim3(32, 8), 0, stream>>>(                                     \
        q.data_ptr<uint8_t>(),                                         \
        reinterpret_cast<const __half*>(scale.data_ptr<mfq_half>()),   \
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),       \
        reinterpret_cast<__half*>(out.data_ptr<mfq_half>()), nullptr,  \
        M, N, groups, K)
    if (M <= 16) {
        NINT8_ZERO_F16_MMQ_LAUNCH(1);
    } else if (M <= 32) {
        NINT8_ZERO_F16_MMQ_LAUNCH(2);
    } else if (M <= 64) {
        NINT8_ZERO_F16_MMQ_LAUNCH(4);
    } else {
        NINT8_ZERO_F16_MMQ_LAUNCH(8);
    }
#undef NINT8_ZERO_F16_MMQ_LAUNCH
    MFQ_RUNTIME_CHECK(
        cudaGetLastError() == cudaSuccess,
        "NINT8-0 common FP16 MMQ kernel launch failed");
    return out;
}

mfq_tensor_backend::Tensor nint8_zero_mmq_f32_packed_cuda(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor scale,
    mfq_tensor_backend::Tensor x,
    int64_t neuron_len)
{
    MFQ_RUNTIME_CHECK(
        q.is_cuda() && q.scalar_type() == mfq_tensor_backend::kUInt8 &&
        q.is_contiguous() && q.dim() == 3 && q.size(2) == 32,
        "NINT8-0 common FP32-output MMQ q must be CUDA contiguous uint8 [N,G,32]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.scalar_type() == mfq_tensor_backend::kFloat16 &&
        scale.is_contiguous() && scale.sizes() == q.sizes().slice(0, 2),
        "NINT8-0 common FP32-output MMQ scale shape mismatch");
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
        x.is_contiguous() && x.dim() == 2,
        "NINT8-0 common FP32-output MMQ x must be CUDA contiguous fp16 rank-2");
    const int M = (int)x.size(0);
    const int K = (int)neuron_len;
    const int N = (int)q.size(0);
    const int groups = (int)q.size(1);
    MFQ_RUNTIME_CHECK(M >= 16, "NINT8-0 common FP32-output MMQ requires M >= 16");
    MFQ_RUNTIME_CHECK(
        K > 0 && K <= groups * 32 && x.size(1) == K,
        "NINT8-0 common FP32-output MMQ neuron_len mismatch");

    auto out = mfq_tensor_backend::empty(
        {M, N}, x.options().dtype(mfq_tensor_backend::kFloat32));
    cudaStream_t stream = mfq_current_cuda_stream();
#define NINT8_ZERO_F32_MMQ_LAUNCH(MTILES_VALUE)                        \
    nint8_zero_mmq_f16_packed_kernel<MTILES_VALUE, true><<<            \
        dim3((N + 63) / 64,                                            \
             (M + 16 * MTILES_VALUE - 1) / (16 * MTILES_VALUE)),      \
        dim3(32, 8), 0, stream>>>(                                     \
        q.data_ptr<uint8_t>(),                                         \
        reinterpret_cast<const __half*>(scale.data_ptr<mfq_half>()),   \
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),       \
        nullptr, out.data_ptr<float>(), M, N, groups, K)
    if (M <= 16) {
        NINT8_ZERO_F32_MMQ_LAUNCH(1);
    } else if (M <= 32) {
        NINT8_ZERO_F32_MMQ_LAUNCH(2);
    } else if (M <= 64) {
        NINT8_ZERO_F32_MMQ_LAUNCH(4);
    } else {
        NINT8_ZERO_F32_MMQ_LAUNCH(8);
    }
#undef NINT8_ZERO_F32_MMQ_LAUNCH
    MFQ_RUNTIME_CHECK(
        cudaGetLastError() == cudaSuccess,
        "NINT8-0 common FP32-output MMQ kernel launch failed");
    return out;
}
