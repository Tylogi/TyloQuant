#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#define MFQ_MX_CUBLAS_CHECK(expr) \
    TORCH_CHECK((expr) == CUBLAS_STATUS_SUCCESS, \
                "MXFP8 cuBLAS call failed: ", #expr)

namespace {

__device__ __forceinline__ float decode_e8m0(std::uint8_t raw) {
    if (raw == 255u) {
        return __int_as_float(0x7fffffff);
    }
    // E8M0 stores the unbiased power-of-two exponent with the same bias
    // as FP32.  Constructing the FP32 exponent is exact and avoids an
    // expensive per-block ldexpf call.  raw=0 denotes 2^-127, the largest
    // FP32 subnormal power of two.
    return raw == 0u
        ? __int_as_float(0x00400000)
        : __int_as_float(unsigned(raw) << 23u);
}

__device__ __forceinline__ float decode_e4m3fn(std::uint8_t raw) {
    const unsigned exponent = (unsigned(raw) >> 3u) & 15u;
    const unsigned mantissa = unsigned(raw) & 7u;
    if (exponent == 15u && mantissa == 7u) {
        return __int_as_float(0x7fffffff);
    }
    // Normal E4M3FN numbers map exactly to FP32 by rebiasing the exponent
    // and shifting the three mantissa bits.  Subnormals are integer
    // multiples of 2^-9.  This removes one transcendental instruction per
    // weight while preserving every represented value exactly.
    const float magnitude = exponent == 0u
        ? float(mantissa) * 0.001953125f
        : __int_as_float(
            ((exponent + 120u) << 23u) |
            (mantissa << 20u));
    return (raw & 128u) == 0u ? magnitude : -magnitude;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

template <
    int TILE_M,
    int OUTPUTS_PER_WARP,
    bool CACHE_ACTIVATION,
    typename Output>
__global__ void mxfp8_small_m_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const __half * __restrict__ x,
        Output * __restrict__ y,
        int rows,
        int outputs,
        int width,
        int groups,
        int outputs_per_group) {
    extern __shared__ float shared_activation[];
    constexpr int kWarps = 8;
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(threadIdx.x) >> 5;
    const int block_first_output =
        int(blockIdx.x) * kWarps * OUTPUTS_PER_WARP;
    const int first_output =
        block_first_output + warp * OUTPUTS_PER_WARP;
    const int first_row = int(blockIdx.y) * TILE_M;
    const int activation_group = block_first_output / outputs_per_group;
    const bool valid_output = first_output < outputs;
    if (first_row >= rows) {
        return;
    }
    if constexpr (CACHE_ACTIVATION) {
        for (int column = int(threadIdx.x); column < width;
             column += int(blockDim.x)) {
            shared_activation[column] = __half2float(x[
                std::size_t(activation_group) * width + column]);
        }
        __syncthreads();
    }
    if (!valid_output) {
        return;
    }

    float accum[TILE_M][OUTPUTS_PER_WARP];
#pragma unroll
    for (int row = 0; row < TILE_M; ++row) {
#pragma unroll
        for (int output = 0; output < OUTPUTS_PER_WARP; ++output) {
            accum[row][output] = 0.0f;
        }
    }

    const int scale_columns = width / 128;
    for (int block_column = 0; block_column < width; block_column += 128) {
        float scale[OUTPUTS_PER_WARP];
#pragma unroll
        for (int output = 0; output < OUTPUTS_PER_WARP; ++output) {
            const int global_output = first_output + output;
            scale[output] = global_output < outputs
                ? decode_e8m0(scales[
                    (global_output / 128) * scale_columns +
                    block_column / 128])
                : 0.0f;
        }
#pragma unroll
        for (int local_column = lane; local_column < 128;
             local_column += 32) {
            const int column = block_column + local_column;
            float activation[TILE_M];
#pragma unroll
            for (int row = 0; row < TILE_M; ++row) {
                const int global_row = first_row + row;
                if constexpr (CACHE_ACTIVATION) {
                    activation[row] = shared_activation[column];
                } else {
                    activation[row] = global_row < rows
                        ? __half2float(x[
                            (std::size_t(global_row) * groups + activation_group) *
                            width + column])
                        : 0.0f;
                }
            }
#pragma unroll
            for (int output = 0; output < OUTPUTS_PER_WARP; ++output) {
                const int global_output = first_output + output;
                if (global_output >= outputs) {
                    continue;
                }
                const float weight = decode_e4m3fn(
                    values[(std::size_t)global_output * width + column]) *
                    scale[output];
#pragma unroll
                for (int row = 0; row < TILE_M; ++row) {
                    accum[row][output] += activation[row] * weight;
                }
            }
        }
    }

#pragma unroll
    for (int row = 0; row < TILE_M; ++row) {
        const int global_row = first_row + row;
#pragma unroll
        for (int output = 0; output < OUTPUTS_PER_WARP; ++output) {
            const int global_output = first_output + output;
            const float total = warp_sum(accum[row][output]);
            if (lane == 0 && global_row < rows && global_output < outputs) {
                if constexpr (std::is_same_v<Output, float>) {
                    y[(std::size_t)global_row * outputs + global_output] =
                        total;
                } else {
                    y[(std::size_t)global_row * outputs + global_output] =
                        __float2half_rn(total);
                }
            }
        }
    }
}

template <int TILE_M, typename Output>
void launch_mxfp8_small_m(
        const torch::Tensor & values,
        const torch::Tensor & scales,
        const torch::Tensor & x,
        torch::Tensor & y,
        int groups,
        int outputs_per_group,
        cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const int outputs = int(values.size(0));
#define MFQ_LAUNCH_MXFP8_OPW_CACHE(OPW, CACHE) \
    do { \
        const int outputs_per_block = warps * (OPW); \
        const dim3 grid( \
            unsigned((outputs + outputs_per_block - 1) / outputs_per_block), \
            1u, 1u); \
        const size_t shared_bytes = (CACHE) \
            ? size_t(values.size(1)) * sizeof(float) : 0; \
        mxfp8_small_m_kernel<TILE_M, OPW, CACHE, Output><<< \
            grid, threads, shared_bytes, stream>>>( \
                values.data_ptr<std::uint8_t>(), \
                scales.data_ptr<std::uint8_t>(), \
                reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), \
                reinterpret_cast<Output *>(y.data_ptr()), \
                int(x.size(0)), outputs, int(values.size(1)), \
                groups, outputs_per_group); \
    } while (false)
#define MFQ_LAUNCH_MXFP8_OPW(OPW) \
    do { \
        if constexpr (TILE_M == 1) { \
            /* Cache activations only for wide output projections. */ \
            if (outputs > 8192) { \
                MFQ_LAUNCH_MXFP8_OPW_CACHE(OPW, true); \
            } else { \
                MFQ_LAUNCH_MXFP8_OPW_CACHE(OPW, false); \
            } \
        } else { \
            MFQ_LAUNCH_MXFP8_OPW_CACHE(OPW, false); \
        } \
    } while (false)
    // One output per warp exposes independent work across the full output
    // axis and avoids a model-shape threshold in the launch policy.
    MFQ_LAUNCH_MXFP8_OPW(1);
#undef MFQ_LAUNCH_MXFP8_OPW
#undef MFQ_LAUNCH_MXFP8_OPW_CACHE
}

__global__ void mxfp8_dequant_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        __half * __restrict__ dense,
        int outputs,
        int width) {
    const std::size_t index =
        std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = std::size_t(outputs) * width;
    if (index >= count) {
        return;
    }
    const int output = int(index / width);
    const int column = int(index - std::size_t(output) * width);
    const int scale_columns = width / 128;
    const float scale = decode_e8m0(
        scales[(output / 128) * scale_columns + column / 128]);
    dense[index] = __float2half_rn(decode_e4m3fn(values[index]) * scale);
}

void validate_mxfp8(
        const torch::Tensor & values,
        const torch::Tensor & scales) {
    TORCH_CHECK(values.is_cuda() && scales.is_cuda(),
                "MXFP8 values and scales must be CUDA tensors");
    TORCH_CHECK(values.scalar_type() == torch::kUInt8 &&
                    scales.scalar_type() == torch::kUInt8,
                "MXFP8 values and scales must be uint8");
    TORCH_CHECK(values.is_contiguous() && scales.is_contiguous(),
                "MXFP8 values and scales must be contiguous");
    TORCH_CHECK(values.dim() == 2 && scales.dim() == 2,
                "MXFP8 values and scales must be rank-2");
    TORCH_CHECK(values.size(1) % 128 == 0,
                "MXFP8 input width must be divisible by 128");
    TORCH_CHECK(scales.size(0) == (values.size(0) + 127) / 128 &&
                    scales.size(1) == values.size(1) / 128,
                "MXFP8 scale geometry mismatch");
}

} // namespace

torch::Tensor mxfp8_dequant_cuda(
        torch::Tensor values,
        torch::Tensor scales) {
    validate_mxfp8(values, scales);
    auto dense = torch::empty(
        values.sizes(), values.options().dtype(torch::kFloat16));
    const std::size_t count = std::size_t(values.numel());
    constexpr int threads = 256;
    const int blocks = int((count + threads - 1) / threads);
    mxfp8_dequant_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            values.data_ptr<std::uint8_t>(),
            scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<__half *>(dense.data_ptr<at::Half>()),
            int(values.size(0)), int(values.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dense;
}

torch::Tensor mxfp8_small_m_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor x) {
    validate_mxfp8(values, scales);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                    x.is_contiguous() && x.dim() == 2,
                "MXFP8 activation must be contiguous CUDA fp16 rank-2");
    TORCH_CHECK(x.size(1) == values.size(1),
                "MXFP8 activation width mismatch");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 small-M kernel supports M in [1, 8]");
    auto y = torch::empty(
        {x.size(0), values.size(0)}, x.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define MFQ_LAUNCH_MXFP8(TILE) \
    launch_mxfp8_small_m<TILE, __half>( \
        values, scales, x, y, 1, int(values.size(0)), stream)
    if (x.size(0) == 1) {
        MFQ_LAUNCH_MXFP8(1);
    } else if (x.size(0) <= 2) {
        MFQ_LAUNCH_MXFP8(2);
    } else if (x.size(0) <= 4) {
        MFQ_LAUNCH_MXFP8(4);
    } else {
        MFQ_LAUNCH_MXFP8(8);
    }
#undef MFQ_LAUNCH_MXFP8
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor mxfp8_small_m_f32_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor x) {
    validate_mxfp8(values, scales);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                    x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == values.size(1),
                "MXFP8 FP32-output activation geometry mismatch");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 FP32-output small-M kernel supports M in [1, 8]");
    auto y = torch::empty(
        {x.size(0), values.size(0)},
        x.options().dtype(torch::kFloat32));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define MFQ_LAUNCH_MXFP8_F32(TILE) \
    launch_mxfp8_small_m<TILE, float>( \
        values, scales, x, y, 1, int(values.size(0)), stream)
    if (x.size(0) == 1) {
        MFQ_LAUNCH_MXFP8_F32(1);
    } else if (x.size(0) <= 2) {
        MFQ_LAUNCH_MXFP8_F32(2);
    } else if (x.size(0) <= 4) {
        MFQ_LAUNCH_MXFP8_F32(4);
    } else {
        MFQ_LAUNCH_MXFP8_F32(8);
    }
#undef MFQ_LAUNCH_MXFP8_F32
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor mxfp8_gemm_f32_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor x) {
    validate_mxfp8(values, scales);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                    x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == values.size(1),
                "MXFP8 FP32-output GEMM activation geometry mismatch");
    auto weight = mxfp8_dequant_cuda(values, scales);
    const int M = int(x.size(0));
    const int K = int(x.size(1));
    const int N = int(values.size(0));
    auto y = torch::empty(
        {M, N}, x.options().dtype(torch::kFloat32));
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    MFQ_MX_CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    MFQ_MX_CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        weight.data_ptr<at::Half>(), CUDA_R_16F, K,
        x.data_ptr<at::Half>(), CUDA_R_16F, K,
        &beta,
        y.data_ptr<float>(), CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return y;
}

torch::Tensor mxfp8_groupwise_small_m_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor x,
        int64_t groups) {
    validate_mxfp8(values, scales);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                    x.is_contiguous() && x.dim() == 3,
                "MXFP8 groupwise activation must be contiguous CUDA fp16 rank-3");
    TORCH_CHECK(groups > 0 && x.size(1) == groups &&
                    values.size(0) % groups == 0 &&
                    x.size(2) == values.size(1),
                "MXFP8 groupwise geometry mismatch");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 groupwise small-M kernel supports M in [1, 8]");
    const int64_t outputs_per_group = values.size(0) / groups;
    TORCH_CHECK(outputs_per_group % 32 == 0,
                "MXFP8 groupwise outputs per group must be divisible by 32");
    auto y = torch::empty({x.size(0), values.size(0)}, x.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define MFQ_LAUNCH_GROUPED_MXFP8(TILE) \
    launch_mxfp8_small_m<TILE, __half>( \
        values, scales, x, y, int(groups), int(outputs_per_group), stream)
    if (x.size(0) == 1) {
        MFQ_LAUNCH_GROUPED_MXFP8(1);
    } else if (x.size(0) <= 2) {
        MFQ_LAUNCH_GROUPED_MXFP8(2);
    } else if (x.size(0) <= 4) {
        MFQ_LAUNCH_GROUPED_MXFP8(4);
    } else {
        MFQ_LAUNCH_GROUPED_MXFP8(8);
    }
#undef MFQ_LAUNCH_GROUPED_MXFP8
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor mxfp8_groupwise_small_m_f32_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor x,
        int64_t groups) {
    validate_mxfp8(values, scales);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                    x.is_contiguous() && x.dim() == 3 &&
                    groups > 0 && x.size(1) == groups &&
                    values.size(0) % groups == 0 &&
                    x.size(2) == values.size(1),
                "MXFP8 groupwise FP32-output geometry mismatch");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 groupwise FP32-output kernel supports M in [1, 8]");
    const int64_t outputs_per_group = values.size(0) / groups;
    TORCH_CHECK(outputs_per_group % 32 == 0,
                "MXFP8 groupwise outputs per group must be divisible by 32");
    auto y = torch::empty(
        {x.size(0), values.size(0)},
        x.options().dtype(torch::kFloat32));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define MFQ_LAUNCH_GROUPED_MXFP8_F32(TILE) \
    launch_mxfp8_small_m<TILE, float>( \
        values, scales, x, y, int(groups), int(outputs_per_group), stream)
    if (x.size(0) == 1) {
        MFQ_LAUNCH_GROUPED_MXFP8_F32(1);
    } else if (x.size(0) <= 2) {
        MFQ_LAUNCH_GROUPED_MXFP8_F32(2);
    } else if (x.size(0) <= 4) {
        MFQ_LAUNCH_GROUPED_MXFP8_F32(4);
    } else {
        MFQ_LAUNCH_GROUPED_MXFP8_F32(8);
    }
#undef MFQ_LAUNCH_GROUPED_MXFP8_F32
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
