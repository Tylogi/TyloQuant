#include <cuda_fp16.h>
#include "../../../cpp_runtime/cuda/mfq_tensor_backend.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

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
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales,
        const mfq_tensor_backend::Tensor & x,
        mfq_tensor_backend::Tensor & y,
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
                reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), \
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

__global__ void mxfp8_embedding_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const int64_t * __restrict__ ids,
        __half * __restrict__ output,
        int count,
        int vocab,
        int width) {
    const int64_t total = static_cast<int64_t>(count) * width;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int token_index = static_cast<int>(linear / width);
        const int column = static_cast<int>(linear % width);
        const int64_t token = ids[token_index];
        float value = 0.0f;
        if (token >= 0 && token < vocab) {
            const int scale_columns = width / 128;
            const float scale = decode_e8m0(scales[
                (token / 128) * scale_columns + column / 128]);
            value = decode_e4m3fn(values[token * width + column]) * scale;
        }
        output[linear] = __float2half_rn(value);
    }
}

template <typename Output>
__global__ void __launch_bounds__(256) mxfp8_matmul_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const __half * __restrict__ input,
        Output * __restrict__ output,
        int rows,
        int outputs,
        int width,
        int row_tiles,
        int output_tiles) {
    constexpr int kRowsPerTile = 8;
    constexpr int kOutputsPerTile = 8;
    const int warp = int(threadIdx.y);
    const int lane = int(threadIdx.x);
    const int64_t tasks = static_cast<int64_t>(row_tiles) * output_tiles;
    for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int output_tile = static_cast<int>(task % output_tiles);
        const int row_tile = static_cast<int>(task / output_tiles);
        const int neuron = output_tile * kOutputsPerTile + warp;
        if (neuron >= outputs) continue;
        const int first_row = row_tile * kRowsPerTile;
        float accumulators[kRowsPerTile];
#pragma unroll
        for (int item = 0; item < kRowsPerTile; ++item) {
            accumulators[item] = 0.0f;
        }
        const int scale_columns = width / 128;
        for (int column = lane; column < width; column += 32) {
            const float weight = decode_e4m3fn(values[
                static_cast<int64_t>(neuron) * width + column]) *
                decode_e8m0(scales[
                    static_cast<int64_t>(neuron / 128) * scale_columns +
                    column / 128]);
#pragma unroll
            for (int item = 0; item < kRowsPerTile; ++item) {
                const int row = first_row + item;
                if (row < rows) {
                    accumulators[item] = fmaf(
                        __half2float(input[
                            static_cast<int64_t>(row) * width + column]),
                        weight,
                        accumulators[item]);
                }
            }
        }
#pragma unroll
        for (int item = 0; item < kRowsPerTile; ++item) {
            const float value = warp_sum(accumulators[item]);
            const int row = first_row + item;
            if (lane == 0 && row < rows) {
                if constexpr (std::is_same_v<Output, float>) {
                    output[static_cast<int64_t>(row) * outputs + neuron] = value;
                } else {
                    output[static_cast<int64_t>(row) * outputs + neuron] =
                        __float2half_rn(value);
                }
            }
        }
    }
}

void validate_mxfp8(
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales) {
    MFQ_RUNTIME_CHECK(values.is_cuda() && scales.is_cuda(),
                "MXFP8 values and scales must be CUDA tensors");
    MFQ_RUNTIME_CHECK(values.scalar_type() == mfq_tensor_backend::kUInt8 &&
                    scales.scalar_type() == mfq_tensor_backend::kUInt8,
                "MXFP8 values and scales must be uint8");
    MFQ_RUNTIME_CHECK(values.is_contiguous() && scales.is_contiguous(),
                "MXFP8 values and scales must be contiguous");
    MFQ_RUNTIME_CHECK(values.dim() == 2 && scales.dim() == 2,
                "MXFP8 values and scales must be rank-2");
    MFQ_RUNTIME_CHECK(values.size(0) > 0 && values.size(1) > 0 &&
                    values.size(0) <= std::numeric_limits<int>::max() &&
                    values.size(1) <= std::numeric_limits<int>::max() &&
                    values.size(1) % 128 == 0,
                "MXFP8 input width must be divisible by 128");
    MFQ_RUNTIME_CHECK(scales.size(0) == (values.size(0) + 127) / 128 &&
                    scales.size(1) == values.size(1) / 128,
                "MXFP8 scale geometry mismatch");
}

} // namespace

mfq_tensor_backend::Tensor mxfp8_dequant_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales) {
    validate_mxfp8(values, scales);
    auto dense = mfq_tensor_backend::empty(
        values.sizes(), values.options().dtype(mfq_tensor_backend::kFloat16));
    const std::size_t count = std::size_t(values.numel());
    constexpr int threads = 256;
    const int blocks = int((count + threads - 1) / threads);
    mxfp8_dequant_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            values.data_ptr<std::uint8_t>(),
            scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<__half *>(dense.data_ptr<mfq_half>()),
            int(values.size(0)), int(values.size(1)));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return dense;
}

mfq_tensor_backend::Tensor mxfp8_embedding_lookup_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor token_ids) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() &&
                    token_ids.scalar_type() == mfq_tensor_backend::kInt64 &&
                    token_ids.get_device() == values.get_device() &&
                    token_ids.numel() <= std::numeric_limits<int>::max(),
                "MXFP8 embedding ids must be contiguous CUDA int64 on the weight device");
    const int count = static_cast<int>(token_ids.numel());
    const int vocab = static_cast<int>(values.size(0));
    const int width = static_cast<int>(values.size(1));
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = mfq_tensor_backend::empty(shape, values.options().dtype(mfq_tensor_backend::kFloat16));
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(count) * width;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 4096));
    if (blocks > 0) {
        mxfp8_embedding_kernel<<<
            blocks, threads, 0, mfq_current_cuda_stream()>>>(
                values.data_ptr<std::uint8_t>(),
                scales.data_ptr<std::uint8_t>(), token_ids.data_ptr<int64_t>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
                count, vocab, width);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

mfq_tensor_backend::Tensor mxfp8_small_m_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor x) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    x.is_contiguous() && x.dim() == 2,
                "MXFP8 activation must be contiguous CUDA fp16 rank-2");
    MFQ_RUNTIME_CHECK(x.size(1) == values.size(1),
                "MXFP8 activation width mismatch");
    MFQ_RUNTIME_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 small-M kernel supports M in [1, 8]");
    auto y = mfq_tensor_backend::empty(
        {x.size(0), values.size(0)}, x.options());
    const cudaStream_t stream = mfq_current_cuda_stream();
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
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

mfq_tensor_backend::Tensor mxfp8_matmul_f16_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor input) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    input.is_contiguous() && input.dim() == 2 &&
                    input.size(0) > 0 && input.size(1) == values.size(1) &&
                    input.size(0) <= std::numeric_limits<int>::max(),
                "MXFP8 activation must be non-empty contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(input.size(1));
    auto result = mfq_tensor_backend::empty({rows, outputs}, input.options());
    constexpr int kRowsPerTile = 8;
    constexpr int kOutputsPerTile = 8;
    const int row_tiles = (rows + kRowsPerTile - 1) / kRowsPerTile;
    const int output_tiles =
        (outputs + kOutputsPerTile - 1) / kOutputsPerTile;
    const int64_t tasks = static_cast<int64_t>(row_tiles) * output_tiles;
    const int blocks = static_cast<int>(std::min<int64_t>(tasks, 4096));
    const dim3 threads(32, kOutputsPerTile);
    mxfp8_matmul_kernel<__half><<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            values.data_ptr<std::uint8_t>(),
            scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
            rows, outputs, width, row_tiles, output_tiles);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

mfq_tensor_backend::Tensor mxfp8_small_m_f32_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor x) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == values.size(1),
                "MXFP8 FP32-output activation geometry mismatch");
    MFQ_RUNTIME_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 FP32-output small-M kernel supports M in [1, 8]");
    auto y = mfq_tensor_backend::empty(
        {x.size(0), values.size(0)},
        x.options().dtype(mfq_tensor_backend::kFloat32));
    const cudaStream_t stream = mfq_current_cuda_stream();
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
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

mfq_tensor_backend::Tensor mxfp8_gemm_f32_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor x) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == values.size(1),
                "MXFP8 FP32-output GEMM activation geometry mismatch");
    const int M = int(x.size(0));
    const int N = int(values.size(0));
    auto y = mfq_tensor_backend::empty(
        {M, N}, x.options().dtype(mfq_tensor_backend::kFloat32));
    constexpr int kRowsPerTile = 8;
    constexpr int kOutputsPerTile = 8;
    const int row_tiles = (M + kRowsPerTile - 1) / kRowsPerTile;
    const int output_tiles = (N + kOutputsPerTile - 1) / kOutputsPerTile;
    const int64_t tasks = static_cast<int64_t>(row_tiles) * output_tiles;
    const int blocks = static_cast<int>(std::min<int64_t>(tasks, 4096));
    const dim3 threads(32, kOutputsPerTile);
    mxfp8_matmul_kernel<float><<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            values.data_ptr<std::uint8_t>(),
            scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
            y.data_ptr<float>(), M, N, int(x.size(1)),
            row_tiles, output_tiles);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

mfq_tensor_backend::Tensor mxfp8_groupwise_small_m_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor x,
        int64_t groups) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    x.is_contiguous() && x.dim() == 3,
                "MXFP8 groupwise activation must be contiguous CUDA fp16 rank-3");
    MFQ_RUNTIME_CHECK(groups > 0 && x.size(1) == groups &&
                    values.size(0) % groups == 0 &&
                    x.size(2) == values.size(1),
                "MXFP8 groupwise geometry mismatch");
    MFQ_RUNTIME_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 groupwise small-M kernel supports M in [1, 8]");
    const int64_t outputs_per_group = values.size(0) / groups;
    MFQ_RUNTIME_CHECK(outputs_per_group % 32 == 0,
                "MXFP8 groupwise outputs per group must be divisible by 32");
    auto y = mfq_tensor_backend::empty({x.size(0), values.size(0)}, x.options());
    const cudaStream_t stream = mfq_current_cuda_stream();
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
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

mfq_tensor_backend::Tensor mxfp8_groupwise_small_m_f32_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor x,
        int64_t groups) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    x.is_contiguous() && x.dim() == 3 &&
                    groups > 0 && x.size(1) == groups &&
                    values.size(0) % groups == 0 &&
                    x.size(2) == values.size(1),
                "MXFP8 groupwise FP32-output geometry mismatch");
    MFQ_RUNTIME_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 groupwise FP32-output kernel supports M in [1, 8]");
    const int64_t outputs_per_group = values.size(0) / groups;
    MFQ_RUNTIME_CHECK(outputs_per_group % 32 == 0,
                "MXFP8 groupwise outputs per group must be divisible by 32");
    auto y = mfq_tensor_backend::empty(
        {x.size(0), values.size(0)},
        x.options().dtype(mfq_tensor_backend::kFloat32));
    const cudaStream_t stream = mfq_current_cuda_stream();
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
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

namespace {

__device__ __forceinline__ float decode_mxfp4_e2m1(std::uint8_t code) {
    constexpr float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float value = magnitude[code & 7u];
    return (code & 8u) == 0u ? value : -value;
}

__global__ void mxfp4_dequant_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        __half * __restrict__ dense,
        int outputs,
        int width) {
    const std::size_t index =
        std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = std::size_t(outputs) * width;
    if (index >= count) return;
    const int output = int(index / width);
    const int column = int(index - std::size_t(output) * width);
    const std::uint8_t packed = values[
        std::size_t(output) * (width / 2) + column / 2];
    const std::uint8_t code = static_cast<std::uint8_t>(
        (packed >> ((column & 1) * 4)) & 15u);
    const float scale = decode_e8m0(
        scales[std::size_t(output) * (width / 32) + column / 32]);
    dense[index] = __float2half_rn(decode_mxfp4_e2m1(code) * scale);
}

__global__ void mxfp4_embedding_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const int64_t * __restrict__ ids,
        __half * __restrict__ output,
        int count,
        int vocab,
        int width) {
    const int64_t total = static_cast<int64_t>(count) * width;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int token_index = static_cast<int>(linear / width);
        const int column = static_cast<int>(linear % width);
        const int64_t token = ids[token_index];
        float value = 0.0f;
        if (token >= 0 && token < vocab) {
            const std::uint8_t byte = values[
                token * (width / 2) + column / 2];
            const std::uint8_t code = static_cast<std::uint8_t>(
                (byte >> ((column & 1) * 4)) & 15u);
            value = decode_mxfp4_e2m1(code) * decode_e8m0(
                scales[token * (width / 32) + column / 32]);
        }
        output[linear] = __float2half_rn(value);
    }
}

__global__ void __launch_bounds__(256) mxfp4_matmul_f16_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const __half * __restrict__ input,
        __half * __restrict__ output,
        int rows,
        int outputs,
        int width,
        int row_tiles,
        int output_tiles) {
    constexpr int kRowsPerTile = 8;
    constexpr int kOutputsPerTile = 8;
    const int warp = int(threadIdx.y);
    const int lane = int(threadIdx.x);
    const int64_t tasks = static_cast<int64_t>(row_tiles) * output_tiles;
    for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int output_tile = static_cast<int>(task % output_tiles);
        const int row_tile = static_cast<int>(task / output_tiles);
        const int neuron = output_tile * kOutputsPerTile + warp;
        if (neuron >= outputs) continue;
        const int first_row = row_tile * kRowsPerTile;
        float accumulators[kRowsPerTile];
#pragma unroll
        for (int item = 0; item < kRowsPerTile; ++item) {
            accumulators[item] = 0.0f;
        }
        const int packed_columns = width / 2;
        const int scale_columns = width / 32;
        for (int column = lane; column < width; column += 32) {
            const std::uint8_t packed = values[
                static_cast<int64_t>(neuron) * packed_columns + column / 2];
            const std::uint8_t code = static_cast<std::uint8_t>(
                (packed >> ((column & 1) * 4)) & 15u);
            const float weight = decode_mxfp4_e2m1(code) * decode_e8m0(
                scales[static_cast<int64_t>(neuron) * scale_columns +
                       column / 32]);
#pragma unroll
            for (int item = 0; item < kRowsPerTile; ++item) {
                const int row = first_row + item;
                if (row < rows) {
                    accumulators[item] = fmaf(
                        __half2float(input[static_cast<int64_t>(row) * width + column]),
                        weight,
                        accumulators[item]);
                }
            }
        }
#pragma unroll
        for (int item = 0; item < kRowsPerTile; ++item) {
            const float value = warp_sum(accumulators[item]);
            const int row = first_row + item;
            if (lane == 0 && row < rows) {
                output[static_cast<int64_t>(row) * outputs + neuron] =
                    __float2half_rn(value);
            }
        }
    }
}

void validate_mxfp4_dense(
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales) {
    MFQ_RUNTIME_CHECK(values.is_cuda() && scales.is_cuda(),
                "MXFP4 values and scales must be CUDA tensors");
    MFQ_RUNTIME_CHECK(values.scalar_type() == mfq_tensor_backend::kUInt8 &&
                    scales.scalar_type() == mfq_tensor_backend::kUInt8,
                "MXFP4 values and scales must be uint8");
    MFQ_RUNTIME_CHECK(values.is_contiguous() && scales.is_contiguous(),
                "MXFP4 values and scales must be contiguous");
    MFQ_RUNTIME_CHECK(values.dim() == 2 && scales.dim() == 2 &&
                    values.size(0) > 0 && values.size(1) > 0 &&
                    values.size(0) <= std::numeric_limits<int>::max() &&
                    values.size(1) <= std::numeric_limits<int>::max() / 2 &&
                    values.size(1) % 16 == 0 &&
                    scales.size(0) == values.size(0) &&
                    scales.size(1) == values.size(1) / 16,
                "MXFP4 weight geometry mismatch");
}

template <bool COMPACT_ROUTE>
__global__ void __launch_bounds__(256) mxfp4_moe_grouped_f16_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const __half * __restrict__ input,
        const std::int32_t * __restrict__ ids,
        const std::int32_t * __restrict__ expert_local,
        const std::int32_t * __restrict__ ids_dst,
        const std::int32_t * __restrict__ expert_bounds,
        const std::int32_t * __restrict__ tile_bounds,
        const std::int32_t * __restrict__ tile_experts,
        __half * __restrict__ output,
        int tokens,
        int routes,
        int global_experts,
        int pool_experts,
        int out_per_expert,
        int width,
        int max_tiles,
        int row_tiles,
        bool routed_input) {
    constexpr int kTileM = 8;
    constexpr int kRowsPerBlock = 8;
    const int warp = int(threadIdx.y);
    const int lane = int(threadIdx.x);
    const int pairs = tokens * routes;
    const int scale_columns = width / 32;
    const int packed_columns = width / 2;
    const int64_t max_tasks = COMPACT_ROUTE
        ? static_cast<int64_t>(max_tiles) * row_tiles
        : static_cast<int64_t>(pairs) * row_tiles;
    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int row_tile = static_cast<int>(task % row_tiles);
        int expert = -1;
        int first = 0;
        int last = 0;
        int single_pair = -1;
        if constexpr (COMPACT_ROUTE) {
            const int tile = static_cast<int>(task / row_tiles);
            if (tile >= tile_bounds[global_experts]) continue;
            expert = tile_experts[tile];
            const int local_tile = tile - tile_bounds[expert];
            first = expert_bounds[expert] + local_tile * kTileM;
            last = min(first + kTileM, expert_bounds[expert + 1]);
        } else {
            single_pair = static_cast<int>(task / row_tiles);
            expert = ids[single_pair];
            first = single_pair;
            last = single_pair + 1;
        }
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned>(local_expert) >=
                static_cast<unsigned>(pool_experts)) {
            continue;
        }
        const int local_row = row_tile * kRowsPerBlock + warp;
        if (local_row >= out_per_expert) continue;
        const int packed_row = local_expert * out_per_expert + local_row;
        float accum[kTileM];
#pragma unroll
        for (int item = 0; item < kTileM; ++item) accum[item] = 0.0f;
        for (int column = lane; column < width; column += 32) {
            const std::uint8_t packed = values[
                static_cast<int64_t>(packed_row) * packed_columns + column / 2];
            const std::uint8_t code = static_cast<std::uint8_t>(
                (packed >> ((column & 1) * 4)) & 15u);
            const float weight = decode_mxfp4_e2m1(code) * decode_e8m0(
                scales[static_cast<int64_t>(packed_row) * scale_columns +
                       column / 32]);
#pragma unroll
            for (int item = 0; item < kTileM; ++item) {
                int pair = -1;
                if constexpr (COMPACT_ROUTE) {
                    const int compact = first + item;
                    if (compact >= last) continue;
                    pair = ids_dst[compact];
                } else {
                    if (item != 0) continue;
                    pair = single_pair;
                }
                const int source_row = routed_input ? pair : pair / routes;
                accum[item] = fmaf(
                    __half2float(input[static_cast<int64_t>(source_row) * width + column]),
                    weight,
                    accum[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < kTileM; ++item) {
            float value = warp_sum(accum[item]);
            if (lane != 0) continue;
            int pair = -1;
            if constexpr (COMPACT_ROUTE) {
                const int compact = first + item;
                if (compact >= last) continue;
                pair = ids_dst[compact];
            } else {
                if (item != 0) continue;
                pair = single_pair;
            }
            output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                __float2half_rn(value);
        }
    }
}

void validate_mxfp4_moe(
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales,
        const mfq_tensor_backend::Tensor & input,
        const mfq_tensor_backend::Tensor & ids,
        const mfq_tensor_backend::Tensor & expert_local,
        const mfq_tensor_backend::Tensor & output,
        int64_t global_experts,
        int64_t pool_experts,
        int64_t out_per_expert,
        int64_t neuron_len) {
    MFQ_RUNTIME_CHECK(values.is_cuda() && scales.is_cuda() && input.is_cuda() &&
                    ids.is_cuda() && expert_local.is_cuda() && output.is_cuda(),
                "MXFP4 routed tensors must be CUDA tensors");
    MFQ_RUNTIME_CHECK(values.scalar_type() == mfq_tensor_backend::kUInt8 &&
                    scales.scalar_type() == mfq_tensor_backend::kUInt8 &&
                    input.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    ids.scalar_type() == mfq_tensor_backend::kInt32 &&
                    expert_local.scalar_type() == mfq_tensor_backend::kInt32 &&
                    output.scalar_type() == mfq_tensor_backend::kFloat16,
                "MXFP4 routed tensor dtypes are invalid");
    MFQ_RUNTIME_CHECK(values.is_contiguous() && scales.is_contiguous() &&
                    input.is_contiguous() && ids.is_contiguous() &&
                    expert_local.is_contiguous() && output.is_contiguous(),
                "MXFP4 routed tensors must be contiguous");
    MFQ_RUNTIME_CHECK(neuron_len > 0 && neuron_len % 32 == 0 &&
                    values.dim() == 2 &&
                    values.size(0) == pool_experts * out_per_expert &&
                    values.size(1) == neuron_len / 2 &&
                    scales.sizes() == mfq_tensor_backend::IntArrayRef(
                        {pool_experts * out_per_expert, neuron_len / 32}),
                "MXFP4 routed weight geometry mismatch");
    MFQ_RUNTIME_CHECK((input.dim() == 2 || input.dim() == 3) &&
                    input.size(-1) == neuron_len && ids.dim() == 2 &&
                    input.size(0) == ids.size(0) &&
                    output.sizes() == mfq_tensor_backend::IntArrayRef(
                        {ids.size(0), ids.size(1), out_per_expert}) &&
                    expert_local.numel() == global_experts,
                "MXFP4 routed activation/route geometry mismatch");
}

} // namespace

mfq_tensor_backend::Tensor mxfp4_dequant_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales) {
    validate_mxfp4_dense(values, scales);
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(values.size(1) * 2);
    auto dense = mfq_tensor_backend::empty(
        {outputs, width}, values.options().dtype(mfq_tensor_backend::kFloat16));
    const std::size_t count = static_cast<std::size_t>(outputs) * width;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    mxfp4_dequant_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            values.data_ptr<std::uint8_t>(),
            scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<__half *>(dense.data_ptr<mfq_half>()),
            outputs, width);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return dense;
}

mfq_tensor_backend::Tensor mxfp4_embedding_lookup_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor token_ids) {
    validate_mxfp4_dense(values, scales);
    MFQ_RUNTIME_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() &&
                    token_ids.scalar_type() == mfq_tensor_backend::kInt64 &&
                    token_ids.get_device() == values.get_device() &&
                    token_ids.numel() <= std::numeric_limits<int>::max(),
                "MXFP4 embedding ids must be contiguous CUDA int64 on the weight device");
    const int count = static_cast<int>(token_ids.numel());
    const int vocab = static_cast<int>(values.size(0));
    const int width = static_cast<int>(values.size(1) * 2);
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = mfq_tensor_backend::empty(shape, values.options().dtype(mfq_tensor_backend::kFloat16));
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(count) * width;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 4096));
    if (blocks > 0) {
        mxfp4_embedding_kernel<<<
            blocks, threads, 0, mfq_current_cuda_stream()>>>(
                values.data_ptr<std::uint8_t>(),
                scales.data_ptr<std::uint8_t>(), token_ids.data_ptr<int64_t>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
                count, vocab, width);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

mfq_tensor_backend::Tensor mxfp4_matmul_f16_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor input) {
    validate_mxfp4_dense(values, scales);
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    input.is_contiguous() && input.dim() == 2 &&
                    input.size(0) > 0 &&
                    input.size(0) <= std::numeric_limits<int>::max() &&
                    input.size(1) == values.size(1) * 2,
                "MXFP4 activation must be contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(input.size(1));
    auto result = mfq_tensor_backend::empty({rows, outputs}, input.options());
    constexpr int kRowsPerTile = 8;
    constexpr int kOutputsPerTile = 8;
    const int row_tiles = (rows + kRowsPerTile - 1) / kRowsPerTile;
    const int output_tiles =
        (outputs + kOutputsPerTile - 1) / kOutputsPerTile;
    const int64_t tasks = static_cast<int64_t>(row_tiles) * output_tiles;
    const int blocks = static_cast<int>(std::min<int64_t>(tasks, 4096));
    const dim3 threads(32, kOutputsPerTile);
    mxfp4_matmul_f16_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            values.data_ptr<std::uint8_t>(),
            scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
            rows, outputs, width, row_tiles, output_tiles);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

mfq_tensor_backend::Tensor mxfp4_moe_grouped_matmul_pool_f16_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor expert_local,
        int64_t global_experts,
        int64_t pool_experts,
        int64_t out_per_expert,
        int64_t neuron_len,
        mfq_tensor_backend::Tensor output,
        mfq_tensor_backend::Tensor ids_dst,
        mfq_tensor_backend::Tensor expert_bounds,
        mfq_tensor_backend::Tensor tile_bounds,
        mfq_tensor_backend::Tensor tile_experts) {
    validate_mxfp4_moe(
        values, scales, input, ids, expert_local, output,
        global_experts, pool_experts, out_per_expert, neuron_len);
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const int row_tiles =
        (static_cast<int>(out_per_expert) + 7) / 8;
    const dim3 threads(32, 8);
    const cudaStream_t stream = mfq_current_cuda_stream();
    if (ids_dst.numel() == ids.numel()) {
        MFQ_RUNTIME_CHECK(expert_bounds.is_cuda() && tile_bounds.is_cuda() &&
                        tile_experts.is_cuda() &&
                        expert_bounds.scalar_type() == mfq_tensor_backend::kInt32 &&
                        tile_bounds.scalar_type() == mfq_tensor_backend::kInt32 &&
                        tile_experts.scalar_type() == mfq_tensor_backend::kInt32 &&
                        expert_bounds.is_contiguous() &&
                        tile_bounds.is_contiguous() && tile_experts.is_contiguous() &&
                        expert_bounds.numel() >= global_experts + 1 &&
                        tile_bounds.numel() >= global_experts + 1,
                    "MXFP4 compact route map is invalid");
        const int max_tiles = (pairs + 7) / 8 + static_cast<int>(global_experts);
        const int blocks = static_cast<int>(std::min<int64_t>(
            static_cast<int64_t>(max_tiles) * row_tiles, 4096));
        mxfp4_moe_grouped_f16_kernel<true><<<blocks, threads, 0, stream>>>(
            values.data_ptr<std::uint8_t>(), scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            ids.data_ptr<std::int32_t>(), expert_local.data_ptr<std::int32_t>(),
            ids_dst.data_ptr<std::int32_t>(), expert_bounds.data_ptr<std::int32_t>(),
            tile_bounds.data_ptr<std::int32_t>(), tile_experts.data_ptr<std::int32_t>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            tokens, routes, static_cast<int>(global_experts),
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),
            static_cast<int>(neuron_len), max_tiles, row_tiles, input.dim() == 3);
    } else {
        const int blocks = static_cast<int>(std::min<int64_t>(
            static_cast<int64_t>(pairs) * row_tiles, 4096));
        mxfp4_moe_grouped_f16_kernel<false><<<blocks, threads, 0, stream>>>(
            values.data_ptr<std::uint8_t>(), scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            ids.data_ptr<std::int32_t>(), expert_local.data_ptr<std::int32_t>(),
            nullptr, nullptr, nullptr, nullptr,
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            tokens, routes, static_cast<int>(global_experts),
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),
            static_cast<int>(neuron_len), 0, row_tiles, input.dim() == 3);
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
