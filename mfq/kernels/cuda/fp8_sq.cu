#include "fp8_sq.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "packed_backward.cuh"

namespace {

constexpr int kDirectPackedMaxRows = 48;

struct Layout {
    int outputs = 0;
    int width = 0;
    int block_rows = 0;
    int block_columns = 0;
    int scale_rows = 0;
    int scale_columns = 0;
    int scale_kind = 0;
    std::size_t palettes = 0;
    std::size_t symbols = 0;
    std::size_t scales = 0;
};

__device__ __forceinline__ unsigned read_bits(
        const std::uint8_t* data, std::size_t index, int bits) {
    const auto bit = index * static_cast<std::size_t>(bits);
    const unsigned shift = static_cast<unsigned>(bit & 7);
    unsigned value = data[bit >> 3];
    if (shift + static_cast<unsigned>(bits) > 8) {
        value |= static_cast<unsigned>(data[(bit >> 3) + 1]) << 8;
    }
    return (value >> shift) & ((1u << bits) - 1u);
}

__device__ __forceinline__ float decode_e4m3fn(std::uint8_t raw) {
    const unsigned magnitude = static_cast<unsigned>(raw & 0x7fu);
    const unsigned exponent = magnitude >> 3u;
    const unsigned mantissa = magnitude & 7u;
    const float value = exponent == 0u
        ? static_cast<float>(mantissa) * 0.001953125f
        : __uint_as_float(((exponent + 120u) << 23u) | (mantissa << 20u));
    return (raw & 0x80u) == 0u ? value : -value;
}

__device__ __forceinline__ float decode_e8m0(std::uint8_t raw) {
    return raw == 0u
        ? __uint_as_float(0x00400000u)
        : __uint_as_float(static_cast<unsigned>(raw) << 23u);
}

__device__ __forceinline__ std::uint16_t load_u16(
        const std::uint8_t* source) {
    return static_cast<std::uint16_t>(source[0]) |
        static_cast<std::uint16_t>(source[1]) << 8;
}

__device__ __forceinline__ std::uint32_t load_u32(
        const std::uint8_t* source) {
    return static_cast<std::uint32_t>(source[0]) |
        static_cast<std::uint32_t>(source[1]) << 8 |
        static_cast<std::uint32_t>(source[2]) << 16 |
        static_cast<std::uint32_t>(source[3]) << 24;
}

__device__ __forceinline__ float decode_fp8_128_scale(
        const std::uint8_t* source, int scale_kind) {
    if (scale_kind == 2) {
        return __uint_as_float(static_cast<unsigned>(load_u16(source)) << 16);
    }
    if (scale_kind == 3) {
        const auto raw = load_u16(source);
        const unsigned exponent = (raw >> 10u) & 31u;
        const unsigned mantissa = raw & 1023u;
        if (exponent == 0u) {
            return static_cast<float>(mantissa) * 0x1p-24f;
        }
        return (1.0f + static_cast<float>(mantissa) / 1024.0f) *
            __uint_as_float((exponent + 112u) << 23u);
    }
    return __uint_as_float(load_u32(source));
}

__device__ __forceinline__ std::uint8_t decode_code(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const Layout& layout,
        int output,
        int column) {
    const int bits = static_cast<int>(row_q[output]);
    const auto* row_symbols = blob + layout.symbols +
        static_cast<std::size_t>(row_symbol_byte_offsets[output]);
    if (bits == 8) {
        return row_symbols[column];
    }
    const auto symbol = read_bits(
        row_symbols, static_cast<std::size_t>(column), bits);
    return blob[layout.palettes + (std::size_t{1} << bits) - 2 + symbol];
}

template <bool MXFP8>
__device__ __forceinline__ float decode_scale(
        const std::uint8_t* blob,
        const Layout& layout,
        int output,
        int column) {
    const auto scale_index =
        static_cast<std::size_t>(output / layout.block_rows) *
            layout.scale_columns +
        column / layout.block_columns;
    if constexpr (MXFP8) {
        return decode_e8m0(blob[layout.scales + scale_index]);
    } else {
        const int itemsize = layout.scale_kind == 4 ? 4 : 2;
        return decode_fp8_128_scale(
            blob + layout.scales + scale_index * itemsize,
            layout.scale_kind);
    }
}

template <bool MXFP8>
__device__ __forceinline__ float decode_weight(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const Layout& layout,
        int output,
        int column) {
    return decode_e4m3fn(decode_code(
        blob, row_q, row_symbol_byte_offsets, layout, output, column)) *
        decode_scale<MXFP8>(blob, layout, output, column);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

template <typename T>
__device__ __forceinline__ float as_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float as_float(__half value) {
    return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half from_float(float value) {
    return __float2half_rn(value);
}

template <bool MXFP8, typename T>
__device__ __forceinline__ void dequant_body(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        T* output,
        Layout layout) {
    const auto count = static_cast<std::size_t>(layout.outputs) * layout.width;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < count;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / layout.width);
        const int column = static_cast<int>(index % layout.width);
        output[index] = from_float<T>(decode_weight<MXFP8>(
            blob, row_q, row_symbol_byte_offsets, layout, row, column));
    }
}

template <typename T>
__global__ void mxfp8_sq_dequant_kernel(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        T* output,
        Layout layout) {
    dequant_body<true>(blob, row_q, row_symbol_byte_offsets, output, layout);
}

template <typename T>
__global__ void fp8_128_sq_dequant_kernel(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        T* output,
        Layout layout) {
    dequant_body<false>(blob, row_q, row_symbol_byte_offsets, output, layout);
}

template <bool MXFP8, int TILE_M, typename T, bool ROUTED>
__device__ __forceinline__ void mmq_body(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        int global_experts,
        int local_experts,
        int out_per_expert,
        int routes,
        bool shared_input) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int logical_outputs = ROUTED ? out_per_expert : layout.outputs;
    const auto output_tiles = (static_cast<std::int64_t>(logical_outputs) + 3) / 4;
    const auto row_tiles = (static_cast<std::int64_t>(rows) + TILE_M - 1) / TILE_M;
    for (std::int64_t task = blockIdx.x;
         task < output_tiles * row_tiles;
         task += gridDim.x) {
        const int logical_output = static_cast<int>(task % output_tiles) * 4 + warp;
        if (logical_output >= logical_outputs) continue;
        const int first_row = static_cast<int>(task / output_tiles) * TILE_M;
        int local_expert = 0;
        if constexpr (ROUTED) {
            const int expert = expert_ids[first_row];
            if (expert < 0 || expert >= global_experts) continue;
            local_expert = expert_local[expert];
            if (local_expert < 0 || local_expert >= local_experts) continue;
        }
        const int weight_row = ROUTED
            ? local_expert * out_per_expert + logical_output
            : logical_output;
        float accumulator[TILE_M] = {};
        for (int column = lane; column < layout.width; column += 32) {
            const float weight = decode_weight<MXFP8>(
                blob,
                row_q,
                row_symbol_byte_offsets,
                layout,
                weight_row,
                column);
#pragma unroll
            for (int item = 0; item < TILE_M; ++item) {
                if (first_row + item >= rows) continue;
                const int source_row = ROUTED && shared_input
                    ? (first_row + item) / routes
                    : first_row + item;
                accumulator[item] = fmaf(
                    weight,
                    as_float(input[
                        static_cast<std::size_t>(source_row) * layout.width + column]),
                    accumulator[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
            const float value = warp_sum(accumulator[item]);
            if (lane == 0 && first_row + item < rows) {
                output[
                    static_cast<std::size_t>(first_row + item) * logical_outputs +
                    logical_output] = from_float<T>(value);
            }
        }
    }
}

template <int TILE_M, typename T, bool ROUTED = false>
__global__ void mxfp8_sq_mmq_kernel(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        int global_experts,
        int local_experts,
        int out_per_expert,
        int routes,
        bool shared_input) {
    mmq_body<true, TILE_M, T, ROUTED>(
        blob, row_q, row_symbol_byte_offsets, input, output, layout, rows,
        expert_ids, expert_local, global_experts, local_experts,
        out_per_expert, routes, shared_input);
}

template <int TILE_M, typename T, bool ROUTED = false>
__global__ void fp8_128_sq_mmq_kernel(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        int global_experts,
        int local_experts,
        int out_per_expert,
        int routes,
        bool shared_input) {
    mmq_body<false, TILE_M, T, ROUTED>(
        blob, row_q, row_symbol_byte_offsets, input, output, layout, rows,
        expert_ids, expert_local, global_experts, local_experts,
        out_per_expert, routes, shared_input);
}

void validate_metadata(
        const mfq_tensor_backend::Tensor& blob,
        const mfq_tensor_backend::Tensor& row_q,
        const mfq_tensor_backend::Tensor& row_symbol_byte_offsets,
        const Layout& layout,
        bool mxfp8) {
    MFQ_RUNTIME_CHECK(
        blob.is_cuda() && blob.scalar_type() == mfq_tensor_backend::kUInt8 &&
        blob.dim() == 1 && blob.is_contiguous(),
        "FP8-SQ blob must be contiguous rank-1 CUDA uint8");
    MFQ_RUNTIME_CHECK(
        row_q.is_cuda() && row_q.get_device() == blob.get_device() &&
        row_q.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q.dim() == 1 && row_q.is_contiguous() &&
        row_q.numel() == layout.outputs &&
        row_symbol_byte_offsets.is_cuda() &&
        row_symbol_byte_offsets.get_device() == blob.get_device() &&
        row_symbol_byte_offsets.scalar_type() == mfq_tensor_backend::kInt32 &&
        row_symbol_byte_offsets.dim() == 1 &&
        row_symbol_byte_offsets.is_contiguous() &&
        row_symbol_byte_offsets.numel() == layout.outputs,
        "FP8-SQ row metadata must be matching CUDA arrays");
    MFQ_RUNTIME_CHECK(
        layout.outputs > 0 && layout.width > 0 &&
        layout.block_rows > 0 && layout.block_columns > 0 &&
        layout.scale_rows == (layout.outputs + layout.block_rows - 1) /
            layout.block_rows &&
        layout.scale_columns == (layout.width + layout.block_columns - 1) /
            layout.block_columns &&
        layout.palettes < layout.symbols && layout.symbols <= layout.scales &&
        layout.scales < static_cast<std::size_t>(blob.numel()),
        "FP8-SQ layout metadata is inconsistent");
    const std::size_t scale_count =
        static_cast<std::size_t>(layout.scale_rows) * layout.scale_columns;
    const std::size_t scale_itemsize = mxfp8 || layout.scale_kind == 1
        ? 1
        : layout.scale_kind == 4 ? 4 : 2;
    MFQ_RUNTIME_CHECK(
        scale_count <=
            (static_cast<std::size_t>(blob.numel()) - layout.scales) /
                scale_itemsize,
        "FP8-SQ scale payload exceeds its blob");
    if (mxfp8) {
        MFQ_RUNTIME_CHECK(
            layout.scale_kind == 1 &&
            ((layout.block_rows == 1 && layout.block_columns == 32) ||
             (layout.block_rows == 32 && layout.block_columns == 32) ||
             (layout.block_rows == 128 && layout.block_columns == 128)),
            "MXFP8-SQ scale contract is invalid");
    } else {
        MFQ_RUNTIME_CHECK(
            layout.block_rows == 128 && layout.block_columns == 128 &&
            (layout.scale_kind == 2 || layout.scale_kind == 3 ||
             layout.scale_kind == 4),
            "FP8-128SQ scale contract is invalid");
    }
}

Layout make_layout(
        std::int64_t outputs,
        std::int64_t width,
        std::int64_t block_rows,
        std::int64_t block_columns,
        std::int64_t scale_rows,
        std::int64_t scale_columns,
        std::int64_t scale_kind,
        std::int64_t palettes,
        std::int64_t symbols,
        std::int64_t scales) {
    MFQ_RUNTIME_CHECK(
        outputs > 0 && outputs <= std::numeric_limits<int>::max() &&
        width > 0 && width <= std::numeric_limits<int>::max() &&
        block_rows > 0 && block_rows <= std::numeric_limits<int>::max() &&
        block_columns > 0 && block_columns <= std::numeric_limits<int>::max() &&
        scale_rows > 0 && scale_rows <= std::numeric_limits<int>::max() &&
        scale_columns > 0 && scale_columns <= std::numeric_limits<int>::max() &&
        palettes >= 0 && symbols >= 0 && scales >= 0,
        "FP8-SQ host layout exceeds CUDA limits");
    return {
        static_cast<int>(outputs),
        static_cast<int>(width),
        static_cast<int>(block_rows),
        static_cast<int>(block_columns),
        static_cast<int>(scale_rows),
        static_cast<int>(scale_columns),
        static_cast<int>(scale_kind),
        static_cast<std::size_t>(palettes),
        static_cast<std::size_t>(symbols),
        static_cast<std::size_t>(scales),
    };
}

template <bool MXFP8, typename T>
void launch_dequant(
        const mfq_tensor_backend::Tensor& blob,
        const mfq_tensor_backend::Tensor& row_q,
        const mfq_tensor_backend::Tensor& row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor& output,
        Layout layout,
        cudaStream_t stream) {
    const auto count = static_cast<std::int64_t>(layout.outputs) * layout.width;
    const int blocks = static_cast<int>(std::min<std::int64_t>(
        (count + 255) / 256, 65535));
    if constexpr (MXFP8) {
        mxfp8_sq_dequant_kernel<<<blocks, 256, 0, stream>>>(
            blob.data_ptr<std::uint8_t>(),
            row_q.data_ptr<std::uint8_t>(),
            row_symbol_byte_offsets.data_ptr<std::int32_t>(),
            reinterpret_cast<T*>(output.data_ptr()),
            layout);
    } else {
        fp8_128_sq_dequant_kernel<<<blocks, 256, 0, stream>>>(
            blob.data_ptr<std::uint8_t>(),
            row_q.data_ptr<std::uint8_t>(),
            row_symbol_byte_offsets.data_ptr<std::int32_t>(),
            reinterpret_cast<T*>(output.data_ptr()),
            layout);
    }
}

template <bool MXFP8, int TILE_M, typename T>
void launch_mmq(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        cudaStream_t stream) {
    const auto tasks =
        (static_cast<std::int64_t>(layout.outputs) + 3) / 4 *
        ((static_cast<std::int64_t>(rows) + TILE_M - 1) / TILE_M);
    const int blocks = static_cast<int>(std::min<std::int64_t>(tasks, 65535));
    if constexpr (MXFP8) {
        mxfp8_sq_mmq_kernel<TILE_M><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets, input, output, layout, rows,
            nullptr, nullptr, 0, 0, layout.outputs, 1, false);
    } else {
        fp8_128_sq_mmq_kernel<TILE_M><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets, input, output, layout, rows,
            nullptr, nullptr, 0, 0, layout.outputs, 1, false);
    }
}

template <bool MXFP8, typename T>
void dispatch_mmq(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        cudaStream_t stream) {
#define MFQ_FP8_SQ_M_CASE(M) \
    case M: launch_mmq<MXFP8, M>( \
        blob, row_q, row_symbol_byte_offsets, input, output, layout, rows, stream); break
    switch (rows) {
        MFQ_FP8_SQ_M_CASE(1);
        MFQ_FP8_SQ_M_CASE(2);
        MFQ_FP8_SQ_M_CASE(3);
        MFQ_FP8_SQ_M_CASE(4);
        MFQ_FP8_SQ_M_CASE(5);
        MFQ_FP8_SQ_M_CASE(6);
        default:
            launch_mmq<MXFP8, 8>(
                blob, row_q, row_symbol_byte_offsets,
                input, output, layout, rows, stream);
            break;
    }
#undef MFQ_FP8_SQ_M_CASE
}

template <bool MXFP8>
void launch_routed(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const __half* input,
        __half* output,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        Layout layout,
        int route_count,
        int routes,
        int global_experts,
        int local_experts,
        int out_per_expert,
        bool shared_input,
        cudaStream_t stream) {
    const auto tasks = static_cast<std::int64_t>(route_count) *
        ((static_cast<std::int64_t>(out_per_expert) + 3) / 4);
    const int blocks = static_cast<int>(std::min<std::int64_t>(tasks, 65535));
    if constexpr (MXFP8) {
        mxfp8_sq_mmq_kernel<1, __half, true><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets, input, output, layout,
            route_count, expert_ids, expert_local, global_experts,
            local_experts, out_per_expert, routes, shared_input);
    } else {
        fp8_128_sq_mmq_kernel<1, __half, true><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets, input, output, layout,
            route_count, expert_ids, expert_local, global_experts,
            local_experts, out_per_expert, routes, shared_input);
    }
}

template <bool MXFP8>
mfq_tensor_backend::Tensor dequant(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        Layout layout,
        bool fp32) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    const MfqCudaGuard guard(blob.device());
    auto output = mfq_tensor_backend::empty(
        {layout.outputs, layout.width},
        blob.options().dtype(
            fp32 ? mfq_tensor_backend::kFloat32 : mfq_tensor_backend::kFloat16));
    const auto stream = mfq_current_cuda_stream();
    if (fp32) {
        launch_dequant<MXFP8, float>(
            blob, row_q, row_symbol_byte_offsets, output, layout, stream);
    } else {
        launch_dequant<MXFP8, __half>(
            blob, row_q, row_symbol_byte_offsets, output, layout, stream);
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void launch_dense_gemm_nt(
        const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& weight,
        mfq_tensor_backend::Tensor& output,
        cudaStream_t stream) {
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_RUNTIME_CHECK(
        cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
        "FP8-SQ cublasSetStream failed");

    // Row-major Y[M,N] = X[M,K] * W[N,K]^T maps to column-major
    // Y^T[N,M] = W[N,K] * X^T[K,M].
    if (input.scalar_type() == mfq_tensor_backend::kFloat16) {
        const __half alpha = __float2half(1.0f);
        const __half beta = __float2half(0.0f);
        MFQ_RUNTIME_CHECK(
            cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                outputs, rows, width,
                &alpha,
                weight.data_ptr<mfq_half>(), CUDA_R_16F, width,
                input.data_ptr<mfq_half>(), CUDA_R_16F, width,
                &beta,
                output.data_ptr<mfq_half>(), CUDA_R_16F, outputs,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS,
            "FP8-SQ FP16 GEMM failed");
        return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    MFQ_RUNTIME_CHECK(
        cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            outputs, rows, width,
            &alpha,
            weight.data_ptr<float>(), CUDA_R_32F, width,
            input.data_ptr<float>(), CUDA_R_32F, width,
            &beta,
            output.data_ptr<float>(), CUDA_R_32F, outputs,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS,
        "FP8-SQ FP32 GEMM failed");
}

template <bool MXFP8>
mfq_tensor_backend::Tensor matmul(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor input,
        Layout layout) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.get_device() == blob.get_device() &&
        input.dim() == 2 && input.is_contiguous() &&
        input.size(1) == layout.width &&
        input.size(0) <= std::numeric_limits<int>::max() &&
        (input.scalar_type() == mfq_tensor_backend::kFloat16 ||
         input.scalar_type() == mfq_tensor_backend::kFloat32),
        "FP8-SQ activation must be contiguous rank-2 CUDA FP16/FP32");
    const MfqCudaGuard guard(blob.device());
    auto output = mfq_tensor_backend::empty(
        {input.size(0), layout.outputs}, input.options());
    const int rows = static_cast<int>(input.size(0));
    if (rows == 0) return output;
    const auto stream = mfq_current_cuda_stream();
    if (rows > kDirectPackedMaxRows) {
        // The decoded matrix is a transient workspace owned by this call.  It
        // is never cached or attached to the packed weight.
        auto weight = dequant<MXFP8>(
            blob, row_q, row_symbol_byte_offsets, layout,
            input.scalar_type() == mfq_tensor_backend::kFloat32);
        launch_dense_gemm_nt(input, weight, output, stream);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }
    if (input.scalar_type() == mfq_tensor_backend::kFloat32) {
        dispatch_mmq<MXFP8>(
            blob.data_ptr<std::uint8_t>(),
            row_q.data_ptr<std::uint8_t>(),
            row_symbol_byte_offsets.data_ptr<std::int32_t>(),
            input.data_ptr<float>(),
            output.template data_ptr<float>(),
            layout, rows, stream);
    } else {
        dispatch_mmq<MXFP8>(
            blob.data_ptr<std::uint8_t>(),
            row_q.data_ptr<std::uint8_t>(),
            row_symbol_byte_offsets.data_ptr<std::int32_t>(),
            reinterpret_cast<const __half*>(input.data_ptr<mfq_half>()),
            reinterpret_cast<__half*>(output.data_ptr<mfq_half>()),
            layout, rows, stream);
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <bool MXFP8>
void routed_matmul(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor expert_ids,
        mfq_tensor_backend::Tensor expert_local,
        std::int64_t n_experts,
        std::int64_t local_experts,
        std::int64_t out_per_expert,
        Layout layout,
        mfq_tensor_backend::Tensor output) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    MFQ_RUNTIME_CHECK(
        layout.outputs == local_experts * out_per_expert &&
        input.is_cuda() && input.get_device() == blob.get_device() &&
        input.is_contiguous() && input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        (input.dim() == 2 || input.dim() == 3) && input.size(-1) == layout.width &&
        expert_ids.is_cuda() && expert_ids.get_device() == blob.get_device() &&
        expert_ids.is_contiguous() &&
        expert_ids.scalar_type() == mfq_tensor_backend::kInt32 &&
        expert_ids.dim() == 2 && expert_ids.size(0) == input.size(0) &&
        expert_local.is_cuda() && expert_local.get_device() == blob.get_device() &&
        expert_local.is_contiguous() &&
        expert_local.scalar_type() == mfq_tensor_backend::kInt32 &&
        expert_local.dim() == 1 && expert_local.numel() == n_experts &&
        output.is_cuda() && output.get_device() == blob.get_device() &&
        output.is_contiguous() && output.scalar_type() == mfq_tensor_backend::kFloat16 &&
        output.sizes() == mfq_tensor_backend::IntArrayRef(
            {expert_ids.size(0), expert_ids.size(1), out_per_expert}) &&
        (input.dim() == 2 || input.size(1) == expert_ids.size(1)) &&
        n_experts > 0 && n_experts <= std::numeric_limits<int>::max() &&
        local_experts > 0 && local_experts <= std::numeric_limits<int>::max() &&
        out_per_expert > 0 && out_per_expert <= std::numeric_limits<int>::max(),
        "FP8-SQ routed matmul requires matching contiguous CUDA FP16 tensors");
    MFQ_RUNTIME_CHECK(
        expert_ids.numel() <= std::numeric_limits<int>::max(),
        "FP8-SQ route count exceeds CUDA limits");
    if (expert_ids.numel() == 0) return;
    const MfqCudaGuard guard(blob.device());
    launch_routed<MXFP8>(
        blob.data_ptr<std::uint8_t>(),
        row_q.data_ptr<std::uint8_t>(),
        row_symbol_byte_offsets.data_ptr<std::int32_t>(),
        reinterpret_cast<const __half*>(input.data_ptr<mfq_half>()),
        reinterpret_cast<__half*>(output.data_ptr<mfq_half>()),
        expert_ids.data_ptr<std::int32_t>(),
        expert_local.data_ptr<std::int32_t>(),
        layout,
        static_cast<int>(expert_ids.numel()),
        static_cast<int>(expert_ids.size(1)),
        static_cast<int>(n_experts),
        static_cast<int>(local_experts),
        static_cast<int>(out_per_expert),
        input.dim() == 2,
        mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool MXFP8>
mfq_tensor_backend::Tensor backward_input(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor output_gradient,
        Layout layout) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.get_device() == blob.get_device() &&
        output_gradient.dim() == 2 && output_gradient.is_contiguous() &&
        output_gradient.size(1) == layout.outputs &&
        output_gradient.size(0) <= std::numeric_limits<int>::max() &&
        (output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 ||
         output_gradient.scalar_type() == mfq_tensor_backend::kBFloat16 ||
         output_gradient.scalar_type() == mfq_tensor_backend::kFloat32),
        "FP8-SQ backward requires contiguous CUDA FP16/BF16/FP32 [M,N]");
    const MfqCudaGuard guard(blob.device());
    auto result = mfq_tensor_backend::empty(
        {output_gradient.size(0), layout.width}, output_gradient.options());
    if (output_gradient.size(0) == 0) return result;
    auto weight = dequant<MXFP8>(
        blob, row_q, row_symbol_byte_offsets, layout, false);
    mfq_packed_backward::launch_dense_half_weight(
        output_gradient, weight, result,
        static_cast<int>(output_gradient.size(0)),
        layout.outputs, layout.width, mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

} // namespace

mfq_tensor_backend::Tensor mxfp8_sq_dequant_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        std::int64_t outputs, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset, bool fp32) {
    return dequant<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets),
        make_layout(outputs, width, block_rows, block_columns,
                    scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset),
        fp32);
}

mfq_tensor_backend::Tensor fp8_128_sq_dequant_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        std::int64_t outputs, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset, bool fp32) {
    return dequant<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets),
        make_layout(outputs, width, 128, 128,
                    (outputs + 127) / 128, (width + 127) / 128,
                    scale_kind, palettes_offset, symbols_offset, scales_offset),
        fp32);
}

mfq_tensor_backend::Tensor mxfp8_sq_matmul_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor input,
        std::int64_t outputs, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return matmul<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(input),
        make_layout(outputs, width, block_rows, block_columns,
                    scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset));
}

mfq_tensor_backend::Tensor fp8_128_sq_matmul_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor input,
        std::int64_t outputs, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return matmul<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(input),
        make_layout(outputs, width, 128, 128,
                    (outputs + 127) / 128, (width + 127) / 128,
                    scale_kind, palettes_offset, symbols_offset, scales_offset));
}

mfq_tensor_backend::Tensor mxfp8_sq_backward_input_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor output_gradient,
        std::int64_t outputs, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return backward_input<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(output_gradient),
        make_layout(outputs, width, block_rows, block_columns,
                    scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset));
}

mfq_tensor_backend::Tensor fp8_128_sq_backward_input_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor output_gradient,
        std::int64_t outputs, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return backward_input<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(output_gradient),
        make_layout(outputs, width, 128, 128,
                    (outputs + 127) / 128, (width + 127) / 128,
                    scale_kind, palettes_offset, symbols_offset, scales_offset));
}

void mxfp8_sq_moe_matmul_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor expert_ids,
        mfq_tensor_backend::Tensor expert_local,
        std::int64_t n_experts, std::int64_t local_experts,
        std::int64_t out_per_expert, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset,
        mfq_tensor_backend::Tensor output) {
    routed_matmul<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(input),
        std::move(expert_ids), std::move(expert_local),
        n_experts, local_experts, out_per_expert,
        make_layout(local_experts * out_per_expert, width,
                    block_rows, block_columns, scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset),
        std::move(output));
}

void fp8_128_sq_moe_matmul_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor expert_ids,
        mfq_tensor_backend::Tensor expert_local,
        std::int64_t n_experts, std::int64_t local_experts,
        std::int64_t out_per_expert, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset,
        mfq_tensor_backend::Tensor output) {
    routed_matmul<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(input),
        std::move(expert_ids), std::move(expert_local),
        n_experts, local_experts, out_per_expert,
        make_layout(local_experts * out_per_expert, width, 128, 128,
                    (local_experts * out_per_expert + 127) / 128,
                    (width + 127) / 128, scale_kind,
                    palettes_offset, symbols_offset, scales_offset),
        std::move(output));
}
