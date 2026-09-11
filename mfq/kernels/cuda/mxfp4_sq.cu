#include "mxfp4_sq.h"
#include <algorithm>
#include <cstdint>

#include "packed_backward.cuh"

namespace {

// Exact frozen catalogs; source tests compare every entry to Metal.
__device__ __constant__ std::uint8_t kSq2Palette[128] = {
    15, 13, 0,  5,  15, 13, 1,  6,  15, 12, 2,  6,  14, 11, 0,  3,  14, 11, 1,
    5,  14, 11, 2,  6,  14, 10, 1,  4,  14, 10, 1,  5,  14, 10, 3,  6,  14, 10,
    4,  7,  14, 9,  5,  7,  13, 10, 1,  4,  13, 9,  3,  6,  13, 0,  5,  7,  12,
    9,  2,  5,  12, 9,  2,  6,  15, 12, 3,  7,  11, 0,  3,  6,  12, 9,  3,  6,
    13, 9,  2,  6,  14, 10, 3,  7,  15, 11, 4,  7,  15, 13, 9,  4,  15, 11, 1,
    5,  15, 11, 2,  6,  15, 12, 1,  6,  15, 12, 2,  7,  15, 13, 0,  6,  14, 11,
    0,  4,  13, 1,  5,  7,  15, 14, 13, 12, 15, 14, 13, 11,
};

__device__ __constant__ std::uint8_t kSq3Palette[256] = {
    15, 14, 13, 11, 0,  3,  5,  7,  15, 13, 11, 0,  3,  5,  6,  7,  15, 14, 13,
    11, 1,  4,  6,  7,  13, 11, 9,  0,  1,  2,  3,  6,  15, 14, 12, 9,  3,  5,
    6,  7, 15, 13, 12, 10, 0,  2,  4,  5,  13, 12, 10, 0,  2,  4,  5,  7,  14,
    12, 10, 0,  2,  4,  5,  7,  15, 14, 12, 10, 0,  3,  5,  7,  15, 13, 11, 0,
    2,  4,  5,  6,  15, 13, 11, 0,  2,  4,  6,  7,  15, 14, 12, 10, 0,  2,  4,
    5,  15, 14, 12, 10, 0,  2,  4,  6,  14, 13, 11, 0,  2,  4,  5,  6,  13, 12,
    10, 0,  3,  5,  6,  7,  15, 13, 12, 10, 0,  2,  4,  6,  14, 13, 12, 10, 0,
    3,  5,  6,  15, 13, 12, 10, 0,  3,  5,  6,  13, 12, 10, 0,  2,  4,  6,  7,
    14, 13, 11, 0,  2,  4,  5,  7,  13, 12, 9,  0,  2,  4,  5,  6,  14, 12, 10,
    0,  2,  4,  6,  7,  15, 14, 13, 11, 0,  3,  6,  7,  15, 14, 11, 0,  3,  5,
    6,  7,  14, 13, 12, 10, 0,  2,  4,  5,  14, 13, 12, 10, 0,  3,  5,  7,  15,
    14, 11, 0,  2,  4,  6,  7,  14, 11, 10, 9,  0,  1,  2,  3,  13, 11, 0,  2,
    4,  5,  6,  7,  15, 13, 12, 10, 0,  2,  4,  7,  15, 12, 10, 0,  2,  4,  5,
    7,  15, 13, 12, 10, 1,  4,  6,  7,
};

template<int BITS>
__device__ __forceinline__ unsigned read_bits(const std::uint8_t* data, std::size_t index) {
    const auto bit = index * BITS;
    const unsigned shift = bit & 7;
    unsigned value = data[bit >> 3];
    if (shift + BITS > 8) value |= unsigned(data[(bit >> 3) + 1]) << 8;
    return (value >> shift) & ((1u << BITS) - 1);
}

template<int BITS>
__device__ __forceinline__ unsigned block_tag(
        const std::uint8_t* symbols, const std::uint8_t* selectors, std::size_t block) {
    const auto* words = reinterpret_cast<const unsigned*>(symbols + block * (BITS * 4));
    unsigned low;
    if constexpr (BITS == 2) {
        unsigned fold = words[0] ^ words[1];
        fold ^= fold >> 16; fold ^= fold >> 8; fold ^= fold >> 4; fold ^= fold >> 2;
        low = fold & 3;
    } else {
        constexpr unsigned m0 = 0x49249249u, m1 = 0x92492492u, m2 = 0x24924924u;
        const unsigned a = words[0], b = words[1], c = words[2];
        const unsigned lo = __popc(a & m0) + __popc(b & m1) + __popc(c & m2);
        const unsigned hi = __popc(a & m1) + __popc(b & m2) + __popc(c & m0);
        low = (lo & 1) | ((hi & 1) << 1);
    }
    return low | (((selectors[block >> 3] >> (block & 7)) & 1) << 2);
}

template<int BITS>
__device__ __forceinline__ float decode_value(unsigned palette, unsigned symbol, unsigned exponent) {
    const unsigned nibble = BITS == 2 ? kSq2Palette[palette * 4 + symbol]
                                     : kSq3Palette[palette * 8 + symbol];
    float magnitude = float((0xc8643210u >> ((nibble & 7) * 4)) & 15) * 0.5f;
    if (nibble & 8) magnitude = -magnitude;
    const float scale = __uint_as_float(exponent == 0 ? 0x00400000u : exponent << 23);
    float result;
    // No .ftz: preserve base-zero E8M0 in --use_fast_math builds too.
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(magnitude), "f"(scale));
    return result;
}

template<typename T> __device__ __forceinline__ float as_float(T x) { return float(x); }
template<> __device__ __forceinline__ float as_float(__half x) { return __half2float(x); }
template<typename T> __device__ __forceinline__ T from_float(float x) { return T(x); }
template<> __device__ __forceinline__ __half from_float(float x) { return __float2half_rn(x); }

template<int BITS, typename T>
__global__ void sq_dequant(const std::uint8_t* blob, T* out, mfq::sq::Layout q) {
    const auto* symbols = blob + q.symbols;
    const std::size_t count = std::size_t(q.outputs) * q.width;
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += std::size_t(gridDim.x) * blockDim.x) {
        const auto state = (index / q.width) * 8 + block_tag<BITS>(symbols, blob + q.selectors, index / 32);
        const auto exponent = q.base + read_bits<2>(blob + q.scales, state);
        out[index] = from_float<T>(decode_value<BITS>(read_bits<5>(blob + q.palettes, state),
            read_bits<BITS>(symbols, index), exponent));
    }
}

// One warp per output, coalesced K loads, decode reuse across TILE_M rows.
// All batch sizes use packed weights; M=1..6 have exact tile specializations.
template<int BITS, int TILE_M, typename T>
__global__ void sq_mmq(const std::uint8_t* blob, const T* x, T* y,
                       mfq::sq::Layout q, int rows) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(threadIdx.x) >> 5;
    const std::int64_t output_tiles = (std::int64_t(q.outputs) + 3) / 4;
    const std::int64_t row_tiles = (std::int64_t(rows) + TILE_M - 1) / TILE_M;
    const auto* symbols = blob + q.symbols;
    for (std::int64_t task = blockIdx.x; task < output_tiles * row_tiles; task += gridDim.x) {
        const auto output = (task % output_tiles) * 4 + warp;
        if (output >= q.outputs) continue; // uniform within a whole warp
        const auto first_row = (task / output_tiles) * TILE_M;
        float accum[TILE_M] = {};
        for (int column = 0; column < q.width; column += 32) {
            const auto block = std::size_t(output) * (q.width / 32) + column / 32;
            unsigned state_value = 0;
            if (lane == 0) {
                const auto state = output * 8 + block_tag<BITS>(symbols, blob + q.selectors, block);
                state_value = (read_bits<5>(blob + q.palettes, state) << 8) |
                              (q.base + read_bits<2>(blob + q.scales, state));
            }
            state_value = __shfl_sync(0xffffffffu, state_value, 0);
            const auto index = std::size_t(output) * q.width + column + lane;
            const float w = decode_value<BITS>(state_value >> 8,
                read_bits<BITS>(symbols, index), state_value & 255);
#pragma unroll
            for (int m = 0; m < TILE_M; ++m)
                if (first_row + m < rows)
                    accum[m] = fmaf(w, as_float(x[(first_row + m) * q.width + column + lane]), accum[m]);
        }
#pragma unroll
        for (int m = 0; m < TILE_M; ++m) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1)
                accum[m] += __shfl_down_sync(0xffffffffu, accum[m], delta);
            if (lane == 0 && first_row + m < rows)
                y[(first_row + m) * q.outputs + output] = from_float<T>(accum[m]);
        }
    }
}

template<int BITS, typename T>
__global__ void sq_backward_input(
        const std::uint8_t* blob,
        const T* output_gradient,
        T* input_gradient,
        mfq::sq::Layout q,
        int rows) {
    const auto* symbols = blob + q.symbols;
    const std::size_t total = std::size_t(rows) * q.width;
    for (std::size_t logical = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         logical < total;
         logical += std::size_t(gridDim.x) * blockDim.x) {
        const int row = int(logical / q.width);
        const int column = int(logical - std::size_t(row) * q.width);
        const int block = column / 32;
        float accumulator = 0.0f;
        for (int output = 0; output < q.outputs; ++output) {
            const auto block_index = std::size_t(output) * (q.width / 32) + block;
            const auto state = std::size_t(output) * 8 +
                block_tag<BITS>(symbols, blob + q.selectors, block_index);
            const auto exponent = q.base + read_bits<2>(blob + q.scales, state);
            const auto palette = read_bits<5>(blob + q.palettes, state);
            const auto symbol = read_bits<BITS>(
                symbols, std::size_t(output) * q.width + column);
            accumulator = fmaf(
                as_float(output_gradient[
                    std::size_t(row) * q.outputs + output]),
                decode_value<BITS>(palette, symbol, exponent),
                accumulator);
        }
        input_gradient[logical] = from_float<T>(accumulator);
    }
}

template<int BITS, int Rows>
__global__ void __launch_bounds__(128) sq_backward_matrix(
        const std::uint8_t* blob,
        const __half* output_gradient,
        float* partials,
        mfq::sq::Layout q,
        int rows,
        int output_tile) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int block = static_cast<int>(blockIdx.x) * 4 + warp;
    const int column = block * 32 + lane;
    if (column >= q.width) return;

    const auto* symbols = blob + q.symbols;
    const int split = static_cast<int>(blockIdx.y);
    const int output0 = split * output_tile;
    const int output_end = min(output0 + output_tile, q.outputs);
    float accumulators[Rows] = {};
    for (int output = output0; output < output_end; ++output) {
        unsigned state_value = 0;
        if (lane == 0) {
            const auto block_index =
                std::size_t(output) * (q.width / 32) + block;
            const auto state = std::size_t(output) * 8 +
                block_tag<BITS>(symbols, blob + q.selectors, block_index);
            state_value =
                (read_bits<5>(blob + q.palettes, state) << 8) |
                (q.base + read_bits<2>(blob + q.scales, state));
        }
        state_value = __shfl_sync(0xffffffffu, state_value, 0);
        const float weight = decode_value<BITS>(
            state_value >> 8,
            read_bits<BITS>(
                symbols, std::size_t(output) * q.width + column),
            state_value & 255);
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float gradient = lane == 0 && row < rows
                ? __half2float(output_gradient[
                    std::size_t(row) * q.outputs + output])
                : 0.0f;
            gradient = __shfl_sync(0xffffffffu, gradient, 0);
            accumulators[row] = fmaf(gradient, weight, accumulators[row]);
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row < rows) {
            partials[(std::size_t(split) * rows + row) * q.width + column] =
                accumulators[row];
        }
    }
}

mfq::sq::Layout validate(const mfq_tensor_backend::Tensor& blob,
        std::int64_t bits, std::int64_t outputs, std::int64_t width, std::int64_t base) {
    const auto q = mfq::sq::layout(bits, outputs, width, base);
    MFQ_RUNTIME_CHECK(blob.is_cuda() && blob.scalar_type() == mfq_tensor_backend::kUInt8 &&
        blob.dim() == 1 && blob.is_contiguous() && std::size_t(blob.numel()) == q.bytes,
        "MXFP4-SQ blob requires contiguous rank-1 CUDA uint8 and exact payload length");
    MFQ_RUNTIME_CHECK(reinterpret_cast<std::uintptr_t>(blob.data_ptr<std::uint8_t>()) % 4 == 0,
        "MXFP4-SQ blob must be 4-byte aligned");
    return q;
}

template<int BITS, int M, typename T>
void launch_mmq(const std::uint8_t* blob, const T* x, T* y,
                mfq::sq::Layout q, int rows, cudaStream_t stream) {
    const auto tasks = ((std::int64_t(q.outputs) + 3) / 4) * ((std::int64_t(rows) + M - 1) / M);
    const int blocks = int(std::min<std::int64_t>(tasks, 65535));
    sq_mmq<BITS, M, T><<<blocks, 128, 0, stream>>>(blob, x, y, q, rows);
}

template<int BITS, typename T>
void dispatch_mmq(const std::uint8_t* blob, const T* x, T* y,
                  mfq::sq::Layout q, int rows, cudaStream_t stream) {
#define MFQ_SQ_M_CASE(M) case M: launch_mmq<BITS, M>(blob, x, y, q, rows, stream); break
    switch (rows) {
        MFQ_SQ_M_CASE(1); MFQ_SQ_M_CASE(2); MFQ_SQ_M_CASE(3);
        MFQ_SQ_M_CASE(4); MFQ_SQ_M_CASE(5); MFQ_SQ_M_CASE(6);
        default: launch_mmq<BITS, 8>(blob, x, y, q, rows, stream); break;
    }
#undef MFQ_SQ_M_CASE
}

template<int BITS, int Rows>
void launch_backward_matrix(
        const std::uint8_t* blob,
        const __half* output_gradient,
        float* partials,
        mfq::sq::Layout q,
        int rows,
        int output_tile,
        cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned>((q.width + 127) / 128),
        static_cast<unsigned>((q.outputs + output_tile - 1) / output_tile));
    sq_backward_matrix<BITS, Rows><<<
        grid, 128, 0, stream>>>(
        blob, output_gradient, partials, q, rows, output_tile);
}

template<int BITS>
void dispatch_backward_matrix(
        const std::uint8_t* blob,
        const __half* output_gradient,
        float* partials,
        mfq::sq::Layout q,
        int rows,
        int output_tile,
        cudaStream_t stream) {
    if (rows == 1) {
        launch_backward_matrix<BITS, 1>(
            blob, output_gradient, partials, q, rows, output_tile, stream);
    } else if (rows <= 2) {
        launch_backward_matrix<BITS, 2>(
            blob, output_gradient, partials, q, rows, output_tile, stream);
    } else if (rows <= 4) {
        launch_backward_matrix<BITS, 4>(
            blob, output_gradient, partials, q, rows, output_tile, stream);
    } else {
        launch_backward_matrix<BITS, 8>(
            blob, output_gradient, partials, q, rows, output_tile, stream);
    }
}

} // namespace

mfq_tensor_backend::Tensor mxfp4_sq_dequant_cuda(
        mfq_tensor_backend::Tensor blob, std::int64_t bits, std::int64_t outputs,
        std::int64_t width, std::int64_t base, bool fp32) {
    const auto q = validate(blob, bits, outputs, width, base);
    const MfqCudaGuard guard(blob.device());
    auto out = mfq_tensor_backend::empty({outputs, width}, blob.options().dtype(
        fp32 ? mfq_tensor_backend::kFloat32 : mfq_tensor_backend::kFloat16));
    const auto count = outputs * width;
    const int blocks = int(std::min<std::int64_t>((count + 255) / 256, 65535));
    const auto* data = blob.data_ptr<std::uint8_t>();
    const auto stream = mfq_current_cuda_stream();
#define MFQ_SQ_DEQUANT(B, T, PTR) sq_dequant<B, T><<<blocks, 256, 0, stream>>>(data, PTR, q)
    if (fp32) {
        if (bits == 2) { MFQ_SQ_DEQUANT(2, float, out.data_ptr<float>()); }
        else { MFQ_SQ_DEQUANT(3, float, out.data_ptr<float>()); }
    } else {
        auto* ptr = reinterpret_cast<__half*>(out.data_ptr<mfq_half>());
        if (bits == 2) { MFQ_SQ_DEQUANT(2, __half, ptr); }
        else { MFQ_SQ_DEQUANT(3, __half, ptr); }
    }
#undef MFQ_SQ_DEQUANT
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

mfq_tensor_backend::Tensor mxfp4_sq_matmul_cuda(
        mfq_tensor_backend::Tensor blob, mfq_tensor_backend::Tensor input,
        std::int64_t bits, std::int64_t outputs, std::int64_t width, std::int64_t base) {
    const auto q = validate(blob, bits, outputs, width, base);
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.get_device() == blob.get_device() &&
        input.dim() == 2 && input.is_contiguous() && input.size(1) == width &&
        input.size(0) <= std::numeric_limits<int>::max() &&
        (input.scalar_type() == mfq_tensor_backend::kFloat16 ||
         input.scalar_type() == mfq_tensor_backend::kFloat32),
        "MXFP4-SQ activation requires contiguous rank-2 CUDA FP16/FP32 on the weight device");
    const MfqCudaGuard guard(blob.device());
    auto out = mfq_tensor_backend::empty({input.size(0), outputs}, input.options());
    const int rows = int(input.size(0));
    if (!rows) return out;
    const auto* data = blob.data_ptr<std::uint8_t>();
    const auto stream = mfq_current_cuda_stream();
    if (input.scalar_type() == mfq_tensor_backend::kFloat32) {
        if (bits == 2) dispatch_mmq<2>(data, input.data_ptr<float>(), out.data_ptr<float>(), q, rows, stream);
        else dispatch_mmq<3>(data, input.data_ptr<float>(), out.data_ptr<float>(), q, rows, stream);
    } else {
        const auto* x = reinterpret_cast<const __half*>(input.data_ptr<mfq_half>());
        auto* y = reinterpret_cast<__half*>(out.data_ptr<mfq_half>());
        if (bits == 2) dispatch_mmq<2>(data, x, y, q, rows, stream);
        else dispatch_mmq<3>(data, x, y, q, rows, stream);
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

mfq_tensor_backend::Tensor mxfp4_sq_backward_input_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor output_gradient,
        std::int64_t bits,
        std::int64_t outputs,
        std::int64_t width,
        std::int64_t base) {
    const auto q = validate(blob, bits, outputs, width, base);
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.get_device() == blob.get_device() &&
        output_gradient.dim() == 2 && output_gradient.is_contiguous() &&
        output_gradient.size(1) == outputs &&
        (output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 ||
         output_gradient.scalar_type() == mfq_tensor_backend::kFloat32),
        "MXFP4-SQ backward requires contiguous CUDA FP16/FP32 [M,N]");
    const MfqCudaGuard guard(blob.device());
    auto result = mfq_tensor_backend::empty(
        {output_gradient.size(0), width}, output_gradient.options());
    const auto count = output_gradient.size(0) * width;
    if (!count) return result;
    constexpr int threads = 256;
    const int blocks = int(std::min<std::int64_t>(
        (count + threads - 1) / threads, 65535));
    const auto stream = mfq_current_cuda_stream();
    const auto* data = blob.data_ptr<std::uint8_t>();
    const int rows = int(output_gradient.size(0));
    if (output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 &&
            rows <= 8) {
        const int output_tile = rows <= 4 ? 16 : 32;
        const int splits = (int(outputs) + output_tile - 1) / output_tile;
        auto partials = mfq_tensor_backend::empty(
            {splits, rows, width},
            output_gradient.options().dtype(mfq_tensor_backend::kFloat32));
        const auto* gradient = reinterpret_cast<const __half*>(
            output_gradient.data_ptr<mfq_half>());
        auto* destination = reinterpret_cast<__half*>(
            result.data_ptr<mfq_half>());
        if (bits == 2) {
            dispatch_backward_matrix<2>(
                data, gradient, partials.data_ptr<float>(), q, rows,
                output_tile, stream);
        } else {
            dispatch_backward_matrix<3>(
                data, gradient, partials.data_ptr<float>(), q, rows,
                output_tile, stream);
        }
        mfq_packed_backward::launch_split_float_reduce_to_half(
            partials.data_ptr<float>(), destination,
            rows, int(width), splits, stream);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    auto weight = mxfp4_sq_dequant_cuda(
        blob, bits, outputs, width, base, false);
    mfq_packed_backward::launch_dense_half_weight(
        output_gradient, weight, result,
        rows, int(outputs), int(width), stream);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
