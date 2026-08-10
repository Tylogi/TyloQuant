// Native CUDA execution for TPQ symmetric int4 and learned product-VQ weights.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

__device__ __forceinline__ float tpq_warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

__device__ __forceinline__ std::uint32_t tpq_load_bits(
        const std::uint8_t * data,
        int64_t nbytes,
        int64_t bit,
        int bits) {
    const int64_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    std::uint32_t word = byte < nbytes ? data[byte] : 0u;
    if (byte + 1 < nbytes) {
        word |= static_cast<std::uint32_t>(data[byte + 1]) << 8;
    }
    if (byte + 2 < nbytes) {
        word |= static_cast<std::uint32_t>(data[byte + 2]) << 16;
    }
    return (word >> shift) & ((1u << bits) - 1u);
}

__global__ void __launch_bounds__(256) tpq_int4_matmul_kernel(
        const std::uint8_t * __restrict__ packed,
        const __half * __restrict__ scales,
        const __half * __restrict__ input,
        __half * __restrict__ output,
        int rows,
        int outputs,
        int width,
        int group_size,
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
        float accumulators[kRowsPerTile] = {};
        const int packed_columns = width / 2;
        const int scale_columns = width / group_size;
        for (int column = lane; column < width; column += 32) {
            const std::uint8_t value = packed[
                static_cast<int64_t>(neuron) * packed_columns + column / 2];
            const int quantized = static_cast<int>(
                (value >> ((column & 1) * 4)) & 15u) - 8;
            const float weight = static_cast<float>(quantized) * __half2float(
                scales[static_cast<int64_t>(neuron) * scale_columns +
                       column / group_size]);
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
            const float value = tpq_warp_sum(accumulators[item]);
            const int row = first_row + item;
            if (lane == 0 && row < rows) {
                output[static_cast<int64_t>(row) * outputs + neuron] =
                    __float2half_rn(value);
            }
        }
    }
}

__global__ void tpq_int4_dequant_kernel(
        const std::uint8_t * __restrict__ packed,
        const __half * __restrict__ scales,
        __half * __restrict__ output,
        int outputs,
        int width,
        int group_size) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const int64_t count = static_cast<int64_t>(outputs) * width;
    if (linear >= count) return;
    const int row = static_cast<int>(linear / width);
    const int column = static_cast<int>(linear - static_cast<int64_t>(row) * width);
    const std::uint8_t value = packed[
        static_cast<int64_t>(row) * (width / 2) + column / 2];
    const int quantized = static_cast<int>(
        (value >> ((column & 1) * 4)) & 15u) - 8;
    output[linear] = __float2half_rn(
        static_cast<float>(quantized) * __half2float(
            scales[static_cast<int64_t>(row) * (width / group_size) +
                   column / group_size]));
}

__global__ void __launch_bounds__(256) tpq_pq_matmul_kernel(
        const std::uint8_t * __restrict__ indices,
        int64_t indices_nbytes,
        const float * __restrict__ codebook,
        int codebook_entries,
        const __half * __restrict__ input,
        __half * __restrict__ output,
        int rows,
        int outputs,
        int width,
        int vector_size,
        int index_bits,
        int row_tiles,
        int output_tiles) {
    constexpr int kRowsPerTile = 8;
    constexpr int kOutputsPerTile = 8;
    const int warp = int(threadIdx.y);
    const int lane = int(threadIdx.x);
    const int vectors = width / vector_size;
    const int first_row_tile_stride = kRowsPerTile;
    const int64_t tasks = static_cast<int64_t>(row_tiles) * output_tiles;
    for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int output_tile = static_cast<int>(task % output_tiles);
        const int row_tile = static_cast<int>(task / output_tiles);
        const int neuron = output_tile * kOutputsPerTile + warp;
        if (neuron >= outputs) continue;
        const int first_row = row_tile * first_row_tile_stride;
        float accumulators[kRowsPerTile] = {};
        for (int vector = lane; vector < vectors; vector += 32) {
            const int64_t index_linear =
                static_cast<int64_t>(neuron) * vectors + vector;
            const std::uint32_t code = tpq_load_bits(
                indices, indices_nbytes, index_linear * index_bits, index_bits);
            if (code >= static_cast<std::uint32_t>(codebook_entries)) continue;
            const float * weight = codebook +
                static_cast<int64_t>(code) * vector_size;
            const int column = vector * vector_size;
            for (int component = 0; component < vector_size; ++component) {
                const float weight_value = weight[component];
#pragma unroll
                for (int item = 0; item < kRowsPerTile; ++item) {
                    const int row = first_row + item;
                    if (row < rows) {
                        accumulators[item] = fmaf(
                            __half2float(input[
                                static_cast<int64_t>(row) * width +
                                column + component]),
                            weight_value,
                            accumulators[item]);
                    }
                }
            }
        }
#pragma unroll
        for (int item = 0; item < kRowsPerTile; ++item) {
            const float value = tpq_warp_sum(accumulators[item]);
            const int row = first_row + item;
            if (lane == 0 && row < rows) {
                output[static_cast<int64_t>(row) * outputs + neuron] =
                    __float2half_rn(value);
            }
        }
    }
}

__global__ void tpq_pq_dequant_kernel(
        const std::uint8_t * __restrict__ indices,
        int64_t indices_nbytes,
        const float * __restrict__ codebook,
        int codebook_entries,
        __half * __restrict__ output,
        int outputs,
        int width,
        int vector_size,
        int index_bits) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const int64_t count = static_cast<int64_t>(outputs) * width;
    if (linear >= count) return;
    const int row = static_cast<int>(linear / width);
    const int column = static_cast<int>(linear - static_cast<int64_t>(row) * width);
    const int vectors = width / vector_size;
    const int vector = column / vector_size;
    const int component = column - vector * vector_size;
    const int64_t index_linear = static_cast<int64_t>(row) * vectors + vector;
    const std::uint32_t code = tpq_load_bits(
        indices, indices_nbytes, index_linear * index_bits, index_bits);
    output[linear] = code < static_cast<std::uint32_t>(codebook_entries)
        ? __float2half_rn(
              codebook[static_cast<int64_t>(code) * vector_size + component])
        : __float2half_rn(0.0f);
}

__global__ void tpq_int4_embedding_kernel(
        const std::uint8_t * __restrict__ packed,
        const __half * __restrict__ scales,
        const int64_t * __restrict__ ids,
        __half * __restrict__ output,
        int count,
        int vocab,
        int width,
        int group_size) {
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
            const std::uint8_t byte = packed[
                token * (width / 2) + column / 2];
            const int quantized = static_cast<int>(
                (byte >> ((column & 1) * 4)) & 15u) - 8;
            value = static_cast<float>(quantized) * __half2float(
                scales[token * (width / group_size) + column / group_size]);
        }
        output[linear] = __float2half_rn(value);
    }
}

__global__ void tpq_pq_embedding_kernel(
        const std::uint8_t * __restrict__ indices,
        int64_t indices_nbytes,
        const float * __restrict__ codebook,
        int codebook_entries,
        const int64_t * __restrict__ ids,
        __half * __restrict__ output,
        int count,
        int vocab,
        int width,
        int vector_size,
        int index_bits) {
    const int64_t total = static_cast<int64_t>(count) * width;
    const int vectors = width / vector_size;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int token_index = static_cast<int>(linear / width);
        const int column = static_cast<int>(linear % width);
        const int64_t token = ids[token_index];
        float value = 0.0f;
        if (token >= 0 && token < vocab) {
            const int vector = column / vector_size;
            const int component = column % vector_size;
            const int64_t index_linear = token * vectors + vector;
            const std::uint32_t code = tpq_load_bits(
                indices, indices_nbytes, index_linear * index_bits, index_bits);
            if (code < static_cast<std::uint32_t>(codebook_entries)) {
                value = codebook[
                    static_cast<int64_t>(code) * vector_size + component];
            }
        }
        output[linear] = __float2half_rn(value);
    }
}

template <bool CompactRoute>
__global__ void __launch_bounds__(256) tpq_pq_moe_kernel(
        const std::uint8_t * __restrict__ indices,
        int64_t indices_nbytes,
        const float * __restrict__ codebook,
        int codebook_entries,
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
        int vector_size,
        int index_bits,
        int max_tiles,
        int output_tiles,
        bool routed_input) {
    constexpr int kRouteTile = 8;
    constexpr int kOutputsPerTile = 8;
    const int warp = int(threadIdx.y);
    const int lane = int(threadIdx.x);
    const int vectors = width / vector_size;
    const int pairs = tokens * routes;
    const int64_t tasks = CompactRoute
        ? static_cast<int64_t>(max_tiles) * output_tiles
        : static_cast<int64_t>(pairs) * output_tiles;
    for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int output_tile = static_cast<int>(task % output_tiles);
        int expert = -1;
        int first = 0;
        int last = 0;
        int single_pair = -1;
        if constexpr (CompactRoute) {
            const int tile = static_cast<int>(task / output_tiles);
            if (tile >= tile_bounds[global_experts]) continue;
            expert = tile_experts[tile];
            if (static_cast<unsigned>(expert) >=
                    static_cast<unsigned>(global_experts)) continue;
            const int local_tile = tile - tile_bounds[expert];
            first = expert_bounds[expert] + local_tile * kRouteTile;
            last = min(first + kRouteTile, expert_bounds[expert + 1]);
        } else {
            single_pair = static_cast<int>(task / output_tiles);
            expert = ids[single_pair];
            if (static_cast<unsigned>(expert) >=
                    static_cast<unsigned>(global_experts)) continue;
            first = single_pair;
            last = single_pair + 1;
        }
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned>(local_expert) >=
                static_cast<unsigned>(pool_experts)) continue;
        const int local_output = output_tile * kOutputsPerTile + warp;
        if (local_output >= out_per_expert) continue;
        const int packed_row = local_expert * out_per_expert + local_output;
        float accumulators[kRouteTile] = {};
        for (int vector = lane; vector < vectors; vector += 32) {
            const int64_t index_linear =
                static_cast<int64_t>(packed_row) * vectors + vector;
            const std::uint32_t code = tpq_load_bits(
                indices, indices_nbytes, index_linear * index_bits, index_bits);
            if (code >= static_cast<std::uint32_t>(codebook_entries)) continue;
            const float * weight = codebook +
                static_cast<int64_t>(code) * vector_size;
            const int column = vector * vector_size;
            for (int component = 0; component < vector_size; ++component) {
                const float weight_value = weight[component];
#pragma unroll
                for (int item = 0; item < kRouteTile; ++item) {
                    int pair = -1;
                    if constexpr (CompactRoute) {
                        const int compact = first + item;
                        if (compact >= last) continue;
                        pair = ids_dst[compact];
                    } else {
                        if (item != 0) continue;
                        pair = single_pair;
                    }
                    const int source_row = routed_input ? pair : pair / routes;
                    accumulators[item] = fmaf(
                        __half2float(input[
                            static_cast<int64_t>(source_row) * width +
                            column + component]),
                        weight_value,
                        accumulators[item]);
                }
            }
        }
#pragma unroll
        for (int item = 0; item < kRouteTile; ++item) {
            const float value = tpq_warp_sum(accumulators[item]);
            if (lane != 0) continue;
            int pair = -1;
            if constexpr (CompactRoute) {
                const int compact = first + item;
                if (compact >= last) continue;
                pair = ids_dst[compact];
            } else {
                if (item != 0) continue;
                pair = single_pair;
            }
            output[static_cast<int64_t>(pair) * out_per_expert + local_output] =
                __float2half_rn(value);
        }
    }
}

void validate_tpq_int4(
        const torch::Tensor & packed,
        const torch::Tensor & scales,
        int64_t group_size) {
    TORCH_CHECK(packed.is_cuda() && scales.is_cuda() &&
                    packed.scalar_type() == torch::kUInt8 &&
                    scales.scalar_type() == torch::kFloat16 &&
                    packed.is_contiguous() && scales.is_contiguous() &&
                    packed.dim() == 2 && scales.dim() == 2 &&
                    packed.size(0) > 0 && packed.size(1) > 0 &&
                    packed.size(0) <= std::numeric_limits<int>::max() &&
                    packed.size(1) <= std::numeric_limits<int>::max() / 2 &&
                    group_size > 0 && (packed.size(1) * 2) % group_size == 0 &&
                    scales.size(0) == packed.size(0) &&
                    scales.size(1) == packed.size(1) * 2 / group_size,
                "TPQ-I4 packed weight geometry is invalid");
}

void validate_tpq_pq(
        const torch::Tensor & indices,
        const torch::Tensor & codebook,
        int64_t outputs,
        int64_t width,
        int64_t vector_size,
        int64_t index_bits) {
    TORCH_CHECK(indices.is_cuda() && codebook.is_cuda() &&
                    indices.scalar_type() == torch::kUInt8 &&
                    codebook.scalar_type() == torch::kFloat32 &&
                    indices.is_contiguous() && codebook.is_contiguous() &&
                    indices.dim() == 1 && codebook.dim() == 2 &&
                    outputs > 0 && width > 0 && vector_size > 0 &&
                    width % vector_size == 0 &&
                    index_bits >= 8 && index_bits <= 16 &&
                    codebook.size(1) == vector_size &&
                    codebook.size(0) > 0 &&
                    codebook.size(0) <= std::numeric_limits<int>::max() &&
                    codebook.size(0) <= (1ll << index_bits),
                "TPQ-PQ packed weight geometry is invalid");
    TORCH_CHECK(
        outputs <= std::numeric_limits<int>::max() &&
            width <= std::numeric_limits<int>::max() &&
            vector_size <= std::numeric_limits<int>::max() &&
            outputs <= std::numeric_limits<int64_t>::max() / (width / vector_size),
        "TPQ-PQ dimensions exceed supported integer ranges");
    const int64_t index_count = outputs * (width / vector_size);
    TORCH_CHECK(indices.numel() == (index_count * index_bits + 7) / 8,
                "TPQ-PQ packed index length is invalid");
}

} // namespace

torch::Tensor tpq_int4_matmul_f16_cuda(
        torch::Tensor packed,
        torch::Tensor scales,
        torch::Tensor input,
        int64_t group_size) {
    validate_tpq_int4(packed, scales, group_size);
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
                    input.is_contiguous() && input.dim() == 2 &&
                    input.size(0) > 0 && input.size(1) == packed.size(1) * 2,
                "TPQ-I4 activation geometry is invalid");
    const int rows = static_cast<int>(input.size(0));
    const int outputs = static_cast<int>(packed.size(0));
    const int width = static_cast<int>(input.size(1));
    auto result = torch::empty({rows, outputs}, input.options());
    const int row_tiles = (rows + 7) / 8;
    const int output_tiles = (outputs + 7) / 8;
    const int blocks = static_cast<int>(std::min<int64_t>(
        static_cast<int64_t>(row_tiles) * output_tiles, 4096));
    tpq_int4_matmul_kernel<<<
        blocks, dim3(32, 8), 0, at::cuda::getCurrentCUDAStream()>>>(
            packed.data_ptr<std::uint8_t>(),
            reinterpret_cast<const __half *>(scales.data_ptr<at::Half>()),
            reinterpret_cast<const __half *>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half *>(result.data_ptr<at::Half>()),
            rows, outputs, width, static_cast<int>(group_size),
            row_tiles, output_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

torch::Tensor tpq_int4_dequant_cuda(
        torch::Tensor packed,
        torch::Tensor scales,
        int64_t group_size) {
    validate_tpq_int4(packed, scales, group_size);
    const int outputs = static_cast<int>(packed.size(0));
    const int width = static_cast<int>(packed.size(1) * 2);
    auto result = torch::empty(
        {outputs, width}, packed.options().dtype(torch::kFloat16));
    const int64_t count = static_cast<int64_t>(outputs) * width;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    tpq_int4_dequant_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            packed.data_ptr<std::uint8_t>(),
            reinterpret_cast<const __half *>(scales.data_ptr<at::Half>()),
            reinterpret_cast<__half *>(result.data_ptr<at::Half>()),
            outputs, width, static_cast<int>(group_size));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

torch::Tensor tpq_pq_matmul_f16_cuda(
        torch::Tensor indices,
        torch::Tensor codebook,
        torch::Tensor input,
        int64_t outputs,
        int64_t width,
        int64_t vector_size,
        int64_t index_bits) {
    validate_tpq_pq(
        indices, codebook, outputs, width, vector_size, index_bits);
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
                    input.is_contiguous() && input.dim() == 2 &&
                    input.size(0) > 0 && input.size(1) == width,
                "TPQ-PQ activation geometry is invalid");
    const int rows = static_cast<int>(input.size(0));
    auto result = torch::empty({rows, outputs}, input.options());
    const int row_tiles = (rows + 7) / 8;
    const int output_tiles = (static_cast<int>(outputs) + 7) / 8;
    const int blocks = static_cast<int>(std::min<int64_t>(
        static_cast<int64_t>(row_tiles) * output_tiles, 4096));
    tpq_pq_matmul_kernel<<<
        blocks, dim3(32, 8), 0, at::cuda::getCurrentCUDAStream()>>>(
            indices.data_ptr<std::uint8_t>(), indices.numel(),
            codebook.data_ptr<float>(), static_cast<int>(codebook.size(0)),
            reinterpret_cast<const __half *>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half *>(result.data_ptr<at::Half>()),
            rows, static_cast<int>(outputs), static_cast<int>(width),
            static_cast<int>(vector_size), static_cast<int>(index_bits),
            row_tiles, output_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

torch::Tensor tpq_pq_dequant_cuda(
        torch::Tensor indices,
        torch::Tensor codebook,
        int64_t outputs,
        int64_t width,
        int64_t vector_size,
        int64_t index_bits) {
    validate_tpq_pq(
        indices, codebook, outputs, width, vector_size, index_bits);
    auto result = torch::empty(
        {outputs, width}, indices.options().dtype(torch::kFloat16));
    const int64_t count = outputs * width;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    tpq_pq_dequant_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            indices.data_ptr<std::uint8_t>(), indices.numel(),
            codebook.data_ptr<float>(), static_cast<int>(codebook.size(0)),
            reinterpret_cast<__half *>(result.data_ptr<at::Half>()),
            static_cast<int>(outputs), static_cast<int>(width),
            static_cast<int>(vector_size), static_cast<int>(index_bits));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

torch::Tensor tpq_int4_embedding_lookup_cuda(
        torch::Tensor packed,
        torch::Tensor scales,
        torch::Tensor token_ids,
        int64_t group_size) {
    validate_tpq_int4(packed, scales, group_size);
    TORCH_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() &&
                    token_ids.scalar_type() == torch::kInt64 &&
                    token_ids.get_device() == packed.get_device() &&
                    token_ids.numel() <= std::numeric_limits<int>::max(),
                "TPQ-I4 embedding ids must be contiguous CUDA int64");
    const int vocab = static_cast<int>(packed.size(0));
    const int width = static_cast<int>(packed.size(1) * 2);
    const int count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = torch::empty(shape, packed.options().dtype(torch::kFloat16));
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(count) * width;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 4096));
    if (blocks > 0) {
        tpq_int4_embedding_kernel<<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                packed.data_ptr<std::uint8_t>(),
                reinterpret_cast<const __half *>(scales.data_ptr<at::Half>()),
                token_ids.data_ptr<int64_t>(),
                reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
                count, vocab, width, static_cast<int>(group_size));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

torch::Tensor tpq_pq_embedding_lookup_cuda(
        torch::Tensor indices,
        torch::Tensor codebook,
        torch::Tensor token_ids,
        int64_t outputs,
        int64_t width,
        int64_t vector_size,
        int64_t index_bits) {
    validate_tpq_pq(
        indices, codebook, outputs, width, vector_size, index_bits);
    TORCH_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() &&
                    token_ids.scalar_type() == torch::kInt64 &&
                    token_ids.get_device() == indices.get_device() &&
                    token_ids.numel() <= std::numeric_limits<int>::max(),
                "TPQ-PQ embedding ids must be contiguous CUDA int64");
    const int count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = torch::empty(shape, indices.options().dtype(torch::kFloat16));
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(count) * width;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 4096));
    if (blocks > 0) {
        tpq_pq_embedding_kernel<<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                indices.data_ptr<std::uint8_t>(), indices.numel(),
                codebook.data_ptr<float>(), static_cast<int>(codebook.size(0)),
                token_ids.data_ptr<int64_t>(),
                reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
                count, static_cast<int>(outputs), static_cast<int>(width),
                static_cast<int>(vector_size), static_cast<int>(index_bits));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

torch::Tensor tpq_pq_moe_grouped_matmul_pool_f16_cuda(
        torch::Tensor indices,
        torch::Tensor codebook,
        torch::Tensor input,
        torch::Tensor ids,
        torch::Tensor expert_local,
        int64_t global_experts,
        int64_t pool_experts,
        int64_t out_per_expert,
        int64_t width,
        int64_t vector_size,
        int64_t index_bits,
        torch::Tensor output,
        torch::Tensor ids_dst,
        torch::Tensor expert_bounds,
        torch::Tensor tile_bounds,
        torch::Tensor tile_experts) {
    validate_tpq_pq(
        indices, codebook, pool_experts * out_per_expert,
        width, vector_size, index_bits);
    TORCH_CHECK(input.is_cuda() && ids.is_cuda() && expert_local.is_cuda() &&
                    output.is_cuda() && input.scalar_type() == torch::kFloat16 &&
                    ids.scalar_type() == torch::kInt32 &&
                    expert_local.scalar_type() == torch::kInt32 &&
                    output.scalar_type() == torch::kFloat16 &&
                    input.is_contiguous() && ids.is_contiguous() &&
                    expert_local.is_contiguous() && output.is_contiguous() &&
                    (input.dim() == 2 || input.dim() == 3) && ids.dim() == 2 &&
                    input.size(0) == ids.size(0) && input.size(-1) == width &&
                    expert_local.numel() == global_experts &&
                    output.sizes() == torch::IntArrayRef(
                        {ids.size(0), ids.size(1), out_per_expert}),
                "TPQ-PQ routed activation geometry is invalid");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const int output_tiles = (static_cast<int>(out_per_expert) + 7) / 8;
    const dim3 threads(32, 8);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (ids_dst.numel() == ids.numel()) {
        TORCH_CHECK(ids_dst.is_cuda() && expert_bounds.is_cuda() &&
                        tile_bounds.is_cuda() && tile_experts.is_cuda() &&
                        ids_dst.scalar_type() == torch::kInt32 &&
                        expert_bounds.scalar_type() == torch::kInt32 &&
                        tile_bounds.scalar_type() == torch::kInt32 &&
                        tile_experts.scalar_type() == torch::kInt32 &&
                        ids_dst.is_contiguous() && expert_bounds.is_contiguous() &&
                        tile_bounds.is_contiguous() && tile_experts.is_contiguous() &&
                        expert_bounds.numel() >= global_experts + 1 &&
                        tile_bounds.numel() >= global_experts + 1,
                    "TPQ-PQ compact route map is invalid");
        const int max_tiles = (pairs + 7) / 8 + static_cast<int>(global_experts);
        const int blocks = static_cast<int>(std::min<int64_t>(
            static_cast<int64_t>(max_tiles) * output_tiles, 4096));
        tpq_pq_moe_kernel<true><<<blocks, threads, 0, stream>>>(
            indices.data_ptr<std::uint8_t>(), indices.numel(),
            codebook.data_ptr<float>(), static_cast<int>(codebook.size(0)),
            reinterpret_cast<const __half *>(input.data_ptr<at::Half>()),
            ids.data_ptr<std::int32_t>(), expert_local.data_ptr<std::int32_t>(),
            ids_dst.data_ptr<std::int32_t>(), expert_bounds.data_ptr<std::int32_t>(),
            tile_bounds.data_ptr<std::int32_t>(), tile_experts.data_ptr<std::int32_t>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            tokens, routes, static_cast<int>(global_experts),
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),
            static_cast<int>(width), static_cast<int>(vector_size),
            static_cast<int>(index_bits), max_tiles, output_tiles,
            input.dim() == 3);
    } else {
        const int blocks = static_cast<int>(std::min<int64_t>(
            static_cast<int64_t>(pairs) * output_tiles, 4096));
        tpq_pq_moe_kernel<false><<<blocks, threads, 0, stream>>>(
            indices.data_ptr<std::uint8_t>(), indices.numel(),
            codebook.data_ptr<float>(), static_cast<int>(codebook.size(0)),
            reinterpret_cast<const __half *>(input.data_ptr<at::Half>()),
            ids.data_ptr<std::int32_t>(), expert_local.data_ptr<std::int32_t>(),
            nullptr, nullptr, nullptr, nullptr,
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            tokens, routes, static_cast<int>(global_experts),
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),
            static_cast<int>(width), static_cast<int>(vector_size),
            static_cast<int>(index_bits), 0, output_tiles, input.dim() == 3);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
