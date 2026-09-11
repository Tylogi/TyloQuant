

#include <cuda.h>
#include "mfq_tensor_backend.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>


namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ float residual_dot(
    const __half * input,
    const __half * dictionary,
    int record,
    int position_bits,
    int block_vectors,
    int block,
    int width) {
    if (record < 0) return 0.0f;
    const int position = record & ((1 << position_bits) - 1);
    const int dictionary_id = record >> position_bits;
    const int vector = block * block_vectors + position;
    if (dictionary_id >= 1024 || vector >= width / 8) return 0.0f;
    const __half * source = input + static_cast<int64_t>(vector) * 8;
    const __half * code = dictionary + static_cast<int64_t>(dictionary_id) * 8;
    float value = 0.0f;
#pragma unroll
    for (int component = 0; component < 8; ++component) {
        value = fmaf(
            __half2float(source[component]),
            __half2float(code[component]),
            value);
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void nepq_sparse_residual_matmul_kernel(
    const __half * dictionary,
    const int16_t * first,
    const int16_t * second,
    const __half * input,
    __half * output,
    int rows,
    int width,
    int input_rows,
    int blocks_per_row,
    int position_bits,
    int block_vectors) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int64_t logical =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const int64_t total = static_cast<int64_t>(input_rows) * rows;
    if (logical >= total) return;
    const int input_row = static_cast<int>(logical / rows);
    const int row = static_cast<int>(logical - static_cast<int64_t>(input_row) * rows);
    const __half * source = input + static_cast<int64_t>(input_row) * width;
    const int16_t * first_row = first + static_cast<int64_t>(row) * blocks_per_row;
    const int16_t * second_row = second + static_cast<int64_t>(row) * blocks_per_row;
    float value = 0.0f;
    for (int block = lane; block < blocks_per_row; block += kWarpSize) {
        value += residual_dot(
            source,
            dictionary,
            first_row[block],
            position_bits,
            block_vectors,
            block,
            width);
        const int record = second_row[block];
        if (record >= 0) {
            value += residual_dot(
                source,
                dictionary,
                record,
                position_bits,
                block_vectors,
                block,
                width);
        }
    }
    value = warp_sum(value);
    if (lane == 0) {
        __half * destination = output + logical;
        *destination = __float2half(__half2float(*destination) + value);
    }
}

__global__ void nepq_sparse_residual_dequant_kernel(
    const __half * dictionary,
    const int16_t * first,
    const int16_t * second,
    __half * weight,
    int width,
    int blocks_per_row,
    int position_bits,
    int block_vectors,
    int64_t total_blocks) {
    const int64_t logical =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (logical >= total_blocks) return;
    const int row = static_cast<int>(logical / blocks_per_row);
    const int block = static_cast<int>(logical - static_cast<int64_t>(row) * blocks_per_row);
    const int records[2] = {first[logical], second[logical]};
#pragma unroll
    for (int stream = 0; stream < 2; ++stream) {
        const int record = records[stream];
        if (record < 0) continue;
        const int position = record & ((1 << position_bits) - 1);
        const int dictionary_id = record >> position_bits;
        const int column = (block * block_vectors + position) * 8;
        if (dictionary_id >= 1024 || column >= width) continue;
        __half * destination = weight + static_cast<int64_t>(row) * width + column;
        const __half * code = dictionary + static_cast<int64_t>(dictionary_id) * 8;
#pragma unroll
        for (int component = 0; component < 8; ++component) {
            destination[component] = __float2half(
                __half2float(destination[component])
                + __half2float(code[component]));
        }
    }
}

__global__ void nepq_sparse_residual_grouped_kernel(
    const __half * dictionary,
    const int16_t * first,
    const int16_t * second,
    const __half * input,
    const int32_t * route_ids,
    const int32_t * expert_local,
    __half * output,
    int tokens,
    int routes,
    int out_per_expert,
    int width,
    int blocks_per_row,
    int position_bits,
    int block_vectors,
    bool routed_input,
    bool mapped_experts,
    int global_experts,
    int local_experts) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int64_t logical =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const int64_t total =
        static_cast<int64_t>(tokens) * routes * out_per_expert;
    if (logical >= total) return;
    const int output_row = static_cast<int>(logical / out_per_expert);
    const int out = static_cast<int>(logical - static_cast<int64_t>(output_row) * out_per_expert);
    const int token = output_row / routes;
    const int route = output_row - token * routes;
    const int expert = route_ids[static_cast<int64_t>(token) * routes + route];
    if (expert < 0 ||
            (mapped_experts ? expert >= global_experts
                            : expert >= local_experts)) return;
    const int local = mapped_experts ? expert_local[expert] : expert;
    if (local < 0 || local >= local_experts) return;
    const int row = local * out_per_expert + out;
    const int input_row = routed_input ? output_row : token;
    const __half * source = input + static_cast<int64_t>(input_row) * width;
    const int16_t * first_row = first + static_cast<int64_t>(row) * blocks_per_row;
    const int16_t * second_row = second + static_cast<int64_t>(row) * blocks_per_row;
    float value = 0.0f;
    for (int block = lane; block < blocks_per_row; block += kWarpSize) {
        value += residual_dot(
            source,
            dictionary,
            first_row[block],
            position_bits,
            block_vectors,
            block,
            width);
        const int record = second_row[block];
        if (record >= 0) {
            value += residual_dot(
                source,
                dictionary,
                record,
                position_bits,
                block_vectors,
                block,
                width);
        }
    }
    value = warp_sum(value);
    if (lane == 0) {
        __half * destination = output + logical;
        *destination = __float2half(__half2float(*destination) + value);
    }
}

__global__ void nepq_sparse_residual_backward_input_kernel(
    const __half * dictionary,
    const int16_t * first,
    const int16_t * second,
    const __half * output_gradient,
    __half * input_gradient,
    int input_rows,
    int rows,
    int width,
    int blocks_per_row,
    int position_bits,
    int block_vectors) {
    extern __shared__ float warp_gradients[];
    const int input_row = static_cast<int>(blockIdx.y);
    const int residual_block = static_cast<int>(blockIdx.x);
    const int values_per_block = block_vectors * 8;
    const int warp_count = static_cast<int>(blockDim.x) / kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int row_group = lane / 8;
    const int component = lane & 7;
    for (int index = static_cast<int>(threadIdx.x);
         index < warp_count * values_per_block;
         index += static_cast<int>(blockDim.x)) {
        warp_gradients[index] = 0.0f;
    }
    __syncthreads();

    const int position_mask = (1 << position_bits) - 1;
    for (int row_base = warp * 4;
         row_base < rows;
         row_base += warp_count * 4) {
        const int row = row_base + row_group;
        float gradient = 0.0f;
        int first_record = -1;
        int second_record = -1;
        if (component == 0 && row < rows) {
            gradient = __half2float(output_gradient[
                static_cast<int64_t>(input_row) * rows + row]);
            const int64_t record_index =
                static_cast<int64_t>(row) * blocks_per_row + residual_block;
            first_record = first[record_index];
            second_record = second[record_index];
        }
        const int source_lane = row_group * 8;
        gradient = __shfl_sync(0xffffffffu, gradient, source_lane);
        first_record = __shfl_sync(0xffffffffu, first_record, source_lane);
        second_record = __shfl_sync(0xffffffffu, second_record, source_lane);
        const int records[2] = {first_record, second_record};
#pragma unroll
        for (int stream = 0; stream < 2; ++stream) {
            const int record = records[stream];
            if (record < 0) continue;
            const int position = record & position_mask;
            const int dictionary_id = record >> position_bits;
            atomicAdd(
                warp_gradients + warp * values_per_block +
                    position * 8 + component,
                gradient * __half2float(dictionary[
                    static_cast<int64_t>(dictionary_id) * 8 + component]));
        }
    }
    __syncthreads();

    for (int index = static_cast<int>(threadIdx.x);
         index < values_per_block;
         index += static_cast<int>(blockDim.x)) {
        const int column = residual_block * values_per_block + index;
        if (column < width) {
            float value = 0.0f;
            for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
                value += warp_gradients[
                    source_warp * values_per_block + index];
            }
            const int64_t logical =
                static_cast<int64_t>(input_row) * width + column;
            input_gradient[logical] = __float2half(
                __half2float(input_gradient[logical]) + value);
        }
    }
}

__global__ void nepq_sparse_residual_backward_transpose_kernel(
    const __half * dictionary,
    const int32_t * transpose_offsets,
    const int32_t * transpose_rows,
    const int16_t * transpose_dictionary,
    const __half * output_gradient,
    const int8_t * rotation_signs,
    __half * input_gradient,
    int input_rows,
    int rows,
    int width,
    int vectors,
    bool fuse_rotation) {
    const int64_t thread =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t group = thread / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int component = static_cast<int>(thread & 7);
    const int record_split = lane / 8;
    const int64_t total_groups = static_cast<int64_t>(input_rows) * vectors;
    if (group >= total_groups) return;

    const int input_row = static_cast<int>(group / vectors);
    const int vector = static_cast<int>(group -
        static_cast<int64_t>(input_row) * vectors);
    const unsigned active = __activemask();
    int begin = lane == 0 ? transpose_offsets[vector] : 0;
    int end = lane == 0 ? transpose_offsets[vector + 1] : 0;
    begin = __shfl_sync(active, begin, 0);
    end = __shfl_sync(active, end, 0);

    float value = 0.0f;
    for (int record = begin + record_split;
         record < end;
         record += 4) {
        const unsigned record_active = __activemask();
        const int source_lane = record_split * 8;
        int output = component == 0 ? transpose_rows[record] : 0;
        int dictionary_id = component == 0
            ? static_cast<int>(transpose_dictionary[record])
            : 0;
        output = __shfl_sync(record_active, output, source_lane);
        dictionary_id = __shfl_sync(
            record_active, dictionary_id, source_lane);
        float gradient = component == 0
            ? __half2float(output_gradient[
                static_cast<int64_t>(input_row) * rows + output])
            : 0.0f;
        gradient = __shfl_sync(record_active, gradient, source_lane);
        value = fmaf(
            gradient,
            __half2float(dictionary[
                static_cast<int64_t>(dictionary_id) * 8 + component]),
            value);
    }

    const unsigned reduce_active = __activemask();
    const float split1 = __shfl_sync(
        reduce_active, value, component + 8);
    const float split2 = __shfl_sync(
        reduce_active, value, component + 16);
    const float split3 = __shfl_sync(
        reduce_active, value, component + 24);
    value += split1 + split2 + split3;
    if (record_split != 0) return;

    const int column = vector * 8 + component;
    const int64_t logical =
        static_cast<int64_t>(input_row) * width + column;
    value += __half2float(input_gradient[logical]);
    if (fuse_rotation) {
        const unsigned rotate_active = __activemask();
#pragma unroll
        for (int stride = 1; stride < 8; stride <<= 1) {
            const float other = __shfl_xor_sync(
                rotate_active, value, stride, 8);
            value = (component & stride) ? other - value : value + other;
        }
        value *= 0.3535533905932738f *
            static_cast<float>(rotation_signs[column]);
    }
    input_gradient[logical] = __float2half(value);
}

void validate_residual(
    const mfq_tensor_backend::Tensor & dictionary,
    const mfq_tensor_backend::Tensor & first,
    const mfq_tensor_backend::Tensor & second,
    int64_t width,
    int64_t position_bits,
    int64_t block_vectors) {
    MFQ_RUNTIME_CHECK(
        dictionary.is_cuda() && dictionary.scalar_type() == mfq_tensor_backend::kFloat16 &&
        dictionary.is_contiguous() && dictionary.dim() == 2 &&
        dictionary.size(0) == 1024 && dictionary.size(1) == 8,
        "NEPQ-A dictionary must be CUDA contiguous fp16 [1024,8]");
    MFQ_RUNTIME_CHECK(
        first.is_cuda() && first.scalar_type() == mfq_tensor_backend::kInt16 &&
        first.is_contiguous() && first.dim() == 2,
        "NEPQ-A first records must be CUDA contiguous int16 rank-2");
    MFQ_RUNTIME_CHECK(
        second.is_cuda() && second.scalar_type() == mfq_tensor_backend::kInt16 &&
        second.is_contiguous() && second.sizes() == first.sizes(),
        "NEPQ-A second records must be int16 and match the first-record layout");
    MFQ_RUNTIME_CHECK(
        dictionary.device() == first.device() &&
        dictionary.device() == second.device(),
        "NEPQ-A residual tensors must share one CUDA device");
    MFQ_RUNTIME_CHECK(width > 0 && width % 8 == 0, "NEPQ-A width must be divisible by 8");
    MFQ_RUNTIME_CHECK(
        position_bits >= 1 && position_bits <= 5 &&
        block_vectors >= 2 && block_vectors <= 32,
        "invalid NEPQ-A residual profile");
    const int64_t expected_blocks = (width / 8 + block_vectors - 1) / block_vectors;
    MFQ_RUNTIME_CHECK(first.size(1) == expected_blocks, "NEPQ-A block count mismatch");
}

}  // namespace


mfq_tensor_backend::Tensor nepq_sparse_residual_matmul_cuda(
    mfq_tensor_backend::Tensor dictionary,
    mfq_tensor_backend::Tensor first,
    mfq_tensor_backend::Tensor second,
    mfq_tensor_backend::Tensor input,
    int64_t position_bits,
    int64_t block_vectors,
    mfq_tensor_backend::Tensor output) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.is_contiguous() && input.dim() == 2,
        "NEPQ-A input must be CUDA contiguous fp16 rank-2");
    MFQ_RUNTIME_CHECK(
        output.is_cuda() && output.scalar_type() == mfq_tensor_backend::kFloat16 &&
        output.is_contiguous() && output.dim() == 2 &&
        output.size(0) == input.size(0) && output.size(1) == first.size(0),
        "NEPQ-A output must be CUDA contiguous fp16 [M,rows]");
    validate_residual(
        dictionary, first, second, input.size(1), position_bits, block_vectors);
    MFQ_RUNTIME_CHECK(
        dictionary.device() == input.device() && input.device() == output.device(),
        "NEPQ-A matmul tensors must share one CUDA device");
    MfqCudaGuard guard(input.device());
    const int64_t total = input.size(0) * first.size(0);
    cudaStream_t stream = mfq_current_cuda_stream();
    nepq_sparse_residual_matmul_kernel<<<
        (total + kWarpsPerBlock - 1) / kWarpsPerBlock,
        kWarpSize * kWarpsPerBlock,
        0,
        stream>>>(
        reinterpret_cast<const __half *>(dictionary.data_ptr<mfq_half>()),
        first.data_ptr<int16_t>(),
        second.data_ptr<int16_t>(),
        reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
        static_cast<int>(first.size(0)),
        static_cast<int>(input.size(1)),
        static_cast<int>(input.size(0)),
        static_cast<int>(first.size(1)),
        static_cast<int>(position_bits),
        static_cast<int>(block_vectors));
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ-A residual matmul kernel launch failed");
    return output;
}


mfq_tensor_backend::Tensor nepq_sparse_residual_dequant_cuda(
    mfq_tensor_backend::Tensor dictionary,
    mfq_tensor_backend::Tensor first,
    mfq_tensor_backend::Tensor second,
    int64_t position_bits,
    int64_t block_vectors,
    mfq_tensor_backend::Tensor weight) {
    MFQ_RUNTIME_CHECK(
        weight.is_cuda() && weight.scalar_type() == mfq_tensor_backend::kFloat16 &&
        weight.is_contiguous() && weight.dim() == 2 &&
        weight.size(0) == first.size(0),
        "NEPQ-A weight must be CUDA contiguous fp16 [rows,K]");
    validate_residual(
        dictionary, first, second, weight.size(1), position_bits, block_vectors);
    MFQ_RUNTIME_CHECK(
        dictionary.device() == weight.device(),
        "NEPQ-A dequant tensors must share one CUDA device");
    MfqCudaGuard guard(weight.device());
    const int64_t total = first.numel();
    cudaStream_t stream = mfq_current_cuda_stream();
    nepq_sparse_residual_dequant_kernel<<<
        (total + 255) / 256,
        256,
        0,
        stream>>>(
        reinterpret_cast<const __half *>(dictionary.data_ptr<mfq_half>()),
        first.data_ptr<int16_t>(),
        second.data_ptr<int16_t>(),
        reinterpret_cast<__half *>(weight.data_ptr<mfq_half>()),
        static_cast<int>(weight.size(1)),
        static_cast<int>(first.size(1)),
        static_cast<int>(position_bits),
        static_cast<int>(block_vectors),
        total);
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ-A residual dequant kernel launch failed");
    return weight;
}


mfq_tensor_backend::Tensor nepq_sparse_residual_grouped_cuda(
    mfq_tensor_backend::Tensor dictionary,
    mfq_tensor_backend::Tensor first,
    mfq_tensor_backend::Tensor second,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor route_ids,
    mfq_tensor_backend::Tensor expert_local,
    int64_t out_per_expert,
    int64_t position_bits,
    int64_t block_vectors,
    mfq_tensor_backend::Tensor output) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.is_contiguous() && (input.dim() == 2 || input.dim() == 3),
        "NEPQ-A routed input must be CUDA contiguous fp16 rank-2 or rank-3");
    MFQ_RUNTIME_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == mfq_tensor_backend::kInt32 &&
        route_ids.is_contiguous() && route_ids.dim() == 2,
        "NEPQ-A route IDs must be CUDA contiguous int32 rank-2");
    MFQ_RUNTIME_CHECK(
        expert_local.is_cuda() && expert_local.scalar_type() == mfq_tensor_backend::kInt32 &&
        expert_local.is_contiguous() && expert_local.dim() == 1,
        "NEPQ-A expert map must be CUDA contiguous int32 rank-1");
    MFQ_RUNTIME_CHECK(
        output.is_cuda() && output.scalar_type() == mfq_tensor_backend::kFloat16 &&
        output.is_contiguous() && output.dim() == 3 &&
        output.size(0) == route_ids.size(0) &&
        output.size(1) == route_ids.size(1) &&
        output.size(2) == out_per_expert,
        "NEPQ-A grouped output shape mismatch");
    const int64_t width = input.size(-1);
    validate_residual(dictionary, first, second, width, position_bits, block_vectors);
    MFQ_RUNTIME_CHECK(
        dictionary.device() == input.device() &&
        input.device() == route_ids.device() &&
        input.device() == expert_local.device() &&
        input.device() == output.device(),
        "NEPQ-A grouped tensors must share one CUDA device");
    MFQ_RUNTIME_CHECK(
        first.size(0) % out_per_expert == 0,
        "NEPQ-A expert row count is not divisible by output width");
    MFQ_RUNTIME_CHECK(
        input.size(0) == route_ids.size(0) &&
        (input.dim() == 2 || input.size(1) == route_ids.size(1)),
        "NEPQ-A routed input leading dimensions mismatch");
    MfqCudaGuard guard(input.device());
    const int64_t total = route_ids.numel() * out_per_expert;
    cudaStream_t stream = mfq_current_cuda_stream();
    nepq_sparse_residual_grouped_kernel<<<
        (total + kWarpsPerBlock - 1) / kWarpsPerBlock,
        kWarpSize * kWarpsPerBlock,
        0,
        stream>>>(
        reinterpret_cast<const __half *>(dictionary.data_ptr<mfq_half>()),
        first.data_ptr<int16_t>(),
        second.data_ptr<int16_t>(),
        reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
        route_ids.data_ptr<int32_t>(),
        expert_local.data_ptr<int32_t>(),
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
        static_cast<int>(route_ids.size(0)),
        static_cast<int>(route_ids.size(1)),
        static_cast<int>(out_per_expert),
        static_cast<int>(width),
        static_cast<int>(first.size(1)),
        static_cast<int>(position_bits),
        static_cast<int>(block_vectors),
        input.dim() == 3,
        expert_local.numel() != 0,
        static_cast<int>(expert_local.numel()),
        static_cast<int>(first.size(0) / out_per_expert));
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ-A residual grouped kernel launch failed");
    return output;
}


mfq_tensor_backend::Tensor nepq_sparse_residual_backward_input_cuda(
    mfq_tensor_backend::Tensor dictionary,
    mfq_tensor_backend::Tensor first,
    mfq_tensor_backend::Tensor second,
    mfq_tensor_backend::Tensor output_gradient,
    int64_t position_bits,
    int64_t block_vectors,
    mfq_tensor_backend::Tensor input_gradient) {
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 &&
        output_gradient.is_contiguous() && output_gradient.dim() == 2 &&
        output_gradient.size(1) == first.size(0),
        "NEPQ-A backward output gradient must be CUDA fp16 [M,rows]");
    MFQ_RUNTIME_CHECK(
        input_gradient.is_cuda() &&
        input_gradient.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input_gradient.is_contiguous() && input_gradient.dim() == 2 &&
        input_gradient.size(0) == output_gradient.size(0),
        "NEPQ-A backward input gradient must be CUDA fp16 [M,K]");
    validate_residual(
        dictionary, first, second, input_gradient.size(1),
        position_bits, block_vectors);
    MFQ_RUNTIME_CHECK(
        dictionary.device() == output_gradient.device() &&
        output_gradient.device() == input_gradient.device(),
        "NEPQ-A backward tensors must share one CUDA device");
    MfqCudaGuard guard(input_gradient.device());
    if (input_gradient.numel() == 0) {
        return input_gradient;
    }
    const int threads = output_gradient.size(0) <= 8 ? 1024 : 256;
    const dim3 blocks(
        static_cast<unsigned>(first.size(1)),
        static_cast<unsigned>(output_gradient.size(0)));
    const size_t shared_bytes =
        static_cast<size_t>(threads / kWarpSize) * block_vectors * 8 * sizeof(float);
    nepq_sparse_residual_backward_input_kernel<<<
        blocks, threads, shared_bytes, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const __half *>(dictionary.data_ptr<mfq_half>()),
        first.data_ptr<int16_t>(), second.data_ptr<int16_t>(),
        reinterpret_cast<const __half *>(
            output_gradient.data_ptr<mfq_half>()),
        reinterpret_cast<__half *>(input_gradient.data_ptr<mfq_half>()),
        static_cast<int>(output_gradient.size(0)),
        static_cast<int>(first.size(0)),
        static_cast<int>(input_gradient.size(1)),
        static_cast<int>(first.size(1)),
        static_cast<int>(position_bits),
        static_cast<int>(block_vectors));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return input_gradient;
}


mfq_tensor_backend::Tensor nepq_sparse_residual_backward_transpose_cuda(
    mfq_tensor_backend::Tensor dictionary,
    mfq_tensor_backend::Tensor transpose_offsets,
    mfq_tensor_backend::Tensor transpose_rows,
    mfq_tensor_backend::Tensor transpose_dictionary,
    mfq_tensor_backend::Tensor output_gradient,
    mfq_tensor_backend::Tensor rotation_signs,
    bool fuse_rotation,
    mfq_tensor_backend::Tensor input_gradient) {
    MFQ_RUNTIME_CHECK(
        dictionary.is_cuda() &&
        dictionary.scalar_type() == mfq_tensor_backend::kFloat16 &&
        dictionary.is_contiguous() && dictionary.dim() == 2 &&
        dictionary.size(0) == 1024 && dictionary.size(1) == 8,
        "NEPQ-A transpose dictionary must be CUDA fp16 [1024,8]");
    MFQ_RUNTIME_CHECK(
        transpose_offsets.is_cuda() &&
        transpose_offsets.scalar_type() == mfq_tensor_backend::kInt32 &&
        transpose_offsets.is_contiguous() && transpose_offsets.dim() == 1,
        "NEPQ-A transpose offsets must be CUDA contiguous int32");
    MFQ_RUNTIME_CHECK(
        transpose_rows.is_cuda() &&
        transpose_rows.scalar_type() == mfq_tensor_backend::kInt32 &&
        transpose_rows.is_contiguous() && transpose_rows.dim() == 1 &&
        transpose_dictionary.is_cuda() &&
        transpose_dictionary.scalar_type() == mfq_tensor_backend::kInt16 &&
        transpose_dictionary.is_contiguous() &&
        transpose_dictionary.dim() == 1 &&
        transpose_dictionary.numel() == transpose_rows.numel(),
        "NEPQ-A transpose records must be matching int32/int16 vectors");
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 &&
        output_gradient.is_contiguous() && output_gradient.dim() == 2,
        "NEPQ-A transpose gradient must be CUDA contiguous fp16 [M,rows]");
    MFQ_RUNTIME_CHECK(
        input_gradient.is_cuda() &&
        input_gradient.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input_gradient.is_contiguous() && input_gradient.dim() == 2 &&
        input_gradient.size(0) == output_gradient.size(0) &&
        input_gradient.size(1) % 8 == 0,
        "NEPQ-A transpose output must be CUDA contiguous fp16 [M,K]");
    const int64_t vectors = input_gradient.size(1) / 8;
    MFQ_RUNTIME_CHECK(
        transpose_offsets.numel() == vectors + 1,
        "NEPQ-A transpose offset count mismatch");
    MFQ_RUNTIME_CHECK(
        rotation_signs.is_cuda() &&
        rotation_signs.scalar_type() == mfq_tensor_backend::kInt8 &&
        rotation_signs.is_contiguous() && rotation_signs.dim() == 1 &&
        (!fuse_rotation || rotation_signs.numel() == input_gradient.size(1)),
        "NEPQ-A fused rotation signs must match K");
    MFQ_RUNTIME_CHECK(
        dictionary.device() == transpose_offsets.device() &&
        dictionary.device() == transpose_rows.device() &&
        dictionary.device() == transpose_dictionary.device() &&
        dictionary.device() == output_gradient.device() &&
        dictionary.device() == rotation_signs.device() &&
        dictionary.device() == input_gradient.device(),
        "NEPQ-A transpose tensors must share one CUDA device");
    if (input_gradient.numel() == 0) return input_gradient;

    MfqCudaGuard guard(input_gradient.device());
    constexpr int threads = 256;
    const int64_t total_threads =
        output_gradient.size(0) * vectors * kWarpSize;
    const int blocks = static_cast<int>(
        (total_threads + threads - 1) / threads);
    nepq_sparse_residual_backward_transpose_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const __half *>(dictionary.data_ptr<mfq_half>()),
        transpose_offsets.data_ptr<int32_t>(),
        transpose_rows.data_ptr<int32_t>(),
        transpose_dictionary.data_ptr<int16_t>(),
        reinterpret_cast<const __half *>(
            output_gradient.data_ptr<mfq_half>()),
        rotation_signs.data_ptr<int8_t>(),
        reinterpret_cast<__half *>(input_gradient.data_ptr<mfq_half>()),
        static_cast<int>(output_gradient.size(0)),
        static_cast<int>(output_gradient.size(1)),
        static_cast<int>(input_gradient.size(1)),
        static_cast<int>(vectors),
        fuse_rotation);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return input_gradient;
}
