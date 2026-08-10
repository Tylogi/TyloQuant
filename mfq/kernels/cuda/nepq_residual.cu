#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
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

void validate_residual(
    const torch::Tensor & dictionary,
    const torch::Tensor & first,
    const torch::Tensor & second,
    int64_t width,
    int64_t position_bits,
    int64_t block_vectors) {
    TORCH_CHECK(
        dictionary.is_cuda() && dictionary.scalar_type() == torch::kFloat16 &&
        dictionary.is_contiguous() && dictionary.dim() == 2 &&
        dictionary.size(0) == 1024 && dictionary.size(1) == 8,
        "NEPQ-A dictionary must be CUDA contiguous fp16 [1024,8]");
    TORCH_CHECK(
        first.is_cuda() && first.scalar_type() == torch::kInt16 &&
        first.is_contiguous() && first.dim() == 2,
        "NEPQ-A first records must be CUDA contiguous int16 rank-2");
    TORCH_CHECK(
        second.is_cuda() && second.scalar_type() == torch::kInt16 &&
        second.is_contiguous() && second.sizes() == first.sizes(),
        "NEPQ-A second records must be int16 and match the first-record layout");
    TORCH_CHECK(
        dictionary.device() == first.device() &&
        dictionary.device() == second.device(),
        "NEPQ-A residual tensors must share one CUDA device");
    TORCH_CHECK(width > 0 && width % 8 == 0, "NEPQ-A width must be divisible by 8");
    TORCH_CHECK(
        position_bits >= 1 && position_bits <= 5 &&
        block_vectors >= 2 && block_vectors <= 32,
        "invalid NEPQ-A residual profile");
    const int64_t expected_blocks = (width / 8 + block_vectors - 1) / block_vectors;
    TORCH_CHECK(first.size(1) == expected_blocks, "NEPQ-A block count mismatch");
}

}  // namespace


torch::Tensor nepq_sparse_residual_matmul_cuda(
    torch::Tensor dictionary,
    torch::Tensor first,
    torch::Tensor second,
    torch::Tensor input,
    int64_t position_bits,
    int64_t block_vectors,
    torch::Tensor output) {
    TORCH_CHECK(
        input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
        input.is_contiguous() && input.dim() == 2,
        "NEPQ-A input must be CUDA contiguous fp16 rank-2");
    TORCH_CHECK(
        output.is_cuda() && output.scalar_type() == torch::kFloat16 &&
        output.is_contiguous() && output.dim() == 2 &&
        output.size(0) == input.size(0) && output.size(1) == first.size(0),
        "NEPQ-A output must be CUDA contiguous fp16 [M,rows]");
    validate_residual(
        dictionary, first, second, input.size(1), position_bits, block_vectors);
    TORCH_CHECK(
        dictionary.device() == input.device() && input.device() == output.device(),
        "NEPQ-A matmul tensors must share one CUDA device");
    c10::cuda::CUDAGuard guard(input.device());
    const int64_t total = input.size(0) * first.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nepq_sparse_residual_matmul_kernel<<<
        (total + kWarpsPerBlock - 1) / kWarpsPerBlock,
        kWarpSize * kWarpsPerBlock,
        0,
        stream>>>(
        reinterpret_cast<const __half *>(dictionary.data_ptr<at::Half>()),
        first.data_ptr<int16_t>(),
        second.data_ptr<int16_t>(),
        reinterpret_cast<const __half *>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
        static_cast<int>(first.size(0)),
        static_cast<int>(input.size(1)),
        static_cast<int>(input.size(0)),
        static_cast<int>(first.size(1)),
        static_cast<int>(position_bits),
        static_cast<int>(block_vectors));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ-A residual matmul kernel launch failed");
    return output;
}


torch::Tensor nepq_sparse_residual_dequant_cuda(
    torch::Tensor dictionary,
    torch::Tensor first,
    torch::Tensor second,
    int64_t position_bits,
    int64_t block_vectors,
    torch::Tensor weight) {
    TORCH_CHECK(
        weight.is_cuda() && weight.scalar_type() == torch::kFloat16 &&
        weight.is_contiguous() && weight.dim() == 2 &&
        weight.size(0) == first.size(0),
        "NEPQ-A weight must be CUDA contiguous fp16 [rows,K]");
    validate_residual(
        dictionary, first, second, weight.size(1), position_bits, block_vectors);
    TORCH_CHECK(
        dictionary.device() == weight.device(),
        "NEPQ-A dequant tensors must share one CUDA device");
    c10::cuda::CUDAGuard guard(weight.device());
    const int64_t total = first.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nepq_sparse_residual_dequant_kernel<<<
        (total + 255) / 256,
        256,
        0,
        stream>>>(
        reinterpret_cast<const __half *>(dictionary.data_ptr<at::Half>()),
        first.data_ptr<int16_t>(),
        second.data_ptr<int16_t>(),
        reinterpret_cast<__half *>(weight.data_ptr<at::Half>()),
        static_cast<int>(weight.size(1)),
        static_cast<int>(first.size(1)),
        static_cast<int>(position_bits),
        static_cast<int>(block_vectors),
        total);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ-A residual dequant kernel launch failed");
    return weight;
}


torch::Tensor nepq_sparse_residual_grouped_cuda(
    torch::Tensor dictionary,
    torch::Tensor first,
    torch::Tensor second,
    torch::Tensor input,
    torch::Tensor route_ids,
    torch::Tensor expert_local,
    int64_t out_per_expert,
    int64_t position_bits,
    int64_t block_vectors,
    torch::Tensor output) {
    TORCH_CHECK(
        input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
        input.is_contiguous() && (input.dim() == 2 || input.dim() == 3),
        "NEPQ-A routed input must be CUDA contiguous fp16 rank-2 or rank-3");
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == torch::kInt32 &&
        route_ids.is_contiguous() && route_ids.dim() == 2,
        "NEPQ-A route IDs must be CUDA contiguous int32 rank-2");
    TORCH_CHECK(
        expert_local.is_cuda() && expert_local.scalar_type() == torch::kInt32 &&
        expert_local.is_contiguous() && expert_local.dim() == 1,
        "NEPQ-A expert map must be CUDA contiguous int32 rank-1");
    TORCH_CHECK(
        output.is_cuda() && output.scalar_type() == torch::kFloat16 &&
        output.is_contiguous() && output.dim() == 3 &&
        output.size(0) == route_ids.size(0) &&
        output.size(1) == route_ids.size(1) &&
        output.size(2) == out_per_expert,
        "NEPQ-A grouped output shape mismatch");
    const int64_t width = input.size(-1);
    validate_residual(dictionary, first, second, width, position_bits, block_vectors);
    TORCH_CHECK(
        dictionary.device() == input.device() &&
        input.device() == route_ids.device() &&
        input.device() == expert_local.device() &&
        input.device() == output.device(),
        "NEPQ-A grouped tensors must share one CUDA device");
    TORCH_CHECK(
        first.size(0) % out_per_expert == 0,
        "NEPQ-A expert row count is not divisible by output width");
    TORCH_CHECK(
        input.size(0) == route_ids.size(0) &&
        (input.dim() == 2 || input.size(1) == route_ids.size(1)),
        "NEPQ-A routed input leading dimensions mismatch");
    c10::cuda::CUDAGuard guard(input.device());
    const int64_t total = route_ids.numel() * out_per_expert;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nepq_sparse_residual_grouped_kernel<<<
        (total + kWarpsPerBlock - 1) / kWarpsPerBlock,
        kWarpSize * kWarpsPerBlock,
        0,
        stream>>>(
        reinterpret_cast<const __half *>(dictionary.data_ptr<at::Half>()),
        first.data_ptr<int16_t>(),
        second.data_ptr<int16_t>(),
        reinterpret_cast<const __half *>(input.data_ptr<at::Half>()),
        route_ids.data_ptr<int32_t>(),
        expert_local.data_ptr<int32_t>(),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
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
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ-A residual grouped kernel launch failed");
    return output;
}
