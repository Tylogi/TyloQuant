#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


namespace {

__global__ void nepq_hadamard_input_kernel(
    const __half * input,
    const int8_t * signs,
    __half * output,
    int rows,
    int width,
    int block_size) {
    extern __shared__ float values[];
    const int row = blockIdx.y;
    const int offset = blockIdx.x * block_size;
    if (row >= rows || offset >= width) return;

    for (int index = threadIdx.x; index < block_size; index += blockDim.x) {
        const int column = offset + index;
        values[index] = __half2float(input[static_cast<int64_t>(row) * width + column])
            * static_cast<float>(signs[column]);
    }
    __syncthreads();

    for (int half = 1; half < block_size; half <<= 1) {
        const int span = half << 1;
        for (int butterfly = threadIdx.x;
             butterfly < block_size / 2;
             butterfly += blockDim.x) {
            const int group = butterfly / half;
            const int within = butterfly - group * half;
            const int first = group * span + within;
            const int second = first + half;
            const float a = values[first];
            const float b = values[second];
            values[first] = a + b;
            values[second] = a - b;
        }
        __syncthreads();
    }

    const float normalization = rsqrtf(static_cast<float>(block_size));
    for (int index = threadIdx.x; index < block_size; index += blockDim.x) {
        const int column = offset + index;
        output[static_cast<int64_t>(row) * width + column] =
            __float2half(values[index] * normalization);
    }
}

}  // namespace


torch::Tensor nepq_hadamard_input_cuda(
    torch::Tensor input,
    torch::Tensor signs,
    int64_t block_size) {
    TORCH_CHECK(
        input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
        input.is_contiguous() && input.dim() == 2,
        "NEPQ Hadamard input must be CUDA contiguous fp16 rank-2");
    TORCH_CHECK(
        signs.is_cuda() && signs.scalar_type() == torch::kInt8 &&
        signs.is_contiguous() && signs.dim() == 1,
        "NEPQ Hadamard signs must be CUDA contiguous int8 rank-1");
    TORCH_CHECK(
        block_size >= 2 && block_size <= 2048 &&
        (block_size & (block_size - 1)) == 0,
        "NEPQ Hadamard block must be a power of two in [2,2048]");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    TORCH_CHECK(signs.numel() == width, "NEPQ Hadamard sign width mismatch");
    TORCH_CHECK(width % block_size == 0, "NEPQ Hadamard block must divide K");
    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(input);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nepq_hadamard_input_kernel<<<
        dim3(width / block_size, rows),
        256,
        static_cast<size_t>(block_size) * sizeof(float),
        stream>>>(
        reinterpret_cast<const __half *>(input.data_ptr<at::Half>()),
        signs.data_ptr<int8_t>(),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
        rows,
        width,
        static_cast<int>(block_size));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ Hadamard input kernel launch failed");
    return output;
}
