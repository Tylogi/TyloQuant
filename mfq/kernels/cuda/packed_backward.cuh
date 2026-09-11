#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace mfq_packed_backward {

inline void launch_half_gemm_nn(
        const __half * output_gradient,
        const __half * weight,
        __half * input_gradient,
        int rows,
        int outputs,
        int width,
        cudaStream_t stream) {
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_RUNTIME_CHECK(
        cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
        "packed backward cublasSetStream failed");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major dX[M,K] = dY[M,N] * W[N,K] maps to
    // column-major dX^T[K,M] = W^T[K,N] * dY^T[N,M].
    MFQ_RUNTIME_CHECK(
        cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            width, rows, outputs,
            &alpha,
            weight, CUDA_R_16F, width,
            output_gradient, CUDA_R_16F, outputs,
            &beta,
            input_gradient, CUDA_R_16F, width,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS,
        "packed backward GEMM failed");
}

template <typename Destination, typename Source>
__device__ __forceinline__ Destination cast_value(Source value) {
    return static_cast<Destination>(value);
}

template <>
__device__ __forceinline__ __half cast_value<__half, __nv_bfloat16>(
        __nv_bfloat16 value) {
    return __float2half_rn(__bfloat162float(value));
}

template <>
__device__ __forceinline__ __nv_bfloat16 cast_value<__nv_bfloat16, __half>(
        __half value) {
    return __float2bfloat16_rn(__half2float(value));
}

template <>
__device__ __forceinline__ float cast_value<float, __half>(__half value) {
    return __half2float(value);
}

template <typename Destination, typename Source>
__global__ void cast_kernel(
        const Source * __restrict__ source,
        Destination * __restrict__ destination,
        int64_t count) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < count;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        destination[index] = cast_value<Destination>(source[index]);
    }
}

template <typename Destination, typename Source>
inline void launch_cast(
        const Source * source,
        Destination * destination,
        int64_t count,
        cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (count + threads - 1) / threads, 65535));
    cast_kernel<Destination, Source><<<blocks, threads, 0, stream>>>(
        source, destination, count);
}

static __global__ void split_float_reduce_to_half_kernel(
        const float * __restrict__ partials,
        __half * __restrict__ destination,
        int rows,
        int width,
        int splits) {
    const int64_t total = static_cast<int64_t>(rows) * width;
    for (int64_t logical = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         logical < total;
         logical += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(logical / width);
        const int column = static_cast<int>(logical -
            static_cast<int64_t>(row) * width);
        float accumulator = 0.0f;
        for (int split = 0; split < splits; ++split) {
            accumulator += partials[
                (static_cast<int64_t>(split) * rows + row) * width + column];
        }
        destination[logical] = __float2half_rn(accumulator);
    }
}

inline void launch_split_float_reduce_to_half(
        const float * partials,
        __half * destination,
        int rows,
        int width,
        int splits,
        cudaStream_t stream) {
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(rows) * width;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 65535));
    split_float_reduce_to_half_kernel<<<blocks, threads, 0, stream>>>(
        partials, destination, rows, width, splits);
}

inline void launch_float_gemm_nn(
        const float * output_gradient,
        const float * weight,
        float * input_gradient,
        int rows,
        int outputs,
        int width,
        cudaStream_t stream) {
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_RUNTIME_CHECK(
        cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
        "packed backward cublasSetStream failed");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    MFQ_RUNTIME_CHECK(
        cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            width, rows, outputs,
            &alpha,
            weight, CUDA_R_32F, width,
            output_gradient, CUDA_R_32F, outputs,
            &beta,
            input_gradient, CUDA_R_32F, width,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS,
        "packed backward FP32 GEMM failed");
}

inline void launch_dense_half_weight(
        const mfq_tensor_backend::Tensor & output_gradient,
        const mfq_tensor_backend::Tensor & half_weight,
        mfq_tensor_backend::Tensor & input_gradient,
        int rows,
        int outputs,
        int width,
        cudaStream_t stream) {
    const auto dtype = output_gradient.scalar_type();
    if (dtype == mfq_tensor_backend::kFloat16) {
        launch_half_gemm_nn(
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<mfq_half>()),
            reinterpret_cast<const __half *>(half_weight.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(input_gradient.data_ptr<mfq_half>()),
            rows, outputs, width, stream);
        return;
    }
    if (dtype == mfq_tensor_backend::kBFloat16) {
        auto half_gradient = mfq_tensor_backend::empty(
            {rows, outputs}, half_weight.options());
        auto half_result = mfq_tensor_backend::empty(
            {rows, width}, half_weight.options());
        launch_cast<__half, __nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16 *>(
                output_gradient.data_ptr<mfq_bfloat16>()),
            reinterpret_cast<__half *>(half_gradient.data_ptr<mfq_half>()),
            static_cast<int64_t>(rows) * outputs, stream);
        launch_half_gemm_nn(
            reinterpret_cast<const __half *>(half_gradient.data_ptr<mfq_half>()),
            reinterpret_cast<const __half *>(half_weight.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(half_result.data_ptr<mfq_half>()),
            rows, outputs, width, stream);
        launch_cast<__nv_bfloat16, __half>(
            reinterpret_cast<const __half *>(half_result.data_ptr<mfq_half>()),
            reinterpret_cast<__nv_bfloat16 *>(
                input_gradient.data_ptr<mfq_bfloat16>()),
            static_cast<int64_t>(rows) * width, stream);
        return;
    }
    auto float_weight = mfq_tensor_backend::empty(
        {outputs, width}, half_weight.options().dtype(
            mfq_tensor_backend::kFloat32));
    launch_cast<float, __half>(
        reinterpret_cast<const __half *>(half_weight.data_ptr<mfq_half>()),
        float_weight.data_ptr<float>(),
        static_cast<int64_t>(outputs) * width, stream);
    launch_float_gemm_nn(
        output_gradient.data_ptr<float>(), float_weight.data_ptr<float>(),
        input_gradient.data_ptr<float>(), rows, outputs, width, stream);
}

}  // namespace mfq_packed_backward
