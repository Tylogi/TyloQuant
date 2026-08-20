#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <optional>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <utility>

#ifdef MFQ_NATIVE_CUDA_RUNTIME

#include "mfq_cuda_context.h"
#include "mfq_native_tensor.h"

namespace mfq_tensor_backend = ::mfq::cuda;
using mfq_half = __half;
using mfq_bfloat16 = __nv_bfloat16;
using MfqBackendError = ::mfq::cuda::Error;
inline constexpr auto mfq_dispatch_half = ::mfq::cuda::ScalarType::float16;
inline constexpr auto mfq_dispatch_bfloat16 = ::mfq::cuda::ScalarType::bfloat16;

inline cudaStream_t mfq_current_cuda_stream() {
    return ::mfq::cuda::current_stream().stream();
}

inline cublasHandle_t mfq_current_cublas_handle() {
    int device = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    auto context = ::mfq::cuda::default_context(device);
    context->blas().set_stream(::mfq::cuda::current_stream(device).stream());
    return context->blas().get();
}

inline void mfq_cuda_synchronize() {
    int device = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(
        ::mfq::cuda::current_stream(device).stream()));
}

template <typename... Values>
[[noreturn]] inline void mfq_runtime_check_failed(Values&&... values) {
    std::ostringstream message;
    (message << ... << std::forward<Values>(values));
    throw std::runtime_error(message.str());
}

#define MFQ_RUNTIME_CHECK(condition, ...) \
    do { \
        if (!(condition)) { \
            ::mfq_runtime_check_failed(__VA_ARGS__); \
        } \
    } while (false)

#define MFQ_CUDA_KERNEL_LAUNCH_CHECK() \
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError())

#define MFQ_CUDA_CHECK(expression) MFQ_NATIVE_CUDA_CHECK(expression)

#define MFQ_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
    ([&]() { \
        switch (TYPE) { \
            case ::mfq::cuda::ScalarType::float64: { using scalar_t = double; return (__VA_ARGS__)(); } \
            case ::mfq::cuda::ScalarType::float32: { using scalar_t = float; return (__VA_ARGS__)(); } \
            case ::mfq::cuda::ScalarType::float16: { using scalar_t = __half; return (__VA_ARGS__)(); } \
            default: ::mfq_runtime_check_failed(NAME, ": unsupported floating dtype"); \
        } \
    })()

#define MFQ_DISPATCH_FLOATING_TYPES_AND2(TYPE1, TYPE2, TYPE, NAME, ...) \
    ([&]() { \
        switch (TYPE) { \
            case ::mfq::cuda::ScalarType::float64: { using scalar_t = double; return (__VA_ARGS__)(); } \
            case ::mfq::cuda::ScalarType::float32: { using scalar_t = float; return (__VA_ARGS__)(); } \
            case ::mfq::cuda::ScalarType::float16: { using scalar_t = __half; return (__VA_ARGS__)(); } \
            case ::mfq::cuda::ScalarType::bfloat16: { using scalar_t = __nv_bfloat16; return (__VA_ARGS__)(); } \
            default: ::mfq_runtime_check_failed(NAME, ": unsupported floating dtype"); \
        } \
    })()

class MfqCudaGuard final {
public:
    explicit MfqCudaGuard(int device) : guard_(device) {}
    explicit MfqCudaGuard(const ::mfq::cuda::Device& device) : guard_(device.index) {}

private:
    ::mfq::cuda::DeviceGuard guard_;
};

using MfqCudaStream = ::mfq::cuda::StreamHandle;
using MfqCudaStreamGuard = ::mfq::cuda::StreamGuard;
using MfqCudaGraph = ::mfq::cuda::Graph;

inline MfqCudaStream mfq_get_current_cuda_stream(int device = -1) {
    return ::mfq::cuda::current_stream(device);
}

inline MfqCudaStream mfq_get_stream_from_pool(
    bool high_priority = false,
    int device = -1) {
    return ::mfq::cuda::stream_from_pool(high_priority, device);
}

inline int mfq_current_cuda_device() {
    int device = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

inline void mfq_cuda_empty_cache() {
    ::mfq::cuda::default_context(mfq_current_cuda_device())->trim();
}

inline bool mfq_cuda_graph_capture_supported() {
    return ::mfq::cuda::default_context(
        mfq_current_cuda_device())->supports_async_allocations();
}

inline void mfq_cuda_record_stream(
    const ::mfq::cuda::Tensor& tensor,
    const MfqCudaStream& stream) {
    tensor.record_stream(reinterpret_cast<std::uintptr_t>(stream.stream()));
}

struct MfqCudaMemoryStats {
    std::uint64_t allocated_bytes = 0;
    std::uint64_t active_bytes = 0;
    std::uint64_t reserved_bytes = 0;
    std::uint64_t inactive_split_bytes = 0;
    std::uint64_t requested_bytes = 0;
    std::uint64_t peak_allocated_bytes = 0;
    std::uint64_t peak_reserved_bytes = 0;
    std::uint64_t allocations = 0;
    std::uint64_t segments = 0;
    std::uint64_t retries = 0;
    std::uint64_t ooms = 0;
};

inline MfqCudaMemoryStats mfq_cuda_memory_stats(int device) {
    ::mfq::cuda::DeviceGuard guard(device);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    MfqCudaMemoryStats stats;
    stats.allocated_bytes = total_bytes - free_bytes;
    stats.active_bytes = stats.allocated_bytes;
    stats.reserved_bytes = stats.allocated_bytes;
    stats.requested_bytes = stats.allocated_bytes;
    stats.peak_allocated_bytes = stats.allocated_bytes;
    stats.peak_reserved_bytes = stats.reserved_bytes;
    return stats;
}

inline std::atomic<int> mfq_host_thread_count{
    static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))};

inline int mfq_get_num_threads() {
    return mfq_host_thread_count.load(std::memory_order_relaxed);
}

inline void mfq_set_num_threads(int count) {
    mfq_host_thread_count.store(std::max(1, count), std::memory_order_relaxed);
}

template <typename Function>
void mfq_parallel_for(
    std::int64_t begin,
    std::int64_t end,
    std::int64_t grain,
    Function&& function) {
    const auto count = std::max<std::int64_t>(0, end - begin);
    const int threads = std::min<int>(
        mfq_get_num_threads(),
        static_cast<int>((count + std::max<std::int64_t>(grain, 1) - 1) /
                         std::max<std::int64_t>(grain, 1)));
    if (threads <= 1) {
        function(begin, end);
        return;
    }
    std::atomic<std::int64_t> next{begin};
    const auto chunk = std::max<std::int64_t>(grain, 1);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (int index = 0; index < threads; ++index) {
        workers.emplace_back([&] {
            while (true) {
                const auto first = next.fetch_add(chunk, std::memory_order_relaxed);
                if (first >= end) break;
                function(first, std::min(end, first + chunk));
            }
        });
    }
    for (auto& worker : workers) worker.join();
}

inline void mfq_disable_tf32_cublas() {
    auto context = ::mfq::cuda::default_context(mfq_current_cuda_device());
    context->blas().set_stream(mfq_current_cuda_stream());
    MFQ_NATIVE_CUDA_CHECK(cublasSetMathMode(
        context->blas().get(), CUBLAS_DEFAULT_MATH));
}

inline void mfq_cuda_manual_seed_all(std::uint64_t seed) {
    ::mfq::cuda::manual_seed(static_cast<std::int64_t>(seed));
}

#else

#include <ATen/Parallel.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace mfq_tensor_backend = ::torch;
using mfq_half = ::at::Half;
using mfq_bfloat16 = ::at::BFloat16;
using MfqBackendError = ::c10::Error;
inline constexpr auto mfq_dispatch_half = ::at::ScalarType::Half;
inline constexpr auto mfq_dispatch_bfloat16 = ::at::ScalarType::BFloat16;

inline cudaStream_t mfq_current_cuda_stream() {
    return ::at::cuda::getCurrentCUDAStream();
}

inline cublasHandle_t mfq_current_cublas_handle() {
    return ::at::cuda::getCurrentCUDABlasHandle();
}

inline void mfq_cuda_synchronize() {
    ::torch::cuda::synchronize();
}

#define MFQ_RUNTIME_CHECK TORCH_CHECK
#define MFQ_CUDA_KERNEL_LAUNCH_CHECK C10_CUDA_KERNEL_LAUNCH_CHECK
#define MFQ_CUDA_CHECK C10_CUDA_CHECK
#define MFQ_DISPATCH_FLOATING_TYPES_AND_HALF AT_DISPATCH_FLOATING_TYPES_AND_HALF
#define MFQ_DISPATCH_FLOATING_TYPES_AND2 AT_DISPATCH_FLOATING_TYPES_AND2
using MfqCudaGuard = ::c10::cuda::CUDAGuard;
using MfqCudaStream = ::at::cuda::CUDAStream;
using MfqCudaStreamGuard = ::c10::cuda::CUDAStreamGuard;
using MfqCudaGraph = ::at::cuda::CUDAGraph;

inline MfqCudaStream mfq_get_current_cuda_stream(int device = -1) {
    return device < 0
        ? ::at::cuda::getCurrentCUDAStream()
        : ::at::cuda::getCurrentCUDAStream(device);
}

inline MfqCudaStream mfq_get_stream_from_pool(
    bool high_priority = false,
    int device = -1) {
    return device < 0
        ? ::at::cuda::getStreamFromPool(high_priority)
        : ::at::cuda::getStreamFromPool(high_priority, device);
}

inline int mfq_current_cuda_device() {
    return ::c10::cuda::current_device();
}

inline void mfq_cuda_empty_cache() {
    ::c10::cuda::CUDACachingAllocator::emptyCache();
}

inline bool mfq_cuda_graph_capture_supported() { return true; }

inline void mfq_cuda_record_stream(
    const ::torch::Tensor& tensor,
    const MfqCudaStream& stream) {
    ::c10::cuda::CUDACachingAllocator::recordStream(
        tensor.storage().data_ptr(), stream);
}

struct MfqCudaMemoryStats {
    std::uint64_t allocated_bytes = 0;
    std::uint64_t active_bytes = 0;
    std::uint64_t reserved_bytes = 0;
    std::uint64_t inactive_split_bytes = 0;
    std::uint64_t requested_bytes = 0;
    std::uint64_t peak_allocated_bytes = 0;
    std::uint64_t peak_reserved_bytes = 0;
    std::uint64_t allocations = 0;
    std::uint64_t segments = 0;
    std::uint64_t retries = 0;
    std::uint64_t ooms = 0;
};

inline MfqCudaMemoryStats mfq_cuda_memory_stats(int device) {
    const auto source = ::c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    constexpr auto aggregate = static_cast<std::size_t>(
        ::c10::CachingDeviceAllocator::StatType::AGGREGATE);
    return {
        static_cast<std::uint64_t>(source.allocated_bytes[aggregate].current),
        static_cast<std::uint64_t>(source.active_bytes[aggregate].current),
        static_cast<std::uint64_t>(source.reserved_bytes[aggregate].current),
        static_cast<std::uint64_t>(source.inactive_split_bytes[aggregate].current),
        static_cast<std::uint64_t>(source.requested_bytes[aggregate].current),
        static_cast<std::uint64_t>(source.allocated_bytes[aggregate].peak),
        static_cast<std::uint64_t>(source.reserved_bytes[aggregate].peak),
        static_cast<std::uint64_t>(source.allocation[aggregate].current),
        static_cast<std::uint64_t>(source.segment[aggregate].current),
        static_cast<std::uint64_t>(source.num_alloc_retries),
        static_cast<std::uint64_t>(source.num_ooms),
    };
}

inline int mfq_get_num_threads() { return ::at::get_num_threads(); }
inline void mfq_set_num_threads(int count) { ::at::set_num_threads(count); }

template <typename Function>
void mfq_parallel_for(
    std::int64_t begin,
    std::int64_t end,
    std::int64_t grain,
    Function&& function) {
    ::at::parallel_for(begin, end, grain, std::forward<Function>(function));
}

inline void mfq_disable_tf32_cublas() {
    ::at::globalContext().setAllowTF32CuBLAS(false);
}


inline void mfq_cuda_manual_seed_all(std::uint64_t seed) {
    ::torch::cuda::manual_seed_all(seed);
}

#endif

template <typename Value>
using MfqOptional = std::optional<Value>;

inline constexpr std::nullopt_t mfq_nullopt = std::nullopt;

inline mfq_tensor_backend::Tensor mfq_linear(
    const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& weight,
    const std::optional<mfq_tensor_backend::Tensor>& bias = std::nullopt) {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    return ::mfq::cuda::linear(input, weight, bias);
#else
    return ::at::linear(input, weight, bias);
#endif
}

inline mfq_tensor_backend::Tensor mfq_linear(
    const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& weight,
    const mfq_tensor_backend::Tensor& bias) {
    return mfq_linear(
        input, weight, std::optional<mfq_tensor_backend::Tensor>(bias));
}

inline mfq_tensor_backend::Tensor mfq_scaled_dot_product_attention(
    const mfq_tensor_backend::Tensor& query,
    const mfq_tensor_backend::Tensor& key,
    const mfq_tensor_backend::Tensor& value,
    const std::optional<mfq_tensor_backend::Tensor>& mask,
    double dropout,
    bool causal,
    const std::optional<double>& scale,
    bool enable_grouped_query_attention) {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    return ::mfq::cuda::scaled_dot_product_attention(
        query, key, value, mask, dropout, causal, scale,
        enable_grouped_query_attention);
#else
    return ::at::scaled_dot_product_attention(
        query, key, value, mask, dropout, causal, scale,
        enable_grouped_query_attention);
#endif
}
