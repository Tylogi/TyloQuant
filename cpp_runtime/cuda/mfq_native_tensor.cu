#include "mfq_native_tensor.h"

#include "mfq_cuda_context.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace mfq::cuda {
namespace {

std::shared_ptr<TensorStorage> allocate_cuda_storage(
    std::size_t bytes,
    const Device& device) {
    auto context = default_context(device.index);
    const auto allocation_stream = current_stream(device.index);
    void* pointer = context->allocate(bytes, allocation_stream.stream());
    auto owner = std::shared_ptr<void>(pointer, [context, allocation_stream, bytes](void* value) {
        context->release(value, bytes, allocation_stream.stream());
    });
    auto storage = std::make_shared<TensorStorage>();
    storage->owner = std::move(owner);
    storage->base = pointer;
    storage->bytes = bytes;
    storage->device = device;
    const auto allocation_stream_value = allocation_stream.stream();
    storage->record_stream = [allocation_stream_value](std::uintptr_t raw_stream) {
        const auto usage_stream = reinterpret_cast<cudaStream_t>(raw_stream);
        if (usage_stream == nullptr || usage_stream == allocation_stream_value) return;
        cudaEvent_t event = nullptr;
        MFQ_NATIVE_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        try {
            MFQ_NATIVE_CUDA_CHECK(cudaEventRecord(event, usage_stream));
            MFQ_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(allocation_stream_value, event, 0));
        } catch (...) {
            (void)cudaEventDestroy(event);
            throw;
        }
        (void)cudaEventDestroy(event);
    };
    return storage;
}

std::shared_ptr<TensorStorage> allocate_pinned_storage(std::size_t bytes) {
    void* pointer = nullptr;
    if (bytes != 0) {
        MFQ_NATIVE_CUDA_CHECK(cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault));
    }
    auto owner = std::shared_ptr<void>(pointer, [](void* value) {
        if (value != nullptr) {
            (void)cudaFreeHost(value);
        }
    });
    auto storage = std::make_shared<TensorStorage>();
    storage->owner = std::move(owner);
    storage->base = pointer;
    storage->bytes = bytes;
    storage->device = Device{DeviceType::cpu, 0};
    storage->pinned = true;
    return storage;
}

__device__ std::int64_t strided_offset(const TensorView& view, std::int64_t linear) {
    std::int64_t offset = 0;
    for (std::size_t reverse = view.rank; reverse > 0; --reverse) {
        const auto dimension = reverse - 1;
        const auto coordinate = linear % view.sizes[dimension];
        linear /= view.sizes[dimension];
        offset += coordinate * view.strides[dimension];
    }
    return offset;
}

template <typename Value>
__device__ double numeric_value(Value value) {
    if constexpr (std::is_same_v<Value, __half>) {
        return static_cast<double>(__half2float(value));
    } else if constexpr (std::is_same_v<Value, __nv_bfloat16>) {
        return static_cast<double>(__bfloat162float(value));
    } else {
        return static_cast<double>(value);
    }
}

template <typename Destination, typename Source>
__device__ Destination convert_value(Source value) {
    const auto promoted = numeric_value(value);
    if constexpr (std::is_same_v<Destination, __half>) {
        return __float2half_rn(static_cast<float>(promoted));
    } else if constexpr (std::is_same_v<Destination, __nv_bfloat16>) {
        return __float2bfloat16_rn(static_cast<float>(promoted));
    } else {
        return static_cast<Destination>(promoted);
    }
}

template <typename Destination, typename Source>
__global__ void convert_strided_kernel(
    TensorView destination,
    TensorView source,
    std::int64_t elements) {
    const auto* input = static_cast<const Source*>(source.data);
    auto* output = static_cast<Destination*>(destination.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const auto source_index = strided_offset(source, linear);
        const auto destination_index = strided_offset(destination, linear);
        output[destination_index] = convert_value<Destination>(input[source_index]);
    }
}

__global__ void convert_contiguous_bf16_to_f16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __half* __restrict__ output,
    std::int64_t elements) {
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        output[linear] = __float2half_rn(__bfloat162float(input[linear]));
    }
}

__global__ void materialize_bf16_head_to_token_d128_kernel(
    const uint4* source,
    uint4* destination,
    std::int64_t tokens,
    std::int64_t heads) {
    constexpr std::int64_t depth_packs = 16;
    const auto head = static_cast<std::int64_t>(blockIdx.x);
    const auto token =
        static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const auto batch_index = static_cast<std::int64_t>(blockIdx.z);
    const auto depth_pack = static_cast<std::int64_t>(threadIdx.x);
    if (token >= tokens) return;
    const auto source_index =
        ((batch_index * heads + head) * tokens + token) * depth_packs +
        depth_pack;
    const auto destination_index =
        ((batch_index * tokens + token) * heads + head) * depth_packs +
        depth_pack;
    destination[destination_index] = source[source_index];
}

template <typename Destination>
__global__ void fill_kernel(TensorView destination, double value, std::int64_t elements) {
    auto* output = static_cast<Destination*>(destination.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        output[strided_offset(destination, linear)] = static_cast<Destination>(value);
    }
}

template <int Threads>
__global__ void row_mean_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    std::int64_t rows,
    std::int64_t columns) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    float value = 0.0f;
    for (std::int64_t column = threadIdx.x; column < columns; column += Threads) {
        value += input[row * columns + column];
    }
    __shared__ float partial[Threads];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int offset = Threads / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[row] = partial[0] / static_cast<float>(columns);
    }
}

template <>
__global__ void fill_kernel<__half>(
    TensorView destination,
    double value,
    std::int64_t elements) {
    auto* output = static_cast<__half*>(destination.data);
    const auto converted = __float2half_rn(static_cast<float>(value));
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        output[strided_offset(destination, linear)] = converted;
    }
}

template <>
__global__ void fill_kernel<__nv_bfloat16>(
    TensorView destination,
    double value,
    std::int64_t elements) {
    auto* output = static_cast<__nv_bfloat16*>(destination.data);
    const auto converted = __float2bfloat16_rn(static_cast<float>(value));
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        output[strided_offset(destination, linear)] = converted;
    }
}

template <typename Destination, typename Source>
void launch_convert(
    Tensor& destination,
    const Tensor& source,
    cudaStream_t stream) {
    if (source.numel() == 0) {
        return;
    }
    constexpr int threads = 256;
    const auto blocks = static_cast<int>(std::min<std::int64_t>(
        4096, (source.numel() + threads - 1) / threads));
    if constexpr (
            std::is_same_v<Destination, __half> &&
            std::is_same_v<Source, __nv_bfloat16>) {
        const char* disabled =
            std::getenv("MFQ_DISABLE_NATIVE_CONTIGUOUS_BF16_TO_F16");
        if (source.is_contiguous() && destination.is_contiguous() &&
                (disabled == nullptr || disabled[0] != '1')) {
            convert_contiguous_bf16_to_f16_kernel<<<
                blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(source.data_ptr()),
                reinterpret_cast<__half*>(destination.data_ptr()),
                source.numel());
            MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    if constexpr (
            std::is_same_v<Destination, __nv_bfloat16> &&
            std::is_same_v<Source, __nv_bfloat16>) {
        const auto batch = source.dim() == 4 ? source.size(0) : 0;
        const auto tokens = source.dim() == 4 ? source.size(1) : 0;
        const auto heads = source.dim() == 4 ? source.size(2) : 0;
        const auto depth = source.dim() == 4 ? source.size(3) : 0;
        const bool head_to_token_layout =
            source.dim() == 4 && destination.dim() == 4 &&
            destination.size(0) == batch && destination.size(1) == tokens &&
            destination.size(2) == heads && destination.size(3) == depth &&
            destination.is_contiguous() && batch > 0 && tokens > 0 &&
            heads > 0 && batch <= 65535 && heads <= 2147483647 &&
            (tokens + 7) / 8 <= 65535 && depth == 128 &&
            source.stride(3) == 1 && source.stride(1) == depth &&
            source.stride(2) == tokens * depth &&
            source.stride(0) == heads * tokens * depth &&
            reinterpret_cast<std::uintptr_t>(source.data_ptr()) %
                    alignof(uint4) == 0 &&
            reinterpret_cast<std::uintptr_t>(destination.data_ptr()) %
                    alignof(uint4) == 0;
        if (head_to_token_layout) {
            const dim3 block(16, 8);
            const dim3 grid(
                static_cast<unsigned int>(heads),
                static_cast<unsigned int>((tokens + block.y - 1) / block.y),
                static_cast<unsigned int>(batch));
            materialize_bf16_head_to_token_d128_kernel<<<
                grid, block, 0, stream>>>(
                reinterpret_cast<const uint4*>(source.data_ptr()),
                reinterpret_cast<uint4*>(destination.data_ptr()),
                tokens, heads);
            MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    convert_strided_kernel<Destination, Source><<<blocks, threads, 0, stream>>>(
        destination.view_descriptor(), source.view_descriptor(), source.numel());
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
}

template <typename Destination>
void dispatch_source(
    Tensor& destination,
    const Tensor& source,
    cudaStream_t stream) {
    switch (source.scalar_type()) {
        case ScalarType::boolean: launch_convert<Destination, bool>(destination, source, stream); break;
        case ScalarType::uint8: launch_convert<Destination, std::uint8_t>(destination, source, stream); break;
        case ScalarType::int8: launch_convert<Destination, std::int8_t>(destination, source, stream); break;
        case ScalarType::int16: launch_convert<Destination, std::int16_t>(destination, source, stream); break;
        case ScalarType::int32: launch_convert<Destination, std::int32_t>(destination, source, stream); break;
        case ScalarType::int64: launch_convert<Destination, std::int64_t>(destination, source, stream); break;
        case ScalarType::float16: launch_convert<Destination, __half>(destination, source, stream); break;
        case ScalarType::bfloat16: launch_convert<Destination, __nv_bfloat16>(destination, source, stream); break;
        case ScalarType::float32: launch_convert<Destination, float>(destination, source, stream); break;
        case ScalarType::float64: launch_convert<Destination, double>(destination, source, stream); break;
        case ScalarType::float8_e4m3fn:
            throw std::invalid_argument("native Float8 conversion requires a format-specific kernel");
    }
}

void dispatch_convert(Tensor& destination, const Tensor& source, cudaStream_t stream) {
    switch (destination.scalar_type()) {
        case ScalarType::boolean: dispatch_source<bool>(destination, source, stream); break;
        case ScalarType::uint8: dispatch_source<std::uint8_t>(destination, source, stream); break;
        case ScalarType::int8: dispatch_source<std::int8_t>(destination, source, stream); break;
        case ScalarType::int16: dispatch_source<std::int16_t>(destination, source, stream); break;
        case ScalarType::int32: dispatch_source<std::int32_t>(destination, source, stream); break;
        case ScalarType::int64: dispatch_source<std::int64_t>(destination, source, stream); break;
        case ScalarType::float16: dispatch_source<__half>(destination, source, stream); break;
        case ScalarType::bfloat16: dispatch_source<__nv_bfloat16>(destination, source, stream); break;
        case ScalarType::float32: dispatch_source<float>(destination, source, stream); break;
        case ScalarType::float64: dispatch_source<double>(destination, source, stream); break;
        case ScalarType::float8_e4m3fn:
            throw std::invalid_argument("native Float8 conversion requires a format-specific kernel");
    }
}

cudaMemcpyKind copy_kind(const Tensor& destination, const Tensor& source) {
    if (destination.is_cuda() && source.is_cuda()) return cudaMemcpyDeviceToDevice;
    if (destination.is_cuda() && source.is_cpu()) return cudaMemcpyHostToDevice;
    if (destination.is_cpu() && source.is_cuda()) return cudaMemcpyDeviceToHost;
    return cudaMemcpyHostToHost;
}

}  // namespace

Tensor empty_cuda(std::span<const std::int64_t> shape, const TensorOptions& options) {
    if (!options.target_device().is_cuda()) {
        throw std::invalid_argument("empty_cuda requires a CUDA device");
    }
    auto descriptor = make_contiguous_view(
        reinterpret_cast<void*>(1), shape, options.scalar_type(), options.target_device());
    auto storage = allocate_cuda_storage(descriptor.nbytes(), options.target_device());
    descriptor.data = storage->base;
    return Tensor(std::move(storage), descriptor);
}

Tensor empty_pinned(std::span<const std::int64_t> shape, const TensorOptions& options) {
    if (!options.target_device().is_cpu()) {
        throw std::invalid_argument("empty_pinned requires a CPU device");
    }
    auto descriptor = make_contiguous_view(
        reinterpret_cast<void*>(1), shape, options.scalar_type(), options.target_device());
    auto storage = allocate_pinned_storage(descriptor.nbytes());
    descriptor.data = storage->base;
    return Tensor(std::move(storage), descriptor);
}

void copy_cuda(Tensor& destination, const Tensor& source) {
    if (destination.numel() != source.numel()) {
        throw std::invalid_argument("native tensor copy changes element count");
    }
    if (!destination.is_cuda() && !source.is_cuda()) {
        throw std::invalid_argument("copy_cuda requires at least one CUDA tensor");
    }
    if (source.numel() == 0) return;
    const auto device = destination.is_cuda() ? destination.device().index : source.device().index;
    auto context = default_context(device);
    const auto stream = current_stream(device);
    DeviceGuard guard(device);
    if (destination.scalar_type() == source.scalar_type() &&
        destination.is_contiguous() && source.is_contiguous()) {
        MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(
            destination.data_ptr(),
            source.data_ptr(),
            destination.nbytes(),
            copy_kind(destination, source),
            stream.stream()));
        if (destination.is_cpu() && source.is_cuda()) {
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        }
        return;
    }
    if (!destination.is_cuda() || !source.is_cuda()) {
        throw std::invalid_argument(
            "strided or dtype-converting host/device copy requires an explicit staging tensor");
    }
    dispatch_convert(destination, source, stream.stream());
}

Tensor copy_or_convert_cuda(const Tensor& source, const TensorOptions& options) {
    auto destination = empty(source.sizes(), options);
    copy_cuda(destination, source);
    return destination;
}

void fill_cuda(Tensor& destination, double value) {
    if (!destination.is_cuda()) {
        throw std::invalid_argument("fill_cuda requires a CUDA tensor");
    }
    auto context = default_context(destination.device().index);
    const auto stream = current_stream(destination.device().index);
    DeviceGuard guard(destination.device().index);
    if (destination.numel() == 0) {
        return;
    }
    if (value == 0.0 && destination.is_contiguous()) {
        MFQ_NATIVE_CUDA_CHECK(cudaMemsetAsync(
            destination.data_ptr(), 0, destination.nbytes(), stream.stream()));
        return;
    }
    constexpr int threads = 256;
    const auto blocks = static_cast<int>(std::min<std::int64_t>(
        4096, (destination.numel() + threads - 1) / threads));
    const auto launch = [&](auto tag) {
        using Value = decltype(tag);
        fill_kernel<Value><<<blocks, threads, 0, stream.stream()>>>(
            destination.view_descriptor(), value, destination.numel());
    };
    switch (destination.scalar_type()) {
        case ScalarType::boolean: launch(bool{}); break;
        case ScalarType::uint8: launch(std::uint8_t{}); break;
        case ScalarType::int8: launch(std::int8_t{}); break;
        case ScalarType::int16: launch(std::int16_t{}); break;
        case ScalarType::int32: launch(std::int32_t{}); break;
        case ScalarType::int64: launch(std::int64_t{}); break;
        case ScalarType::float16: launch(__half{}); break;
        case ScalarType::bfloat16: launch(__nv_bfloat16{}); break;
        case ScalarType::float32: launch(float{}); break;
        case ScalarType::float64: launch(double{}); break;
        case ScalarType::float8_e4m3fn:
            throw std::invalid_argument("native Float8 fill is unsupported");
    }
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
}

Tensor make_contiguous_cuda(const Tensor& source) {
    if (!source.is_cuda()) {
        throw std::invalid_argument("make_contiguous_cuda requires a CUDA tensor");
    }
    auto destination = empty(source.sizes(), source.options());
    copy_cuda(destination, source);
    return destination;
}

Tensor mean_cuda(const Tensor& source, std::int64_t dimension, bool keep_dimension) {
    if (!source.is_cuda() || !source.is_contiguous() ||
        source.scalar_type() != ScalarType::float32 || source.dim() == 0) {
        throw std::invalid_argument(
            "native mean currently requires a contiguous CUDA f32 tensor");
    }
    auto normalized = dimension < 0 ? dimension + source.dim() : dimension;
    if (normalized != source.dim() - 1) {
        throw std::invalid_argument("native mean currently reduces only the last dimension");
    }
    const auto columns = source.size(-1);
    if (columns <= 0) {
        throw std::invalid_argument("native mean cannot reduce an empty dimension");
    }
    const auto rows = source.numel() / columns;
    auto shape = source.sizes().vec();
    if (keep_dimension) {
        shape.back() = 1;
    } else {
        shape.pop_back();
    }
    auto output = empty(shape, source.options());
    auto context = default_context(source.device().index);
    const auto stream = current_stream(source.device().index);
    constexpr int threads = 256;
    row_mean_f32_kernel<threads><<<
        static_cast<unsigned>(rows), threads, 0, stream.stream()>>>(
        source.data_ptr<float>(), output.data_ptr<float>(), rows, columns);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

}  // namespace mfq::cuda
