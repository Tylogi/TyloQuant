#include "mfq_native_tensor.h"

#include "mfq_cuda_context.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace mfq::cuda {
namespace {

std::atomic<std::int64_t> random_seed{0};

std::size_t normalize_dimension(std::int64_t dimension, std::size_t rank) {
    const auto normalized = dimension < 0
        ? dimension + static_cast<std::int64_t>(rank)
        : dimension;
    if (normalized < 0 || normalized >= static_cast<std::int64_t>(rank)) {
        throw std::out_of_range("tensor dimension is out of range");
    }
    return static_cast<std::size_t>(normalized);
}

bool floating(ScalarType type) {
    return type == kFloat16 || type == kBFloat16 ||
           type == kFloat32 || type == kFloat64;
}

ScalarType promote(ScalarType left, ScalarType right) {
    if (left == right) return left;
    if (left == kFloat64 || right == kFloat64) return kFloat64;
    if (left == kFloat32 || right == kFloat32) return kFloat32;
    if ((left == kFloat16 && right == kBFloat16) ||
        (left == kBFloat16 && right == kFloat16)) return kFloat32;
    if (left == kBFloat16 || right == kBFloat16) return kBFloat16;
    if (left == kFloat16 || right == kFloat16) return kFloat16;
    if (left == kInt64 || right == kInt64) return kInt64;
    if (left == kInt32 || right == kInt32) return kInt32;
    if (left == kInt16 || right == kInt16) return kInt16;
    if (left == kInt8 || right == kInt8) return kInt8;
    if (left == kUInt8 || right == kUInt8) return kUInt8;
    return kBool;
}

TensorView align_for_broadcast(
    const Tensor& input,
    std::span<const std::int64_t> output_shape) {
    if (input.dim() > static_cast<std::int64_t>(output_shape.size())) {
        throw std::invalid_argument("broadcast rank mismatch");
    }
    auto result = input.view_descriptor();
    std::array<std::int64_t, kMaximumTensorRank> sizes{};
    std::array<std::int64_t, kMaximumTensorRank> strides{};
    const auto offset = output_shape.size() - static_cast<std::size_t>(input.dim());
    for (std::size_t dimension = 0; dimension < output_shape.size(); ++dimension) {
        sizes[dimension] = output_shape[dimension];
        if (dimension < offset) {
            strides[dimension] = 0;
            continue;
        }
        const auto source_dimension = dimension - offset;
        const auto source_extent = input.size(static_cast<std::int64_t>(source_dimension));
        if (source_extent != output_shape[dimension] && source_extent != 1) {
            throw std::invalid_argument("tensor dimensions cannot be broadcast");
        }
        strides[dimension] = source_extent == 1
            ? 0
            : input.stride(static_cast<std::int64_t>(source_dimension));
    }
    result.rank = static_cast<std::uint8_t>(output_shape.size());
    result.sizes = sizes;
    result.strides = strides;
    return result;
}

std::vector<std::int64_t> broadcast_shape(const Tensor& left, const Tensor& right) {
    const auto rank = static_cast<std::size_t>(std::max(left.dim(), right.dim()));
    if (rank > kMaximumTensorRank) throw std::invalid_argument("broadcast rank exceeds ABI");
    std::vector<std::int64_t> result(rank, 1);
    for (std::size_t reverse = 0; reverse < rank; ++reverse) {
        const auto left_dimension = left.dim() - 1 - static_cast<std::int64_t>(reverse);
        const auto right_dimension = right.dim() - 1 - static_cast<std::int64_t>(reverse);
        const auto l = left_dimension >= 0 ? left.size(left_dimension) : 1;
        const auto r = right_dimension >= 0 ? right.size(right_dimension) : 1;
        if (l != r && l != 1 && r != 1) {
            throw std::invalid_argument("tensor dimensions cannot be broadcast");
        }
        result[rank - 1 - reverse] = std::max(l, r);
    }
    return result;
}

__device__ std::int64_t tensor_offset(const TensorView& view, std::int64_t linear) {
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
__device__ double load_number(const Value* values, std::int64_t index) {
    if constexpr (std::is_same_v<Value, __half>) {
        return static_cast<double>(__half2float(values[index]));
    } else if constexpr (std::is_same_v<Value, __nv_bfloat16>) {
        return static_cast<double>(__bfloat162float(values[index]));
    } else {
        return static_cast<double>(values[index]);
    }
}

template <typename Value>
__device__ Value store_number(double value) {
    if constexpr (std::is_same_v<Value, __half>) {
        return __float2half_rn(static_cast<float>(value));
    } else if constexpr (std::is_same_v<Value, __nv_bfloat16>) {
        return __float2bfloat16_rn(static_cast<float>(value));
    } else {
        return static_cast<Value>(value);
    }
}

template <typename Value>
__global__ void unary_kernel(
    TensorView output,
    TensorView input,
    std::int64_t elements,
    int operation,
    double first,
    double second) {
    const auto* source = static_cast<const Value*>(input.data);
    auto* destination = static_cast<Value*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const double value = load_number(source, tensor_offset(input, linear));
        double result = value;
        switch (operation) {
            case 0: result = ::fabs(value); break;
            case 1: result = value * value; break;
            case 2: result = ::exp(value); break;
            case 3: result = ::sqrt(value); break;
            case 4: result = ::fmin(::fmax(value, first), second); break;
            case 5: result = ::fmax(value, first); break;
            case 6: result = ::fmin(value, first); break;
            case 7: result = value - ::floor(value / first) * first; break;
            case 9: result = ::sin(value); break;
            case 10: result = ::cos(value); break;
            case 11: result = 1.0 / (1.0 + ::exp(-value)); break;
            case 12: result = ::tanh(value); break;
            case 13: result = ::fmax(value, 0.0); break;
            case 14: result = 1.0 / ::sqrt(value); break;
            case 15: result = 1.0 / value; break;
            case 16: result = ::ceil(value); break;
            case 17: result = value > 20.0 ? value : ::log1p(::exp(value)); break;
            case 18: result = ::log(value); break;
            case 19: result = ::log1p(value); break;
            case 20: result = ::log2(value); break;
            case 21: result = ::exp2(value); break;
            case 22: result = value / (1.0 + ::exp(-value)); break;
            case 23: {
                constexpr double inv_sqrt_two = 0.7071067811865475244;
                result = 0.5 * value * (1.0 + ::erf(value * inv_sqrt_two));
                break;
            }
            case 24: result = ::pow(value, first); break;
        }
        destination[tensor_offset(output, linear)] = store_number<Value>(result);
    }
}

template <typename Input>
__global__ void unary_bool_kernel(
    TensorView output,
    TensorView input,
    std::int64_t elements,
    int operation) {
    const auto* source = static_cast<const Input*>(input.data);
    auto* destination = static_cast<bool*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const double value = load_number(source, tensor_offset(input, linear));
        bool result = false;
        if (operation == 8) result = value == 0.0;
        if (operation == 25) result = ::isfinite(value);
        if (operation == 26) result = ::isinf(value) && value < 0.0;
        destination[tensor_offset(output, linear)] = result;
    }
}

template <typename Value>
__global__ void binary_kernel(
    TensorView output,
    TensorView left,
    TensorView right,
    std::int64_t elements,
    int operation) {
    const auto* a = static_cast<const Value*>(left.data);
    const auto* b = static_cast<const Value*>(right.data);
    auto* destination = static_cast<Value*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const double x = load_number(a, tensor_offset(left, linear));
        const double y = load_number(b, tensor_offset(right, linear));
        double result = 0.0;
        if (operation == 0) result = x + y;
        if (operation == 1) result = x - y;
        if (operation == 2) result = x * y;
        if (operation == 3) result = x / y;
        if (operation == 8) result = static_cast<double>(
            static_cast<std::int64_t>(x) & static_cast<std::int64_t>(y));
        if (operation == 9) result = static_cast<double>(
            static_cast<std::int64_t>(x) | static_cast<std::int64_t>(y));
        if (operation == 12) result = ::pow(x, y);
        destination[tensor_offset(output, linear)] = store_number<Value>(result);
    }
}

template <typename Value>
__global__ void comparison_kernel(
    TensorView output,
    TensorView left,
    TensorView right,
    std::int64_t elements,
    int operation) {
    const auto* a = static_cast<const Value*>(left.data);
    const auto* b = static_cast<const Value*>(right.data);
    auto* destination = static_cast<bool*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const double x = load_number(a, tensor_offset(left, linear));
        const double y = load_number(b, tensor_offset(right, linear));
        bool result = false;
        if (operation == 4) result = x < y;
        if (operation == 5) result = x <= y;
        if (operation == 6) result = x > y;
        if (operation == 7) result = x >= y;
        if (operation == 10) result = x == y;
        if (operation == 11) result = x != y;
        destination[tensor_offset(output, linear)] = result;
    }
}

template <typename Value>
__global__ void scalar_binary_kernel(
    TensorView output,
    TensorView input,
    std::int64_t elements,
    double scalar,
    int operation,
    bool scalar_first) {
    const auto* source = static_cast<const Value*>(input.data);
    auto* destination = static_cast<Value*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const double value = load_number(source, tensor_offset(input, linear));
        const double x = scalar_first ? scalar : value;
        const double y = scalar_first ? value : scalar;
        double result = 0.0;
        if (operation == 0) result = x + y;
        if (operation == 1) result = x - y;
        if (operation == 2) result = x * y;
        if (operation == 3) result = x / y;
        if (operation == 8) result = static_cast<double>(
            static_cast<std::int64_t>(x) & static_cast<std::int64_t>(y));
        if (operation == 9) result = static_cast<double>(
            static_cast<std::int64_t>(x) | static_cast<std::int64_t>(y));
        if (operation == 12) result = ::pow(x, y);
        destination[tensor_offset(output, linear)] = store_number<Value>(result);
    }
}

template <typename Value>
__global__ void scalar_comparison_kernel(
    TensorView output,
    TensorView input,
    std::int64_t elements,
    double scalar,
    int operation,
    bool scalar_first) {
    const auto* source = static_cast<const Value*>(input.data);
    auto* destination = static_cast<bool*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const double value = load_number(source, tensor_offset(input, linear));
        const double x = scalar_first ? scalar : value;
        const double y = scalar_first ? value : scalar;
        bool result = false;
        if (operation == 4) result = x < y;
        if (operation == 5) result = x <= y;
        if (operation == 6) result = x > y;
        if (operation == 7) result = x >= y;
        if (operation == 10) result = x == y;
        if (operation == 11) result = x != y;
        destination[tensor_offset(output, linear)] = result;
    }
}

template <typename Function>
void dispatch_numeric(ScalarType type, Function&& function) {
    switch (type) {
        case kBool: function.template operator()<bool>(); break;
        case kUInt8: function.template operator()<std::uint8_t>(); break;
        case kInt8: function.template operator()<std::int8_t>(); break;
        case kInt16: function.template operator()<std::int16_t>(); break;
        case kInt32: function.template operator()<std::int32_t>(); break;
        case kInt64: function.template operator()<std::int64_t>(); break;
        case kFloat16: function.template operator()<__half>(); break;
        case kBFloat16: function.template operator()<__nv_bfloat16>(); break;
        case kFloat32: function.template operator()<float>(); break;
        case kFloat64: function.template operator()<double>(); break;
        case kFloat8E4M3FN:
            throw std::invalid_argument("generic native op does not accept FP8 storage");
    }
}

std::pair<int, int> launch_geometry(std::int64_t elements) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<std::int64_t>(
        4096, std::max<std::int64_t>(1, (elements + threads - 1) / threads)));
    return {blocks, threads};
}

cudaDataType_t cuda_data_type(ScalarType type) {
    switch (type) {
        case kFloat16: return CUDA_R_16F;
        case kBFloat16: return CUDA_R_16BF;
        case kFloat32: return CUDA_R_32F;
        case kFloat64: return CUDA_R_64F;
        default: throw std::invalid_argument("cuBLAS matmul requires a floating dtype");
    }
}

std::vector<std::int64_t> matmul_batch_shape(const Tensor& left, const Tensor& right) {
    const auto left_rank = static_cast<std::size_t>(left.dim() - 2);
    const auto right_rank = static_cast<std::size_t>(right.dim() - 2);
    const auto rank = std::max(left_rank, right_rank);
    std::vector<std::int64_t> shape(rank, 1);
    for (std::size_t reverse = 0; reverse < rank; ++reverse) {
        const auto left_axis = static_cast<std::int64_t>(left_rank) - 1 -
            static_cast<std::int64_t>(reverse);
        const auto right_axis = static_cast<std::int64_t>(right_rank) - 1 -
            static_cast<std::int64_t>(reverse);
        const auto l = left_axis >= 0 ? left.size(left_axis) : 1;
        const auto r = right_axis >= 0 ? right.size(right_axis) : 1;
        if (l != r && l != 1 && r != 1) {
            throw std::invalid_argument("matmul batch dimensions cannot be broadcast");
        }
        shape[rank - 1 - reverse] = std::max(l, r);
    }
    return shape;
}

std::vector<std::int64_t> batch_coordinates(
    std::int64_t linear,
    std::span<const std::int64_t> shape) {
    std::vector<std::int64_t> coordinates(shape.size(), 0);
    for (std::size_t reverse = shape.size(); reverse > 0; --reverse) {
        const auto axis = reverse - 1;
        coordinates[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    return coordinates;
}

std::int64_t matmul_batch_offset(
    const Tensor& tensor,
    std::span<const std::int64_t> output_shape,
    std::span<const std::int64_t> coordinates) {
    const auto input_rank = static_cast<std::size_t>(tensor.dim() - 2);
    const auto offset = output_shape.size() - input_rank;
    std::int64_t result = 0;
    for (std::size_t axis = 0; axis < input_rank; ++axis) {
        const auto extent = tensor.size(static_cast<std::int64_t>(axis));
        if (extent != 1) {
            result += coordinates[offset + axis] *
                tensor.stride(static_cast<std::int64_t>(axis));
        }
    }
    return result;
}

bool matrix_layout_supported(const Tensor& tensor) {
    return tensor.stride(-1) == 1 || tensor.stride(-2) == 1;
}

struct ParallelBatchMatmulContext {
    static constexpr std::size_t kStreams = 4;

    explicit ParallelBatchMatmulContext(int device) : ready() {
        streams.reserve(kStreams);
        handles.reserve(kStreams);
        done.reserve(kStreams);
        for (std::size_t index = 0; index < kStreams; ++index) {
            streams.emplace_back(device);
            handles.emplace_back(device);
            handles.back().set_stream(streams.back().get());
            done.emplace_back();
        }
    }

    std::mutex mutex;
    Event ready;
    std::vector<Stream> streams;
    std::vector<BlasHandle> handles;
    std::vector<Event> done;
};

std::mutex parallel_batch_matmul_contexts_mutex;
std::unordered_map<int, std::unique_ptr<ParallelBatchMatmulContext>>
    parallel_batch_matmul_contexts;

ParallelBatchMatmulContext& parallel_batch_matmul_context(int device) {
    std::lock_guard<std::mutex> lock(parallel_batch_matmul_contexts_mutex);
    auto& context = parallel_batch_matmul_contexts[device];
    if (!context) {
        DeviceGuard guard(device);
        context = std::make_unique<ParallelBatchMatmulContext>(device);
    }
    return *context;
}

template <typename Value>
__global__ void conv1d_kernel(
    TensorView output,
    TensorView input,
    TensorView weight,
    TensorView bias,
    bool has_bias,
    std::int64_t elements,
    std::int64_t output_length,
    std::int64_t stride,
    std::int64_t padding,
    std::int64_t dilation,
    std::int64_t groups) {
    const auto input_channels = input.sizes[1];
    const auto output_channels = output.sizes[1];
    const auto input_length = input.sizes[2];
    const auto kernel = weight.sizes[2];
    const auto input_per_group = input_channels / groups;
    const auto output_per_group = output_channels / groups;
    const auto* x = static_cast<const Value*>(input.data);
    const auto* w = static_cast<const Value*>(weight.data);
    const auto* b = has_bias ? static_cast<const Value*>(bias.data) : nullptr;
    auto* y = static_cast<Value*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        const auto position = residual % output_length;
        residual /= output_length;
        const auto channel = residual % output_channels;
        const auto batch = residual / output_channels;
        const auto group = channel / output_per_group;
        using Accumulator = std::conditional_t<std::is_same_v<Value, double>, double, float>;
        Accumulator accumulator = has_bias
            ? static_cast<Accumulator>(load_number(b, channel))
            : Accumulator{0};
        for (std::int64_t local_channel = 0; local_channel < input_per_group;
             ++local_channel) {
            const auto input_channel = group * input_per_group + local_channel;
            for (std::int64_t tap = 0; tap < kernel; ++tap) {
                const auto source_position = position * stride - padding + tap * dilation;
                if (source_position < 0 || source_position >= input_length) continue;
                const auto input_offset =
                    (batch * input_channels + input_channel) * input_length + source_position;
                const auto weight_offset =
                    (channel * input_per_group + local_channel) * kernel + tap;
                accumulator += static_cast<Accumulator>(load_number(x, input_offset)) *
                    static_cast<Accumulator>(load_number(w, weight_offset));
            }
        }
        y[linear] = store_number<Value>(accumulator);
    }
}

template <typename Value>
__global__ void conv2d_kernel(
    TensorView output,
    TensorView input,
    TensorView weight,
    TensorView bias,
    bool has_bias,
    std::int64_t elements,
    std::int64_t stride_h,
    std::int64_t stride_w,
    std::int64_t padding_h,
    std::int64_t padding_w,
    std::int64_t dilation_h,
    std::int64_t dilation_w,
    std::int64_t groups) {
    const auto input_channels = input.sizes[1];
    const auto output_channels = output.sizes[1];
    const auto input_h = input.sizes[2];
    const auto input_w = input.sizes[3];
    const auto output_h = output.sizes[2];
    const auto output_w = output.sizes[3];
    const auto kernel_h = weight.sizes[2];
    const auto kernel_w = weight.sizes[3];
    const auto input_per_group = input_channels / groups;
    const auto output_per_group = output_channels / groups;
    const auto* x = static_cast<const Value*>(input.data);
    const auto* w = static_cast<const Value*>(weight.data);
    const auto* b = has_bias ? static_cast<const Value*>(bias.data) : nullptr;
    auto* y = static_cast<Value*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        const auto output_x = residual % output_w;
        residual /= output_w;
        const auto output_y = residual % output_h;
        residual /= output_h;
        const auto channel = residual % output_channels;
        const auto batch = residual / output_channels;
        const auto group = channel / output_per_group;
        using Accumulator = std::conditional_t<std::is_same_v<Value, double>, double, float>;
        Accumulator accumulator = has_bias
            ? static_cast<Accumulator>(load_number(b, channel))
            : Accumulator{0};
        for (std::int64_t local_channel = 0; local_channel < input_per_group;
             ++local_channel) {
            const auto input_channel = group * input_per_group + local_channel;
            for (std::int64_t ky = 0; ky < kernel_h; ++ky) {
                const auto source_y = output_y * stride_h - padding_h + ky * dilation_h;
                if (source_y < 0 || source_y >= input_h) continue;
                for (std::int64_t kx = 0; kx < kernel_w; ++kx) {
                    const auto source_x = output_x * stride_w - padding_w + kx * dilation_w;
                    if (source_x < 0 || source_x >= input_w) continue;
                    const auto input_offset =
                        ((batch * input_channels + input_channel) * input_h + source_y) *
                            input_w + source_x;
                    const auto weight_offset =
                        ((channel * input_per_group + local_channel) * kernel_h + ky) *
                            kernel_w + kx;
                    accumulator += static_cast<Accumulator>(load_number(x, input_offset)) *
                        static_cast<Accumulator>(load_number(w, weight_offset));
                }
            }
        }
        y[linear] = store_number<Value>(accumulator);
    }
}

template <typename Value>
__global__ void avg_pool1d_kernel(
    TensorView output,
    TensorView input,
    std::int64_t elements,
    std::int64_t kernel,
    std::int64_t stride,
    std::int64_t padding) {
    const auto output_length = output.sizes[2];
    const auto input_length = input.sizes[2];
    const auto* source = static_cast<const Value*>(input.data);
    auto* destination = static_cast<Value*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const auto output_position = linear % output_length;
        const auto row = linear / output_length;
        using Accumulator = std::conditional_t<std::is_same_v<Value, double>, double, float>;
        Accumulator accumulator = 0;
        for (std::int64_t tap = 0; tap < kernel; ++tap) {
            const auto position = output_position * stride - padding + tap;
            if (position >= 0 && position < input_length) {
                accumulator += static_cast<Accumulator>(
                    load_number(source, row * input_length + position));
            }
        }
        destination[linear] = store_number<Value>(
            accumulator / static_cast<Accumulator>(kernel));
    }
}

template <typename Value>
__global__ void cumsum_kernel(
    TensorView output,
    TensorView input,
    std::int64_t outer,
    std::int64_t length,
    std::int64_t inner) {
    const auto rows = outer * inner;
    const auto* source = static_cast<const Value*>(input.data);
    auto* destination = static_cast<Value*>(output.data);
    for (std::int64_t row =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const auto outer_index = row / inner;
        const auto inner_index = row % inner;
        using Accumulator = std::conditional_t<
            std::is_same_v<Value, double>, double,
            std::conditional_t<std::is_integral_v<Value>, std::int64_t, float>>;
        Accumulator accumulator = 0;
        for (std::int64_t index = 0; index < length; ++index) {
            const auto offset = (outer_index * length + index) * inner + inner_index;
            if constexpr (std::is_integral_v<Value>) {
                accumulator += static_cast<Accumulator>(source[offset]);
            } else {
                accumulator += static_cast<Accumulator>(load_number(source, offset));
            }
            if constexpr (std::is_integral_v<Value>) {
                destination[offset] = static_cast<Value>(accumulator);
            } else {
                destination[offset] = store_number<Value>(accumulator);
            }
        }
    }
}

template <typename Value>
__global__ void multinomial_one_kernel(
    const Value* probabilities,
    std::int64_t* output,
    const double* uniforms,
    std::int64_t rows,
    std::int64_t columns) {
    for (std::int64_t row =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        double total = 0.0;
        for (std::int64_t column = 0; column < columns; ++column) {
            total += ::fmax(0.0, load_number(probabilities, row * columns + column));
        }
        const double target = uniforms[row] * total;
        double cumulative = 0.0;
        std::int64_t selected = columns - 1;
        for (std::int64_t column = 0; column < columns; ++column) {
            cumulative += ::fmax(0.0, load_number(probabilities, row * columns + column));
            if (target < cumulative) {
                selected = column;
                break;
            }
        }
        output[row] = selected;
    }
}

template <typename Value, typename Output>
__global__ void reduce_kernel(
    TensorView output,
    TensorView input,
    std::int64_t outer_elements,
    std::int64_t reduced,
    int dimension,
    int operation,
    bool keep_dimension) {
    auto* destination = static_cast<Output*>(output.data);
    const auto* source = static_cast<const Value*>(input.data);
    for (std::int64_t output_linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         output_linear < outer_elements;
         output_linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        std::array<std::int64_t, kMaximumTensorRank> coordinates{};
        auto residual = output_linear;
        for (std::size_t reverse = output.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            coordinates[axis] = residual % output.sizes[axis];
            residual /= output.sizes[axis];
        }
        std::int64_t base = 0;
        std::size_t output_axis = 0;
        for (std::size_t input_axis = 0; input_axis < input.rank; ++input_axis) {
            if (static_cast<int>(input_axis) == dimension) continue;
            const auto coordinate_axis = keep_dimension ? input_axis : output_axis++;
            base += coordinates[coordinate_axis] * input.strides[input_axis];
        }
        using Accumulator = std::conditional_t<
            std::is_same_v<Value, double>, double,
            std::conditional_t<
                std::is_same_v<Value, float> || std::is_same_v<Value, __half> ||
                    std::is_same_v<Value, __nv_bfloat16>,
                float,
                std::int64_t>>;
        Accumulator accumulator = operation == 2 || operation == 3
            ? std::numeric_limits<Accumulator>::lowest()
            : Accumulator{0};
        std::int64_t best = 0;
        bool boolean = operation == 4;
        if (operation == 5) boolean = false;
        for (std::int64_t index = 0; index < reduced; ++index) {
            const auto source_offset = base + index * input.strides[dimension];
            const auto value = [&] {
                if constexpr (std::is_integral_v<Value>) {
                    return static_cast<Accumulator>(source[source_offset]);
                } else {
                    return static_cast<Accumulator>(load_number(source, source_offset));
                }
            }();
            if (operation == 0 || operation == 1) accumulator += value;
            if (operation == 2 || operation == 3) {
                if (value > accumulator) {
                    accumulator = value;
                    best = index;
                }
            }
            if (operation == 4) boolean = boolean && value != 0.0;
            if (operation == 5) boolean = boolean || value != 0.0;
        }
        if (operation == 1) accumulator /= static_cast<Accumulator>(reduced);
        if (operation == 3) accumulator = static_cast<Accumulator>(best);
        if (operation == 4 || operation == 5) accumulator = boolean ? 1.0 : 0.0;
        if constexpr (std::is_integral_v<Output>) {
            destination[output_linear] = static_cast<Output>(accumulator);
        } else {
            destination[output_linear] = store_number<Output>(accumulator);
        }
    }
}

template <int Threads>
__global__ void argmax_last_contiguous_bf16_kernel(
    const __nv_bfloat16* input,
    std::int64_t* output,
    std::int64_t rows,
    std::int64_t columns) {
    __shared__ float values[Threads];
    __shared__ std::int64_t indices[Threads];
    constexpr std::int64_t invalid_index = 0x7fffffffffffffffLL;
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto* row_input = input + row * columns;
    float best_value = 0.0f;
    std::int64_t best_index = invalid_index;
    for (std::int64_t column = threadIdx.x;
         column < columns;
         column += Threads) {
        const float value = __bfloat162float(row_input[column]);
        if (best_index == invalid_index ||
            value > best_value ||
            (value == best_value && column < best_index)) {
            best_value = value;
            best_index = column;
        }
    }
    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    for (int offset = Threads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            const auto other_index = indices[threadIdx.x + offset];
            const auto other_value = values[threadIdx.x + offset];
            if (other_index != invalid_index &&
                (indices[threadIdx.x] == invalid_index ||
                 other_value > values[threadIdx.x] ||
                 (other_value == values[threadIdx.x] &&
                  other_index < indices[threadIdx.x]))) {
                values[threadIdx.x] = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = indices[0];
}

ScalarType reduction_type(ScalarType input, int operation) {
    if (operation == 3) return kInt64;
    if (operation == 4 || operation == 5) return kBool;
    if (operation == 0 && !floating(input)) return kInt64;
    return input;
}

template <typename Input>
void launch_reduction_output(
    Tensor& output,
    const Tensor& input,
    std::int64_t outer,
    std::int64_t reduced,
    int dimension,
    int operation,
    bool keep_dimension,
    cudaStream_t stream) {
    const auto [blocks, threads] = launch_geometry(outer);
    auto launch = [&]<typename Output>() {
        reduce_kernel<Input, Output><<<blocks, threads, 0, stream>>>(
            output.view_descriptor(), input.view_descriptor(), outer,
            reduced, dimension, operation, keep_dimension);
    };
    dispatch_numeric(output.scalar_type(), launch);
}

}  // namespace

void manual_seed(std::int64_t seed) {
    random_seed.store(seed, std::memory_order_relaxed);
}

Tensor unary_cuda(
    const Tensor& source,
    int operation,
    double first = 0.0,
    double second = 0.0) {
    if (!source.is_cuda()) {
        throw std::invalid_argument("native unary CUDA op requires a CUDA tensor");
    }
    const bool boolean_output = operation == 8 || operation == 25 || operation == 26;
    auto options = source.options().dtype(boolean_output ? kBool : source.scalar_type());
    auto output = empty(source.sizes(), options);
    const auto [blocks, threads] = launch_geometry(source.numel());
    const auto stream = current_stream(source.get_device()).stream();
    auto launch = [&]<typename Value>() {
        if (boolean_output) {
            unary_bool_kernel<Value><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), source.view_descriptor(), source.numel(), operation);
        } else {
            unary_kernel<Value><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), source.view_descriptor(), source.numel(),
                operation, first, second);
        }
    };
    dispatch_numeric(source.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor binary_cuda(const Tensor& left_source, const Tensor& right_source, int operation) {
    if (!left_source.is_cuda() || !right_source.is_cuda()) {
        throw std::invalid_argument("native binary CUDA op requires CUDA tensors");
    }
    const auto type = promote(left_source.scalar_type(), right_source.scalar_type());
    auto left = left_source.scalar_type() == type ? left_source : left_source.to(type);
    auto right = right_source.scalar_type() == type ? right_source : right_source.to(type);
    const auto shape = broadcast_shape(left, right);
    const bool comparison = operation >= 4 && operation <= 7 ||
                            operation == 10 || operation == 11;
    auto output = empty(shape, left.options().dtype(comparison ? kBool : type));
    const auto left_view = align_for_broadcast(left, shape);
    const auto right_view = align_for_broadcast(right, shape);
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream = current_stream(left.get_device()).stream();
    auto launch = [&]<typename Value>() {
        if (comparison) {
            comparison_kernel<Value><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), left_view, right_view, output.numel(), operation);
        } else {
            binary_kernel<Value><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), left_view, right_view, output.numel(), operation);
        }
    };
    dispatch_numeric(type, launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor scalar_binary_cuda(
    const Tensor& left,
    double right,
    int operation,
    bool scalar_first = false) {
    if (!left.is_cuda()) {
        throw std::invalid_argument("native scalar CUDA op requires a CUDA tensor");
    }
    const bool comparison = operation >= 4 && operation <= 7 ||
                            operation == 10 || operation == 11;
    auto output = empty(
        left.sizes(), left.options().dtype(comparison ? kBool : left.scalar_type()));
    const auto [blocks, threads] = launch_geometry(left.numel());
    const auto stream = current_stream(left.get_device()).stream();
    auto launch = [&]<typename Value>() {
        if (comparison) {
            scalar_comparison_kernel<Value><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), left.view_descriptor(), left.numel(),
                right, operation, scalar_first);
        } else {
            scalar_binary_kernel<Value><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), left.view_descriptor(), left.numel(),
                right, operation, scalar_first);
        }
    };
    dispatch_numeric(left.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor reduce_cuda(
    const Tensor& source,
    std::int64_t dimension,
    bool keep_dimension,
    int operation) {
    if (!source.is_cuda() || source.dim() == 0) {
        throw std::invalid_argument("native reduction requires a ranked CUDA tensor");
    }
    if (operation == 1 && !floating(source.scalar_type())) {
        throw std::invalid_argument("native mean requires a floating dtype");
    }
    const auto selected = normalize_dimension(dimension, source.dim());
    const auto reduced = source.size(static_cast<std::int64_t>(selected));
    if (reduced == 0) throw std::invalid_argument("cannot reduce an empty dimension");
    auto shape = source.sizes().vec();
    if (keep_dimension) shape[selected] = 1;
    else shape.erase(shape.begin() + static_cast<std::ptrdiff_t>(selected));
    auto output = empty(shape, source.options().dtype(
        reduction_type(source.scalar_type(), operation)));
    const auto outer = output.numel();
    const auto stream = current_stream(source.get_device()).stream();
    if (operation == 3 &&
        source.scalar_type() == kBFloat16 &&
        source.is_contiguous() &&
        outer <= std::numeric_limits<unsigned int>::max() &&
        selected + 1 == static_cast<std::size_t>(source.dim())) {
        constexpr int threads = 256;
        argmax_last_contiguous_bf16_kernel<threads>
            <<<static_cast<unsigned int>(outer), threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(source.data_ptr()),
                output.data_ptr<std::int64_t>(), outer, reduced);
        MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
        return output;
    }
    auto launch = [&]<typename Input>() {
        launch_reduction_output<Input>(
            output, source, outer, reduced, static_cast<int>(selected), operation,
            keep_dimension, stream);
    };
    dispatch_numeric(source.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor operator+(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 0); }
Tensor operator+(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 0); }
Tensor operator+(double left, const Tensor& right) { return scalar_binary_cuda(right, left, 0, true); }
Tensor operator-(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 1); }
Tensor operator-(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 1); }
Tensor operator-(double left, const Tensor& right) { return scalar_binary_cuda(right, left, 1, true); }
Tensor operator-(const Tensor& value) { return scalar_binary_cuda(value, -1.0, 2); }
Tensor operator*(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 2); }
Tensor operator*(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 2); }
Tensor operator*(double left, const Tensor& right) { return scalar_binary_cuda(right, left, 2, true); }
Tensor operator/(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 3); }
Tensor operator/(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 3); }
Tensor operator/(double left, const Tensor& right) { return scalar_binary_cuda(right, left, 3, true); }
Tensor operator<(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 4); }
Tensor operator<(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 4); }
Tensor operator<=(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 5); }
Tensor operator<=(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 5); }
Tensor operator>(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 6); }
Tensor operator>(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 6); }
Tensor operator>=(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 7); }
Tensor operator>=(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 7); }
Tensor operator&(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 8); }
Tensor operator|(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 9); }
Tensor operator~(const Tensor& value) { return value.logical_not(); }
Tensor operator==(const Tensor& left, const Tensor& right) { return left.eq(right); }
Tensor operator==(const Tensor& left, double right) { return left.eq(right); }
Tensor operator!=(const Tensor& left, const Tensor& right) { return left.ne(right); }
Tensor operator!=(const Tensor& left, double right) { return left.ne(right); }

Tensor sin(const Tensor& input) { return unary_cuda(input, 9); }
Tensor cos(const Tensor& input) { return unary_cuda(input, 10); }
Tensor sigmoid(const Tensor& input) { return unary_cuda(input, 11); }
Tensor tanh(const Tensor& input) { return unary_cuda(input, 12); }
Tensor relu(const Tensor& input) { return unary_cuda(input, 13); }
Tensor rsqrt(const Tensor& input) { return unary_cuda(input, 14); }
Tensor reciprocal(const Tensor& input) { return unary_cuda(input, 15); }
Tensor ceil(const Tensor& input) { return unary_cuda(input, 16); }
Tensor softplus(const Tensor& input) { return unary_cuda(input, 17); }
Tensor log(const Tensor& input) { return unary_cuda(input, 18); }
Tensor log1p(const Tensor& input) { return unary_cuda(input, 19); }
Tensor log2(const Tensor& input) { return unary_cuda(input, 20); }
Tensor exp2(const Tensor& input) { return unary_cuda(input, 21); }
Tensor silu(const Tensor& input) { return unary_cuda(input, 22); }
Tensor gelu(const Tensor& input, const std::string&) { return unary_cuda(input, 23); }
Tensor exp(const Tensor& input) { return input.exp(); }
Tensor sqrt(const Tensor& input) { return input.sqrt(); }
Tensor pow(const Tensor& input, double exponent) { return unary_cuda(input, 24, exponent); }
Tensor pow(const Tensor& input, const Tensor& exponent) { return binary_cuda(input, exponent, 12); }
Tensor remainder(const Tensor& input, double divisor) { return input.remainder(divisor); }
Tensor clamp(const Tensor& input, double minimum, double maximum) { return input.clamp(minimum, maximum); }
Tensor clamp_min(const Tensor& input, double minimum) { return input.clamp_min(minimum); }
Tensor clamp_max(const Tensor& input, double maximum) { return input.clamp_max(maximum); }
Tensor isfinite(const Tensor& input) { return unary_cuda(input, 25); }
Tensor isneginf(const Tensor& input) { return unary_cuda(input, 26); }
Tensor sum(const Tensor& input, std::int64_t dimension, bool keep_dimension) {
    return input.sum(dimension, keep_dimension);
}
Tensor mean(const Tensor& input, std::int64_t dimension, bool keep_dimension) {
    return input.mean(dimension, keep_dimension);
}
Tensor argmax(const Tensor& input, std::int64_t dimension, bool keep_dimension) {
    return input.argmax(dimension, keep_dimension);
}

namespace {

template <typename Index>
__device__ std::int64_t load_index(const Index* indices, std::int64_t offset) {
    return static_cast<std::int64_t>(indices[offset]);
}

template <typename Value, typename Index>
__global__ void index_select_kernel(
    TensorView output,
    TensorView source,
    TensorView indices,
    std::int64_t elements,
    int dimension) {
    auto* destination = static_cast<Value*>(output.data);
    const auto* input = static_cast<const Value*>(source.data);
    const auto* selected = static_cast<const Index*>(indices.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        std::int64_t source_offset = 0;
        for (std::size_t reverse = output.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            auto coordinate = residual % output.sizes[axis];
            residual /= output.sizes[axis];
            if (static_cast<int>(axis) == dimension) {
                coordinate = load_index(selected, coordinate);
            }
            source_offset += coordinate * source.strides[axis];
        }
        destination[linear] = input[source_offset];
    }
}

template <typename Value, typename Index>
__global__ void gather_kernel(
    TensorView output,
    TensorView source,
    TensorView indices,
    std::int64_t elements,
    int dimension) {
    auto* destination = static_cast<Value*>(output.data);
    const auto* input = static_cast<const Value*>(source.data);
    const auto* selected = static_cast<const Index*>(indices.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        std::int64_t source_offset = 0;
        for (std::size_t reverse = output.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            auto coordinate = residual % output.sizes[axis];
            residual /= output.sizes[axis];
            if (static_cast<int>(axis) == dimension) {
                coordinate = load_index(selected, tensor_offset(indices, linear));
            }
            source_offset += coordinate * source.strides[axis];
        }
        destination[linear] = input[source_offset];
    }
}

template <typename Value>
__global__ void repeat_kernel(
    TensorView output,
    TensorView source,
    std::int64_t elements) {
    auto* destination = static_cast<Value*>(output.data);
    const auto* input = static_cast<const Value*>(source.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        std::int64_t source_offset = 0;
        for (std::size_t reverse = output.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            const auto coordinate = residual % output.sizes[axis];
            residual /= output.sizes[axis];
            source_offset += (coordinate % source.sizes[axis]) * source.strides[axis];
        }
        destination[linear] = input[source_offset];
    }
}

template <typename Value>
__global__ void repeat_interleave_kernel(
    TensorView output,
    TensorView source,
    std::int64_t elements,
    int dimension,
    std::int64_t repeats) {
    auto* destination = static_cast<Value*>(output.data);
    const auto* input = static_cast<const Value*>(source.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        std::int64_t source_offset = 0;
        for (std::size_t reverse = output.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            auto coordinate = residual % output.sizes[axis];
            residual /= output.sizes[axis];
            if (static_cast<int>(axis) == dimension) coordinate /= repeats;
            source_offset += coordinate * source.strides[axis];
        }
        destination[linear] = input[source_offset];
    }
}

template <typename Value>
__device__ void atomic_add_16bit(Value* destination, Value value) {
    const auto address = reinterpret_cast<std::uintptr_t>(destination);
    auto* word = reinterpret_cast<unsigned int*>(address & ~std::uintptr_t{3});
    const bool upper = (address & std::uintptr_t{2}) != 0;
    unsigned int observed = *word;
    unsigned int expected = 0;
    do {
        expected = observed;
        const auto bits = static_cast<std::uint16_t>(
            upper ? expected >> 16 : expected & 0xffffU);
        Value current;
        if constexpr (std::is_same_v<Value, __half>) {
            current = __ushort_as_half(bits);
        } else {
            current = __ushort_as_bfloat16(bits);
        }
        const auto sum = store_number<Value>(
            load_number(&current, 0) + load_number(&value, 0));
        std::uint16_t sum_bits = 0;
        if constexpr (std::is_same_v<Value, __half>) {
            sum_bits = __half_as_ushort(sum);
        } else {
            sum_bits = __bfloat16_as_ushort(sum);
        }
        const auto replacement = upper
            ? (expected & 0x0000ffffU) | (static_cast<unsigned int>(sum_bits) << 16)
            : (expected & 0xffff0000U) | static_cast<unsigned int>(sum_bits);
        observed = atomicCAS(word, expected, replacement);
    } while (observed != expected);
}

template <typename Value>
__device__ void atomic_add_value(Value* destination, Value value) {
    if constexpr (std::is_same_v<Value, float>) {
        atomicAdd(destination, value);
    } else if constexpr (std::is_same_v<Value, double>) {
        atomicAdd(destination, value);
    } else if constexpr (std::is_same_v<Value, int>) {
        atomicAdd(destination, value);
    } else if constexpr (std::is_same_v<Value, unsigned int>) {
        atomicAdd(destination, value);
    } else if constexpr (std::is_same_v<Value, unsigned long long>) {
        atomicAdd(destination, value);
    } else if constexpr (std::is_same_v<Value, std::int64_t>) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(destination),
            static_cast<unsigned long long>(value));
    } else if constexpr (std::is_same_v<Value, __half>) {
#if __CUDA_ARCH__ >= 700
        atomicAdd(destination, value);
#else
        atomic_add_16bit(destination, value);
#endif
    } else if constexpr (std::is_same_v<Value, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
        atomicAdd(destination, value);
#else
        atomic_add_16bit(destination, value);
#endif
    }
}

template <typename Value, typename Index>
__global__ void scatter_kernel(
    TensorView destination,
    TensorView indices,
    TensorView source,
    std::int64_t elements,
    int dimension,
    bool add) {
    auto* output = static_cast<Value*>(destination.data);
    const auto* input = static_cast<const Value*>(source.data);
    const auto* selected = static_cast<const Index*>(indices.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        std::int64_t destination_offset = 0;
        for (std::size_t reverse = source.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            auto coordinate = residual % source.sizes[axis];
            residual /= source.sizes[axis];
            if (static_cast<int>(axis) == dimension) {
                coordinate = load_index(selected, tensor_offset(indices, linear));
            }
            destination_offset += coordinate * destination.strides[axis];
        }
        const auto value = input[tensor_offset(source, linear)];
        if (add) atomic_add_value(output + destination_offset, value);
        else output[destination_offset] = value;
    }
}

template <typename Value, typename Index>
__global__ void index_copy_kernel(
    TensorView destination,
    TensorView indices,
    TensorView source,
    std::int64_t elements,
    int dimension) {
    auto* output = static_cast<Value*>(destination.data);
    const auto* input = static_cast<const Value*>(source.data);
    const auto* selected = static_cast<const Index*>(indices.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        std::int64_t destination_offset = 0;
        for (std::size_t reverse = source.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            auto coordinate = residual % source.sizes[axis];
            residual /= source.sizes[axis];
            if (static_cast<int>(axis) == dimension) {
                coordinate = load_index(selected, coordinate);
            }
            destination_offset += coordinate * destination.strides[axis];
        }
        output[destination_offset] = input[tensor_offset(source, linear)];
    }
}

template <typename Value, typename Index>
__global__ void index_fill_kernel(
    TensorView destination,
    TensorView iteration,
    TensorView indices,
    std::int64_t elements,
    int dimension,
    double fill) {
    auto* output = static_cast<Value*>(destination.data);
    const auto* selected = static_cast<const Index*>(indices.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        std::int64_t destination_offset = 0;
        for (std::size_t reverse = iteration.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            auto coordinate = residual % iteration.sizes[axis];
            residual /= iteration.sizes[axis];
            if (static_cast<int>(axis) == dimension) {
                coordinate = load_index(selected, coordinate);
            }
            destination_offset += coordinate * destination.strides[axis];
        }
        output[destination_offset] = store_number<Value>(fill);
    }
}

template <typename Value>
__global__ void masked_fill_kernel(
    TensorView destination,
    TensorView mask,
    std::int64_t elements,
    double fill) {
    auto* output = static_cast<Value*>(destination.data);
    const auto* selected = static_cast<const bool*>(mask.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        if (selected[tensor_offset(mask, linear)]) {
            output[tensor_offset(destination, linear)] = store_number<Value>(fill);
        }
    }
}

template <typename Function>
void dispatch_index_type(ScalarType type, Function&& function) {
    if (type == kInt64) function.template operator()<std::int64_t>();
    else if (type == kInt32) function.template operator()<std::int32_t>();
    else throw std::invalid_argument("index tensor must be int32 or int64");
}

}  // namespace

Tensor index_select_cuda(
    const Tensor& source,
    std::int64_t dimension,
    const Tensor& indices_source) {
    if (!source.is_cuda() || indices_source.dim() != 1) {
        throw std::invalid_argument("index_select expects CUDA source and one-dimensional indices");
    }
    auto indices = indices_source.device() == source.device()
        ? indices_source.contiguous()
        : indices_source.to(source.device()).contiguous();
    const auto selected = normalize_dimension(dimension, source.dim());
    auto shape = source.sizes().vec();
    shape[selected] = indices.numel();
    auto output = empty(shape, source.options());
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream = current_stream(source.get_device()).stream();
    auto launch_value = [&]<typename Value>() {
        auto launch_index = [&]<typename Index>() {
            index_select_kernel<Value, Index><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), source.view_descriptor(),
                indices.view_descriptor(), output.numel(), static_cast<int>(selected));
        };
        dispatch_index_type(indices.scalar_type(), launch_index);
    };
    dispatch_numeric(source.scalar_type(), launch_value);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor gather_cuda(
    const Tensor& source,
    std::int64_t dimension,
    const Tensor& indices_source) {
    if (!source.is_cuda() || source.dim() != indices_source.dim()) {
        throw std::invalid_argument("gather requires matching CUDA tensor ranks");
    }
    auto indices = indices_source.device() == source.device()
        ? indices_source.contiguous()
        : indices_source.to(source.device()).contiguous();
    const auto selected = normalize_dimension(dimension, source.dim());
    auto output = empty(indices.sizes(), source.options());
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream = current_stream(source.get_device()).stream();
    auto launch_value = [&]<typename Value>() {
        auto launch_index = [&]<typename Index>() {
            gather_kernel<Value, Index><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), source.view_descriptor(),
                indices.view_descriptor(), output.numel(), static_cast<int>(selected));
        };
        dispatch_index_type(indices.scalar_type(), launch_index);
    };
    dispatch_numeric(source.scalar_type(), launch_value);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor repeat_cuda(const Tensor& source, std::span<const std::int64_t> repeats) {
    if (!source.is_cuda() || repeats.size() < static_cast<std::size_t>(source.dim())) {
        throw std::invalid_argument("repeat geometry is invalid");
    }
    auto aligned = source;
    while (aligned.dim() < static_cast<std::int64_t>(repeats.size())) aligned = aligned.unsqueeze(0);
    auto shape = aligned.sizes().vec();
    for (std::size_t index = 0; index < repeats.size(); ++index) {
        if (repeats[index] < 0) throw std::invalid_argument("repeat count cannot be negative");
        shape[index] *= repeats[index];
    }
    auto output = empty(shape, source.options());
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream = current_stream(source.get_device()).stream();
    auto launch = [&]<typename Value>() {
        repeat_kernel<Value><<<blocks, threads, 0, stream>>>(
            output.view_descriptor(), aligned.view_descriptor(), output.numel());
    };
    dispatch_numeric(source.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor repeat_interleave_cuda(
    const Tensor& source,
    std::int64_t repeats,
    std::int64_t dimension) {
    if (!source.is_cuda() || repeats < 0) {
        throw std::invalid_argument("repeat_interleave geometry is invalid");
    }
    const auto selected = normalize_dimension(dimension, source.dim());
    auto shape = source.sizes().vec();
    shape[selected] *= repeats;
    auto output = empty(shape, source.options());
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream = current_stream(source.get_device()).stream();
    auto launch = [&]<typename Value>() {
        repeat_interleave_kernel<Value><<<blocks, threads, 0, stream>>>(
            output.view_descriptor(), source.view_descriptor(), output.numel(),
            static_cast<int>(selected), repeats);
    };
    dispatch_numeric(source.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

void scatter_cuda(
    Tensor& destination,
    std::int64_t dimension,
    const Tensor& index_source,
    const Tensor& source_value,
    bool add) {
    if (!destination.is_cuda() || destination.scalar_type() != source_value.scalar_type() ||
        index_source.sizes() != source_value.sizes()) {
        throw std::invalid_argument("scatter tensor geometry or dtype mismatch");
    }
    if (add && destination.scalar_type() != kInt32 &&
        destination.scalar_type() != kInt64 &&
        destination.scalar_type() != kFloat16 &&
        destination.scalar_type() != kBFloat16 &&
        destination.scalar_type() != kFloat32 &&
        destination.scalar_type() != kFloat64) {
        throw std::invalid_argument(
            "native scatter_add supports int32, int64, FP16, BF16, FP32, and FP64");
    }
    auto index = index_source.to(destination.device()).contiguous();
    auto source = source_value.to(destination.device()).contiguous();
    const auto selected = normalize_dimension(dimension, destination.dim());
    const auto [blocks, threads] = launch_geometry(source.numel());
    const auto stream = current_stream(destination.get_device()).stream();
    auto launch_value = [&]<typename Value>() {
        auto launch_index = [&]<typename Index>() {
            scatter_kernel<Value, Index><<<blocks, threads, 0, stream>>>(
                destination.view_descriptor(), index.view_descriptor(),
                source.view_descriptor(), source.numel(), static_cast<int>(selected), add);
        };
        dispatch_index_type(index.scalar_type(), launch_index);
    };
    dispatch_numeric(destination.scalar_type(), launch_value);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
}

void index_copy_cuda(
    Tensor& destination,
    std::int64_t dimension,
    const Tensor& index_source,
    const Tensor& source_value) {
    if (!destination.is_cuda() || destination.scalar_type() != source_value.scalar_type()) {
        throw std::invalid_argument("index_copy tensor geometry or dtype mismatch");
    }
    auto index = index_source.to(destination.device()).contiguous();
    auto source = source_value.to(destination.device()).contiguous();
    const auto selected = normalize_dimension(dimension, destination.dim());
    if (source.size(static_cast<std::int64_t>(selected)) != index.numel()) {
        throw std::invalid_argument("index_copy source extent does not match indices");
    }
    const auto [blocks, threads] = launch_geometry(source.numel());
    const auto stream = current_stream(destination.get_device()).stream();
    auto launch_value = [&]<typename Value>() {
        auto launch_index = [&]<typename Index>() {
            index_copy_kernel<Value, Index><<<blocks, threads, 0, stream>>>(
                destination.view_descriptor(), index.view_descriptor(),
                source.view_descriptor(), source.numel(), static_cast<int>(selected));
        };
        dispatch_index_type(index.scalar_type(), launch_index);
    };
    dispatch_numeric(destination.scalar_type(), launch_value);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
}

void index_fill_cuda(
    Tensor& destination,
    std::int64_t dimension,
    const Tensor& index_source,
    double value) {
    if (!destination.is_cuda()) throw std::invalid_argument("index_fill requires CUDA tensor");
    auto index = index_source.to(destination.device()).contiguous();
    const auto selected = normalize_dimension(dimension, destination.dim());
    auto indexed_shape = destination.sizes().vec();
    indexed_shape[selected] = index.numel();
    const auto elements = std::accumulate(
        indexed_shape.begin(), indexed_shape.end(), std::int64_t{1},
        std::multiplies<>());
    auto indexed_view = make_contiguous_view(
        destination.data_ptr(), indexed_shape, destination.scalar_type(), destination.device());
    const auto [blocks, threads] = launch_geometry(elements);
    const auto stream = current_stream(destination.get_device()).stream();
    auto launch_value = [&]<typename Value>() {
        auto launch_index = [&]<typename Index>() {
            index_fill_kernel<Value, Index><<<blocks, threads, 0, stream>>>(
                destination.view_descriptor(), indexed_view,
                index.view_descriptor(), elements,
                static_cast<int>(selected), value);
        };
        dispatch_index_type(index.scalar_type(), launch_index);
    };
    dispatch_numeric(destination.scalar_type(), launch_value);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
}

void masked_fill_cuda(Tensor& destination, const Tensor& mask_source, double value) {
    if (!destination.is_cuda()) throw std::invalid_argument("masked_fill requires CUDA tensor");
    auto mask = mask_source.to(destination.device(), kBool);
    const auto aligned = align_for_broadcast(mask, destination.sizes());
    const auto [blocks, threads] = launch_geometry(destination.numel());
    const auto stream = current_stream(destination.get_device()).stream();
    auto launch = [&]<typename Value>() {
        masked_fill_kernel<Value><<<blocks, threads, 0, stream>>>(
            destination.view_descriptor(), aligned, destination.numel(), value);
    };
    dispatch_numeric(destination.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
}

Tensor masked_select_cuda(const Tensor& source_value, const Tensor& mask_value) {
    auto source = source_value.contiguous();
    auto mask = mask_value.to(source.device(), kBool).expand(source.sizes()).contiguous();
    auto mask_host = mask.to(kCPU).contiguous();
    const auto* flags = mask_host.data_ptr<bool>();
    std::int64_t selected = 0;
    for (std::int64_t index = 0; index < mask_host.numel(); ++index) selected += flags[index];
    auto output = empty({selected}, source.options());
    if (selected == 0) return output;
    const auto stream = current_stream(source.get_device()).stream();
    auto context = default_context(source.get_device());
    auto launch = [&]<typename Value>() {
        std::size_t temporary_bytes = 0;
        cub::DeviceSelect::Flagged(
            nullptr, temporary_bytes,
            source.data_ptr<Value>(), mask.data_ptr<bool>(), output.data_ptr<Value>(),
            static_cast<int*>(nullptr), source.numel(), stream);
        Buffer temporary(context, temporary_bytes);
        auto selected_count = empty(
            {1}, TensorOptions{}.dtype(kInt32).device(source.device()));
        cub::DeviceSelect::Flagged(
            temporary.data(), temporary_bytes,
            source.data_ptr<Value>(), mask.data_ptr<bool>(), output.data_ptr<Value>(),
            selected_count.data_ptr<int>(), source.numel(), stream);
    };
    dispatch_numeric(source.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

namespace {

template <typename Value>
__global__ void arange_kernel(
    Value* output,
    std::int64_t elements,
    double start,
    double step) {
    for (std::int64_t index =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        output[index] = store_number<Value>(start + step * static_cast<double>(index));
    }
}

template <typename Value>
__global__ void where_kernel(
    TensorView output,
    TensorView condition,
    TensorView yes,
    TensorView no,
    std::int64_t elements) {
    auto* destination = static_cast<Value*>(output.data);
    const auto* selected = static_cast<const bool*>(condition.data);
    const auto* yes_values = static_cast<const Value*>(yes.data);
    const auto* no_values = static_cast<const Value*>(no.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        destination[linear] = selected[tensor_offset(condition, linear)]
            ? yes_values[tensor_offset(yes, linear)]
            : no_values[tensor_offset(no, linear)];
    }
}

__global__ void initialize_sort_indices_kernel(
    std::int64_t* indices,
    std::int64_t elements,
    std::int64_t columns) {
    for (std::int64_t index =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        indices[index] = index % columns;
    }
}

}  // namespace

Tensor cat(std::span<const Tensor> tensors, std::int64_t dimension) {
    if (tensors.empty()) throw std::invalid_argument("cat requires at least one tensor");
    const auto selected = normalize_dimension(dimension, tensors.front().dim());
    auto shape = tensors.front().sizes().vec();
    shape[selected] = 0;
    for (const auto& tensor : tensors) {
        if (tensor.dim() != tensors.front().dim()) {
            throw std::invalid_argument("cat tensor ranks differ");
        }
        for (std::int64_t axis = 0; axis < tensor.dim(); ++axis) {
            if (axis != static_cast<std::int64_t>(selected) &&
                tensor.size(axis) != tensors.front().size(axis)) {
                throw std::invalid_argument("cat non-concatenated extents differ");
            }
        }
        shape[selected] += tensor.size(static_cast<std::int64_t>(selected));
    }
    auto output = empty(shape, tensors.front().options());
    std::int64_t offset = 0;
    for (const auto& tensor : tensors) {
        auto value = tensor.device() == output.device() &&
                     tensor.scalar_type() == output.scalar_type()
            ? tensor
            : tensor.to(output.device(), output.scalar_type());
        output.narrow(
            static_cast<std::int64_t>(selected), offset,
            value.size(static_cast<std::int64_t>(selected))).copy_(value);
        offset += value.size(static_cast<std::int64_t>(selected));
    }
    return output;
}

Tensor cat(std::initializer_list<Tensor> tensors, std::int64_t dimension) {
    return cat(std::span<const Tensor>(tensors.begin(), tensors.size()), dimension);
}

Tensor cat(const std::vector<Tensor>& tensors, std::int64_t dimension) {
    return cat(std::span<const Tensor>(tensors), dimension);
}

Tensor stack(std::span<const Tensor> tensors, std::int64_t dimension) {
    if (tensors.empty()) throw std::invalid_argument("stack requires at least one tensor");
    auto selected = dimension;
    if (selected < 0) selected += tensors.front().dim() + 1;
    if (selected < 0 || selected > tensors.front().dim()) {
        throw std::out_of_range("stack dimension is out of range");
    }
    std::vector<Tensor> expanded;
    expanded.reserve(tensors.size());
    for (const auto& tensor : tensors) expanded.push_back(tensor.unsqueeze(selected));
    return cat(expanded, selected);
}

Tensor stack(std::initializer_list<Tensor> tensors, std::int64_t dimension) {
    return stack(std::span<const Tensor>(tensors.begin(), tensors.size()), dimension);
}

Tensor stack(const std::vector<Tensor>& tensors, std::int64_t dimension) {
    return stack(std::span<const Tensor>(tensors), dimension);
}

Tensor arange(
    std::int64_t start,
    std::int64_t end,
    std::int64_t step,
    const TensorOptions& options) {
    if (step == 0) throw std::invalid_argument("arange step cannot be zero");
    const auto distance = end - start;
    const auto elements = distance == 0 || (distance > 0) != (step > 0)
        ? 0
        : (std::llabs(distance) + std::llabs(step) - 1) / std::llabs(step);
    auto output = empty({elements}, options);
    if (!output.is_cuda()) {
        for (std::int64_t index = 0; index < elements; ++index) {
            switch (output.scalar_type()) {
                case kInt64: output.data_ptr<std::int64_t>()[index] = start + index * step; break;
                case kInt32: output.data_ptr<std::int32_t>()[index] = static_cast<std::int32_t>(start + index * step); break;
                case kFloat32: output.data_ptr<float>()[index] = static_cast<float>(start + index * step); break;
                case kFloat64: output.data_ptr<double>()[index] = static_cast<double>(start + index * step); break;
                default: throw std::invalid_argument("CPU arange dtype is unsupported");
            }
        }
        return output;
    }
    const auto [blocks, threads] = launch_geometry(elements);
    const auto stream = current_stream(output.get_device()).stream();
    auto launch = [&]<typename Value>() {
        arange_kernel<Value><<<blocks, threads, 0, stream>>>(
            output.data_ptr<Value>(), elements,
            static_cast<double>(start), static_cast<double>(step));
    };
    dispatch_numeric(output.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor arange(std::int64_t start, std::int64_t end, const TensorOptions& options) {
    return arange(start, end, 1, options);
}

Tensor arange(std::int64_t end, const TensorOptions& options) {
    return arange(0, end, 1, options);
}

Tensor randn(std::span<const std::int64_t> shape, const TensorOptions& options) {
    const auto elements = std::accumulate(
        shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>());
    std::mt19937_64 generator(static_cast<std::uint64_t>(
        random_seed.fetch_add(1, std::memory_order_relaxed)));
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> values(static_cast<std::size_t>(elements));
    for (auto& value : values) value = distribution(generator);
    auto result = tensor(
        values,
        TensorOptions{}.dtype(kFloat32).device(options.target_device()));
    return result.to(options.scalar_type()).reshape(shape);
}

Tensor randn(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return randn(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}

Tensor randint(
    std::int64_t low,
    std::int64_t high,
    std::span<const std::int64_t> shape,
    const TensorOptions& options) {
    if (high <= low) throw std::invalid_argument("randint range is empty");
    const auto elements = std::accumulate(
        shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>());
    std::mt19937_64 generator(static_cast<std::uint64_t>(
        random_seed.fetch_add(1, std::memory_order_relaxed)));
    std::uniform_int_distribution<std::int64_t> distribution(low, high - 1);
    std::vector<std::int64_t> values(static_cast<std::size_t>(elements));
    for (auto& value : values) value = distribution(generator);
    auto result = tensor(
        values,
        TensorOptions{}.dtype(kInt64).device(options.target_device()));
    return result.to(options.scalar_type()).reshape(shape);
}

Tensor randint(
    std::int64_t high,
    std::span<const std::int64_t> shape,
    const TensorOptions& options) {
    return randint(0, high, shape, options);
}

Tensor randperm(std::int64_t size, const TensorOptions& options) {
    std::vector<std::int64_t> values(static_cast<std::size_t>(size));
    std::iota(values.begin(), values.end(), 0);
    std::mt19937_64 generator(static_cast<std::uint64_t>(
        random_seed.fetch_add(1, std::memory_order_relaxed)));
    std::shuffle(values.begin(), values.end(), generator);
    return tensor(values, TensorOptions{}.dtype(kInt64).device(options.target_device()))
        .to(options.scalar_type());
}

Tensor where(
    const Tensor& condition_source,
    const Tensor& yes_source,
    const Tensor& no_source) {
    auto shape = broadcast_shape(yes_source, no_source);
    auto type = promote(yes_source.scalar_type(), no_source.scalar_type());
    auto yes = yes_source.to(yes_source.device(), type);
    auto no = no_source.to(yes.device(), type);
    auto condition = condition_source.to(yes.device(), kBool);
    auto output = empty(shape, yes.options());
    const auto yes_view = align_for_broadcast(yes, shape);
    const auto no_view = align_for_broadcast(no, shape);
    const auto condition_view = align_for_broadcast(condition, shape);
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream = current_stream(output.get_device()).stream();
    auto launch = [&]<typename Value>() {
        where_kernel<Value><<<blocks, threads, 0, stream>>>(
            output.view_descriptor(), condition_view, yes_view, no_view, output.numel());
    };
    dispatch_numeric(type, launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor where(const Tensor& condition, const Tensor& yes, double no) {
    return where(condition, yes, full_like(yes, no));
}

Tensor where(const Tensor& condition, double yes, const Tensor& no) {
    return where(condition, full_like(no, yes), no);
}

Tensor matmul(const Tensor& left_source, const Tensor& right_source) {
    if (!left_source.defined() || !right_source.defined() ||
        !left_source.is_cuda() || !right_source.is_cuda()) {
        throw std::invalid_argument("native matmul requires CUDA tensors");
    }
    if (left_source.device() != right_source.device()) {
        throw std::invalid_argument("native matmul requires one CUDA device");
    }
    if (left_source.dim() < 1 || right_source.dim() < 1) {
        throw std::invalid_argument("matmul requires tensors with at least one dimension");
    }
    if (left_source.scalar_type() != right_source.scalar_type() ||
        !floating(left_source.scalar_type())) {
        throw std::invalid_argument("native matmul requires matching floating dtypes");
    }
    const bool left_vector = left_source.dim() == 1;
    const bool right_vector = right_source.dim() == 1;
    auto left = left_vector ? left_source.unsqueeze(0) : left_source;
    auto right = right_vector ? right_source.unsqueeze(-1) : right_source;
    if (left.size(-1) != right.size(-2)) {
        throw std::invalid_argument("matmul contraction dimensions do not match");
    }
    if (!matrix_layout_supported(left)) left = left.contiguous();
    if (!matrix_layout_supported(right)) right = right.contiguous();

    const auto batch_shape = matmul_batch_shape(left, right);
    const auto batches = std::accumulate(
        batch_shape.begin(), batch_shape.end(), std::int64_t{1}, std::multiplies<>());
    const auto rows = left.size(-2);
    const auto contraction = left.size(-1);
    const auto columns = right.size(-1);
    auto output_shape = batch_shape;
    output_shape.push_back(rows);
    output_shape.push_back(columns);
    auto output = empty(output_shape, left.options());
    if (output.numel() == 0) {
        if (left_vector) output = output.squeeze(-2);
        if (right_vector) output = output.squeeze(-1);
        return output;
    }

    DeviceGuard guard(left.get_device());
    auto context = default_context(left.get_device());
    const auto stream = current_stream(left.get_device()).stream();
    context->blas().set_stream(stream);
    const auto handle = context->blas().get();
    const auto data_type = cuda_data_type(left.scalar_type());

    const bool left_row_major = left.stride(-1) == 1;
    const bool right_row_major = right.stride(-1) == 1;
    const auto left_operation = left_row_major ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto right_operation = right_row_major ? CUBLAS_OP_N : CUBLAS_OP_T;
    if (rows > std::numeric_limits<int>::max() ||
        columns > std::numeric_limits<int>::max() ||
        contraction > std::numeric_limits<int>::max()) {
        throw std::overflow_error("matmul dimension exceeds cuBLAS integer ABI");
    }
    const int left_leading = static_cast<int>(left_row_major ? contraction : rows);
    const int right_leading = static_cast<int>(right_row_major ? columns : contraction);

    const char* parallel_batch_disabled =
        std::getenv("MFQ_DISABLE_NATIVE_PARALLEL_BATCH_MATMUL");
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool parallel_batch_eligible =
        batches >= static_cast<std::int64_t>(
            ParallelBatchMatmulContext::kStreams) &&
        rows >= 32 &&
        (parallel_batch_disabled == nullptr ||
         parallel_batch_disabled[0] != '1');
    if (parallel_batch_eligible) {
        MFQ_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    }
    const bool parallel_batch_enabled =
        parallel_batch_eligible &&
        capture_status == cudaStreamCaptureStatusNone;
    if (parallel_batch_enabled) {
        auto& parallel = parallel_batch_matmul_context(left.get_device());
        std::lock_guard<std::mutex> lock(parallel.mutex);
        parallel.ready.record(stream);
        for (auto& worker : parallel.streams) {
            worker.wait(parallel.ready);
        }
        for (std::int64_t batch = 0; batch < batches; ++batch) {
            const auto coordinates = batch_coordinates(batch, batch_shape);
            const auto left_offset =
                matmul_batch_offset(left, batch_shape, coordinates);
            const auto right_offset =
                matmul_batch_offset(right, batch_shape, coordinates);
            const auto* left_pointer =
                static_cast<const std::byte*>(left.data_ptr()) +
                left_offset * left.element_size();
            const auto* right_pointer =
                static_cast<const std::byte*>(right.data_ptr()) +
                right_offset * right.element_size();
            auto* output_pointer = static_cast<std::byte*>(output.data_ptr()) +
                batch * rows * columns * output.element_size();
            const auto worker = static_cast<std::size_t>(batch) %
                ParallelBatchMatmulContext::kStreams;
            const auto worker_handle = parallel.handles[worker].get();
            if (left.scalar_type() == kFloat64) {
                const double alpha = 1.0;
                const double beta = 0.0;
                MFQ_NATIVE_CUDA_CHECK(cublasGemmEx(
                    worker_handle,
                    right_operation, left_operation,
                    static_cast<int>(columns), static_cast<int>(rows),
                    static_cast<int>(contraction),
                    &alpha,
                    right_pointer, data_type, right_leading,
                    left_pointer, data_type, left_leading,
                    &beta,
                    output_pointer, data_type, static_cast<int>(columns),
                    CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
            } else {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                MFQ_NATIVE_CUDA_CHECK(cublasGemmEx(
                    worker_handle,
                    right_operation, left_operation,
                    static_cast<int>(columns), static_cast<int>(rows),
                    static_cast<int>(contraction),
                    &alpha,
                    right_pointer, data_type, right_leading,
                    left_pointer, data_type, left_leading,
                    &beta,
                    output_pointer, data_type, static_cast<int>(columns),
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }
        for (std::size_t worker = 0;
             worker < ParallelBatchMatmulContext::kStreams; ++worker) {
            parallel.done[worker].record(parallel.streams[worker].get());
            MFQ_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(
                stream, parallel.done[worker].get(), 0));
        }
        if (left_vector) output = output.squeeze(-2);
        if (right_vector) output = output.squeeze(-1);
        return output;
    }

    for (std::int64_t batch = 0; batch < batches; ++batch) {
        const auto coordinates = batch_coordinates(batch, batch_shape);
        const auto left_offset = matmul_batch_offset(left, batch_shape, coordinates);
        const auto right_offset = matmul_batch_offset(right, batch_shape, coordinates);
        const auto* left_pointer = static_cast<const std::byte*>(left.data_ptr()) +
            left_offset * left.element_size();
        const auto* right_pointer = static_cast<const std::byte*>(right.data_ptr()) +
            right_offset * right.element_size();
        auto* output_pointer = static_cast<std::byte*>(output.data_ptr()) +
            batch * rows * columns * output.element_size();
        if (left.scalar_type() == kFloat64) {
            const double alpha = 1.0;
            const double beta = 0.0;
            MFQ_NATIVE_CUDA_CHECK(cublasGemmEx(
                handle,
                right_operation, left_operation,
                static_cast<int>(columns), static_cast<int>(rows),
                static_cast<int>(contraction),
                &alpha,
                right_pointer, data_type, right_leading,
                left_pointer, data_type, left_leading,
                &beta,
                output_pointer, data_type, static_cast<int>(columns),
                CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
        } else {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            MFQ_NATIVE_CUDA_CHECK(cublasGemmEx(
                handle,
                right_operation, left_operation,
                static_cast<int>(columns), static_cast<int>(rows),
                static_cast<int>(contraction),
                &alpha,
                right_pointer, data_type, right_leading,
                left_pointer, data_type, left_leading,
                &beta,
                output_pointer, data_type, static_cast<int>(columns),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
    }
    if (left_vector) output = output.squeeze(-2);
    if (right_vector) output = output.squeeze(-1);
    return output;
}

Tensor bmm(const Tensor& left, const Tensor& right) {
    if (left.dim() != 3 || right.dim() != 3 || left.size(0) != right.size(0)) {
        throw std::invalid_argument("bmm requires equally batched rank-three tensors");
    }
    return matmul(left, right);
}

Tensor baddbmm(const Tensor& input, const Tensor& left, const Tensor& right) {
    return input + bmm(left, right);
}

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
    if (weight.dim() != 2) throw std::invalid_argument("linear weight must be a matrix");
    auto output = matmul(input, weight.transpose(0, 1));
    if (bias.has_value() && bias->defined()) output = output + *bias;
    return output;
}

Tensor scaled_dot_product_attention(
    const Tensor& query_source,
    const Tensor& key_source,
    const Tensor& value_source,
    const std::optional<Tensor>& mask,
    double dropout,
    bool causal,
    const std::optional<double>& scale,
    bool enable_grouped_query_attention) {
    if (dropout != 0.0) {
        throw std::invalid_argument("native inference attention does not support dropout");
    }
    if (query_source.dim() < 3 || key_source.dim() != query_source.dim() ||
        value_source.dim() != query_source.dim()) {
        throw std::invalid_argument("attention tensors have incompatible ranks");
    }
    auto key = key_source;
    auto value = value_source;
    const auto head_dimension = query_source.dim() - 3;
    if (query_source.size(head_dimension) != key.size(head_dimension)) {
        if (!enable_grouped_query_attention ||
            query_source.size(head_dimension) % key.size(head_dimension) != 0) {
            throw std::invalid_argument("attention head counts are incompatible");
        }
        const auto repeat = query_source.size(head_dimension) / key.size(head_dimension);
        key = key.repeat_interleave(repeat, head_dimension).contiguous();
        value = value.repeat_interleave(repeat, head_dimension).contiguous();
    }
    const auto factor = scale.value_or(
        1.0 / std::sqrt(static_cast<double>(query_source.size(-1))));
    auto scores = matmul(query_source, key.transpose(-2, -1)) * factor;
    if (causal) {
        if (mask.has_value()) {
            throw std::invalid_argument("attention cannot combine explicit and causal masks");
        }
        auto rows = arange(
            scores.size(-2), TensorOptions{}.dtype(kInt64).device(scores.device()))
            .unsqueeze(-1);
        auto columns = arange(
            scores.size(-1), TensorOptions{}.dtype(kInt64).device(scores.device()))
            .unsqueeze(0);
        scores = where(
            columns <= rows,
            scores,
            -std::numeric_limits<double>::infinity());
    } else if (mask.has_value()) {
        auto selected = mask->to(scores.device());
        if (selected.scalar_type() == kBool) {
            scores = where(
                selected,
                scores,
                -std::numeric_limits<double>::infinity());
        } else {
            scores = scores + selected.to(scores.scalar_type());
        }
    }
    auto probabilities = softmax(scores, -1).to(value.scalar_type());
    return matmul(probabilities, value);
}

Tensor conv1d(
    const Tensor& input_source,
    const Tensor& weight_source,
    const Tensor& bias_source,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding,
    std::span<const std::int64_t> dilation,
    std::int64_t groups) {
    if (input_source.dim() != 3 || weight_source.dim() != 3 || groups <= 0 ||
        stride.size() != 1 || padding.size() != 1 || dilation.size() != 1) {
        throw std::invalid_argument("conv1d geometry is invalid");
    }
    auto input = input_source.contiguous();
    auto weight = weight_source.to(input.device(), input.scalar_type()).contiguous();
    auto bias = bias_source.defined()
        ? bias_source.to(input.device(), input.scalar_type()).contiguous()
        : Tensor{};
    if (!floating(input.scalar_type()) || input.size(1) % groups != 0 ||
        weight.size(0) % groups != 0 || weight.size(1) != input.size(1) / groups) {
        throw std::invalid_argument("conv1d channel geometry is invalid");
    }
    const auto output_length =
        (input.size(2) + 2 * padding[0] - dilation[0] * (weight.size(2) - 1) - 1) /
            stride[0] + 1;
    if (output_length < 0) throw std::invalid_argument("conv1d output is empty");
    auto output = empty(
        {input.size(0), weight.size(0), output_length}, input.options());
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream_value = current_stream(input.get_device()).stream();
    auto launch = [&]<typename Value>() {
        conv1d_kernel<Value><<<blocks, threads, 0, stream_value>>>(
            output.view_descriptor(), input.view_descriptor(), weight.view_descriptor(),
            bias.defined() ? bias.view_descriptor() : TensorView{}, bias.defined(),
            output.numel(), output_length, stride[0], padding[0], dilation[0], groups);
    };
    dispatch_numeric(input.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor conv2d(
    const Tensor& input_source,
    const Tensor& weight_source,
    const Tensor& bias_source,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding,
    std::span<const std::int64_t> dilation,
    std::int64_t groups) {
    if (input_source.dim() != 4 || weight_source.dim() != 4 || groups <= 0 ||
        stride.size() != 2 || padding.size() != 2 || dilation.size() != 2) {
        throw std::invalid_argument("conv2d geometry is invalid");
    }
    auto input = input_source.contiguous();
    auto weight = weight_source.to(input.device(), input.scalar_type()).contiguous();
    auto bias = bias_source.defined()
        ? bias_source.to(input.device(), input.scalar_type()).contiguous()
        : Tensor{};
    if (!floating(input.scalar_type()) || input.size(1) % groups != 0 ||
        weight.size(0) % groups != 0 || weight.size(1) != input.size(1) / groups) {
        throw std::invalid_argument("conv2d channel geometry is invalid");
    }
    const auto output_h =
        (input.size(2) + 2 * padding[0] - dilation[0] * (weight.size(2) - 1) - 1) /
            stride[0] + 1;
    const auto output_w =
        (input.size(3) + 2 * padding[1] - dilation[1] * (weight.size(3) - 1) - 1) /
            stride[1] + 1;
    if (output_h < 0 || output_w < 0) {
        throw std::invalid_argument("conv2d output is empty");
    }
    auto output = empty(
        {input.size(0), weight.size(0), output_h, output_w}, input.options());
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream_value = current_stream(input.get_device()).stream();
    auto launch = [&]<typename Value>() {
        conv2d_kernel<Value><<<blocks, threads, 0, stream_value>>>(
            output.view_descriptor(), input.view_descriptor(), weight.view_descriptor(),
            bias.defined() ? bias.view_descriptor() : TensorView{}, bias.defined(),
            output.numel(), stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1], groups);
    };
    dispatch_numeric(input.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor avg_pool1d(
    const Tensor& input_source,
    std::span<const std::int64_t> kernel,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding) {
    if (input_source.dim() != 3 || kernel.size() != 1 || stride.size() != 1 ||
        (!padding.empty() && padding.size() != 1)) {
        throw std::invalid_argument("avg_pool1d geometry is invalid");
    }
    auto input = input_source.contiguous();
    const auto selected_padding = padding.empty() ? 0 : padding[0];
    const auto output_length =
        (input.size(2) + 2 * selected_padding - kernel[0]) / stride[0] + 1;
    auto output = empty(
        {input.size(0), input.size(1), output_length}, input.options());
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream_value = current_stream(input.get_device()).stream();
    auto launch = [&]<typename Value>() {
        avg_pool1d_kernel<Value><<<blocks, threads, 0, stream_value>>>(
            output.view_descriptor(), input.view_descriptor(), output.numel(),
            kernel[0], stride[0], selected_padding);
    };
    dispatch_numeric(input.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor cumsum(const Tensor& input_source, std::int64_t dimension) {
    auto input = input_source.contiguous();
    const auto selected = normalize_dimension(dimension, input.dim());
    std::int64_t outer = 1;
    std::int64_t inner = 1;
    for (std::size_t axis = 0; axis < selected; ++axis) outer *= input.size(axis);
    for (std::size_t axis = selected + 1; axis < static_cast<std::size_t>(input.dim()); ++axis) {
        inner *= input.size(axis);
    }
    auto output = empty(input.sizes(), input.options());
    const auto [blocks, threads] = launch_geometry(outer * inner);
    const auto stream_value = current_stream(input.get_device()).stream();
    auto launch = [&]<typename Value>() {
        cumsum_kernel<Value><<<blocks, threads, 0, stream_value>>>(
            output.view_descriptor(), input.view_descriptor(),
            outer, input.size(selected), inner);
    };
    dispatch_numeric(input.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor multinomial(
    const Tensor& probabilities_source,
    std::int64_t samples,
    bool replacement) {
    if (samples != 1 || replacement) {
        throw std::invalid_argument(
            "native inference multinomial currently supports one sample without replacement");
    }
    auto probabilities = probabilities_source.contiguous();
    if (!probabilities.is_cuda() || !floating(probabilities.scalar_type()) ||
        probabilities.dim() < 1 || probabilities.size(-1) <= 0) {
        throw std::invalid_argument("multinomial probabilities are invalid");
    }
    const auto columns = probabilities.size(-1);
    const auto rows = probabilities.numel() / columns;
    auto shape = probabilities.sizes().vec();
    shape.back() = 1;
    auto output = empty(shape, probabilities.options().dtype(kInt64));
    std::mt19937_64 generator(static_cast<std::uint64_t>(
        random_seed.fetch_add(1, std::memory_order_relaxed)));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::vector<double> host_uniforms(static_cast<std::size_t>(rows));
    for (auto& value : host_uniforms) value = distribution(generator);
    auto uniforms = tensor(
        host_uniforms,
        TensorOptions{}.dtype(kFloat64).device(probabilities.device()));
    const auto [blocks, threads] = launch_geometry(rows);
    const auto stream_value = current_stream(probabilities.get_device()).stream();
    auto launch = [&]<typename Value>() {
        multinomial_one_kernel<Value><<<blocks, threads, 0, stream_value>>>(
            probabilities.data_ptr<Value>(), output.data_ptr<std::int64_t>(),
            uniforms.data_ptr<double>(), rows, columns);
    };
    dispatch_numeric(probabilities.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor einsum(const std::string& equation, std::span<const Tensor> operands) {
    if (equation != "bmhd,bkd->bmhk" || operands.size() != 2) {
        throw std::invalid_argument("native einsum equation is unsupported");
    }
    const auto& query = operands[0];
    const auto& key = operands[1];
    if (query.dim() != 4 || key.dim() != 3 || query.size(0) != key.size(0) ||
        query.size(3) != key.size(2)) {
        throw std::invalid_argument("native indexer einsum geometry is invalid");
    }
    const auto batch = query.size(0);
    const auto rows = query.size(1);
    const auto heads = query.size(2);
    const auto width = query.size(3);
    const auto keys = key.size(1);
    auto flattened = query.contiguous().reshape({batch, rows * heads, width});
    return bmm(flattened, key.transpose(1, 2))
        .reshape({batch, rows, heads, keys});
}

Tensor logsumexp(const Tensor& input, std::int64_t dimension, bool keep_dimension) {
    auto working = input.scalar_type() == kFloat64
        ? input
        : input.to(kFloat32);
    auto maximum = working.amax(dimension, true);
    auto result = log((working - maximum).exp().sum(dimension, true)) + maximum;
    if (!keep_dimension) result = result.squeeze(dimension);
    return result;
}

Tensor softmax(const Tensor& input, std::int64_t dimension) {
    auto working = input.scalar_type() == kFloat64
        ? input
        : input.to(kFloat32);
    auto maximum = working.amax(dimension, true);
    auto numerator = (working - maximum).exp();
    auto result = numerator / numerator.sum(dimension, true);
    return result.to(input.scalar_type());
}

Tensor log_softmax(const Tensor& input, std::int64_t dimension) {
    auto working = input.scalar_type() == kFloat64
        ? input
        : input.to(kFloat32);
    auto result = working - logsumexp(working, dimension, true);
    return result.to(input.scalar_type());
}

Tensor dot(const Tensor& left, const Tensor& right) {
    if (left.numel() != right.numel()) throw std::invalid_argument("dot size mismatch");
    return (left.reshape({-1}) * right.reshape({-1})).sum();
}

bool equal(const Tensor& left, const Tensor& right) {
    if (left.sizes() != right.sizes() || left.scalar_type() != right.scalar_type()) {
        return false;
    }
    if (left.numel() == 0) return true;
    return !left.ne(right).reshape({-1}).any(0).item<bool>();
}

std::tuple<Tensor, Tensor> max(
    const Tensor& input,
    std::int64_t dimension,
    bool keep_dimension) {
    return {
        input.amax(dimension, keep_dimension),
        input.argmax(dimension, keep_dimension)};
}

std::tuple<Tensor, Tensor> sort(
    const Tensor& input_source,
    std::int64_t dimension,
    bool descending) {
    const auto selected = normalize_dimension(dimension, input_source.dim());
    if (selected != static_cast<std::size_t>(input_source.dim() - 1)) {
        throw std::invalid_argument("native sort currently requires the last dimension");
    }
    auto input = input_source.contiguous();
    const auto columns = input.size(-1);
    const auto rows = input.numel() / columns;
    auto keys = floating(input.scalar_type()) && input.scalar_type() != kFloat64
        ? input.to(kFloat32)
        : input;
    auto output_keys = empty(keys.sizes(), keys.options());
    auto input_indices = empty(keys.sizes(), keys.options().dtype(kInt64));
    auto output_indices = empty(keys.sizes(), keys.options().dtype(kInt64));
    const auto [blocks, threads] = launch_geometry(keys.numel());
    const auto stream = current_stream(keys.get_device()).stream();
    initialize_sort_indices_kernel<<<blocks, threads, 0, stream>>>(
        input_indices.data_ptr<std::int64_t>(), keys.numel(), columns);
    std::vector<std::int32_t> offsets(static_cast<std::size_t>(rows + 1));
    for (std::int64_t row = 0; row <= rows; ++row) {
        offsets[static_cast<std::size_t>(row)] = static_cast<std::int32_t>(row * columns);
    }
    auto segment_offsets = tensor(
        offsets, TensorOptions{}.dtype(kInt32).device(keys.device()));
    auto context = default_context(keys.get_device());
    auto launch = [&]<typename Value>() {
        std::size_t bytes = 0;
        if (descending) {
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                nullptr, bytes, keys.data_ptr<Value>(), output_keys.data_ptr<Value>(),
                input_indices.data_ptr<std::int64_t>(), output_indices.data_ptr<std::int64_t>(),
                keys.numel(), rows, segment_offsets.data_ptr<std::int32_t>(),
                segment_offsets.data_ptr<std::int32_t>() + 1, 0, sizeof(Value) * 8, stream);
        } else {
            cub::DeviceSegmentedRadixSort::SortPairs(
                nullptr, bytes, keys.data_ptr<Value>(), output_keys.data_ptr<Value>(),
                input_indices.data_ptr<std::int64_t>(), output_indices.data_ptr<std::int64_t>(),
                keys.numel(), rows, segment_offsets.data_ptr<std::int32_t>(),
                segment_offsets.data_ptr<std::int32_t>() + 1, 0, sizeof(Value) * 8, stream);
        }
        Buffer temporary(context, bytes);
        if (descending) {
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                temporary.data(), bytes, keys.data_ptr<Value>(), output_keys.data_ptr<Value>(),
                input_indices.data_ptr<std::int64_t>(), output_indices.data_ptr<std::int64_t>(),
                keys.numel(), rows, segment_offsets.data_ptr<std::int32_t>(),
                segment_offsets.data_ptr<std::int32_t>() + 1, 0, sizeof(Value) * 8, stream);
        } else {
            cub::DeviceSegmentedRadixSort::SortPairs(
                temporary.data(), bytes, keys.data_ptr<Value>(), output_keys.data_ptr<Value>(),
                input_indices.data_ptr<std::int64_t>(), output_indices.data_ptr<std::int64_t>(),
                keys.numel(), rows, segment_offsets.data_ptr<std::int32_t>(),
                segment_offsets.data_ptr<std::int32_t>() + 1, 0, sizeof(Value) * 8, stream);
        }
    };
    dispatch_numeric(keys.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return {output_keys.to(input_source.scalar_type()), output_indices};
}

std::tuple<Tensor, Tensor> topk(
    const Tensor& input,
    std::int64_t count,
    std::int64_t dimension,
    bool largest,
    bool) {
    auto [values, indices] = sort(input, dimension, largest);
    if (count < 0 || count > values.size(dimension)) {
        throw std::invalid_argument("topk count is out of range");
    }
    return {
        values.narrow(dimension, 0, count).contiguous(),
        indices.narrow(dimension, 0, count).contiguous()};
}

Tensor layer_norm(
    const Tensor& input,
    std::span<const std::int64_t> normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double epsilon) {
    if (normalized_shape.size() != 1 || normalized_shape.front() != input.size(-1)) {
        throw std::invalid_argument("native layer_norm currently supports the final dimension");
    }
    auto working = input.to(kFloat32);
    auto average = working.mean(-1, true);
    auto variance = (working - average).square().mean(-1, true);
    auto result = (working - average) * rsqrt(variance + epsilon);
    if (weight.defined()) result = result * weight.to(result.device(), kFloat32);
    if (bias.defined()) result = result + bias.to(result.device(), kFloat32);
    return result.to(input.scalar_type());
}

Tensor constant_pad_nd(
    const Tensor& input,
    std::span<const std::int64_t> padding,
    double value) {
    if (padding.size() % 2 != 0 || padding.size() / 2 > static_cast<std::size_t>(input.dim())) {
        throw std::invalid_argument("constant_pad_nd padding rank is invalid");
    }
    auto shape = input.sizes().vec();
    for (std::size_t pair = 0; pair < padding.size() / 2; ++pair) {
        const auto dimension = shape.size() - 1 - pair;
        shape[dimension] += padding[2 * pair] + padding[2 * pair + 1];
    }
    auto output = full(shape, value, input.options());
    auto target = output;
    for (std::size_t pair = 0; pair < padding.size() / 2; ++pair) {
        const auto dimension = static_cast<std::int64_t>(shape.size() - 1 - pair);
        target = target.narrow(dimension, padding[2 * pair], input.size(dimension));
    }
    target.copy_(input);
    return output;
}

}  // namespace mfq::cuda
