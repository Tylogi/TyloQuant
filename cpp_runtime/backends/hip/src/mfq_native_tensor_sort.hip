#include "hip/hip_runtime.h"
#include "mfq_native_tensor_ops_common.cuh"

#include <hipcub/hipcub.hpp>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace mfq::cuda {
using namespace native_ops_detail;
namespace {
__global__ void exact_bf16_softmax_max_kernel(
    const __hip_bfloat16* input,
    float* maximum,
    std::int64_t rows,
    std::int64_t columns) {
    for (std::int64_t row =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        float value = std::numeric_limits<float>::lowest();
        const auto* row_input = input + row * columns;
        for (std::int64_t column = 0; column < columns; ++column) {
            const float candidate = __bfloat162float(row_input[column]);
            if (candidate > value) value = candidate;
        }
        maximum[row] = value;
    }
}

__global__ void exact_bf16_softmax_numerator_kernel(
    const __hip_bfloat16* input,
    const float* maximum,
    float* numerator,
    std::int64_t elements,
    std::int64_t columns) {
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const auto row = linear / columns;
        const float shifted = static_cast<float>(
            static_cast<double>(__bfloat162float(input[linear])) -
            static_cast<double>(maximum[row]));
        numerator[linear] = static_cast<float>(
            ::exp(static_cast<double>(shifted)));
    }
}

__global__ void exact_bf16_softmax_sum_kernel(
    const float* numerator,
    float* denominator,
    std::int64_t rows,
    std::int64_t columns) {
    for (std::int64_t row =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const auto offset = row * columns;
        float value = 0.0f;
        for (std::int64_t column = 0; column < columns; ++column) {
            value += numerator[offset + column];
        }
        denominator[row] = value;
    }
}

__global__ void exact_bf16_softmax_normalize_element_kernel(
    const float* numerator,
    const float* denominator,
    __hip_bfloat16* output,
    std::int64_t elements,
    std::int64_t columns) {
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        output[linear] = store_number<__hip_bfloat16>(
            static_cast<double>(numerator[linear]) /
            static_cast<double>(denominator[linear / columns]));
    }
}

template <int Threads>
__device__ float block_reduce_max(float value, float* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int offset = Threads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] = ::fmaxf(
                shared[threadIdx.x], shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
}

template <int Threads>
__device__ float block_reduce_sum(float value, float* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int offset = Threads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
}

template <typename Value, int Threads>
__global__ void row_softmax_last_contiguous_kernel(
    const Value* __restrict__ input,
    Value* __restrict__ output,
    std::int64_t rows,
    std::int64_t columns) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto* row_input = input + row * columns;
    auto* row_output = output + row * columns;
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::int64_t column = threadIdx.x;
         column < columns;
         column += Threads) {
        maximum = ::fmaxf(maximum, static_cast<float>(
            load_number(row_input, column)));
    }
    __shared__ float partial[Threads];
    maximum = block_reduce_max<Threads>(maximum, partial);
    float sum = 0.0f;
    for (std::int64_t column = threadIdx.x;
         column < columns;
         column += Threads) {
        sum += ::expf(static_cast<float>(
            load_number(row_input, column)) - maximum);
    }
    sum = block_reduce_sum<Threads>(sum, partial);
    const float inverse = 1.0f / sum;
    for (std::int64_t column = threadIdx.x;
         column < columns;
         column += Threads) {
        row_output[column] = store_number<Value>(::expf(
            static_cast<float>(load_number(row_input, column)) - maximum) *
            inverse);
    }
}

template <typename Value, int Threads>
__global__ void row_log_softmax_last_contiguous_kernel(
    const Value* __restrict__ input,
    Value* __restrict__ output,
    std::int64_t rows,
    std::int64_t columns) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto* row_input = input + row * columns;
    auto* row_output = output + row * columns;
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::int64_t column = threadIdx.x;
         column < columns;
         column += Threads) {
        maximum = ::fmaxf(maximum, static_cast<float>(
            load_number(row_input, column)));
    }
    __shared__ float partial[Threads];
    maximum = block_reduce_max<Threads>(maximum, partial);
    float sum = 0.0f;
    for (std::int64_t column = threadIdx.x;
         column < columns;
         column += Threads) {
        sum += ::expf(static_cast<float>(
            load_number(row_input, column)) - maximum);
    }
    sum = block_reduce_sum<Threads>(sum, partial);
    const float shift = maximum + ::logf(sum);
    for (std::int64_t column = threadIdx.x;
         column < columns;
         column += Threads) {
        row_output[column] = store_number<Value>(
            static_cast<float>(load_number(row_input, column)) - shift);
    }
}

template <typename Value>
void launch_row_softmax(
    const Value* input,
    Value* output,
    std::int64_t rows,
    std::int64_t columns,
    bool logarithmic,
    hipStream_t stream) {
    const auto launch = [&]<int Threads>() {
        if (logarithmic) {
            row_log_softmax_last_contiguous_kernel<Value, Threads>
                <<<static_cast<unsigned int>(rows), Threads, 0, stream>>>(
                    input, output, rows, columns);
        } else {
            row_softmax_last_contiguous_kernel<Value, Threads>
                <<<static_cast<unsigned int>(rows), Threads, 0, stream>>>(
                    input, output, rows, columns);
        }
    };
    if (columns <= 32) launch.template operator()<32>();
    else if (columns <= 64) launch.template operator()<64>();
    else if (columns <= 128) launch.template operator()<128>();
    else launch.template operator()<256>();
}

template <bool Descending, int Items>
__global__ void topk_tile_kernel(
    const float* source, const std::int64_t* source_indices,
    float* output, std::int64_t* output_indices,
    std::int64_t rows, std::int64_t columns, int count) {
    constexpr int threads = 256, items = Items, tile = threads * items;
    using Sort = hipcub::BlockRadixSort<float, threads, items, std::int64_t>;
    extern __shared__ __align__(16) unsigned char temporary[];
    auto& storage = *reinterpret_cast<typename Sort::TempStorage*>(temporary);
    const auto tiles = (columns + tile - 1) / tile;
    for (std::int64_t task = blockIdx.x; task < rows * tiles; task += gridDim.x) {
        const auto row = task / tiles;
        const auto begin = (task % tiles) * tile;
        float keys[items];
        std::int64_t indices[items];
#pragma unroll
        for (int i = 0; i < items; ++i) {
            const auto column = begin + threadIdx.x * items + i;
            // Extremal radix keys include signed NaNs, unlike +/- infinity.
            keys[i] = column < columns ? source[row * columns + column]
                : __uint_as_float(Descending ? 0xffffffffu : 0x7fffffffu);
            indices[i] = column < columns
                ? (source_indices ? source_indices[row * columns + column] : column)
                : std::numeric_limits<std::int64_t>::max();
        }
        if constexpr (Descending) Sort(storage).SortDescending(keys, indices);
        else Sort(storage).Sort(keys, indices);
#pragma unroll
        for (int i = 0; i < items; ++i) {
            const int rank = threadIdx.x * items + i;
            if (rank < count) {
                output[task * count + rank] = keys[i];
                output_indices[task * count + rank] = indices[i];
            }
        }
        __syncthreads();
    }
}

__global__ void initialize_sort_indices_kernel(
    std::int64_t* indices,
    std::int32_t* segment_offsets,
    std::int64_t elements,
    std::int64_t columns) {
    for (std::int64_t index =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        indices[index] = index % columns;
    }
    for (std::int64_t row =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row <= elements / columns;
         row += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        segment_offsets[row] = static_cast<std::int32_t>(row * columns);
    }
}

}  // namespace

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
    const auto selected = normalize_dimension(dimension, input.dim());
    const auto columns = input.size(-1);
    const auto rows = columns == 0 ? 0 : input.numel() / columns;
    bool exact_bf16_disabled = false;
    if (input.scalar_type() == kBFloat16) {
        const char* value =
            std::getenv("MFQ_DISABLE_NATIVE_EXACT_BF16_SOFTMAX");
        exact_bf16_disabled = value != nullptr && value[0] == '1';
    }
    const bool row_softmax = input.is_cuda() && input.is_contiguous() &&
        selected + 1 == static_cast<std::size_t>(input.dim()) &&
        (input.scalar_type() == kFloat32 ||
         input.scalar_type() == kFloat16 ||
         input.scalar_type() == kBFloat16) &&
        (input.scalar_type() != kBFloat16 || exact_bf16_disabled) &&
        columns > 0 &&
        rows > 0 &&
        rows <= std::numeric_limits<unsigned int>::max();
    if (row_softmax) {
        auto output = empty(input.sizes(), input.options());
        const auto stream = current_stream(input.get_device()).stream();
        if (input.scalar_type() == kFloat32) {
            launch_row_softmax(
                input.data_ptr<float>(), output.data_ptr<float>(),
                rows, columns, false, stream);
        } else if (input.scalar_type() == kFloat16) {
            launch_row_softmax(
                    static_cast<const __half*>(input.data_ptr()),
                    static_cast<__half*>(output.data_ptr()),
                    rows, columns, false, stream);
        } else {
            launch_row_softmax(
                    static_cast<const __hip_bfloat16*>(input.data_ptr()),
                    static_cast<__hip_bfloat16*>(output.data_ptr()),
                    rows, columns, false, stream);
        }
        MFQ_NATIVE_CUDA_CHECK(hipGetLastError());
        return output;
    }
    const bool exact_bf16 = input.is_cuda() &&
        input.scalar_type() == kBFloat16 && input.is_contiguous() &&
        selected + 1 == static_cast<std::size_t>(input.dim()) &&
        input.size(-1) >= 32 &&
        !exact_bf16_disabled;
    if (exact_bf16) {
        auto maximum = empty(
            {rows}, input.options().dtype(kFloat32));
        auto numerator = empty(
            input.sizes(), input.options().dtype(kFloat32));
        auto output = empty(input.sizes(), input.options());
        const auto stream = current_stream(input.get_device()).stream();
        const auto [row_blocks, row_threads] = launch_geometry(rows);
        exact_bf16_softmax_max_kernel<<<
            row_blocks, row_threads, 0, stream>>>(
            static_cast<const __hip_bfloat16*>(input.data_ptr()),
            maximum.data_ptr<float>(), rows, columns);
        const auto [element_blocks, element_threads] =
            launch_geometry(input.numel());
        exact_bf16_softmax_numerator_kernel<<<
            element_blocks, element_threads, 0, stream>>>(
            static_cast<const __hip_bfloat16*>(input.data_ptr()),
            maximum.data_ptr<float>(), numerator.data_ptr<float>(),
            input.numel(), columns);
        auto denominator = empty(
            {rows}, input.options().dtype(kFloat32));
        exact_bf16_softmax_sum_kernel<<<
            row_blocks, row_threads, 0, stream>>>(
            numerator.data_ptr<float>(), denominator.data_ptr<float>(),
            rows, columns);
        exact_bf16_softmax_normalize_element_kernel<<<
            element_blocks, element_threads, 0, stream>>>(
            numerator.data_ptr<float>(), denominator.data_ptr<float>(),
            static_cast<__hip_bfloat16*>(output.data_ptr()),
            input.numel(), columns);
        MFQ_NATIVE_CUDA_CHECK(hipGetLastError());
        return output;
    }
    auto working = input.scalar_type() == kFloat64
        ? input
        : input.to(kFloat32);
    auto maximum = working.amax(dimension, true);
    auto numerator = (working - maximum).exp();
    auto result = numerator / numerator.sum(dimension, true);
    return result.to(input.scalar_type());
}

Tensor log_softmax(const Tensor& input, std::int64_t dimension) {
    const auto selected = normalize_dimension(dimension, input.dim());
    const auto columns = input.size(-1);
    const auto rows = columns == 0 ? 0 : input.numel() / columns;
    const bool row_log_softmax = input.is_cuda() && input.is_contiguous() &&
        selected + 1 == static_cast<std::size_t>(input.dim()) &&
        (input.scalar_type() == kFloat32 ||
         input.scalar_type() == kFloat16 ||
         input.scalar_type() == kBFloat16) &&
        columns > 0 &&
        rows > 0 &&
        rows <= std::numeric_limits<unsigned int>::max();
    if (row_log_softmax) {
        auto output = empty(input.sizes(), input.options());
        const auto stream = current_stream(input.get_device()).stream();
        if (input.scalar_type() == kFloat32) {
            launch_row_softmax(
                input.data_ptr<float>(), output.data_ptr<float>(),
                rows, columns, true, stream);
        } else if (input.scalar_type() == kFloat16) {
            launch_row_softmax(
                    static_cast<const __half*>(input.data_ptr()),
                    static_cast<__half*>(output.data_ptr()),
                    rows, columns, true, stream);
        } else {
            launch_row_softmax(
                    static_cast<const __hip_bfloat16*>(input.data_ptr()),
                    static_cast<__hip_bfloat16*>(output.data_ptr()),
                    rows, columns, true, stream);
        }
        MFQ_NATIVE_CUDA_CHECK(hipGetLastError());
        return output;
    }
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
    if (input.numel() == 0) {
        return {input.clone(), empty(input.sizes(), input.options().dtype(kInt64))};
    }
    const auto rows = input.numel() / columns;
    auto keys = floating(input.scalar_type()) && input.scalar_type() != kFloat64
        ? input.to(kFloat32)
        : input;
    auto output_keys = empty(keys.sizes(), keys.options());
    auto input_indices = empty(keys.sizes(), keys.options().dtype(kInt64));
    auto output_indices = empty(keys.sizes(), keys.options().dtype(kInt64));
    const auto [blocks, threads] = launch_geometry(keys.numel());
    const auto stream = current_stream(keys.get_device()).stream();
    auto segment_offsets = empty({rows + 1}, keys.options().dtype(kInt32));
    // Offsets are a fixed arithmetic progression. Generate them in the
    // existing initialization launch, avoiding a pinned host allocation,
    // host-to-device transfer and synchronizing hipHostFree on every sort.
    initialize_sort_indices_kernel<<<blocks, threads, 0, stream>>>(
        input_indices.data_ptr<std::int64_t>(), segment_offsets.data_ptr<std::int32_t>(),
        keys.numel(), columns);
    auto context = default_context(keys.get_device());
    auto launch = [&]<typename Value>() {
        std::size_t bytes = 0;
        if (descending) {
            hipcub::DeviceSegmentedRadixSort::SortPairsDescending(
                nullptr, bytes, keys.data_ptr<Value>(), output_keys.data_ptr<Value>(),
                input_indices.data_ptr<std::int64_t>(), output_indices.data_ptr<std::int64_t>(),
                keys.numel(), rows, segment_offsets.data_ptr<std::int32_t>(),
                segment_offsets.data_ptr<std::int32_t>() + 1, 0, sizeof(Value) * 8, stream);
        } else {
            hipcub::DeviceSegmentedRadixSort::SortPairs(
                nullptr, bytes, keys.data_ptr<Value>(), output_keys.data_ptr<Value>(),
                input_indices.data_ptr<std::int64_t>(), output_indices.data_ptr<std::int64_t>(),
                keys.numel(), rows, segment_offsets.data_ptr<std::int32_t>(),
                segment_offsets.data_ptr<std::int32_t>() + 1, 0, sizeof(Value) * 8, stream);
        }
        Buffer temporary(context, bytes);
        if (descending) {
            hipcub::DeviceSegmentedRadixSort::SortPairsDescending(
                temporary.data(), bytes, keys.data_ptr<Value>(), output_keys.data_ptr<Value>(),
                input_indices.data_ptr<std::int64_t>(), output_indices.data_ptr<std::int64_t>(),
                keys.numel(), rows, segment_offsets.data_ptr<std::int32_t>(),
                segment_offsets.data_ptr<std::int32_t>() + 1, 0, sizeof(Value) * 8, stream);
        } else {
            hipcub::DeviceSegmentedRadixSort::SortPairs(
                temporary.data(), bytes, keys.data_ptr<Value>(), output_keys.data_ptr<Value>(),
                input_indices.data_ptr<std::int64_t>(), output_indices.data_ptr<std::int64_t>(),
                keys.numel(), rows, segment_offsets.data_ptr<std::int32_t>(),
                segment_offsets.data_ptr<std::int32_t>() + 1, 0, sizeof(Value) * 8, stream);
        }
    };
    dispatch_numeric(keys.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(hipGetLastError());
    return {output_keys.to(input_source.scalar_type()), output_indices};
}

std::tuple<Tensor, Tensor> topk(
    const Tensor& input,
    std::int64_t count,
    std::int64_t dimension,
    bool largest,
    bool) {
    const auto selected = normalize_dimension(dimension, input.dim());
    if (count < 0 || count > input.size(selected)) {
        throw std::invalid_argument("topk count is out of range");
    }
    auto shape = input.sizes().vec();
    shape[selected] = count;
    if (count == 0 || input.numel() == 0) {
        return {empty(shape, input.options()), empty(shape, input.options().dtype(kInt64))};
    }
    if (input.is_cuda() && selected + 1 == input.dim() &&
        (count <= 256 || (count <= 2048 && count * 4 <= input.size(-1))) &&
        (input.scalar_type() == kFloat32 || input.scalar_type() == kFloat16 ||
         input.scalar_type() == kBFloat16)) {
        DeviceGuard guard(input.get_device());
        auto keys = input.to(kFloat32).contiguous();
        Tensor indices;
        const auto rows = input.numel() / input.size(-1);
        auto columns = input.size(-1);
        const auto stream = current_stream(input.get_device()).stream();
        do {
            const int items = count <= 256 ? 4 : count <= 1024 ? 16 : 32;
            const int tile = 256 * items;
            const auto tiles = (columns + tile - 1) / tile;
            auto next_keys = empty({rows, tiles * count}, keys.options());
            auto next_indices = empty(next_keys.sizes(), keys.options().dtype(kInt64));
            const int blocks = static_cast<int>(std::min<std::int64_t>(rows * tiles, 65535));
            const auto launch = [&]<bool Descending, int Items>() {
                constexpr auto bytes = sizeof(typename hipcub::BlockRadixSort<float, 256, Items, std::int64_t>::TempStorage);
                if constexpr (bytes > 49152) {
                    MFQ_NATIVE_CUDA_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(topk_tile_kernel<Descending, Items>),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
                }
                topk_tile_kernel<Descending, Items><<<blocks, 256, bytes, stream>>>(
                    keys.data_ptr<float>(), indices.defined() ? indices.data_ptr<std::int64_t>() : nullptr,
                    next_keys.data_ptr<float>(), next_indices.data_ptr<std::int64_t>(),
                    rows, columns, static_cast<int>(count));
            };
            if (items == 4) {
                if (largest) launch.template operator()<true, 4>();
                else launch.template operator()<false, 4>();
            } else if (items == 16) {
                if (largest) launch.template operator()<true, 16>();
                else launch.template operator()<false, 16>();
            } else {
                if (largest) launch.template operator()<true, 32>();
                else launch.template operator()<false, 32>();
            }
            MFQ_NATIVE_CUDA_CHECK(hipGetLastError());
            keys = std::move(next_keys);
            indices = std::move(next_indices);
            columns = tiles * count;
        } while (columns > count);
        return {keys.to(input.scalar_type()).reshape(shape), indices.reshape(shape)};
    }
    auto [values, indices] = sort(input, dimension, largest);
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
