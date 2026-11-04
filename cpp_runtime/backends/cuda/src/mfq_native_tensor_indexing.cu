#include "mfq_native_tensor_ops_common.cuh"
#include "mfq_cuda_input_error.cuh"

#include <cub/cub.cuh>

#include <stdexcept>
#include <vector>

namespace mfq::cuda {
using namespace native_ops_detail;
namespace {
template <typename Index>
__device__ std::int64_t load_index(const Index* indices, std::int64_t offset) {
    return static_cast<std::int64_t>(indices[offset]);
}

template <typename Index>
__global__ void index_select_dim0_contiguous_vec16_kernel(
    uint4* __restrict__ output,
    const uint4* __restrict__ source,
    const Index* __restrict__ indices,
    std::int64_t packs_per_row,
    std::int64_t total_packs, std::int64_t extent, DeviceInputError* error) {
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total_packs;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const auto output_row = linear / packs_per_row;
        const auto pack = linear - output_row * packs_per_row;
        const auto source_row = load_index(indices, output_row);
        output[linear] = valid_device_index(source_row, extent, error, 1)
            ? source[source_row * packs_per_row + pack] : uint4{};
    }
}

template <typename Value, typename Index>
__global__ void index_select_kernel(
    TensorView output,
    TensorView source,
    TensorView indices,
    std::int64_t elements,
    int dimension, DeviceInputError* error) {
    auto* destination = static_cast<Value*>(output.data);
    const auto* input = static_cast<const Value*>(source.data);
    const auto* selected = static_cast<const Index*>(indices.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        bool valid = true;
        std::int64_t source_offset = 0;
        for (std::size_t reverse = output.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            auto coordinate = residual % output.sizes[axis];
            residual /= output.sizes[axis];
            if (static_cast<int>(axis) == dimension) {
                coordinate = load_index(selected, coordinate);
                if (!valid_device_index(coordinate, source.sizes[axis], error, 1)) {
                    valid = false;
                    coordinate = 0;
                }
            }
            source_offset += coordinate * source.strides[axis];
        }
        if (!valid) { destination[linear] = Value{}; continue; }
        destination[linear] = input[source_offset];
    }
}

template <typename Value, typename Index>
__global__ void gather_kernel(
    TensorView output,
    TensorView source,
    TensorView indices,
    std::int64_t elements,
    int dimension, DeviceInputError* error) {
    auto* destination = static_cast<Value*>(output.data);
    const auto* input = static_cast<const Value*>(source.data);
    const auto* selected = static_cast<const Index*>(indices.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        bool valid = true;
        std::int64_t source_offset = 0;
        for (std::size_t reverse = output.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            auto coordinate = residual % output.sizes[axis];
            residual /= output.sizes[axis];
            if (static_cast<int>(axis) == dimension) {
                coordinate = load_index(selected, tensor_offset(indices, linear));
                if (!valid_device_index(coordinate, source.sizes[axis], error, 2)) {
                    valid = false;
                    coordinate = 0;
                }
            }
            source_offset += coordinate * source.strides[axis];
        }
        if (!valid) { destination[linear] = Value{}; continue; }
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
    bool add, DeviceInputError* error) {
    auto* output = static_cast<Value*>(destination.data);
    const auto* input = static_cast<const Value*>(source.data);
    const auto* selected = static_cast<const Index*>(indices.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        bool valid = true;
        std::int64_t destination_offset = 0;
        for (std::size_t reverse = source.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            auto coordinate = residual % source.sizes[axis];
            residual /= source.sizes[axis];
            if (static_cast<int>(axis) == dimension) {
                coordinate = load_index(selected, tensor_offset(indices, linear));
                if (!valid_device_index(coordinate, destination.sizes[axis], error, 3)) {
                    valid = false;
                    coordinate = 0;
                }
            }
            destination_offset += coordinate * destination.strides[axis];
        }
        if (!valid) {  continue; }
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
    int dimension, DeviceInputError* error) {
    auto* output = static_cast<Value*>(destination.data);
    const auto* input = static_cast<const Value*>(source.data);
    const auto* selected = static_cast<const Index*>(indices.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        bool valid = true;
        std::int64_t destination_offset = 0;
        for (std::size_t reverse = source.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            auto coordinate = residual % source.sizes[axis];
            residual /= source.sizes[axis];
            if (static_cast<int>(axis) == dimension) {
                coordinate = load_index(selected, coordinate);
                if (!valid_device_index(coordinate, destination.sizes[axis], error, 4)) {
                    valid = false;
                    coordinate = 0;
                }
            }
            destination_offset += coordinate * destination.strides[axis];
        }
        if (!valid) {  continue; }
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
    double fill, DeviceInputError* error) {
    auto* output = static_cast<Value*>(destination.data);
    const auto* selected = static_cast<const Index*>(indices.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        bool valid = true;
        std::int64_t destination_offset = 0;
        for (std::size_t reverse = iteration.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            auto coordinate = residual % iteration.sizes[axis];
            residual /= iteration.sizes[axis];
            if (static_cast<int>(axis) == dimension) {
                coordinate = load_index(selected, coordinate);
                if (!valid_device_index(coordinate, destination.sizes[axis], error, 5)) {
                    valid = false;
                    coordinate = 0;
                }
            }
            destination_offset += coordinate * destination.strides[axis];
        }
        if (!valid) {  continue; }
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
    const auto stream = current_stream(source.get_device()).stream();
    if (selected == 0 && source.is_contiguous() && source.size(0) > 0) {
        const auto row_bytes = static_cast<std::size_t>(
            source.numel() / source.size(0)) * source.element_size();
        const auto addresses = reinterpret_cast<std::uintptr_t>(source.data_ptr()) |
            reinterpret_cast<std::uintptr_t>(output.data_ptr());
        if (row_bytes >= sizeof(uint4) &&
            row_bytes % sizeof(uint4) == 0 &&
            (addresses & (alignof(uint4) - 1)) == 0) {
            const auto packs_per_row = static_cast<std::int64_t>(
                row_bytes / sizeof(uint4));
            const auto total_packs = indices.numel() * packs_per_row;
            const auto [blocks, threads] = launch_geometry(total_packs);
            auto launch_index = [&]<typename Index>() {
                index_select_dim0_contiguous_vec16_kernel<Index>
                    <<<blocks, threads, 0, stream>>>(
                        static_cast<uint4*>(output.data_ptr()),
                        static_cast<const uint4*>(source.data_ptr()),
                        indices.data_ptr<Index>(), packs_per_row,
                        total_packs, source.size(0), default_context(source.get_device())->input_error(stream));
            };
            dispatch_index_type(indices.scalar_type(), launch_index);
            MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
            return output;
        }
    }
    const auto [blocks, threads] = launch_geometry(output.numel());
    auto launch_value = [&]<typename Value>() {
        auto launch_index = [&]<typename Index>() {
            index_select_kernel<Value, Index><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), source.view_descriptor(),
                indices.view_descriptor(), output.numel(), static_cast<int>(selected),
                default_context(source.get_device())->input_error(stream));
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
    for (int64_t axis = 0; axis < source.dim(); ++axis)
        if (axis != int64_t(selected) && indices.size(axis) > source.size(axis))
            throw std::invalid_argument("gather index extent exceeds source");
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream = current_stream(source.get_device()).stream();
    auto launch_value = [&]<typename Value>() {
        auto launch_index = [&]<typename Index>() {
            gather_kernel<Value, Index><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), source.view_descriptor(),
                indices.view_descriptor(), output.numel(), static_cast<int>(selected),
                default_context(source.get_device())->input_error(stream));
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
    if (source.dim() != destination.dim()) throw std::invalid_argument("scatter rank mismatch");
    for (int64_t axis = 0; axis < destination.dim(); ++axis)
        if (axis != int64_t(selected) && source.size(axis) > destination.size(axis))
            throw std::invalid_argument("scatter source extent exceeds destination");
    const auto [blocks, threads] = launch_geometry(source.numel());
    const auto stream = current_stream(destination.get_device()).stream();
    auto launch_value = [&]<typename Value>() {
        auto launch_index = [&]<typename Index>() {
            scatter_kernel<Value, Index><<<blocks, threads, 0, stream>>>(
                destination.view_descriptor(), index.view_descriptor(),
                source.view_descriptor(), source.numel(), static_cast<int>(selected), add,
                default_context(destination.get_device())->input_error(stream));
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
    if (index.dim() != 1 || source.dim() != destination.dim() ||
        source.size(static_cast<std::int64_t>(selected)) != index.numel()) {
        throw std::invalid_argument("index_copy source extent does not match indices");
    }
    for (int64_t axis = 0; axis < destination.dim(); ++axis)
        if (axis != int64_t(selected) && source.size(axis) != destination.size(axis))
            throw std::invalid_argument("index_copy source extent does not match destination");
    const auto [blocks, threads] = launch_geometry(source.numel());
    const auto stream = current_stream(destination.get_device()).stream();
    auto launch_value = [&]<typename Value>() {
        auto launch_index = [&]<typename Index>() {
            index_copy_kernel<Value, Index><<<blocks, threads, 0, stream>>>(
                destination.view_descriptor(), index.view_descriptor(),
                source.view_descriptor(), source.numel(), static_cast<int>(selected),
                default_context(destination.get_device())->input_error(stream));
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
    if (index_source.dim() != 1) throw std::invalid_argument("index_fill requires one-dimensional indices");
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
                static_cast<int>(selected), value,
                default_context(destination.get_device())->input_error(stream));
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

}  // namespace mfq::cuda
