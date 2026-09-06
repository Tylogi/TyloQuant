#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::cuda {

enum class ScalarType : std::uint8_t {
    boolean,
    uint8,
    int8,
    int16,
    int32,
    int64,
    float16,
    bfloat16,
    float32,
    float64,
    float8_e4m3fn,
};

enum class DeviceType : std::uint8_t {
    cpu,
    cuda,
};

inline constexpr std::size_t kMaximumTensorRank = 8;

constexpr std::size_t scalar_size(ScalarType type) {
    switch (type) {
        case ScalarType::boolean:
        case ScalarType::uint8:
        case ScalarType::int8:
        case ScalarType::float8_e4m3fn:
            return 1;
        case ScalarType::int16:
        case ScalarType::float16:
        case ScalarType::bfloat16:
            return 2;
        case ScalarType::int32:
        case ScalarType::float32:
            return 4;
        case ScalarType::int64:
        case ScalarType::float64:
            return 8;
    }
    return 0;
}

struct Device {
    DeviceType type = DeviceType::cpu;
    std::int32_t index = 0;

    constexpr Device() noexcept = default;
    constexpr explicit Device(DeviceType device_type, std::int32_t device_index = 0) noexcept
        : type(device_type), index(device_index) {}

    constexpr bool is_cpu() const noexcept { return type == DeviceType::cpu; }
    constexpr bool is_cuda() const noexcept { return type == DeviceType::cuda; }
    constexpr bool operator==(const Device&) const noexcept = default;
};

struct TensorView {
    void* data = nullptr;
    std::array<std::int64_t, kMaximumTensorRank> sizes{};
    std::array<std::int64_t, kMaximumTensorRank> strides{};
    std::uint8_t rank = 0;
    ScalarType scalar_type = ScalarType::float32;
    Device device{};

    constexpr bool defined() const noexcept { return data != nullptr; }

    std::int64_t size(std::int64_t dimension) const {
        const auto normalized = normalize_dimension(dimension);
        return sizes[normalized];
    }

    std::int64_t stride(std::int64_t dimension) const {
        const auto normalized = normalize_dimension(dimension);
        return strides[normalized];
    }

    std::int64_t numel() const {
        if (rank == 0) {
            return defined() ? 1 : 0;
        }
        std::int64_t result = 1;
        for (std::size_t index = 0; index < rank; ++index) {
            if (sizes[index] < 0 ||
                (sizes[index] != 0 &&
                 result > std::numeric_limits<std::int64_t>::max() / sizes[index])) {
                throw std::overflow_error("tensor element count overflow");
            }
            result *= sizes[index];
        }
        return result;
    }

    std::size_t nbytes() const {
        const auto elements = numel();
        const auto width = scalar_size(scalar_type);
        if (elements < 0 ||
            static_cast<std::uint64_t>(elements) >
                std::numeric_limits<std::size_t>::max() / width) {
            throw std::overflow_error("tensor byte count overflow");
        }
        return static_cast<std::size_t>(elements) * width;
    }

    bool is_contiguous() const noexcept {
        std::int64_t expected = 1;
        for (std::size_t reverse = rank; reverse > 0; --reverse) {
            const auto index = reverse - 1;
            if (sizes[index] > 1 && strides[index] != expected) {
                return false;
            }
            expected *= sizes[index];
        }
        return true;
    }

    template <typename T>
    T* data_as() const noexcept {
        return static_cast<T*>(data);
    }

private:
    std::size_t normalize_dimension(std::int64_t dimension) const {
        const auto normalized = dimension < 0 ? dimension + rank : dimension;
        if (normalized < 0 || normalized >= rank) {
            throw std::out_of_range("tensor dimension is out of range");
        }
        return static_cast<std::size_t>(normalized);
    }
};

inline TensorView make_contiguous_view(
    void* data,
    std::span<const std::int64_t> shape,
    ScalarType scalar_type,
    Device device) {
    if (shape.size() > kMaximumTensorRank) {
        throw std::invalid_argument("tensor rank exceeds the native CUDA ABI limit");
    }
    TensorView result;
    result.data = data;
    result.rank = static_cast<std::uint8_t>(shape.size());
    result.scalar_type = scalar_type;
    result.device = device;
    std::int64_t stride = 1;
    for (std::size_t reverse = shape.size(); reverse > 0; --reverse) {
        const auto index = reverse - 1;
        if (shape[index] < 0) {
            throw std::invalid_argument("tensor shape cannot contain a negative extent");
        }
        result.sizes[index] = shape[index];
        result.strides[index] = stride;
        if (shape[index] != 0 &&
            stride > std::numeric_limits<std::int64_t>::max() / shape[index]) {
            throw std::overflow_error("tensor stride overflow");
        }
        stride *= shape[index];
    }
    return result;
}

inline TensorView reshape_view(
    const TensorView& source,
    std::span<const std::int64_t> requested_shape) {
    if (!source.is_contiguous()) {
        throw std::invalid_argument("reshape_view requires contiguous storage");
    }
    if (requested_shape.size() > kMaximumTensorRank) {
        throw std::invalid_argument("tensor rank exceeds the native CUDA ABI limit");
    }

    std::vector<std::int64_t> resolved(requested_shape.begin(), requested_shape.end());
    std::int64_t inferred_index = -1;
    std::int64_t known_elements = 1;
    for (std::size_t index = 0; index < resolved.size(); ++index) {
        const auto extent = resolved[index];
        if (extent == -1) {
            if (inferred_index >= 0) {
                throw std::invalid_argument("reshape can infer only one dimension");
            }
            inferred_index = static_cast<std::int64_t>(index);
            continue;
        }
        if (extent < 0) {
            throw std::invalid_argument("reshape extent must be non-negative or -1");
        }
        if (extent != 0 &&
            known_elements > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("reshape element count overflow");
        }
        known_elements *= extent;
    }

    const auto source_elements = source.numel();
    if (inferred_index >= 0) {
        if (known_elements == 0 || source_elements % known_elements != 0) {
            throw std::invalid_argument("reshape cannot infer a compatible extent");
        }
        resolved[static_cast<std::size_t>(inferred_index)] = source_elements / known_elements;
    } else if (known_elements != source_elements) {
        throw std::invalid_argument("reshape changes the tensor element count");
    }

    return make_contiguous_view(
        source.data,
        resolved,
        source.scalar_type,
        source.device);
}

inline TensorView transpose_view(
    const TensorView& source,
    std::int64_t first_dimension,
    std::int64_t second_dimension) {
    auto result = source;
    auto normalize = [&](std::int64_t dimension) -> std::size_t {
        const auto normalized = dimension < 0 ? dimension + source.rank : dimension;
        if (normalized < 0 || normalized >= source.rank) {
            throw std::out_of_range("transpose dimension is out of range");
        }
        return static_cast<std::size_t>(normalized);
    };
    const auto first = normalize(first_dimension);
    const auto second = normalize(second_dimension);
    std::swap(result.sizes[first], result.sizes[second]);
    std::swap(result.strides[first], result.strides[second]);
    return result;
}

}  // namespace mfq::cuda
