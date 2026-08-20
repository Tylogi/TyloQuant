#pragma once

#include "mfq_cuda_tensor_view.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <optional>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace mfq::cuda {

inline constexpr ScalarType kBool = ScalarType::boolean;
inline constexpr ScalarType kUInt8 = ScalarType::uint8;
inline constexpr ScalarType kInt8 = ScalarType::int8;
inline constexpr ScalarType kInt16 = ScalarType::int16;
inline constexpr ScalarType kInt32 = ScalarType::int32;
inline constexpr ScalarType kInt64 = ScalarType::int64;
inline constexpr ScalarType kFloat16 = ScalarType::float16;
inline constexpr ScalarType kHalf = ScalarType::float16;
inline constexpr ScalarType kBFloat16 = ScalarType::bfloat16;
inline constexpr ScalarType kFloat32 = ScalarType::float32;
inline constexpr ScalarType kFloat64 = ScalarType::float64;
inline constexpr ScalarType kFloat8E4M3FN = ScalarType::float8_e4m3fn;
inline constexpr DeviceType kCPU = DeviceType::cpu;
inline constexpr DeviceType kCUDA = DeviceType::cuda;

class TensorOptions {
public:
    TensorOptions& dtype(ScalarType value) & noexcept {
        scalar_type_ = value;
        has_scalar_type_ = true;
        return *this;
    }
    TensorOptions dtype(ScalarType value) const & noexcept {
        auto result = *this;
        result.scalar_type_ = value;
        result.has_scalar_type_ = true;
        return result;
    }
    TensorOptions&& dtype(ScalarType value) && noexcept {
        scalar_type_ = value;
        has_scalar_type_ = true;
        return std::move(*this);
    }

    TensorOptions& device(Device value) & noexcept {
        device_ = value;
        has_device_ = true;
        return *this;
    }
    TensorOptions device(Device value) const & noexcept {
        auto result = *this;
        result.device_ = value;
        result.has_device_ = true;
        return result;
    }
    TensorOptions&& device(Device value) && noexcept {
        device_ = value;
        has_device_ = true;
        return std::move(*this);
    }

    TensorOptions& device(DeviceType value) & noexcept {
        device_ = Device{value, 0};
        has_device_ = true;
        return *this;
    }
    TensorOptions device(DeviceType value) const & noexcept {
        return device(Device{value, 0});
    }
    TensorOptions&& device(DeviceType value) && noexcept {
        device_ = Device{value, 0};
        has_device_ = true;
        return std::move(*this);
    }

    TensorOptions& pinned_memory(bool value) & noexcept {
        pinned_ = value;
        has_pinned_ = true;
        return *this;
    }
    TensorOptions pinned_memory(bool value) const & noexcept {
        auto result = *this;
        result.pinned_ = value;
        result.has_pinned_ = true;
        return result;
    }
    TensorOptions&& pinned_memory(bool value) && noexcept {
        pinned_ = value;
        has_pinned_ = true;
        return std::move(*this);
    }

    ScalarType scalar_type() const noexcept { return scalar_type_; }
    const Device& target_device() const noexcept { return device_; }
    bool pinned() const noexcept { return pinned_; }
    bool has_dtype() const noexcept { return has_scalar_type_; }
    bool has_device() const noexcept { return has_device_; }
    bool has_pinned_memory() const noexcept { return has_pinned_; }

private:
    ScalarType scalar_type_ = ScalarType::float32;
    Device device_{};
    bool pinned_ = false;
    bool has_scalar_type_ = false;
    bool has_device_ = false;
    bool has_pinned_ = false;
};

struct TensorStorage {
    std::shared_ptr<void> owner;
    void* base = nullptr;
    std::size_t bytes = 0;
    Device device{};
    bool pinned = false;
    std::function<void(std::uintptr_t)> record_stream;
};

class IntArrayRef {
public:
    IntArrayRef() = default;
    IntArrayRef(std::span<const std::int64_t> values) : values_(values.begin(), values.end()) {}
    IntArrayRef(std::initializer_list<std::int64_t> values) : values_(values) {}
    explicit IntArrayRef(std::vector<std::int64_t> values) : values_(std::move(values)) {}

    std::size_t size() const noexcept { return values_.size(); }
    bool empty() const noexcept { return values_.empty(); }
    const std::int64_t* data() const noexcept { return values_.data(); }
    const std::int64_t& operator[](std::size_t index) const { return values_.at(index); }
    auto begin() const noexcept { return values_.begin(); }
    auto end() const noexcept { return values_.end(); }
    std::vector<std::int64_t> vec() const { return values_; }
    operator std::span<const std::int64_t>() const noexcept { return values_; }

    IntArrayRef slice(std::size_t start, std::size_t length) const {
        if (start > values_.size() || length > values_.size() - start) {
            throw std::out_of_range("shape slice is out of range");
        }
        return IntArrayRef(std::span<const std::int64_t>(values_).subspan(start, length));
    }

    bool operator==(const IntArrayRef&) const noexcept = default;

private:
    std::vector<std::int64_t> values_;
};

inline std::ostream& operator<<(std::ostream& stream, const Device& device) {
    stream << (device.is_cuda() ? "cuda" : "cpu");
    if (device.is_cuda() || device.index != 0) stream << ':' << device.index;
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const IntArrayRef& values) {
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) stream << ", ";
        stream << values[index];
    }
    return stream << ']';
}

namespace indexing {

struct NoneType final {};
inline constexpr NoneType None{};

class Slice final {
public:
    Slice() = default;
    Slice(std::int64_t start, std::int64_t end, std::int64_t step = 1)
        : start_(start), end_(end), step_(step), has_start_(true), has_end_(true) {}
    Slice(std::int64_t start, NoneType, std::int64_t step = 1)
        : start_(start), step_(step), has_start_(true) {}

    bool has_start() const noexcept { return has_start_; }
    bool has_end() const noexcept { return has_end_; }
    std::int64_t start() const noexcept { return start_; }
    std::int64_t end() const noexcept { return end_; }
    std::int64_t step() const noexcept { return step_; }

private:
    std::int64_t start_ = 0;
    std::int64_t end_ = 0;
    std::int64_t step_ = 1;
    bool has_start_ = false;
    bool has_end_ = false;
};

}  // namespace indexing

class TensorIndex;

class Tensor {
public:
    Tensor() = default;

    Tensor(
        std::shared_ptr<TensorStorage> storage,
        TensorView view,
        std::size_t byte_offset = 0)
        : storage_(std::move(storage)), view_(view), byte_offset_(byte_offset) {
        if (storage_ && storage_->base != nullptr) {
            view_.data = static_cast<std::byte*>(storage_->base) + byte_offset_;
        }
    }

    bool defined() const noexcept { return storage_ != nullptr; }
    std::int64_t dim() const noexcept { return defined() ? view_.rank : 0; }
    std::int64_t numel() const { return defined() ? view_.numel() : 0; }
    std::size_t nbytes() const { return defined() ? view_.nbytes() : 0; }
    std::int64_t size(std::int64_t dimension) const { return view_.size(dimension); }
    std::int64_t stride(std::int64_t dimension) const { return view_.stride(dimension); }
    ScalarType scalar_type() const noexcept { return view_.scalar_type; }
    ScalarType dtype() const noexcept { return scalar_type(); }
    const Device& device() const noexcept { return view_.device; }
    int get_device() const noexcept { return device().index; }
    bool is_cpu() const noexcept { return defined() && device().is_cpu(); }
    bool is_cuda() const noexcept { return defined() && device().is_cuda(); }
    bool is_contiguous() const noexcept { return defined() && view_.is_contiguous(); }
    std::size_t element_size() const noexcept { return scalar_size(scalar_type()); }

    IntArrayRef sizes() const {
        return IntArrayRef(std::span<const std::int64_t>(view_.sizes).first(view_.rank));
    }

    const TensorView& view_descriptor() const noexcept { return view_; }
    std::size_t byte_offset() const noexcept { return byte_offset_; }
    void record_stream(std::uintptr_t stream) const {
        if (storage_ && storage_->record_stream) storage_->record_stream(stream);
    }

    template <typename T>
    T* data_ptr() const noexcept {
        return static_cast<T*>(view_.data);
    }

    void* data_ptr() const noexcept { return view_.data; }

    TensorOptions options() const noexcept {
        return TensorOptions{}.dtype(scalar_type()).device(device()).pinned_memory(
            storage_ && storage_->pinned);
    }

    Tensor reshape(std::span<const std::int64_t> shape) const;
    Tensor reshape(std::initializer_list<std::int64_t> shape) const {
        return reshape(std::span<const std::int64_t>(shape.begin(), shape.size()));
    }
    Tensor view(std::span<const std::int64_t> shape) const { return reshape(shape); }
    Tensor view(std::initializer_list<std::int64_t> shape) const { return reshape(shape); }
    Tensor transpose(std::int64_t first, std::int64_t second) const;
    Tensor permute(std::span<const std::int64_t> order) const;
    Tensor permute(std::initializer_list<std::int64_t> order) const {
        return permute(std::span<const std::int64_t>(order.begin(), order.size()));
    }
    Tensor unsqueeze(std::int64_t dimension) const;
    Tensor squeeze(std::int64_t dimension) const;
    Tensor squeeze() const;
    Tensor flatten(std::int64_t start = 0, std::int64_t end = -1) const;
    Tensor narrow(std::int64_t dimension, std::int64_t start, std::int64_t length) const;
    Tensor select(std::int64_t dimension, std::int64_t index) const;
    Tensor slice(
        std::int64_t dimension,
        std::int64_t start,
        std::int64_t end,
        std::int64_t step = 1) const;
    Tensor index(std::initializer_list<TensorIndex> indices) const;
    Tensor operator[](std::int64_t index) const { return select(0, index); }
    Tensor index_select(std::int64_t dimension, const Tensor& indices) const;
    Tensor gather(std::int64_t dimension, const Tensor& indices) const;
    Tensor expand(std::span<const std::int64_t> shape) const;
    Tensor expand(std::initializer_list<std::int64_t> shape) const {
        return expand(std::span<const std::int64_t>(shape.begin(), shape.size()));
    }
    Tensor repeat(std::span<const std::int64_t> repeats) const;
    Tensor repeat(std::initializer_list<std::int64_t> repeats) const {
        return repeat(std::span<const std::int64_t>(repeats.begin(), repeats.size()));
    }
    Tensor repeat_interleave(std::int64_t repeats, std::int64_t dimension) const;
    Tensor unfold(
        std::int64_t dimension,
        std::int64_t size,
        std::int64_t step) const;
    std::vector<Tensor> split_with_sizes(
        std::span<const std::int64_t> sections,
        std::int64_t dimension = 0) const;
    std::vector<Tensor> split_with_sizes(
        std::initializer_list<std::int64_t> sections,
        std::int64_t dimension = 0) const {
        return split_with_sizes(
            std::span<const std::int64_t>(sections.begin(), sections.size()),
            dimension);
    }
    std::vector<Tensor> chunk(std::int64_t chunks, std::int64_t dimension = 0) const;
    Tensor mean(std::int64_t dimension, bool keep_dimension = false) const;
    Tensor mean() const;
    Tensor sum(std::int64_t dimension, bool keep_dimension = false) const;
    Tensor sum() const;
    Tensor max() const;
    Tensor amax(std::int64_t dimension, bool keep_dimension = false) const;
    Tensor norm() const;
    Tensor abs() const;
    Tensor square() const;
    Tensor exp() const;
    Tensor sqrt() const;
    Tensor clamp(double minimum, double maximum) const;
    Tensor clamp_min(double minimum) const;
    Tensor clamp_max(double maximum) const;
    Tensor remainder(double divisor) const;
    Tensor argmax(std::int64_t dimension, bool keep_dimension = false) const;
    Tensor eq(const Tensor& other) const;
    Tensor eq(double other) const;
    Tensor ne(const Tensor& other) const;
    Tensor ne(double other) const;
    Tensor all() const;
    Tensor all(std::int64_t dimension, bool keep_dimension = false) const;
    Tensor any(std::int64_t dimension, bool keep_dimension = false) const;
    Tensor logical_not() const;
    Tensor detach() const { return *this; }
    bool equal(const Tensor& other) const;
    Tensor masked_select(const Tensor& mask) const;
    Tensor tril(std::int64_t diagonal = 0) const;
    Tensor masked_fill(const Tensor& mask, double value) const;
    Tensor& masked_fill_(const Tensor& mask, double value);
    Tensor reshape_as(const Tensor& other) const { return reshape(other.sizes()); }

    Tensor contiguous() const;
    Tensor clone() const;
    Tensor to(ScalarType type) const;
    Tensor to(Device device) const;
    Tensor to(DeviceType device) const { return to(Device{device, 0}); }
    Tensor to(Device device, bool, bool) const { return to(device); }
    Tensor to(const TensorOptions& options, bool = false, bool = false) const {
        const auto destination = options.has_device() ? options.target_device() : device();
        const auto type = options.has_dtype() ? options.scalar_type() : scalar_type();
        auto moved = destination == device()
            ? *this
            : to(destination);
        return type == moved.scalar_type()
            ? moved
            : moved.to(type);
    }
    Tensor to(DeviceType device, ScalarType type) const {
        return to(Device{device, 0}, type);
    }
    Tensor to(Device device, ScalarType type) const {
        if (device != this->device() && type != scalar_type()) {
            return to(type).to(device);
        }
        auto moved = device == this->device() ? *this : to(device);
        return type == moved.scalar_type() ? moved : moved.to(type);
    }
    Tensor cpu() const { return to(Device{DeviceType::cpu, 0}); }

    Tensor& copy_(const Tensor& source, bool non_blocking = false);
    Tensor& zero_();
    Tensor& fill_(double value);
    Tensor& add_(const Tensor& value);
    Tensor& scatter_add_(std::int64_t dimension, const Tensor& index, const Tensor& source);
    Tensor& scatter_(std::int64_t dimension, const Tensor& index, const Tensor& source);
    Tensor& index_copy_(std::int64_t dimension, const Tensor& index, const Tensor& source);
    Tensor& index_fill_(std::int64_t dimension, const Tensor& index, double value);
    Tensor& index_put_(std::initializer_list<TensorIndex> indices, const Tensor& value);
    Tensor& index_put_(std::initializer_list<TensorIndex> indices, double value);

    template <typename T>
    T item() const {
        if (!defined() || numel() != 1) {
            throw std::invalid_argument("item() requires a defined scalar tensor");
        }
        const Tensor host = is_cpu() && is_contiguous() ? *this : cpu().contiguous();
        return scalar_cast<T>(host.data_ptr(), host.scalar_type());
    }

private:
    template <typename T>
    static T scalar_cast(const void* pointer, ScalarType type) {
        switch (type) {
            case ScalarType::boolean: return static_cast<T>(*static_cast<const bool*>(pointer));
            case ScalarType::uint8: return static_cast<T>(*static_cast<const std::uint8_t*>(pointer));
            case ScalarType::int8: return static_cast<T>(*static_cast<const std::int8_t*>(pointer));
            case ScalarType::int16: return static_cast<T>(*static_cast<const std::int16_t*>(pointer));
            case ScalarType::int32: return static_cast<T>(*static_cast<const std::int32_t*>(pointer));
            case ScalarType::int64: return static_cast<T>(*static_cast<const std::int64_t*>(pointer));
            case ScalarType::float32: return static_cast<T>(*static_cast<const float*>(pointer));
            case ScalarType::float64: return static_cast<T>(*static_cast<const double*>(pointer));
            case ScalarType::float16: {
                const auto bits = *static_cast<const std::uint16_t*>(pointer);
                const auto sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
                auto exponent = static_cast<std::uint32_t>((bits >> 10) & 0x1fu);
                auto mantissa = static_cast<std::uint32_t>(bits & 0x03ffu);
                std::uint32_t value = 0;
                if (exponent == 0) {
                    if (mantissa != 0) {
                        exponent = 127 - 15 + 1;
                        while ((mantissa & 0x0400u) == 0) {
                            mantissa <<= 1;
                            --exponent;
                        }
                        mantissa &= 0x03ffu;
                        value = sign | (exponent << 23) | (mantissa << 13);
                    } else {
                        value = sign;
                    }
                } else if (exponent == 0x1fu) {
                    value = sign | 0x7f800000u | (mantissa << 13);
                } else {
                    value = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
                }
                float decoded = 0.0f;
                std::memcpy(&decoded, &value, sizeof(decoded));
                return static_cast<T>(decoded);
            }
            case ScalarType::bfloat16: {
                const auto value = static_cast<std::uint32_t>(
                    *static_cast<const std::uint16_t*>(pointer)) << 16;
                float decoded = 0.0f;
                std::memcpy(&decoded, &value, sizeof(decoded));
                return static_cast<T>(decoded);
            }
            case ScalarType::float8_e4m3fn: {
                const auto bits = *static_cast<const std::uint8_t*>(pointer);
                const double sign = (bits & 0x80u) != 0 ? -1.0 : 1.0;
                const auto exponent = static_cast<int>((bits >> 3) & 0x0fu);
                const auto mantissa = static_cast<int>(bits & 0x07u);
                if (exponent == 0) {
                    return static_cast<T>(sign * std::ldexp(
                        static_cast<double>(mantissa) / 8.0, -6));
                }
                if (exponent == 0x0f && mantissa == 0x07) {
                    return static_cast<T>(std::numeric_limits<double>::quiet_NaN());
                }
                return static_cast<T>(sign * std::ldexp(
                    1.0 + static_cast<double>(mantissa) / 8.0,
                    exponent - 7));
            }
        }
        throw std::invalid_argument("unknown tensor scalar type");
    }

    std::shared_ptr<TensorStorage> storage_;
    TensorView view_{};
    std::size_t byte_offset_ = 0;
};

class TensorIndex final {
public:
    enum class Kind : std::uint8_t { integer, slice, tensor };

    TensorIndex(std::int64_t value) : kind_(Kind::integer), integer_(value) {}
    TensorIndex(int value) : TensorIndex(static_cast<std::int64_t>(value)) {}
    TensorIndex(const indexing::Slice& value) : kind_(Kind::slice), slice_(value) {}
    TensorIndex(const Tensor& value)
        : kind_(Kind::tensor), tensor_(std::make_shared<Tensor>(value)) {}

    Kind kind() const noexcept { return kind_; }
    std::int64_t integer() const noexcept { return integer_; }
    const indexing::Slice& slice() const noexcept { return slice_; }
    const Tensor& tensor() const { return *tensor_; }

private:
    Kind kind_ = Kind::integer;
    std::int64_t integer_ = 0;
    indexing::Slice slice_{};
    std::shared_ptr<Tensor> tensor_;
};

class NoGradGuard final {
public:
    NoGradGuard() = default;
};

Tensor empty(std::span<const std::int64_t> shape, const TensorOptions& options = {});
Tensor empty(std::initializer_list<std::int64_t> shape, const TensorOptions& options = {});
Tensor empty_like(const Tensor& source, const TensorOptions* options = nullptr);
Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options = {});
Tensor zeros(std::initializer_list<std::int64_t> shape, const TensorOptions& options = {});
Tensor zeros_like(const Tensor& source);
Tensor ones(std::span<const std::int64_t> shape, const TensorOptions& options = {});
Tensor ones(std::initializer_list<std::int64_t> shape, const TensorOptions& options = {});
Tensor full(std::span<const std::int64_t> shape, double value, const TensorOptions& options = {});
Tensor full(std::initializer_list<std::int64_t> shape, double value, const TensorOptions& options = {});
Tensor full_like(const Tensor& source, double value);
Tensor ones_like(const Tensor& source);
Tensor from_blob(
    void* data,
    std::span<const std::int64_t> shape,
    const TensorOptions& options = {});
Tensor from_blob(
    void* data,
    std::initializer_list<std::int64_t> shape,
    const TensorOptions& options = {});

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias = std::nullopt);
Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& mask,
    double dropout,
    bool causal,
    const std::optional<double>& scale,
    bool enable_grouped_query_attention);

Tensor cat(std::span<const Tensor> tensors, std::int64_t dimension = 0);
Tensor cat(std::initializer_list<Tensor> tensors, std::int64_t dimension = 0);
Tensor cat(const std::vector<Tensor>& tensors, std::int64_t dimension = 0);
Tensor stack(std::span<const Tensor> tensors, std::int64_t dimension = 0);
Tensor stack(std::initializer_list<Tensor> tensors, std::int64_t dimension = 0);
Tensor stack(const std::vector<Tensor>& tensors, std::int64_t dimension = 0);
Tensor arange(std::int64_t end, const TensorOptions& options = {});
Tensor arange(std::int64_t start, std::int64_t end, const TensorOptions& options = {});
Tensor arange(
    std::int64_t start,
    std::int64_t end,
    std::int64_t step,
    const TensorOptions& options = {});
Tensor randn(std::span<const std::int64_t> shape, const TensorOptions& options = {});
Tensor randn(std::initializer_list<std::int64_t> shape, const TensorOptions& options = {});
Tensor randint(
    std::int64_t high,
    std::span<const std::int64_t> shape,
    const TensorOptions& options = {});
Tensor randint(
    std::int64_t low,
    std::int64_t high,
    std::span<const std::int64_t> shape,
    const TensorOptions& options = {});
inline Tensor randint(
    std::int64_t low,
    std::int64_t high,
    std::initializer_list<std::int64_t> shape,
    const TensorOptions& options = {}) {
    return randint(
        low, high,
        std::span<const std::int64_t>(shape.begin(), shape.size()),
        options);
}
Tensor randperm(std::int64_t size, const TensorOptions& options = {});
void manual_seed(std::int64_t seed);

Tensor matmul(const Tensor& left, const Tensor& right);
Tensor bmm(const Tensor& left, const Tensor& right);
Tensor baddbmm(const Tensor& input, const Tensor& left, const Tensor& right);
Tensor where(const Tensor& condition, const Tensor& yes, const Tensor& no);
Tensor where(const Tensor& condition, const Tensor& yes, double no);
Tensor where(const Tensor& condition, double yes, const Tensor& no);
Tensor silu(const Tensor& input);
Tensor gelu(const Tensor& input, const std::string& approximation = "none");
Tensor relu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor tanh(const Tensor& input);
Tensor exp(const Tensor& input);
Tensor exp2(const Tensor& input);
Tensor log(const Tensor& input);
Tensor log1p(const Tensor& input);
Tensor log2(const Tensor& input);
Tensor sqrt(const Tensor& input);
Tensor rsqrt(const Tensor& input);
Tensor reciprocal(const Tensor& input);
Tensor sin(const Tensor& input);
Tensor cos(const Tensor& input);
Tensor ceil(const Tensor& input);
Tensor softplus(const Tensor& input);
Tensor pow(const Tensor& input, double exponent);
Tensor pow(const Tensor& input, const Tensor& exponent);
Tensor remainder(const Tensor& input, double divisor);
Tensor clamp(const Tensor& input, double minimum, double maximum);
Tensor clamp_min(const Tensor& input, double minimum);
Tensor clamp_max(const Tensor& input, double maximum);
Tensor softmax(const Tensor& input, std::int64_t dimension);
inline Tensor softmax(
    const Tensor& input,
    std::int64_t dimension,
    ScalarType dtype) {
    return softmax(input, dimension).to(dtype);
}
Tensor log_softmax(const Tensor& input, std::int64_t dimension);
Tensor logsumexp(const Tensor& input, std::int64_t dimension, bool keep_dimension = false);
Tensor sum(const Tensor& input, std::int64_t dimension, bool keep_dimension = false);
Tensor mean(const Tensor& input, std::int64_t dimension, bool keep_dimension = false);
Tensor argmax(const Tensor& input, std::int64_t dimension, bool keep_dimension = false);
std::tuple<Tensor, Tensor> max(
    const Tensor& input,
    std::int64_t dimension,
    bool keep_dimension = false);
Tensor dot(const Tensor& left, const Tensor& right);
Tensor isfinite(const Tensor& input);
Tensor isneginf(const Tensor& input);
bool equal(const Tensor& left, const Tensor& right);
std::tuple<Tensor, Tensor> topk(
    const Tensor& input,
    std::int64_t count,
    std::int64_t dimension = -1,
    bool largest = true,
    bool sorted = true);
std::tuple<Tensor, Tensor> sort(
    const Tensor& input,
    std::int64_t dimension = -1,
    bool descending = false);
Tensor layer_norm(
    const Tensor& input,
    std::span<const std::int64_t> normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double epsilon);
inline Tensor layer_norm(
    const Tensor& input,
    std::initializer_list<std::int64_t> normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double epsilon) {
    return layer_norm(
        input,
        std::span<const std::int64_t>(
            normalized_shape.begin(), normalized_shape.size()),
        weight, bias, epsilon);
}
Tensor constant_pad_nd(
    const Tensor& input,
    std::span<const std::int64_t> padding,
    double value = 0.0);
inline Tensor constant_pad_nd(
    const Tensor& input,
    std::initializer_list<std::int64_t> padding,
    double value = 0.0) {
    return constant_pad_nd(
        input,
        std::span<const std::int64_t>(padding.begin(), padding.size()),
        value);
}
Tensor conv1d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding,
    std::span<const std::int64_t> dilation,
    std::int64_t groups);
Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding,
    std::span<const std::int64_t> dilation,
    std::int64_t groups);
Tensor avg_pool1d(
    const Tensor& input,
    std::span<const std::int64_t> kernel,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding = {});
Tensor cumsum(const Tensor& input, std::int64_t dimension);
Tensor multinomial(
    const Tensor& probabilities,
    std::int64_t samples,
    bool replacement = false);
Tensor einsum(const std::string& equation, std::span<const Tensor> operands);
inline Tensor einsum(
    const std::string& equation,
    std::initializer_list<Tensor> operands) {
    return einsum(
        equation,
        std::span<const Tensor>(operands.begin(), operands.size()));
}

struct NativeIValue final {
    Tensor value;
    bool isTensor() const noexcept { return value.defined(); }
    Tensor toTensor() const { return value; }
};

std::vector<char> pickle_save(const Tensor& value);
NativeIValue pickle_load(const std::vector<char>& bytes);

Tensor operator+(const Tensor& left, const Tensor& right);
Tensor operator+(const Tensor& left, double right);
Tensor operator+(double left, const Tensor& right);
Tensor operator-(const Tensor& left, const Tensor& right);
Tensor operator-(const Tensor& left, double right);
Tensor operator-(double left, const Tensor& right);
Tensor operator-(const Tensor& value);
Tensor operator*(const Tensor& left, const Tensor& right);
Tensor operator*(const Tensor& left, double right);
Tensor operator*(double left, const Tensor& right);
Tensor operator/(const Tensor& left, const Tensor& right);
Tensor operator/(const Tensor& left, double right);
Tensor operator/(double left, const Tensor& right);
Tensor operator<(const Tensor& left, const Tensor& right);
Tensor operator<(const Tensor& left, double right);
Tensor operator<=(const Tensor& left, const Tensor& right);
Tensor operator<=(const Tensor& left, double right);
Tensor operator>(const Tensor& left, const Tensor& right);
Tensor operator>(const Tensor& left, double right);
Tensor operator>=(const Tensor& left, const Tensor& right);
Tensor operator>=(const Tensor& left, double right);
Tensor operator&(const Tensor& left, const Tensor& right);
Tensor operator|(const Tensor& left, const Tensor& right);
Tensor operator~(const Tensor& value);
Tensor operator==(const Tensor& left, const Tensor& right);
Tensor operator==(const Tensor& left, double right);
Tensor operator!=(const Tensor& left, const Tensor& right);
Tensor operator!=(const Tensor& left, double right);

template <typename Value>
ScalarType scalar_type_for();

template <> inline ScalarType scalar_type_for<bool>() { return kBool; }
template <> inline ScalarType scalar_type_for<std::uint8_t>() { return kUInt8; }
template <> inline ScalarType scalar_type_for<std::int8_t>() { return kInt8; }
template <> inline ScalarType scalar_type_for<std::int16_t>() { return kInt16; }
template <> inline ScalarType scalar_type_for<std::int32_t>() { return kInt32; }
template <> inline ScalarType scalar_type_for<std::int64_t>() { return kInt64; }
template <> inline ScalarType scalar_type_for<float>() { return kFloat32; }
template <> inline ScalarType scalar_type_for<double>() { return kFloat64; }

template <typename Value>
Tensor tensor(const std::vector<Value>& values, TensorOptions options = {}) {
    if (options.scalar_type() == kFloat32 && !std::is_same_v<Value, float>) {
        options.dtype(scalar_type_for<Value>());
    }
    auto host_options = options;
    const auto target = options.target_device();
    host_options.device(kCPU).pinned_memory(target.is_cuda());
    auto host = from_blob(
        const_cast<Value*>(values.data()),
        {static_cast<std::int64_t>(values.size())},
        host_options).clone();
    return target.is_cuda() ? host.to(target) : host;
}

template <typename Value>
Tensor tensor(std::initializer_list<Value> values, TensorOptions options = {}) {
    return tensor(std::vector<Value>(values), options);
}

}  // namespace mfq::cuda
