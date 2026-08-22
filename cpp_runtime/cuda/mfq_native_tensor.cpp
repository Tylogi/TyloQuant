#include "mfq_native_tensor.h"

#include <cstdlib>
#include <cstring>
#include <bit>
#include <cmath>
#include <limits>
#include <new>
#include <string_view>
#include <unordered_set>

namespace mfq::cuda {
namespace {

std::size_t normalize_dimension(std::int64_t dimension, std::size_t rank, bool allow_end = false) {
    const auto upper = static_cast<std::int64_t>(rank) + (allow_end ? 1 : 0);
    auto normalized = dimension;
    if (normalized < 0) {
        normalized += upper;
    }
    if (normalized < 0 || normalized >= upper) {
        throw std::out_of_range("tensor dimension is out of range");
    }
    return static_cast<std::size_t>(normalized);
}

std::shared_ptr<TensorStorage> allocate_cpu_storage(std::size_t bytes, bool pinned) {
    if (pinned) {
        throw std::invalid_argument(
            "pinned host allocation is provided by the native CUDA translation unit");
    }
    const auto allocation_bytes = std::max<std::size_t>(bytes, 1);
    void* pointer = ::operator new(allocation_bytes, std::align_val_t{64});
    auto owner = std::shared_ptr<void>(pointer, [](void* value) {
        ::operator delete(value, std::align_val_t{64});
    });
    auto storage = std::make_shared<TensorStorage>();
    storage->owner = std::move(owner);
    storage->base = pointer;
    storage->bytes = bytes;
    storage->device = Device{DeviceType::cpu, 0};
    storage->pinned = false;
    return storage;
}

Tensor empty_cpu(std::span<const std::int64_t> shape, const TensorOptions& options) {
    auto descriptor = make_contiguous_view(
        reinterpret_cast<void*>(1), shape, options.scalar_type(), options.target_device());
    auto storage = allocate_cpu_storage(descriptor.nbytes(), options.pinned());
    descriptor.data = storage->base;
    return Tensor(std::move(storage), descriptor);
}

float decode_float16(std::uint16_t bits) {
    const auto sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    auto exponent = static_cast<std::uint32_t>((bits >> 10) & 0x1fu);
    auto mantissa = static_cast<std::uint32_t>(bits & 0x03ffu);
    std::uint32_t value = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            value = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            value = sign | (exponent << 23) | ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | (mantissa << 13);
    } else {
        value = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(value);
}

std::uint16_t encode_float16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const auto exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    auto mantissa = bits & 0x007fffffu;
    if (((bits >> 23) & 0xffu) == 0xffu) {
        if (mantissa == 0) return static_cast<std::uint16_t>(sign | 0x7c00u);
        return static_cast<std::uint16_t>(sign | 0x7e00u | (mantissa >> 13));
    }
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        mantissa |= 0x00800000u;
        const auto shift = static_cast<unsigned>(14 - exponent);
        auto rounded = mantissa >> shift;
        const auto remainder = mantissa & ((std::uint32_t{1} << shift) - 1);
        const auto halfway = std::uint32_t{1} << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    auto rounded = mantissa >> 13;
    const auto remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
        if (rounded == 0x0400u) {
            rounded = 0;
            if (exponent + 1 >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
            return static_cast<std::uint16_t>(sign | ((exponent + 1) << 10));
        }
    }
    return static_cast<std::uint16_t>(sign | (exponent << 10) | rounded);
}

float decode_bfloat16(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

std::uint16_t encode_bfloat16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    const auto lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(bits >> 16);
}

double read_cpu_number(const void* pointer, ScalarType type) {
    switch (type) {
        case kBool: return *static_cast<const bool*>(pointer) ? 1.0 : 0.0;
        case kUInt8: return *static_cast<const std::uint8_t*>(pointer);
        case kInt8: return *static_cast<const std::int8_t*>(pointer);
        case kInt16: return *static_cast<const std::int16_t*>(pointer);
        case kInt32: return *static_cast<const std::int32_t*>(pointer);
        case kInt64: return static_cast<double>(*static_cast<const std::int64_t*>(pointer));
        case kFloat16: return decode_float16(*static_cast<const std::uint16_t*>(pointer));
        case kBFloat16: return decode_bfloat16(*static_cast<const std::uint16_t*>(pointer));
        case kFloat32: return *static_cast<const float*>(pointer);
        case kFloat64: return *static_cast<const double*>(pointer);
        case kFloat8E4M3FN:
            throw std::invalid_argument("CPU FP8 conversion is format-specific");
    }
    throw std::invalid_argument("unknown CPU tensor dtype");
}

void write_cpu_number(void* pointer, ScalarType type, double value) {
    switch (type) {
        case kBool: *static_cast<bool*>(pointer) = value != 0.0; return;
        case kUInt8: *static_cast<std::uint8_t*>(pointer) = static_cast<std::uint8_t>(value); return;
        case kInt8: *static_cast<std::int8_t*>(pointer) = static_cast<std::int8_t>(value); return;
        case kInt16: *static_cast<std::int16_t*>(pointer) = static_cast<std::int16_t>(value); return;
        case kInt32: *static_cast<std::int32_t*>(pointer) = static_cast<std::int32_t>(value); return;
        case kInt64: *static_cast<std::int64_t*>(pointer) = static_cast<std::int64_t>(value); return;
        case kFloat16: *static_cast<std::uint16_t*>(pointer) = encode_float16(static_cast<float>(value)); return;
        case kBFloat16: *static_cast<std::uint16_t*>(pointer) = encode_bfloat16(static_cast<float>(value)); return;
        case kFloat32: *static_cast<float*>(pointer) = static_cast<float>(value); return;
        case kFloat64: *static_cast<double*>(pointer) = value; return;
        case kFloat8E4M3FN:
            throw std::invalid_argument("CPU FP8 conversion is format-specific");
    }
    throw std::invalid_argument("unknown CPU tensor dtype");
}

std::int64_t cpu_tensor_offset(const TensorView& view, std::int64_t linear) {
    std::int64_t result = 0;
    for (std::size_t reverse = view.rank; reverse > 0; --reverse) {
        const auto dimension = reverse - 1;
        const auto coordinate = linear % view.sizes[dimension];
        linear /= view.sizes[dimension];
        result += coordinate * view.strides[dimension];
    }
    return result;
}

void copy_contiguous_cpu(Tensor& destination, const Tensor& source) {
    if (!destination.is_cpu() || !source.is_cpu() ||
        destination.numel() != source.numel()) {
        throw std::invalid_argument("CPU copy shape mismatch");
    }
    if (destination.scalar_type() == source.scalar_type() &&
        destination.is_contiguous() && source.is_contiguous()) {
        std::memcpy(destination.data_ptr(), source.data_ptr(), destination.nbytes());
        return;
    }
    const auto source_width = scalar_size(source.scalar_type());
    const auto destination_width = scalar_size(destination.scalar_type());
    for (std::int64_t linear = 0; linear < source.numel(); ++linear) {
        const auto source_offset = cpu_tensor_offset(source.view_descriptor(), linear);
        const auto destination_offset = cpu_tensor_offset(destination.view_descriptor(), linear);
        const auto* input = static_cast<const std::byte*>(source.data_ptr()) +
            source_offset * source_width;
        auto* output = static_cast<std::byte*>(destination.data_ptr()) +
            destination_offset * destination_width;
        if (destination.scalar_type() == source.scalar_type()) {
            std::memcpy(output, input, source_width);
        } else {
            write_cpu_number(output, destination.scalar_type(),
                             read_cpu_number(input, source.scalar_type()));
        }
    }
}

}  // namespace

// Implemented in mfq_native_tensor.cu for CUDA allocations and copies.
Tensor empty_cuda(std::span<const std::int64_t> shape, const TensorOptions& options);
Tensor empty_pinned(std::span<const std::int64_t> shape, const TensorOptions& options);
Tensor copy_or_convert_cuda(const Tensor& source, const TensorOptions& options);
void fill_cuda(Tensor& destination, double value);
void copy_cuda(Tensor& destination, const Tensor& source);
Tensor make_contiguous_cuda(const Tensor& source);
Tensor mean_cuda(const Tensor& source, std::int64_t dimension, bool keep_dimension);
Tensor reduce_cuda(
    const Tensor& source,
    std::int64_t dimension,
    bool keep_dimension,
    int operation);
Tensor unary_cuda(const Tensor& source, int operation, double first = 0.0, double second = 0.0);
Tensor binary_cuda(const Tensor& left, const Tensor& right, int operation);
Tensor scalar_binary_cuda(const Tensor& left, double right, int operation, bool scalar_first = false);
Tensor index_select_cuda(const Tensor& source, std::int64_t dimension, const Tensor& indices);
Tensor gather_cuda(const Tensor& source, std::int64_t dimension, const Tensor& indices);
Tensor repeat_cuda(const Tensor& source, std::span<const std::int64_t> repeats);
Tensor repeat_interleave_cuda(const Tensor& source, std::int64_t repeats, std::int64_t dimension);
Tensor masked_select_cuda(const Tensor& source, const Tensor& mask);
void scatter_cuda(Tensor& destination, std::int64_t dimension, const Tensor& index,
                  const Tensor& source, bool add);
void index_copy_cuda(Tensor& destination, std::int64_t dimension, const Tensor& index,
                     const Tensor& source);
void index_fill_cuda(Tensor& destination, std::int64_t dimension, const Tensor& index,
                     double value);
void masked_fill_cuda(Tensor& destination, const Tensor& mask, double value);

Tensor Tensor::reshape(std::span<const std::int64_t> shape) const {
    if (!defined()) {
        throw std::invalid_argument("cannot reshape an undefined tensor");
    }
    if (!is_contiguous()) return contiguous().reshape(shape);
    return Tensor(storage_, reshape_view(view_, shape), byte_offset_);
}

Tensor Tensor::transpose(std::int64_t first, std::int64_t second) const {
    if (!defined()) {
        throw std::invalid_argument("cannot transpose an undefined tensor");
    }
    return Tensor(storage_, transpose_view(view_, first, second), byte_offset_);
}

Tensor Tensor::permute(std::span<const std::int64_t> order) const {
    if (!defined() || order.size() != view_.rank) {
        throw std::invalid_argument("permute order must contain every tensor dimension");
    }
    TensorView result = view_;
    std::unordered_set<std::size_t> seen;
    for (std::size_t output = 0; output < order.size(); ++output) {
        const auto input = normalize_dimension(order[output], view_.rank);
        if (!seen.insert(input).second) {
            throw std::invalid_argument("permute order contains a duplicate dimension");
        }
        result.sizes[output] = view_.sizes[input];
        result.strides[output] = view_.strides[input];
    }
    return Tensor(storage_, result, byte_offset_);
}

Tensor Tensor::unsqueeze(std::int64_t dimension) const {
    if (!defined() || view_.rank == kMaximumTensorRank) {
        throw std::invalid_argument("cannot unsqueeze this tensor");
    }
    const auto inserted = normalize_dimension(dimension, view_.rank, true);
    TensorView result = view_;
    for (std::size_t index = result.rank; index > inserted; --index) {
        result.sizes[index] = result.sizes[index - 1];
        result.strides[index] = result.strides[index - 1];
    }
    result.sizes[inserted] = 1;
    result.strides[inserted] = inserted < result.rank
        ? result.strides[inserted + 1] * result.sizes[inserted + 1]
        : 1;
    ++result.rank;
    return Tensor(storage_, result, byte_offset_);
}

Tensor Tensor::squeeze(std::int64_t dimension) const {
    if (!defined()) {
        throw std::invalid_argument("cannot squeeze an undefined tensor");
    }
    const auto removed = normalize_dimension(dimension, view_.rank);
    if (view_.sizes[removed] != 1) {
        return *this;
    }
    TensorView result = view_;
    for (std::size_t index = removed; index + 1 < result.rank; ++index) {
        result.sizes[index] = result.sizes[index + 1];
        result.strides[index] = result.strides[index + 1];
    }
    --result.rank;
    return Tensor(storage_, result, byte_offset_);
}

Tensor Tensor::squeeze() const {
    if (!defined()) {
        throw std::invalid_argument("cannot squeeze an undefined tensor");
    }
    auto result = *this;
    for (std::int64_t dimension = result.dim() - 1; dimension >= 0; --dimension) {
        if (result.size(dimension) == 1) {
            result = result.squeeze(dimension);
        }
    }
    return result;
}

Tensor Tensor::flatten(std::int64_t start, std::int64_t end) const {
    if (!defined()) {
        throw std::invalid_argument("flatten requires a defined tensor");
    }
    if (!is_contiguous()) return contiguous().flatten(start, end);
    const auto first = normalize_dimension(start, view_.rank);
    const auto last = normalize_dimension(end, view_.rank);
    if (first > last) {
        throw std::invalid_argument("flatten start must not exceed end");
    }
    std::vector<std::int64_t> shape;
    for (std::size_t index = 0; index < first; ++index) {
        shape.push_back(view_.sizes[index]);
    }
    std::int64_t collapsed = 1;
    for (std::size_t index = first; index <= last; ++index) {
        collapsed *= view_.sizes[index];
    }
    shape.push_back(collapsed);
    for (std::size_t index = last + 1; index < view_.rank; ++index) {
        shape.push_back(view_.sizes[index]);
    }
    return reshape(shape);
}

Tensor Tensor::narrow(
    std::int64_t dimension,
    std::int64_t start,
    std::int64_t length) const {
    if (!defined()) {
        throw std::invalid_argument("cannot narrow an undefined tensor");
    }
    const auto selected = normalize_dimension(dimension, view_.rank);
    const auto extent = view_.sizes[selected];
    auto normalized_start = start < 0 ? start + extent : start;
    if (normalized_start < 0 || length < 0 || normalized_start + length > extent) {
        throw std::out_of_range("narrow range is out of bounds");
    }
    auto result = view_;
    result.sizes[selected] = length;
    const auto offset_elements = normalized_start * view_.strides[selected];
    const auto offset_bytes = static_cast<std::size_t>(offset_elements) * element_size();
    result.data = static_cast<std::byte*>(view_.data) + offset_bytes;
    return Tensor(storage_, result, byte_offset_ + offset_bytes);
}

Tensor Tensor::select(std::int64_t dimension, std::int64_t index) const {
    const auto selected = normalize_dimension(dimension, view_.rank);
    const auto narrowed = narrow(dimension, index, 1);
    TensorView result = narrowed.view_;
    for (std::size_t position = selected; position + 1 < result.rank; ++position) {
        result.sizes[position] = result.sizes[position + 1];
        result.strides[position] = result.strides[position + 1];
    }
    --result.rank;
    return Tensor(narrowed.storage_, result, narrowed.byte_offset_);
}

Tensor Tensor::slice(
    std::int64_t dimension,
    std::int64_t start,
    std::int64_t end,
    std::int64_t step) const {
    if (step <= 0) {
        throw std::invalid_argument("native slice requires a positive step");
    }
    const auto selected = normalize_dimension(dimension, view_.rank);
    const auto extent = view_.sizes[selected];
    auto first = start < 0 ? start + extent : start;
    auto last = end < 0 ? end + extent : end;
    first = std::clamp<std::int64_t>(first, 0, extent);
    last = std::clamp<std::int64_t>(last, 0, extent);
    const auto length = last <= first ? 0 : (last - first + step - 1) / step;
    auto result = narrow(dimension, first, std::max<std::int64_t>(0, last - first));
    result.view_.sizes[selected] = length;
    result.view_.strides[selected] *= step;
    return result;
}

Tensor Tensor::index(std::initializer_list<TensorIndex> indices) const {
    Tensor result = *this;
    std::int64_t dimension = 0;
    for (const auto& index : indices) {
        if (index.kind() == TensorIndex::Kind::integer) {
            result = result.select(dimension, index.integer());
            continue;
        }
        if (index.kind() == TensorIndex::Kind::tensor) {
            result = result.index_select(dimension, index.tensor());
            ++dimension;
            continue;
        }
        const auto& selected = index.slice();
        const auto extent = result.size(dimension);
        const auto start = selected.has_start() ? selected.start() : 0;
        const auto end = selected.has_end() ? selected.end() : extent;
        result = result.slice(dimension, start, end, selected.step());
        ++dimension;
    }
    return result;
}

Tensor Tensor::index_select(std::int64_t dimension, const Tensor& indices) const {
    return index_select_cuda(*this, dimension, indices);
}

Tensor Tensor::gather(std::int64_t dimension, const Tensor& indices) const {
    return gather_cuda(*this, dimension, indices);
}

Tensor Tensor::expand(std::span<const std::int64_t> shape) const {
    if (shape.size() < view_.rank || shape.size() > kMaximumTensorRank) {
        throw std::invalid_argument("expand rank is incompatible with tensor rank");
    }
    TensorView result = view_;
    std::array<std::int64_t, kMaximumTensorRank> sizes{};
    std::array<std::int64_t, kMaximumTensorRank> strides{};
    const auto offset = shape.size() - view_.rank;
    for (std::size_t output = 0; output < shape.size(); ++output) {
        const auto requested = shape[output];
        if (output < offset) {
            if (requested < 0) throw std::invalid_argument("new expand dimensions must be explicit");
            sizes[output] = requested;
            strides[output] = 0;
            continue;
        }
        const auto input = output - offset;
        const auto extent = view_.sizes[input];
        const auto resolved = requested == -1 ? extent : requested;
        if (resolved != extent && extent != 1) {
            throw std::invalid_argument("cannot expand a non-singleton dimension");
        }
        sizes[output] = resolved;
        strides[output] = resolved == extent ? view_.strides[input] : 0;
    }
    result.rank = static_cast<std::uint8_t>(shape.size());
    result.sizes = sizes;
    result.strides = strides;
    return Tensor(storage_, result, byte_offset_);
}

Tensor Tensor::repeat(std::span<const std::int64_t> repeats) const {
    return repeat_cuda(*this, repeats);
}

Tensor Tensor::repeat_interleave(std::int64_t repeats, std::int64_t dimension) const {
    return repeat_interleave_cuda(*this, repeats, dimension);
}

Tensor Tensor::unfold(
    std::int64_t dimension,
    std::int64_t window,
    std::int64_t step) const {
    if (window <= 0 || step <= 0 || view_.rank == kMaximumTensorRank) {
        throw std::invalid_argument("invalid unfold geometry");
    }
    const auto selected = normalize_dimension(dimension, view_.rank);
    const auto extent = view_.sizes[selected];
    if (window > extent) throw std::invalid_argument("unfold window exceeds dimension");
    TensorView result = view_;
    result.sizes[selected] = (extent - window) / step + 1;
    result.strides[selected] *= step;
    result.sizes[result.rank] = window;
    result.strides[result.rank] = view_.strides[selected];
    ++result.rank;
    return Tensor(storage_, result, byte_offset_);
}

std::vector<Tensor> Tensor::split_with_sizes(
    std::span<const std::int64_t> sections,
    std::int64_t dimension) const {
    std::vector<Tensor> result;
    result.reserve(sections.size());
    std::int64_t offset = 0;
    for (const auto length : sections) {
        result.push_back(narrow(dimension, offset, length));
        offset += length;
    }
    if (offset != size(dimension)) {
        throw std::invalid_argument("split sizes do not cover the selected dimension");
    }
    return result;
}

std::vector<Tensor> Tensor::chunk(std::int64_t chunks, std::int64_t dimension) const {
    if (chunks <= 0) throw std::invalid_argument("chunk count must be positive");
    const auto extent = size(dimension);
    const auto width = (extent + chunks - 1) / chunks;
    std::vector<Tensor> result;
    for (std::int64_t start = 0; start < extent; start += width) {
        result.push_back(narrow(dimension, start, std::min(width, extent - start)));
    }
    return result;
}

Tensor Tensor::mean(std::int64_t dimension, bool keep_dimension) const {
    if (!defined()) {
        throw std::invalid_argument("mean requires a defined tensor");
    }
    if (!is_cuda()) {
        throw std::invalid_argument("native CPU mean is not implemented");
    }
    return reduce_cuda(*this, dimension, keep_dimension, 1);
}

Tensor Tensor::mean() const { return reshape({-1}).mean(0, false); }
Tensor Tensor::sum(std::int64_t dimension, bool keep_dimension) const {
    return reduce_cuda(*this, dimension, keep_dimension, 0);
}
Tensor Tensor::sum() const { return reshape({-1}).sum(0, false); }
Tensor Tensor::max() const { return reshape({-1}).amax(0, false); }
Tensor Tensor::amax(std::int64_t dimension, bool keep_dimension) const {
    return reduce_cuda(*this, dimension, keep_dimension, 2);
}
Tensor Tensor::norm() const { return square().sum().sqrt(); }
Tensor Tensor::abs() const { return unary_cuda(*this, 0); }
Tensor Tensor::square() const { return unary_cuda(*this, 1); }
Tensor Tensor::exp() const { return unary_cuda(*this, 2); }
Tensor Tensor::sqrt() const { return unary_cuda(*this, 3); }
Tensor Tensor::clamp(double minimum, double maximum) const {
    return unary_cuda(*this, 4, minimum, maximum);
}
Tensor Tensor::clamp_min(double minimum) const {
    return unary_cuda(*this, 5, minimum);
}
Tensor Tensor::clamp_max(double maximum) const {
    return unary_cuda(*this, 6, maximum);
}
Tensor Tensor::remainder(double divisor) const {
    return unary_cuda(*this, 7, divisor);
}
Tensor Tensor::argmax(std::int64_t dimension, bool keep_dimension) const {
    return reduce_cuda(*this, dimension, keep_dimension, 3);
}
Tensor Tensor::eq(const Tensor& other) const { return binary_cuda(*this, other, 10); }
Tensor Tensor::eq(double other) const { return scalar_binary_cuda(*this, other, 10); }
Tensor Tensor::ne(const Tensor& other) const { return binary_cuda(*this, other, 11); }
Tensor Tensor::ne(double other) const { return scalar_binary_cuda(*this, other, 11); }
Tensor Tensor::all() const { return reshape({-1}).all(0, false); }
Tensor Tensor::all(std::int64_t dimension, bool keep_dimension) const {
    return reduce_cuda(*this, dimension, keep_dimension, 4);
}
Tensor Tensor::any(std::int64_t dimension, bool keep_dimension) const {
    return reduce_cuda(*this, dimension, keep_dimension, 5);
}
Tensor Tensor::logical_not() const { return unary_cuda(*this, 8); }
bool Tensor::equal(const Tensor& other) const { return mfq::cuda::equal(*this, other); }
Tensor Tensor::masked_select(const Tensor& mask) const { return masked_select_cuda(*this, mask); }
Tensor Tensor::tril(std::int64_t diagonal) const {
    if (dim() < 2) throw std::invalid_argument("tril requires at least two dimensions");
    auto rows = arange(size(-2), TensorOptions{}.dtype(kInt64).device(device())).unsqueeze(-1);
    auto columns = arange(size(-1), TensorOptions{}.dtype(kInt64).device(device())).unsqueeze(0);
    return where(columns <= rows + diagonal, *this, 0.0);
}
Tensor Tensor::masked_fill(const Tensor& mask, double value) const {
    auto result = clone();
    result.masked_fill_(mask, value);
    return result;
}
Tensor& Tensor::masked_fill_(const Tensor& mask, double value) {
    masked_fill_cuda(*this, mask, value);
    return *this;
}

Tensor Tensor::contiguous() const {
    if (!defined()) {
        return {};
    }
    if (is_contiguous()) {
        return *this;
    }
    if (is_cuda()) {
        return make_contiguous_cuda(*this);
    }
    auto result = empty(sizes(), options());
    copy_contiguous_cpu(result, *this);
    return result;
}

Tensor Tensor::clone() const {
    if (!defined()) {
        return {};
    }
    auto result = empty(sizes(), options());
    result.copy_(*this);
    return result;
}

Tensor Tensor::to(ScalarType type) const {
    auto target = options().dtype(type);
    if (type == scalar_type()) {
        return *this;
    }
    if (is_cpu()) {
        auto result = empty(sizes(), target);
        copy_contiguous_cpu(result, *this);
        return result;
    }
    return copy_or_convert_cuda(*this, target);
}

Tensor Tensor::to(Device target_device) const {
    auto target = options().device(target_device);
    if (target_device == device()) {
        return *this;
    }
    return copy_or_convert_cuda(*this, target);
}

Tensor& Tensor::copy_(const Tensor& source, bool) {
    if (!defined() || !source.defined()) {
        throw std::invalid_argument("copy_ requires defined tensors");
    }
    if (is_cpu() && source.is_cpu()) {
        copy_contiguous_cpu(*this, source);
    } else {
        copy_cuda(*this, source);
    }
    return *this;
}

Tensor& Tensor::zero_() {
    return fill_(0.0);
}

Tensor& Tensor::fill_(double value) {
    if (!defined()) {
        throw std::invalid_argument("fill_ requires a defined tensor");
    }
    if (is_cuda()) {
        fill_cuda(*this, value);
        return *this;
    }
    if (!is_contiguous()) {
        throw std::invalid_argument("CPU fill requires a contiguous tensor");
    }
    if (value == 0.0) {
        std::memset(data_ptr(), 0, nbytes());
        return *this;
    }
    switch (scalar_type()) {
        case ScalarType::boolean:
            std::fill_n(data_ptr<bool>(), numel(), value != 0.0);
            break;
        case ScalarType::uint8:
            std::fill_n(data_ptr<std::uint8_t>(), numel(), static_cast<std::uint8_t>(value));
            break;
        case ScalarType::int8:
            std::fill_n(data_ptr<std::int8_t>(), numel(), static_cast<std::int8_t>(value));
            break;
        case ScalarType::int16:
            std::fill_n(data_ptr<std::int16_t>(), numel(), static_cast<std::int16_t>(value));
            break;
        case ScalarType::int32:
            std::fill_n(data_ptr<std::int32_t>(), numel(), static_cast<std::int32_t>(value));
            break;
        case ScalarType::int64:
            std::fill_n(data_ptr<std::int64_t>(), numel(), static_cast<std::int64_t>(value));
            break;
        case ScalarType::float16:
            std::fill_n(data_ptr<std::uint16_t>(), numel(), encode_float16(static_cast<float>(value)));
            break;
        case ScalarType::bfloat16:
            std::fill_n(data_ptr<std::uint16_t>(), numel(), encode_bfloat16(static_cast<float>(value)));
            break;
        case ScalarType::float32:
            std::fill_n(data_ptr<float>(), numel(), static_cast<float>(value));
            break;
        case ScalarType::float64:
            std::fill_n(data_ptr<double>(), numel(), value);
            break;
        default:
            throw std::invalid_argument("CPU fill for this scalar type is not implemented yet");
    }
    return *this;
}

Tensor& Tensor::add_(const Tensor& value) {
    auto result = *this + value;
    return copy_(result);
}

Tensor& Tensor::scatter_add_(
    std::int64_t dimension, const Tensor& index, const Tensor& source) {
    scatter_cuda(*this, dimension, index, source, true);
    return *this;
}

Tensor& Tensor::scatter_(
    std::int64_t dimension, const Tensor& index, const Tensor& source) {
    scatter_cuda(*this, dimension, index, source, false);
    return *this;
}

Tensor& Tensor::index_copy_(
    std::int64_t dimension, const Tensor& index, const Tensor& source) {
    index_copy_cuda(*this, dimension, index, source);
    return *this;
}

Tensor& Tensor::index_fill_(
    std::int64_t dimension, const Tensor& index, double value) {
    index_fill_cuda(*this, dimension, index, value);
    return *this;
}

Tensor& Tensor::index_put_(
    std::initializer_list<TensorIndex> indices, const Tensor& value) {
    index(indices).copy_(value);
    return *this;
}

Tensor& Tensor::index_put_(
    std::initializer_list<TensorIndex> indices, double value) {
    index(indices).fill_(value);
    return *this;
}

Tensor empty(std::span<const std::int64_t> shape, const TensorOptions& options) {
    if (options.target_device().is_cuda()) {
        return empty_cuda(shape, options);
    }
    return options.pinned() ? empty_pinned(shape, options) : empty_cpu(shape, options);
}

Tensor empty(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return empty(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}

Tensor empty_like(const Tensor& source, const TensorOptions* options) {
    if (options == nullptr) return empty(source.sizes(), source.options());
    auto merged = source.options();
    if (options->has_dtype()) merged.dtype(options->scalar_type());
    if (options->has_device()) merged.device(options->target_device());
    if (options->has_pinned_memory()) merged.pinned_memory(options->pinned());
    return empty(source.sizes(), merged);
}

Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    auto result = empty(shape, options);
    result.zero_();
    return result;
}

Tensor zeros(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return zeros(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}

Tensor zeros_like(const Tensor& source) {
    return zeros(source.sizes(), source.options());
}

Tensor ones(std::span<const std::int64_t> shape, const TensorOptions& options) {
    return full(shape, 1.0, options);
}

Tensor ones(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return ones(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}

Tensor full(std::span<const std::int64_t> shape, double value, const TensorOptions& options) {
    auto result = empty(shape, options);
    result.fill_(value);
    return result;
}

Tensor full(
    std::initializer_list<std::int64_t> shape,
    double value,
    const TensorOptions& options) {
    return full(std::span<const std::int64_t>(shape.begin(), shape.size()), value, options);
}

Tensor full_like(const Tensor& source, double value) {
    return full(source.sizes(), value, source.options());
}

Tensor ones_like(const Tensor& source) {
    return ones(source.sizes(), source.options());
}

Tensor from_blob(
    void* data,
    std::span<const std::int64_t> shape,
    const TensorOptions& options) {
    if (data == nullptr) {
        throw std::invalid_argument("from_blob requires non-null storage");
    }
    auto descriptor = make_contiguous_view(
        data, shape, options.scalar_type(), options.target_device());
    auto storage = std::make_shared<TensorStorage>();
    storage->base = data;
    storage->bytes = descriptor.nbytes();
    storage->device = options.target_device();
    storage->pinned = options.pinned();
    return Tensor(std::move(storage), descriptor);
}

Tensor from_blob(
    void* data,
    std::initializer_list<std::int64_t> shape,
    const TensorOptions& options) {
    return from_blob(
        data,
        std::span<const std::int64_t>(shape.begin(), shape.size()),
        options);
}

std::vector<char> pickle_save(const Tensor& value) {
    if (!value.defined()) throw std::invalid_argument("cannot serialize an undefined tensor");
    const auto host = value.cpu().contiguous();
    constexpr std::string_view magic{"MFQTNSR1", 8};
    const auto header_bytes = std::size_t{16} +
        static_cast<std::size_t>(host.dim()) * sizeof(std::int64_t);
    std::vector<char> result(header_bytes + host.nbytes());
    std::memcpy(result.data(), magic.data(), magic.size());
    result[8] = static_cast<char>(host.scalar_type());
    result[9] = static_cast<char>(host.dim());
    std::fill(result.begin() + 10, result.begin() + 16, 0);
    for (std::int64_t axis = 0; axis < host.dim(); ++axis) {
        const auto extent = host.size(axis);
        std::memcpy(
            result.data() + 16 + static_cast<std::size_t>(axis) * sizeof(extent),
            &extent,
            sizeof(extent));
    }
    std::memcpy(result.data() + header_bytes, host.data_ptr(), host.nbytes());
    return result;
}

NativeIValue pickle_load(const std::vector<char>& bytes) {
    constexpr std::string_view magic{"MFQTNSR1", 8};
    if (bytes.size() < 16 ||
        std::string_view(bytes.data(), magic.size()) != magic) {
        throw std::invalid_argument(
            "native tensor file is not MFQTNSR1; legacy Torch pickle files require "
            "the optional mfq-decode-torch compatibility runtime");
    }
    const auto raw_type = static_cast<std::uint8_t>(bytes[8]);
    const auto rank = static_cast<std::uint8_t>(bytes[9]);
    if (raw_type > static_cast<std::uint8_t>(ScalarType::float8_e4m3fn) ||
        rank > kMaximumTensorRank) {
        throw std::invalid_argument("native tensor header is invalid");
    }
    const auto header_bytes = std::size_t{16} +
        static_cast<std::size_t>(rank) * sizeof(std::int64_t);
    if (bytes.size() < header_bytes) {
        throw std::invalid_argument("native tensor header is truncated");
    }
    std::vector<std::int64_t> shape(rank);
    for (std::size_t axis = 0; axis < rank; ++axis) {
        std::memcpy(
            &shape[axis],
            bytes.data() + 16 + axis * sizeof(std::int64_t),
            sizeof(std::int64_t));
        if (shape[axis] < 0) throw std::invalid_argument("native tensor shape is invalid");
    }
    auto output = empty(
        shape,
        TensorOptions{}.dtype(static_cast<ScalarType>(raw_type)).device(kCPU));
    if (bytes.size() != header_bytes + output.nbytes()) {
        throw std::invalid_argument("native tensor payload size does not match its shape");
    }
    std::memcpy(output.data_ptr(), bytes.data() + header_bytes, output.nbytes());
    return NativeIValue{std::move(output)};
}

}  // namespace mfq::cuda
