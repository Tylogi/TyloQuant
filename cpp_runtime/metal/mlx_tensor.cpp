#include "mlx_tensor.h"

#include <mlx/allocator.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::array;

class DenseCursor {
public:
    explicit DenseCursor(std::span<const std::uint8_t> blob)
        : blob_(blob) {}

    template <typename T>
    T scalar(const char* name) {
        if (sizeof(T) > blob_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated dense tensor ") + name);
        }
        T value{};
        std::memcpy(&value, blob_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    const std::uint8_t* data() const noexcept {
        return blob_.data() + offset_;
    }
    std::size_t remaining() const noexcept {
        return blob_.size() - offset_;
    }

private:
    std::span<const std::uint8_t> blob_;
    std::size_t offset_ = 0;
};

std::pair<Dtype, std::size_t> dense_dtype(const std::string& name) {
    if (name == "BF16") {
        return {mlx::core::bfloat16, 2};
    }
    if (name == "F16") {
        return {mlx::core::float16, 2};
    }
    if (name == "F32") {
        return {mlx::core::float32, 4};
    }
    if (name == "I32") {
        return {mlx::core::int32, 4};
    }
    if (name == "I64") {
        return {mlx::core::int64, 8};
    }
    throw std::runtime_error("unsupported dense MFQ dtype: " + name);
}

array load_weight(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" &&
        record.dtype != "F16" &&
        record.dtype != "F32") {
        throw std::runtime_error(
            "dense linear/embedding requires BF16, F16, or F32 tensor: " + name);
    }
    const auto mapped = model.map_record(name);
    auto result = load_dense_array(record.dtype, mapped.view());
    if (result.ndim() != 2) {
        throw std::runtime_error(
            "linear/embedding weight must have rank two: " + name);
    }
    return result;
}

} // namespace

array load_dense_array(
    const std::string& dtype_name,
    std::span<const std::uint8_t> blob) {
    DenseCursor cursor(blob);
    const auto dimensions =
        cursor.scalar<std::uint32_t>("dimension count");
    if (dimensions == 0 || dimensions > 8) {
        throw std::runtime_error("invalid dense MFQ dimension count");
    }
    Shape shape;
    shape.reserve(dimensions);
    std::size_t elements = 1;
    for (std::uint32_t index = 0; index < dimensions; ++index) {
        const auto value = cursor.scalar<std::int64_t>("shape");
        if (value <= 0 ||
            value > std::numeric_limits<std::int32_t>::max() ||
            elements >
                std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(value)) {
            throw std::runtime_error("invalid dense MFQ shape");
        }
        shape.push_back(static_cast<std::int32_t>(value));
        elements *= static_cast<std::size_t>(value);
    }
    const auto [dtype, item_size] = dense_dtype(dtype_name);
    if (elements >
            std::numeric_limits<std::size_t>::max() / item_size ||
        cursor.remaining() != elements * item_size) {
        throw std::runtime_error("dense MFQ payload length mismatch");
    }

    auto result = array(
        mlx::core::allocator::malloc(cursor.remaining()),
        std::move(shape),
        dtype);
    std::memcpy(
        result.data<std::uint8_t>(),
        cursor.data(),
        cursor.remaining());
    return result;
}

MlxLinear MlxLinear::load(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (is_nint8_zero_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return MlxLinear(
            MlxNint8ZeroWeight::from_blob(mapped.view()));
    }
    if (is_nint_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return MlxLinear(MlxNintWeight::from_blob(mapped.view()));
    }
    if (is_vq_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return MlxLinear(
            MlxVqWeight::from_blob(record.dtype, mapped.view()));
    }
    if (record.dtype == "TPQ-I4G64" ||
        record.dtype == "CCCP-I4G64") {
        return MlxLinear(
            MlxCccpInt4Weight::from_blob(model.read(name)));
    }
    if (is_cccp_dtype(record.dtype)) {
        return MlxLinear(
            MlxCccpPqWeight::from_blob(
                record.dtype,
                model.read(name)));
    }
    if (is_mx_dtype(record.dtype)) {
        return MlxLinear(
            MlxMxWeight::from_blob(record.dtype, model.read(name)));
    }
    return MlxLinear(load_weight(model, name));
}

MlxLinear::MlxLinear(MlxNintWeight weight)
    : input_size_(weight.input_size()),
      output_size_(weight.output_size()),
      weight_(std::move(weight)) {}

MlxLinear::MlxLinear(MlxNint8ZeroWeight weight)
    : input_size_(weight.input_size()),
      output_size_(weight.output_size()),
      weight_(std::move(weight)) {}

MlxLinear::MlxLinear(MlxVqWeight weight)
    : input_size_(weight.input_size()),
      output_size_(weight.output_size()),
      weight_(std::move(weight)) {}

MlxLinear::MlxLinear(MlxCccpInt4Weight weight)
    : input_size_(weight.input_size()),
      output_size_(weight.output_size()),
      weight_(std::move(weight)) {}

MlxLinear::MlxLinear(MlxCccpPqWeight weight)
    : input_size_(weight.input_size()),
      output_size_(weight.output_size()),
      weight_(std::move(weight)) {}

MlxLinear::MlxLinear(MlxMxWeight weight)
    : input_size_(weight.input_size()),
      output_size_(weight.output_size()),
      weight_(std::move(weight)) {}

MlxLinear::MlxLinear(array weight)
    : weight_(std::move(weight)) {
    const auto& dense = std::get<array>(weight_);
    if (dense.ndim() != 2) {
        throw std::runtime_error("dense linear weight must have rank two");
    }
    output_size_ = dense.shape(0);
    input_size_ = dense.shape(1);
}

array MlxLinear::operator()(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error("linear input width mismatch");
    }
    if (const auto* packed = std::get_if<MlxNintWeight>(&weight_)) {
        return packed->matmul(input);
    }
    if (const auto* packed =
            std::get_if<MlxNint8ZeroWeight>(&weight_)) {
        return packed->matmul(input);
    }
    if (const auto* packed = std::get_if<MlxVqWeight>(&weight_)) {
        return packed->matmul(input);
    }
    if (const auto* packed =
            std::get_if<MlxCccpInt4Weight>(&weight_)) {
        return packed->matmul(input);
    }
    if (const auto* packed =
            std::get_if<MlxCccpPqWeight>(&weight_)) {
        return packed->matmul(input);
    }
    if (const auto* packed = std::get_if<MlxMxWeight>(&weight_)) {
        return packed->matmul(input);
    }
    const auto& dense = std::get<array>(weight_);
    auto source = input;
    if (source.dtype() != dense.dtype()) {
        source = mlx::core::astype(source, dense.dtype());
    }
    return mlx::core::matmul(source, mlx::core::transpose(dense));
}

array MlxLinear::grouped_row_matmul(
    const array& input,
    int group_count) const {
    if (group_count <= 0 ||
        input.ndim() < 2 ||
        input.shape(-2) != group_count ||
        input.shape(-1) != input_size_ ||
        output_size_ % group_count != 0) {
        throw std::runtime_error(
            "grouped-row linear shape/group mismatch");
    }
    if (const auto* packed =
            std::get_if<MlxCccpInt4Weight>(&weight_)) {
        return packed->grouped_row_matmul(
            input,
            group_count);
    }
    if (const auto* packed =
            std::get_if<MlxNint8ZeroWeight>(&weight_)) {
        return packed->grouped_row_matmul(
            input,
            group_count);
    }
    if (const auto* dense =
            std::get_if<array>(&weight_)) {
        auto source = input;
        if (source.dtype() != dense->dtype()) {
            source = mlx::core::astype(
                source,
                dense->dtype());
        }
        const auto grouped_weight =
            mlx::core::reshape(
                *dense,
                Shape{
                    group_count,
                    output_size_ / group_count,
                    input_size_,
                });
        return mlx::core::sum(
            mlx::core::expand_dims(source, -2) *
                grouped_weight,
            -1);
    }

    // The fallback intentionally stays on the original packed representation:
    // project each input group, then keep the output rows assigned to that
    // group. This matches the reference implementation for NINT/VQ/CCCP-PQ
    // and keeps correctness for uncommon O-LoRA weight formats.
    const auto complete = (*this)(input);
    const int output_per_group =
        output_size_ / group_count;
    std::vector<array> pieces;
    pieces.reserve(
        static_cast<std::size_t>(group_count));
    for (int group = 0; group < group_count; ++group) {
        auto selected = mlx::core::take(
            complete,
            group,
            complete.ndim() - 2);
        Shape starts(
            static_cast<std::size_t>(selected.ndim()),
            0);
        Shape stops = selected.shape();
        starts.back() = group * output_per_group;
        stops.back() = (group + 1) * output_per_group;
        pieces.push_back(
            mlx::core::slice(
                selected,
                starts,
                stops));
    }
    return mlx::core::stack(pieces, input.ndim() - 2);
}

std::optional<MlxGroupedLinearWeightRef>
MlxLinear::grouped_weight_ref() const noexcept {
    if (const auto* packed =
            std::get_if<MlxNintWeight>(&weight_)) {
        return MlxGroupedLinearWeightRef{packed};
    }
    if (const auto* packed =
            std::get_if<MlxNint8ZeroWeight>(&weight_)) {
        return MlxGroupedLinearWeightRef{packed};
    }
    if (const auto* packed =
            std::get_if<MlxVqWeight>(&weight_)) {
        return MlxGroupedLinearWeightRef{packed};
    }
    if (const auto* packed =
            std::get_if<MlxCccpInt4Weight>(&weight_)) {
        return MlxGroupedLinearWeightRef{packed};
    }
    if (const auto* packed =
            std::get_if<MlxCccpPqWeight>(&weight_)) {
        return MlxGroupedLinearWeightRef{packed};
    }
    return std::nullopt;
}

const MlxNintWeight* MlxLinear::nint_weight_ref() const noexcept {
    return std::get_if<MlxNintWeight>(&weight_);
}

const MlxNint8ZeroWeight*
MlxLinear::nint8_zero_weight_ref() const noexcept {
    return std::get_if<MlxNint8ZeroWeight>(&weight_);
}

const array* MlxLinear::dense_weight_ref() const noexcept {
    return std::get_if<array>(&weight_);
}

MlxEmbedding MlxEmbedding::load(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (is_nint8_zero_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return MlxEmbedding(
            MlxNint8ZeroWeight::from_blob(mapped.view()));
    }
    if (is_nint_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return MlxEmbedding(
            MlxNintWeight::from_blob(mapped.view()));
    }
    if (is_vq_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return MlxEmbedding(
            MlxVqWeight::from_blob(record.dtype, mapped.view()));
    }
    if (record.dtype == "TPQ-I4G64" ||
        record.dtype == "CCCP-I4G64") {
        return MlxEmbedding(
            MlxCccpInt4Weight::from_blob(model.read(name)));
    }
    if (is_cccp_dtype(record.dtype)) {
        throw std::runtime_error(
            "CCCP learned-PQ tensors do not support embedding lookup: " +
            name);
    }
    if (is_mx_dtype(record.dtype)) {
        return MlxEmbedding(
            MlxMxWeight::from_blob(record.dtype, model.read(name)));
    }
    return MlxEmbedding(load_weight(model, name));
}

MlxEmbedding::MlxEmbedding(MlxNintWeight weight)
    : vocabulary_size_(weight.output_size()),
      hidden_size_(weight.input_size()),
      weight_(std::move(weight)) {}

MlxEmbedding::MlxEmbedding(MlxNint8ZeroWeight weight)
    : vocabulary_size_(weight.output_size()),
      hidden_size_(weight.input_size()),
      weight_(std::move(weight)) {}

MlxEmbedding::MlxEmbedding(MlxVqWeight weight)
    : vocabulary_size_(weight.output_size()),
      hidden_size_(weight.input_size()),
      weight_(std::move(weight)) {}

MlxEmbedding::MlxEmbedding(MlxCccpInt4Weight weight)
    : vocabulary_size_(weight.output_size()),
      hidden_size_(weight.input_size()),
      weight_(std::move(weight)) {}

MlxEmbedding::MlxEmbedding(MlxMxWeight weight)
    : vocabulary_size_(weight.output_size()),
      hidden_size_(weight.input_size()),
      weight_(std::move(weight)) {}

MlxEmbedding::MlxEmbedding(array weight)
    : weight_(std::move(weight)) {
    const auto& dense = std::get<array>(weight_);
    if (dense.ndim() != 2) {
        throw std::runtime_error("dense embedding weight must have rank two");
    }
    vocabulary_size_ = dense.shape(0);
    hidden_size_ = dense.shape(1);
}

array MlxEmbedding::operator()(
    const array& token_ids,
    Dtype dtype) const {
    if (const auto* packed = std::get_if<MlxNintWeight>(&weight_)) {
        return packed->embedding(token_ids, dtype);
    }
    if (const auto* packed =
            std::get_if<MlxNint8ZeroWeight>(&weight_)) {
        return packed->embedding(token_ids, dtype);
    }
    if (const auto* packed = std::get_if<MlxVqWeight>(&weight_)) {
        return packed->embedding(token_ids, dtype);
    }
    if (const auto* packed =
            std::get_if<MlxCccpInt4Weight>(&weight_)) {
        return packed->embedding(token_ids, dtype);
    }
    if (const auto* packed = std::get_if<MlxMxWeight>(&weight_)) {
        return packed->embedding(token_ids, dtype);
    }
    auto ids = token_ids;
    if (ids.dtype() != mlx::core::int32 &&
        ids.dtype() != mlx::core::uint32) {
        ids = mlx::core::astype(ids, mlx::core::int32);
    }
    auto result = mlx::core::take(
        std::get<array>(weight_),
        ids,
        0);
    return result.dtype() == dtype
        ? result
        : mlx::core::astype(result, dtype);
}

array MlxEmbedding::project(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != hidden_size_) {
        throw std::runtime_error(
            "embedding projection input width mismatch");
    }
    if (const auto* packed =
            std::get_if<MlxNintWeight>(&weight_)) {
        return packed->matmul(input);
    }
    if (const auto* packed =
            std::get_if<MlxNint8ZeroWeight>(&weight_)) {
        return packed->matmul(input);
    }
    if (const auto* packed = std::get_if<MlxVqWeight>(&weight_)) {
        return packed->matmul(input);
    }
    if (const auto* packed =
            std::get_if<MlxCccpInt4Weight>(&weight_)) {
        return packed->matmul(input);
    }
    if (const auto* packed = std::get_if<MlxMxWeight>(&weight_)) {
        return packed->matmul(input);
    }
    const auto& dense = std::get<array>(weight_);
    auto source = input;
    if (source.dtype() != dense.dtype()) {
        source = mlx::core::astype(source, dense.dtype());
    }
    return mlx::core::matmul(
        source,
        mlx::core::transpose(dense));
}

} // namespace mfq::metal
