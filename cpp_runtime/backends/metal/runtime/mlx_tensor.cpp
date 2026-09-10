#include "mlx_tensor.h"
#include "mlx_reference.h"

#include <mlx/allocator.h>

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Dtype;
using mlx::core::CompileOptions;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

std::atomic_bool g_predequantize_fp16{false};

// Decode verification presents two through six hidden states at once.  MLX's
// general GEMM path does not reuse a dense weight row efficiently at this M,
// so one SIMD group owns an output and accumulates every input row while the
// weight vector is resident in registers.
constexpr const char* kDenseSmallM = R"METAL(
    constexpr uint K_LANES_VALUE = uint(K_LANES);
    constexpr uint SIMD_GROUPS_VALUE = uint(SIMD_GROUPS);
    constexpr uint OUTPUTS_PER_SIMD = 32u / K_LANES_VALUE;
    constexpr uint OUTPUTS_PER_TG =
        SIMD_GROUPS_VALUE * OUTPUTS_PER_SIMD;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES_VALUE - 1u);
    uint simd_output = lane / K_LANES_VALUE;
    uint output_index =
        threadgroup_position_in_grid.y * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD + simd_output;
    uint output = min(output_index, uint(OUT) - 1u);

    float accumulators[M];
    for (uint row = 0u; row < uint(M); ++row) {
        accumulators[row] = 0.0f;
    }

    uint weight_base = output * uint(K);
    for (uint vector = k_lane;
         vector < uint(K) / 4u;
         vector += K_LANES_VALUE) {
        uint column = vector * 4u;
        vec<T, 4> packed_weight = *(device const vec<T, 4>*)(
            weight + weight_base + column);
        float4 weight_values = float4(packed_weight);
        for (uint row = 0u; row < uint(M); ++row) {
            vec<T, 4> packed_input = *(device const vec<T, 4>*)(
                x + row * uint(K) + column);
            accumulators[row] += dot(float4(packed_input), weight_values);
        }
    }

    for (uint row = 0u; row < uint(M); ++row) {
        if (K_LANES_VALUE >= 32u) {
            accumulators[row] += simd_shuffle_down(accumulators[row], 16u);
        }
        if (K_LANES_VALUE >= 16u) {
            accumulators[row] += simd_shuffle_down(accumulators[row], 8u);
        }
        if (K_LANES_VALUE >= 8u) {
            accumulators[row] += simd_shuffle_down(accumulators[row], 4u);
        }
        accumulators[row] += simd_shuffle_down(accumulators[row], 2u);
        accumulators[row] += simd_shuffle_down(accumulators[row], 1u);
        if (k_lane == 0u && output_index < uint(OUT)) {
            y[row * uint(OUT) + output_index] = T(accumulators[row]);
        }
    }
)METAL";

// M=2..6 verifier GEMV with the same 32-lane reduction geometry as MLX's
// one-token large-output GEMV. Four output rows share every activation load
// inside each SIMD group, while each weight is reused across all M inputs.
//
// Scheduling derived from oMLX 0.6.4 verify_qmv.py.
// Copyright © 2026 Apple Inc.  Licensed under Apache-2.0.
constexpr const char* kDenseSmallMExact = R"METAL(
    constexpr uint OUTPUTS_PER_SIMD = 4u;
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint OUTPUTS_PER_TG = OUTPUTS_PER_SIMD * SIMD_GROUPS;
    constexpr uint VALUES_PER_LANE = 4u;
    constexpr uint K_BLOCK = VALUES_PER_LANE * 32u;

    uint simd_group = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint output_base =
        threadgroup_position_in_grid.x * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD;
    if (output_base >= uint(OUT)) {
        return;
    }

    uint k_base = lane * VALUES_PER_LANE;
    float accum[M][OUTPUTS_PER_SIMD] = {{0.0f}};
    for (uint block = 0u; block < uint(K); block += K_BLOCK) {
        float input_values[M][VALUES_PER_LANE];
        for (uint row = 0u; row < uint(M); ++row) {
            for (uint column = 0u;
                 column < VALUES_PER_LANE;
                 ++column) {
                input_values[row][column] =
                    float(x[row * uint(K) + k_base + column]);
            }
        }
        for (uint result = 0u;
             result < OUTPUTS_PER_SIMD;
             ++result) {
            device const T* row_weight =
                weight + (output_base + result) * uint(K) + k_base;
            float weight_values[VALUES_PER_LANE];
            for (uint column = 0u;
                 column < VALUES_PER_LANE;
                 ++column) {
                weight_values[column] = float(row_weight[column]);
            }
            for (uint row = 0u; row < uint(M); ++row) {
                for (uint column = 0u;
                     column < VALUES_PER_LANE;
                     ++column) {
                    accum[row][result] +=
                        weight_values[column] * input_values[row][column];
                }
            }
        }
        k_base += K_BLOCK;
    }

    for (uint row = 0u; row < uint(M); ++row) {
        for (uint result = 0u;
             result < OUTPUTS_PER_SIMD;
             ++result) {
            for (ushort step = 16u; step >= 1u; step >>= 1u) {
                accum[row][result] +=
                    simd_shuffle_down(accum[row][result], step);
            }
            if (lane == 0u) {
                y[row * uint(OUT) + output_base + result] =
                    T(accum[row][result]);
            }
        }
    }
)METAL";

const mlx::core::fast::CustomKernelFunction& dense_small_m_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_dense_small_m_m2_6",
            {"weight", "x"},
            {"y"},
            kDenseSmallM,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
dense_small_m_exact_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_dense_small_m_exact_m2_6",
            {"weight", "x"},
            {"y"},
            kDenseSmallMExact,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

array dense_small_m_matmul(
    const array& weight,
    const array& input,
    int rows,
    int input_size,
    int output_size) {
    const auto* layout = std::getenv(
        "MFQ_METAL_DENSE_SMALL_M_LAYOUT");
    const bool exact =
        input_size % 128 == 0 &&
        output_size >= 4096 &&
        output_size % 32 == 0 &&
        (layout == nullptr || std::strcmp(layout, "legacy") != 0);
    if (exact) {
        auto source = mlx::core::reshape(
            input.flags().row_contiguous
                ? input
                : mlx::core::contiguous(input),
            Shape{rows, input_size});
        auto outputs = dense_small_m_exact_kernel()(
            {weight, std::move(source)},
            {Shape{rows, output_size}},
            {weight.dtype()},
            {
                ((output_size + 31) / 32) * 256,
                1,
                1,
            },
            {256, 1, 1},
            {
                {"T", weight.dtype()},
                {"M", rows},
                {"K", input_size},
                {"OUT", output_size},
            },
            std::nullopt,
            false,
            {});
        Shape output_shape = input.shape();
        output_shape.back() = output_size;
        return mlx::core::reshape(
            std::move(outputs.front()),
            std::move(output_shape));
    }
    constexpr int simd_groups = 4;
    const int k_lanes = input_size >= 16384 ? 16 : 32;
    const int outputs_per_threadgroup = simd_groups * 32 / k_lanes;
    auto source = mlx::core::reshape(
        input.flags().row_contiguous ? input : mlx::core::contiguous(input),
        Shape{rows, input_size});
    auto outputs = dense_small_m_kernel()(
        {weight, std::move(source)},
        {Shape{rows, output_size}},
        {weight.dtype()},
        {
            simd_groups * 32,
            (output_size + outputs_per_threadgroup - 1) /
                outputs_per_threadgroup,
            1,
        },
        {simd_groups * 32, 1, 1},
        {
            {"T", weight.dtype()},
            {"M", rows},
            {"K", input_size},
            {"OUT", output_size},
            {"K_LANES", k_lanes},
            {"SIMD_GROUPS", simd_groups},
        },
        std::nullopt,
        false,
        {});
    Shape output_shape = input.shape();
    output_shape.back() = output_size;
    return mlx::core::reshape(
        std::move(outputs.front()),
        std::move(output_shape));
}

array dense_decode_consistent_matmul(
    const array& weight,
    const array& input,
    int rows,
    int input_size,
    int output_size) {
    // This is the same fallback used by oMLX's decode-consistency patch:
    // express M independent projections as a batch of matrix-vector
    // products. MLX consequently keeps its M=1 GEMV reduction for every row
    // instead of selecting a width-M GEMM kernel.
    auto source = mlx::core::reshape(
        input.flags().row_contiguous
            ? input
            : mlx::core::contiguous(input),
        Shape{rows, input_size});
    auto projected = mlx::core::matmul(
        weight,
        mlx::core::expand_dims(std::move(source), -1));
    projected = mlx::core::squeeze(std::move(projected), -1);
    Shape output_shape = input.shape();
    output_shape.back() = output_size;
    return mlx::core::reshape(
        std::move(projected),
        std::move(output_shape));
}

template <typename Variant>
array materialize_weight_fp16(
    const Variant& weight,
    int output_size) {
    const auto row_ids = mlx::core::arange(
        0,
        output_size,
        1,
        mlx::core::int32);
    auto dense = std::visit(
        [&](const auto& value) -> array {
            using Weight = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Weight, array>) {
                return value.dtype() == mlx::core::float16
                    ? value
                    : mlx::core::astype(value, mlx::core::float16);
            } else if constexpr (
                std::is_same_v<Weight, MlxTpqPqWeight>
                || std::is_same_v<Weight, MlxMxWeight>
            ) {
                return value.dequantize(mlx::core::float16);
            } else {
                return value.embedding(
                    row_ids,
                    mlx::core::float16);
            }
        },
        weight);
    dense.eval();
    return dense;
}

template <typename Variant>
std::optional<array> unpack_quantized_weight(
    const Variant& weight,
    int output_size) {
    const auto row_ids = mlx::core::arange(
        0,
        output_size,
        1,
        mlx::core::int32);
    return std::visit(
        [&](const auto& value) -> std::optional<array> {
            using Weight = std::decay_t<decltype(value)>;
            if constexpr (
                std::is_same_v<Weight, MlxNintWeight>
                || std::is_same_v<Weight, MlxNint8ZeroWeight>
                || std::is_same_v<Weight, MlxVqWeight>
                || std::is_same_v<Weight, MlxTpqInt4Weight>
            ) {
                return mlx::core::astype(
                    value.embedding(
                        row_ids,
                        mlx::core::float32),
                    mlx::core::float16);
            } else if constexpr (
                std::is_same_v<Weight, MlxTpqPqWeight>
            ) {
                return mlx::core::astype(
                    value.dequantize(
                        mlx::core::float32),
                    mlx::core::float16);
            } else {
                return std::nullopt;
            }
        },
        weight);
}

array dense_reference_matmul(
    const array& dense,
    const array& input) {
    auto source = input.dtype() == mlx::core::float16
        ? input
        : mlx::core::astype(
              input,
              mlx::core::float16);
    return mlx::core::matmul(
        source,
        mlx::core::transpose(dense));
}

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

void set_mlx_predequantize_fp16(bool enabled) noexcept {
    g_predequantize_fp16.store(enabled, std::memory_order_relaxed);
}

bool mlx_predequantize_fp16_enabled() noexcept {
    return g_predequantize_fp16.load(std::memory_order_relaxed);
}

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
    const auto finish = [](MlxLinear result) {
        if (mlx_predequantize_fp16_enabled()) {
            result.materialize_fp16();
        }
        return result;
    };
    const auto& record = model.record(name);
    if (is_nint8_zero_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return finish(MlxLinear(
            MlxNint8ZeroWeight::from_blob(mapped.view())));
    }
    if (is_nint_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return finish(MlxLinear(
            MlxNintWeight::from_blob(mapped.view())));
    }
    if (is_vq_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return finish(MlxLinear(
            MlxVqWeight::from_blob(record.dtype, mapped.view())));
    }
    if (record.dtype == "TPQ-I4G64" ||
        record.dtype == "TPQ-I4G64") {
        return finish(MlxLinear(
            MlxTpqInt4Weight::from_blob(model.read(name))));
    }
    if (is_tpq_dtype(record.dtype)) {
        return finish(MlxLinear(
            MlxTpqPqWeight::from_blob(
                record.dtype,
                model.read(name))));
    }
    if (is_mx_dtype(record.dtype)) {
        return finish(MlxLinear(
            MlxMxWeight::from_blob(record.dtype, model.read(name))));
    }
    return finish(MlxLinear(load_weight(model, name)));
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

MlxLinear::MlxLinear(MlxTpqInt4Weight weight)
    : input_size_(weight.input_size()),
      output_size_(weight.output_size()),
      weight_(std::move(weight)) {}

MlxLinear::MlxLinear(MlxTpqPqWeight weight)
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

std::optional<array> MlxLinear::greedy_argmax(
    const array& input) const {
    if (const auto* packed = std::get_if<MlxNintWeight>(&weight_)) {
        return packed->greedy_argmax(input);
    }
    return std::nullopt;
}

array MlxLinear::operator()(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error("linear input width mismatch");
    }
    if (mlx_reference_enabled()) {
        if (auto dense = unpack_quantized_weight(
                weight_, output_size_)) {
            return dense_reference_matmul(*dense, input);
        }
    }
    const auto preserve_input_dtype = [&](array result) {
        return input.dtype() == mlx::core::bfloat16 &&
                result.dtype() != input.dtype()
            ? mlx::core::astype(result, input.dtype())
            : result;
    };
    if (const auto* packed = std::get_if<MlxNintWeight>(&weight_)) {
        return preserve_input_dtype(packed->matmul(input));
    }
    if (const auto* packed =
            std::get_if<MlxNint8ZeroWeight>(&weight_)) {
        return preserve_input_dtype(packed->matmul(input));
    }
    if (const auto* packed = std::get_if<MlxVqWeight>(&weight_)) {
        return preserve_input_dtype(packed->matmul(input));
    }
    if (const auto* packed =
            std::get_if<MlxTpqInt4Weight>(&weight_)) {
        return preserve_input_dtype(packed->matmul(input));
    }
    if (const auto* packed =
            std::get_if<MlxTpqPqWeight>(&weight_)) {
        return preserve_input_dtype(packed->matmul(input));
    }
    if (const auto* packed = std::get_if<MlxMxWeight>(&weight_)) {
        return preserve_input_dtype(packed->matmul(input));
    }
    const auto& dense = std::get<array>(weight_);
    auto source = input;
    if (source.dtype() != dense.dtype()) {
        source = mlx::core::astype(source, dense.dtype());
    }
    const auto rows = source.size() / static_cast<std::size_t>(input_size_);
    if (rows >= 2 && rows <= 6 &&
        input_size_ % 4 == 0 &&
        dense.size() >= 65536 &&
        dense.flags().row_contiguous) {
        const auto* layout = std::getenv(
            "MFQ_METAL_DENSE_SMALL_M_LAYOUT");
        const bool legacy = layout != nullptr &&
            std::strcmp(layout, "legacy") == 0;
        if (!legacy &&
            output_size_ < 4096) {
            return dense_decode_consistent_matmul(
                dense,
                source,
                static_cast<int>(rows),
                input_size_,
                output_size_);
        }
        if (dense.dtype() == mlx::core::float16 ||
            dense.dtype() == mlx::core::bfloat16) {
            return dense_small_m_matmul(
                dense,
                source,
                static_cast<int>(rows),
                input_size_,
                output_size_);
        }
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
    if (mlx_reference_enabled()) {
        if (auto dense = unpack_quantized_weight(
                weight_, output_size_)) {
            auto source = input.dtype() == mlx::core::float16
                ? input
                : mlx::core::astype(
                      input,
                      mlx::core::float16);
            const auto grouped_weight = mlx::core::reshape(
                *dense,
                Shape{
                    group_count,
                    output_size_ / group_count,
                    input_size_,
                });
            return mlx::core::sum(
                mlx::core::expand_dims(source, -2)
                    * grouped_weight,
                -1);
        }
    }
    if (const auto* packed = std::get_if<MlxNintWeight>(&weight_)) {
        if (auto result = packed->grouped_row_matmul(input, group_count)) {
            return input.dtype() == mlx::core::bfloat16
                    && result->dtype() != input.dtype()
                ? mlx::core::astype(*result, input.dtype())
                : *result;
        }
    }
    if (const auto* packed =
            std::get_if<MlxTpqInt4Weight>(&weight_)) {
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
    if (const auto* packed =
            std::get_if<MlxMxWeight>(&weight_);
        packed != nullptr && packed->bits() == 8) {
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
    // group. This matches the reference implementation for NINT/VQ/TPQ-PQ
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
    if (mlx_reference_enabled()) {
        if (const auto* packed =
                std::get_if<MlxMxWeight>(&weight_)) {
            return MlxGroupedLinearWeightRef{packed};
        }
        return std::nullopt;
    }
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
            std::get_if<MlxTpqInt4Weight>(&weight_)) {
        return MlxGroupedLinearWeightRef{packed};
    }
    if (const auto* packed =
            std::get_if<MlxTpqPqWeight>(&weight_)) {
        return MlxGroupedLinearWeightRef{packed};
    }
    if (const auto* packed =
            std::get_if<MlxMxWeight>(&weight_)) {
        return MlxGroupedLinearWeightRef{packed};
    }
    return std::nullopt;
}

const MlxNintWeight* MlxLinear::nint_weight_ref() const noexcept {
    if (mlx_reference_enabled()) return nullptr;
    return std::get_if<MlxNintWeight>(&weight_);
}

const MlxNint8ZeroWeight*
MlxLinear::nint8_zero_weight_ref() const noexcept {
    if (mlx_reference_enabled()) return nullptr;
    return std::get_if<MlxNint8ZeroWeight>(&weight_);
}

const MlxMxWeight* MlxLinear::mx_weight_ref() const noexcept {
    return std::get_if<MlxMxWeight>(&weight_);
}

const array* MlxLinear::dense_weight_ref() const noexcept {
    return std::get_if<array>(&weight_);
}

void MlxLinear::materialize_fp16() {
    if (std::holds_alternative<array>(weight_)) return;
    auto dense = materialize_weight_fp16(weight_, output_size_);
    weight_ = std::move(dense);
}

MlxEmbedding MlxEmbedding::load(
    const MfqContainer& model,
    const std::string& name) {
    const auto finish = [](MlxEmbedding result) {
        if (mlx_predequantize_fp16_enabled()) {
            result.materialize_fp16();
        }
        return result;
    };
    const auto& record = model.record(name);
    if (is_nint8_zero_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return finish(MlxEmbedding(
            MlxNint8ZeroWeight::from_blob(mapped.view())));
    }
    if (is_nint_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return finish(MlxEmbedding(
            MlxNintWeight::from_blob(mapped.view())));
    }
    if (is_vq_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return finish(MlxEmbedding(
            MlxVqWeight::from_blob(record.dtype, mapped.view())));
    }
    if (record.dtype == "TPQ-I4G64" ||
        record.dtype == "TPQ-I4G64") {
        return finish(MlxEmbedding(
            MlxTpqInt4Weight::from_blob(model.read(name))));
    }
    if (is_tpq_dtype(record.dtype)) {
        throw std::runtime_error(
            "TPQ learned-PQ tensors do not support embedding lookup: " +
            name);
    }
    if (is_mx_dtype(record.dtype)) {
        return finish(MlxEmbedding(
            MlxMxWeight::from_blob(record.dtype, model.read(name))));
    }
    return finish(MlxEmbedding(load_weight(model, name)));
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

MlxEmbedding::MlxEmbedding(MlxTpqInt4Weight weight)
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

void MlxEmbedding::materialize_fp16() {
    if (std::holds_alternative<array>(weight_)) return;
    auto dense = materialize_weight_fp16(weight_, vocabulary_size_);
    weight_ = std::move(dense);
}

array MlxEmbedding::operator()(
    const array& token_ids,
    Dtype dtype) const {
    const bool reference = mlx_reference_enabled();
    const auto finish_quantized = [&](const auto& packed) {
        auto result = reference
            ? mlx::core::astype(
                  packed.embedding(
                      token_ids,
                      mlx::core::float32),
                  mlx::core::float16)
            : packed.embedding(
                  token_ids,
                  dtype == mlx::core::bfloat16
                      ? mlx::core::float16
                      : dtype);
        return result.dtype() == dtype
            ? result
            : mlx::core::astype(result, dtype);
    };
    if (const auto* packed = std::get_if<MlxNintWeight>(&weight_)) {
        return finish_quantized(*packed);
    }
    if (const auto* packed =
            std::get_if<MlxNint8ZeroWeight>(&weight_)) {
        return finish_quantized(*packed);
    }
    if (const auto* packed = std::get_if<MlxVqWeight>(&weight_)) {
        return finish_quantized(*packed);
    }
    if (const auto* packed =
            std::get_if<MlxTpqInt4Weight>(&weight_)) {
        return finish_quantized(*packed);
    }
    if (const auto* packed = std::get_if<MlxMxWeight>(&weight_)) {
        auto result = packed->embedding(
            token_ids,
            dtype == mlx::core::bfloat16
                ? mlx::core::float16
                : dtype);
        return result.dtype() == dtype
            ? result
            : mlx::core::astype(result, dtype);
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
    if (mlx_reference_enabled()) {
        if (auto dense = unpack_quantized_weight(
                weight_, vocabulary_size_)) {
            return dense_reference_matmul(*dense, input);
        }
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
            std::get_if<MlxTpqInt4Weight>(&weight_)) {
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
