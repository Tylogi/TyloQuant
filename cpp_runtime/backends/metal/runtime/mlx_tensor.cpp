#include "mlx_tensor.h"
#include "mlx_kernel_prepare.h"
#include "mlx_moe_ops.h"
#include "mlx_reference.h"
#include "mlx_transformer.h"
#include "mlx_weight_residency.h"

#include <mlx/allocator.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iterator>
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
thread_local unsigned int g_fp16_weight_scopes = 0;

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
                std::is_same_v<Weight, MlxFp8SqWeight>
                || std::is_same_v<Weight, MlxMxWeight>
                || std::is_same_v<Weight, MlxMxfp4SqWeight>
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
                || std::is_same_v<Weight, MlxTqWeight>
            ) {
                return mlx::core::astype(
                    value.embedding(
                        row_ids,
                        mlx::core::float32),
                    mlx::core::float16);
            } else if constexpr (
                std::is_same_v<Weight, MlxFp8SqWeight>
                || std::is_same_v<Weight, MlxMxfp4SqWeight>
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
    return g_fp16_weight_scopes > 0 || g_predequantize_fp16.load(std::memory_order_relaxed);
}

MlxFp16WeightScope::MlxFp16WeightScope(bool enabled) noexcept : enabled_(enabled) {
    if (enabled_) ++g_fp16_weight_scopes;
}
MlxFp16WeightScope::~MlxFp16WeightScope() {
    if (enabled_) --g_fp16_weight_scopes;
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
    MlxWeightResidency::track(result);
    return result;
}

MlxLinear MlxLinear::load(
    const MfqContainer& model,
    const std::string& name) {
    const auto finish = [](MlxLinear result) {
        if (mlx_predequantize_fp16_enabled()) {
            result.materialize_fp16();
        }
        if (auto* preparation = MlxKernelPreparation::current();
            preparation && !preparation->model_paths() && result.packed()) {
            preparation->collect([&] {
                for (const auto rows : preparation->row_buckets()) {
                    for (const auto dtype : {mlx::core::float16, mlx::core::float32}) {
                        const auto input = mlx::core::zeros(Shape{rows, result.input_size()}, dtype);
                        preparation->add(result(input));
                        if (rows >= 64 && dtype == mlx::core::float16) {
                            if (const auto* nint = result.nint_weight_ref()) {
                                preparation->add(nint->matmul_packed(input));
                                if (std::int64_t(nint->input_size()) * nint->output_size() <= std::numeric_limits<int>::max())
                                    preparation->add(nint->dequantize(mlx::core::float16));
                            }
                        }
                        if (rows == 1) {
                            if (auto greedy = result.greedy_argmax(input)) preparation->add(*greedy);
                        }
                    }
                }
            });
        }
        return result;
    };
    const auto& record = model.record(name);
    if (record.dtype == "TQ") return finish(MlxLinear(MlxTqWeight::load(model, name)));
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
    if (is_mxfp4_sq_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return finish(MlxLinear(
            MlxMxfp4SqWeight::from_blob(mapped.view())));
    }
    if (is_fp8_sq_dtype(record.dtype)) {
        const auto mapped = model.map_record(name);
        return finish(MlxLinear(
            MlxFp8SqWeight::from_blob(record.dtype, mapped.view())));
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

MlxLinear::MlxLinear(MlxTqWeight weight)
    : weight_(weight), input_size_(weight.input_size()), output_size_(weight.output_size()) {}

MlxLinear::MlxLinear(MlxNint8ZeroWeight weight)
    : input_size_(weight.input_size()),
      output_size_(weight.output_size()),
      weight_(std::move(weight)) {}

MlxLinear::MlxLinear(MlxVqWeight weight)
    : input_size_(weight.input_size()),
      output_size_(weight.output_size()),
      weight_(std::move(weight)) {}

MlxLinear::MlxLinear(MlxFp8SqWeight weight)
    : input_size_(weight.input_size()),
      output_size_(weight.output_size()),
      weight_(std::move(weight)) {}

MlxLinear::MlxLinear(MlxMxWeight weight)
    : input_size_(weight.input_size()),
      output_size_(weight.output_size()),
      weight_(std::move(weight)) {}

MlxLinear::MlxLinear(MlxMxfp4SqWeight weight)
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
    if (const auto* tq = std::get_if<MlxTqWeight>(&weight_)) return tq->matmul(input);
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
    if (const auto* packed = std::get_if<MlxFp8SqWeight>(&weight_)) {
        return preserve_input_dtype(packed->matmul(input));
    }
    if (const auto* packed = std::get_if<MlxMxWeight>(&weight_)) {
        return preserve_input_dtype(packed->matmul(input));
    }
    if (const auto* packed =
            std::get_if<MlxMxfp4SqWeight>(&weight_)) {
        return preserve_input_dtype(packed->matmul(input));
    }
    const auto& dense = std::get<array>(weight_);
    auto source = input;
    if (source.dtype() != dense.dtype()) {
        source = mlx::core::astype(source, dense.dtype());
    }
    const auto rows = source.size() / static_cast<std::size_t>(input_size_);
    if (rows == 1 &&
        moe_dense_router_logits_supported(source, dense)) {
        auto output_shape = input.shape();
        output_shape.back() = output_size_;
        return mlx::core::reshape(
            moe_dense_router_logits(source, dense),
            std::move(output_shape));
    }
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

array MlxLinear::moe_shared(
    const array& input,
    const array& routed_pairs,
    const array& route_weights,
    const array& gate_logits) const {
    if (const auto* packed = std::get_if<MlxNintWeight>(&weight_);
        packed != nullptr && !mlx_reference_enabled()) {
        return packed->matmul_moe_shared(
            input,
            routed_pairs,
            route_weights,
            gate_logits);
    }
    return moe_weighted_reduce_shared_gate(
        routed_pairs,
        route_weights,
        (*this)(input),
        gate_logits);
}

void MlxLinear::prepare_grouped_row_matmul(int group_count) const {
    if (!tq_weight_ref()) return;
    if (auto* preparation = MlxKernelPreparation::current()) preparation->collect([&] {
        for (const auto rows : preparation->row_buckets())
            for (auto dtype : {mlx::core::float16, mlx::core::float32})
                preparation->add(grouped_row_matmul(
                    mlx::core::zeros(Shape{rows, group_count, input_size_}, dtype), group_count));
    });
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
    if (const auto* tq = std::get_if<MlxTqWeight>(&weight_)) return tq->grouped_row_matmul(input, group_count);
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
    // group. This keeps correctness for uncommon O-LoRA weight formats.
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

array MlxLinear::grouped_row_matmul_inverse_rope(
    const array& value,
    int group_count,
    const array& cosine,
    const array& sine,
    int head_dimension,
    int rotary_dimension) const {
    if (value.ndim() != 4 ||
        group_count <= 0 ||
        head_dimension <= 0 ||
        rotary_dimension <= 0 ||
        rotary_dimension > head_dimension ||
        rotary_dimension % 2 != 0 ||
        value.shape(-1) != head_dimension ||
        value.shape(-2) * head_dimension != group_count * input_size_) {
        throw std::runtime_error(
            "inverse-RoPE grouped-row linear shape/group mismatch");
    }
    const int batch = value.shape(0);
    const int tokens = value.shape(1);
    const auto grouped_shape = Shape{
        batch,
        tokens,
        group_count,
        input_size_,
    };
    auto grouped = mlx::core::reshape(value, grouped_shape);

    if (const auto* packed =
            std::get_if<MlxNint8ZeroWeight>(&weight_)) {
        return packed->grouped_row_matmul_inverse_rope(
            grouped,
            group_count,
            cosine,
            sine,
            head_dimension,
            rotary_dimension);
    }
    if (const auto* packed = std::get_if<MlxMxWeight>(&weight_)) {
        const auto rows = static_cast<std::int64_t>(batch) * tokens;
        const bool block32_eligible =
            packed->scale_column_block_size() == 32 &&
            rows == 1 &&
            value.dtype() == mlx::core::float32;
        const bool block128_eligible =
            packed->scale_column_block_size() == 128 &&
            rows >= 1 && rows <= 6;
        if (packed->bits() == 8 &&
            cosine.ndim() == 2 &&
            cosine.shape(0) == rows &&
            (block32_eligible || block128_eligible)) {
            return packed->grouped_row_matmul_inverse_rope(
                grouped,
                group_count,
                cosine,
                sine,
                head_dimension,
                rotary_dimension);
        }
    }

    // mlx_rope_adjacent treats rotary_dimension as a suffix width, matching
    // DeepSeek's decoupled-RoPE layout while preserving the leading content
    // channels in the same kernel dispatch.
    auto unrotated = mlx_rope_adjacent(
        value,
        rotary_dimension,
        cosine,
        sine,
        true);
    return grouped_row_matmul(
        mlx::core::reshape(std::move(unrotated), grouped_shape),
        group_count);
}

std::optional<MlxGroupedLinearWeightRef>
MlxLinear::grouped_weight_ref() const noexcept {
    if (mlx_reference_enabled()) {
        if (const auto* packed =
                std::get_if<MlxMxWeight>(&weight_)) {
            if (packed->bits() == 8 &&
                (packed->scale_row_block_size() != 128 ||
                 packed->scale_column_block_size() != 128)) {
                return std::nullopt;
            }
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
            std::get_if<MlxFp8SqWeight>(&weight_)) {
        return MlxGroupedLinearWeightRef{packed};
    }
    if (const auto* packed =
            std::get_if<MlxMxfp4SqWeight>(&weight_)) {
        return MlxGroupedLinearWeightRef{packed};
    }
    if (const auto* packed =
            std::get_if<MlxMxWeight>(&weight_)) {
        return MlxGroupedLinearWeightRef{packed};
    }
    if (const auto* dense =
            std::get_if<array>(&weight_)) {
        return MlxGroupedLinearWeightRef{dense};
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

const MlxTqWeight* MlxLinear::tq_weight_ref() const noexcept {
    return std::get_if<MlxTqWeight>(&weight_);
}

const array* MlxLinear::dense_weight_ref() const noexcept {
    return std::get_if<array>(&weight_);
}

void MlxLinear::materialize_fp16() {
    if (std::holds_alternative<MlxTqWeight>(weight_)) return;
    if (const auto* dense = std::get_if<array>(&weight_);
        dense && (!g_fp16_weight_scopes || dense->dtype() == mlx::core::float16)) return;
    auto dense = materialize_weight_fp16(weight_, output_size_);
    weight_ = std::move(dense);
}

std::optional<MlxGroupedLinear> mlx_group_linears(
    std::span<const MlxLinear* const> linears) {
    if (linears.size() < 2) return std::nullopt;
    std::vector<MlxGroupedLinearWeightRef> references;
    references.reserve(linears.size());
    for (const auto* linear : linears) {
        if (linear == nullptr) return std::nullopt;
        auto reference = linear->grouped_weight_ref();
        if (!reference) return std::nullopt;
        references.push_back(*reference);
    }
    try {
        return MlxGroupedLinear(std::move(references));
    } catch (const MlxGroupedLinearUnsupported&) {
        return std::nullopt;
    }
}

struct MlxProjectionBatch::Impl {
    struct Segment {
        std::vector<std::size_t> indices;
        std::optional<MlxGroupedLinear> grouped;
        std::vector<MlxTqWeight> tq;
    };

    explicit Impl(std::vector<const MlxLinear*> sources) {
        if (sources.empty() || std::any_of(
                sources.begin(),
                sources.end(),
                [](const MlxLinear* linear) {
                    return linear == nullptr;
                })) {
            throw std::invalid_argument(
                "projection batch requires non-null projections");
        }
        linears.reserve(sources.size());
        for (const auto* source : sources) {
            linears.push_back(*source);
        }

        std::vector<const MlxLinear*> references;
        references.reserve(linears.size());
        for (const auto& linear : linears) {
            references.push_back(&linear);
        }

        std::size_t begin = 0;
        while (begin < references.size()) {
            const std::size_t remaining = references.size() - begin;
            if (const auto* first = references[begin]->tq_weight_ref()) {
                Segment segment;
                for (std::size_t offset = 0; offset < std::min(remaining, std::size_t(8)); ++offset) {
                    const auto* weight = references[begin + offset]->tq_weight_ref();
                    if (!weight || weight->input_size() != first->input_size()) break;
                    segment.indices.push_back(begin + offset);
                    segment.tq.push_back(*weight);
                }
                if (segment.tq.size() >= 2) {
                    grouped_projection_count += segment.tq.size();
                    begin += segment.tq.size();
                    segments.push_back(std::move(segment));
                    continue;
                }
            }
            std::optional<MlxGroupedLinear> selected;
            std::size_t selected_count = 0;
            for (std::size_t count = remaining; count >= 2; --count) {
                // Avoid leaving a lone compatible projection when two
                // balanced groups can cover the same four-member suffix.
                if (remaining - count == 1 && count > 2) {
                    continue;
                }
                auto candidate = mlx_group_linears(
                    std::span<const MlxLinear* const>(
                        references.data() + begin,
                        count));
                if (candidate && candidate->has_projection_fusion()) {
                    selected = std::move(candidate);
                    selected_count = count;
                    break;
                }
            }

            const std::size_t count = selected ? selected_count : 1;
            std::vector<std::size_t> indices;
            indices.reserve(count);
            for (std::size_t offset = 0; offset < count; ++offset)
                indices.push_back(begin + offset);
            segments.push_back(Segment{
                .indices = std::move(indices),
                .grouped = std::move(selected),
            });
            if (segments.back().grouped) {
                grouped_projection_count += count;
            }
            begin += count;
        }
        for (std::size_t first = 0; first < segments.size(); ++first) {
            if (segments[first].grouped || !segments[first].tq.empty()) continue;
            for (std::size_t second = first + 1; second < segments.size(); ++second) {
                if (segments[second].grouped || !segments[second].tq.empty()) continue;
                const std::array<const MlxLinear*, 2> pair{
                    references[segments[first].indices.front()],
                    references[segments[second].indices.front()]};
                auto candidate = mlx_group_linears(pair);
                if (!candidate || !candidate->has_projection_fusion()) continue;
                segments[first].indices.push_back(segments[second].indices.front());
                segments[first].grouped = std::move(candidate);
                grouped_projection_count += 2;
                segments.erase(segments.begin() + second);
                break;
            }
        }
    }

    std::vector<MlxLinear> linears;
    std::vector<Segment> segments;
    std::size_t grouped_projection_count = 0;
};

MlxProjectionBatch::MlxProjectionBatch(
    std::vector<const MlxLinear*> linears)
    : impl_(std::make_shared<Impl>(std::move(linears))) {
    if (auto* preparation = MlxKernelPreparation::current(); preparation && !preparation->model_paths()) {
        preparation->collect([&] {
            for (const auto rows : preparation->row_buckets()) {
                for (const auto dtype : {mlx::core::float16, mlx::core::float32}) {
                    const auto input = mlx::core::zeros(
                        Shape{rows, impl_->linears.front().input_size()}, dtype);
                    for (const auto& output : (*this)(input)) preparation->add(output);
                    if (impl_->linears.size() == 2 &&
                        impl_->linears[0].output_size() == impl_->linears[1].output_size())
                        preparation->add(swiglu(input));
                }
            }
        });
    }
}

std::vector<array> MlxProjectionBatch::operator()(
    const array& input) const {
    const std::size_t rows =
        input.ndim() == 0 || input.shape(-1) <= 0
        ? 0
        : input.size() /
            static_cast<std::size_t>(input.shape(-1));
    std::vector<array> outputs(impl_->linears.size(), input);
    for (const auto& segment : impl_->segments) {
        if (!segment.tq.empty()) {
            auto values = MlxTqWeight::matmul_many(segment.tq, input);
            for (std::size_t offset = 0; offset < segment.indices.size(); ++offset)
                outputs[segment.indices[offset]] = std::move(values[offset]);
            continue;
        }
        const bool use_grouped = segment.grouped &&
            segment.grouped->supports(input) &&
            (rows > 1 ||
             segment.grouped
                 ->supports_single_row_projection_fusion());
        if (use_grouped) {
            auto values = segment.grouped->matmul(input);
            for (std::size_t offset = 0; offset < segment.indices.size(); ++offset)
                outputs[segment.indices[offset]] = std::move(values[offset]);
            continue;
        }
        for (const auto index : segment.indices)
            outputs[index] = impl_->linears[index](input);
    }
    return outputs;
}

array MlxProjectionBatch::swiglu(
    const array& input,
    float limit) const {
    if (impl_->linears.size() != 2 ||
        impl_->linears[0].output_size() !=
            impl_->linears[1].output_size()) {
        throw std::logic_error(
            "SwiGLU projection batch requires two equal-width outputs");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "SwiGLU projection limit must be finite and non-negative");
    }
    if (const auto* gate = impl_->linears[0].tq_weight_ref()) {
        if (const auto* up = impl_->linears[1].tq_weight_ref()) {
            const int tokens = input.size() / gate->input_size();
            auto result = gate->swiglu(*up, input, mlx::core::zeros(Shape{tokens, 1}, mlx::core::int32), limit);
            auto shape = input.shape(); shape.back() = gate->output_size();
            return mlx::core::reshape(result, shape);
        }
    }
    if (impl_->segments.size() == 1 &&
        impl_->segments.front().indices.size() == 2 &&
        impl_->segments.front().grouped) {
        const auto& grouped = *impl_->segments.front().grouped;
        if (grouped.supports_single_row_swiglu(input)) {
            return grouped.single_row_swiglu(input, limit);
        }
        if (grouped.supports_small_m_swiglu(input)) {
            return grouped.small_m_swiglu(input, limit);
        }
    }
    auto projected = (*this)(input);
    if (limit > 0.0f) {
        return moe_limited_swiglu_pair(
            projected.at(0), projected.at(1), limit);
    }
    auto gate = std::move(projected.at(0));
    auto up = std::move(projected.at(1));
    if (up.dtype() != gate.dtype()) {
        up = mlx::core::astype(up, gate.dtype());
    }
    return gate * mlx::core::sigmoid(gate) * up;
}

bool MlxProjectionBatch::supports_fused_swiglu(
    const array& input) const noexcept {
    if (impl_->linears.size() == 2 && impl_->linears[0].tq_weight_ref()
        && impl_->linears[1].tq_weight_ref()) return true;
    if (impl_->linears.size() != 2 ||
        impl_->segments.size() != 1 ||
        impl_->segments.front().indices.size() != 2 ||
        !impl_->segments.front().grouped) {
        return false;
    }
    const auto& grouped = *impl_->segments.front().grouped;
    return grouped.supports_single_row_swiglu(input) ||
        grouped.supports_small_m_swiglu(input);
}

std::size_t MlxProjectionBatch::projection_count() const noexcept {
    return impl_->linears.size();
}

std::size_t
MlxProjectionBatch::grouped_projection_count() const noexcept {
    return impl_->grouped_projection_count;
}

bool MlxProjectionBatch::projections_share_group(
    std::size_t begin,
    std::size_t count) const noexcept {
    if (count < 2 || begin > impl_->linears.size() ||
        count > impl_->linears.size() - begin) {
        return false;
    }
    return std::any_of(
        impl_->segments.begin(),
        impl_->segments.end(),
        [begin, count](const Impl::Segment& segment) {
            if (!segment.grouped && segment.tq.empty()) return false;
            for (std::size_t index = begin; index < begin + count; ++index)
                if (std::find(segment.indices.begin(), segment.indices.end(), index)
                    == segment.indices.end()) return false;
            return true;
        });
}

MlxEmbedding MlxEmbedding::load(
    const MfqContainer& model,
    const std::string& name) {
    const auto finish = [](MlxEmbedding result) {
        if (mlx_predequantize_fp16_enabled()) {
            result.materialize_fp16();
        }
        if (auto* preparation = MlxKernelPreparation::current()) {
            preparation->collect([&] {
                for (const auto rows : preparation->row_buckets()) {
                    preparation->add(result(mlx::core::zeros(Shape{rows}, mlx::core::int32)));
                    const std::vector<std::int32_t> ids(rows, 0);
                    preparation->add(result(array(ids.begin(), Shape{rows})));
                }
            });
        }
        return result;
    };
    const auto& record = model.record(name);
    if (record.dtype == "TQ") return finish(MlxEmbedding(MlxTqWeight::load(model, name)));
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
    if (is_mxfp4_sq_dtype(record.dtype)) {
        throw std::runtime_error(
            "MXFP4-SQ tensors do not support embedding lookup: " + name);
    }
    if (is_fp8_sq_dtype(record.dtype)) {
        throw std::runtime_error(
            "FP8-SQ tensors do not support embedding lookup: " + name);
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

MlxEmbedding::MlxEmbedding(MlxTqWeight weight)
    : weight_(weight), vocabulary_size_(weight.output_size()), hidden_size_(weight.input_size()) {}

MlxEmbedding::MlxEmbedding(MlxNint8ZeroWeight weight)
    : vocabulary_size_(weight.output_size()),
      hidden_size_(weight.input_size()),
      weight_(std::move(weight)) {}

MlxEmbedding::MlxEmbedding(MlxVqWeight weight)
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
    if (std::holds_alternative<MlxTqWeight>(weight_)) return;
    if (const auto* dense = std::get_if<array>(&weight_);
        dense && (!g_fp16_weight_scopes || dense->dtype() == mlx::core::float16)) return;
    auto dense = materialize_weight_fp16(weight_, vocabulary_size_);
    weight_ = std::move(dense);
}

array MlxEmbedding::operator()(
    const array& token_ids,
    Dtype dtype) const {
    if (const auto* tq = std::get_if<MlxTqWeight>(&weight_)) return tq->embedding(token_ids, dtype);
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
    if (const auto* tq = std::get_if<MlxTqWeight>(&weight_)) return tq->matmul(input);
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
