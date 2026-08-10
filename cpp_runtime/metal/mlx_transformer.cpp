#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;
using mlx::core::CompileOptions;
using mlx::core::MathMode;

constexpr const char* kMropeSource = R"METAL(
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }

    uint column = index % uint(DIM);
    if (column >= uint(ROTARY_DIM)) {
        y[index] = x[index];
        return;
    }

    constexpr uint HALF = uint(ROTARY_DIM / 2);
    uint row = index / uint(DIM);
    uint token = row % uint(TOKENS);
    uint batch = row / uint(HEADS * TOKENS);
    uint pair = column < HALF ? column : column - HALF;

    uint axis = 0u;
    if (S0 + S1 + S2 > 0 && POS_AXES == 3) {
        if (INTERLEAVED != 0) {
            uint residue = pair % 3u;
            if (residue == 1u && pair < uint(S1 * 3)) {
                axis = 1u;
            } else if (residue == 2u && pair < uint(S2 * 3)) {
                axis = 2u;
            }
        } else {
            axis = pair < uint(S0)
                ? 0u
                : (pair < uint(S0 + S1) ? 1u : 2u);
        }
    }

    int position = POSITION_BATCH != 0
        ? positions[batch * uint(TOKENS) + token]
        : positions[axis * uint(TOKENS) + token];
    float exponent =
        -2.0f * float(pair) / float(ROTARY_DIM);
    float angle =
        float(position) * pow(params[0], exponent);
    float cosine = cos(angle);
    float sine = sin(angle);

    uint row_offset = row * uint(DIM);
    float first = float(x[row_offset + pair]);
    float second = float(x[row_offset + pair + HALF]);
    float value = column < HALF
        ? first * cosine - second * sine
        : second * cosine + first * sine;
    y[index] = T(value);
)METAL";

mlx::core::fast::CustomKernelFunction make_mrope_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_mrope",
        {"x", "positions", "params"},
        {"y"},
        kMropeSource,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& mrope_kernel() {
    static const auto kernel = make_mrope_kernel();
    return kernel;
}

array floating(const array& input, mlx::core::Dtype dtype) {
    auto result = input;
    if (result.dtype() != dtype) {
        result = mlx::core::astype(result, dtype);
    }
    return mlx::core::contiguous(result);
}

void require_attention_shape(const array& value, const char* name) {
    if (value.ndim() != 4) {
        throw std::runtime_error(
            std::string(name) + " must have [batch,heads,tokens,head_dim] shape");
    }
}

} // namespace

MlxRmsNorm::MlxRmsNorm(
    array weight,
    float eps,
    float weight_offset)
    : weight_(mlx::core::contiguous(
          mlx::core::astype(std::move(weight), mlx::core::float32)) +
          weight_offset),
      eps_(eps),
      width_(weight_.ndim() == 1 ? weight_.shape(0) : 0) {
    if (width_ <= 0) {
        throw std::runtime_error(
            "RMSNorm weight must be a non-empty vector");
    }
    if (!std::isfinite(eps_) || eps_ <= 0.0f ||
        !std::isfinite(weight_offset)) {
        throw std::runtime_error("invalid RMSNorm parameters");
    }
}

array MlxRmsNorm::operator()(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != width_) {
        throw std::runtime_error("RMSNorm input width mismatch");
    }
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::bfloat16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    auto normalized = mlx::core::fast::rms_norm(
        source,
        std::optional<array>(weight_),
        eps_);
    // MLX promotes an FP16 activation to FP32 when the RMSNorm weight is
    // FP32. Keep the stored/offset weight in FP32 for accuracy, but preserve
    // the model activation dtype at the operator boundary.
    return normalized.dtype() == source.dtype()
        ? normalized
        : mlx::core::astype(normalized, source.dtype());
}

array apply_rope(
    const array& input,
    int rotary_dimension,
    float base,
    int offset) {
    if (input.ndim() < 2 ||
        rotary_dimension <= 0 ||
        rotary_dimension > input.shape(-1) ||
        rotary_dimension % 2 != 0 ||
        !std::isfinite(base) ||
        base <= 0.0f ||
        offset < 0) {
        throw std::runtime_error("invalid RoPE input or parameters");
    }
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::bfloat16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    return mlx::core::fast::rope(
        source,
        rotary_dimension,
        false,
        base,
        1.0f,
        offset);
}

array apply_rope(
    const array& input,
    const array& positions,
    int rotary_dimension,
    float base,
    const std::vector<std::int64_t>& sections,
    bool interleaved) {
    if (input.ndim() < 2 ||
        input.shape(-2) <= 0 ||
        input.shape(-1) <= 0 ||
        rotary_dimension <= 0 ||
        rotary_dimension > input.shape(-1) ||
        rotary_dimension % 2 != 0 ||
        !std::isfinite(base) ||
        base <= 0.0f) {
        throw std::runtime_error(
            "invalid explicit-position RoPE input or parameters");
    }

    const int tokens = input.shape(-2);
    int position_axes = 0;
    bool position_batch = false;
    if (positions.ndim() == 1) {
        position_axes = 1;
    } else if (positions.ndim() == 2 &&
               positions.shape(0) == input.shape(0) &&
               sections.empty()) {
        position_axes = 1;
        position_batch = positions.shape(0) > 1;
    } else if (positions.ndim() == 2 &&
               (positions.shape(0) == 1 ||
                positions.shape(0) == 3)) {
        position_axes = positions.shape(0);
    } else {
        throw std::runtime_error(
            "RoPE positions must have [tokens], [1,tokens], "
            "[batch,tokens], or [3,tokens] shape");
    }
    if (positions.shape(-1) != tokens) {
        throw std::runtime_error(
            "RoPE position length must match the input token count");
    }

    int section_values[3] = {0, 0, 0};
    if (!sections.empty()) {
        if (sections.size() != 3) {
            throw std::runtime_error(
                "MRoPE sections must contain exactly three entries");
        }
        std::int64_t total = 0;
        for (std::size_t index = 0; index < sections.size(); ++index) {
            const auto section = sections[index];
            if (section < 0 ||
                section > std::numeric_limits<int>::max()) {
                throw std::runtime_error(
                    "MRoPE sections must be nonnegative int values");
            }
            section_values[index] = static_cast<int>(section);
            total += section;
        }
        if (total != rotary_dimension / 2) {
            throw std::runtime_error(
                "MRoPE sections must sum to rotary_dim / 2");
        }
    } else if (interleaved) {
        throw std::runtime_error(
            "interleaved MRoPE requires three section sizes");
    }

    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::bfloat16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::contiguous(source);
    auto position_values = positions;
    if (position_values.dtype() != mlx::core::int32) {
        position_values =
            mlx::core::astype(position_values, mlx::core::int32);
    }
    position_values = mlx::core::contiguous(position_values);

    if (source.size() >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "explicit-position RoPE Metal grid exceeds MLX limits");
    }
    const int size = static_cast<int>(source.size());
    const array parameters({base}, Shape{1});
    std::vector<std::pair<
        std::string,
        mlx::core::fast::TemplateArg>> templates{
        {"T", source.dtype()},
        {"SIZE", size},
        {"TOKENS", tokens},
        {"DIM", source.shape(-1)},
        {"HEADS", source.shape(-3)},
        {"ROTARY_DIM", rotary_dimension},
        {"POS_AXES", position_axes},
        {"POSITION_BATCH", static_cast<int>(position_batch)},
        {"S0", section_values[0]},
        {"S1", section_values[1]},
        {"S2", section_values[2]},
        {"INTERLEAVED", static_cast<int>(interleaved)},
    };
    auto outputs = mrope_kernel()(
        {source, position_values, parameters},
        {source.shape()},
        {source.dtype()},
        {size, 1, 1},
        {std::min(256, size), 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array scaled_dot_product_attention(
    const array& query,
    const array& key,
    const array& value,
    bool causal,
    float scale,
    const std::optional<array>& mask) {
    require_attention_shape(query, "attention query");
    require_attention_shape(key, "attention key");
    require_attention_shape(value, "attention value");
    if (key.shape() != value.shape() ||
        query.shape(0) != key.shape(0) ||
        query.shape(3) != key.shape(3) ||
        key.shape(1) <= 0 ||
        query.shape(1) % key.shape(1) != 0) {
        throw std::runtime_error("incompatible attention dimensions");
    }
    auto dtype = query.dtype();
    if (dtype != mlx::core::float16 &&
        dtype != mlx::core::bfloat16 &&
        dtype != mlx::core::float32) {
        dtype = mlx::core::float16;
    }
    auto q = floating(query, dtype);
    auto k = floating(key, dtype);
    auto v = floating(value, dtype);
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(q.shape(3)));
    }
    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::runtime_error("attention scale must be positive");
    }
    return mlx::core::fast::scaled_dot_product_attention(
        q,
        k,
        v,
        scale,
        causal ? "causal" : "",
        mask);
}

MlxKvCache::MlxKvCache(
    int batch,
    int heads,
    int maximum_sequence,
    int head_dimension,
    int initial_capacity,
    mlx::core::Dtype dtype)
    : batch_(batch),
      heads_(heads),
      maximum_sequence_(maximum_sequence),
      head_dimension_(head_dimension),
      dtype_(dtype),
      key_(mlx::core::zeros(
          Shape{
              batch,
              heads,
              std::min(std::max(initial_capacity, 0), maximum_sequence),
              head_dimension,
          },
          dtype)),
      value_(mlx::core::zeros(key_.shape(), dtype)) {
    if (batch_ <= 0 || heads_ <= 0 ||
        maximum_sequence_ <= 0 || head_dimension_ <= 0 ||
        initial_capacity < 0 ||
        (dtype_ != mlx::core::float16 &&
         dtype_ != mlx::core::bfloat16 &&
         dtype_ != mlx::core::float32)) {
        throw std::runtime_error("invalid KV cache dimensions or dtype");
    }
}

void MlxKvCache::materialize() {
    mlx::core::eval(key_, value_);
}

void MlxKvCache::ensure_capacity(int required) {
    if (required <= capacity()) {
        return;
    }
    if (required > maximum_sequence_) {
        throw std::runtime_error("KV cache exceeds maximum sequence length");
    }
    int next_capacity = std::max(1, capacity());
    while (next_capacity < required) {
        if (next_capacity > maximum_sequence_ / 2) {
            next_capacity = maximum_sequence_;
            break;
        }
        next_capacity *= 2;
    }
    const Shape next_shape{
        batch_,
        heads_,
        next_capacity,
        head_dimension_,
    };
    auto next_key = mlx::core::zeros(next_shape, dtype_);
    auto next_value = mlx::core::zeros(next_shape, dtype_);
    if (capacity() > 0) {
        const Shape start{0, 0, 0, 0};
        const Shape stop{
            batch_,
            heads_,
            capacity(),
            head_dimension_,
        };
        next_key = mlx::core::slice_update(
            next_key,
            key_,
            start,
            stop);
        next_value = mlx::core::slice_update(
            next_value,
            value_,
            start,
            stop);
    }
    key_ = std::move(next_key);
    value_ = std::move(next_value);
}

std::pair<array, array> MlxKvCache::append(
    const array& key,
    const array& value) {
    require_attention_shape(key, "KV key");
    require_attention_shape(value, "KV value");
    if (key.shape() != value.shape() ||
        key.shape(0) != batch_ ||
        key.shape(1) != heads_ ||
        key.shape(3) != head_dimension_) {
        throw std::runtime_error("KV append shape mismatch");
    }
    const int tokens = key.shape(2);
    if (tokens < 0 || position_ > maximum_sequence_ - tokens) {
        throw std::runtime_error("KV append exceeds maximum sequence length");
    }
    ensure_capacity(position_ + tokens);
    if (tokens > 0) {
        const Shape start{0, 0, position_, 0};
        const Shape stop{
            batch_,
            heads_,
            position_ + tokens,
            head_dimension_,
        };
        key_ = mlx::core::slice_update(
            key_,
            floating(key, dtype_),
            start,
            stop);
        value_ = mlx::core::slice_update(
            value_,
            floating(value, dtype_),
            start,
            stop);
        position_ += tokens;
    }
    return view();
}

std::pair<array, array> MlxKvCache::view() const {
    const Shape start{0, 0, 0, 0};
    const Shape stop{
        batch_,
        heads_,
        position_,
        head_dimension_,
    };
    return {
        mlx::core::slice(key_, start, stop),
        mlx::core::slice(value_, start, stop),
    };
}

} // namespace mfq::metal
