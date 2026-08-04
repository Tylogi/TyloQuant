#include "mlx_deepseek_v4_attention.h"
#include "mlx_eval_timing.h"

#include "mlx_grouped_linear.h"
#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr const char* kHadamardSource = R"METAL(
    uint row = thread_position_in_grid.x / 256u;
    uint lane = thread_index_in_threadgroup;
    if (row >= uint(M)) {
        return;
    }
    threadgroup float values[BLOCK];
    for (
        uint local_block = 0u;
        local_block < uint(K) / uint(BLOCK);
        ++local_block
    ) {
        uint column_base = local_block * uint(BLOCK);
        for (
            uint index = lane;
            index < uint(BLOCK);
            index += 256u
        ) {
            values[index] = float(
                x[row * uint(K) + column_base + index]
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (
            uint stride = 1u;
            stride < uint(BLOCK);
            stride <<= 1u
        ) {
            for (
                uint pair = lane;
                pair < uint(BLOCK) / 2u;
                pair += 256u
            ) {
                uint pair_block = pair / stride;
                uint within = pair - pair_block * stride;
                uint first =
                    pair_block * (stride << 1u) + within;
                uint second = first + stride;
                float first_value = values[first];
                float second_value = values[second];
                values[first] = first_value + second_value;
                values[second] = first_value - second_value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inverse = rsqrt(float(BLOCK));
        for (
            uint index = lane;
            index < uint(BLOCK);
            index += 256u
        ) {
            y[
                row * uint(K) + column_base + index
            ] = T(values[index] * inverse);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
)METAL";

constexpr const char* kPartialAdjacentRopeSource = R"METAL(
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }
    uint column = index % uint(DIM);
    uint row = index / uint(DIM);
    constexpr uint PREFIX = uint(DIM - ROTARY);
    if (column < PREFIX) {
        y[index] = x[index];
        return;
    }

    uint rotary_column = column - PREFIX;
    uint pair = rotary_column >> 1u;
    uint token = (row / uint(HEADS)) % uint(TOKENS);
    float cosine = float(cos_values[token * uint(PAIRS) + pair]);
    float sine = float(sin_values[token * uint(PAIRS) + pair]);
    if (INVERSE != 0) {
        sine = -sine;
    }
    uint pair_base =
        row * uint(DIM) + PREFIX + (pair << 1u);
    float first = float(x[pair_base]);
    float second = float(x[pair_base + 1u]);
    float result = (rotary_column & 1u) == 0u
        ? first * cosine - second * sine
        : first * sine + second * cosine;
    y[index] = T(result);
)METAL";

constexpr const char* kRmsPartialAdjacentRopeSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint local_thread = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }

    threadgroup float reductions[8];
    float sum_squares = 0.0f;
    uint row_base = row * uint(DIM);
    for (uint column = local_thread;
         column < uint(DIM);
         column += 256u) {
        float value = float(x[row_base + column]);
        sum_squares += value * value;
    }
    float subtotal = simd_sum(sum_squares);
    if (lane == 0u) {
        reductions[simd_group] = subtotal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_thread == 0u) {
        float total = 0.0f;
        for (uint group = 0u; group < 8u; ++group) {
            total += reductions[group];
        }
        reductions[0] = rsqrt(total / float(DIM) + params[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse_rms = reductions[0];

    constexpr uint PREFIX = uint(DIM - ROTARY);
    uint token = (row / uint(HEADS)) % uint(TOKENS);
    for (uint column = local_thread;
         column < uint(DIM);
         column += 256u) {
        // Preserve the original FP16/BF16 boundary between RMSNorm and RoPE.
        float weight = WEIGHTED != 0
            ? float(weights[column])
            : 1.0f;
        T normalized = T(
            float(x[row_base + column]) * inverse_rms * weight);
        if (column < PREFIX) {
            y[row_base + column] = normalized;
            continue;
        }
        uint rotary_column = column - PREFIX;
        uint pair = rotary_column >> 1u;
        uint pair_base = row_base + PREFIX + (pair << 1u);
        float first_weight = WEIGHTED != 0
            ? float(weights[PREFIX + (pair << 1u)])
            : 1.0f;
        float second_weight = WEIGHTED != 0
            ? float(weights[PREFIX + (pair << 1u) + 1u])
            : 1.0f;
        float first = float(T(
            float(x[pair_base]) * inverse_rms * first_weight));
        float second = float(T(
            float(x[pair_base + 1u]) * inverse_rms * second_weight));
        float cosine =
            float(cos_values[token * uint(PAIRS) + pair]);
        float sine =
            float(sin_values[token * uint(PAIRS) + pair]);
        float result = (rotary_column & 1u) == 0u
            ? first * cosine - second * sine
            : first * sine + second * cosine;
        y[row_base + column] = T(result);
    }
)METAL";

const mlx::core::fast::CustomKernelFunction&
hadamard_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_dsv4_index_hadamard",
            {"x"},
            {"y"},
            kHadamardSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
partial_adjacent_rope_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_dsv4_partial_adjacent_rope",
            {"x", "cos_values", "sin_values"},
            {"y"},
            kPartialAdjacentRopeSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
rms_partial_adjacent_rope_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_dsv4_rms_partial_adjacent_rope",
            {"x", "cos_values", "sin_values", "weights", "params"},
            {"y"},
            kRmsPartialAdjacentRopeSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

int checked_int(
    std::int64_t value,
    const char* label) {
    if (value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string("invalid DeepSeek-V4 ") + label);
    }
    return static_cast<int>(value);
}

int checked_product(
    std::initializer_list<int> factors,
    const char* label) {
    std::int64_t result = 1;
    for (const int factor : factors) {
        if (factor < 0 ||
            (factor != 0 &&
             result >
                 std::numeric_limits<int>::max() /
                     factor)) {
            throw std::invalid_argument(
                std::string("DeepSeek-V4 ") + label +
                " exceeds MLX limits");
        }
        result *= factor;
    }
    return static_cast<int>(result);
}

array typed_contiguous(
    const array& input,
    Dtype dtype) {
    auto result = input;
    if (result.dtype() != dtype) {
        result = mlx::core::astype(result, dtype);
    }
    return mlx::core::contiguous(result);
}

array floating_contiguous(const array& input) {
    auto result = input;
    if (result.dtype() != mlx::core::float16 &&
        result.dtype() != mlx::core::float32) {
        result = mlx::core::astype(
            result,
            mlx::core::float16);
    }
    return mlx::core::contiguous(result);
}

array slice_axis(
    const array& input,
    int axis,
    int begin,
    int end) {
    if (axis < 0) {
        axis += static_cast<int>(input.ndim());
    }
    if (axis < 0 ||
        axis >= static_cast<int>(input.ndim()) ||
        begin < 0 || end < begin ||
        end > input.shape(axis)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 array slice");
    }
    Shape start(input.ndim(), 0);
    Shape stop = input.shape();
    start[axis] = begin;
    stop[axis] = end;
    return mlx::core::slice(
        input,
        std::move(start),
        std::move(stop));
}

array replace_last_rope(
    const array& input,
    int rotary,
    const array& cosine,
    const array& sine,
    bool inverse = false) {
    if (input.ndim() < 2 || input.ndim() > 4 ||
        input.shape(-1) <= 0 ||
        rotary <= 0 ||
        rotary > input.shape(-1) ||
        rotary % 2 != 0 ||
        cosine.shape() != sine.shape() ||
        cosine.shape(-1) != rotary / 2) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 partial adjacent RoPE input");
    }

    auto source = floating_contiguous(input);
    const int dimension = source.shape(-1);
    const int tokens = source.ndim() == 2
        ? source.shape(0)
        : source.shape(1);
    const int heads = source.ndim() == 4
        ? source.shape(2)
        : 1;
    const int pairs = rotary / 2;
    if (tokens <= 0 || heads <= 0 ||
        cosine.size() !=
            static_cast<std::size_t>(tokens) * pairs) {
        throw std::invalid_argument(
            "DeepSeek-V4 RoPE table is not token-broadcastable");
    }
    auto cos_values = typed_contiguous(
        cosine,
        mlx::core::float32);
    auto sin_values = typed_contiguous(
        sine,
        mlx::core::float32);
    const auto size = source.size();
    if (size >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "DeepSeek-V4 RoPE tensor exceeds Metal grid limits");
    }
    const int grid = static_cast<int>(size);
    auto outputs = partial_adjacent_rope_kernel()(
        {source, cos_values, sin_values},
        {source.shape()},
        {source.dtype()},
        {grid, 1, 1},
        {std::min(256, grid), 1, 1},
        {
            {"T", source.dtype()},
            {"SIZE", grid},
            {"DIM", dimension},
            {"ROTARY", rotary},
            {"PAIRS", pairs},
            {"TOKENS", tokens},
            {"HEADS", heads},
            {"INVERSE", inverse ? 1 : 0},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array rms_replace_last_rope(
    const array& input,
    int rotary,
    const array& cosine,
    const array& sine,
    const array& weights,
    bool weighted,
    const array& params) {
    if ((input.ndim() != 3 && input.ndim() != 4) ||
        input.shape(-1) <= 0 ||
        rotary <= 0 ||
        rotary > input.shape(-1) ||
        rotary % 2 != 0 ||
        cosine.shape() != sine.shape() ||
        cosine.shape(-1) != rotary / 2 ||
        params.dtype() != mlx::core::float32 ||
        params.size() != 1 ||
        (weighted &&
         (weights.dtype() != mlx::core::float32 ||
          weights.ndim() != 1 ||
          weights.size() !=
              static_cast<std::size_t>(input.shape(-1))))) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 fused RMS/RoPE input");
    }
    auto source = floating_contiguous(input);
    const int dimension = source.shape(-1);
    const int tokens = source.shape(1);
    const int heads = source.ndim() == 4
        ? source.shape(2)
        : 1;
    const int pairs = rotary / 2;
    if (tokens <= 0 || heads <= 0 ||
        cosine.size() !=
            static_cast<std::size_t>(tokens) * pairs) {
        throw std::invalid_argument(
            "DeepSeek-V4 fused RMS/RoPE table mismatch");
    }
    const auto rows = source.size() /
        static_cast<std::size_t>(dimension);
    if (rows == 0 ||
        rows > static_cast<std::size_t>(
            std::numeric_limits<int>::max() / 256)) {
        throw std::invalid_argument(
            "DeepSeek-V4 fused RMS/RoPE grid exceeds Metal limits");
    }
    auto outputs = rms_partial_adjacent_rope_kernel()(
        {
            source,
            typed_contiguous(cosine, mlx::core::float32),
            typed_contiguous(sine, mlx::core::float32),
            weights,
            params,
        },
        {source.shape()},
        {source.dtype()},
        {static_cast<int>(rows) * 256, 1, 1},
        {256, 1, 1},
        {
            {"T", source.dtype()},
            {"ROWS", static_cast<int>(rows)},
            {"DIM", dimension},
            {"ROTARY", rotary},
            {"PAIRS", pairs},
            {"TOKENS", tokens},
            {"HEADS", heads},
            {"WEIGHTED", weighted ? 1 : 0},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array load_float_array(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" &&
        record.dtype != "F16" &&
        record.dtype != "F32") {
        throw std::runtime_error(
            "DeepSeek-V4 attention tensor must be BF16/F16/F32: " +
            name);
    }
    const auto mapped = model.map_record(name);
    return typed_contiguous(
        load_dense_array(
            record.dtype,
            mapped.view()),
        mlx::core::float32);
}

array signed_hadamard(
    const array& input,
    int block) {
    auto source = floating_contiguous(input);
    if (source.ndim() == 0 ||
        block <= 0 ||
        (block & (block - 1)) != 0 ||
        block > 8192 ||
        source.shape(-1) % block != 0) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 Hadamard input");
    }
    Shape output_shape = source.shape();
    int rows = 1;
    for (std::size_t axis = 0;
         axis + 1 < source.ndim();
         ++axis) {
        rows = checked_product(
            {rows, source.shape(static_cast<int>(axis))},
            "Hadamard row count");
    }
    const int width = source.shape(-1);
    source = mlx::core::contiguous(
        mlx::core::reshape(
            source,
            Shape{rows, width}));
    const int grid = checked_product(
        {rows, 256},
        "Hadamard grid");
    auto outputs = hadamard_kernel()(
        {source},
        {Shape{rows, width}},
        {source.dtype()},
        {grid, 1, 1},
        {256, 1, 1},
        {
            {"T", source.dtype()},
            {"M", rows},
            {"K", width},
            {"BLOCK", block},
        },
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        std::move(outputs.front()),
        std::move(output_shape));
}

array pool_prefix(
    const MlxDeepseekV4PoolState& state) {
    return slice_axis(
        state.pool(),
        1,
        0,
        state.pool_len());
}

array gather_selected(
    const array& cache,
    const array& indices) {
    const int batch = cache.shape(0);
    const int tokens = indices.shape(1);
    const int selected = indices.shape(2);
    const int sequence = cache.shape(1);
    const int dimension = cache.shape(2);
    auto expanded_cache = mlx::core::broadcast_to(
        mlx::core::expand_dims(cache, 1),
        Shape{
            batch,
            tokens,
            sequence,
            dimension,
        });
    auto expanded_indices = mlx::core::broadcast_to(
        mlx::core::expand_dims(indices, -1),
        Shape{
            batch,
            tokens,
            selected,
            dimension,
        });
    return mlx::core::take_along_axis(
        expanded_cache,
        expanded_indices,
        2);
}

array generic_sparse_attention(
    const array& query,
    const array& cache,
    const array& indices,
    const array& mask,
    const array& sinks) {
    const int heads = query.shape(2);
    const int dimension = query.shape(3);
    auto selected = mlx::core::astype(
        gather_selected(cache, indices),
        mlx::core::float32);
    auto query_values = mlx::core::astype(
        query,
        mlx::core::float32);
    auto scores = mlx::core::sum(
        mlx::core::expand_dims(query_values, 3) *
            mlx::core::expand_dims(selected, 2),
        -1) /
        std::sqrt(static_cast<double>(dimension));
    scores = scores +
        mlx::core::expand_dims(
            mlx::core::astype(
                mask,
                mlx::core::float32),
            2);
    auto sink_values = mlx::core::reshape(
        typed_contiguous(
            sinks,
            mlx::core::float32),
        Shape{1, 1, heads});
    auto maximum = mlx::core::maximum(
        mlx::core::max(scores, -1),
        sink_values);
    auto exponentials = mlx::core::exp(
        scores -
        mlx::core::expand_dims(maximum, -1));
    auto denominator =
        mlx::core::sum(exponentials, -1) +
        mlx::core::exp(sink_values - maximum);
    auto probabilities =
        exponentials /
        mlx::core::expand_dims(denominator, -1);
    return mlx::core::sum(
        mlx::core::expand_dims(probabilities, -1) *
            mlx::core::expand_dims(selected, 2),
        3);
}

array generic_index_topk(
    const array& query,
    const array& keys,
    const array& weights,
    const array& positions,
    int ratio,
    int count) {
    const int batch = query.shape(0);
    const int tokens = query.shape(1);
    const int heads = query.shape(2);
    const int pool_len = keys.shape(1);
    auto query_values = mlx::core::astype(
        query,
        mlx::core::float32);
    auto key_values = mlx::core::astype(
        keys,
        mlx::core::float32);
    auto dots = mlx::core::sum(
        mlx::core::expand_dims(query_values, 3) *
            mlx::core::expand_dims(
                mlx::core::expand_dims(
                    key_values,
                    1),
                1),
        -1);
    auto positive = mlx::core::maximum(
        dots,
        array(0.0f, mlx::core::float32));
    auto score = mlx::core::sum(
        positive *
            mlx::core::expand_dims(
                mlx::core::astype(
                    weights,
                    mlx::core::float32),
                -1),
        2) /
        std::sqrt(
            static_cast<double>(
                query.shape(3) * heads));
    auto visible_count = mlx::core::floor_divide(
        positions + array(1, mlx::core::int32),
        array(ratio, mlx::core::int32));
    auto key_ids = mlx::core::reshape(
        mlx::core::arange(
            pool_len,
            mlx::core::int32),
        Shape{1, 1, pool_len});
    auto visible = mlx::core::less(
        key_ids,
        mlx::core::reshape(
            visible_count,
            Shape{1, tokens, 1}));
    score = mlx::core::where(
        mlx::core::broadcast_to(
            visible,
            Shape{batch, tokens, pool_len}),
        score,
        array(
            -std::numeric_limits<float>::infinity(),
            mlx::core::float32));
    auto partitioned = mlx::core::argpartition(
        score,
        pool_len - count,
        -1);
    return typed_contiguous(
        slice_axis(
            partitioned,
            -1,
            pool_len - count,
            pool_len),
        mlx::core::int32);
}

class ProjectionGroup {
public:
    explicit ProjectionGroup(
        std::vector<const MlxLinear*> projections)
        : projections_(std::move(projections)) {
        std::vector<MlxGroupedLinearWeightRef> refs;
        refs.reserve(projections_.size());
        for (const auto* projection : projections_) {
            const auto reference =
                projection->grouped_weight_ref();
            if (!reference.has_value()) {
                return;
            }
            refs.push_back(*reference);
        }
        if (refs.size() < 2) {
            return;
        }
        try {
            grouped_.emplace(std::move(refs));
        } catch (const MlxGroupedLinearUnsupported&) {
            grouped_.reset();
        }
    }

    std::vector<array> operator()(
        const array& input) const {
        const std::size_t rows =
            input.ndim() == 0 || input.shape(-1) <= 0
            ? 0
            : input.size() /
                static_cast<std::size_t>(input.shape(-1));
        if (grouped_.has_value() &&
            grouped_->supports(input) &&
            (rows > 1 ||
             grouped_->has_single_row_nint_fast_path() ||
             grouped_->has_single_row_mxfp8_fast_path())) {
            return (*grouped_)(input);
        }
        std::vector<array> result;
        result.reserve(projections_.size());
        for (const auto* projection : projections_) {
            result.push_back((*projection)(input));
        }
        return result;
    }

private:
    std::vector<const MlxLinear*> projections_;
    std::optional<MlxGroupedLinear> grouped_;
};

void require_linear(
    const MlxLinear& linear,
    int input,
    int output,
    const char* name) {
    if (linear.input_size() != input ||
        linear.output_size() != output) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 ") + name +
            " projection shape mismatch");
    }
}

void require_vector(
    const array& value,
    int size,
    const char* name) {
    if (value.ndim() != 1 ||
        value.size() !=
            static_cast<std::size_t>(size)) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 ") + name +
            " shape mismatch");
    }
}

void require_matrix(
    const array& value,
    int rows,
    int columns,
    const char* name) {
    if (value.shape() != Shape{rows, columns}) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 ") + name +
            " shape mismatch");
    }
}

} // namespace

std::pair<array, array> deepseek_v4_yarn_tables(
    int dimension,
    int length,
    float theta,
    const DeepseekV4RopeScaling& scaling) {
    if (dimension <= 0 ||
        dimension % 2 != 0 ||
        length <= 0 ||
        !std::isfinite(theta) ||
        theta <= 0.0f) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 Yarn table parameters");
    }
    const int pairs = dimension / 2;
    std::vector<float> frequency(pairs);
    for (int pair = 0; pair < pairs; ++pair) {
        frequency[pair] =
            1.0f /
            std::pow(
                theta,
                static_cast<float>(2 * pair) /
                    static_cast<float>(dimension));
    }
    if (scaling.enabled &&
        scaling.original_max_position_embeddings > 0) {
        if (!std::isfinite(scaling.factor) ||
            scaling.factor <= 0.0 ||
            !std::isfinite(scaling.beta_fast) ||
            scaling.beta_fast <= 0.0 ||
            !std::isfinite(scaling.beta_slow) ||
            scaling.beta_slow <= 0.0) {
            throw std::invalid_argument(
                "invalid DeepSeek-V4 Yarn scaling");
        }
        const auto correction =
            [&](double rotations) {
                return static_cast<double>(dimension) *
                    std::log(
                        static_cast<double>(
                            scaling
                                .original_max_position_embeddings) /
                        (rotations * 2.0 * std::acos(-1.0))) /
                    (2.0 *
                     std::log(
                         static_cast<double>(theta)));
            };
        const int low = std::max(
            static_cast<int>(
                std::floor(
                    correction(scaling.beta_fast))),
            0);
        double high = std::min(
            std::ceil(
                correction(scaling.beta_slow)),
            static_cast<double>(dimension - 1));
        if (static_cast<double>(low) == high) {
            high += 0.001;
        }
        for (int pair = 0; pair < pairs; ++pair) {
            const float ramp = std::clamp(
                static_cast<float>(
                    (static_cast<double>(pair) - low) /
                    (high - low)),
                0.0f,
                1.0f);
            const float smooth = 1.0f - ramp;
            frequency[pair] =
                frequency[pair] /
                    static_cast<float>(scaling.factor) *
                    (1.0f - smooth) +
                frequency[pair] * smooth;
        }
    }
    std::vector<float> cosine(
        static_cast<std::size_t>(length) * pairs);
    std::vector<float> sine(cosine.size());
    for (int position = 0;
         position < length;
         ++position) {
        for (int pair = 0; pair < pairs; ++pair) {
            const float angle =
                static_cast<float>(position) *
                frequency[pair];
            cosine[position * pairs + pair] =
                std::cos(angle);
            sine[position * pairs + pair] =
                std::sin(angle);
        }
    }
    return {
        array(
            cosine.begin(),
            Shape{length, pairs}),
        array(
            sine.begin(),
            Shape{length, pairs}),
    };
}

array deepseek_v4_rope_adjacent(
    const array& value,
    const array& cosine,
    const array& sine,
    bool inverse) {
    if (value.ndim() == 0 ||
        value.shape(-1) <= 0 ||
        value.shape(-1) % 2 != 0 ||
        cosine.shape() != sine.shape() ||
        cosine.size() == 0 ||
        cosine.shape(-1) != value.shape(-1) / 2) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 adjacent RoPE input");
    }
    return replace_last_rope(
        value,
        value.shape(-1),
        cosine,
        sine,
        inverse);
}

array deepseek_v4_unweighted_rms(
    const array& value,
    float eps) {
    if (value.ndim() == 0 ||
        value.shape(-1) <= 0 ||
        !std::isfinite(eps) ||
        eps <= 0.0f) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 unweighted RMS input");
    }
    auto source = mlx::core::astype(
        value,
        mlx::core::float32);
    auto inverse = mlx::core::rsqrt(
        mlx::core::mean(
            source * source,
            -1,
            true) +
        eps);
    auto result = source * inverse;
    return result.dtype() == value.dtype()
        ? result
        : mlx::core::astype(result, value.dtype());
}

MlxDeepseekV4PoolState::MlxDeepseekV4PoolState(
    int ratio,
    int head_dim,
    bool overlap,
    int batch,
    int capacity,
    Dtype dtype,
    array pool,
    array state_kv,
    array state_gate,
    std::optional<array> prev_kv,
    std::optional<array> prev_gate)
    : ratio_(ratio),
      head_dim_(head_dim),
      overlap_(overlap),
      batch_(batch),
      capacity_(capacity),
      dtype_(dtype),
      pool_(std::move(pool)),
      state_kv_(std::move(state_kv)),
      state_gate_(std::move(state_gate)),
      prev_kv_(std::move(prev_kv)),
      prev_gate_(std::move(prev_gate)) {}

MlxDeepseekV4PoolState
MlxDeepseekV4PoolState::allocate(
    int ratio,
    int head_dim,
    bool overlap,
    int batch,
    int max_context,
    Dtype dtype) {
    if (ratio <= 0 ||
        ratio > 128 ||
        (head_dim != 128 && head_dim != 512) ||
        batch <= 0 ||
        max_context <= 0 ||
        (dtype != mlx::core::float16 &&
         dtype != mlx::core::float32)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 pool allocation");
    }
    const int capacity =
        std::max(1, (max_context + ratio - 1) / ratio);
    const int output_dim =
        head_dim * (overlap ? 2 : 1);
    const Shape previous_shape{
        batch,
        ratio,
        head_dim,
    };
    std::optional<array> previous_kv;
    std::optional<array> previous_gate;
    if (overlap) {
        previous_kv = mlx::core::zeros(
            previous_shape,
            dtype);
        previous_gate = mlx::core::full(
            previous_shape,
            -std::numeric_limits<float>::infinity(),
            dtype);
    }
    return MlxDeepseekV4PoolState(
        ratio,
        head_dim,
        overlap,
        batch,
        capacity,
        dtype,
        mlx::core::zeros(
            Shape{batch, capacity, head_dim},
            dtype),
        mlx::core::zeros(
            Shape{batch, ratio, output_dim},
            dtype),
        mlx::core::full(
            Shape{batch, ratio, output_dim},
            -std::numeric_limits<float>::infinity(),
            dtype),
        std::move(previous_kv),
        std::move(previous_gate));
}

void MlxDeepseekV4PoolState::update(
    const array& kv_token,
    const array& gate_token,
    const array& ape,
    const array& norm,
    int length,
    const array& cosine,
    const array& sine,
    int quant_mode,
    float eps) {
    const int expected =
        pool_len_ * ratio_ + remainder_ + 1;
    if (length != expected ||
        length <= 0 ||
        length > capacity_ * ratio_) {
        throw std::invalid_argument(
            "DeepSeek-V4 compressor update is not contiguous "
            "or exceeds capacity");
    }
    const int required_rows =
        (length + ratio_ - 1) / ratio_;
    if (cosine.ndim() != 2 ||
        sine.shape() != cosine.shape() ||
        cosine.shape(0) < required_rows ||
        cosine.shape(1) < 32) {
        throw std::invalid_argument(
            "DeepSeek-V4 compressor rotary table is too short");
    }
    auto step = dsv4_decode_pool_step(
        kv_token,
        gate_token,
        ape,
        norm,
        state_kv_,
        state_gate_,
        prev_kv_,
        prev_gate_,
        mlx::core::full(
            Shape{batch_},
            length,
            mlx::core::int32),
        cosine,
        sine,
        ratio_,
        overlap_,
        quant_mode,
        eps);
    state_kv_ = std::move(step.state_kv);
    state_gate_ = std::move(step.state_gate);
    prev_kv_ = std::move(step.prev_kv);
    prev_gate_ = std::move(step.prev_gate);
    remainder_ = length % ratio_;
    if (remainder_ != 0) {
        return;
    }
    const int row = length / ratio_ - 1;
    auto row_indices = mlx::core::full(
        Shape{batch_, 1},
        row,
        mlx::core::int32);
    pool_ = dsv4_cache_write_inplace(
        pool_,
        step.emitted,
        row_indices);
    pool_len_ = std::max(pool_len_, row + 1);
}

void MlxDeepseekV4PoolState::prefill(
    const array& kv,
    const array& gate,
    const array& ape,
    const array& norm,
    int start_position,
    const array& cosine,
    const array& sine,
    int quant_mode,
    float eps) {
    auto values = typed_contiguous(kv, dtype_);
    auto gates = typed_contiguous(gate, dtype_);
    const int output_dim =
        head_dim_ * (overlap_ ? 2 : 1);
    const int tokens = values.ndim() == 3
        ? values.shape(1)
        : 0;
    const int expected =
        pool_len_ * ratio_ + remainder_;
    if (values.ndim() != 3 ||
        values.shape(0) != batch_ ||
        values.shape(2) != output_dim ||
        gates.shape() != values.shape() ||
        tokens <= 0 ||
        remainder_ != 0 ||
        start_position != expected ||
        start_position > capacity_ * ratio_ - tokens) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 compressor prefill input");
    }

    const int windows = tokens / ratio_;
    const int cutoff = windows * ratio_;
    array tail_kv = state_kv_;
    array tail_gate = state_gate_;
    if (windows > 0) {
        auto grouped_kv = mlx::core::contiguous(
            mlx::core::reshape(
                slice_axis(values, 1, 0, cutoff),
                Shape{
                    batch_,
                    windows,
                    ratio_,
                    output_dim,
                }));
        auto grouped_gate = mlx::core::contiguous(
            mlx::core::reshape(
                slice_axis(gates, 1, 0, cutoff),
                Shape{
                    batch_,
                    windows,
                    ratio_,
                    output_dim,
                }));
        auto positions = mlx::core::contiguous(
            mlx::core::broadcast_to(
                mlx::core::reshape(
                    mlx::core::arange(
                        pool_len_,
                        pool_len_ + windows,
                        1,
                        mlx::core::int32),
                    Shape{1, windows}),
                Shape{batch_, windows}));
        const bool has_previous =
            overlap_ && pool_len_ > 0;
        auto compressed = dsv4_compress(
            grouped_kv,
            grouped_gate,
            ape,
            norm,
            has_previous ? prev_kv_ : std::nullopt,
            has_previous ? prev_gate_ : std::nullopt,
            positions,
            cosine,
            sine,
            ratio_,
            overlap_,
            quant_mode,
            eps);
        pool_ = dsv4_cache_write_inplace(
            pool_,
            compressed,
            positions);

        tail_kv = mlx::core::contiguous(
            slice_axis(
                values,
                1,
                cutoff - ratio_,
                cutoff));
        tail_gate = mlx::core::contiguous(
            slice_axis(
                gates,
                1,
                cutoff - ratio_,
                cutoff));
        if (overlap_) {
            prev_kv_ = mlx::core::contiguous(
                slice_axis(
                    tail_kv,
                    2,
                    0,
                    head_dim_));
            prev_gate_ = mlx::core::contiguous(
                slice_axis(
                    tail_gate,
                    2,
                    0,
                    head_dim_));
        }
        pool_len_ += windows;
    }

    const int remainder = tokens - cutoff;
    if (remainder > 0) {
        tail_kv = mlx::core::concatenate(
            {
                slice_axis(values, 1, cutoff, tokens),
                slice_axis(tail_kv, 1, remainder, ratio_),
            },
            1);
        tail_gate = mlx::core::concatenate(
            {
                slice_axis(gates, 1, cutoff, tokens),
                slice_axis(tail_gate, 1, remainder, ratio_),
            },
            1);
    }
    state_kv_ = mlx::core::contiguous(tail_kv);
    state_gate_ = mlx::core::contiguous(tail_gate);
    remainder_ = remainder;
}

MlxDeepseekV4LayerState::MlxDeepseekV4LayerState(
    array local,
    std::optional<MlxDeepseekV4PoolState> main,
    std::optional<MlxDeepseekV4PoolState> indexer)
    : local_(std::move(local)),
      main_(std::move(main)),
      indexer_(std::move(indexer)) {}

array MlxDeepseekV4LayerState::local_positions() const {
    const int window = local_.shape(1);
    auto slots = mlx::core::arange(
        window,
        mlx::core::int32);
    array positions = mlx::core::full(
        Shape{window},
        -1,
        mlx::core::int32);
    if (position_ > 0 && position_ < window) {
        positions = mlx::core::where(
            mlx::core::less(
                slots,
                array(position_, mlx::core::int32)),
            slots,
            positions);
    } else if (position_ >= window) {
        const int last = position_ - 1;
        positions =
            array(last, mlx::core::int32) -
            mlx::core::remainder(
                array(last, mlx::core::int32) - slots,
                array(window, mlx::core::int32));
    }
    return mlx::core::broadcast_to(
        mlx::core::reshape(
            positions,
            Shape{1, window}),
        Shape{batch(), window});
}

MlxDeepseekV4LayerState
MlxDeepseekV4LayerState::allocate(
    const DeepseekV4Config& config,
    int ratio,
    int batch,
    int max_context,
    Dtype dtype) {
    if (ratio != 0 && ratio != 4 && ratio != 128) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 layer compression ratio");
    }
    const int window = checked_int(
        config.sliding_window,
        "sliding window");
    const int head_dim = checked_int(
        config.head_dim,
        "head dimension");
    if (batch <= 0 || max_context <= 0) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 layer cache allocation");
    }
    std::optional<MlxDeepseekV4PoolState> main;
    std::optional<MlxDeepseekV4PoolState> indexer;
    if (ratio != 0) {
        main = MlxDeepseekV4PoolState::allocate(
            ratio,
            head_dim,
            ratio == 4,
            batch,
            max_context,
            dtype);
    }
    if (ratio == 4) {
        indexer = MlxDeepseekV4PoolState::allocate(
            ratio,
            checked_int(
                config.index_head_dim,
                "indexer head dimension"),
            true,
            batch,
            max_context,
            dtype);
    }
    return MlxDeepseekV4LayerState(
        mlx::core::zeros(
            Shape{batch, window, head_dim},
            dtype),
        std::move(main),
        std::move(indexer));
}

struct MlxDeepseekV4Attention::Impl {
    DeepseekV4Config config;
    int layer;
    int ratio;
    int maximum_context;
    MlxDeepseekV4AttentionComponents components;
    std::pair<array, array> rope;
    std::optional<std::pair<array, array>> pool_rope;
    array rms_params;
    MlxRmsNorm q_norm;
    std::optional<ProjectionGroup> projections;

    Impl(
        DeepseekV4Config selected_config,
        int selected_layer,
        int selected_ratio,
        int max_context,
        MlxDeepseekV4AttentionComponents selected_components,
        std::pair<array, array> selected_rope)
        : config(std::move(selected_config)),
          layer(selected_layer),
          ratio(selected_ratio),
          maximum_context(max_context),
          components(std::move(selected_components)),
          rope{
              typed_contiguous(
                  selected_rope.first,
                  mlx::core::float32),
              typed_contiguous(
                  selected_rope.second,
                  mlx::core::float32),
          },
          rms_params(
              {static_cast<float>(config.rms_eps)},
              mlx::core::float32),
          q_norm(
              components.q_norm,
              static_cast<float>(config.rms_eps)) {
        validate();
        if (ratio != 0) {
            const Shape strides{ratio, 1};
            pool_rope.emplace(
                mlx::core::contiguous(
                    mlx::core::slice(
                        rope.first,
                        Shape{0, 0},
                        rope.first.shape(),
                        strides)),
                mlx::core::contiguous(
                    mlx::core::slice(
                        rope.second,
                        Shape{0, 0},
                        rope.second.shape(),
                        strides)));
        }
        projections.emplace(projection_list());
    }

    std::vector<const MlxLinear*> projection_list() {
        std::vector<const MlxLinear*> result{
            &components.q_a,
            &components.kv,
        };
        if (ratio != 0) {
            result.push_back(&*components.main_kv);
            result.push_back(&*components.main_gate);
        }
        if (ratio == 4) {
            result.push_back(&*components.index_kv);
            result.push_back(&*components.index_gate);
            result.push_back(&*components.index_weights);
        }
        return result;
    }

    void validate() {
        config.validate();
        const int hidden = checked_int(
            config.hidden,
            "hidden size");
        const int heads = checked_int(
            config.n_heads,
            "attention heads");
        const int head_dim = checked_int(
            config.head_dim,
            "attention head dimension");
        const int q_rank = checked_int(
            config.q_lora_rank,
            "query rank");
        const int groups = checked_int(
            config.o_groups,
            "output groups");
        const int o_rank = checked_int(
            config.o_lora_rank,
            "output rank");
        const int attention = checked_product(
            {heads, head_dim},
            "attention width");
        if (layer < 0 ||
            layer >= config.n_layers ||
            config.compress_ratios[layer] != ratio ||
            maximum_context <= 0 ||
            maximum_context >
                config.max_position_embeddings ||
            (ratio != 0 && ratio != 4 && ratio != 128)) {
            throw std::invalid_argument(
                "invalid DeepSeek-V4 attention layer schedule");
        }
        require_linear(
            components.q_a,
            hidden,
            q_rank,
            "q_a");
        require_linear(
            components.kv,
            hidden,
            head_dim,
            "kv");
        require_linear(
            components.q_b,
            q_rank,
            attention,
            "q_b");
        require_linear(
            components.wo_a,
            attention / groups,
            groups * o_rank,
            "wo_a");
        require_linear(
            components.wo_b,
            groups * o_rank,
            hidden,
            "wo_b");
        require_vector(
            components.q_norm,
            q_rank,
            "q_norm");
        require_vector(
            components.kv_norm,
            head_dim,
            "kv_norm");
        require_vector(
            components.sinks,
            heads,
            "attention sinks");
        const int rotary = checked_int(
            config.qk_rope_head_dim,
            "rotary dimension");
        if (rope.first.shape() != rope.second.shape() ||
            rope.first.ndim() != 2 ||
            rope.first.shape(0) < maximum_context ||
            rope.first.shape(1) != rotary / 2) {
            throw std::invalid_argument(
                "DeepSeek-V4 attention RoPE table mismatch");
        }
        if (ratio == 0) {
            if (components.main_kv ||
                components.main_gate ||
                components.main_ape ||
                components.main_norm ||
                components.index_q_b ||
                components.index_kv ||
                components.index_gate ||
                components.index_weights ||
                components.index_ape ||
                components.index_norm) {
                throw std::invalid_argument(
                    "ratio-zero DeepSeek-V4 layer received "
                    "compressor components");
            }
            return;
        }
        const int main_width =
            head_dim * (ratio == 4 ? 2 : 1);
        if (!components.main_kv ||
            !components.main_gate ||
            !components.main_ape ||
            !components.main_norm) {
            throw std::invalid_argument(
                "DeepSeek-V4 main compressor components missing");
        }
        require_linear(
            *components.main_kv,
            hidden,
            main_width,
            "main compressor KV");
        require_linear(
            *components.main_gate,
            hidden,
            main_width,
            "main compressor gate");
        require_matrix(
            *components.main_ape,
            ratio,
            main_width,
            "main compressor APE");
        require_vector(
            *components.main_norm,
            head_dim,
            "main compressor norm");
        if (ratio != 4) {
            if (components.index_q_b ||
                components.index_kv ||
                components.index_gate ||
                components.index_weights ||
                components.index_ape ||
                components.index_norm) {
                throw std::invalid_argument(
                    "ratio-128 DeepSeek-V4 layer received "
                    "Indexer components");
            }
            return;
        }
        const int index_heads = checked_int(
            config.index_n_heads,
            "index heads");
        const int index_dim = checked_int(
            config.index_head_dim,
            "index dimension");
        const int index_width = checked_product(
            {index_heads, index_dim},
            "index query width");
        if (!components.index_q_b ||
            !components.index_kv ||
            !components.index_gate ||
            !components.index_weights ||
            !components.index_ape ||
            !components.index_norm) {
            throw std::invalid_argument(
                "DeepSeek-V4 Indexer components missing");
        }
        require_linear(
            *components.index_q_b,
            q_rank,
            index_width,
            "index query");
        require_linear(
            *components.index_kv,
            hidden,
            2 * index_dim,
            "index compressor KV");
        require_linear(
            *components.index_gate,
            hidden,
            2 * index_dim,
            "index compressor gate");
        require_linear(
            *components.index_weights,
            hidden,
            index_heads,
            "index weights");
        require_matrix(
            *components.index_ape,
            ratio,
            2 * index_dim,
            "index compressor APE");
        require_vector(
            *components.index_norm,
            index_dim,
            "index compressor norm");
    }

    std::vector<array> project(
        const array& input) const {
        return (*projections)(input);
    }

    array index_query(
        const array& q_rank,
        const array& positions) const {
        const int batch = q_rank.shape(0);
        const int tokens = q_rank.shape(1);
        const int heads = checked_int(
            config.index_n_heads,
            "index heads");
        const int dimension = checked_int(
            config.index_head_dim,
            "index dimension");
        auto query = mlx::core::reshape(
            (*components.index_q_b)(q_rank),
            Shape{batch, tokens, heads, dimension});
        const int rotary = checked_int(
            config.qk_rope_head_dim,
            "rotary dimension");
        auto cosine = mlx::core::take(
            rope.first,
            positions,
            0);
        auto sine = mlx::core::take(
            rope.second,
            positions,
            0);
        cosine = mlx::core::expand_dims(
            mlx::core::expand_dims(cosine, 0),
            2);
        sine = mlx::core::expand_dims(
            mlx::core::expand_dims(sine, 0),
            2);
        query = replace_last_rope(
            query,
            rotary,
            cosine,
            sine);
        if (config.fast_indexer()) {
            query = signed_hadamard(
                query,
                dimension);
        }
        return query;
    }

    array topk(
        const array& q_rank,
        const array& index_weights,
        const MlxDeepseekV4LayerState& state,
        const array& positions,
        int pos0) const {
        const int batch = q_rank.shape(0);
        const int tokens = q_rank.shape(1);
        const int pool_len =
            state.main_.has_value()
            ? state.main_->pool_len()
            : 0;
        if (pool_len <= 0) {
            return mlx::core::zeros(
                Shape{batch, tokens, 0},
                mlx::core::int32);
        }
        const int requested = checked_int(
            config.index_topk,
            "index top-k");
        if (ratio != 4 || pool_len <= requested) {
            return mlx::core::broadcast_to(
                mlx::core::reshape(
                    mlx::core::arange(
                        pool_len,
                        mlx::core::int32),
                    Shape{1, 1, pool_len}),
                Shape{batch, tokens, pool_len});
        }
        auto query = index_query(
            q_rank,
            positions);
        if (config.fast_indexer() &&
            requested == 512) {
            if (tokens == 1) {
                return dsv4_topk512(
                    dsv4_indexer_scores_decode(
                        query,
                        state.indexer_->pool(),
                        index_weights,
                        pos0,
                        ratio,
                        pool_len),
                    true,
                    pool_len);
            }
            return dsv4_topk512(
                dsv4_indexer_scores(
                    query,
                    pool_prefix(*state.indexer_),
                    index_weights,
                    pos0,
                    ratio));
        }
        return generic_index_topk(
            query,
            pool_prefix(*state.indexer_),
            index_weights,
            positions,
            ratio,
            requested);
    }

    array output_projection(
        const array& value,
        const array& cosine,
        const array& sine) const {
        const int batch = value.shape(0);
        const int tokens = value.shape(1);
        const int groups = checked_int(
            config.o_groups,
            "output groups");
        const int rank = checked_int(
            config.o_lora_rank,
            "output rank");
        const int input_width =
            checked_product(
                {
                    checked_int(
                        config.n_heads,
                        "attention heads"),
                    checked_int(
                        config.head_dim,
                        "head dimension"),
                },
                "attention width") /
            groups;
        const int head_dim = checked_int(
            config.head_dim,
            "head dimension");
        const int rotary = checked_int(
            config.qk_rope_head_dim,
            "rotary dimension");
        auto grouped = mlx::core::reshape(
            value,
            Shape{
                batch,
                tokens,
                groups,
                input_width,
            });
        array low_rank =
            components.wo_a.nint8_zero_weight_ref()
            ? components.wo_a.nint8_zero_weight_ref()
                  ->grouped_row_matmul_inverse_rope(
                      grouped,
                      groups,
                      cosine,
                      sine,
                      head_dim,
                      rotary)
            : components.wo_a.mx_weight_ref() && tokens == 1
            ? components.wo_a.mx_weight_ref()
                  ->grouped_row_matmul_inverse_rope(
                      grouped,
                      groups,
                      cosine,
                      sine,
                      head_dim,
                      rotary)
            : components.wo_a.grouped_row_matmul(
                  mlx::core::reshape(
                      replace_last_rope(
                          value,
                          rotary,
                          cosine,
                          sine,
                          true),
                      Shape{
                          batch,
                          tokens,
                          groups,
                          input_width,
                      }),
                  groups);
        low_rank = mlx::core::reshape(
            std::move(low_rank),
            Shape{
                batch,
                tokens,
                groups * rank,
            });
        if (detail::component_profile_active()) {
            detail::profile_eval(
                std::string("attention.r")
                    + std::to_string(ratio)
                    + ".output_projection_a",
                low_rank);
        }
        auto result = components.wo_b(low_rank);
        if (detail::component_profile_active()) {
            detail::profile_eval(
                std::string("attention.r")
                    + std::to_string(ratio)
                    + ".output_projection_b",
                result);
        }
        return result;
    }
};

MlxDeepseekV4Attention MlxDeepseekV4Attention::load(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    int layer,
    int ratio,
    int max_context) {
    auto base = deepseek_v4_yarn_tables(
        checked_int(
            config.qk_rope_head_dim,
            "rotary dimension"),
        max_context,
        static_cast<float>(config.rope_theta),
        config.rope_scaling);
    auto compressed = deepseek_v4_yarn_tables(
        checked_int(
            config.qk_rope_head_dim,
            "rotary dimension"),
        max_context,
        static_cast<float>(
            config.compress_rope_theta),
        config.rope_scaling);
    return load(
        model,
        config,
        layer,
        ratio,
        max_context,
        std::move(base),
        std::move(compressed));
}

MlxDeepseekV4Attention MlxDeepseekV4Attention::load(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    int layer,
    int ratio,
    int max_context,
    std::pair<array, array> rope_base,
    std::pair<array, array> rope_compressed) {
    config.validate();
    if (layer < 0 ||
        layer >= config.n_layers ||
        static_cast<std::size_t>(layer) >=
            config.compress_ratios.size() ||
        config.compress_ratios[layer] != ratio ||
        max_context <= 0 ||
        max_context >
            config.max_position_embeddings) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 attention load schedule");
    }
    const auto name =
        [layer](std::string_view suffix) {
            return DeepseekV4TensorNames::layer(
                static_cast<std::size_t>(layer),
                suffix);
        };
    MlxDeepseekV4AttentionComponents components{
        MlxLinear::load(
            model,
            name("attn.wq_a.weight")),
        MlxLinear::load(
            model,
            name("attn.wkv.weight")),
        MlxLinear::load(
            model,
            name("attn.wq_b.weight")),
        MlxLinear::load(
            model,
            name("attn.wo_a.weight")),
        MlxLinear::load(
            model,
            name("attn.wo_b.weight")),
        load_float_array(
            model,
            name("attn.q_norm.weight")),
        load_float_array(
            model,
            name("attn.kv_norm.weight")),
        load_float_array(
            model,
            name("attn.attn_sink")),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
    };
    if (ratio != 0) {
        components.main_kv.emplace(
            MlxLinear::load(
                model,
                name("attn.compressor.wkv.weight")));
        components.main_gate.emplace(
            MlxLinear::load(
                model,
                name("attn.compressor.wgate.weight")));
        components.main_ape = load_float_array(
            model,
            name("attn.compressor.ape"));
        components.main_norm = load_float_array(
            model,
            name("attn.compressor.norm.weight"));
    }
    if (ratio == 4) {
        components.index_q_b.emplace(
            MlxLinear::load(
                model,
                name("attn.indexer.wq_b.weight")));
        components.index_kv.emplace(
            MlxLinear::load(
                model,
                name(
                    "attn.indexer.compressor.wkv.weight")));
        components.index_gate.emplace(
            MlxLinear::load(
                model,
                name(
                    "attn.indexer.compressor.wgate.weight")));
        components.index_weights.emplace(
            MlxLinear::load(
                model,
                name("attn.indexer.weights_proj.weight")));
        components.index_ape = load_float_array(
            model,
            name("attn.indexer.compressor.ape"));
        components.index_norm = load_float_array(
            model,
            name(
                "attn.indexer.compressor.norm.weight"));
    }
    return MlxDeepseekV4Attention(
        config,
        layer,
        ratio,
        max_context,
        std::move(components),
        std::move(rope_base),
        std::move(rope_compressed));
}

MlxDeepseekV4Attention::MlxDeepseekV4Attention(
    DeepseekV4Config config,
    int layer,
    int ratio,
    int max_context,
    MlxDeepseekV4AttentionComponents components,
    std::pair<array, array> rope_base,
    std::pair<array, array> rope_compressed)
    : impl_(std::make_shared<Impl>(
          std::move(config),
          layer,
          ratio,
          max_context,
          std::move(components),
          ratio == 0
              ? std::move(rope_base)
              : std::move(rope_compressed))) {}

array MlxDeepseekV4Attention::operator()(
    const array& input,
    MlxDeepseekV4LayerState& state,
    int pos0) const {
    auto source = floating_contiguous(input);
    const auto& config = impl_->config;
    const int hidden = checked_int(
        config.hidden,
        "hidden size");
    const int heads = checked_int(
        config.n_heads,
        "attention heads");
    const int head_dim = checked_int(
        config.head_dim,
        "head dimension");
    if (source.ndim() != 3 ||
        source.shape(0) <= 0 ||
        source.shape(1) <= 0 ||
        source.shape(2) != hidden ||
        state.batch() != source.shape(0) ||
        state.position_ != pos0 ||
        pos0 < 0 ||
        source.shape(1) >
            impl_->maximum_context - pos0 ||
        state.local_.shape() != Shape{
            source.shape(0),
            checked_int(
                config.sliding_window,
                "sliding window"),
            head_dim,
        } ||
        state.main_.has_value() !=
            (impl_->ratio != 0) ||
        state.indexer_.has_value() !=
            (impl_->ratio == 4)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 attention input/cache state");
    }
    const int batch = source.shape(0);
    const int tokens = source.shape(1);
    auto positions = mlx::core::arange(
        pos0,
        pos0 + tokens,
        1,
        mlx::core::int32);
    auto projected = impl_->project(source);
    const auto profile_component =
        [ratio = impl_->ratio](const char* component) {
            return std::string("attention.r")
                + std::to_string(ratio)
                + "." + component;
        };
    if (detail::component_profile_active()) {
        detail::profile_eval(
            profile_component("input_projections"),
            projected);
    }
    std::size_t output = 0;
    auto q_rank = impl_->q_norm(
        projected.at(output++));
    auto kv = projected.at(output++);
    if (detail::component_profile_active()) {
        detail::profile_eval(
            profile_component("q_rank_norm"),
            q_rank);
    }
    auto q_projected = mlx::core::reshape(
        impl_->components.q_b(q_rank),
        Shape{batch, tokens, heads, head_dim});
    if (detail::component_profile_active()) {
        detail::profile_eval(
            profile_component("q_b_projection"),
            q_projected);
    }
    const int rotary = checked_int(
        config.qk_rope_head_dim,
        "rotary dimension");
    auto cosine = mlx::core::take(
        impl_->rope.first,
        positions,
        0);
    auto sine = mlx::core::take(
        impl_->rope.second,
        positions,
        0);
    auto q_cosine = mlx::core::expand_dims(
        mlx::core::expand_dims(cosine, 0),
        2);
    auto q_sine = mlx::core::expand_dims(
        mlx::core::expand_dims(sine, 0),
        2);
    auto q = rms_replace_last_rope(
        q_projected,
        rotary,
        q_cosine,
        q_sine,
        impl_->rms_params,
        false,
        impl_->rms_params);
    kv = rms_replace_last_rope(
        kv,
        rotary,
        mlx::core::expand_dims(cosine, 0),
        mlx::core::expand_dims(sine, 0),
        impl_->components.kv_norm,
        true,
        impl_->rms_params);
    if (detail::component_profile_active()) {
        detail::profile_eval(
            profile_component("q_rms_rope"),
            q);
        detail::profile_eval(
            profile_component("kv_norm_rope"),
            kv);
    }

    array index_weights = mlx::core::zeros(
        Shape{batch, tokens, 0},
        source.dtype());
    if (impl_->ratio != 0) {
        auto main_kv = projected.at(output++);
        auto main_gate = projected.at(output++);
        array index_kv = mlx::core::zeros(
            Shape{1},
            source.dtype());
        array index_gate = mlx::core::zeros(
            Shape{1},
            source.dtype());
        if (impl_->ratio == 4) {
            index_kv = projected.at(output++);
            index_gate = projected.at(output++);
            index_weights = projected.at(output++);
        }
        const bool batch_prefill =
            tokens > 1 &&
            state.main_->remainder() == 0 &&
            (!state.indexer_ ||
             state.indexer_->remainder() == 0);
        if (batch_prefill) {
            state.main_->prefill(
                main_kv,
                main_gate,
                *impl_->components.main_ape,
                *impl_->components.main_norm,
                pos0,
                impl_->pool_rope->first,
                impl_->pool_rope->second,
                0,
                static_cast<float>(config.rms_eps));
            if (impl_->ratio == 4) {
                state.indexer_->prefill(
                    index_kv,
                    index_gate,
                    *impl_->components.index_ape,
                    *impl_->components.index_norm,
                    pos0,
                    impl_->pool_rope->first,
                    impl_->pool_rope->second,
                    config.fast_indexer() ? 2 : 0,
                    static_cast<float>(config.rms_eps));
            }
        } else {
            for (int token = 0; token < tokens; ++token) {
                const int length = pos0 + token + 1;
                state.main_->update(
                    slice_axis(
                        main_kv,
                        1,
                        token,
                        token + 1),
                    slice_axis(
                        main_gate,
                        1,
                        token,
                        token + 1),
                    *impl_->components.main_ape,
                    *impl_->components.main_norm,
                    length,
                    impl_->pool_rope->first,
                    impl_->pool_rope->second,
                    0,
                    static_cast<float>(config.rms_eps));
                if (impl_->ratio == 4) {
                    state.indexer_->update(
                        slice_axis(
                            index_kv,
                            1,
                            token,
                            token + 1),
                        slice_axis(
                            index_gate,
                            1,
                            token,
                            token + 1),
                        *impl_->components.index_ape,
                        *impl_->components.index_norm,
                        length,
                        impl_->pool_rope->first,
                        impl_->pool_rope->second,
                        config.fast_indexer() ? 2 : 0,
                        static_cast<float>(config.rms_eps));
                }
            }
        }
        if (state.indexer_ &&
            state.indexer_->pool_len() !=
                state.main_->pool_len()) {
            throw std::runtime_error(
                "DeepSeek-V4 main and Indexer "
                "pool lengths diverged");
        }
    }
    if (output != projected.size()) {
        throw std::runtime_error(
            "DeepSeek-V4 projection group output mismatch");
    }

    if (
        detail::component_profile_active()
        && impl_->ratio != 0
    ) {
        std::vector<array> compressor_state;
        const auto append_pool =
            [&compressor_state](
                const MlxDeepseekV4PoolState& pool) {
                compressor_state.push_back(pool.pool());
                compressor_state.push_back(pool.state_kv());
                compressor_state.push_back(pool.state_gate());
                if (pool.prev_kv()) {
                    compressor_state.push_back(*pool.prev_kv());
                }
                if (pool.prev_gate()) {
                    compressor_state.push_back(*pool.prev_gate());
                }
            };
        append_pool(*state.main_);
        if (state.indexer_) {
            append_pool(*state.indexer_);
        }
        detail::profile_eval(
            profile_component("compressor_update"),
            std::move(compressor_state));
    }

    auto topk = impl_->topk(
        q_rank,
        index_weights,
        state,
        positions,
        pos0);
    if (detail::component_profile_active()) {
        detail::profile_eval(
            profile_component("indexer_topk"),
            topk);
    }
    const int window = checked_int(
        config.sliding_window,
        "sliding window");
    const int pool_len =
        state.main_.has_value()
        ? state.main_->pool_len()
        : 0;
    array unified = state.local_;
    std::optional<std::pair<array, array>> plan;
    std::optional<array> direct_decode;
    if (tokens == 1) {
        const int slot = pos0 % window;
        auto local_index = mlx::core::full(
            Shape{batch, 1},
            slot,
            mlx::core::int32);
        state.local_ = dsv4_cache_write_inplace(
            state.local_,
            mlx::core::astype(
                kv,
                state.local_.dtype()),
            local_index);
        unified = state.local_;
        if (config.fast_attention()) {
            std::optional<array> pooled;
            if (state.main_.has_value()) {
                pooled = state.main_->pool();
            }
            direct_decode = attention_dsv4_sparse_decode(
                mlx::core::transpose(
                    q,
                    {0, 2, 1, 3}),
                state.local_,
                pooled,
                pool_len,
                topk,
                impl_->components.sinks,
                pos0 + 1,
                impl_->ratio == 0
                    ? 1
                    : impl_->ratio,
                window);
        } else {
            if (state.main_.has_value()) {
                unified = mlx::core::concatenate(
                    {
                        state.local_,
                        pool_prefix(*state.main_),
                    },
                    1);
            }
            plan = dsv4_build_decode_plan(
                topk,
                mlx::core::full(
                    Shape{batch},
                    pos0 + 1,
                    mlx::core::int32),
                pool_len,
                impl_->ratio == 0
                    ? 1
                    : impl_->ratio,
                window);
        }
    } else {
        const int history = std::min(pos0, window);
        array history_values = slice_axis(
            state.local_,
            1,
            0,
            0);
        if (history != 0) {
            auto history_positions = mlx::core::arange(
                pos0 - history,
                pos0,
                1,
                mlx::core::int32);
            auto history_slots =
                mlx::core::remainder(
                    history_positions,
                    array(window, mlx::core::int32));
            history_values = mlx::core::take(
                state.local_,
                history_slots,
                1);
        }
        std::vector<array> parts{
            history_values,
            mlx::core::astype(
                kv,
                state.local_.dtype()),
        };
        if (state.main_.has_value()) {
            parts.push_back(
                pool_prefix(*state.main_));
        }
        unified = mlx::core::concatenate(
            std::move(parts),
            1);
        plan = dsv4_build_prefill_plan(
            topk,
            pos0,
            history,
            pool_len,
            impl_->ratio == 0
                ? 1
                : impl_->ratio,
            window);
        const int recent = std::min(tokens, window);
        auto recent_values = slice_axis(
            kv,
            1,
            tokens - recent,
            tokens);
        auto recent_positions = mlx::core::arange(
            pos0 + tokens - recent,
            pos0 + tokens,
            1,
            mlx::core::int32);
        auto recent_slots = mlx::core::remainder(
            recent_positions,
            array(window, mlx::core::int32));
        auto local_indices = mlx::core::broadcast_to(
            mlx::core::reshape(
                recent_slots,
                Shape{1, recent}),
            Shape{batch, recent});
        state.local_ = dsv4_cache_write_inplace(
            state.local_,
            mlx::core::astype(
                recent_values,
                state.local_.dtype()),
            local_indices);
    }

    array attended = direct_decode
        ? *direct_decode
        : config.fast_attention()
            ? attention_dsv4_sparse(
                  mlx::core::transpose(
                      q,
                      {0, 2, 1, 3}),
                  unified,
                  plan->first,
                  plan->second,
                  impl_->components.sinks)
            : generic_sparse_attention(
                  q,
                  unified,
                  plan->first,
                  plan->second,
                  impl_->components.sinks);
    if (detail::component_profile_active()) {
        if (direct_decode) {
            detail::profile_eval(
                profile_component("cache_update"),
                std::vector<array>{state.local_});
        } else {
            detail::profile_eval(
                profile_component("cache_plan"),
                std::vector<array>{
                    state.local_,
                    unified,
                    plan->first,
                    plan->second,
                });
        }
        detail::profile_eval(
            profile_component("sparse_core"),
            attended);
    }
    auto result = impl_->output_projection(
        attended,
        cosine,
        sine);
    state.position_ += tokens;
    return result;
}

int MlxDeepseekV4Attention::ratio() const noexcept {
    return impl_->ratio;
}

int MlxDeepseekV4Attention::layer() const noexcept {
    return impl_->layer;
}

int MlxDeepseekV4Attention::max_context() const noexcept {
    return impl_->maximum_context;
}

} // namespace mfq::metal
