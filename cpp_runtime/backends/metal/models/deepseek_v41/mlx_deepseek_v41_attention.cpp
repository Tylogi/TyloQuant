#include "mlx_deepseek_v41_attention.h"

#include "mlx_deepseek_v4_attention.h"
#include "mlx_deepseek_v4_sparse.h"
#include "mlx_detached_copy.h"
#include "mlx_sparse_attention.h"
#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr const char* kMxfpSimHeader = R"METAL(
    METAL_FUNC float mfq_v41_e4m3(float value) {
        float magnitude = min(abs(value), 448.0f);
        float quantized;
        if (magnitude < 0x1p-6f) {
            quantized = rint(magnitude * 512.0f) / 512.0f;
        } else {
            float exponent = floor(log2(magnitude));
            float step = exp2(exponent - 3.0f);
            quantized = min(rint(magnitude / step) * step, 448.0f);
        }
        return copysign(quantized, value);
    }
    METAL_FUNC float mfq_v41_pow2_ceil(float value) {
        uint bits = as_type<uint>(value);
        uint exponent = (bits >> 23u) & 0xffu;
        bool has_mantissa = (bits & 0x7fffffu) != 0u;
        return as_type<float>((exponent + uint(has_mantissa)) << 23u);
    }
    METAL_FUNC float mfq_v41_e2m1(float value, float scale) {
        float normalized = clamp(value / scale, -6.0f, 6.0f);
        float magnitude = abs(normalized);
        float quantized;
        if (magnitude <= 0.25f) quantized = 0.0f;
        else if (magnitude < 0.75f) quantized = 0.5f;
        else if (magnitude <= 1.25f) quantized = 1.0f;
        else if (magnitude < 1.75f) quantized = 1.5f;
        else if (magnitude <= 2.5f) quantized = 2.0f;
        else if (magnitude < 3.5f) quantized = 3.0f;
        else if (magnitude <= 5.0f) quantized = 4.0f;
        else quantized = 6.0f;
        return copysign(quantized * scale, normalized);
    }
)METAL";

constexpr const char* kMxfp8SimSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint group = threadgroup_position_in_grid.x;
    uint offset = group * 32u + lane;
    float value = float(x[offset]);
    float maximum = simd_max(abs(value));
    float scale = mfq_v41_pow2_ceil(
        max(maximum, 448.0f * 0x1p-126f) / 448.0f);
    out[offset] = half(mfq_v41_e4m3(value / scale) * scale);
)METAL";

constexpr const char* kMxfp4E4m3ScaleSimSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint group = threadgroup_position_in_grid.x;
    uint offset = group * 16u + min(lane, 15u);
    float value = lane < 16u ? float(x[offset]) : 0.0f;
    float maximum = simd_max(abs(value));
    float raw_scale = max(maximum, 6.0f * 0x1p-9f) / 6.0f;
    float scale = max(mfq_v41_e4m3(raw_scale), 0x1p-9f);
    if (lane < 16u) {
        out[offset] = half(mfq_v41_e2m1(value, scale));
    }
)METAL";

const mlx::core::fast::CustomKernelFunction& mxfp8_sim_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_deepseek_v41_mxfp8_sim",
            {"x"}, {"out"}, kMxfp8SimSource, kMxfpSimHeader,
            true, false, options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& mxfp4_e4m3_sim_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_deepseek_v41_mxfp4_e4m3_scale_sim",
            {"x"}, {"out"}, kMxfp4E4m3ScaleSimSource, kMxfpSimHeader,
            true, false, options);
    }();
    return kernel;
}

int checked_int(std::int64_t value, const char* label) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string("invalid DeepSeek-V4.1 ") + label);
    }
    return static_cast<int>(value);
}

array typed_contiguous(const array& input, mlx::core::Dtype dtype) {
    auto value = input.dtype() == dtype ? input : mlx::core::astype(input, dtype);
    return mlx::core::contiguous(value);
}

array dense_array(const MfqContainer& model, const std::string& name) {
    const auto mapped = model.map_record(name);
    return load_dense_array(model.record(name).dtype, mapped.view());
}

array slice_axis(const array& input, int axis, int begin, int end) {
    if (axis < 0) axis += static_cast<int>(input.ndim());
    if (axis < 0 || axis >= static_cast<int>(input.ndim()) ||
        begin < 0 || end < begin || end > input.shape(axis)) {
        throw std::invalid_argument("invalid DeepSeek-V4.1 tensor slice");
    }
    Shape start(input.ndim(), 0);
    Shape stop = input.shape();
    start[axis] = begin;
    stop[axis] = end;
    return mlx::core::slice(input, std::move(start), std::move(stop));
}

array pool_prefix(const array& cache, int length) {
    return slice_axis(cache, 1, 0, length);
}

array weighted_rms(const array& input, const array& weight, float eps) {
    auto source = mlx::core::astype(input, mlx::core::float32);
    auto scale = mlx::core::rsqrt(
        mlx::core::mean(source * source, -1, true) + eps);
    auto result = source * scale * mlx::core::astype(weight, mlx::core::float32);
    return mlx::core::astype(result, input.dtype());
}

array rope_tail(
    const array& input,
    int rotary,
    const array& cosine,
    const array& sine,
    bool inverse = false) {
    const int width = input.shape(-1);
    if (rotary == width) {
        return deepseek_v4_rope_adjacent(input, cosine, sine, inverse);
    }
    return mlx::core::concatenate(
        {
            slice_axis(input, -1, 0, width - rotary),
            deepseek_v4_rope_adjacent(
                slice_axis(input, -1, width - rotary, width),
                cosine,
                sine,
                inverse),
        },
        -1);
}

array positions(int begin, int end) {
    return mlx::core::arange(begin, end, 1, mlx::core::int32);
}

array repeated_rows(int batch, int begin, int count) {
    return mlx::core::broadcast_to(
        mlx::core::reshape(
            positions(begin, begin + count), Shape{1, count}),
        Shape{batch, count});
}

array empty_topk(int batch, int tokens) {
    return mlx::core::zeros(Shape{batch, tokens, 0}, mlx::core::int32);
}

array normalize_topk(
    const array& selected,
    const array& valid,
    int pool_length) {
    auto sentinel = array(pool_length, mlx::core::int32);
    auto ordered = mlx::core::sort(
        mlx::core::where(valid, selected, sentinel), -1);
    return mlx::core::where(
        mlx::core::less(ordered, sentinel),
        ordered,
        array(-1, mlx::core::int32));
}

array topk_from_scores(
    const array& scores,
    int requested,
    int pool_length,
    const std::optional<array>& source_indices = std::nullopt) {
    const int width = scores.shape(-1);
    const int count = std::min(requested, width);
    if (count <= 0) {
        return empty_topk(scores.shape(0), scores.shape(1));
    }
    auto partition = mlx::core::argpartition(scores, width - count, -1);
    auto relative = slice_axis(partition, -1, width - count, width);
    auto values = mlx::core::take_along_axis(scores, relative, -1);
    auto selected = source_indices
        ? mlx::core::take_along_axis(*source_indices, relative, -1)
        : relative;
    auto valid = mlx::core::logical_and(
        mlx::core::greater(
            values,
            array(-std::numeric_limits<float>::infinity(), values.dtype())),
        mlx::core::logical_and(
            mlx::core::greater_equal(selected, array(0, mlx::core::int32)),
            mlx::core::less(selected, array(pool_length, mlx::core::int32))));
    return normalize_topk(selected, valid, pool_length);
}

array all_visible_indices(
    int batch,
    int tokens,
    int pool_length,
    int ratio,
    int pos0) {
    auto ids = mlx::core::broadcast_to(
        mlx::core::reshape(
            positions(0, pool_length), Shape{1, 1, pool_length}),
        Shape{batch, tokens, pool_length});
    auto query_positions = mlx::core::reshape(
        positions(pos0, pos0 + tokens), Shape{1, tokens, 1});
    auto visible = mlx::core::floor_divide(
        query_positions + array(1, mlx::core::int32),
        array(ratio, mlx::core::int32));
    return mlx::core::where(
        mlx::core::less(ids, visible), ids, array(-1, mlx::core::int32));
}

array candidate_blocks_from_scores(
    const array& scores,
    int ratio,
    int pos0,
    int requested_blocks,
    int block_size) {
    const int batch = scores.shape(0);
    const int tokens = scores.shape(1);
    const int width = scores.shape(2);
    const int blocks = (width + block_size - 1) / block_size;
    const int padded = blocks * block_size;
    auto padded_scores = scores;
    if (padded != width) {
        padded_scores = mlx::core::concatenate(
            {
                scores,
                mlx::core::full(
                    Shape{batch, tokens, padded - width},
                    -std::numeric_limits<float>::infinity(),
                    scores.dtype()),
            },
            -1);
    }
    auto block_scores = mlx::core::max(
        mlx::core::reshape(
            padded_scores, Shape{batch, tokens, blocks, block_size}),
        -1);
    auto query_positions = mlx::core::reshape(
        positions(pos0, pos0 + tokens), Shape{1, tokens, 1});
    auto visible = mlx::core::floor_divide(
        query_positions + array(1, mlx::core::int32),
        array(ratio, mlx::core::int32));
    auto last = mlx::core::floor_divide(
        visible - array(1, mlx::core::int32),
        array(block_size, mlx::core::int32));
    auto block_ids = mlx::core::reshape(
        positions(0, blocks), Shape{1, 1, blocks});
    block_scores = mlx::core::where(
        mlx::core::equal(block_ids, last),
        array(std::numeric_limits<float>::infinity(), block_scores.dtype()),
        block_scores);
    const int count = std::min(requested_blocks, blocks);
    auto partition = mlx::core::argpartition(block_scores, blocks - count, -1);
    auto selected = slice_axis(partition, -1, blocks - count, blocks);
    auto values = mlx::core::take_along_axis(block_scores, selected, -1);
    return mlx::core::where(
        mlx::core::greater(
            values,
            array(-std::numeric_limits<float>::infinity(), values.dtype())),
        selected,
        array(-1, mlx::core::int32));
}

array candidate_positions(
    const array& blocks,
    int block_size,
    int pool_length) {
    const int count = blocks.shape(-1);
    auto within = mlx::core::reshape(
        positions(0, block_size), Shape{1, 1, 1, block_size});
    auto expanded = mlx::core::expand_dims(blocks, -1) * block_size + within;
    expanded = mlx::core::reshape(
        expanded,
        Shape{blocks.shape(0), blocks.shape(1), count * block_size});
    return mlx::core::where(
        mlx::core::logical_and(
            mlx::core::greater_equal(expanded, array(0, mlx::core::int32)),
            mlx::core::less(expanded, array(pool_length, mlx::core::int32))),
        expanded,
        array(-1, mlx::core::int32));
}

array index_scores(
    const array& query,
    const array& keys,
    const array& weights,
    int ratio,
    int pos0) {
    const int batch = query.shape(0);
    const int tokens = query.shape(1);
    const int heads = query.shape(2);
    const int pool_length = keys.shape(1);
    auto q = mlx::core::astype(query, mlx::core::float32);
    auto k = mlx::core::astype(keys, mlx::core::float32);
    auto dots = mlx::core::sum(
        mlx::core::expand_dims(q, 3) *
            mlx::core::expand_dims(
                mlx::core::expand_dims(k, 1), 1),
        -1);
    auto positive = mlx::core::maximum(
        dots, array(0.0f, mlx::core::float32));
    auto score = mlx::core::sum(
        positive * mlx::core::expand_dims(
            mlx::core::astype(weights, mlx::core::float32), -1),
        2) /
        std::sqrt(static_cast<double>(query.shape(3) * heads));
    auto visible = mlx::core::floor_divide(
        mlx::core::reshape(
            positions(pos0, pos0 + tokens), Shape{1, tokens, 1}) +
            array(1, mlx::core::int32),
        array(ratio, mlx::core::int32));
    auto ids = mlx::core::reshape(
        positions(0, pool_length), Shape{1, 1, pool_length});
    return mlx::core::where(
        mlx::core::broadcast_to(
            mlx::core::less(ids, visible),
            Shape{batch, tokens, pool_length}),
        score,
        array(-std::numeric_limits<float>::infinity(), mlx::core::float32));
}

array filter_candidate_scores(
    const array& scores,
    const array& blocks,
    int block_size,
    int pool_length) {
    auto candidates = candidate_positions(blocks, block_size, pool_length);
    auto safe = mlx::core::maximum(candidates, array(0, mlx::core::int32));
    auto selected = mlx::core::take_along_axis(scores, safe, -1);
    selected = mlx::core::where(
        mlx::core::greater_equal(candidates, array(0, mlx::core::int32)),
        selected,
        array(-std::numeric_limits<float>::infinity(), scores.dtype()));
    return topk_from_scores(selected, 512, pool_length, candidates);
}

array apply_attention(
    const array& query,
    const array& local,
    const std::optional<array>& pooled,
    int pool_length,
    const array& topk,
    const array& sinks,
    int pos0,
    int ratio,
    int window) {
    const int batch = query.shape(0);
    const int tokens = query.shape(1);
    auto transposed = mlx::core::transpose(query, {0, 2, 1, 3});
    if (tokens == 1) {
        if (query.shape(2) == 64 && query.shape(3) == 512) {
            return mlx_sparse_circular_mla_decode_attention(
                transposed,
                local,
                pooled,
                pool_length,
                topk,
                sinks,
                pos0 + 1,
                ratio == 0 ? 1 : ratio,
                window);
        }
        const int local_length = std::min(pos0 + 1, window);
        auto local_positions = positions(
            pos0 + 1 - local_length, pos0 + 1);
        auto local_slots = mlx::core::remainder(
            local_positions, array(window, mlx::core::int32));
        auto chronological = mlx::core::take(local, local_slots, 1);
        auto plan = dsv4_build_prefill_plan(
            topk,
            pos0,
            local_length - 1,
            pool_length,
            ratio == 0 ? 1 : ratio,
            window);
        std::vector<array> parts{chronological};
        if (pooled && pool_length > 0) {
            parts.push_back(pool_prefix(*pooled, pool_length));
        }
        auto unified = parts.size() == 1
            ? parts.front()
            : mlx::core::concatenate(std::move(parts), 1);
        return mlx_sparse_selected_mla_attention(
            transposed, unified, plan.first, plan.second, sinks);
    }
    if (pooled && pool_length > 0 && topk.shape(2) > 0 &&
        query.shape(2) == 64 && query.shape(3) == 512) {
        return mlx_sparse_circular_mla_attention(
            transposed,
            local,
            *pooled,
            pool_length,
            topk,
            sinks,
            pos0,
            ratio,
            window);
    }
    auto plan = dsv4_build_prefill_plan(
        topk,
        pos0,
        std::max(0, local.shape(1) - tokens),
        pool_length,
        ratio == 0 ? 1 : ratio,
        window);
    std::vector<array> parts{local};
    if (pooled && pool_length > 0) {
        parts.push_back(pool_prefix(*pooled, pool_length));
    }
    auto unified = parts.size() == 1
        ? parts.front()
        : mlx::core::concatenate(std::move(parts), 1);
    return mlx_sparse_selected_mla_attention(
        transposed, unified, plan.first, plan.second, sinks);
}

} // namespace

struct MlxDeepseekV41AttentionSpeculation {
    int confirmed_tokens = 0;
    int total_tokens = 0;
    int start_position = 0;
    int compressed_length = 0;
    int partial_length = 0;
    array local_backup = array(0.0f);
    std::optional<array> partial_kv_backup;
    std::optional<array> partial_score_backup;
    std::optional<array> local_kv;
    std::optional<array> projected_kv;
    std::optional<array> projected_score;
};

array deepseek_v41_mxfp8_e4m3_sim(const array& input) {
    auto source = typed_contiguous(input, mlx::core::float16);
    if (source.size() == 0 || source.shape(-1) % 32 != 0 ||
        source.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 MXFP8 simulation expects nonempty [...,32*n]");
    }
    const int size = static_cast<int>(source.size());
    auto output = mxfp8_sim_kernel()(
        {source}, {source.shape()}, {mlx::core::float16},
        {size, 1, 1}, {32, 1, 1}, {}, std::nullopt, false, {});
    return std::move(output.front());
}

array deepseek_v41_mxfp4_e4m3_scale_sim(const array& input) {
    auto source = typed_contiguous(input, mlx::core::float16);
    if (source.size() == 0 || source.shape(-1) % 16 != 0 ||
        source.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 MXFP4 simulation expects nonempty [...,16*n]");
    }
    const int groups = static_cast<int>(source.size() / 16);
    auto output = mxfp4_e4m3_sim_kernel()(
        {source}, {source.shape()}, {mlx::core::float16},
        {groups * 32, 1, 1}, {32, 1, 1}, {}, std::nullopt, false, {});
    return std::move(output.front());
}

void MlxDeepseekV41SharedAttentionState::reset() noexcept {
    compressed_kv.reset();
    index_k.reset();
    topk.reset();
    candidate_blocks.reset();
    compressed_length = 0;
    ratio = 0;
}

MlxDeepseekV41AttentionState MlxDeepseekV41AttentionState::allocate(
    const DeepseekV41Config& config,
    int layer,
    int batch,
    int max_context,
    mlx::core::Dtype dtype) {
    config.validate();
    if (layer < 0 || layer >= config.n_layers || batch <= 0 ||
        max_context <= 0 || max_context > config.max_position_embeddings) {
        throw std::invalid_argument("invalid DeepSeek-V4.1 attention state");
    }
    const int window = checked_int(config.sliding_window, "sliding window");
    const int head_dim = checked_int(config.head_dim, "head dimension");
    MlxDeepseekV41AttentionState state{
        mlx::core::zeros(Shape{batch, window, head_dim}, dtype)};
    if (config.is_kv_source(layer)) {
        const int ratio = static_cast<int>(config.compress_ratios[layer]);
        const int capacity = std::max(1, max_context / ratio);
        state.compressed_kv = mlx::core::zeros(
            Shape{batch, capacity, head_dim}, dtype);
        state.index_k = mlx::core::zeros(
            Shape{batch, capacity, checked_int(config.index_head_dim, "index dimension")},
            dtype);
        if (ratio > 1) {
            state.partial_kv = mlx::core::zeros(
                Shape{batch, ratio, head_dim}, mlx::core::float32);
            state.partial_score = mlx::core::full(
                Shape{batch, ratio, head_dim},
                -std::numeric_limits<float>::infinity(),
                mlx::core::float32);
        }
    }
    return state;
}

MlxDeepseekV41Attention MlxDeepseekV41Attention::load(
    const MfqContainer& model,
    const DeepseekV41Config& config,
    int layer,
    int max_context,
    std::pair<array, array> rope_base,
    std::pair<array, array> rope_compressed) {
    const auto name = [layer](std::string_view suffix) {
        return DeepseekV41TensorNames::layer(
            static_cast<std::size_t>(layer), suffix);
    };
    const bool kv_source = config.is_kv_source(layer);
    const bool index_source = config.is_index_source(layer);
    MlxDeepseekV41AttentionComponents components{
        MlxLinear::load(model, name("attention.query_a.weight")),
        MlxLinear::load(model, name("attention.query_b.weight")),
        MlxLinear::load(model, name("attention.key_value.weight")),
        MlxLinear::load(model, name("attention.output_a.weight")),
        MlxLinear::load(model, name("attention.output_b.weight")),
        dense_array(model, name("attention.query_a_norm.weight")),
        dense_array(model, name("attention.key_value_norm.weight")),
        dense_array(model, name("attention.sink")),
        kv_source
            ? std::optional<MlxLinear>(MlxLinear::load(
                  model, name("attention.compressor.key_value.weight")))
            : std::nullopt,
        kv_source && config.compress_ratios[layer] > 1
            ? std::optional<MlxLinear>(MlxLinear::load(
                  model, name("attention.compressor.gate.weight")))
            : std::nullopt,
        kv_source
            ? std::optional<array>(dense_array(
                  model, name("attention.compressor.norm.weight")))
            : std::nullopt,
        index_source
            ? std::optional<MlxLinear>(MlxLinear::load(
                  model, name("attention.indexer.query.weight")))
            : std::nullopt,
        kv_source
            ? std::optional<MlxLinear>(MlxLinear::load(
                  model, name("attention.indexer.key.weight")))
            : std::nullopt,
        kv_source
            ? std::optional<array>(dense_array(
                  model, name("attention.indexer.key_norm.weight")))
            : std::nullopt,
        index_source
            ? std::optional<MlxLinear>(MlxLinear::load(
                  model, name("attention.indexer.score.weight")))
            : std::nullopt,
    };
    return MlxDeepseekV41Attention(
        config,
        layer,
        max_context,
        std::move(components),
        std::move(rope_base),
        std::move(rope_compressed));
}

MlxDeepseekV41Attention::MlxDeepseekV41Attention(
    DeepseekV41Config config,
    int layer,
    int max_context,
    MlxDeepseekV41AttentionComponents components,
    std::pair<array, array> rope_base,
    std::pair<array, array> rope_compressed)
    : config_(std::move(config)),
      layer_(layer),
      ratio_(static_cast<int>(config_.compress_ratios.at(layer))),
      max_context_(max_context),
      components_(std::move(components)),
      rope_(ratio_ == 0 ? std::move(rope_base) : std::move(rope_compressed)) {
    config_.validate();
    const int hidden = checked_int(config_.hidden, "hidden size");
    const int head_dim = checked_int(config_.head_dim, "head dimension");
    const int heads = checked_int(config_.n_heads, "head count");
    const int q_rank = checked_int(config_.q_lora_rank, "query rank");
    const int groups = checked_int(config_.o_groups, "output groups");
    const int o_rank = checked_int(config_.o_lora_rank, "output rank");
    if (layer_ < 0 || layer_ >= config_.n_layers || max_context_ <= 0 ||
        max_context_ > config_.max_position_embeddings ||
        components_.query_a.input_size() != hidden ||
        components_.query_a.output_size() != q_rank ||
        components_.query_b.input_size() != q_rank ||
        components_.query_b.output_size() != heads * head_dim ||
        components_.key_value.input_size() != hidden ||
        components_.key_value.output_size() != head_dim ||
        components_.output_a.input_size() != heads * head_dim / groups ||
        components_.output_a.output_size() != groups * o_rank ||
        components_.output_b.input_size() != groups * o_rank ||
        components_.output_b.output_size() != hidden ||
        components_.query_a_norm.size() != static_cast<std::size_t>(q_rank) ||
        components_.key_value_norm.size() != static_cast<std::size_t>(head_dim) ||
        components_.sinks.size() != static_cast<std::size_t>(heads)) {
        throw std::runtime_error("DeepSeek-V4.1 attention geometry disagrees");
    }
    const bool kv_source = config_.is_kv_source(layer_);
    const bool index_source = config_.is_index_source(layer_);
    if (kv_source != components_.compressor_key_value.has_value() ||
        kv_source != components_.compressor_norm.has_value() ||
        (kv_source && ratio_ > 1) != components_.compressor_gate.has_value() ||
        index_source != components_.index_query.has_value() ||
        index_source != components_.index_score.has_value() ||
        kv_source != components_.index_key.has_value() ||
        kv_source != components_.index_key_norm.has_value()) {
        throw std::runtime_error("DeepSeek-V4.1 CSA2 components disagree with schedule");
    }
}

std::vector<array> MlxDeepseekV41Attention::begin_speculative(
    MlxDeepseekV41AttentionState& state,
    int confirmed_tokens,
    int total_tokens) const {
    const int batch = state.local_kv.shape(0);
    const int window = state.local_kv.shape(1);
    if (state.speculation || confirmed_tokens <= 0 ||
        total_tokens <= confirmed_tokens || total_tokens > window ||
        state.position < 0 || state.position + total_tokens > max_context_) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 speculative cache transaction");
    }
    auto slots = mlx::core::remainder(
        positions(state.position, state.position + total_tokens),
        array(window, mlx::core::int32));
    auto local_backup = detached_copy(
        mlx::core::take(state.local_kv, slots, 1));
    std::optional<array> partial_kv_backup;
    std::optional<array> partial_score_backup;
    if (state.partial_kv) {
        partial_kv_backup = detached_copy(*state.partial_kv);
    }
    if (state.partial_score) {
        partial_score_backup = detached_copy(*state.partial_score);
    }
    state.speculation = std::make_shared<
        MlxDeepseekV41AttentionSpeculation>(
            MlxDeepseekV41AttentionSpeculation{
                confirmed_tokens,
                total_tokens,
                state.position,
                state.compressed_length,
                state.partial_length,
                std::move(local_backup),
                std::move(partial_kv_backup),
                std::move(partial_score_backup),
            });
    std::vector<array> checkpoints{state.speculation->local_backup};
    if (state.speculation->partial_kv_backup) {
        checkpoints.push_back(*state.speculation->partial_kv_backup);
    }
    if (state.speculation->partial_score_backup) {
        checkpoints.push_back(*state.speculation->partial_score_backup);
    }
    return checkpoints;
}

void MlxDeepseekV41Attention::commit_speculative(
    MlxDeepseekV41AttentionState& state) const noexcept {
    state.speculation.reset();
}

std::vector<array> MlxDeepseekV41Attention::rollback_speculative(
    MlxDeepseekV41AttentionState& state,
    int accepted_drafts) const {
    auto transaction = std::move(state.speculation);
    state.speculation.reset();
    if (!transaction || !transaction->local_kv) {
        throw std::runtime_error(
            "DeepSeek-V4.1 speculative cache capture is unavailable");
    }
    const int speculative_tokens =
        transaction->total_tokens - transaction->confirmed_tokens;
    if (accepted_drafts < 0 || accepted_drafts > speculative_tokens) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 speculative acceptance count is invalid");
    }
    const int keep = transaction->confirmed_tokens + accepted_drafts;
    const int batch = state.local_kv.shape(0);
    const int window = state.local_kv.shape(1);
    auto all_slots = mlx::core::remainder(
        positions(
            transaction->start_position,
            transaction->start_position + transaction->total_tokens),
        array(window, mlx::core::int32));
    auto all_rows = mlx::core::broadcast_to(
        mlx::core::reshape(
            all_slots, Shape{1, transaction->total_tokens}),
        Shape{batch, transaction->total_tokens});
    state.local_kv = dsv4_cache_write_inplace(
        state.local_kv, transaction->local_backup, all_rows);
    if (keep > 0) {
        auto kept_slots = slice_axis(all_slots, 0, 0, keep);
        auto kept_rows = mlx::core::broadcast_to(
            mlx::core::reshape(kept_slots, Shape{1, keep}),
            Shape{batch, keep});
        state.local_kv = dsv4_cache_write_inplace(
            state.local_kv,
            slice_axis(*transaction->local_kv, 1, 0, keep),
            kept_rows);
    }

    state.compressed_length = transaction->compressed_length;
    state.partial_length = transaction->partial_length;
    const bool kv_source = config_.is_kv_source(layer_);
    if (kv_source && ratio_ > 0) {
        state.compressed_length +=
            (transaction->partial_length + keep) / ratio_;
    }
    if (kv_source && ratio_ > 1) {
        if (!state.partial_kv || !state.partial_score ||
            !transaction->partial_kv_backup ||
            !transaction->partial_score_backup ||
            !transaction->projected_kv ||
            !transaction->projected_score) {
            throw std::runtime_error(
                "DeepSeek-V4.1 speculative compressor capture is unavailable");
        }
        *state.partial_kv = std::move(*transaction->partial_kv_backup);
        *state.partial_score = std::move(*transaction->partial_score_backup);
        for (int token = 0; token < keep; ++token) {
            const int slot =
                (transaction->start_position + token) % ratio_;
            auto row = mlx::core::full(
                Shape{batch, 1}, slot, mlx::core::int32);
            *state.partial_kv = dsv4_cache_write_inplace(
                *state.partial_kv,
                slice_axis(
                    *transaction->projected_kv, 1, token, token + 1),
                row);
            *state.partial_score = dsv4_cache_write_inplace(
                *state.partial_score,
                slice_axis(
                    *transaction->projected_score, 1, token, token + 1),
                row);
        }
        state.partial_length =
            (transaction->partial_length + keep) % ratio_;
    }
    state.position = transaction->start_position + keep;

    std::vector<array> restored{state.local_kv};
    if (state.partial_kv) restored.push_back(*state.partial_kv);
    if (state.partial_score) restored.push_back(*state.partial_score);
    return restored;
}

array MlxDeepseekV41Attention::forward(
    const array& input,
    MlxDeepseekV41AttentionState& state,
    MlxDeepseekV41SharedAttentionState& shared,
    int pos0) const {
    const int hidden = checked_int(config_.hidden, "hidden size");
    const int heads = checked_int(config_.n_heads, "head count");
    const int head_dim = checked_int(config_.head_dim, "head dimension");
    const int rotary = checked_int(config_.rope_head_dim, "rotary dimension");
    const int window = checked_int(config_.sliding_window, "sliding window");
    if (input.ndim() != 3 || input.shape(0) <= 0 || input.shape(1) <= 0 ||
        input.shape(2) != hidden || state.position != pos0 ||
        pos0 < 0 || pos0 + input.shape(1) > max_context_ ||
        state.local_kv.shape() != Shape{input.shape(0), window, head_dim}) {
        throw std::invalid_argument("DeepSeek-V4.1 attention input/cache mismatch");
    }
    const int batch = input.shape(0);
    const int tokens = input.shape(1);
    const float eps = static_cast<float>(config_.rms_eps);
    auto cosine = slice_axis(rope_.first, 0, pos0, pos0 + tokens);
    auto sine = slice_axis(rope_.second, 0, pos0, pos0 + tokens);

    auto q_rank = weighted_rms(
        components_.query_a(input), components_.query_a_norm, eps);
    auto query = mlx::core::reshape(
        components_.query_b(q_rank), Shape{batch, tokens, heads, head_dim});
    query = rope_tail(query, rotary, cosine, sine);

    auto local_kv = weighted_rms(
        components_.key_value(input), components_.key_value_norm, eps);
    local_kv = rope_tail(local_kv, rotary, cosine, sine);
    local_kv = deepseek_v41_mxfp8_e4m3_sim(local_kv);
    if (state.speculation) {
        if (state.speculation->start_position != pos0 ||
            state.speculation->total_tokens != tokens ||
            state.speculation->local_kv) {
            throw std::runtime_error(
                "DeepSeek-V4.1 speculative target shape changed");
        }
        state.speculation->local_kv = local_kv;
    }

    if (config_.is_kv_source(layer_)) {
        if (!state.compressed_kv || !state.index_k ||
            !components_.compressor_key_value || !components_.compressor_norm) {
            throw std::runtime_error("DeepSeek-V4.1 KV source state is incomplete");
        }
        std::optional<array> latent;
        if (ratio_ == 1) {
            latent = weighted_rms(
                (*components_.compressor_key_value)(input),
                *components_.compressor_norm,
                eps);
        } else {
            if (!components_.compressor_gate || !state.partial_kv ||
                !state.partial_score) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 ratio-two compressor state is incomplete");
            }
            auto projected_kv = mlx::core::astype(
                (*components_.compressor_key_value)(input), mlx::core::float32);
            auto projected_score = mlx::core::astype(
                (*components_.compressor_gate)(input), mlx::core::float32);
            if (state.speculation) {
                state.speculation->projected_kv = projected_kv;
                state.speculation->projected_score = projected_score;
            }
            std::vector<array> emitted;
            if (state.partial_length == 0 && pos0 % ratio_ == 0) {
                const int complete = tokens / ratio_;
                const int cutoff = complete * ratio_;
                if (complete > 0) {
                    auto kv_groups = mlx::core::reshape(
                        slice_axis(projected_kv, 1, 0, cutoff),
                        Shape{batch, complete, ratio_, head_dim});
                    auto score_groups = mlx::core::reshape(
                        slice_axis(projected_score, 1, 0, cutoff),
                        Shape{batch, complete, ratio_, head_dim});
                    auto pooled = mlx::core::sum(
                        kv_groups * mlx::core::softmax(score_groups, 2, true), 2);
                    emitted.push_back(weighted_rms(
                        mlx::core::astype(pooled, input.dtype()),
                        *components_.compressor_norm,
                        eps));
                }
                const int remainder = tokens - cutoff;
                if (remainder > 0) {
                    auto rows = repeated_rows(batch, 0, remainder);
                    *state.partial_kv = dsv4_cache_write_inplace(
                        *state.partial_kv,
                        slice_axis(projected_kv, 1, cutoff, tokens),
                        rows);
                    *state.partial_score = dsv4_cache_write_inplace(
                        *state.partial_score,
                        slice_axis(projected_score, 1, cutoff, tokens),
                        rows);
                }
                state.partial_length = remainder;
            } else {
                for (int token = 0; token < tokens; ++token) {
                    const int slot = (pos0 + token) % ratio_;
                    auto row = mlx::core::full(
                        Shape{batch, 1}, slot, mlx::core::int32);
                    *state.partial_kv = dsv4_cache_write_inplace(
                        *state.partial_kv,
                        slice_axis(projected_kv, 1, token, token + 1),
                        row);
                    *state.partial_score = dsv4_cache_write_inplace(
                        *state.partial_score,
                        slice_axis(projected_score, 1, token, token + 1),
                        row);
                    if (slot + 1 == ratio_) {
                        auto pooled = mlx::core::sum(
                            *state.partial_kv *
                                mlx::core::softmax(*state.partial_score, 1, true),
                            1,
                            true);
                        emitted.push_back(weighted_rms(
                            mlx::core::astype(pooled, input.dtype()),
                            *components_.compressor_norm,
                            eps));
                    }
                }
                state.partial_length = (pos0 + tokens) % ratio_;
            }
            if (!emitted.empty()) {
                latent = emitted.size() == 1
                    ? std::move(emitted.front())
                    : mlx::core::concatenate(std::move(emitted), 1);
            }
        }

        if (latent) {
            const int count = latent->shape(1);
            const int begin = state.compressed_length;
            if (begin + count > state.compressed_kv->shape(1)) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 compressed cache capacity exceeded");
            }
            auto group_positions = positions(
                begin * ratio_, (begin + count) * ratio_);
            if (ratio_ > 1) {
                group_positions = slice_axis(group_positions, 0, 0, count * ratio_);
                auto every = mlx::core::arange(
                    begin * ratio_,
                    (begin + count) * ratio_,
                    ratio_,
                    mlx::core::int32);
                group_positions = std::move(every);
            }
            auto group_cosine = mlx::core::take(rope_.first, group_positions, 0);
            auto group_sine = mlx::core::take(rope_.second, group_positions, 0);

            auto index_key = weighted_rms(
                (*components_.index_key)(*latent),
                *components_.index_key_norm,
                eps);
            index_key = rope_tail(
                index_key, rotary, group_cosine, group_sine);
            index_key = dsv4_fp4_sim(index_key);
            auto cache_rows = repeated_rows(batch, begin, count);
            *state.index_k = dsv4_cache_write_inplace(
                *state.index_k, index_key, cache_rows);

            auto compressed = rope_tail(
                *latent, rotary, group_cosine, group_sine);
            compressed = deepseek_v41_mxfp4_e4m3_scale_sim(compressed);
            *state.compressed_kv = dsv4_cache_write_inplace(
                *state.compressed_kv, compressed, cache_rows);
            state.compressed_length += count;
        }
        shared.compressed_kv = *state.compressed_kv;
        shared.index_k = *state.index_k;
        shared.compressed_length = state.compressed_length;
        shared.ratio = ratio_;
    } else if (ratio_ > 0) {
        if (!shared.compressed_kv || !shared.index_k ||
            shared.ratio != ratio_) {
            throw std::runtime_error(
                "DeepSeek-V4.1 CSA2 consumer ran before its source layer");
        }
    }

    array topk = empty_topk(batch, tokens);
    if (ratio_ > 0) {
        const int pool_length = shared.compressed_length;
        if (config_.is_index_source(layer_)) {
            if (!components_.index_query || !components_.index_score) {
                throw std::runtime_error("DeepSeek-V4.1 index source is incomplete");
            }
            if (pool_length > 0) {
                auto index_query = mlx::core::reshape(
                    (*components_.index_query)(q_rank),
                    Shape{
                        batch,
                        tokens,
                        checked_int(config_.index_n_heads, "index heads"),
                        checked_int(config_.index_head_dim, "index dimension"),
                    });
                index_query = rope_tail(
                    index_query, rotary, cosine, sine);
                index_query = dsv4_fp4_sim(index_query);
                auto weights = (*components_.index_score)(input);
                auto scores = index_scores(
                    index_query,
                    pool_prefix(*shared.index_k, pool_length),
                    weights,
                    ratio_,
                    pos0);
                if (layer_ == config_.candidate_source_layer) {
                    shared.candidate_blocks = candidate_blocks_from_scores(
                        scores,
                        ratio_,
                        pos0,
                        checked_int(config_.candidate_topk_blocks, "candidate blocks"),
                        checked_int(config_.candidate_block_size, "candidate block size"));
                }
                if (layer_ > config_.candidate_source_layer) {
                    if (!shared.candidate_blocks) {
                        throw std::runtime_error(
                            "DeepSeek-V4.1 hierarchical indexer has no candidates");
                    }
                    auto candidates = candidate_positions(
                        *shared.candidate_blocks,
                        checked_int(config_.candidate_block_size, "candidate block size"),
                        pool_length);
                    auto safe = mlx::core::maximum(
                        candidates, array(0, mlx::core::int32));
                    auto candidate_scores = mlx::core::take_along_axis(
                        scores, safe, -1);
                    candidate_scores = mlx::core::where(
                        mlx::core::greater_equal(
                            candidates, array(0, mlx::core::int32)),
                        candidate_scores,
                        array(
                            -std::numeric_limits<float>::infinity(),
                            candidate_scores.dtype()));
                    topk = topk_from_scores(
                        candidate_scores,
                        checked_int(config_.index_topk, "index top-k"),
                        pool_length,
                        candidates);
                } else {
                    topk = topk_from_scores(
                        scores,
                        checked_int(config_.index_topk, "index top-k"),
                        pool_length);
                }
            }
            shared.topk = topk;
        } else {
            if (!shared.topk) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 CSA2 consumer has no published index plan");
            }
            topk = *shared.topk;
        }
    }

    array local_for_attention = state.local_kv;
    if (tokens == 1) {
        auto rows = mlx::core::full(
            Shape{batch, 1}, pos0 % window, mlx::core::int32);
        state.local_kv = dsv4_cache_write_inplace(
            state.local_kv, local_kv, rows);
        local_for_attention = state.local_kv;
    } else {
        const int history = std::min(pos0, window);
        array history_values = slice_axis(state.local_kv, 1, 0, 0);
        if (history > 0) {
            auto history_positions = positions(pos0 - history, pos0);
            auto slots = mlx::core::remainder(
                history_positions, array(window, mlx::core::int32));
            history_values = mlx::core::take(state.local_kv, slots, 1);
        }
        local_for_attention = mlx::core::concatenate(
            {history_values, local_kv}, 1);
        const int recent = std::min(tokens, window);
        auto recent_values = slice_axis(
            local_kv, 1, tokens - recent, tokens);
        auto recent_positions = positions(
            pos0 + tokens - recent, pos0 + tokens);
        auto recent_slots = mlx::core::remainder(
            recent_positions, array(window, mlx::core::int32));
        auto rows = mlx::core::broadcast_to(
            mlx::core::reshape(recent_slots, Shape{1, recent}),
            Shape{batch, recent});
        state.local_kv = dsv4_cache_write_inplace(
            state.local_kv, recent_values, rows);
    }

    auto attended = apply_attention(
        query,
        local_for_attention,
        ratio_ > 0 ? shared.compressed_kv : std::nullopt,
        ratio_ > 0 ? shared.compressed_length : 0,
        topk,
        components_.sinks,
        pos0,
        ratio_,
        window);
    attended = rope_tail(attended, rotary, cosine, sine, true);
    const int groups = checked_int(config_.o_groups, "output groups");
    const int group_width = heads * head_dim / groups;
    auto grouped = mlx::core::reshape(
        attended, Shape{batch, tokens, groups, group_width});
    auto low_rank = components_.output_a.grouped_row_matmul(grouped, groups);
    low_rank = mlx::core::reshape(
        low_rank,
        Shape{
            batch,
            tokens,
            groups * checked_int(config_.o_lora_rank, "output rank"),
        });
    state.position += tokens;
    return components_.output_b(low_rank);
}

} // namespace mfq::metal
