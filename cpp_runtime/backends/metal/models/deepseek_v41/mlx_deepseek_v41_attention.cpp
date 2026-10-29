#include "mlx_deepseek_v41_attention.h"

#include "mlx_dsa.h"
#include "mlx_dsa_cache_ops.h"
#include "mlx_csa2.h"
#include "mlx_detached_copy.h"
#include "mlx_mx.h"
#include "mlx_sparse_attention.h"
#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

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
        return mlx_rope_adjacent(input, cosine, sine, inverse);
    }
    return mlx::core::concatenate(
        {
            slice_axis(input, -1, 0, width - rotary),
            mlx_rope_adjacent(
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

array topk_from_scores(
    const array& scores,
    int requested,
    const std::optional<array>& source_indices = std::nullopt) {
    const int width = scores.shape(-1);
    const int count = std::min(requested, width);
    if (count <= 0) {
        return empty_topk(scores.shape(0), scores.shape(1));
    }
    return mlx_csa2_topk(scores, count, source_indices);
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

array all_visible_blocks(
    int batch,
    int tokens,
    int pool_length,
    int ratio,
    int pos0,
    int block_size) {
    const int blocks = (pool_length + block_size - 1) / block_size;
    auto ids = mlx::core::broadcast_to(
        mlx::core::reshape(
            positions(0, blocks), Shape{1, 1, blocks}),
        Shape{batch, tokens, blocks});
    auto query_positions = mlx::core::reshape(
        positions(pos0, pos0 + tokens), Shape{1, tokens, 1});
    auto visible = mlx::core::floor_divide(
        query_positions + array(1, mlx::core::int32),
        array(ratio, mlx::core::int32));
    auto visible_blocks = mlx::core::floor_divide(
        visible + array(block_size - 1, mlx::core::int32),
        array(block_size, mlx::core::int32));
    return mlx::core::where(
        mlx::core::less(ids, visible_blocks),
        ids,
        array(-1, mlx::core::int32));
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
    auto selected = count == blocks
        ? mlx::core::broadcast_to(
              block_ids,
              Shape{batch, tokens, blocks})
        : slice_axis(
              mlx::core::argpartition(
                  block_scores,
                  blocks - count,
                  -1),
              -1,
              blocks - count,
              blocks);
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
    int pos0,
    const std::optional<array>& candidates = std::nullopt) {
    const int batch = query.shape(0);
    const int tokens = query.shape(1);
    const int heads = query.shape(2);
    const int pool_length = keys.shape(1);
    if (keys.dtype() == mlx::core::uint8) {
        return mlx_csa2_index_scores(query, keys, weights, pos0, ratio, candidates);
    }
    if (candidates) {
        auto scores = index_scores(query, keys, weights, ratio, pos0);
        auto safe = mlx::core::maximum(*candidates, array(0, mlx::core::int32));
        auto selected = mlx::core::take_along_axis(scores, safe, -1);
        return mlx::core::where(*candidates >= array(0, mlx::core::int32), selected,
            array(-std::numeric_limits<float>::infinity(), selected.dtype()));
    }
    if ((heads == 32 || heads == 64) && query.shape(3) == 128) {
        return tokens == 1
            ? mlx_dsa_indexer_scores_decode(
                  query,
                  keys,
                  weights,
                  pos0,
                  ratio,
                  pool_length)
            : mlx_dsa_indexer_scores(
                  query,
                  keys,
                  weights,
                  pos0,
                  ratio);
    }
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
    if (local.dtype() == mlx::core::uint8) {
        return mlx_csa2_attention(query, local, pooled, pool_length, topk, sinks,
            pos0, ratio == 0 ? 1 : ratio, window);
    }
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
        auto chronological = mlx_circular_cache_history(
            local, pos0 + 1);
        auto plan = mlx_dsa_build_prefill_plan(
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
    if (query.shape(2) == 64 && query.shape(3) == 512) {
        return mlx_sparse_circular_mla_attention(
            transposed,
            local,
            pooled,
            pool_length,
            topk,
            sinks,
            pos0,
            ratio == 0 ? 1 : ratio,
            window);
    }
    auto plan = mlx_dsa_build_prefill_plan(
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
    const bool packed = head_dim % 32 == 0 && config.index_head_dim % 32 == 0;
    MlxDeepseekV41AttentionState state{
        mlx::core::zeros(Shape{batch, window, packed ? head_dim + head_dim / 32 : head_dim},
            packed ? mlx::core::uint8 : dtype)};
    if (config.is_kv_source(layer)) {
        const int ratio = static_cast<int>(config.compress_ratios[layer]);
        const int capacity = std::max(1, max_context / ratio);
        state.compressed_kv = mlx::core::zeros(
            Shape{batch, capacity, packed ? head_dim / 2 + head_dim / 16 : head_dim},
            packed ? mlx::core::uint8 : dtype);
        state.index_k = mlx::core::zeros(
            Shape{batch, capacity, packed ? checked_int(config.index_head_dim, "index dimension") / 2 +
                checked_int(config.index_head_dim, "index dimension") / 32 : checked_int(config.index_head_dim, "index dimension")},
            packed ? mlx::core::uint8 : dtype);
        if (ratio > 1) {
            state.partial_kv = mlx::core::zeros(
                Shape{batch, ratio, head_dim}, mlx::core::float32);
            state.partial_score = mlx::core::full(
                Shape{batch, ratio, head_dim},
                -std::numeric_limits<float>::infinity(),
                mlx::core::float32);
        }
    }
    state.resources.set({state.nbytes(), static_cast<std::size_t>(batch)});
    return state;
}

void MlxDeepseekV41AttentionState::reset() noexcept {
    // Cache contents are addressed only through these logical lengths. Keep
    // the large fixed-capacity arrays resident and let subsequent writes
    // replace rows as they become reachable again.
    speculation.reset();
    position = 0;
    compressed_length = 0;
    partial_length = 0;
}

MlxDeepseekV41AttentionState
MlxDeepseekV41AttentionState::snapshot(bool copy_arrays) const {
    if (speculation) {
        throw std::runtime_error(
            "cannot snapshot a speculative DeepSeek-V4.1 attention cache");
    }
    const auto copy_optional = [copy_arrays](const std::optional<array>& value) {
        return value
            ? std::optional<array>(copy_arrays ? detached_copy(*value) : *value)
            : std::nullopt;
    };
    MlxDeepseekV41AttentionState result;
    result.local_kv = copy_arrays ? detached_copy(local_kv) : local_kv;
    result.compressed_kv = copy_optional(compressed_kv);
    result.index_k = copy_optional(index_k);
    result.partial_kv = copy_optional(partial_kv);
    result.partial_score = copy_optional(partial_score);
    result.position = position;
    result.compressed_length = compressed_length;
    result.partial_length = partial_length;
    return result;
}

void MlxDeepseekV41AttentionState::restore_snapshot(
    MlxDeepseekV41AttentionState snapshot) {
    const auto same_layout = [](
        const std::optional<array>& live,
        const std::optional<array>& saved) {
        return live.has_value() == saved.has_value() &&
            (!live || (live->shape() == saved->shape() &&
                       live->dtype() == saved->dtype()));
    };
    if (snapshot.speculation ||
        local_kv.shape() != snapshot.local_kv.shape() ||
        local_kv.dtype() != snapshot.local_kv.dtype() ||
        !same_layout(compressed_kv, snapshot.compressed_kv) ||
        !same_layout(index_k, snapshot.index_k) ||
        !same_layout(partial_kv, snapshot.partial_kv) ||
        !same_layout(partial_score, snapshot.partial_score) ||
        snapshot.position < 0 || snapshot.compressed_length < 0 ||
        snapshot.partial_length < 0 ||
        (snapshot.compressed_kv &&
         snapshot.compressed_length > snapshot.compressed_kv->shape(1)) ||
        (snapshot.partial_kv &&
         snapshot.partial_length > snapshot.partial_kv->shape(1))) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 attention snapshot topology mismatch");
    }
    local_kv = std::move(snapshot.local_kv);
    compressed_kv = std::move(snapshot.compressed_kv);
    index_k = std::move(snapshot.index_k);
    partial_kv = std::move(snapshot.partial_kv);
    partial_score = std::move(snapshot.partial_score);
    position = snapshot.position;
    compressed_length = snapshot.compressed_length;
    partial_length = snapshot.partial_length;
    speculation.reset();
    resources.set({nbytes(), static_cast<std::size_t>(local_kv.shape(0))});
}

std::size_t MlxDeepseekV41AttentionState::nbytes() const noexcept {
    std::size_t bytes = local_kv.nbytes();
    const auto append = [&bytes](const std::optional<array>& value) {
        if (value) bytes += value->nbytes();
    };
    append(compressed_kv);
    append(index_k);
    append(partial_kv);
    append(partial_score);
    return bytes;
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
    components_.output_a.prepare_grouped_row_matmul(checked_int(config_.o_groups, "output groups"));
    std::vector<const MlxLinear*> input_projections{
        &components_.query_a,
        &components_.key_value,
    };
    if (components_.compressor_key_value) {
        input_projections.push_back(&*components_.compressor_key_value);
    }
    if (components_.compressor_gate) {
        input_projections.push_back(&*components_.compressor_gate);
    }
    if (components_.index_score) {
        // The index score is another projection of the same hidden state.
        // Keep it in the common projection batch so index-source layers do
        // not pay a separate small-M dispatch on every decode/verify step.
        input_projections.push_back(&*components_.index_score);
    }
    input_projections_.emplace(std::move(input_projections));
    if (components_.index_query) {
        query_projections_.emplace(
            std::vector<const MlxLinear*>{
                &components_.query_b,
                &*components_.index_query,
            });
    }
}

std::vector<array> MlxDeepseekV41Attention::begin_speculative(
    MlxDeepseekV41AttentionState& state,
    int confirmed_tokens,
    int total_tokens) const {
    const int batch = state.local_kv.shape(0);
    const int window = state.local_kv.shape(1);
    if (state.speculation || confirmed_tokens < 0 ||
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
    state.local_kv = mlx_cache_write_inplace(
        state.local_kv, transaction->local_backup, all_rows);
    if (keep > 0) {
        auto kept_slots = slice_axis(all_slots, 0, 0, keep);
        auto kept_rows = mlx::core::broadcast_to(
            mlx::core::reshape(kept_slots, Shape{1, keep}),
            Shape{batch, keep});
        state.local_kv = mlx_cache_write_inplace(
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
        if (keep > 0) {
            auto restored = mlx_dsa_compress_window(
                slice_axis(*transaction->projected_kv, 1, 0, keep),
                slice_axis(*transaction->projected_score, 1, 0, keep),
                *state.partial_kv, *state.partial_score,
                *components_.compressor_norm, transaction->partial_length,
                mlx::core::float32, static_cast<float>(config_.rms_eps));
            *state.partial_kv = std::move(restored.partial_kv);
            *state.partial_score = std::move(restored.partial_score);
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
    int pos0,
    bool ced_replay,
    const array* encoder_source) const {
    const int hidden = checked_int(config_.hidden, "hidden size");
    const int heads = checked_int(config_.n_heads, "head count");
    const int head_dim = checked_int(config_.head_dim, "head dimension");
    const int rotary = checked_int(config_.rope_head_dim, "rotary dimension");
    const int window = checked_int(config_.sliding_window, "sliding window");
    const bool packed = state.local_kv.dtype() == mlx::core::uint8;
    if (input.ndim() != 3 || input.shape(0) <= 0 || input.shape(1) <= 0 ||
        input.shape(2) != hidden ||
        (ced_replay ? state.position > pos0 : state.position != pos0) ||
        pos0 < 0 || pos0 + input.shape(1) > max_context_ ||
        state.local_kv.shape() != Shape{input.shape(0), window, packed ? head_dim + head_dim / 32 : head_dim}) {
        throw std::invalid_argument("DeepSeek-V4.1 attention input/cache mismatch");
    }
    const int batch = input.shape(0);
    const int tokens = input.shape(1);
    if (ced_replay && (tokens < window || state.speculation || ratio_ != 1 ||
            layer_ < config_.n_layers / 2 ||
            config_.is_kv_source(layer_) != (encoder_source != nullptr))) {
        throw std::invalid_argument("invalid DeepSeek-V4.1 CED replay window");
    }
    if (encoder_source && (!ced_replay || !config_.is_kv_source(layer_) ||
            encoder_source->ndim() != 3 || encoder_source->shape(0) != batch ||
            encoder_source->shape(2) != hidden ||
            state.position + encoder_source->shape(1) != pos0 + tokens ||
            state.compressed_length != state.position)) {
        throw std::invalid_argument("invalid DeepSeek-V4.1 CED encoder source");
    }
    const float eps = static_cast<float>(config_.rms_eps);
    auto cosine = slice_axis(rope_.first, 0, pos0, pos0 + tokens);
    auto sine = slice_axis(rope_.second, 0, pos0, pos0 + tokens);

    auto input_projections = (*input_projections_)(input);
    std::size_t projection_offset = 0;
    auto query_a = std::move(input_projections.at(projection_offset++));
    auto key_value = std::move(input_projections.at(projection_offset++));
    std::optional<array> compressor_key_value;
    std::optional<array> compressor_gate;
    std::optional<array> index_score_projection;
    if (components_.compressor_key_value) {
        compressor_key_value.emplace(
            std::move(input_projections.at(projection_offset++)));
    }
    if (components_.compressor_gate) {
        compressor_gate.emplace(
            std::move(input_projections.at(projection_offset++)));
    }
    if (components_.index_score) {
        index_score_projection.emplace(
            std::move(input_projections.at(projection_offset++)));
    }
    if (projection_offset != input_projections.size()) {
        throw std::logic_error(
            "DeepSeek-V4.1 input projection group output mismatch");
    }
    if (encoder_source) {
        compressor_key_value = (*components_.compressor_key_value)(*encoder_source);
    }

    auto q_rank = weighted_rms(
        query_a, components_.query_a_norm, eps);

    auto local_kv = mlx_weighted_rms_rope_mxfp8_sim(
        key_value,
        components_.key_value_norm,
        eps,
        rotary,
        cosine,
        sine);
    if (packed) local_kv = mlx_csa2_pack(local_kv, 8, 32);
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
            latent = packed ? *compressor_key_value : weighted_rms(
                *compressor_key_value, *components_.compressor_norm, eps);
        } else {
            if (!components_.compressor_gate || !state.partial_kv ||
                !state.partial_score) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 ratio-two compressor state is incomplete");
            }
            auto projected_kv = mlx::core::astype(
                *compressor_key_value, mlx::core::float32);
            auto projected_score = mlx::core::astype(
                *compressor_gate, mlx::core::float32);
            if (state.speculation) {
                state.speculation->projected_kv = projected_kv;
                state.speculation->projected_score = projected_score;
            }
            auto compressed = mlx_dsa_compress_window(projected_kv, projected_score,
                *state.partial_kv, *state.partial_score, *components_.compressor_norm,
                state.partial_length, input.dtype(), eps);
            *state.partial_kv = std::move(compressed.partial_kv);
            *state.partial_score = std::move(compressed.partial_score);
            state.partial_length = (state.partial_length + tokens) % ratio_;
            if (compressed.latent.shape(1) > 0) latent = std::move(compressed.latent);
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

            auto cache_rows = repeated_rows(batch, begin, count);
            if (packed) {
                if (ratio_ == 1) {
                    auto update = mlx_dsa_norm_rope_pack_store(*state.compressed_kv,
                        *latent, *components_.compressor_norm, group_cosine, group_sine,
                        begin, rotary, 16, true, eps, true, true);
                    *state.compressed_kv = std::move(update.cache);
                    latent = std::move(update.normalized);
                } else {
                    *state.compressed_kv = mlx_dsa_norm_rope_pack_store(*state.compressed_kv,
                        *latent, *components_.compressor_norm, group_cosine, group_sine,
                        begin, rotary, 16, true, eps, false).cache;
                }
                *state.index_k = mlx_dsa_norm_rope_pack_store(*state.index_k,
                    (*components_.index_key)(*latent), *components_.index_key_norm,
                    group_cosine, group_sine, begin, rotary, 32, false, eps, true).cache;
            } else {
                auto index_key = weighted_rms((*components_.index_key)(*latent),
                    *components_.index_key_norm, eps);
                index_key = mlx_mxfp4_sim(rope_tail(index_key, rotary, group_cosine, group_sine));
                *state.index_k = mlx_cache_write_inplace(*state.index_k, index_key, cache_rows);
                auto compressed = mlx_mxfp4_e4m3_scale_sim(
                    rope_tail(*latent, rotary, group_cosine, group_sine));
                *state.compressed_kv = mlx_cache_write_inplace(*state.compressed_kv, compressed, cache_rows);
            }
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

    const bool needs_index_query = [&]() {
        if (ratio_ <= 0 ||
            !config_.is_index_source(layer_) ||
            shared.compressed_length <= 0) {
            return false;
        }
        const int requested_topk = checked_int(
            config_.index_topk, "index top-k");
        const int candidate_block_size = checked_int(
            config_.candidate_block_size, "candidate block size");
        const int candidate_block_count =
            (shared.compressed_length + candidate_block_size - 1) /
            candidate_block_size;
        const bool all_candidates_fit =
            layer_ != config_.candidate_source_layer ||
            candidate_block_count <= checked_int(
                config_.candidate_topk_blocks,
                "candidate blocks");
        return shared.compressed_length > requested_topk ||
            !all_candidates_fit;
    }();
    std::optional<array> index_query_projection;
    array query_projection = [&]() {
        if (!needs_index_query) {
            return components_.query_b(q_rank);
        }
        auto values = (*query_projections_)(q_rank);
        if (values.size() != 2) {
            throw std::logic_error(
                "DeepSeek-V4.1 query projection group output mismatch");
        }
        index_query_projection.emplace(std::move(values[1]));
        return std::move(values[0]);
    }();
    auto query = mlx::core::reshape(
        std::move(query_projection),
        Shape{batch, tokens, heads, head_dim});
    query = rope_tail(query, rotary, cosine, sine);

    array topk = empty_topk(batch, tokens);
    if (ratio_ > 0) {
        const int pool_length = shared.compressed_length;
        if (config_.is_index_source(layer_)) {
            if (!components_.index_query || !components_.index_score) {
                throw std::runtime_error("DeepSeek-V4.1 index source is incomplete");
            }
            if (pool_length > 0) {
                const int requested_topk = checked_int(
                    config_.index_topk, "index top-k");
                const int candidate_block_size = checked_int(
                    config_.candidate_block_size, "candidate block size");
                if (!needs_index_query) {
                    topk = all_visible_indices(
                        batch, tokens, pool_length, ratio_, pos0);
                    if (layer_ == config_.candidate_source_layer) {
                        shared.candidate_blocks = all_visible_blocks(
                            batch,
                            tokens,
                            pool_length,
                            ratio_,
                            pos0,
                            candidate_block_size);
                    }
                } else {
                    if (!index_query_projection) {
                        throw std::logic_error(
                            "DeepSeek-V4.1 Indexer query projection is unavailable");
                    }
                    auto index_query = mlx::core::reshape(
                        *index_query_projection,
                        Shape{
                            batch,
                            tokens,
                            checked_int(config_.index_n_heads, "index heads"),
                            checked_int(config_.index_head_dim, "index dimension"),
                        });
                    index_query = rope_tail(
                        index_query, rotary, cosine, sine);
                    index_query = mlx_mxfp4_sim(index_query);
                    if (!index_score_projection) {
                        throw std::logic_error(
                            "DeepSeek-V4.1 Indexer score projection is unavailable");
                    }
                    const auto& weights = *index_score_projection;
                    std::optional<array> candidates;
                    if (layer_ > config_.candidate_source_layer) {
                        if (!shared.candidate_blocks)
                            throw std::runtime_error("DeepSeek-V4.1 hierarchical indexer has no candidates");
                        candidates = candidate_positions(*shared.candidate_blocks, candidate_block_size, pool_length);
                    }
                    auto scores = index_scores(
                        index_query,
                        pool_prefix(*shared.index_k, pool_length),
                        weights,
                        ratio_,
                        pos0,
                        candidates);
                    if (layer_ == config_.candidate_source_layer) {
                        shared.candidate_blocks = candidate_blocks_from_scores(
                            scores,
                            ratio_,
                            pos0,
                            checked_int(
                                config_.candidate_topk_blocks,
                                "candidate blocks"),
                            candidate_block_size);
                    }
                    if (candidates) {
                        topk = topk_from_scores(
                            scores,
                            requested_topk,
                            candidates);
                    } else {
                        topk = topk_from_scores(
                            scores,
                            requested_topk);
                    }
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
    std::optional<array> pending_local_values;
    std::optional<array> pending_local_rows;
    if (tokens == 1) {
        auto rows = mlx::core::full(
            Shape{batch, 1}, pos0 % window, mlx::core::int32);
        state.local_kv = mlx_cache_write_inplace(
            state.local_kv, local_kv, rows);
        local_for_attention = state.local_kv;
    } else {
        if (ced_replay) {
            local_for_attention = local_kv;
        } else {
            auto history_values = mlx_circular_cache_history(
                state.local_kv, pos0);
            local_for_attention = mlx::core::concatenate(
                {history_values, local_kv}, 1);
        }
        const int recent = std::min(tokens, window);
        pending_local_values = slice_axis(
            local_kv, 1, tokens - recent, tokens);
        auto recent_positions = positions(
            pos0 + tokens - recent, pos0 + tokens);
        auto recent_slots = mlx::core::remainder(
            recent_positions, array(window, mlx::core::int32));
        pending_local_rows = mlx::core::broadcast_to(
            mlx::core::reshape(recent_slots, Shape{1, recent}),
            Shape{batch, recent});
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
    if (pending_local_values) {
        // Multi-token attention reads the previous circular-cache contents.
        // Preserve that dependency before the in-place write can reuse a
        // wrapped slot; this is also required by MTP target verification.
        auto ordered_local = mlx::core::depends(
            std::vector<array>{state.local_kv},
            std::vector<array>{attended});
        state.local_kv = mlx_cache_write_inplace(
            ordered_local.front(),
            *pending_local_values,
            *pending_local_rows);
    }
    const int groups = checked_int(config_.o_groups, "output groups");
    auto low_rank = components_.output_a.grouped_row_matmul_inverse_rope(
        attended,
        groups,
        cosine,
        sine,
        head_dim,
        rotary);
    low_rank = mlx::core::reshape(
        low_rank,
        Shape{
            batch,
            tokens,
            groups * checked_int(config_.o_lora_rank, "output rank"),
        });
    state.position = pos0 + tokens;
    return components_.output_b(low_rank);
}

} // namespace mfq::metal
