#include "mlx_qwen35_full_attention.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

int checked_int(std::int64_t value, const char* name) {
    if (value <= 0 ||
        value > static_cast<std::int64_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string("invalid Qwen3.5 ") + name);
    }
    return static_cast<int>(value);
}

std::optional<MlxRmsNorm> load_optional_norm(
    const MfqContainer& model,
    const std::string& name,
    const Qwen35Config& config) {
    if (!model.contains(name)) {
        return std::nullopt;
    }
    return load_qwen35_rms_norm(
        model,
        name,
        config.rms_norm_eps,
        config.norm_weight_offset);
}

std::optional<MlxGroupedLinear> make_grouped_linear(
    std::initializer_list<const MlxLinear*> projections) {
    std::vector<MlxGroupedLinearWeightRef> weights;
    weights.reserve(projections.size());
    for (const auto* projection : projections) {
        const auto weight =
            projection->grouped_weight_ref();
        if (!weight) {
            return std::nullopt;
        }
        weights.push_back(*weight);
    }
    return MlxGroupedLinear(std::move(weights));
}

void validate_position_shape(
    const array& positions,
    int tokens) {
    const bool valid_rank =
        positions.ndim() == 1 ||
        (positions.ndim() == 2 &&
         (positions.shape(0) == 1 ||
          positions.shape(0) == 3));
    if (!valid_rank) {
        throw std::runtime_error(
            "Qwen3.5 RoPE positions must have [tokens], "
            "[1,tokens], or [3,tokens] shape");
    }
    if (positions.shape(-1) != tokens) {
        throw std::runtime_error(
            "Qwen3.5 RoPE position length must match the token count");
    }
}

} // namespace

MlxQwen35DenseSwiGlu MlxQwen35DenseSwiGlu::load(
    const MfqContainer& model,
    const Qwen35ResolvedLayerNames& names) {
    const auto gate_high = names.ffn_gate + ".in_high";
    const auto up_high = names.ffn_up + ".in_high";
    const auto down_high = names.ffn_down + ".in_high";
    const bool has_gate = model.contains(gate_high);
    const bool has_up = model.contains(up_high);
    const bool has_down = model.contains(down_high);
    if ((has_gate || has_up || has_down) &&
        !(has_gate && has_up && has_down)) {
        throw std::runtime_error(
            "important-neuron FFN requires matching "
            "gate/up/down .in_high records");
    }
    std::shared_ptr<MlxQwen35DenseSwiGlu> high;
    if (has_gate) {
        high = std::make_shared<MlxQwen35DenseSwiGlu>(
            MlxLinear::load(model, gate_high),
            MlxLinear::load(model, up_high),
            MlxLinear::load(model, down_high));
    }
    return MlxQwen35DenseSwiGlu(
        MlxLinear::load(model, names.ffn_gate),
        MlxLinear::load(model, names.ffn_up),
        MlxLinear::load(model, names.ffn_down),
        std::move(high));
}

MlxQwen35DenseSwiGlu::MlxQwen35DenseSwiGlu(
    MlxLinear gate,
    MlxLinear up,
    MlxLinear down,
    std::shared_ptr<MlxQwen35DenseSwiGlu>
        important_neurons)
    : gate_(std::move(gate)),
      up_(std::move(up)),
      down_(std::move(down)),
      important_neurons_(std::move(important_neurons)),
      input_size_(gate_.input_size()),
      intermediate_size_(gate_.output_size()),
      output_size_(down_.output_size()) {
    if (input_size_ <= 0 ||
        intermediate_size_ <= 0 ||
        output_size_ <= 0 ||
        up_.input_size() != input_size_ ||
        up_.output_size() != intermediate_size_ ||
        down_.input_size() != intermediate_size_) {
        throw std::runtime_error(
            "incompatible Qwen3.5 SwiGLU projection dimensions");
    }
    gate_up_ = make_grouped_linear({&gate_, &up_});
    if (important_neurons_) {
        if (important_neurons_->input_size() != input_size_ ||
            important_neurons_->output_size() != output_size_) {
            throw std::runtime_error(
                "important-neuron FFN tensor shapes disagree "
                "with the low branch");
        }
        intermediate_size_ +=
            important_neurons_->intermediate_size();
    }
}

array MlxQwen35DenseSwiGlu::operator()(
    const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error(
            "Qwen3.5 SwiGLU input width mismatch");
    }
    auto output = forward_branch(input);
    if (important_neurons_) {
        output = output + (*important_neurons_)(input);
    }
    return output;
}

array MlxQwen35DenseSwiGlu::forward_branch(
    const array& input) const {
    // The heterogeneous grouped kernel reduces launch count, but its
    // descriptor/branching overhead and lower memory throughput make it
    // slower than two ordinary packed GEMVs for single-token decode. Keep
    // grouping for multi-row prefill, where launch amortization is useful.
    const bool use_grouped_rows =
        detail::qwen35_use_grouped_projection_rows(
            input.size(), input_size_);
    if (!use_grouped_rows) {
        const auto* gate_nint = gate_.nint_weight_ref();
        const auto* up_nint = up_.nint_weight_ref();
        if (gate_nint != nullptr &&
            up_nint != nullptr &&
            gate_nint->can_fuse_swiglu(*up_nint)) {
            return down_(
                gate_nint->swiglu(*up_nint, input));
        }
    }
    if (use_grouped_rows &&
        gate_up_ &&
        gate_up_->supports(input)) {
        auto projected = (*gate_up_)(input);
        const auto& gate = projected.at(0);
        const auto& up = projected.at(1);
        return down_(
            gate * mlx::core::sigmoid(gate) * up);
    }
    const auto gate = gate_(input);
    const auto up = up_(input);
    return down_(gate * mlx::core::sigmoid(gate) * up);
}

MlxQwen35FullAttentionBlock
MlxQwen35FullAttentionBlock::load(
    const MfqContainer& model,
    const Qwen35Config& config,
    const Qwen35TensorNames& names,
    std::size_t layer_index) {
    const int layer_count =
        checked_int(config.num_hidden_layers, "num_hidden_layers");
    if (layer_index >= static_cast<std::size_t>(layer_count)) {
        throw std::runtime_error(
            "Qwen3.5 full-attention layer index is out of range");
    }
    if (!config.layer_types.empty() &&
        config.layer_types.at(layer_index) != "full_attention") {
        throw std::runtime_error(
            "Qwen3.5 layer is not a full-attention layer");
    }
    const auto resolved = names.layer(layer_index);
    return MlxQwen35FullAttentionBlock(
        config,
        load_qwen35_rms_norm(
            model,
            resolved.attention_norm,
            config.rms_norm_eps,
            config.norm_weight_offset),
        MlxLinear::load(model, resolved.attention_query),
        MlxLinear::load(model, resolved.attention_key),
        MlxLinear::load(model, resolved.attention_value),
        MlxLinear::load(model, resolved.attention_output),
        load_optional_norm(
            model,
            resolved.attention_query_norm,
            config),
        load_optional_norm(
            model,
            resolved.attention_key_norm,
            config),
        load_qwen35_rms_norm(
            model,
            resolved.ffn_norm,
            config.rms_norm_eps,
            config.norm_weight_offset),
        MlxQwen35DenseSwiGlu::load(model, resolved));
}

MlxQwen35FullAttentionBlock::MlxQwen35FullAttentionBlock(
    Qwen35Config config,
    MlxRmsNorm attention_norm,
    MlxLinear query,
    MlxLinear key,
    MlxLinear value,
    MlxLinear output,
    std::optional<MlxRmsNorm> query_norm,
    std::optional<MlxRmsNorm> key_norm,
    MlxRmsNorm ffn_norm,
    MlxQwen35DenseSwiGlu ffn)
    : config_(std::move(config)),
      attention_norm_(std::move(attention_norm)),
      query_(std::move(query)),
      key_(std::move(key)),
      value_(std::move(value)),
      output_(std::move(output)),
      query_norm_(std::move(query_norm)),
      key_norm_(std::move(key_norm)),
      ffn_norm_(std::move(ffn_norm)),
      ffn_(std::move(ffn)) {
    validate_components();
    qkv_ = make_grouped_linear(
        {&query_, &key_, &value_});
}

void MlxQwen35FullAttentionBlock::validate_components() const {
    const int hidden = checked_int(config_.hidden_size, "hidden_size");
    const int intermediate =
        checked_int(config_.intermediate_size, "intermediate_size");
    const int query_heads =
        checked_int(config_.num_attention_heads, "num_attention_heads");
    const int kv_heads =
        checked_int(config_.num_key_value_heads, "num_key_value_heads");
    const int head_dimension =
        checked_int(config_.head_dim, "head_dim");
    const int rotary_dimension =
        checked_int(config_.rotary_dim, "rotary_dim");
    checked_int(
        config_.max_position_embeddings,
        "max_position_embeddings");

    if (query_heads % kv_heads != 0 ||
        rotary_dimension > head_dimension ||
        rotary_dimension % 2 != 0 ||
        !std::isfinite(config_.rope_base) ||
        config_.rope_base <= 0.0 ||
        !std::isfinite(config_.rms_norm_eps) ||
        config_.rms_norm_eps <= 0.0) {
        throw std::runtime_error(
            "invalid Qwen3.5 full-attention configuration");
    }
    if (!config_.rope_sections.empty()) {
        if (config_.rope_sections.size() != 3) {
            throw std::runtime_error(
                "Qwen3.5 MRoPE sections must have three entries");
        }
        std::int64_t section_sum = 0;
        for (const auto section : config_.rope_sections) {
            if (section < 0) {
                throw std::runtime_error(
                    "Qwen3.5 MRoPE sections cannot be negative");
            }
            section_sum += section;
        }
        if (section_sum != rotary_dimension / 2) {
            throw std::runtime_error(
                "Qwen3.5 MRoPE sections must sum to rotary_dim / 2");
        }
    }

    const std::int64_t attention_size =
        static_cast<std::int64_t>(query_heads) * head_dimension;
    const std::int64_t kv_size =
        static_cast<std::int64_t>(kv_heads) * head_dimension;
    const std::int64_t query_output = attention_size *
        (config_.attention_output_gate ? 2 : 1);
    if (attention_size > std::numeric_limits<int>::max() ||
        kv_size > std::numeric_limits<int>::max() ||
        query_output > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "Qwen3.5 attention projection size exceeds int range");
    }

    if (attention_norm_.width() != hidden ||
        ffn_norm_.width() != hidden ||
        query_.input_size() != hidden ||
        query_.output_size() != query_output ||
        key_.input_size() != hidden ||
        key_.output_size() != kv_size ||
        value_.input_size() != hidden ||
        value_.output_size() != kv_size ||
        output_.input_size() != attention_size ||
        output_.output_size() != hidden ||
        ffn_.input_size() != hidden ||
        ffn_.intermediate_size() != intermediate ||
        ffn_.output_size() != hidden) {
        throw std::runtime_error(
            "Qwen3.5 full-attention block projection size mismatch");
    }
    if ((query_norm_ && query_norm_->width() != head_dimension) ||
        (key_norm_ && key_norm_->width() != head_dimension)) {
        throw std::runtime_error(
            "Qwen3.5 Q/K norm width must equal head_dim");
    }
}

void MlxQwen35FullAttentionBlock::reset_cache(
    int batch,
    int initial_capacity) {
    if (batch <= 0) {
        throw std::runtime_error(
            "Qwen3.5 KV cache batch must be positive");
    }
    if (initial_capacity < 0) {
        throw std::runtime_error(
            "Qwen3.5 KV cache initial capacity is invalid");
    }
    const int capacity = std::min(
        initial_capacity,
        checked_int(
            config_.max_position_embeddings,
            "max_position_embeddings"));
    if (cache_ && cache_batch_ == batch &&
        cache_->capacity() >= capacity) {
        cache_->reset();
        return;
    }
    cache_ = std::make_unique<MlxKvCache>(
        batch,
        checked_int(
            config_.num_key_value_heads,
            "num_key_value_heads"),
        checked_int(
            config_.max_position_embeddings,
            "max_position_embeddings"),
        checked_int(config_.head_dim, "head_dim"),
        capacity);
    cache_batch_ = batch;
}

void MlxQwen35FullAttentionBlock::materialize_cache() {
    if (cache_) {
        cache_->materialize();
    }
}

void MlxQwen35FullAttentionBlock::clear_cache() noexcept {
    cache_.reset();
    cache_batch_ = 0;
}

MlxKvCacheSnapshot
MlxQwen35FullAttentionBlock::snapshot_cache() const {
    if (!cache_ || cache_batch_ <= 0) {
        throw std::runtime_error(
            "Qwen3.5 full-attention cache is unavailable");
    }
    return cache_->snapshot();
}

void MlxQwen35FullAttentionBlock::restore_cache(
    const MlxKvCacheSnapshot& snapshot) {
    if (snapshot.batch <= 0 || snapshot.position <= 0) {
        throw std::runtime_error(
            "Qwen3.5 full-attention cache snapshot is invalid");
    }
    reset_cache(snapshot.batch, snapshot.capacity);
    cache_->restore_snapshot(snapshot);
    cache_batch_ = snapshot.batch;
}

void MlxQwen35FullAttentionBlock::trim_cache(int tokens) {
    if (!cache_ || cache_batch_ <= 0) {
        throw std::runtime_error(
            "Qwen3.5 full-attention cache is unavailable");
    }
    cache_->trim(tokens);
}

int MlxQwen35FullAttentionBlock::cache_position() const noexcept {
    return cache_ ? cache_->position() : 0;
}

array MlxQwen35FullAttentionBlock::forward(
    const array& input,
    bool use_cache) {
    if (input.ndim() != 3) {
        throw std::runtime_error(
            "Qwen3.5 block input must have [batch,tokens,hidden] shape");
    }
    const int batch = input.shape(0);
    if (use_cache && !cache_) {
        reset_cache(batch);
    }
    const int offset = use_cache ? cache_position() : 0;
    return forward(input, offset, use_cache);
}

array MlxQwen35FullAttentionBlock::forward(
    const array& input,
    int position_offset,
    bool use_cache) {
    return forward_impl(
        input,
        nullptr,
        position_offset,
        use_cache);
}

array MlxQwen35FullAttentionBlock::forward(
    const array& input,
    const array& positions,
    bool use_cache) {
    if (input.ndim() != 3) {
        throw std::runtime_error(
            "Qwen3.5 block input must have [batch,tokens,hidden] shape");
    }
    const int batch = input.shape(0);
    if (use_cache && !cache_) {
        reset_cache(batch);
    }
    const int offset = use_cache ? cache_position() : 0;
    return forward(
        input,
        positions,
        offset,
        use_cache);
}

array MlxQwen35FullAttentionBlock::forward(
    const array& input,
    const array& positions,
    int position_offset,
    bool use_cache) {
    return forward_impl(
        input,
        &positions,
        position_offset,
        use_cache);
}

array MlxQwen35FullAttentionBlock::forward_impl(
    const array& input,
    const array* positions,
    int position_offset,
    bool use_cache) {
    if (input.ndim() != 3 ||
        input.shape(0) <= 0 ||
        input.shape(1) <= 0 ||
        input.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Qwen3.5 block input must have non-empty "
            "[batch,tokens,hidden] shape");
    }
    const int batch = input.shape(0);
    const int tokens = input.shape(1);
    if (positions) {
        validate_position_shape(*positions, tokens);
    }
    const int maximum_sequence =
        checked_int(
            config_.max_position_embeddings,
            "max_position_embeddings");
    if (position_offset < 0 ||
        position_offset > maximum_sequence - tokens) {
        throw std::runtime_error(
            "Qwen3.5 position range exceeds maximum sequence length");
    }
    if (use_cache) {
        if (!cache_) {
            if (position_offset != 0) {
                throw std::runtime_error(
                    "Qwen3.5 cache must start at position zero");
            }
            reset_cache(batch);
        }
        if (cache_batch_ != batch) {
            throw std::runtime_error(
                "Qwen3.5 KV cache batch mismatch");
        }
        if (position_offset != cache_->position()) {
            throw std::runtime_error(
                "Qwen3.5 KV cache position must be appended contiguously");
        }
    }

    const int query_heads =
        static_cast<int>(config_.num_attention_heads);
    const int kv_heads =
        static_cast<int>(config_.num_key_value_heads);
    const int head_dimension =
        static_cast<int>(config_.head_dim);
    const int attention_size =
        static_cast<int>(config_.attention_size());

    const auto normalized = attention_norm_(input);
    std::vector<array> projected;
    // As with the FFN gate/up pair, the single-row heterogeneous QKV kernel
    // is bandwidth-limited well below the individual packed GEMVs. Preserve
    // it for multi-row prefill only.
    if (detail::qwen35_use_grouped_projection_rows(
            normalized.size(),
            static_cast<int>(config_.hidden_size)) &&
        qkv_ &&
        qkv_->supports(normalized)) {
        projected = (*qkv_)(normalized);
    } else {
        projected = {
            query_(normalized),
            key_(normalized),
            value_(normalized),
        };
    }
    auto query_full = std::move(projected.at(0));
    auto key_full = std::move(projected.at(1));
    auto value_full = std::move(projected.at(2));

    array query_raw = query_full;
    std::optional<array> query_gate;
    if (config_.attention_output_gate) {
        const auto query_pair = mlx::core::reshape(
            query_full,
            Shape{
                batch,
                tokens,
                query_heads,
                head_dimension * 2,
            });
        auto parts = mlx::core::split(query_pair, 2, -1);
        query_raw = std::move(parts.at(0));
        query_gate = std::move(parts.at(1));
    } else {
        query_raw = mlx::core::reshape(
            query_full,
            Shape{
                batch,
                tokens,
                query_heads,
                head_dimension,
            });
    }

    auto query = mlx::core::transpose(
        query_raw,
        {0, 2, 1, 3});
    auto key = mlx::core::transpose(
        mlx::core::reshape(
            key_full,
            Shape{
                batch,
                tokens,
                kv_heads,
                head_dimension,
            }),
        {0, 2, 1, 3});
    auto value = mlx::core::transpose(
        mlx::core::reshape(
            value_full,
            Shape{
                batch,
                tokens,
                kv_heads,
                head_dimension,
            }),
        {0, 2, 1, 3});

    if (query_norm_) {
        query = (*query_norm_)(query);
    }
    if (key_norm_) {
        key = (*key_norm_)(key);
    }
    if (positions) {
        query = apply_rope(
            query,
            *positions,
            static_cast<int>(config_.rotary_dim),
            static_cast<float>(config_.rope_base),
            config_.rope_sections,
            config_.mrope_interleaved);
        key = apply_rope(
            key,
            *positions,
            static_cast<int>(config_.rotary_dim),
            static_cast<float>(config_.rope_base),
            config_.rope_sections,
            config_.mrope_interleaved);
    } else {
        query = apply_rope(
            query,
            static_cast<int>(config_.rotary_dim),
            static_cast<float>(config_.rope_base),
            position_offset);
        key = apply_rope(
            key,
            static_cast<int>(config_.rotary_dim),
            static_cast<float>(config_.rope_base),
            position_offset);
    }

    array key_cache = key;
    array value_cache = value;
    if (use_cache) {
        auto cached = cache_->append(key, value);
        key_cache = std::move(cached.first);
        value_cache = std::move(cached.second);
    }

    auto attended = scaled_dot_product_attention(
        query,
        key_cache,
        value_cache,
        true);
    attended = mlx::core::reshape(
        mlx::core::transpose(attended, {0, 2, 1, 3}),
        Shape{batch, tokens, attention_size});
    if (query_gate) {
        const auto gate = mlx::core::reshape(
            *query_gate,
            Shape{batch, tokens, attention_size});
        attended = attended * mlx::core::sigmoid(gate);
    }

    auto residual = input + output_(attended);
    residual = residual + ffn_(ffn_norm_(residual));
    return residual;
}

} // namespace mfq::metal
