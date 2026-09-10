#include "mlx_deepseek_v41_causal_lm.h"

#include "mlx_deepseek_v4_attention.h"
#include "mlx_eval_timing.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

array dense_array(const MfqContainer& model, const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" && record.dtype != "F16" &&
        record.dtype != "F32") {
        throw std::runtime_error(
            "DeepSeek-V4.1 requires dense control tensor " + name);
    }
    const auto mapped = model.map_record(name);
    return mlx::core::contiguous(
        load_dense_array(record.dtype, mapped.view()));
}

DeepseekV4RopeScaling rope_scaling(
    const DeepseekV41Config& config,
    bool enabled) {
    DeepseekV4RopeScaling result;
    result.enabled = enabled;
    result.type = enabled ? "yarn" : "";
    result.factor = config.rope_scaling.factor;
    result.beta_fast = config.rope_scaling.beta_fast;
    result.beta_slow = config.rope_scaling.beta_slow;
    result.original_max_position_embeddings =
        config.rope_scaling.original_max_position_embeddings;
    return result;
}

array slice_tokens(
    const array& value,
    int begin,
    int end) {
    Shape start(value.ndim(), 0);
    Shape stop = value.shape();
    start[1] = begin;
    stop[1] = end;
    return mlx::core::slice(value, start, stop);
}

} // namespace

MlxDeepseekV41Layer MlxDeepseekV41Layer::load(
    const MfqContainer& model,
    const DeepseekV41Config& config,
    int index,
    int max_context,
    std::pair<array, array> rope_base,
    std::pair<array, array> rope_compressed) {
    config.validate();
    if (index < 0 || index >= config.n_layers) {
        throw std::out_of_range(
            "DeepSeek-V4.1 layer index is out of range");
    }
    const auto name = [index](std::string_view suffix) {
        return DeepseekV41TensorNames::layer(
            static_cast<std::size_t>(index), suffix);
    };
    std::unique_ptr<MlxDeepseekV41Engram> engram;
    if (config.has_engram(index)) {
        engram = std::make_unique<MlxDeepseekV41Engram>(
            MlxDeepseekV41Engram::load(model, config, index));
    }
    return MlxDeepseekV41Layer(
        config,
        index,
        MlxDeepseekV41Attention::load(
            model,
            config,
            index,
            max_context,
            std::move(rope_base),
            std::move(rope_compressed)),
        MlxDeepseekV41Mhc::load(
            model,
            config,
            name("attention.mhc.pre"),
            name("attention.norm.weight")),
        MlxDeepseekV41Mhc::load(
            model,
            config,
            name("mlp.mhc.pre"),
            name("mlp.norm.weight")),
        MlxDeepseekV41Moe::load(model, config, name("mlp")),
        std::move(engram));
}

MlxDeepseekV41Layer::MlxDeepseekV41Layer(
    DeepseekV41Config config,
    int index,
    MlxDeepseekV41Attention attention,
    MlxDeepseekV41Mhc attention_mhc,
    MlxDeepseekV41Mhc ffn_mhc,
    MlxDeepseekV41Moe moe,
    std::unique_ptr<MlxDeepseekV41Engram> engram)
    : config_(std::move(config)),
      index_(index),
      attention_(std::move(attention)),
      attention_mhc_(std::move(attention_mhc)),
      ffn_mhc_(std::move(ffn_mhc)),
      moe_(std::move(moe)),
      engram_(std::move(engram)) {
    if ((engram_ != nullptr) != config_.has_engram(index_)) {
        throw std::runtime_error(
            "DeepSeek-V4.1 layer Engram schedule disagrees");
    }
}

MlxDeepseekV41LayerResult MlxDeepseekV41Layer::forward(
    const array& hidden,
    const array& previous_pre,
    const DeepseekV41EngramHashBatch* hashes,
    const std::optional<array>& image_mask,
    MlxDeepseekV41LayerState& state,
    MlxDeepseekV41SharedAttentionState& shared_attention,
    int pos0) const {
    if (hidden.ndim() != 4 || hidden.shape(2) != config_.hc_mult ||
        hidden.shape(3) != config_.hidden) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 layer hidden-state geometry disagrees");
    }
    auto value = hidden;
    if (engram_) {
        if (hashes == nullptr) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 Engram layer has no hash batch");
        }
        std::optional<array> participation;
        if (image_mask) {
            participation = mlx::core::logical_not(
                mlx::core::astype(*image_mask, mlx::core::bool_));
        }
        value = engram_->forward(value, *hashes, participation);
    }
    auto attention_input = mlx::core::mean(value, 2);

    const auto attention_residual = value;
    auto attention_mix = attention_mhc_.collapse(
        attention_residual, previous_pre);
    auto attention_branch = attention_.forward(
        attention_mix.branch,
        state.attention,
        shared_attention,
        pos0);
    value = attention_mhc_.expand(
        attention_branch,
        attention_residual,
        attention_mix.expansion);

    const auto ffn_residual = value;
    auto ffn_mix = ffn_mhc_.collapse(
        ffn_residual, attention_mix.next_pre);
    auto ffn_branch = moe_.forward(ffn_mix.branch, image_mask);
    value = ffn_mhc_.expand(
        ffn_branch,
        ffn_residual,
        ffn_mix.expansion);
    return {
        std::move(value),
        std::move(ffn_mix.next_pre),
        std::move(attention_input),
    };
}

std::vector<array> MlxDeepseekV41Layer::begin_speculative(
    MlxDeepseekV41LayerState& state,
    int confirmed_tokens,
    int total_tokens) const {
    return attention_.begin_speculative(
        state.attention, confirmed_tokens, total_tokens);
}

void MlxDeepseekV41Layer::commit_speculative(
    MlxDeepseekV41LayerState& state) const noexcept {
    attention_.commit_speculative(state.attention);
}

std::vector<array> MlxDeepseekV41Layer::rollback_speculative(
    MlxDeepseekV41LayerState& state,
    int accepted_drafts) const {
    return attention_.rollback_speculative(
        state.attention, accepted_drafts);
}

MlxDeepseekV41CausalLm MlxDeepseekV41CausalLm::load(
    const MfqContainer& model,
    int max_context) {
    auto config = DeepseekV41Config::from_mfq(model);
    config.validate();
    if (max_context <= 0 || max_context > config.max_position_embeddings ||
        max_context > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 runtime context length");
    }
    auto base_rope = deepseek_v4_yarn_tables(
        static_cast<int>(config.rope_head_dim),
        max_context,
        static_cast<float>(config.rope_theta),
        rope_scaling(config, false));
    auto compressed_rope = deepseek_v4_yarn_tables(
        static_cast<int>(config.rope_head_dim),
        max_context,
        static_cast<float>(config.compress_rope_theta),
        rope_scaling(config, true));
    std::vector<MlxDeepseekV41Layer> layers;
    layers.reserve(static_cast<std::size_t>(config.n_layers));
    for (int index = 0; index < config.n_layers; ++index) {
        layers.push_back(MlxDeepseekV41Layer::load(
            model,
            config,
            index,
            max_context,
            base_rope,
            compressed_rope));
    }
    auto embedding = MlxEmbedding::load(
        model, "model.token_embedding.weight");
    auto output = MlxLinear::load(model, "model.output.weight");
    std::optional<MlxDeepseekV41Vision> vision;
    if (config.has_vision()) {
        vision.emplace(MlxDeepseekV41Vision::load(model, config));
    }
    auto dspark = MlxDeepseekV41DSpark::load_if_present(
        model, config, embedding, output, max_context);
    return MlxDeepseekV41CausalLm(
        config,
        std::move(embedding),
        std::move(layers),
        dense_array(model, "model.output_norm.weight"),
        std::move(output),
        MlxDeepseekV41EngramHashState::load(model, config),
        max_context,
        std::move(vision),
        std::move(dspark));
}

MlxDeepseekV41CausalLm::MlxDeepseekV41CausalLm(
    DeepseekV41Config config,
    MlxEmbedding embedding,
    std::vector<MlxDeepseekV41Layer> layers,
    array output_norm,
    MlxLinear output,
    MlxDeepseekV41EngramHashState engram_hash,
    int max_context,
    std::optional<MlxDeepseekV41Vision> vision,
    std::optional<MlxDeepseekV41DSpark> dspark)
    : config_(std::move(config)),
      embedding_(std::move(embedding)),
      layers_(std::move(layers)),
      output_norm_(std::move(output_norm), static_cast<float>(config_.rms_eps)),
      output_(std::move(output)),
      engram_hash_(std::move(engram_hash)),
      vision_(std::move(vision)),
      dspark_(std::move(dspark)),
      max_context_(max_context) {
    config_.validate();
    if (layers_.size() != static_cast<std::size_t>(config_.n_layers) ||
        embedding_.vocabulary_size() != config_.vocab ||
        embedding_.hidden_size() != config_.hidden ||
        output_norm_.width() != config_.hidden ||
        output_.input_size() != config_.hidden ||
        output_.output_size() != config_.vocab ||
        vision_.has_value() != config_.has_vision() ||
        (dspark_.has_value() && !config_.has_dspark()) ||
        max_context_ <= 0 || max_context_ > config_.max_position_embeddings) {
        throw std::runtime_error(
            "DeepSeek-V4.1 text runtime geometry disagrees");
    }
}

array MlxDeepseekV41CausalLm::normalized_ids(
    const array& token_ids) const {
    auto ids = token_ids.ndim() == 1
        ? mlx::core::expand_dims(token_ids, 0)
        : token_ids;
    if (ids.ndim() != 2 || ids.shape(0) <= 0 || ids.shape(1) <= 0) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 token IDs must have nonempty [B,T] shape");
    }
    if (ids.dtype() != mlx::core::int32 &&
        ids.dtype() != mlx::core::int64) {
        ids = mlx::core::astype(ids, mlx::core::int32);
    }
    return ids;
}

void MlxDeepseekV41CausalLm::reset_cache(int batch) {
    if (batch <= 0) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 cache batch must be positive");
    }
    states_.clear();
    states_.reserve(layers_.size());
    for (int index = 0; index < static_cast<int>(layers_.size()); ++index) {
        states_.push_back({MlxDeepseekV41AttentionState::allocate(
            config_, index, batch, max_context_)});
    }
    engram_hash_.reset(batch);
    dspark_state_.reset();
    if (dspark_ && mtp_context_requested_) {
        dspark_state_.emplace(
            dspark_->make_state(batch, mlx::core::float16));
    }
    speculative_engram_snapshot_.reset();
    speculative_token_ids_.reset();
    speculative_cache_start_ = -1;
    cache_position_ = 0;
    cache_batch_ = batch;
}

void MlxDeepseekV41CausalLm::clear_cache() noexcept {
    states_.clear();
    engram_hash_.clear();
    dspark_state_.reset();
    speculative_engram_snapshot_.reset();
    speculative_token_ids_.reset();
    speculative_cache_start_ = -1;
    mtp_context_requested_ = false;
    cache_position_ = 0;
    cache_batch_ = 0;
}

array MlxDeepseekV41CausalLm::forward_impl(
    const array& raw_token_ids,
    const std::optional<array>& image_mask,
    bool reuse_cache,
    const std::optional<array>& input_embeddings,
    array* dspark_hidden,
    bool update_dspark) {
    auto token_ids = normalized_ids(raw_token_ids);
    const int batch = token_ids.shape(0);
    const int tokens = token_ids.shape(1);
    if (!reuse_cache) reset_cache(batch);
    if (cache_batch_ != batch || states_.size() != layers_.size()) {
        throw std::runtime_error(
            "DeepSeek-V4.1 decode cache is not initialized for this batch");
    }
    if (cache_position_ + tokens > max_context_) {
        throw std::out_of_range(
            "DeepSeek-V4.1 decode exceeds the configured context");
    }
    if (image_mask && image_mask->shape() != token_ids.shape()) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 image mask shape disagrees with token IDs");
    }
    if (input_embeddings &&
        input_embeddings->shape() !=
            Shape{batch, tokens, static_cast<int>(config_.hidden)}) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 input embedding shape disagrees with token IDs");
    }
    std::optional<array> participation;
    if (image_mask) {
        participation = mlx::core::logical_not(
            mlx::core::astype(*image_mask, mlx::core::bool_));
    }
    auto hashes = engram_hash_.forward(
        token_ids,
        cache_position_,
        participation,
        true);
    auto hidden = input_embeddings
        ? *input_embeddings
        : embedding_(token_ids, mlx::core::float16);
    hidden = mlx::core::broadcast_to(
        mlx::core::expand_dims(std::move(hidden), 2),
        Shape{
            batch,
            tokens,
            static_cast<int>(config_.hc_mult),
            static_cast<int>(config_.hidden),
        });
    auto previous_pre = MlxDeepseekV41Mhc::identity_pre(batch, tokens);
    MlxDeepseekV41SharedAttentionState shared_attention;
    std::vector<std::optional<array>> target_hiddens;
    array automatic_dspark_hidden(0.0f);
    auto* selected_dspark_hidden = dspark_hidden;
    if (selected_dspark_hidden == nullptr && dspark_state_ && update_dspark) {
        selected_dspark_hidden = &automatic_dspark_hidden;
    }
    if (selected_dspark_hidden != nullptr) {
        target_hiddens.resize(config_.dspark_target_layer_ids.size());
    }
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        auto result = layers_[index].forward(
            hidden,
            previous_pre,
            &hashes,
            image_mask,
            states_[index],
            shared_attention,
            cache_position_);
        hidden = std::move(result.hidden);
        previous_pre = std::move(result.next_pre);
        if (selected_dspark_hidden != nullptr) {
            for (std::size_t target = 0;
                 target < config_.dspark_target_layer_ids.size();
                 ++target) {
                if (config_.dspark_target_layer_ids[target] ==
                    static_cast<std::int64_t>(index)) {
                    target_hiddens[target] = result.attention_input;
                }
            }
        }
    }
    auto collapsed = mlx::core::sum(
        mlx::core::expand_dims(
            mlx::core::astype(previous_pre, mlx::core::float32), -1) *
            mlx::core::astype(hidden, mlx::core::float32),
        2);
    collapsed = output_norm_(
        mlx::core::astype(std::move(collapsed), hidden.dtype()));
    if (selected_dspark_hidden != nullptr) {
        std::vector<array> captured;
        captured.reserve(target_hiddens.size());
        for (auto& target : target_hiddens) {
            if (!target) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 DSpark target layer was not captured");
            }
            captured.push_back(std::move(*target));
        }
        *selected_dspark_hidden = captured.size() == 1
            ? std::move(captured.front())
            : mlx::core::concatenate(std::move(captured), -1);
    }
    if (dspark_state_ && selected_dspark_hidden == &automatic_dspark_hidden) {
        dspark_->append_context(
            automatic_dspark_hidden,
            *dspark_state_,
            cache_position_);
    }
    cache_position_ += tokens;
    return output_(collapsed);
}

void MlxDeepseekV41CausalLm::begin_speculative_target(
    const array& token_ids,
    int confirmed_tokens) {
    if (token_ids.ndim() != 2 || token_ids.shape(0) != cache_batch_ ||
        token_ids.shape(1) <= confirmed_tokens || confirmed_tokens <= 0 ||
        speculative_engram_snapshot_ || speculative_token_ids_ ||
        speculative_cache_start_ >= 0) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 target cache transaction");
    }
    const int total_tokens = token_ids.shape(1);
    std::vector<array> checkpoints;
    checkpoints.reserve(states_.size() * 3);
    try {
        for (std::size_t index = 0; index < layers_.size(); ++index) {
            auto layer_checkpoints = layers_[index].begin_speculative(
                states_[index], confirmed_tokens, total_tokens);
            checkpoints.insert(
                checkpoints.end(),
                std::make_move_iterator(layer_checkpoints.begin()),
                std::make_move_iterator(layer_checkpoints.end()));
        }
        speculative_engram_snapshot_ = engram_hash_.snapshot();
        speculative_token_ids_ = token_ids;
        speculative_cache_start_ = cache_position_;
        detail::eval_with_timing(std::move(checkpoints));
    } catch (...) {
        abort_speculative_target();
        throw;
    }
}

void MlxDeepseekV41CausalLm::commit_speculative_target() noexcept {
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        layers_[index].commit_speculative(states_[index]);
    }
    speculative_engram_snapshot_.reset();
    speculative_token_ids_.reset();
    speculative_cache_start_ = -1;
}

void MlxDeepseekV41CausalLm::rollback_speculative_target(
    int accepted_drafts,
    int draft_tokens) {
    if (!speculative_engram_snapshot_ || !speculative_token_ids_ ||
        speculative_cache_start_ < 0 || draft_tokens <= 0 ||
        accepted_drafts < 0 || accepted_drafts >= draft_tokens ||
        speculative_token_ids_->shape(1) != draft_tokens + 1) {
        throw std::runtime_error(
            "invalid DeepSeek-V4.1 target cache rollback");
    }
    const int start = speculative_cache_start_;
    const int keep = accepted_drafts + 1;
    std::vector<array> restored;
    restored.reserve(states_.size() * 3);
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        auto layer_restored = layers_[index].rollback_speculative(
            states_[index], accepted_drafts);
        restored.insert(
            restored.end(),
            std::make_move_iterator(layer_restored.begin()),
            std::make_move_iterator(layer_restored.end()));
    }
    engram_hash_.restore(std::move(*speculative_engram_snapshot_));
    auto committed_ids = slice_tokens(*speculative_token_ids_, 0, keep);
    (void)engram_hash_.forward(
        committed_ids, start, std::nullopt, true);
    cache_position_ = start + keep;
    speculative_engram_snapshot_.reset();
    speculative_token_ids_.reset();
    speculative_cache_start_ = -1;
    detail::eval_with_timing(std::move(restored));
}

void MlxDeepseekV41CausalLm::abort_speculative_target() noexcept {
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        layers_[index].commit_speculative(states_[index]);
    }
    if (speculative_engram_snapshot_) {
        try {
            engram_hash_.restore(std::move(*speculative_engram_snapshot_));
        } catch (...) {
            engram_hash_.clear();
        }
    }
    speculative_engram_snapshot_.reset();
    speculative_token_ids_.reset();
    speculative_cache_start_ = -1;
}

array MlxDeepseekV41CausalLm::forward(
    const array& token_ids,
    bool use_cache) {
    return forward_impl(token_ids, std::nullopt, use_cache);
}

array MlxDeepseekV41CausalLm::prefill(
    const array& raw_token_ids,
    int chunk_size,
    bool full_logits) {
    auto token_ids = normalized_ids(raw_token_ids);
    if (chunk_size <= 0) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 prefill chunk size must be positive");
    }
    const int batch = token_ids.shape(0);
    const int tokens = token_ids.shape(1);
    if (tokens > max_context_) {
        throw std::out_of_range(
            "DeepSeek-V4.1 prefill exceeds the configured context");
    }
    reset_cache(batch);
    std::vector<array> chunks;
    if (full_logits) {
        chunks.reserve(static_cast<std::size_t>(
            (tokens + chunk_size - 1) / chunk_size));
    }
    std::optional<array> last;
    for (int begin = 0; begin < tokens; begin += chunk_size) {
        const int end = std::min(tokens, begin + chunk_size);
        auto logits = forward_impl(
            slice_tokens(token_ids, begin, end),
            std::nullopt,
            true);
        if (full_logits) chunks.push_back(std::move(logits));
        else last = std::move(logits);
    }
    if (full_logits) {
        return chunks.size() == 1
            ? std::move(chunks.front())
            : mlx::core::concatenate(std::move(chunks), 1);
    }
    auto final = slice_tokens(*last, last->shape(1) - 1, last->shape(1));
    return mlx::core::squeeze(std::move(final), 1);
}

array MlxDeepseekV41CausalLm::decode(const array& token_ids) {
    if (cache_batch_ == 0) {
        throw std::runtime_error(
            "DeepSeek-V4.1 decode requires a prefill cache");
    }
    return forward_impl(token_ids, std::nullopt, true);
}

array MlxDeepseekV41CausalLm::prefill_multimodal(
    const std::vector<std::int64_t>& token_ids,
    const std::vector<MlxDeepseekV41ImageInput>& images) {
    if (!vision_ || token_ids.empty() || images.empty() ||
        token_ids.size() > static_cast<std::size_t>(max_context_)) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 multimodal prefill input is invalid");
    }
    std::vector<std::int32_t> ids;
    ids.reserve(token_ids.size());
    for (const auto token : token_ids) {
        if (token < 0 || token >= config_.vocab) {
            throw std::out_of_range(
                "DeepSeek-V4.1 multimodal token is outside the vocabulary");
        }
        ids.push_back(static_cast<std::int32_t>(token));
    }
    const auto token_array = array(
        ids.begin(), Shape{1, static_cast<int>(ids.size())});
    auto embeddings = vision_->embed_prompt(
        token_ids, images, embedding_, mlx::core::float16);
    auto mask = vision_->image_mask(
        static_cast<int>(token_ids.size()), images);
    auto logits = forward_impl(
        token_array,
        mask,
        false,
        std::move(embeddings));
    auto final = slice_tokens(logits, logits.shape(1) - 1, logits.shape(1));
    return mlx::core::squeeze(std::move(final), 1);
}

MlxDeepseekV41DSparkDraft MlxDeepseekV41CausalLm::draft_mtp(
    const array& anchor_ids,
    const MlxMtpTokenSelector& select_token,
    int width) {
    if (!dspark_ || !dspark_state_ ||
        dspark_state_->position() != cache_position_) {
        throw std::runtime_error(
            "DeepSeek-V4.1 DSpark state is not synchronized with the target");
    }
    return dspark_->draft(
        anchor_ids, *dspark_state_, select_token, width);
}

std::int32_t MlxDeepseekV41CausalLm::generate_from_prefill(
    array logits,
    const std::vector<std::int64_t>& history,
    const MlxSamplingParams& sampling,
    std::int32_t limit,
    const std::function<bool(std::int64_t)>& callback,
    const MfqTokenConstraintPtr& token_constraint) {
    const int vocab = static_cast<int>(config_.vocab);
    std::optional<array> counts;
    if (sampling.has_penalties()) {
        std::vector<std::int32_t> values;
        values.reserve(history.size());
        for (const auto token : history) {
            values.push_back(static_cast<std::int32_t>(token));
        }
        counts = sample_token_counts_add(
            mlx::core::zeros(Shape{vocab}, mlx::core::int32),
            array(values.begin(), Shape{1, static_cast<int>(values.size())},
                  mlx::core::int32));
    }
    const bool mtp_active = dspark_.has_value() &&
        dspark_state_.has_value() && sampling.enable_mtp &&
        !token_constraint && limit > 1;
    last_mtp_stats_ = {
        dspark_.has_value(),
        mtp_active,
        0,
        0,
        0,
    };
    if (mtp_active) {
        MlxMtpEngineCallbacks callbacks;
        callbacks.target_cache_position = [this] {
            return cache_position_;
        };
        callbacks.prepare_draft =
            [&](const MlxMtpDraftContext& context,
                const MlxMtpTokenSelector& select_token) {
                if (!context.initial) {
                    const int expected_width = static_cast<int>(
                        config_.hidden *
                        static_cast<std::int64_t>(
                            config_.dspark_target_layer_ids.size()));
                    auto committed_hidden = mlx_mtp_committed_hidden(
                        context, 1, expected_width);
                    dspark_->append_context(
                        committed_hidden,
                        *dspark_state_,
                        context.target_cache_start);
                }
                if (context.requested_depth == 0) {
                    return;
                }
                const array anchor_ids(
                    {context.pending_token},
                    Shape{1, 1},
                    mlx::core::int32);
                (void)draft_mtp(
                    anchor_ids, select_token, context.requested_depth);
            };
        callbacks.verify_target =
            [&](std::int32_t pending_token,
                const array& draft_tokens,
                int draft_count) {
                auto verify_ids = mlx_mtp_verification_ids(
                    pending_token, draft_tokens, draft_count);
                if (draft_count > 0) {
                    begin_speculative_target(verify_ids, 1);
                }
                try {
                    array target_hidden(0.0f);
                    auto verified_logits = forward_impl(
                        verify_ids,
                        std::nullopt,
                        true,
                        std::nullopt,
                        &target_hidden,
                        false);
                    return MlxMtpTargetBatch{
                        mlx::core::reshape(
                            verified_logits,
                            Shape{draft_count + 1, vocab}),
                        std::move(target_hidden),
                    };
                } catch (...) {
                    clear_cache();
                    throw;
                }
            };
        callbacks.resolve_target =
            [&](int accepted_drafts, int draft_count) {
                if (draft_count == 0) {
                    return;
                }
                if (accepted_drafts == draft_count) {
                    commit_speculative_target();
                    return;
                }
                rollback_speculative_target(
                    accepted_drafts, draft_count);
            };
        callbacks.plain_decode = [&](std::int32_t pending_token) {
            const array token_ids(
                {pending_token}, Shape{1, 1}, mlx::core::int32);
            return mlx_last_token_logits(
                forward_impl(
                    token_ids,
                    std::nullopt,
                    true,
                    std::nullopt,
                    nullptr,
                    false),
                vocab);
        };
        return run_mlx_mtp_generation(
            MlxMtpEngineRequest{
                vocab,
                limit,
                max_context_,
                dspark_->block_size(),
                logits,
                sampling,
                counts,
                std::span<const std::int64_t>(config_.eos_token_ids),
                callback,
            },
            callbacks,
            last_mtp_stats_);
    }

    MlxSampler sampler(sampling);
    std::int32_t generated = 0;
    while (generated < limit) {
        auto sampled = counts
            ? sampler.sample(logits, *counts)
            : sampler.sample(logits);
        sampled.eval();
        auto token = sampled.data<std::int32_t>()[0];
        if (token < 0 || token >= vocab) {
            throw std::runtime_error(
                "DeepSeek-V4.1 sampler returned an invalid token");
        }
        if (token_constraint && token_constraint->allows &&
            !token_constraint->allows(token)) {
            auto adjusted = counts
                ? sampler.apply_penalties(logits, *counts)
                : logits;
            adjusted = mlx::core::contiguous(
                mlx::core::astype(adjusted, mlx::core::float32));
            adjusted.eval();
            std::vector<float> masked(
                adjusted.data<float>(), adjusted.data<float>() + vocab);
            token_constraint->apply(masked.data(), masked.size());
            sampled = sampler.sample(array(
                masked.begin(), Shape{1, vocab}, mlx::core::float32));
            sampled.eval();
            token = sampled.data<std::int32_t>()[0];
            if (token < 0 || token >= vocab ||
                !token_constraint->allows(token)) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 constrained sampler returned an invalid token");
            }
        }
        if (token_constraint && token_constraint->accept) {
            token_constraint->accept(token);
        }
        const array token_ids(
            {token}, Shape{1, 1}, mlx::core::int32);
        if (counts) {
            *counts = sample_token_counts_add(*counts, token_ids);
        }
        ++generated;
        const bool delivered = !callback || callback(token);
        if (!delivered ||
            mlx_token_in_set(config_.eos_token_ids, token) ||
            generated == limit) {
            break;
        }
        logits = mlx_last_token_logits(decode(token_ids), vocab);
    }
    return generated;
}

std::int32_t MlxDeepseekV41CausalLm::generate(
    const std::vector<std::int64_t>& prompt,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint,
    std::optional<std::size_t>,
    int prefill_chunk_size) {
    if (prompt.empty() || max_tokens < 0 || prefill_chunk_size <= 0) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 generation request");
    }
    mlx_validate_token_set(prompt, static_cast<int>(config_.vocab));
    if (prompt.size() > static_cast<std::size_t>(max_context_)) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 prompt exceeds context capacity");
    }
    last_mtp_stats_ = {supports_mtp(), false, 0, 0, 0};
    if (max_tokens == 0) {
        mtp_context_requested_ = false;
        reset_cache(1);
        return 0;
    }
    std::vector<std::int32_t> values;
    values.reserve(prompt.size());
    for (const auto token : prompt) {
        values.push_back(static_cast<std::int32_t>(token));
    }
    const auto started = std::chrono::steady_clock::now();
    mtp_context_requested_ = supports_mtp() && sampling.enable_mtp &&
        !token_constraint && max_tokens > 1;
    array logits(0.0f);
    try {
        logits = prefill(
            array(values.begin(), Shape{1, static_cast<int>(values.size())},
                  mlx::core::int32),
            prefill_chunk_size,
            false);
    } catch (...) {
        mtp_context_requested_ = false;
        throw;
    }
    mtp_context_requested_ = false;
    logits.eval();
    mlx::core::synchronize();
    if (prefill_callback) {
        prefill_callback(
            prompt.size(),
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
    }
    const auto limit = std::min<std::int32_t>(
        max_tokens,
        max_context_ - static_cast<int>(prompt.size()) + 1);
    return generate_from_prefill(
        std::move(logits), prompt, sampling, limit, callback,
        token_constraint);
}

std::int32_t MlxDeepseekV41CausalLm::generate_multimodal(
    const std::vector<std::int64_t>& prompt,
    const std::vector<MlxDeepseekV41ImageInput>& images,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint) {
    if (prompt.empty() || images.empty() || max_tokens < 0) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 multimodal generation request");
    }
    mlx_validate_token_set(prompt, static_cast<int>(config_.vocab));
    last_mtp_stats_ = {supports_mtp(), false, 0, 0, 0};
    if (max_tokens == 0) {
        mtp_context_requested_ = false;
        reset_cache(1);
        return 0;
    }
    const auto started = std::chrono::steady_clock::now();
    mtp_context_requested_ = supports_mtp() && sampling.enable_mtp &&
        !token_constraint && max_tokens > 1;
    array logits(0.0f);
    try {
        logits = prefill_multimodal(prompt, images);
    } catch (...) {
        mtp_context_requested_ = false;
        throw;
    }
    mtp_context_requested_ = false;
    logits.eval();
    mlx::core::synchronize();
    if (prefill_callback) {
        prefill_callback(
            prompt.size(),
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
    }
    const auto limit = std::min<std::int32_t>(
        max_tokens,
        max_context_ - static_cast<int>(prompt.size()) + 1);
    return generate_from_prefill(
        std::move(logits), prompt, sampling, limit, callback,
        token_constraint);
}

MlxDeepseekV41TextSessionState
MlxDeepseekV41CausalLm::capture_text_session_state(
    const std::vector<std::int64_t>&) const {
    throw std::runtime_error(
        "DeepSeek-V4.1 persistent text-session snapshots are not enabled yet");
}

void MlxDeepseekV41CausalLm::restore_text_session_state(
    const MlxDeepseekV41TextSessionState&) {
    throw std::runtime_error(
        "DeepSeek-V4.1 persistent text-session snapshots are not enabled yet");
}

} // namespace mfq::metal
