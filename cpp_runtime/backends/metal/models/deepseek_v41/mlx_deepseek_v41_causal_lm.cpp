#include "mlx_deepseek_v41_causal_lm.h"

#include "mlx_eval_timing.h"
#include "mlx_legacy_tensor_compat.h"
#include "mlx_transformer.h"
#include "mfe_expert_store.h"

#include <algorithm>
#include <array>
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

MlxYarnScaling rope_scaling(
    const DeepseekV41Config& config,
    bool enabled) {
    MlxYarnScaling result;
    result.enabled = enabled;
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
    std::pair<array, array> rope_compressed,
    std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache,
    std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache,
    std::shared_ptr<MlxTqExpertCache> tq_expert_cache) {
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
        MlxDeepseekV41Moe::load(
            model,
            config,
            name("mlp"),
            false,
            std::move(ssd_expert_cache),
            std::move(mfe_offload_cache),
            static_cast<std::size_t>(index), std::move(tq_expert_cache)),
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
    int pos0,
    MlxSsdPrefetchedExpertLayer* prefetched,
    bool ced_replay,
    MlxMfeCachedRoute* deferred_route) const {
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

    auto attention_residual = value;
    auto attention_mix = attention_mhc_.collapse(
        attention_residual, previous_pre);
    std::optional<array> encoder_source;
    if (ced_replay && index_ == config_.n_layers / 2) {
        const int tokens = value.shape(1);
        const int tail = static_cast<int>(config_.sliding_window);
        const int begin = tokens - tail;
        if (begin <= 0 || image_mask || engram_) {
            throw std::invalid_argument("invalid DeepSeek-V4.1 CED boundary replay");
        }
        encoder_source = attention_mix.branch;
        attention_mix.branch = slice_tokens(attention_mix.branch, begin, tokens);
        attention_mix.next_pre = slice_tokens(attention_mix.next_pre, begin, tokens);
        attention_mix.expansion.pre = slice_tokens(attention_mix.expansion.pre, begin, tokens);
        attention_mix.expansion.post = slice_tokens(attention_mix.expansion.post, begin, tokens);
        attention_mix.expansion.combination = slice_tokens(attention_mix.expansion.combination, begin, tokens);
        attention_residual = slice_tokens(attention_residual, begin, tokens);
        attention_input = slice_tokens(attention_input, begin, tokens);
        pos0 += begin;
    }
    auto attention_branch = attention_.forward(
        attention_mix.branch,
        state.attention,
        shared_attention,
        pos0,
        ced_replay,
        encoder_source ? &*encoder_source : nullptr);
    auto ffn_residual = attention_residual;
    auto ffn_mix = [&] {
        if (ced_replay && index_ + 1 == config_.n_layers) {
            ffn_residual = attention_mhc_.expand(attention_branch, attention_residual,
                attention_mix.expansion);
            if (ffn_residual.shape(1) > 1) {
                const int last = ffn_residual.shape(1) - 1;
                ffn_residual = slice_tokens(ffn_residual, last, last + 1);
                attention_mix.next_pre = slice_tokens(attention_mix.next_pre, last, last + 1);
            }
            return ffn_mhc_.collapse(ffn_residual, attention_mix.next_pre);
        }
        return ffn_mhc_.collapse_after(attention_branch, attention_residual,
            attention_mix.expansion, attention_mix.next_pre, ffn_residual);
    }();
    auto ffn_branches = moe_.forward(
        ffn_mix.branch, image_mask, prefetched, deferred_route);
    value = ffn_mhc_.expand_sum(
        ffn_branches.routed,
        ffn_branches.shared,
        ffn_residual,
        ffn_mix.expansion);
    return {
        std::move(value),
        std::move(ffn_mix.next_pre),
        std::move(attention_input),
    };
}

void MlxDeepseekV41Layer::append_ced_encoder_context(
    const array& hidden, const array& previous_pre,
    MlxDeepseekV41LayerState& state, int pos0) const {
    if (index_ != config_.n_layers / 2 || hidden.shape(1) <= config_.sliding_window)
        throw std::invalid_argument("invalid DeepSeek-V4.1 deferred Decoder boundary");
    auto mixed = attention_mhc_.collapse(hidden, previous_pre);
    const auto& source = mixed.branch;
    const int begin = source.shape(1) - static_cast<int>(config_.sliding_window);
    auto local = state.attention.local_kv;
    MlxDeepseekV41SharedAttentionState shared;
    (void)attention_.forward(slice_tokens(source, begin, source.shape(1)),
        state.attention, shared, pos0 + begin, true, &source);
    state.attention.local_kv = std::move(local);
    detail::eval_with_timing({*state.attention.compressed_kv, *state.attention.index_k});
}

std::optional<MlxQuantizedRowStoreStats>
MlxDeepseekV41Layer::engram_ssd_stats() const {
    return engram_
        ? std::optional<MlxQuantizedRowStoreStats>(engram_->ssd_stats())
        : std::nullopt;
}

void MlxDeepseekV41Layer::prefetch_engram(
    DeepseekV41EngramHashBatch& hashes) const {
    if (engram_) engram_->prefetch(hashes);
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

std::vector<array> MlxDeepseekV41Layer::begin_route_replay(
    MlxDeepseekV41LayerState& state,
    int tokens) const {
    return attention_.begin_speculative(state.attention, 0, tokens);
}

void MlxDeepseekV41Layer::commit_route_replay(
    MlxDeepseekV41LayerState& state) const noexcept {
    attention_.commit_speculative(state.attention);
}

std::vector<array> MlxDeepseekV41Layer::rollback_route_replay(
    MlxDeepseekV41LayerState& state) const {
    return attention_.rollback_speculative(state.attention, 0);
}

MlxDeepseekV41CausalLm MlxDeepseekV41CausalLm::load(
    const MfqContainer& model,
    int max_context,
    std::optional<std::size_t> expert_cache_bytes) {
    auto config = DeepseekV41Config::from_mfq(
        model, effective_model_graph(model));
    config.validate();
    if (max_context <= 0 || max_context > config.max_position_embeddings ||
        max_context > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 runtime context length");
    }
    auto base_rope = mlx_yarn_tables(
        static_cast<int>(config.rope_head_dim),
        max_context,
        static_cast<float>(config.rope_theta),
        rope_scaling(config, false));
    auto compressed_rope = mlx_yarn_tables(
        static_cast<int>(config.rope_head_dim),
        max_context,
        static_cast<float>(config.compress_rope_theta),
        rope_scaling(config, true));
    std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache;
    std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache;
    std::shared_ptr<MlxTqExpertCache> tq_expert_cache;
    if (expert_cache_bytes.has_value() && *expert_cache_bytes > 0) {
        std::vector<std::string> prefixes;
        std::vector<std::size_t> experts_per_layer;
        prefixes.reserve(static_cast<std::size_t>(
            config.n_layers + config.n_mtp_layers));
        experts_per_layer.reserve(prefixes.capacity());
        for (std::int64_t layer = 0; layer < config.n_layers; ++layer) {
            prefixes.push_back("model.block." + std::to_string(layer));
            experts_per_layer.push_back(
                static_cast<std::size_t>(config.n_experts));
        }
        if (config.has_dspark() && model.contains("predictor.stage.0.main_projection.weight")) {
            for (std::int64_t stage = 0;
                 stage < config.n_mtp_layers; ++stage) {
                prefixes.push_back(
                    "predictor.stage." + std::to_string(stage));
                experts_per_layer.push_back(
                    static_cast<std::size_t>(config.dspark_n_experts));
            }
        }
        constexpr std::size_t prefill_buffers_minimum =
            std::size_t{7} << 30;
        if (model.contains("model.block.0.mlp.experts.gate.weight")
            && model.record("model.block.0.mlp.experts.gate.weight").dtype == "TQ") {
            tq_expert_cache = std::make_shared<MlxTqExpertCache>(model, prefixes, *expert_cache_bytes, 8);
        } else try {
            ssd_expert_cache = std::make_shared<MlxMoeSsdExpertCache>(
                model,
                std::move(prefixes),
                static_cast<std::size_t>(config.hidden),
                static_cast<std::size_t>(config.moe_inter),
                std::move(experts_per_layer),
                *expert_cache_bytes,
                8,
                *expert_cache_bytes >= prefill_buffers_minimum);
        } catch (const MlxMfeMxfp4Unsupported&) {
            mfe_offload_cache = std::make_shared<MlxMfeOffloadCache>(
                model, *expert_cache_bytes);
        }
    }
    std::vector<MlxDeepseekV41Layer> layers;
    layers.reserve(static_cast<std::size_t>(config.n_layers));
    for (int index = 0; index < config.n_layers; ++index) {
        layers.push_back(MlxDeepseekV41Layer::load(
            model,
            config,
            index,
            max_context,
            base_rope,
            compressed_rope,
            ssd_expert_cache,
            mfe_offload_cache, tq_expert_cache));
    }
    auto embedding = MlxEmbedding::load(
        model, "model.token_embedding.weight");
    auto output = MlxLinear::load(model, "model.output.weight");
    std::optional<MlxDeepseekV41Vision> vision;
    if (config.has_vision()) {
        vision.emplace(MlxDeepseekV41Vision::load(model, config));
    }
    auto dspark = MlxDeepseekV41DSpark::load_if_present(
        model,
        config,
        embedding,
        output,
        max_context,
        ssd_expert_cache,
        mfe_offload_cache,
        static_cast<std::size_t>(config.n_layers), tq_expert_cache);
    return MlxDeepseekV41CausalLm(
        config,
        std::move(embedding),
        std::move(layers),
        dense_array(model, "model.output_norm.weight"),
        std::move(output),
        MlxDeepseekV41EngramHashState::load(model, config),
        max_context,
        std::move(ssd_expert_cache),
        std::move(mfe_offload_cache),
        std::move(vision),
        std::move(dspark),
        std::make_shared<MfqContainer>(model), std::move(tq_expert_cache));
}

MlxDeepseekV41CausalLm::MlxDeepseekV41CausalLm(
    DeepseekV41Config config,
    MlxEmbedding embedding,
    std::vector<MlxDeepseekV41Layer> layers,
    array output_norm,
    MlxLinear output,
    MlxDeepseekV41EngramHashState engram_hash,
    int max_context,
    std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache,
    std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache,
    std::optional<MlxDeepseekV41Vision> vision,
    std::optional<MlxDeepseekV41DSpark> dspark,
    std::shared_ptr<MfqContainer> source,
    std::shared_ptr<MlxTqExpertCache> tq_expert_cache)
    : config_(std::move(config)),
      embedding_(std::move(embedding)),
      layers_(std::move(layers)),
      output_norm_(std::move(output_norm), static_cast<float>(config_.rms_eps)),
      output_(std::move(output)),
      engram_hash_(std::move(engram_hash)),
      ssd_expert_cache_(std::move(ssd_expert_cache)),
      mfe_offload_cache_(std::move(mfe_offload_cache)),
      tq_expert_cache_(std::move(tq_expert_cache)),
      source_(std::move(source)),
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
        (int(bool(ssd_expert_cache_)) + int(bool(mfe_offload_cache_)) + int(bool(tq_expert_cache_)) > 1) ||
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
    const bool reuse_storage =
        cache_batch_ == batch && states_.size() == layers_.size();
    if (reuse_storage) {
        for (auto& state : states_) state.attention.reset();
    } else {
        states_.clear();
        states_.reserve(layers_.size());
        for (int index = 0; index < static_cast<int>(layers_.size()); ++index) {
            states_.push_back({MlxDeepseekV41AttentionState::allocate(
                config_, index, batch, max_context_)});
        }
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
    stable_cache_tokens_.clear();
    stable_dspark_state_.reset();
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
    stable_cache_tokens_.clear();
    stable_dspark_state_.reset();
    cache_position_ = 0;
    cache_batch_ = 0;
}

void MlxDeepseekV41CausalLm::materialize_states(
    const std::vector<MlxDeepseekV41LayerState>& states) const {
    std::vector<array> arrays;
    arrays.reserve(states.size() * 5);
    for (const auto& layer : states) {
        const auto& state = layer.attention;
        arrays.push_back(state.local_kv);
        if (state.compressed_kv) arrays.push_back(*state.compressed_kv);
        if (state.index_k) arrays.push_back(*state.index_k);
        if (state.partial_kv) arrays.push_back(*state.partial_kv);
        if (state.partial_score) arrays.push_back(*state.partial_score);
    }
    detail::eval_with_timing(std::move(arrays));
}

namespace {

void materialize_deepseek_v41_dspark_state(
    const MlxDeepseekV41DSparkState& state) {
    std::vector<array> rings;
    rings.reserve(state.stages());
    for (std::size_t stage = 0; stage < state.stages(); ++stage) {
        rings.push_back(state.ring(stage));
    }
    if (!rings.empty()) detail::eval_with_timing(std::move(rings));
}

} // namespace

std::size_t
MlxDeepseekV41CausalLm::expert_cache_limit_bytes() const noexcept {
    if (tq_expert_cache_) return tq_expert_cache_->limit_bytes();
    if (ssd_expert_cache_) {
        return ssd_expert_cache_->cache_limit_bytes();
    }
    return mfe_offload_cache_
        ? mfe_offload_cache_->cache_limit_bytes()
        : 0;
}

std::size_t MlxDeepseekV41CausalLm::resident_full_expert_bytes() const noexcept {
    std::size_t bytes = dspark_ ? dspark_->resident_routed_bytes() : 0;
    for (const auto& layer : layers_) bytes += layer.resident_routed_bytes();
    return bytes;
}

std::size_t MlxDeepseekV41CausalLm::dynamic_weight_bytes() const noexcept {
    if (tq_expert_cache_) return tq_expert_cache_->resident_bytes();
    return mfe_offload_cache_ ? mfe_offload_cache_->resident_packed_bytes() : 0;
}

std::size_t MlxDeepseekV41CausalLm::reclaimable_expert_bytes() const noexcept {
    if (tq_expert_cache_) return tq_expert_cache_->resident_bytes();
    return ssd_expert_cache_ ? ssd_expert_cache_->cache_limit_bytes()
        : resident_full_expert_bytes() + dynamic_weight_bytes();
}

std::size_t MlxDeepseekV41CausalLm::ssd_expert_payload_bytes() const noexcept {
    if (tq_expert_cache_) return tq_expert_cache_->backing_bytes();
    if (mfe_offload_cache_) return mfe_offload_cache_->backing_payload_bytes();
    if (!ssd_expert_cache_ || !source_) return 0;
    std::size_t bytes = 0;
    for (const auto& [name, record] : source_->records()) {
        if (name.find(".experts.") != std::string::npos && record.dtype == "MFE") {
            bytes += record.nbytes;
        }
    }
    return bytes;
}

std::optional<std::size_t> MlxDeepseekV41CausalLm::trim_preallocated_expert_cache(std::size_t bytes) {
    if (tq_expert_cache_) return tq_expert_cache_->trim(bytes);
    if (source_ && source_->contains("model.block.0.mlp.experts.gate.weight")
        && source_->record("model.block.0.mlp.experts.gate.weight").dtype == "TQ")
        return set_expert_cache_limit(bytes);
    return std::nullopt;
}

std::size_t MlxDeepseekV41CausalLm::set_expert_cache_limit(std::size_t bytes) {
    if (tq_expert_cache_) {
        return tq_expert_cache_->set_limit(bytes);
    }
    const auto before = mlx::core::get_active_memory();
    if (!ssd_expert_cache_ && !mfe_offload_cache_ && source_
        && source_->contains("model.block.0.mlp.experts.gate.weight")
        && source_->record("model.block.0.mlp.experts.gate.weight").dtype == "TQ") {
        if (resident_full_expert_bytes() <= bytes) return 0;
        std::vector<std::string> prefixes;
        for (std::size_t layer = 0; layer < layers_.size(); ++layer)
            prefixes.push_back("model.block." + std::to_string(layer));
        if (dspark_) for (std::size_t stage = 0; stage < dspark_->stage_count(); ++stage)
            prefixes.push_back("predictor.stage." + std::to_string(stage));
        auto cache = std::make_shared<MlxTqExpertCache>(*source_, prefixes, bytes);
        for (const auto& layer : layers_) layer.validate_tq_offload(cache);
        if (dspark_) dspark_->validate_tq_offload(cache);
        for (auto& layer : layers_) layer.use_tq_offload(cache);
        if (dspark_) dspark_->use_tq_offload(cache);
        tq_expert_cache_ = std::move(cache);
        const auto after = mlx::core::get_active_memory();
        return before > after ? before - after : 0;
    }
    if (!mfe_offload_cache_) {
        if (!ssd_expert_cache_ && resident_full_expert_bytes() <= bytes) return 0;
        if (!source_) throw std::runtime_error("DeepSeek-V4.1 expert backing is unavailable");
        auto cache = std::make_shared<MlxMfeOffloadCache>(*source_, bytes);
        for (const auto& layer : layers_) layer.validate_offload(cache);
        if (dspark_) dspark_->validate_offload(cache);
        if (ssd_expert_cache_) {
            ssd_expert_cache_->release_deferred();
            for (auto& layer : layers_) layer.use_offload(cache);
            if (dspark_) dspark_->use_offload(cache);
            ssd_expert_cache_.reset();
        }
        mfe_offload_cache_ = std::move(cache);
        update_expert_prefill_priority();
    }
    auto resident = resident_full_expert_bytes();
    if (ced_prefill_active_ && dspark_) {
        dspark_->offload_until(mfe_offload_cache_, resident, bytes);
    }
    const auto first = ced_prefill_active_ ? layers_.size() / 2 : 0;
    for (std::size_t offset = 0; offset < layers_.size(); ++offset) {
        if (resident <= bytes) break;
        auto& layer = layers_[(first + offset) % layers_.size()];
        const auto released = layer.resident_routed_bytes();
        if (released) {
            layer.use_offload(mfe_offload_cache_);
            resident -= released;
        }
    }
    if (resident > bytes && dspark_) {
        dspark_->offload_until(mfe_offload_cache_, resident, bytes);
    }
    mfe_offload_cache_->set_cache_limit(bytes > resident ? bytes - resident : 0);
    const auto after = mlx::core::get_active_memory();
    return before > after ? before - after : 0;
}

void MlxDeepseekV41CausalLm::update_expert_prefill_priority() {
    if (!mfe_offload_cache_) return;
    std::vector<std::string> names;
    if (ced_prefill_active_ && source_) {
        const auto encoder_layers = layers_.size() / 2;
        names.reserve(encoder_layers * 3);
        for (std::size_t layer = 0; layer < encoder_layers; ++layer) {
            const auto prefix = "model.block." + std::to_string(layer) + ".mlp.experts.";
            if (source_->contains(prefix + "gate.weight")) {
                names.push_back(prefix + "gate.weight");
                names.push_back(prefix + "up.weight");
            } else names.push_back(prefix + "gate_up.weight");
            names.push_back(prefix + "down.weight");
        }
    }
    mfe_offload_cache_->set_prefill_priority(names);
}

int MlxDeepseekV41CausalLm::preferred_prefill_chunk_size(
    int portable_default) const noexcept {
    if (portable_default <= 0 || layers_.empty()) return portable_default;
    int recommendation = 0;
    for (const auto& layer : layers_) {
        const int candidate = layer.recommended_prefill_chunk_size();
        if (candidate <= 0) return portable_default;
        recommendation = recommendation == 0
            ? candidate
            : std::min(recommendation, candidate);
    }
    return recommendation > 0 ? recommendation : portable_default;
}

std::optional<MlxSsdExpertCacheStats>
MlxDeepseekV41CausalLm::ssd_expert_cache_stats() const {
    if (tq_expert_cache_) {
        const auto cache = tq_expert_cache_->stats();
        MlxSsdExpertCacheStats result;
        result.requests = cache.requests; result.hits = cache.hits;
        result.misses = cache.misses; result.loads = cache.misses;
        result.evictions = cache.evictions; result.bytes_read = cache.bytes_read;
        result.read_calls = cache.read_calls; result.io_seconds = cache.io_seconds;
        result.route_sync_seconds = cache.route_sync_seconds;
        result.prepare_seconds = cache.prepare_seconds;
        result.routed_graph_seconds = cache.graph_seconds;
        result.overlapped_read_batches = cache.overlapped_read_batches;
        result.resident_experts = cache.cached_experts;
        result.resident_bytes = cache.allocated_bytes;
        result.cached_payload_bytes = cache.cached_bytes;
        result.cache_metadata_bytes = cache.metadata_bytes;
        return result;
    }
    if (ssd_expert_cache_) {
        return ssd_expert_cache_->stats();
    }
    if (!mfe_offload_cache_) {
        return std::nullopt;
    }
    MlxSsdExpertCacheStats stats;
    const auto reuse = mfe_offload_cache_->reuse_stats();
    stats.loads = reuse.expert_loads;
    stats.misses = reuse.expert_loads;
    stats.hits = reuse.expert_hits;
    stats.requests = stats.hits + stats.misses;
    stats.coalesced = reuse.load_waits;
    stats.packed_reuses = reuse.packed_hits;
    stats.bytes_read = reuse.bytes_read;
    stats.read_calls = reuse.read_calls;
    stats.evictions = reuse.evictions;
    stats.io_seconds = reuse.io_seconds;
    stats.wait_seconds = reuse.wait_seconds;
    stats.prepare_seconds = reuse.prepare_seconds;
    stats.view_seconds = reuse.view_seconds;
    stats.device_route_hits = reuse.cached_route_hits;
    stats.device_route_misses = reuse.cached_route_misses;
    stats.device_route_layers = reuse.cached_route_hits + reuse.cached_route_misses;
    stats.overlapped_read_batches = reuse.overlapped_read_batches;
    stats.overlapped_read_submissions = reuse.overlapped_read_submissions;
    stats.parallel_body_read_batches = reuse.parallel_body_read_batches;
    stats.resident_experts = mfe_offload_cache_->cached_expert_count();
    stats.resident_bytes = mfe_offload_cache_->resident_packed_bytes();
    return stats;
}

std::optional<MlxQuantizedRowStoreStats>
MlxDeepseekV41CausalLm::engram_ssd_stats() const {
    std::optional<MlxQuantizedRowStoreStats> result;
    for (const auto& layer : layers_) {
        const auto current = layer.engram_ssd_stats();
        if (!current) continue;
        if (!result) result.emplace();
        result->row_requests += current->row_requests;
        result->cache_hits += current->cache_hits;
        result->cache_misses += current->cache_misses;
        result->rows_loaded += current->rows_loaded;
        result->bytes_read += current->bytes_read;
        result->read_calls += current->read_calls;
        result->io_seconds += current->io_seconds;
        result->resident_rows += current->resident_rows;
        result->resident_payload_bytes += current->resident_payload_bytes;
        result->cache_limit_bytes += current->cache_limit_bytes;
    }
    return result;
}

void MlxDeepseekV41CausalLm::prewarm_ssd_expert_arena() {
    if (ssd_expert_cache_) {
        ssd_expert_cache_->prewarm_metal();
    }
}

void MlxDeepseekV41CausalLm::clear_expert_cache() {
    if (tq_expert_cache_) tq_expert_cache_->clear();
    if (ssd_expert_cache_) {
        ssd_expert_cache_->clear();
    }
    if (mfe_offload_cache_) {
        mfe_offload_cache_->clear();
    }
}

array MlxDeepseekV41CausalLm::forward_impl(
    const array& raw_token_ids,
    const std::optional<array>& image_mask,
    bool reuse_cache,
    const std::optional<array>& input_embeddings,
    array* dspark_hidden,
    bool update_dspark,
    bool skip_lm_head,
    bool ced_prefill,
    bool defer_ced_decoder) {
    auto token_ids = normalized_ids(raw_token_ids);
    const int batch = token_ids.shape(0);
    const int tokens = token_ids.shape(1);
    const bool replay_decoder = ced_prefill && !image_mask &&
        speculative_cache_start_ < 0 && tokens > config_.sliding_window &&
        supports_ced_prefill();
    const int decoder_begin = static_cast<int>(config_.n_layers / 2);
    const bool encoder_only = defer_ced_decoder && replay_decoder && skip_lm_head;
    const auto layer_count = encoder_only ? static_cast<std::size_t>(decoder_begin) : layers_.size();
    const int replay_offset = replay_decoder
        ? tokens - static_cast<int>(config_.sliding_window) : 0;
    struct PrefillPriorityScope {
        bool& active;
        std::shared_ptr<MlxMfeOffloadCache>& cache;
        bool enabled;
        ~PrefillPriorityScope() {
            if (enabled) {
                active = false;
                if (cache) cache->set_prefill_priority({});
            }
        }
    } prefill_priority{ced_prefill_active_, mfe_offload_cache_, replay_decoder};
    if (replay_decoder) {
        ced_prefill_active_ = true;
        update_expert_prefill_priority();
    }
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
    if (hashes.layers > 0) {
        hashes.prefetched_rows.resize(
            static_cast<std::size_t>(hashes.layers));
        for (const auto& layer : layers_) {
            layer.prefetch_engram(hashes);
        }
    }
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
    std::array<
        std::optional<MlxSsdPrefetchedExpertLayer>,
        2> routed_pipeline;
    const auto routed_rows = static_cast<std::size_t>(batch) *
        static_cast<std::size_t>(tokens);
    if (tokens > 1 && ssd_expert_cache_ && !layers_.empty()) {
        routed_pipeline[0] = layers_[0].prefetch_routed(routed_rows);
        if (routed_pipeline[0].has_value() && layers_.size() > 1) {
            routed_pipeline[1] = layers_[1].prefetch_routed(routed_rows);
        }
    }
    const auto capture_target = [this, replay_decoder](
        std::size_t index,
        const array& attention_input,
        std::vector<std::optional<array>>& targets) {
        if (targets.empty()) return;
        for (std::size_t target = 0;
             target < config_.dspark_target_layer_ids.size();
             ++target) {
            if (config_.dspark_target_layer_ids[target] ==
                static_cast<std::int64_t>(index)) {
                targets[target] = replay_decoder &&
                        attention_input.shape(1) > config_.sliding_window
                    ? slice_tokens(attention_input,
                          attention_input.shape(1) - static_cast<int>(config_.sliding_window),
                          attention_input.shape(1))
                    : attention_input;
            }
        }
    };
    const bool route_transaction = ssd_expert_cache_ && tokens == 1 &&
        !image_mask.has_value() && speculative_cache_start_ < 0 &&
        mlx_ssd_route_transactions_enabled();
    if (!route_transaction) {
        for (std::size_t index = 0; index < layer_count; ++index) {
            if (mfe_offload_cache_ && batch == 1 && tokens == 1 && cache_position_ > 0 &&
                !image_mask && speculative_cache_start_ < 0 && !detail::component_profile_active()) {
                std::vector<MlxMfeCachedRoute> routes;
                for (auto next = index; next < layers_.size() && next < index + 4; ++next) {
                    auto route = layers_[next].cached_route();
                    if (!route) break;
                    routes.push_back(std::move(*route));
                }
                if (routes.size() > 1) {
                    const auto restore = [&](std::size_t first = 0) {
                        std::vector<array> restored;
                        for (std::size_t offset = first; offset < routes.size(); ++offset) {
                            auto& state = states_[index + offset];
                            if (!state.attention.speculation) continue;
                            auto values = layers_[index + offset].rollback_route_replay(state);
                            restored.insert(restored.end(), std::make_move_iterator(values.begin()),
                                std::make_move_iterator(values.end()));
                        }
                        detail::eval_with_timing(std::move(restored));
                    };
                    auto trial_hidden = hidden;
                    auto trial_pre = previous_pre;
                    auto trial_shared = shared_attention;
                    auto trial_targets = target_hiddens;
                    struct Prefix {
                        array hidden;
                        array pre;
                        MlxDeepseekV41SharedAttentionState shared;
                        std::vector<std::optional<array>> targets;
                    };
                    std::vector<Prefix> prefixes;
                    prefixes.reserve(routes.size());
                    try {
                        std::vector<array> backups;
                        for (std::size_t offset = 0; offset < routes.size(); ++offset) {
                            auto values = layers_[index + offset].begin_route_replay(
                                states_[index + offset], tokens);
                            backups.insert(backups.end(), std::make_move_iterator(values.begin()),
                                std::make_move_iterator(values.end()));
                        }
                        mlx::core::async_eval(std::move(backups));
                        for (std::size_t offset = 0; offset < routes.size(); ++offset) {
                            auto result = layers_[index + offset].forward(trial_hidden, trial_pre,
                                &hashes, image_mask, states_[index + offset], trial_shared,
                                cache_position_, nullptr, false, &routes[offset]);
                            trial_hidden = std::move(result.hidden);
                            trial_pre = std::move(result.next_pre);
                            capture_target(index + offset, result.attention_input, trial_targets);
                            prefixes.push_back({trial_hidden, trial_pre, trial_shared, trial_targets});
                        }
                        mlx::core::async_eval({*routes.back().ids});
                        mlx::core::async_eval({trial_hidden});
                        std::size_t completed = 0;
                        while (completed < routes.size() && routes[completed].all_resident()) {
                            routes[completed].resolve();
                            ++completed;
                        }
                        if (completed == routes.size()) {
                            for (std::size_t offset = 0; offset < routes.size(); ++offset)
                                layers_[index + offset].commit_route_replay(states_[index + offset]);
                            hidden = std::move(trial_hidden);
                            previous_pre = std::move(trial_pre);
                            shared_attention = std::move(trial_shared);
                            target_hiddens = std::move(trial_targets);
                            index += routes.size() - 1;
                            continue;
                        }
                        routes[completed].resolve();
                        detail::eval_with_timing(trial_hidden);
                        restore(completed);
                        if (completed > 0) {
                            for (std::size_t offset = 0; offset < completed; ++offset)
                                layers_[index + offset].commit_route_replay(states_[index + offset]);
                            auto& prefix = prefixes[completed - 1];
                            hidden = std::move(prefix.hidden);
                            previous_pre = std::move(prefix.pre);
                            shared_attention = std::move(prefix.shared);
                            target_hiddens = std::move(prefix.targets);
                            index += completed - 1;
                            continue;
                        }
                    } catch (...) {
                        try { detail::eval_with_timing(trial_hidden); } catch (...) {}
                        try { restore(); } catch (...) {}
                        throw;
                    }
                }
            }
            auto* prefetched = routed_pipeline[index % 2].has_value()
                ? &*routed_pipeline[index % 2]
                : nullptr;
            auto result = layers_[index].forward(
                hidden,
                previous_pre,
                &hashes,
                image_mask,
                states_[index],
                shared_attention,
                cache_position_ + (replay_decoder && index > decoder_begin ? replay_offset : 0),
                prefetched,
                replay_decoder && index >= decoder_begin);
            hidden = std::move(result.hidden);
            previous_pre = std::move(result.next_pre);
            if (prefetched != nullptr) {
                routed_pipeline[index % 2].reset();
                if (index + 2 < layer_count) {
                    routed_pipeline[index % 2] =
                        layers_[index + 2].prefetch_routed(
                            replay_decoder && index + 2 >= decoder_begin
                                ? static_cast<std::size_t>(batch) * config_.sliding_window
                                : routed_rows);
                }
            }
            capture_target(index, result.attention_input, target_hiddens);
        }
    } else {
        const bool force_transactions =
            mlx_ssd_force_route_transactions();
        const auto group_layers = static_cast<std::size_t>(
            mlx_ssd_route_transaction_group_layers(
                static_cast<int>(layers_.size())));
        std::size_t group_begin = 0;
        while (group_begin < layers_.size()) {
            if (!force_transactions &&
                !ssd_expert_cache_->route_layer_likely_hit(group_begin)) {
                auto result = layers_[group_begin].forward(
                    hidden,
                    previous_pre,
                    &hashes,
                    image_mask,
                    states_[group_begin],
                    shared_attention,
                    cache_position_);
                hidden = std::move(result.hidden);
                previous_pre = std::move(result.next_pre);
                capture_target(
                    group_begin,
                    result.attention_input,
                    target_hiddens);
                ++group_begin;
                continue;
            }
            auto group_end = std::min(
                layers_.size(), group_begin + group_layers);
            for (std::size_t index = group_begin + 1;
                 index < group_end;
                 ++index) {
                if (!force_transactions &&
                    !ssd_expert_cache_->route_layer_likely_hit(index)) {
                    group_end = index;
                    break;
                }
            }
            const auto hidden_checkpoint = hidden;
            const auto previous_pre_checkpoint = previous_pre;
            const auto shared_attention_checkpoint = shared_attention;
            const auto target_checkpoint = target_hiddens;
            std::vector<std::optional<MlxSsdPreparedExperts>> pins(
                group_end - group_begin);
            bool completed = false;
            for (std::size_t attempt = 0;
                 attempt <= group_end - group_begin;
                 ++attempt) {
                std::vector<array> backups;
                try {
                    for (std::size_t index = group_begin;
                         index < group_end;
                         ++index) {
                        auto values = layers_[index].begin_route_replay(
                            states_[index], tokens);
                        backups.insert(
                            backups.end(),
                            std::make_move_iterator(values.begin()),
                            std::make_move_iterator(values.end()));
                    }
                    detail::eval_with_timing(std::move(backups));
                    ssd_expert_cache_->begin_route_transaction();
                    auto trial_hidden = hidden_checkpoint;
                    auto trial_previous_pre = previous_pre_checkpoint;
                    auto trial_shared_attention =
                        shared_attention_checkpoint;
                    auto trial_targets = target_checkpoint;
                    for (std::size_t index = group_begin;
                         index < group_end;
                         ++index) {
                        auto result = layers_[index].forward(
                            trial_hidden,
                            trial_previous_pre,
                            &hashes,
                            image_mask,
                            states_[index],
                            trial_shared_attention,
                            cache_position_);
                        trial_hidden = std::move(result.hidden);
                        trial_previous_pre = std::move(result.next_pre);
                        capture_target(
                            index,
                            result.attention_input,
                            trial_targets);
                    }
                    detail::eval_with_timing(trial_hidden);
                    auto transaction =
                        ssd_expert_cache_->resolve_route_transaction();
                    if (transaction.all_hit) {
                        for (std::size_t index = group_begin;
                             index < group_end;
                             ++index) {
                            layers_[index].commit_route_replay(
                                states_[index]);
                        }
                        hidden = std::move(trial_hidden);
                        previous_pre = std::move(trial_previous_pre);
                        shared_attention =
                            std::move(trial_shared_attention);
                        target_hiddens = std::move(trial_targets);
                        pins.clear();
                        ssd_expert_cache_->release_deferred();
                        completed = true;
                        break;
                    }
                    std::vector<array> restored;
                    for (std::size_t index = group_begin;
                         index < group_end;
                         ++index) {
                        auto values = layers_[index].rollback_route_replay(
                            states_[index]);
                        restored.insert(
                            restored.end(),
                            std::make_move_iterator(values.begin()),
                            std::make_move_iterator(values.end()));
                    }
                    detail::eval_with_timing(std::move(restored));
                    for (const auto& route : transaction.routes) {
                        auto prepared = ssd_expert_cache_->prepare(
                            route.layer,
                            route.experts);
                        pins.at(route.layer - group_begin).emplace(
                            std::move(prepared));
                    }
                    ssd_expert_cache_->release_deferred();
                } catch (...) {
                    ssd_expert_cache_->cancel_route_transaction();
                    for (std::size_t index = group_begin;
                         index < group_end;
                         ++index) {
                        if (states_[index].attention.speculation) {
                            try {
                                auto restored =
                                    layers_[index].rollback_route_replay(
                                        states_[index]);
                                detail::eval_with_timing(
                                    std::move(restored));
                            } catch (...) {
                            }
                        }
                    }
                    pins.clear();
                    ssd_expert_cache_->release_deferred();
                    throw;
                }
            }
            if (!completed) {
                pins.clear();
                ssd_expert_cache_->release_deferred();
                throw std::runtime_error(
                    "DeepSeek-V4.1 SSD route group did not converge");
            }
            group_begin = group_end;
        }
    }
    if (encoder_only) {
        layers_[decoder_begin].append_ced_encoder_context(
            hidden, previous_pre, states_[decoder_begin], cache_position_);
        cache_position_ += tokens;
        return mlx::core::zeros(
            Shape{batch, 0, static_cast<int>(config_.vocab)}, mlx::core::float32);
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
            cache_position_ + replay_offset,
            replay_decoder);
    }
    cache_position_ += tokens;
    if (skip_lm_head) {
        return mlx::core::zeros(
            Shape{batch, 0, static_cast<int>(config_.vocab)},
            mlx::core::float32);
    }
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
    stable_cache_tokens_.clear();
    stable_dspark_state_.reset();
    return forward_impl(token_ids, std::nullopt, use_cache);
}

bool MlxDeepseekV41CausalLm::supports_ced_prefill() const noexcept {
    const int midpoint = static_cast<int>(config_.n_layers / 2);
    if (midpoint <= 0 || config_.n_layers % 2 ||
        !config_.is_kv_source(midpoint) || !config_.is_index_source(midpoint)) return false;
    for (int layer = midpoint; layer < config_.n_layers; ++layer) {
        if (config_.compress_ratios[layer] != 1 || config_.has_engram(layer) ||
            (layer > midpoint && config_.is_kv_source(layer))) return false;
    }
    return true;
}

array MlxDeepseekV41CausalLm::prefill(
    const array& raw_token_ids,
    int chunk_size,
    bool full_logits) {
    mfq::engine::PrefillActivity prefill_activity;
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
    const int final_tokens = tokens - ((tokens - 1) / chunk_size) * chunk_size;
    for (int begin = 0; begin < tokens; begin += chunk_size) {
        const int end = std::min(tokens, begin + chunk_size);
        auto logits = forward_impl(
            slice_tokens(token_ids, begin, end),
            std::nullopt,
            true,
            std::nullopt,
            nullptr,
            true,
            !full_logits && end < tokens,
            !full_logits,
            !full_logits && end < tokens && final_tokens > config_.sliding_window);
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
    stable_cache_tokens_.clear();
    stable_dspark_state_.reset();
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
    mfq::engine::PrefillActivity prefill_activity;
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
    const MfqTokenConstraintPtr& token_constraint,
    const MlxPrefixCacheHooks& prefix_cache) {
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
        limit > 1;
    last_mtp_stats_ = {
        dspark_.has_value(),
        mtp_active,
        0,
        0,
        0,
    };
    if (mtp_active) {
        const auto commit_history = [&](const MlxMtpDraftContext& context) {
            if (context.initial || dspark_state_->position() == cache_position_) return;
            const int expected_width = static_cast<int>(config_.hidden *
                static_cast<std::int64_t>(config_.dspark_target_layer_ids.size()));
            dspark_->append_context(mlx_mtp_committed_hidden(context, 1, expected_width),
                *dspark_state_, context.target_cache_start);
        };
        MlxMtpEngineCallbacks callbacks;
        callbacks.predictor = dspark_->mtp_descriptor();
        callbacks.target_cache_position = [this] {
            return cache_position_;
        };
        callbacks.prepare_draft =
            [&](const MlxMtpDraftContext& context,
                const MlxMtpTokenSelector& select_token) {
                commit_history(context);
                if (context.requested_depth == 0) {
                    return;
                }
                const array anchor_ids(
                    {context.pending_token},
                    Shape{1, 1},
                    mlx::core::int32);
                if (!dspark_state_ ||
                    dspark_state_->position() != cache_position_) {
                    throw std::runtime_error(
                        "DeepSeek-V4.1 DSpark state is not synchronized "
                        "with the target");
                }
                dspark_->propose(
                    anchor_ids,
                    *dspark_state_,
                    select_token,
                    context.requested_depth);
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
        if (prefix_cache) {
            callbacks.draft_limit = [&](int position, int depth) {
                return prefix_cache.draft_limit(position, depth);
            };
            callbacks.committed_target = [&](const MlxMtpDraftContext& context) {
                commit_history(context);
                prefix_cache.checkpoint(cache_position_);
            };
        }
        const auto generated = run_mlx_mtp_generation(
            MlxMtpEngineRequest{
                vocab,
                limit,
                max_context_,
                logits,
                sampling,
                counts,
                std::span<const std::int64_t>(config_.eos_token_ids),
                callback,
                0u,
                token_constraint,
            },
            callbacks,
            last_mtp_stats_);
        prefix_cache.checkpoint(cache_position_, true);
        return generated;
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
        if (token_constraint &&
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
        if (token_constraint) {
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
            (!sampling.warmup && mlx_token_in_set(config_.eos_token_ids, token)) ||
            generated == limit) {
            break;
        }
        logits = mlx_last_token_logits(decode(token_ids), vocab);
        prefix_cache.checkpoint(cache_position_);
    }
    prefix_cache.checkpoint(cache_position_, true);
    return generated;
}

std::int32_t MlxDeepseekV41CausalLm::generate(
    const std::vector<std::int64_t>& prompt,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint,
    std::optional<std::size_t> stable_prefix_tokens,
    int prefill_chunk_size,
    const MlxPrefixCacheHooks& prefix_cache) {
    return generate_impl(prompt, sampling, max_tokens, callback, prefill_callback,
        token_constraint, stable_prefix_tokens, prefill_chunk_size, prefix_cache);
}

std::int32_t MlxDeepseekV41CausalLm::generate_impl(
    const std::vector<std::int64_t>& prompt, const MlxSamplingParams& sampling,
    std::int32_t max_tokens, const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint,
    std::optional<std::size_t> stable_prefix_tokens, int prefill_chunk_size,
    const MlxPrefixCacheHooks& requested_prefix_cache,
    const std::vector<MlxDeepseekV41ImageInput>* images) {
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
    const array prompt_ids(
        values.begin(), Shape{1, static_cast<int>(values.size())},
        mlx::core::int32);
    const bool mtp_candidate = supports_mtp() && sampling.enable_mtp &&
        max_tokens > 1;
    mtp_context_requested_ = mtp_candidate;
    std::size_t requested_stable_count = stable_prefix_tokens
        ? std::min(*stable_prefix_tokens, prompt.size())
        : 0;
    auto prefix_cache = requested_prefix_cache;
    if (prefix_cache) requested_stable_count = std::min(requested_stable_count, prompt.size() - 1);
    if (images) for (const auto& image : *images)
        if (requested_stable_count > std::size_t(image.begin) && requested_stable_count < std::size_t(image.end))
            requested_stable_count = image.begin;
    prefix_cache.input_tokens = requested_stable_count;
    const bool retain_stable_prefix =
        requested_stable_count > 0;
    const std::size_t stable_count = retain_stable_prefix
        ? requested_stable_count
        : 0;
    std::size_t reused_tokens = 0;
    const bool dspark_prefix_ready =
        !mtp_candidate ||
        (stable_dspark_state_ &&
         stable_dspark_state_->batch() == 1 &&
         stable_dspark_state_->position() ==
             static_cast<int>(stable_cache_tokens_.size()));
    if (retain_stable_prefix && dspark_prefix_ready &&
        cache_batch_ == 1 &&
        cache_position_ == static_cast<int>(stable_cache_tokens_.size()) &&
        !stable_cache_tokens_.empty() &&
        stable_cache_tokens_.size() <= stable_count &&
        stable_cache_tokens_.size() < prompt.size() &&
        std::equal(
            stable_cache_tokens_.begin(),
            stable_cache_tokens_.end(),
            prompt.begin())) {
        reused_tokens = stable_cache_tokens_.size();
    } else if (retain_stable_prefix) {
        reset_cache(1);
    }
    if (mtp_candidate && reused_tokens > 0) {
        dspark_state_.emplace(stable_dspark_state_->snapshot());
        materialize_deepseek_v41_dspark_state(*dspark_state_);
    } else if (!mtp_candidate) {
        dspark_state_.reset();
    }

    struct StableCacheRestore {
        std::vector<MlxDeepseekV41LayerState>& target_states;
        MlxDeepseekV41EngramHashState& target_hash;
        int& target_position;
        int& target_batch;
        std::vector<std::int64_t>& target_tokens;
        std::optional<MlxDeepseekV41DSparkState>& target_active_dspark;
        std::optional<MlxDeepseekV41DSparkState>& target_stable_dspark;
        std::optional<std::vector<MlxDeepseekV41LayerState>> saved_states;
        std::optional<DeepseekV41EngramHashSnapshot> saved_hash;
        std::optional<MlxDeepseekV41DSparkState> saved_dspark;
        std::vector<std::int64_t> saved_tokens;
        int saved_position = 0;
        int saved_batch = 0;

        StableCacheRestore(
            std::vector<MlxDeepseekV41LayerState>& states,
            MlxDeepseekV41EngramHashState& hash,
            int& position,
            int& batch,
            std::vector<std::int64_t>& tokens,
            std::optional<MlxDeepseekV41DSparkState>& active_dspark,
            std::optional<MlxDeepseekV41DSparkState>& stable_dspark)
            : target_states(states),
              target_hash(hash),
              target_position(position),
              target_batch(batch),
              target_tokens(tokens),
              target_active_dspark(active_dspark),
              target_stable_dspark(stable_dspark) {}

        void capture(
            const std::vector<std::int64_t>& prompt_tokens,
            std::size_t count) {
            std::vector<MlxDeepseekV41LayerState> snapshots;
            snapshots.reserve(target_states.size());
            for (const auto& state : target_states) {
                snapshots.push_back({state.attention.snapshot()});
            }
            saved_states = std::move(snapshots);
            saved_hash = target_hash.snapshot();
            saved_tokens.assign(
                prompt_tokens.begin(),
                prompt_tokens.begin() + static_cast<std::ptrdiff_t>(count));
            saved_position = static_cast<int>(count);
            saved_batch = target_batch;
            if (target_active_dspark) {
                if (target_active_dspark->batch() != target_batch ||
                    target_active_dspark->position() != saved_position) {
                    throw std::runtime_error(
                        "DeepSeek-V4.1 stable DSpark checkpoint mismatch");
                }
                saved_dspark.emplace(target_active_dspark->snapshot());
            } else {
                saved_dspark.reset();
            }
        }

        const std::vector<MlxDeepseekV41LayerState>& states() const {
            return *saved_states;
        }

        const std::optional<MlxDeepseekV41DSparkState>& dspark() const {
            return saved_dspark;
        }

        ~StableCacheRestore() noexcept {
            if (!saved_states || !saved_hash) return;
            try {
                if (target_states.size() != saved_states->size()) {
                    throw std::runtime_error(
                        "DeepSeek-V4.1 stable cache layer count changed");
                }
                for (std::size_t index = 0;
                     index < target_states.size(); ++index) {
                    target_states[index].attention.restore_snapshot(
                        std::move((*saved_states)[index].attention));
                }
                target_hash.restore(std::move(*saved_hash));
                target_position = saved_position;
                target_batch = saved_batch;
                target_tokens = std::move(saved_tokens);
                target_active_dspark.reset();
                target_stable_dspark = std::move(saved_dspark);
            } catch (...) {
                target_states.clear();
                target_hash.clear();
                target_position = 0;
                target_batch = 0;
                target_tokens.clear();
                target_active_dspark.reset();
                target_stable_dspark.reset();
            }
        }
    } stable_restore(
        states_,
        engram_hash_,
        cache_position_,
        cache_batch_,
        stable_cache_tokens_,
        dspark_state_,
        stable_dspark_state_);

    const auto started = std::chrono::steady_clock::now();
    array logits(0.0f);
    try {
        if (!retain_stable_prefix) {
            logits = images ? prefill_multimodal(prompt, *images) : prefill(prompt_ids, prefill_chunk_size, false);
        } else {
            const auto prefill_range = [&](std::size_t begin, std::size_t end) {
                std::optional<array> last;
                const auto length = end - begin;
                const auto chunk = static_cast<std::size_t>(prefill_chunk_size);
                const auto final_tokens = length ? length - ((length - 1) / chunk) * chunk : 0;
                for (std::size_t offset = begin; offset < end;) {
                    auto stop = prefix_cache.prefill_end(offset, std::min(
                        end,
                        offset + static_cast<std::size_t>(prefill_chunk_size)));
                    std::optional<array> image_mask, embeddings;
                    if (images) {
                        for (const auto& image : *images)
                            if (stop > std::size_t(image.begin) && stop < std::size_t(image.end)) stop = image.end;
                        std::vector<MlxDeepseekV41ImageInput> selected;
                        for (const auto& image : *images) if (std::size_t(image.end) > offset && std::size_t(image.begin) < stop) {
                            if (std::size_t(image.begin) < offset || std::size_t(image.end) > stop || stop > end)
                                throw std::runtime_error("DeepSeek-V4.1 cache boundary splits an image");
                            selected.push_back(image);
                            selected.back().begin -= static_cast<int>(offset);
                            selected.back().end -= static_cast<int>(offset);
                        }
                        if (!selected.empty()) {
                            const std::vector<std::int64_t> ids(prompt.begin() + offset, prompt.begin() + stop);
                            embeddings = vision_->embed_prompt(ids, selected, embedding_, mlx::core::float16);
                            image_mask = vision_->image_mask(static_cast<int>(stop - offset), selected);
                        }
                    }
                    last = forward_impl(
                        slice_tokens(
                            prompt_ids,
                            static_cast<int>(offset),
                            static_cast<int>(stop)),
                        image_mask,
                        true,
                        embeddings,
                        nullptr,
                        true,
                        stop < end,
                        true,
                        stop < end && final_tokens > config_.sliding_window && !prefix_cache);
                    if (prefix_cache.wants(stop)) {
                        materialize_states(states_);
                        prefix_cache.checkpoint(stop);
                    }
                    offset = stop;
                }
                if (!last) {
                    throw std::runtime_error(
                        "DeepSeek-V4.1 stable prefill range is empty");
                }
                auto final = slice_tokens(
                    *last, last->shape(1) - 1, last->shape(1));
                return mlx::core::squeeze(std::move(final), 1);
            };

            std::optional<array> stable_logits;
            if (reused_tokens < stable_count) {
                stable_logits = prefill_range(reused_tokens, stable_count);
            }
            materialize_states(states_);
            if (!prefix_cache) {
                stable_restore.capture(prompt, stable_count);
                materialize_states(stable_restore.states());
                if (stable_restore.dspark())
                    materialize_deepseek_v41_dspark_state(*stable_restore.dspark());
            } else {
                prefix_cache.checkpoint(stable_count);
            }
            if (stable_count < prompt.size()) {
                logits = prefill_range(stable_count, prompt.size());
            } else if (stable_logits) {
                logits = std::move(*stable_logits);
            } else {
                throw std::runtime_error(
                    "DeepSeek-V4.1 stable cache has no logits for sampling");
            }
        }
    } catch (...) {
        mtp_context_requested_ = false;
        throw;
    }
    mtp_context_requested_ = false;
    logits.eval();
    mlx::core::synchronize();
    if (prefill_callback) {
        prefill_callback(
            prompt.size() - reused_tokens,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
    }
    const auto limit = std::min<std::int32_t>(
        max_tokens,
        max_context_ - static_cast<int>(prompt.size()) + 1);
    return generate_from_prefill(
        std::move(logits), prompt, sampling, limit, callback,
        token_constraint, prefix_cache);
}

std::int32_t MlxDeepseekV41CausalLm::generate_multimodal(
    const std::vector<std::int64_t>& prompt,
    const std::vector<MlxDeepseekV41ImageInput>& images,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint,
    std::optional<std::size_t> stable_prefix_tokens, int prefill_chunk_size,
    const MlxPrefixCacheHooks& prefix_cache) {
    if (prompt.empty() || images.empty() || max_tokens < 0) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 multimodal generation request");
    }
    return generate_impl(prompt, sampling, max_tokens, callback, prefill_callback,
        token_constraint, stable_prefix_tokens, prefill_chunk_size, prefix_cache, &images);
}

MlxDeepseekV41TextSessionState
MlxDeepseekV41CausalLm::capture_text_session_state(
    const std::vector<std::int64_t>& tokens, bool copy_arrays) const {
    if (cache_batch_ != 1 || cache_position_ <= 0 ||
        static_cast<std::size_t>(cache_position_) != tokens.size() ||
        states_.size() != layers_.size() ||
        speculative_engram_snapshot_ || speculative_token_ids_ ||
        speculative_cache_start_ >= 0) {
        throw std::runtime_error(
            "DeepSeek-V4.1 text session token count does not match cache");
    }
    auto hash = engram_hash_.snapshot();
    if (hash.batch != cache_batch_ || hash.position != cache_position_) {
        throw std::runtime_error(
            "DeepSeek-V4.1 Engram state does not match attention cache");
    }
    MlxDeepseekV41TextSessionState state;
    state.tokens = tokens;
    state.engram_hash = std::move(hash);
    state.cache_position = cache_position_;
    state.cache_batch = cache_batch_;
    state.bytes = state.engram_hash.context.size() * sizeof(std::int64_t);
    state.layers.reserve(states_.size());
    for (const auto& layer : states_) {
        auto snapshot = layer.attention.snapshot(copy_arrays);
        state.bytes += snapshot.nbytes();
        state.layers.push_back({std::move(snapshot)});
    }
    const auto& predictor = dspark_state_ ? dspark_state_ : stable_dspark_state_;
    if (predictor) {
        if (!dspark_ || predictor->batch() != cache_batch_ ||
            predictor->position() != cache_position_) {
            throw std::runtime_error(
                "DeepSeek-V4.1 stable DSpark state does not match cache");
        }
        state.dspark.emplace(predictor->snapshot(copy_arrays));
        state.bytes += state.dspark->nbytes();
    }
    materialize_states(state.layers);
    if (state.dspark) {
        materialize_deepseek_v41_dspark_state(*state.dspark);
    }
    return state;
}

void MlxDeepseekV41CausalLm::restore_text_session_state(
    const MlxDeepseekV41TextSessionState& state) {
    if (state.cache_batch != 1 || state.cache_position <= 0 ||
        static_cast<std::size_t>(state.cache_position) !=
            state.tokens.size() ||
        state.layers.size() != layers_.size() ||
        state.engram_hash.batch != state.cache_batch ||
        state.engram_hash.position != state.cache_position) {
        throw std::runtime_error(
            "DeepSeek-V4.1 text session state is incompatible");
    }
    if (state.dspark &&
        (!dspark_ || state.dspark->batch() != state.cache_batch ||
         state.dspark->position() != state.cache_position)) {
        throw std::runtime_error(
            "DeepSeek-V4.1 text session DSpark state is incompatible");
    }
    try {
        mtp_context_requested_ = false;
        reset_cache(1);
        for (std::size_t index = 0; index < states_.size(); ++index) {
            states_[index].attention.restore_snapshot(
                state.layers[index].attention.snapshot());
        }
        materialize_states(states_);
        std::optional<MlxDeepseekV41DSparkState> restored_dspark;
        if (state.dspark) {
            restored_dspark.emplace(
                dspark_->make_state(state.cache_batch, mlx::core::float16));
            restored_dspark->restore_snapshot(state.dspark->snapshot());
            materialize_deepseek_v41_dspark_state(*restored_dspark);
        }
        engram_hash_.restore(state.engram_hash);
        cache_position_ = state.cache_position;
        cache_batch_ = state.cache_batch;
        stable_cache_tokens_ = state.tokens;
        stable_dspark_state_ = std::move(restored_dspark);
    } catch (...) {
        clear_cache();
        throw;
    }
}

} // namespace mfq::metal
