#include "mlx_qwen35_causal_lm.h"
#include "mlx_eval_timing.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

int checked_positive_int(
    std::int64_t value,
    const char* name) {
    if (value <= 0 ||
        value > static_cast<std::int64_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string("invalid Qwen3.5 causal LM ") + name);
    }
    return static_cast<int>(value);
}

template <typename Function>
void visit_layers(
    std::vector<MlxQwen35Layer>& layers,
    Function&& function) {
    for (auto& layer : layers) {
        std::visit(function, layer);
    }
}

array last_token_logits(
    const array& logits,
    int vocab) {
    if (logits.ndim() != 3 ||
        logits.shape(0) != 1 ||
        logits.shape(1) <= 0 ||
        logits.shape(2) != vocab) {
        throw std::runtime_error(
            "Qwen3.5 generation logits must have [1,tokens,vocab] shape");
    }
    return mlx::core::reshape(
        mlx::core::slice(
            logits,
            Shape{0, logits.shape(1) - 1, 0},
            Shape{1, logits.shape(1), vocab}),
        Shape{1, vocab});
}

array validate_positions(
    const array& positions,
    int tokens,
    int maximum_sequence) {
    const bool valid_rank =
        positions.ndim() == 1 ||
        (positions.ndim() == 2 &&
         (positions.shape(0) == 1 ||
          positions.shape(0) == 3));
    if (!valid_rank) {
        throw std::runtime_error(
            "Qwen3.5 positions must have [tokens], [1,tokens], "
            "or [3,tokens] shape");
    }
    if (positions.shape(-1) != tokens) {
        throw std::runtime_error(
            "Qwen3.5 position length must match the token count");
    }
    if (positions.dtype() != mlx::core::int32 &&
        positions.dtype() != mlx::core::int64) {
        throw std::runtime_error(
            "Qwen3.5 positions must use int32 or int64 values");
    }

    auto result = mlx::core::contiguous(positions);
    result.eval();
    if (result.dtype() == mlx::core::int32) {
        const auto* values = result.data<std::int32_t>();
        for (std::size_t index = 0; index < result.size(); ++index) {
            if (values[index] < 0 ||
                values[index] >= maximum_sequence) {
                throw std::runtime_error(
                    "Qwen3.5 explicit position is outside "
                    "the configured context range");
            }
        }
        return result;
    }

    const auto* values = result.data<std::int64_t>();
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (values[index] < 0 ||
            values[index] >= maximum_sequence) {
            throw std::runtime_error(
                "Qwen3.5 explicit position is outside "
                "the configured context range");
        }
    }
    return mlx::core::contiguous(
        mlx::core::astype(result, mlx::core::int32));
}

} // namespace

std::optional<array>
detail::qwen35_generation_token_counts(
    const MlxSamplingParams& sampling,
    const array& prompt_ids,
    int vocab) {
    if (!sampling.has_penalties()) {
        return std::nullopt;
    }
    return sample_token_counts_add(
        mlx::core::zeros(
            Shape{vocab},
            mlx::core::int32),
        prompt_ids);
}

MlxQwen35CausalLm MlxQwen35CausalLm::load(
    const MfqContainer& model) {
    const auto config = Qwen35Config::from_mfq(model);
    const auto names = Qwen35TensorNames::detect(model);
    return load(model, config, names);
}

MlxQwen35CausalLm MlxQwen35CausalLm::load(
    const MfqContainer& model,
    const Qwen35Config& config,
    const Qwen35TensorNames& names) {
    const auto runtime_config =
        adapt_qwen35_config_for_tensor_names(config, names);
    validate_qwen35_model_bindings(
        model,
        runtime_config,
        names);

    std::vector<MlxQwen35Layer> layers;
    layers.reserve(
        static_cast<std::size_t>(
            runtime_config.num_hidden_layers));
    for (std::size_t index = 0;
         index < runtime_config.layer_types.size();
         ++index) {
        const auto& type =
            runtime_config.layer_types[index];
        if (type == "full_attention") {
            layers.emplace_back(
                MlxQwen35FullAttentionBlock::load(
                    model,
                    runtime_config,
                    names,
                    index));
        } else if (type == "linear_attention") {
            layers.emplace_back(
                MlxQwen35LinearAttentionBlock::load(
                    model,
                    runtime_config,
                    names,
                    index));
        } else {
            throw std::runtime_error(
                "unsupported Qwen3.5 layer type at index " +
                std::to_string(index) + ": " + type);
        }
    }

    auto embedding =
        Qwen35Embedding::load(model, names.token_embedding);
    std::optional<Qwen35Linear> output;
    if (!runtime_config.tie_word_embeddings) {
        output.emplace(
            Qwen35Linear::load(model, names.output));
    }

    return MlxQwen35CausalLm(
        runtime_config,
        std::move(embedding),
        std::move(layers),
        load_qwen35_rms_norm(
            model,
            names.output_norm,
            runtime_config.rms_norm_eps,
            runtime_config.norm_weight_offset),
        std::move(output));
}

MlxQwen35CausalLm::MlxQwen35CausalLm(
    Qwen35Config config,
    Qwen35Embedding embedding,
    std::vector<MlxQwen35Layer> layers,
    MlxRmsNorm output_norm,
    std::optional<Qwen35Linear> output,
    mlx::core::Dtype activation_dtype)
    : config_(std::move(config)),
      embedding_(std::move(embedding)),
      layers_(std::move(layers)),
      output_norm_(std::move(output_norm)),
      output_(std::move(output)),
      activation_dtype_(activation_dtype) {
    validate_components();
}

void MlxQwen35CausalLm::validate_components() const {
    const int vocab =
        checked_positive_int(config_.vocab_size, "vocab_size");
    const int hidden =
        checked_positive_int(config_.hidden_size, "hidden_size");
    const int layer_count =
        checked_positive_int(
            config_.num_hidden_layers,
            "num_hidden_layers");
    checked_positive_int(
        config_.max_position_embeddings,
        "max_position_embeddings");
    if (activation_dtype_ != mlx::core::float16 &&
        activation_dtype_ != mlx::core::float32) {
        throw std::runtime_error(
            "Qwen3.5 activation dtype must be float16 or float32");
    }
    if (embedding_.vocabulary_size() != vocab ||
        embedding_.hidden_size() != hidden ||
        output_norm_.width() != hidden) {
        throw std::runtime_error(
            "Qwen3.5 causal LM embedding/norm dimensions mismatch");
    }
    if (config_.tie_word_embeddings) {
        if (output_.has_value()) {
            throw std::runtime_error(
                "Qwen3.5 tied embeddings must not provide a separate "
                "output weight");
        }
    } else if (!output_.has_value()) {
        throw std::runtime_error(
            "Qwen3.5 untied embeddings require an output weight");
    } else if (
        output_->input_size() != hidden ||
        output_->output_size() != vocab) {
        throw std::runtime_error(
            "Qwen3.5 causal LM output dimensions mismatch");
    }
    if (layers_.size() != static_cast<std::size_t>(layer_count) ||
        config_.layer_types.size() !=
            static_cast<std::size_t>(layer_count)) {
        throw std::runtime_error(
            "Qwen3.5 causal LM layer count does not match config");
    }

    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const bool full = std::holds_alternative<
            MlxQwen35FullAttentionBlock>(layers_[index]);
        const std::string_view actual =
            full ? "full_attention" : "linear_attention";
        if (config_.layer_types[index] != actual) {
            throw std::runtime_error(
                "Qwen3.5 causal LM layer variant mismatch at index " +
                std::to_string(index));
        }
        std::visit(
            [&](const auto& layer) {
                const auto& layer_config = layer.config();
                if (layer_config.hidden_size != config_.hidden_size ||
                    layer_config.max_position_embeddings !=
                        config_.max_position_embeddings) {
                    throw std::runtime_error(
                        "Qwen3.5 causal LM layer config mismatch at index " +
                        std::to_string(index));
                }
            },
            layers_[index]);
    }
}

array MlxQwen35CausalLm::forward(
    const array& token_ids,
    bool use_cache) {
    return forward_impl(token_ids, nullptr, use_cache);
}

array MlxQwen35CausalLm::forward(
    const array& token_ids,
    const array& positions,
    bool use_cache) {
    return forward_impl(token_ids, &positions, use_cache);
}

array MlxQwen35CausalLm::forward_impl(
    const array& token_ids,
    const array* positions,
    bool use_cache) {
    if (token_ids.ndim() != 2 ||
        token_ids.shape(0) <= 0 ||
        token_ids.shape(1) <= 0) {
        throw std::runtime_error(
            "Qwen3.5 token ids must have non-empty [batch,tokens] shape");
    }
    const int batch = token_ids.shape(0);
    const int tokens = token_ids.shape(1);
    const int maximum_sequence =
        static_cast<int>(config_.max_position_embeddings);
    const int position = use_cache ? cache_position_ : 0;
    std::optional<array> explicit_positions;
    if (positions) {
        explicit_positions = validate_positions(
            *positions,
            tokens,
            maximum_sequence);
    }
    if (tokens > maximum_sequence - position) {
        throw std::runtime_error(
            "Qwen3.5 causal LM position range exceeds context capacity");
    }

    if (use_cache) {
        if (cache_batch_ == 0) {
            reset_cache(batch);
        } else if (cache_batch_ != batch) {
            throw std::runtime_error(
                "Qwen3.5 causal LM cache batch mismatch");
        }
    }

    auto hidden = embedding_(token_ids, activation_dtype_);
    for (auto& layer : layers_) {
        hidden = std::visit(
            [&](auto& block) {
                if (explicit_positions) {
                    return block.forward(
                        hidden,
                        *explicit_positions,
                        position,
                        use_cache);
                }
                return block.forward(
                    hidden,
                    position,
                    use_cache);
            },
            layer);
    }
    auto normalized = output_norm_(hidden);
    auto logits = config_.tie_word_embeddings
        ? embedding_.project(normalized)
        : (*output_)(normalized);
    if (use_cache) {
        cache_position_ += tokens;
    }
    return logits;
}

void MlxQwen35CausalLm::reset_cache(int batch) {
    if (batch <= 0) {
        throw std::runtime_error(
            "Qwen3.5 causal LM cache batch must be positive");
    }
    visit_layers(
        layers_,
        [batch](auto& layer) {
            layer.reset_cache(batch);
        });
    cache_position_ = 0;
    cache_batch_ = batch;
    stable_cache_tokens_.clear();
}

void MlxQwen35CausalLm::prepare_cache_for_prefill(
    int batch,
    int prompt_tokens) {
    if (batch <= 0 || prompt_tokens <= 0 ||
        prompt_tokens > config_.max_position_embeddings) {
        throw std::runtime_error(
            "Qwen3.5 prefill cache dimensions are invalid");
    }
    int initial_capacity = 16;
    while (initial_capacity < prompt_tokens &&
           initial_capacity < config_.max_position_embeddings) {
        if (initial_capacity >
            config_.max_position_embeddings / 2) {
            initial_capacity =
                config_.max_position_embeddings;
        } else {
            initial_capacity *= 2;
        }
    }
    visit_layers(
        layers_,
        [batch, initial_capacity](auto& layer) {
            using Layer = std::decay_t<decltype(layer)>;
            if constexpr (std::is_same_v<
                              Layer,
                              MlxQwen35FullAttentionBlock>) {
                layer.reset_cache(
                    batch,
                    initial_capacity);
            } else {
                layer.reset_cache(batch);
            }
            layer.materialize_cache();
        });
    cache_position_ = 0;
    cache_batch_ = batch;
    stable_cache_tokens_.clear();
}

void MlxQwen35CausalLm::clear_cache() noexcept {
    visit_layers(
        layers_,
        [](auto& layer) {
            layer.clear_cache();
        });
    cache_position_ = 0;
    cache_batch_ = 0;
    stable_cache_tokens_.clear();
}

MlxQwen35TextSessionState
MlxQwen35CausalLm::capture_text_session_state(
    const std::vector<std::int64_t>& tokens) const {
    if (cache_batch_ != 1 || cache_position_ <= 0 ||
        static_cast<std::size_t>(cache_position_) != tokens.size() ||
        layers_.empty()) {
        throw std::runtime_error(
            "Qwen3.5 text session token count does not match the cache");
    }
    MlxQwen35TextSessionState state;
    state.tokens = tokens;
    state.cache_position = cache_position_;
    state.cache_batch = cache_batch_;
    state.layers.reserve(layers_.size());
    for (const auto& layer : layers_) {
        std::visit(
            [&](const auto& block) {
                auto snapshot = block.snapshot_cache();
                state.bytes += snapshot.nbytes();
                state.layers.emplace_back(std::move(snapshot));
            },
            layer);
    }
    return state;
}

void MlxQwen35CausalLm::restore_text_session_state(
    const MlxQwen35TextSessionState& state) {
    if (state.cache_batch != 1 || state.cache_position <= 0 ||
        static_cast<std::size_t>(state.cache_position) !=
            state.tokens.size() ||
        state.layers.size() != layers_.size()) {
        throw std::runtime_error(
            "Qwen3.5 text session state is incompatible");
    }
    try {
        for (std::size_t index = 0; index < layers_.size(); ++index) {
            std::visit(
                [&](auto& block) {
                    using Block = std::decay_t<decltype(block)>;
                    if constexpr (std::is_same_v<
                                      Block,
                                      MlxQwen35FullAttentionBlock>) {
                        const auto* snapshot = std::get_if<
                            MlxKvCacheSnapshot>(&state.layers[index]);
                        if (!snapshot) {
                            throw std::runtime_error(
                                "Qwen3.5 session layer type changed");
                        }
                        block.restore_cache(*snapshot);
                    } else {
                        const auto* snapshot = std::get_if<
                            MlxQwen35LinearAttentionCacheSnapshot>(
                                &state.layers[index]);
                        if (!snapshot) {
                            throw std::runtime_error(
                                "Qwen3.5 session layer type changed");
                        }
                        block.restore_cache(*snapshot);
                    }
                },
                layers_[index]);
        }
        cache_position_ = state.cache_position;
        cache_batch_ = state.cache_batch;
        stable_cache_tokens_ = state.tokens;
    } catch (...) {
        clear_cache();
        throw;
    }
}

std::string_view MlxQwen35CausalLm::layer_type(
    std::size_t index) const {
    if (index >= layers_.size()) {
        throw std::out_of_range(
            "Qwen3.5 causal LM layer index is out of range");
    }
    return std::holds_alternative<
        MlxQwen35FullAttentionBlock>(layers_[index])
        ? "full_attention"
        : "linear_attention";
}

std::int32_t MlxQwen35CausalLm::generate(
    const std::vector<std::int64_t>& prompt,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const MlxTokenCallback& callback,
    const std::function<void(std::size_t, double)>&
        prefill_callback,
    const MfqTokenConstraintPtr& token_constraint,
    std::optional<std::size_t> stable_prefix_tokens) {
    if (prompt.empty()) {
        throw std::invalid_argument(
            "Qwen3.5 generation prompt cannot be empty");
    }
    if (max_tokens < 0) {
        throw std::invalid_argument(
            "Qwen3.5 generation max_tokens cannot be negative");
    }
    const int vocab = static_cast<int>(config_.vocab_size);
    const int maximum_sequence =
        static_cast<int>(config_.max_position_embeddings);
    if (prompt.size() >
        static_cast<std::size_t>(maximum_sequence)) {
        throw std::invalid_argument(
            "Qwen3.5 generation prompt exceeds context capacity");
    }

    std::vector<std::int32_t> prompt_values;
    prompt_values.reserve(prompt.size());
    for (const auto token : prompt) {
        if (token < 0 || token >= vocab) {
            throw std::invalid_argument(
                "Qwen3.5 generation prompt token is out of range");
        }
        prompt_values.push_back(
            static_cast<std::int32_t>(token));
    }

    const auto prompt_count =
        static_cast<int>(prompt_values.size());
    if (max_tokens == 0) {
        reset_cache(1);
        return 0;
    }
    const std::size_t stable_count = stable_prefix_tokens
        ? std::min(*stable_prefix_tokens, prompt.size())
        : 0;
    std::size_t reused_tokens = 0;
    if (stable_count > 0 && cache_batch_ == 1 &&
        cache_position_ == static_cast<int>(stable_cache_tokens_.size()) &&
        !stable_cache_tokens_.empty() &&
        stable_cache_tokens_.size() <= stable_count &&
        stable_cache_tokens_.size() < prompt.size() &&
        std::equal(
            stable_cache_tokens_.begin(),
            stable_cache_tokens_.end(),
            prompt.begin())) {
        reused_tokens = stable_cache_tokens_.size();
    } else {
        // Cache allocation/zeroing is request setup, not model prefill
        // compute. Materialize it before the evaluation-only metric starts.
        prepare_cache_for_prefill(1, prompt_count);
    }

    const array prompt_ids(
        prompt_values.begin(),
        Shape{1, prompt_count},
        mlx::core::int32);
    auto counts =
        detail::qwen35_generation_token_counts(
            sampling,
            prompt_ids,
            vocab);
    double prefill_evaluation_ms = 0.0;
    std::optional<MlxQwen35TextSessionState> stable_snapshot;
    struct StableStateRestore {
        MlxQwen35CausalLm& model;
        std::optional<MlxQwen35TextSessionState>& snapshot;
        ~StableStateRestore() noexcept {
            if (!snapshot) return;
            try {
                model.restore_text_session_state(*snapshot);
            } catch (...) {
                model.clear_cache();
            }
        }
    } stable_restore{*this, stable_snapshot};

    auto logits = [&]() {
        detail::ScopedMlxEvaluationTiming timing(
            prefill_callback
                ? &prefill_evaluation_ms
                : nullptr);
        const auto forward_range = [&](std::size_t begin, std::size_t end) {
            if (begin >= end || end > prompt.size()) {
                throw std::runtime_error(
                    "Qwen3.5 session prefill range is invalid");
            }
            auto ids = mlx::core::slice(
                prompt_ids,
                Shape{0, static_cast<int>(begin)},
                Shape{1, static_cast<int>(end)});
            return last_token_logits(forward(ids, true), vocab);
        };
        std::optional<array> stable_logits;
        array value = [&]() {
            if (stable_count == 0) {
                return forward_range(0, prompt.size());
            }
            if (reused_tokens < stable_count) {
                stable_logits = forward_range(
                    reused_tokens, stable_count);
            }
            if (cache_position_ != static_cast<int>(stable_count)) {
                throw std::runtime_error(
                    "Qwen3.5 stable session cache position mismatch");
            }
            std::vector<std::int64_t> stable_tokens(
                prompt.begin(),
                prompt.begin() +
                    static_cast<std::ptrdiff_t>(stable_count));
            stable_snapshot =
                capture_text_session_state(stable_tokens);
            if (stable_count < prompt.size()) {
                return forward_range(stable_count, prompt.size());
            }
            if (!stable_logits) {
                throw std::runtime_error(
                    "Qwen3.5 stable prefix has no sampling logits");
            }
            return std::move(*stable_logits);
        }();
        if (prefill_callback) {
            // Build the lazy graph before entering eval_with_timing().
            // Only MLX execution/synchronization contributes to prefill.
            detail::eval_with_timing(value);
        }
        return value;
    }();
    if (prefill_callback) {
        prefill_callback(
            prompt.size() - reused_tokens,
            prefill_evaluation_ms);
    }
    MlxSampler sampler(sampling);
    const auto context_samples =
        maximum_sequence - prompt_count + 1;
    const auto generation_limit =
        std::min(max_tokens, context_samples);

    std::int32_t generated = 0;
    while (generated < generation_limit) {
        auto sampled = counts.has_value()
            ? sampler.sample(
                  logits,
                  *counts)
            : sampler.sample(logits);
        sampled.eval();
        auto token =
            sampled.data<std::int32_t>()[0];
        if (token < 0 || token >= vocab) {
            throw std::runtime_error(
                "Qwen3.5 sampler returned an out-of-range token");
        }
        if (token_constraint && token_constraint->allows &&
            !token_constraint->allows(token)) {
            auto adjusted = counts.has_value()
                ? sampler.apply_penalties(logits, *counts)
                : logits;
            adjusted = mlx::core::contiguous(
                mlx::core::astype(
                    adjusted, mlx::core::float32));
            adjusted.eval();
            std::vector<float> masked(
                adjusted.data<float>(),
                adjusted.data<float>() + vocab);
            token_constraint->apply(masked.data(), masked.size());
            const array constrained_logits(
                masked.begin(), Shape{1, vocab}, mlx::core::float32);
            sampled = sampler.sample(constrained_logits);
            sampled.eval();
            token = sampled.data<std::int32_t>()[0];
            if (token < 0 || token >= vocab ||
                !token_constraint->allows(token)) {
                throw std::runtime_error(
                    "Qwen3.5 constrained sampler returned an invalid token");
            }
        }
        if (token_constraint && token_constraint->accept) {
            token_constraint->accept(token);
        }

        const array token_ids(
            {token},
            Shape{1, 1},
            mlx::core::int32);
        if (counts.has_value()) {
            *counts = sample_token_counts_add(
                *counts,
                token_ids);
        }
        ++generated;

        if (callback &&
            !callback(static_cast<std::int64_t>(token))) {
            break;
        }
        if (generated == generation_limit) {
            break;
        }
        logits = last_token_logits(
            forward(token_ids, true),
            vocab);
    }
    return generated;
}

} // namespace mfq::metal
