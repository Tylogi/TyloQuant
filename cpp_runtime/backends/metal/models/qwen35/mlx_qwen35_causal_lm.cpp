#include "mlx_qwen35_causal_lm.h"
#include "mlx_legacy_tensor_compat.h"
#include "mlx_mtp.h"
#include "mlx_eval_timing.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
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

// One predictor layer is recurrently chained; it does not limit the chain to
// one token. This is the model-family capability consumed by the common MTP
// scheduler, independently of the number of predictor layers in the file.
constexpr int kQwen35MtpMaximumDraftDepth = 5;

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

std::optional<MlxQwen35MtpModule>
MlxQwen35MtpModule::load_if_present(
    const MfqContainer& model,
    const Qwen35Config& config,
    const Qwen35TensorNames&) {
    const bool complete = model.contains("predictor.fusion.weight");
    const bool has_any = complete ||
        model.contains("predictor.hidden_norm.weight") ||
        model.contains("predictor.embedding_norm.weight") ||
        model.contains("predictor.output_norm.weight") ||
        model.contains("predictor.block.0.attention.query.weight");
    if (!complete) {
        if (has_any) {
            throw std::runtime_error(
                "Qwen3.5 MFQ contains an incomplete MTP head");
        }
        return std::nullopt;
    }
    if (config.mtp_num_hidden_layers <= 0) {
        throw std::runtime_error(
            "Qwen3.5 MFQ contains MTP tensors but the model config "
            "does not advertise an MTP layer");
    }
    if (config.mtp_use_dedicated_embeddings) {
        throw std::runtime_error(
            "Qwen3.5 dedicated MTP embeddings are not supported");
    }

    Qwen35Config mtp_config = config;
    const auto mtp_names = Qwen35TensorNames::predictor();
    mtp_config.num_hidden_layers = config.mtp_num_hidden_layers;
    mtp_config.layer_types.assign(
        static_cast<std::size_t>(config.mtp_num_hidden_layers),
        "full_attention");

    std::vector<MlxQwen35FullAttentionBlock> layers;
    layers.reserve(
        static_cast<std::size_t>(config.mtp_num_hidden_layers));
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(config.mtp_num_hidden_layers);
         ++index) {
        layers.emplace_back(
            MlxQwen35FullAttentionBlock::load(
                model,
                mtp_config,
                mtp_names,
                index));
    }

    return MlxQwen35MtpModule(
        mtp_config,
        load_qwen35_rms_norm(
            model,
            "predictor.hidden_norm.weight",
            config.rms_norm_eps,
            config.norm_weight_offset),
        load_qwen35_rms_norm(
            model,
            "predictor.embedding_norm.weight",
            config.rms_norm_eps,
            config.norm_weight_offset),
        Qwen35Linear::load(
            model,
            "predictor.fusion.weight"),
        std::move(layers),
        load_qwen35_rms_norm(
            model,
            "predictor.output_norm.weight",
            config.rms_norm_eps,
            config.norm_weight_offset));
}

MlxQwen35MtpModule::MlxQwen35MtpModule(
    Qwen35Config config,
    MlxRmsNorm hidden_norm,
    MlxRmsNorm embedding_norm,
    Qwen35Linear fusion,
    std::vector<MlxQwen35FullAttentionBlock> layers,
    MlxRmsNorm output_norm)
    : config_(std::move(config)),
      hidden_norm_(std::move(hidden_norm)),
      embedding_norm_(std::move(embedding_norm)),
      fusion_(std::move(fusion)),
      layers_(std::move(layers)),
      output_norm_(std::move(output_norm)) {
    validate_components();
}

void MlxQwen35MtpModule::validate_components() const {
    const int hidden = checked_positive_int(
        config_.hidden_size, "MTP hidden_size");
    if (layers_.empty() ||
        fusion_.input_size() != hidden * 2 ||
        fusion_.output_size() != hidden ||
        hidden_norm_.width() != hidden ||
        embedding_norm_.width() != hidden ||
        output_norm_.width() != hidden) {
        throw std::runtime_error(
            "Qwen3.5 MTP component dimensions are incompatible");
    }
}

array MlxQwen35MtpModule::forward(
    const array& hidden_states,
    const array& next_token_ids,
    const Qwen35Embedding& embedding,
    bool use_cache) {
    return forward_impl(
        hidden_states,
        next_token_ids,
        nullptr,
        embedding,
        use_cache);
}

array MlxQwen35MtpModule::forward(
    const array& hidden_states,
    const array& next_token_ids,
    const array& positions,
    const Qwen35Embedding& embedding,
    bool use_cache) {
    return forward_impl(
        hidden_states,
        next_token_ids,
        &positions,
        embedding,
        use_cache);
}

array MlxQwen35MtpModule::forward_impl(
    const array& hidden_states,
    const array& next_token_ids,
    const array* positions,
    const Qwen35Embedding& embedding,
    bool use_cache) {
    if (hidden_states.ndim() != 3 || next_token_ids.ndim() != 2 ||
        hidden_states.shape(0) != next_token_ids.shape(0) ||
        hidden_states.shape(1) != next_token_ids.shape(1) ||
        hidden_states.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Qwen3.5 MTP inputs have incompatible shapes");
    }
    auto embedded = embedding(next_token_ids, hidden_states.dtype());
    auto fused = fusion_(mlx::core::concatenate(
        {embedding_norm_(embedded), hidden_norm_(hidden_states)},
        -1));
    if (positions) {
        (void)validate_positions(
            *positions,
            hidden_states.shape(1),
            static_cast<int>(config_.max_position_embeddings));
    }
    for (auto& layer : layers_) {
        fused = positions
            ? layer.forward(fused, *positions, use_cache)
            : layer.forward(fused, use_cache);
    }
    return output_norm_(fused);
}

void MlxQwen35MtpModule::reset_cache(
    int batch,
    int initial_capacity) {
    for (auto& layer : layers_) {
        layer.reset_cache(batch, initial_capacity);
    }
}

void MlxQwen35MtpModule::materialize_cache() {
    for (auto& layer : layers_) {
        layer.materialize_cache();
    }
}

void MlxQwen35MtpModule::trim_cache_to(int position) {
    if (position < 0) {
        throw std::runtime_error(
            "Qwen3.5 MTP cache target position is invalid");
    }
    for (auto& layer : layers_) {
        const int extra = layer.cache_position() - position;
        if (extra < 0) {
            throw std::runtime_error(
                "Qwen3.5 MTP cache is behind committed history");
        }
        if (extra > 0) {
            layer.trim_cache(extra);
        }
    }
}

void MlxQwen35MtpModule::clear_cache() noexcept {
    for (auto& layer : layers_) {
        layer.clear_cache();
    }
}

int MlxQwen35MtpModule::cache_position() const noexcept {
    return layers_.empty() ? 0 : layers_.front().cache_position();
}

MlxQwen35CausalLm MlxQwen35CausalLm::load(
    const MfqContainer& model) {
    const auto graph = effective_model_graph(model);
    if (graph.graph_kind != "causal_lm" || graph.backbone != "qwen3_5" ||
        !graph.has_component("text")) {
        throw std::runtime_error(
            "model graph does not describe a Qwen3.5 causal runtime");
    }
    const auto config = adapt_qwen35_config_for_storage(
        Qwen35Config::from_mfq(model),
        model.legacy_tensor_layout().qwen_gdn_gguf_layout);
    auto runtime = load(model, config, Qwen35TensorNames::canonical());
    const bool predictor_declared = graph.has_component("predictor");
    if (predictor_declared != runtime.mtp_.has_value()) {
        throw std::runtime_error(
            predictor_declared
                ? "model graph declares a missing predictor component"
                : "canonical predictor tensors are absent from the model graph");
    }
    if (const auto* vision = graph.component("vision")) {
        if (vision->implementation != "grid_vit" ||
            !config.vision || !config.image_token_id ||
            !config.video_token_id) {
            throw std::runtime_error(
                "model graph Vision component is incompatible with Qwen3.5");
        }
        runtime.vision_.emplace(MlxGridVisionPromptComponent::load(
            model,
            *config.vision,
            *config.image_token_id,
            *config.video_token_id,
            vision->input_contract,
            vision->position_policy));
    }
    return runtime;
}

MlxQwen35CausalLm MlxQwen35CausalLm::load(
    const MfqContainer& model,
    const Qwen35Config& config,
    const Qwen35TensorNames& names) {
    const auto& runtime_config = config;
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
    auto mtp = MlxQwen35MtpModule::load_if_present(
        model,
        runtime_config,
        names);

    return MlxQwen35CausalLm(
        runtime_config,
        std::move(embedding),
        std::move(layers),
        load_qwen35_rms_norm(
            model,
            names.output_norm,
            runtime_config.rms_norm_eps,
            runtime_config.norm_weight_offset),
        std::move(output),
        mlx::core::float16,
        std::move(mtp));
}

MlxQwen35CausalLm::MlxQwen35CausalLm(
    Qwen35Config config,
    Qwen35Embedding embedding,
    std::vector<MlxQwen35Layer> layers,
    MlxRmsNorm output_norm,
    std::optional<Qwen35Linear> output,
    mlx::core::Dtype activation_dtype,
    std::optional<MlxQwen35MtpModule> mtp,
    std::optional<MlxGridVisionPromptComponent> vision)
    : config_(std::move(config)),
      embedding_(std::move(embedding)),
      layers_(std::move(layers)),
      output_norm_(std::move(output_norm)),
      output_(std::move(output)),
      activation_dtype_(activation_dtype),
      mtp_(std::move(mtp)),
      vision_(std::move(vision)) {
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
    if (token_ids.ndim() != 2 ||
        token_ids.shape(0) <= 0 ||
        token_ids.shape(1) <= 0) {
        throw std::runtime_error(
            "Qwen3.5 token ids must have non-empty [batch,tokens] shape");
    }
    return forward_embeddings(embed_tokens(token_ids), use_cache);
}

array MlxQwen35CausalLm::forward(
    const array& token_ids,
    const array& positions,
    bool use_cache) {
    if (token_ids.ndim() != 2 ||
        token_ids.shape(0) <= 0 ||
        token_ids.shape(1) <= 0) {
        throw std::runtime_error(
            "Qwen3.5 token ids must have non-empty [batch,tokens] shape");
    }
    return forward_embeddings(embed_tokens(token_ids), positions, use_cache);
}

array MlxQwen35CausalLm::embed_tokens(const array& token_ids) const {
    if (token_ids.ndim() != 2 || token_ids.shape(0) <= 0 ||
        token_ids.shape(1) <= 0) {
        throw std::runtime_error(
            "Qwen3.5 token ids must have non-empty [batch,tokens] shape");
    }
    return embedding_(token_ids, activation_dtype_);
}

MlxPreparedPrompt MlxQwen35CausalLm::prepare_multimodal_prompt(
    const std::vector<std::int64_t>& prompt,
    const MlxGridMediaInput& media) const {
    if (!vision_) {
        throw std::runtime_error(
            "model graph does not provide an enabled Vision component");
    }
    if (prompt.empty() ||
        prompt.size() > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::invalid_argument("multimodal prompt length is invalid");
    }
    std::vector<std::int32_t> ids;
    ids.reserve(prompt.size());
    for (const auto token : prompt) {
        if (token < 0 || token >= config_.vocab_size) {
            throw std::invalid_argument(
                "multimodal prompt token is out of range");
        }
        ids.push_back(static_cast<std::int32_t>(token));
    }
    const array token_ids(
        ids.begin(), Shape{1, static_cast<int>(ids.size())},
        mlx::core::int32);
    return vision_->prepare(prompt, embed_tokens(token_ids), media);
}

array MlxQwen35CausalLm::forward_embeddings(
    const array& embeddings,
    bool use_cache) {
    return forward_embeddings_impl(
        embeddings,
        nullptr,
        use_cache).first;
}

array MlxQwen35CausalLm::forward_embeddings(
    const array& embeddings,
    const array& positions,
    bool use_cache) {
    return forward_embeddings_impl(
        embeddings,
        &positions,
        use_cache).first;
}

std::pair<array, array>
MlxQwen35CausalLm::forward_embeddings_with_hidden(
    const array& embeddings,
    bool use_cache) {
    return forward_embeddings_impl(
        embeddings,
        nullptr,
        use_cache);
}

std::pair<array, array>
MlxQwen35CausalLm::forward_embeddings_with_hidden(
    const array& embeddings,
    const array& positions,
    bool use_cache) {
    return forward_embeddings_impl(
        embeddings,
        &positions,
        use_cache);
}

std::pair<array, array>
MlxQwen35CausalLm::forward_embeddings_impl(
    const array& embeddings,
    const array* positions,
    bool use_cache,
    int speculative_confirmed,
    bool full_logits) {
    if (embeddings.ndim() != 3 ||
        embeddings.shape(0) <= 0 ||
        embeddings.shape(1) <= 0 ||
        embeddings.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Qwen3.5 embeddings must have non-empty "
            "[batch,tokens,hidden] shape");
    }
    if (embeddings.dtype() != mlx::core::float16 &&
        embeddings.dtype() != mlx::core::bfloat16 &&
        embeddings.dtype() != mlx::core::float32) {
        throw std::runtime_error(
            "Qwen3.5 embeddings must use a floating-point dtype");
    }
    const int batch = embeddings.shape(0);
    const int tokens = embeddings.shape(1);
    if (speculative_confirmed < 0 ||
        speculative_confirmed >= tokens ||
        (speculative_confirmed > 0 && !use_cache)) {
        throw std::runtime_error(
            "Qwen3.5 speculative forward configuration is invalid");
    }
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

    auto hidden = embeddings.dtype() == activation_dtype_
        ? embeddings
        : mlx::core::astype(embeddings, activation_dtype_);
    for (auto& layer : layers_) {
        hidden = std::visit(
            [&](auto& block) {
                using Block = std::decay_t<decltype(block)>;
                if constexpr (std::is_same_v<
                                  Block,
                                  MlxQwen35LinearAttentionBlock>) {
                    if (speculative_confirmed > 0) {
                        return block.forward_speculative(
                            hidden,
                            position,
                            speculative_confirmed);
                    }
                }
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
    auto head_input = full_logits
        ? hidden
        : mlx::core::slice(
              hidden,
              Shape{0, tokens - 1, 0},
              Shape{
                  batch,
                  tokens,
                  static_cast<int>(config_.hidden_size)});
    auto logits = project_logits(head_input);
    if (use_cache) {
        cache_position_ += tokens;
    }
    return {std::move(logits), std::move(hidden)};
}

std::pair<array, array>
MlxQwen35CausalLm::forward_with_hidden(
    const array& token_ids,
    bool use_cache,
    int speculative_confirmed) {
    if (token_ids.ndim() != 2 ||
        token_ids.shape(0) <= 0 ||
        token_ids.shape(1) <= 0) {
        throw std::runtime_error(
            "Qwen3.5 token ids must have non-empty [batch,tokens] shape");
    }
    return forward_embeddings_impl(
        embed_tokens(token_ids),
        nullptr,
        use_cache,
        speculative_confirmed);
}

array MlxQwen35CausalLm::project_logits(
    const array& hidden) const {
    return project_normalized(output_norm_(hidden));
}

array MlxQwen35CausalLm::project_normalized(
    const array& hidden) const {
    return config_.tie_word_embeddings
        ? embedding_.project(hidden)
        : (*output_)(hidden);
}

void MlxQwen35CausalLm::commit_speculative() {
    for (auto& layer : layers_) {
        std::visit(
            [](auto& block) {
                using Block = std::decay_t<decltype(block)>;
                if constexpr (std::is_same_v<
                                  Block,
                                  MlxQwen35LinearAttentionBlock>) {
                    block.commit_speculative();
                }
            },
            layer);
    }
}

void MlxQwen35CausalLm::rollback_speculative(
    int accepted_tokens,
    int draft_tokens) {
    if (accepted_tokens < 0 || draft_tokens <= 0 ||
        accepted_tokens >= draft_tokens ||
        draft_tokens > cache_position_) {
        throw std::runtime_error(
            "Qwen3.5 speculative rollback range is invalid");
    }
    const int rejected_tokens = draft_tokens - accepted_tokens;
    for (auto& layer : layers_) {
        std::visit(
            [accepted_tokens, rejected_tokens](auto& block) {
                using Block = std::decay_t<decltype(block)>;
                if constexpr (std::is_same_v<
                                  Block,
                                  MlxQwen35LinearAttentionBlock>) {
                    block.rollback_speculative(accepted_tokens);
                } else {
                    block.trim_cache(rejected_tokens);
                }
            },
            layer);
    }
    cache_position_ -= rejected_tokens;
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
    if (mtp_) {
        mtp_->reset_cache(batch);
    }
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
    if (mtp_) {
        mtp_->reset_cache(batch, initial_capacity);
        mtp_->materialize_cache();
    }
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
    if (mtp_) {
        mtp_->clear_cache();
    }
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
        if (mtp_) {
            mtp_->clear_cache();
        }
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
    MlxPreparedPrompt prepared;
    prepared.token_ids = prompt;
    return generate_prepared(
        prepared,
        sampling,
        max_tokens,
        callback,
        prefill_callback,
        token_constraint,
        stable_prefix_tokens);
}

std::int32_t MlxQwen35CausalLm::generate_prepared(
    const MlxPreparedPrompt& prepared,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const MlxTokenCallback& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint,
    std::optional<std::size_t> stable_prefix_tokens) {
    const auto& prompt = prepared.token_ids;
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
    if (prepared.embeddings &&
        (prepared.embeddings->ndim() != 3 ||
         prepared.embeddings->shape(0) != 1 ||
         prepared.embeddings->shape(1) != static_cast<int>(prompt.size()) ||
         prepared.embeddings->shape(2) != config_.hidden_size)) {
        throw std::invalid_argument(
            "prepared prompt embeddings disagree with token geometry");
    }
    if (prepared.positions) {
        (void)validate_positions(
            *prepared.positions,
            static_cast<int>(prompt.size()),
            maximum_sequence);
    }
    const auto next_logical_position =
        static_cast<std::int64_t>(prompt.size()) +
        prepared.decode_position_delta;
    if (next_logical_position < 0 ||
        next_logical_position > maximum_sequence) {
        throw std::invalid_argument(
            "prepared prompt decode position is outside context capacity");
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
        last_mtp_stats_ = {mtp_.has_value(), false, 0, 0, 0};
        reset_cache(1);
        return 0;
    }
    const bool mtp_candidate =
        mtp_.has_value() && sampling.enable_mtp &&
        !token_constraint &&
        max_tokens > 1;
    // MTP head state is not yet part of the persistent session snapshot.
    // Prefer a complete MTP prefill over restoring only the backbone, which
    // would leave the proposal head with an invalid history.
    const std::size_t stable_count =
        !prepared.transformed() && !mtp_candidate && stable_prefix_tokens
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
    const bool mtp_active =
        mtp_candidate &&
        stable_count == 0 &&
        reused_tokens == 0 &&
        max_tokens > 1;
    last_mtp_stats_ = {
        mtp_.has_value(),
        mtp_active,
        0,
        0,
        0,
    };

    const array prompt_ids(
        prompt_values.begin(),
        Shape{1, prompt_count},
        mlx::core::int32);
    auto counts = mlx_generation_token_counts(
        sampling, prompt_ids, vocab);
    double prefill_evaluation_ms = 0.0;
    std::optional<array> prefill_hidden;
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

    const bool profile_prefill =
        detail::component_profile_requested();
    detail::ComponentProfile component_profile;
    const auto profile_started = std::chrono::steady_clock::now();
    auto logits = [&]() {
        detail::ScopedComponentProfile component_scope(
            profile_prefill ? &component_profile : nullptr);
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
            std::optional<array> embeddings;
            if (prepared.embeddings) {
                embeddings = mlx::core::slice(
                    *prepared.embeddings,
                    Shape{0, static_cast<int>(begin), 0},
                    Shape{
                        1,
                        static_cast<int>(end),
                        static_cast<int>(config_.hidden_size)});
            }
            std::optional<array> positions;
            if (prepared.positions) {
                const auto rank = prepared.positions->ndim();
                if (rank == 1) {
                    positions = mlx::core::slice(
                        *prepared.positions,
                        Shape{static_cast<int>(begin)},
                        Shape{static_cast<int>(end)});
                } else {
                    positions = mlx::core::slice(
                        *prepared.positions,
                        Shape{0, static_cast<int>(begin)},
                        Shape{
                            prepared.positions->shape(0),
                            static_cast<int>(end)});
                }
            }
            const auto run = [&](bool retain_hidden) {
                if (embeddings) {
                    auto result = forward_embeddings_impl(
                        *embeddings,
                        positions ? &*positions : nullptr,
                        true,
                        0,
                        false);
                    if (retain_hidden) {
                        prefill_hidden = result.second;
                    }
                    return result.first;
                }
                if (positions) {
                    auto result = forward_embeddings_impl(
                        embed_tokens(ids),
                        &*positions,
                        true,
                        0,
                        false);
                    if (retain_hidden) {
                        prefill_hidden = result.second;
                    }
                    return result.first;
                }
                auto result = forward_embeddings_impl(
                    embed_tokens(ids),
                    nullptr,
                    true,
                    0,
                    false);
                if (retain_hidden) {
                    prefill_hidden = std::move(result.second);
                }
                return result.first;
            };
            if (mtp_active && begin == 0 && end == prompt.size()) {
                return run(true);
            }
            return run(false);
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
        if (profile_prefill) {
            detail::profile_eval("qwen35.output", value);
        } else if (prefill_callback) {
            // Build the lazy graph before entering eval_with_timing().
            // Only MLX execution/synchronization contributes to prefill.
            detail::eval_with_timing(value);
        }
        return value;
    }();
    if (profile_prefill) {
        const double wall_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - profile_started)
                .count();
        const double evaluated_ms = component_profile.evaluated_ms();
        prefill_evaluation_ms = wall_ms;
        std::cout
            << "component_profile model=qwen35 phase=prefill"
            << " tokens=" << (prompt.size() - reused_tokens)
            << " wall_ms=" << std::fixed << std::setprecision(3) << wall_ms
            << " evaluated_ms=" << evaluated_ms
            << " unscoped_ms=" << std::max(0.0, wall_ms - evaluated_ms)
            << std::endl;
        for (const auto& [name, component] : component_profile.timings()) {
            std::cout
                << "component_cost model=qwen35 phase=prefill"
                << " name=" << name
                << " ms=" << component.elapsed_ms
                << " calls=" << component.evaluations
                << " pct_evaluated="
                << (evaluated_ms > 0.0
                        ? 100.0 * component.elapsed_ms / evaluated_ms
                        : 0.0)
                << std::endl;
        }
    }
    if (prefill_callback) {
        prefill_callback(
            prompt.size() - reused_tokens,
            prefill_evaluation_ms);
    }
    MlxSampler sampler(sampling);
    const auto occupied_context = std::max<std::int64_t>(
        prompt_count,
        next_logical_position);
    const auto context_samples = static_cast<int>(
        maximum_sequence - occupied_context + 1);
    const auto generation_limit =
        std::min(max_tokens, context_samples);

    const auto make_positions = [&] (
            int cache_start,
            int tokens) -> std::optional<array> {
        if (!prepared.positions && prepared.decode_position_delta == 0) {
            return std::nullopt;
        }
        if (tokens <= 0) {
            throw std::runtime_error("decode position length must be positive");
        }
        const auto start = static_cast<std::int64_t>(cache_start) +
            prepared.decode_position_delta;
        if (start < 0 || start + tokens > maximum_sequence) {
            throw std::runtime_error(
                "prepared prompt decode position exceeds context capacity");
        }
        const int axes = prepared.positions &&
                prepared.positions->ndim() == 2 &&
                prepared.positions->shape(0) == 3
            ? 3
            : 1;
        std::vector<std::int32_t> values(
            static_cast<std::size_t>(axes * tokens));
        for (int axis = 0; axis < axes; ++axis) {
            for (int index = 0; index < tokens; ++index) {
                values[static_cast<std::size_t>(axis * tokens + index)] =
                    static_cast<std::int32_t>(start + index);
            }
        }
        if (axes == 1) {
            return array(
                values.begin(), Shape{tokens}, mlx::core::int32);
        }
        return array(
            values.begin(), Shape{axes, tokens}, mlx::core::int32);
    };
    const auto make_decode_positions = [&](int tokens) {
        return make_positions(cache_position_, tokens);
    };
    const auto forward_decode = [&](const array& ids) {
        const auto positions = make_decode_positions(ids.shape(1));
        return positions
            ? forward(ids, *positions, true)
            : forward(ids, true);
    };
    const auto forward_decode_with_hidden = [&, this](
            const array& ids,
            int speculative_confirmed = 0) {
        const auto positions = make_decode_positions(ids.shape(1));
        return forward_embeddings_impl(
            embed_tokens(ids),
            positions ? &*positions : nullptr,
            true,
            speculative_confirmed);
    };

    std::int32_t generated = 0;
    if (mtp_active) {
        if (!prefill_hidden) {
            throw std::runtime_error(
                "Qwen3.5 MTP prefill did not retain backbone hidden states");
        }

        // Prime the predictor cache from teacher-forced prompt pairs. The
        // final prompt hidden row is reserved for the first live proposal.
        if (prompt_count > 1) {
            mlx_prime_mtp_history(
                *prefill_hidden,
                prompt_ids,
                [&](const array& hidden_rows,
                    const array& shifted_ids,
                    int pair_offset) {
                    if (!prepared.positions) {
                        return mtp_->forward(
                            hidden_rows,
                            shifted_ids,
                            embedding_,
                            true);
                    }
                    const int begin = pair_offset + 1;
                    const int end = begin + shifted_ids.shape(1);
                    const auto rank = prepared.positions->ndim();
                    auto positions = rank == 1
                        ? mlx::core::slice(
                              *prepared.positions,
                              Shape{begin},
                              Shape{end})
                        : mlx::core::slice(
                              *prepared.positions,
                              Shape{0, begin},
                              Shape{
                                  prepared.positions->shape(0),
                                  end,
                              });
                    return mtp_->forward(
                        hidden_rows,
                        shifted_ids,
                        positions,
                        embedding_,
                        true);
                });
        }

        auto initial_hidden = mlx::core::slice(
            *prefill_hidden,
            Shape{0, prompt_count - 1, 0},
            Shape{
                1,
                prompt_count,
                static_cast<int>(config_.hidden_size),
            });
        int predictor_history_position = mtp_->cache_position();

        MlxMtpEngineCallbacks mtp_callbacks;
        mtp_callbacks.target_cache_position = [this] {
            return cache_position_;
        };
        mtp_callbacks.prepare_draft =
            [&, initial_hidden](
                const MlxMtpDraftContext& context,
                const MlxMtpTokenSelector& select_token) {
                array hidden_rows = initial_hidden;
                std::vector<std::int32_t> next_ids{
                    context.pending_token,
                };
                int logical_position = cache_position_;
                if (!context.initial) {
                    if (context.verified_hidden == nullptr ||
                        context.next_token_ids.empty()) {
                        throw std::runtime_error(
                            "Qwen3.5 MTP verified history is unavailable");
                    }
                    const int committed = context.accepted_drafts + 1;
                    if (context.next_token_ids.size() !=
                        static_cast<std::size_t>(committed)) {
                        throw std::runtime_error(
                            "Qwen3.5 MTP shifted history is inconsistent");
                    }
                    hidden_rows = mlx_mtp_committed_hidden(
                        context, 1, static_cast<int>(config_.hidden_size));
                    next_ids.assign(
                        context.next_token_ids.begin(),
                        context.next_token_ids.end());
                    logical_position = context.target_cache_start + 1;
                }

                if (hidden_rows.ndim() != 3 ||
                    hidden_rows.shape(0) != 1 ||
                    hidden_rows.shape(1) !=
                        static_cast<int>(next_ids.size()) ||
                    hidden_rows.shape(2) != config_.hidden_size) {
                    throw std::runtime_error(
                        "Qwen3.5 MTP history fold has incompatible shapes");
                }
                mtp_->trim_cache_to(predictor_history_position);
                const int committed = static_cast<int>(next_ids.size());
                const array committed_ids(
                    next_ids.begin(),
                    Shape{1, committed},
                    mlx::core::int32);
                const auto positions = make_positions(
                    logical_position, committed);
                auto head_hidden = positions
                    ? mtp_->forward(
                          hidden_rows,
                          committed_ids,
                          *positions,
                          embedding_,
                          true)
                    : mtp_->forward(
                          hidden_rows,
                          committed_ids,
                          embedding_,
                          true);
                predictor_history_position += committed;

                auto last_head = mlx::core::slice(
                    head_hidden,
                    Shape{0, committed - 1, 0},
                    Shape{
                        1,
                        committed,
                        static_cast<int>(config_.hidden_size),
                    });
                for (int position = 0;
                     position < context.requested_depth;
                     ++position) {
                    auto token = select_token(mlx_last_token_logits(
                        project_normalized(last_head), vocab));
                    if (position + 1 == context.requested_depth) {
                        break;
                    }
                    const auto draft_positions = make_positions(
                        logical_position + committed + position,
                        1);
                    last_head = draft_positions
                        ? mtp_->forward(
                              last_head,
                              mlx::core::reshape(token, Shape{1, 1}),
                              *draft_positions,
                              embedding_,
                              true)
                        : mtp_->forward(
                              last_head,
                              mlx::core::reshape(token, Shape{1, 1}),
                              embedding_,
                              true);
                }
            };
        mtp_callbacks.verify_target =
            [&](std::int32_t pending_token,
                const array& draft_tokens,
                int draft_count) {
                auto verify_ids = mlx_mtp_verification_ids(
                    pending_token, draft_tokens, draft_count);
                auto verified = forward_decode_with_hidden(
                    verify_ids,
                    draft_count > 0 ? 1 : 0);
                return MlxMtpTargetBatch{
                    mlx::core::reshape(
                        verified.first,
                        Shape{draft_count + 1, vocab}),
                    std::move(verified.second),
                };
            };
        mtp_callbacks.resolve_target =
            [this](int accepted_drafts, int draft_count) {
                if (draft_count <= 0) {
                    return;
                }
                if (accepted_drafts == draft_count) {
                    commit_speculative();
                } else {
                    rollback_speculative(
                        accepted_drafts, draft_count);
                }
            };
        mtp_callbacks.plain_decode = [&](std::int32_t pending_token) {
            const array ids(
                {pending_token},
                Shape{1, 1},
                mlx::core::int32);
            return mlx_last_token_logits(forward_decode(ids), vocab);
        };

        return run_mlx_mtp_generation(
            MlxMtpEngineRequest{
                vocab,
                generation_limit,
                maximum_sequence,
                kQwen35MtpMaximumDraftDepth,
                logits,
                sampling,
                counts,
                {},
                callback,
            },
            mtp_callbacks,
            last_mtp_stats_);
    }

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
        logits = mlx_last_token_logits(
            forward_decode(token_ids),
            vocab);
    }
    return generated;
}

} // namespace mfq::metal
