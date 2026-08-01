#include "mlx_deepseek_v4_causal_lm.h"
#include "mlx_eval_timing.h"

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using json = nlohmann::json;
using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kConnections = 4;
constexpr int kHcProjectionWidth =
    2 * kConnections + kConnections * kConnections;

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
    int left,
    int right,
    const char* label) {
    if (left <= 0 || right <= 0 ||
        left > std::numeric_limits<int>::max() / right) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 ") + label +
            " exceeds MLX limits");
    }
    return left * right;
}

array floating_contiguous(
    const array& input,
    Dtype preferred = mlx::core::float16) {
    auto result = input;
    if (result.dtype() != mlx::core::float16 &&
        result.dtype() != mlx::core::float32) {
        result = mlx::core::astype(result, preferred);
    }
    return mlx::core::contiguous(result);
}

array float32_contiguous(const array& input) {
    auto result = input;
    if (result.dtype() != mlx::core::float32) {
        result = mlx::core::astype(
            result,
            mlx::core::float32);
    }
    return mlx::core::contiguous(result);
}

array load_float_array(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" &&
        record.dtype != "F16" &&
        record.dtype != "F32") {
        throw std::runtime_error(
            "DeepSeek-V4 runtime tensor must be BF16/F16/F32: " +
            name);
    }
    return float32_contiguous(
        load_dense_array(
            record.dtype,
            model.read(name)));
}

array slice_last(
    const array& input,
    int begin,
    int end) {
    if (input.ndim() == 0 ||
        begin < 0 ||
        end < begin ||
        end > input.shape(-1)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 last-axis slice");
    }
    Shape start(input.ndim(), 0);
    Shape stop = input.shape();
    start.back() = begin;
    stop.back() = end;
    return mlx::core::slice(
        input,
        std::move(start),
        std::move(stop));
}

array softmax_last(const array& input) {
    auto source = float32_contiguous(input);
    auto maximum =
        mlx::core::max(source, -1, true);
    auto exponential =
        mlx::core::exp(source - maximum);
    return exponential /
        mlx::core::sum(exponential, -1, true);
}

MlxDeepseekV4HcPreResult hc_pre_generic(
    const array& residual,
    const array& mixes,
    const array& scale,
    const array& base,
    float eps) {
    if (residual.ndim() != 4 ||
        residual.shape(2) != kConnections ||
        residual.shape(3) <= 0 ||
        mixes.shape() != Shape{
            residual.shape(0),
            residual.shape(1),
            kHcProjectionWidth,
        } ||
        scale.size() != 3 ||
        base.size() != kHcProjectionWidth ||
        !std::isfinite(eps) ||
        eps <= 0.0f) {
        throw std::invalid_argument(
            "invalid generic DeepSeek-V4 HC input");
    }
    auto mix_values = float32_contiguous(mixes);
    auto scale_values = mlx::core::reshape(
        float32_contiguous(scale),
        Shape{3});
    auto base_values = mlx::core::reshape(
        float32_contiguous(base),
        Shape{kHcProjectionWidth});

    auto pre = mlx::core::sigmoid(
        slice_last(
            mix_values,
            0,
            kConnections) *
            slice_last(scale_values, 0, 1) +
        slice_last(
            base_values,
            0,
            kConnections)) +
        eps;
    auto post = 2.0f * mlx::core::sigmoid(
        slice_last(
            mix_values,
            kConnections,
            2 * kConnections) *
            slice_last(scale_values, 1, 2) +
        slice_last(
            base_values,
            kConnections,
            2 * kConnections));

    Shape matrix_shape{
        residual.shape(0),
        residual.shape(1),
        kConnections,
        kConnections,
    };
    auto combination = softmax_last(
        mlx::core::reshape(
            slice_last(
                mix_values,
                2 * kConnections,
                kHcProjectionWidth),
            matrix_shape) *
            slice_last(scale_values, 2, 3) +
        mlx::core::reshape(
            slice_last(
                base_values,
                2 * kConnections,
                kHcProjectionWidth),
            Shape{kConnections, kConnections})) +
        eps;
    combination = combination /
        (mlx::core::sum(
             combination,
             -2,
             true) +
         eps);
    for (int iteration = 1;
         iteration < 20;
         ++iteration) {
        combination = combination /
            (mlx::core::sum(
                 combination,
                 -1,
                 true) +
             eps);
        combination = combination /
            (mlx::core::sum(
                 combination,
                 -2,
                 true) +
             eps);
    }

    auto reduced = mlx::core::sum(
        mlx::core::expand_dims(pre, -1) *
            residual,
        2);
    return {
        std::move(reduced),
        std::move(post),
        std::move(combination),
    };
}

array hc_post_generic(
    const array& branch,
    const array& residual,
    const array& post,
    const array& combination) {
    if (residual.ndim() != 4 ||
        residual.shape(2) != kConnections ||
        branch.shape() != Shape{
            residual.shape(0),
            residual.shape(1),
            residual.shape(3),
        } ||
        post.shape() != Shape{
            residual.shape(0),
            residual.shape(1),
            kConnections,
        } ||
        combination.shape() != Shape{
            residual.shape(0),
            residual.shape(1),
            kConnections,
            kConnections,
        }) {
        throw std::invalid_argument(
            "invalid generic DeepSeek-V4 HC post input");
    }
    auto mixed = mlx::core::sum(
        mlx::core::expand_dims(
            combination,
            -1) *
            mlx::core::expand_dims(
                residual,
                3),
        2);
    auto result =
        mlx::core::expand_dims(post, -1) *
            mlx::core::expand_dims(branch, 2) +
        mixed;
    return result.dtype() == residual.dtype()
        ? result
        : mlx::core::astype(
              result,
              residual.dtype());
}

array all_experts_available(int experts) {
    return mlx::core::ones(
        Shape{experts},
        mlx::core::bool_);
}

std::vector<array> expert_availability(
    const MfqContainer& model,
    const DeepseekV4Config& config) {
    const int layers =
        checked_int(config.n_layers, "layer count");
    const int experts =
        checked_int(config.n_experts, "expert count");
    std::vector<array> result;
    result.reserve(layers);
    for (int layer = 0; layer < layers; ++layer) {
        result.push_back(
            all_experts_available(experts));
    }

    const auto found =
        model.header().extra_json.find(
            "cccp_manifest");
    if (found ==
        model.header().extra_json.end()) {
        return result;
    }

    json manifest;
    try {
        manifest = json::parse(found->second);
    } catch (const json::exception& error) {
        throw std::runtime_error(
            std::string(
                "invalid DeepSeek-V4 CCCP manifest: ") +
            error.what());
    }
    const auto assignments =
        manifest.find("tiers_per_layer");
    if (assignments == manifest.end() ||
        assignments->is_null()) {
        return result;
    }
    if (!assignments->is_object()) {
        throw std::runtime_error(
            "DeepSeek-V4 tiers_per_layer must be an object");
    }
    for (int layer = 0; layer < layers; ++layer) {
        const auto assigned =
            assignments->find(std::to_string(layer));
        if (assigned == assignments->end()) {
            continue;
        }
        if (!assigned->is_string()) {
            throw std::runtime_error(
                "DeepSeek-V4 tier assignment must be a string");
        }
        const auto tiers =
            assigned->get<std::string>();
        if (tiers.size() !=
            static_cast<std::size_t>(experts)) {
            throw std::runtime_error(
                "DeepSeek-V4 layer " +
                std::to_string(layer) +
                " tier assignment length does not match "
                "n_experts");
        }
        std::vector<std::uint8_t> available(
            static_cast<std::size_t>(experts));
        int count = 0;
        for (int expert = 0;
             expert < experts;
             ++expert) {
            const char tier =
                tiers[static_cast<std::size_t>(expert)];
            if (tier != 'x' &&
                tier != 'w' &&
                tier != 'v' &&
                tier != 'V' &&
                tier != 'd') {
                throw std::runtime_error(
                    "DeepSeek-V4 layer " +
                    std::to_string(layer) +
                    " contains an invalid expert tier");
            }
            available[static_cast<std::size_t>(expert)] =
                static_cast<std::uint8_t>(
                    tier != 'd');
            count += tier != 'd' ? 1 : 0;
        }
        if (count < config.top_k) {
            throw std::runtime_error(
                "DeepSeek-V4 layer " +
                std::to_string(layer) +
                " has fewer than top_k available experts");
        }
        result[static_cast<std::size_t>(layer)] =
            mlx::core::astype(
                array(
                    available.begin(),
                    Shape{experts}),
                mlx::core::bool_);
    }
    return result;
}

array normalize_eos_ids(
    const std::vector<std::int64_t>& values,
    int vocab) {
    std::vector<std::int32_t> result;
    result.reserve(values.size());
    for (const auto token : values) {
        if (token < 0 || token >= vocab) {
            throw std::invalid_argument(
                "DeepSeek-V4 EOS token is out of range");
        }
        result.push_back(
            static_cast<std::int32_t>(token));
    }
    return array(
        result.begin(),
        Shape{static_cast<int>(result.size())},
        mlx::core::int32);
}

array last_token_logits(
    const array& logits,
    int vocab,
    bool preserve_batch_axis = true) {
    if (logits.ndim() != 3 ||
        logits.shape(0) <= 0 ||
        logits.shape(1) <= 0 ||
        logits.shape(2) != vocab) {
        throw std::runtime_error(
            "DeepSeek-V4 logits must have "
            "[batch,tokens,vocab] shape");
    }
    auto sliced = mlx::core::slice(
        logits,
        Shape{0, logits.shape(1) - 1, 0},
        Shape{
            logits.shape(0),
            logits.shape(1),
            vocab,
        });
    if (!preserve_batch_axis) {
        return sliced;
    }
    return mlx::core::reshape(
        std::move(sliced),
        Shape{logits.shape(0), vocab});
}

bool contains_token(
    const std::vector<std::int64_t>& values,
    std::int64_t token) {
    return std::find(
        values.begin(),
        values.end(),
        token) != values.end();
}

} // namespace

MlxDeepseekV4Layer MlxDeepseekV4Layer::load(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    std::size_t index,
    int max_context,
    const array& available,
    std::pair<array, array> rope_base,
    std::pair<array, array> rope_compressed,
    std::shared_ptr<MlxCccpExpertResidency> residency) {
    config.validate();
    if (index >=
        static_cast<std::size_t>(
            config.n_layers)) {
        throw std::out_of_range(
            "DeepSeek-V4 layer index is out of range");
    }
    const int ratio = static_cast<int>(
        config.compress_ratios[index]);
    const auto name =
        [index](std::string_view suffix) {
            return DeepseekV4TensorNames::layer(
                index,
                suffix);
        };
    return MlxDeepseekV4Layer(
        config,
        index,
        {
            MlxDeepseekV4Attention::load(
                model,
                config,
                static_cast<int>(index),
                ratio,
                max_context,
                std::move(rope_base),
                std::move(rope_compressed)),
            MlxDeepseekV4Moe::load(
                model,
                config,
                index,
                available,
                std::move(residency)),
            load_float_array(
                model,
                name("attn_norm.weight")),
            load_float_array(
                model,
                name("ffn_norm.weight")),
            MlxLinear::load(
                model,
                name("hc_attn_fn")),
            load_float_array(
                model,
                name("hc_attn_base")),
            load_float_array(
                model,
                name("hc_attn_scale")),
            MlxLinear::load(
                model,
                name("hc_ffn_fn")),
            load_float_array(
                model,
                name("hc_ffn_base")),
            load_float_array(
                model,
                name("hc_ffn_scale")),
        });
}

MlxDeepseekV4Layer::MlxDeepseekV4Layer(
    DeepseekV4Config config,
    std::size_t index,
    MlxDeepseekV4LayerComponents components)
    : config_(std::move(config)),
      index_(index),
      ratio_(
          index_ < config_.compress_ratios.size()
          ? static_cast<int>(
                config_.compress_ratios[index_])
          : -1),
      components_(std::move(components)),
      attention_norm_(
          components_.attention_norm,
          static_cast<float>(
              config_.rms_eps)),
      ffn_norm_(
          components_.ffn_norm,
          static_cast<float>(
              config_.rms_eps)) {
    validate_components();
}

void MlxDeepseekV4Layer::validate_components() const {
    config_.validate();
    const int layers =
        checked_int(config_.n_layers, "layer count");
    const int hidden =
        checked_int(config_.hidden, "hidden size");
    const int hc_width = checked_product(
        kConnections,
        hidden,
        "hyper-connection width");
    if (index_ >=
            static_cast<std::size_t>(layers) ||
        ratio_ < 0 ||
        components_.attention.layer() !=
            static_cast<int>(index_) ||
        components_.attention.ratio() != ratio_ ||
        attention_norm_.width() != hidden ||
        ffn_norm_.width() != hidden ||
        components_.hc_attention_fn.input_size() !=
            hc_width ||
        components_.hc_attention_fn.output_size() !=
            kHcProjectionWidth ||
        components_.hc_ffn_fn.input_size() != hc_width ||
        components_.hc_ffn_fn.output_size() !=
            kHcProjectionWidth ||
        components_.hc_attention_base.size() !=
            kHcProjectionWidth ||
        components_.hc_attention_scale.size() != 3 ||
        components_.hc_ffn_base.size() !=
            kHcProjectionWidth ||
        components_.hc_ffn_scale.size() != 3) {
        throw std::invalid_argument(
            "DeepSeek-V4 layer component dimensions mismatch");
    }
}

MlxDeepseekV4HcPreResult
MlxDeepseekV4Layer::hc_pre(
    const array& residual,
    const MlxLinear& function,
    const array& scale,
    const array& base) const {
    const int batch = residual.shape(0);
    const int tokens = residual.shape(1);
    const int hidden =
        checked_int(config_.hidden, "hidden size");
    auto flat = mlx::core::reshape(
        residual,
        Shape{
            batch,
            tokens,
            checked_product(
                kConnections,
                hidden,
                "hyper-connection width"),
        });
    auto flat_float =
        mlx::core::astype(
            flat,
            mlx::core::float32);
    auto inverse = mlx::core::rsqrt(
        mlx::core::mean(
            flat_float * flat_float,
            -1,
            true) +
        static_cast<float>(
            config_.rms_eps));
    auto mixes =
        mlx::core::astype(
            function(flat),
            mlx::core::float32) *
        inverse;
    if (config_.fast_hyper_connections()) {
        return deepseek_v4_hc_pre(
            residual,
            mixes,
            scale,
            base,
            checked_int(
                config_.hc_sinkhorn_iters,
                "HC Sinkhorn iterations"),
            static_cast<float>(
                config_.hc_eps));
    }
    return hc_pre_generic(
        residual,
        mixes,
        scale,
        base,
        static_cast<float>(
            config_.hc_eps));
}

array MlxDeepseekV4Layer::hc_post(
    const array& branch,
    const array& residual,
    const array& post,
    const array& combination) const {
    if (config_.fast_hyper_connections()) {
        return deepseek_v4_hc_post(
            branch,
            residual,
            post,
            combination);
    }
    return hc_post_generic(
        branch,
        residual,
        post,
        combination);
}

array MlxDeepseekV4Layer::forward(
    const array& hidden,
    const array& token_ids,
    MlxDeepseekV4LayerState& state,
    int pos0) const {
    auto source = floating_contiguous(hidden);
    const int expected_hidden =
        checked_int(config_.hidden, "hidden size");
    if (source.ndim() != 4 ||
        source.shape(0) <= 0 ||
        source.shape(1) <= 0 ||
        source.shape(2) != kConnections ||
        source.shape(3) != expected_hidden ||
        token_ids.size() !=
            static_cast<std::size_t>(
                source.shape(0)) *
                source.shape(1) ||
        state.batch() != source.shape(0) ||
        state.position() != pos0) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 layer input/cache state");
    }

    auto residual = source;
    auto attention_hc = hc_pre(
        source,
        components_.hc_attention_fn,
        components_.hc_attention_scale,
        components_.hc_attention_base);
    auto branch = attention_norm_(
        attention_hc.reduced);
    branch = components_.attention(
        branch,
        state,
        pos0);
    auto result = hc_post(
        branch,
        residual,
        attention_hc.post,
        attention_hc.combination);

    residual = result;
    auto ffn_hc = hc_pre(
        result,
        components_.hc_ffn_fn,
        components_.hc_ffn_scale,
        components_.hc_ffn_base);
    branch = ffn_norm_(ffn_hc.reduced);
    branch = components_.moe(
        branch,
        token_ids);
    return hc_post(
        branch,
        residual,
        ffn_hc.post,
        ffn_hc.combination);
}

MlxDeepseekV4CausalLm
MlxDeepseekV4CausalLm::load(
    const MfqContainer& model,
    int max_context,
    std::size_t expert_cache_bytes) {
    return load(
        model,
        DeepseekV4Config::from_mfq(model),
        {},
        max_context,
        expert_cache_bytes);
}

MlxDeepseekV4CausalLm
MlxDeepseekV4CausalLm::load(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    const DeepseekV4TensorNames& names,
    int max_context,
    std::size_t expert_cache_bytes) {
    config.validate();
    validate_deepseek_v4_model_bindings(
        model,
        config,
        names);
    const int context = std::min(
        max_context,
        checked_int(
            config.max_position_embeddings,
            "maximum context"));
    if (context <= 0) {
        throw std::invalid_argument(
            "DeepSeek-V4 max_context must be positive");
    }
    auto rope_base =
        deepseek_v4_yarn_tables(
            checked_int(
                config.qk_rope_head_dim,
                "rotary dimension"),
            context,
            static_cast<float>(
                config.rope_theta));
    auto rope_compressed =
        deepseek_v4_yarn_tables(
            checked_int(
                config.qk_rope_head_dim,
                "rotary dimension"),
            context,
            static_cast<float>(
                config.compress_rope_theta),
            config.rope_scaling);
    auto availability =
        expert_availability(model, config);
    auto expert_residency =
        std::make_shared<MlxCccpExpertResidency>(
            model,
            expert_cache_bytes,
            checked_int(
                config.n_experts,
                "expert count"));
    std::vector<MlxDeepseekV4Layer> layers;
    layers.reserve(
        static_cast<std::size_t>(
            config.n_layers));
    for (std::size_t index = 0;
         index <
             static_cast<std::size_t>(
                 config.n_layers);
         ++index) {
        layers.push_back(
            MlxDeepseekV4Layer::load(
                model,
                config,
                index,
                context,
                availability.at(index),
                rope_base,
                rope_compressed,
                expert_residency));
    }
    return MlxDeepseekV4CausalLm(
        config,
        MlxEmbedding::load(
            model,
            names.embedding),
        std::move(layers),
        load_float_array(
            model,
            names.output_norm),
        MlxLinear::load(
            model,
            names.output),
        MlxLinear::load(
            model,
            names.hc_head_fn),
        load_float_array(
            model,
            names.hc_head_base),
        load_float_array(
            model,
            names.hc_head_scale),
        context,
        mlx::core::float16,
        std::move(expert_residency));
}

MlxDeepseekV4CausalLm::MlxDeepseekV4CausalLm(
    DeepseekV4Config config,
    MlxEmbedding embedding,
    std::vector<MlxDeepseekV4Layer> layers,
    array output_norm,
    MlxLinear output,
    MlxLinear hc_head_fn,
    array hc_head_base,
    array hc_head_scale,
    int max_context,
    Dtype activation_dtype,
    std::shared_ptr<MlxCccpExpertResidency>
        expert_residency)
    : config_(std::move(config)),
      embedding_(std::move(embedding)),
      layers_(std::move(layers)),
      output_norm_(
          std::move(output_norm),
          static_cast<float>(
              config_.rms_eps)),
      output_(std::move(output)),
      hc_head_fn_(std::move(hc_head_fn)),
      hc_head_base_(
          float32_contiguous(
              hc_head_base)),
      hc_head_scale_(
          float32_contiguous(
              hc_head_scale)),
      expert_residency_(
          std::move(expert_residency)),
      max_context_(max_context),
      activation_dtype_(activation_dtype) {
    validate_components();
}

void MlxDeepseekV4CausalLm::validate_components() const {
    config_.validate();
    const int layers =
        checked_int(config_.n_layers, "layer count");
    const int hidden =
        checked_int(config_.hidden, "hidden size");
    const int vocab =
        checked_int(config_.vocab, "vocabulary size");
    const int hc_width = checked_product(
        kConnections,
        hidden,
        "hyper-connection width");
    if (max_context_ <= 0 ||
        max_context_ >
            config_.max_position_embeddings ||
        (activation_dtype_ != mlx::core::float16 &&
         activation_dtype_ != mlx::core::float32) ||
        embedding_.vocabulary_size() != vocab ||
        embedding_.hidden_size() != hidden ||
        output_norm_.width() != hidden ||
        output_.input_size() != hidden ||
        output_.output_size() != vocab ||
        hc_head_fn_.input_size() != hc_width ||
        hc_head_fn_.output_size() != kConnections ||
        hc_head_base_.size() != kConnections ||
        hc_head_scale_.size() != 1 ||
        layers_.size() !=
            static_cast<std::size_t>(layers)) {
        throw std::invalid_argument(
            "DeepSeek-V4 causal LM component dimensions mismatch");
    }
    for (std::size_t index = 0;
         index < layers_.size();
         ++index) {
        const auto& layer = layers_[index];
        if (layer.index() != index ||
            layer.config().hidden != config_.hidden ||
            layer.config().vocab != config_.vocab ||
            layer.ratio() !=
                config_.compress_ratios[index] ||
            layer.max_context() != max_context_) {
            throw std::invalid_argument(
                "DeepSeek-V4 causal LM layer mismatch at index " +
                std::to_string(index));
        }
    }
}

array MlxDeepseekV4CausalLm::normalize_ids(
    const array& token_ids,
    bool allow_empty) const {
    auto ids = token_ids;
    if (ids.ndim() == 1) {
        ids = mlx::core::expand_dims(
            ids,
            0);
    }
    if (ids.ndim() != 2 ||
        ids.shape(0) <= 0 ||
        (!allow_empty && ids.shape(1) <= 0)) {
        throw std::invalid_argument(
            "DeepSeek-V4 input IDs must have "
            "[batch,tokens] shape");
    }
    if (ids.dtype() != mlx::core::int32 &&
        ids.dtype() != mlx::core::int64 &&
        ids.dtype() != mlx::core::uint32 &&
        ids.dtype() != mlx::core::uint64) {
        throw std::invalid_argument(
            "DeepSeek-V4 input IDs must be integers");
    }
    if (ids.dtype() != mlx::core::int32) {
        ids = mlx::core::astype(
            ids,
            mlx::core::int32);
    }
    return mlx::core::contiguous(ids);
}

void MlxDeepseekV4CausalLm::reset_cache(
    int batch) {
    if (batch <= 0) {
        throw std::invalid_argument(
            "DeepSeek-V4 cache batch must be positive");
    }
    std::vector<MlxDeepseekV4LayerState> states;
    states.reserve(layers_.size());
    for (const auto& layer : layers_) {
        states.push_back(
            MlxDeepseekV4LayerState::allocate(
                config_,
                layer.ratio(),
                batch,
                max_context_,
                activation_dtype_));
    }
    states_ = std::move(states);
    cache_position_ = 0;
    cache_batch_ = batch;
}

void MlxDeepseekV4CausalLm::clear_cache() noexcept {
    states_.clear();
    cache_position_ = 0;
    cache_batch_ = 0;
}

void MlxDeepseekV4CausalLm::materialize_state(
    const MlxDeepseekV4LayerState& state) const {
    std::vector<array> arrays{
        state.local(),
        state.local_positions(),
    };
    const auto append_pool =
        [&](const MlxDeepseekV4PoolState& pool) {
            arrays.push_back(pool.pool());
            arrays.push_back(pool.state_kv());
            arrays.push_back(pool.state_gate());
            if (pool.prev_kv()) {
                arrays.push_back(*pool.prev_kv());
            }
            if (pool.prev_gate()) {
                arrays.push_back(*pool.prev_gate());
            }
        };
    if (state.main()) {
        append_pool(*state.main());
    }
    if (state.indexer()) {
        append_pool(*state.indexer());
    }
    detail::eval_with_timing(std::move(arrays));
}

array MlxDeepseekV4CausalLm::head(
    const array& hidden) const {
    const int batch = hidden.shape(0);
    const int tokens = hidden.shape(1);
    const int width = checked_product(
        kConnections,
        checked_int(
            config_.hidden,
            "hidden size"),
        "hyper-connection width");
    auto flat = mlx::core::reshape(
        hidden,
        Shape{batch, tokens, width});
    auto flat_float =
        mlx::core::astype(
            flat,
            mlx::core::float32);
    auto inverse = mlx::core::rsqrt(
        mlx::core::mean(
            flat_float * flat_float,
            -1,
            true) +
        static_cast<float>(
            config_.rms_eps));
    auto mixes =
        mlx::core::astype(
            hc_head_fn_(flat),
            mlx::core::float32) *
        inverse;
    auto pre = mlx::core::sigmoid(
        mixes *
            mlx::core::reshape(
                hc_head_scale_,
                Shape{1}) +
        mlx::core::reshape(
            hc_head_base_,
            Shape{kConnections})) +
        static_cast<float>(
            config_.hc_eps);
    auto reduced = mlx::core::sum(
        mlx::core::expand_dims(pre, -1) *
            hidden,
        2);
    return output_norm_(reduced);
}

array MlxDeepseekV4CausalLm::forward_chunk(
    const array& token_ids,
    int pos0) {
    if (cache_batch_ == 0 ||
        states_.size() != layers_.size() ||
        token_ids.ndim() != 2 ||
        token_ids.shape(0) != cache_batch_ ||
        token_ids.shape(1) <= 0 ||
        pos0 != cache_position_) {
        throw std::runtime_error(
            "invalid DeepSeek-V4 forward chunk/cache state");
    }
    const int batch = token_ids.shape(0);
    const int tokens = token_ids.shape(1);
    const int hidden =
        checked_int(config_.hidden, "hidden size");
    auto embedded =
        embedding_(
            token_ids,
            activation_dtype_);
    auto streams = mlx::core::broadcast_to(
        mlx::core::expand_dims(
            embedded,
            2),
        Shape{
            batch,
            tokens,
            kConnections,
            hidden,
        });
    auto hidden_values =
        mlx::core::contiguous(streams);
    for (std::size_t index = 0;
         index < layers_.size();
         ++index) {
        hidden_values = layers_[index](
            hidden_values,
            token_ids,
            states_[index],
            pos0);
        detail::eval_with_timing(hidden_values);
        materialize_state(
            states_[index]);
    }
    auto logits = mlx::core::astype(
        output_(head(hidden_values)),
        mlx::core::float32);
    for (const auto& state : states_) {
        materialize_state(state);
    }
    return logits;
}

array MlxDeepseekV4CausalLm::forward(
    const array& token_ids,
    bool use_cache) {
    auto ids = normalize_ids(
        token_ids,
        true);
    const int batch = ids.shape(0);
    const int tokens = ids.shape(1);
    if (tokens == 0) {
        return mlx::core::zeros(
            Shape{
                batch,
                0,
                checked_int(
                    config_.vocab,
                    "vocabulary size"),
            },
            mlx::core::float32);
    }
    if (!use_cache ||
        cache_batch_ == 0 ||
        cache_batch_ != batch) {
        reset_cache(batch);
    }
    if (tokens >
        max_context_ - cache_position_) {
        throw std::invalid_argument(
            "DeepSeek-V4 decode exceeds max_context");
    }
    const int start = cache_position_;
    auto logits = forward_chunk(ids, start);
    cache_position_ += tokens;
    return logits;
}

array MlxDeepseekV4CausalLm::prefill(
    const array& token_ids,
    int chunk_size,
    bool full_logits) {
    return prefill_impl(
        token_ids,
        chunk_size,
        full_logits,
        true);
}

array MlxDeepseekV4CausalLm::prefill_impl(
    const array& token_ids,
    int chunk_size,
    bool full_logits,
    bool reset) {
    auto ids = normalize_ids(
        token_ids,
        false);
    const int batch = ids.shape(0);
    const int tokens = ids.shape(1);
    if (chunk_size <= 0) {
        throw std::invalid_argument(
            "DeepSeek-V4 prefill chunk_size must be positive");
    }
    if (tokens > max_context_) {
        throw std::invalid_argument(
            "DeepSeek-V4 prefill exceeds max_context");
    }
    if (reset) {
        reset_cache(batch);
    } else if (cache_batch_ != batch ||
               cache_position_ != 0 ||
               states_.size() != layers_.size()) {
        throw std::runtime_error(
            "DeepSeek-V4 prepared prefill cache is invalid");
    }
    std::vector<array> outputs;
    if (full_logits) {
        outputs.reserve(
            static_cast<std::size_t>(
                (tokens + chunk_size - 1) /
                chunk_size));
    }
    std::optional<array> last;
    for (int start = 0;
         start < tokens;
         start += chunk_size) {
        const int end = std::min(
            tokens,
            start + chunk_size);
        auto chunk_ids = mlx::core::slice(
            ids,
            Shape{0, start},
            Shape{batch, end});
        auto chunk = forward_chunk(
            chunk_ids,
            start);
        cache_position_ = end;
        if (full_logits) {
            outputs.push_back(
                std::move(chunk));
        } else {
            last = std::move(chunk);
        }
    }
    if (full_logits) {
        return outputs.size() == 1
            ? std::move(outputs.front())
            : mlx::core::concatenate(
                  std::move(outputs),
                  1);
    }
    return last_token_logits(
        *last,
        checked_int(
            config_.vocab,
            "vocabulary size"));
}

array MlxDeepseekV4CausalLm::decode(
    const array& token_ids) {
    if (cache_batch_ == 0) {
        throw std::runtime_error(
            "DeepSeek-V4 decode requires prefill first");
    }
    return forward(token_ids, true);
}

const MlxDeepseekV4LayerState&
MlxDeepseekV4CausalLm::layer_state(
    std::size_t index) const {
    if (cache_batch_ == 0) {
        throw std::runtime_error(
            "DeepSeek-V4 layer state requires an active cache");
    }
    return states_.at(index);
}

bool MlxDeepseekV4CausalLm::uses_streamed_experts()
    const noexcept {
    return std::any_of(
        layers_.begin(),
        layers_.end(),
        [](const MlxDeepseekV4Layer& layer) {
            return layer.uses_streamed_experts();
        });
}

std::size_t
MlxDeepseekV4CausalLm::expert_cache_limit_bytes()
    const noexcept {
    return expert_residency_
        ? expert_residency_->cache_limit_bytes()
        : 0;
}

std::size_t
MlxDeepseekV4CausalLm::expert_resident_packed_bytes()
    const {
    return expert_residency_
        ? expert_residency_->resident_packed_bytes()
        : 0;
}

std::size_t
MlxDeepseekV4CausalLm::cached_expert_count() const {
    return expert_residency_
        ? expert_residency_->cached_expert_count()
        : 0;
}

void MlxDeepseekV4CausalLm::clear_expert_cache() {
    if (expert_residency_) {
        expert_residency_->clear();
    }
}

std::int32_t MlxDeepseekV4CausalLm::generate(
    const std::vector<std::int64_t>& prompt,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const MlxDeepseekV4TokenCallback& callback,
    const std::optional<std::vector<std::int64_t>>&
        eos_token_ids,
    int chunk_size,
    const std::function<void(std::size_t, double)>&
        prefill_callback) {
    if (prompt.empty()) {
        throw std::invalid_argument(
            "DeepSeek-V4 generation prompt cannot be empty");
    }
    if (max_tokens < 0) {
        throw std::invalid_argument(
            "DeepSeek-V4 generation max_tokens cannot be negative");
    }
    if (chunk_size <= 0) {
        throw std::invalid_argument(
            "DeepSeek-V4 generation chunk_size must be positive");
    }
    const int vocab =
        checked_int(config_.vocab, "vocabulary size");
    if (prompt.size() >
            static_cast<std::size_t>(
                max_context_) ||
        static_cast<std::size_t>(
            max_tokens) >
            static_cast<std::size_t>(
                max_context_) -
                prompt.size()) {
        throw std::invalid_argument(
            "DeepSeek-V4 generation exceeds max_context");
    }
    std::vector<std::int32_t> prompt_values;
    prompt_values.reserve(prompt.size());
    for (const auto token : prompt) {
        if (token < 0 || token >= vocab) {
            throw std::invalid_argument(
                "DeepSeek-V4 generation prompt token is out of range");
        }
        prompt_values.push_back(
            static_cast<std::int32_t>(token));
    }
    const auto& eos = eos_token_ids.has_value()
        ? *eos_token_ids
        : config_.eos_token_id;
    (void)normalize_eos_ids(eos, vocab);
    if (max_tokens == 0) {
        return 0;
    }

    const int prompt_count =
        static_cast<int>(
            prompt_values.size());
    const array prompt_ids(
        prompt_values.begin(),
        Shape{1, prompt_count},
        mlx::core::int32);
    auto counts = sample_token_counts_add(
        mlx::core::zeros(
            Shape{vocab},
            mlx::core::int32),
        prompt_ids);
    // State allocation/zeroing is request setup. Keep it in TTFT while
    // materializing it before the model-evaluation prefill metric starts.
    reset_cache(1);
    for (const auto& state : states_) {
        materialize_state(state);
    }
    double prefill_evaluation_ms = 0.0;
    auto logits = [&]() {
        detail::ScopedMlxEvaluationTiming timing(
            prefill_callback
                ? &prefill_evaluation_ms
                : nullptr);
        auto value = prefill_impl(
            prompt_ids,
            chunk_size,
            false,
            false);
        if (prefill_callback) {
            // DeepSeek evaluates layer/state chunks eagerly. Their wrapped
            // eval calls accumulate above; this final eval adds only the
            // remaining output-head graph.
            detail::eval_with_timing(value);
        }
        return value;
    }();
    if (prefill_callback) {
        prefill_callback(
            prompt.size(),
            prefill_evaluation_ms);
    }
    MlxSampler sampler(sampling);

    std::int32_t generated = 0;
    while (generated < max_tokens) {
        auto sampled = sampler.sample(
            logits,
            counts);
        sampled.eval();
        const auto token =
            sampled.data<std::int32_t>()[0];
        if (token < 0 || token >= vocab) {
            throw std::runtime_error(
                "DeepSeek-V4 sampler returned an "
                "out-of-range token");
        }
        const array token_ids(
            {token},
            Shape{1, 1},
            mlx::core::int32);
        counts = sample_token_counts_add(
            counts,
            token_ids);
        ++generated;
        if (callback &&
            !callback(
                static_cast<std::int64_t>(
                    token))) {
            break;
        }
        if (contains_token(eos, token) ||
            generated == max_tokens) {
            break;
        }
        logits = last_token_logits(
            decode(token_ids),
            vocab);
    }
    return generated;
}

} // namespace mfq::metal
