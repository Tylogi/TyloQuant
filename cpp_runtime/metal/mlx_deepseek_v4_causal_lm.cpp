#include "mlx_deepseek_v4_causal_lm.h"
#include "mlx_eval_timing.h"

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
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
    const auto mapped = model.map_record(name);
    return float32_contiguous(
        load_dense_array(
            record.dtype,
            mapped.view()));
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
            "tpq_manifest");
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
                "invalid DeepSeek-V4 TPQ manifest: ") +
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
    std::shared_ptr<MlxNintMoeOffloadCache> offload) {
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
                std::move(offload)),
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
MlxDeepseekV4Layer::hc_pre_norm(
    const array& residual,
    const MlxLinear& function,
    const array& scale,
    const array& base,
    const array& norm,
    const MlxRmsNorm& normalizer) const {
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
    auto raw_mixes = mlx::core::astype(
        function(flat),
        mlx::core::float32);
    if (config_.fast_hyper_connections()) {
        return deepseek_v4_hc_pre_norm(
            residual,
            raw_mixes,
            scale,
            base,
            norm,
            checked_int(
                config_.hc_sinkhorn_iters,
                "HC Sinkhorn iterations"),
            static_cast<float>(
                config_.hc_eps),
            static_cast<float>(
                config_.rms_eps),
            true);
    }
    auto flat_float = mlx::core::astype(
        flat,
        mlx::core::float32);
    auto inverse = mlx::core::rsqrt(
        mlx::core::mean(
            flat_float * flat_float,
            -1,
            true) +
        static_cast<float>(
            config_.rms_eps));
    auto mixes = raw_mixes * inverse;
    auto result = hc_pre_generic(
        residual,
        mixes,
        scale,
        base,
        static_cast<float>(
            config_.hc_eps));
    result.reduced = normalizer(result.reduced);
    return result;
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
    auto attention_hc = hc_pre_norm(
        source,
        components_.hc_attention_fn,
        components_.hc_attention_scale,
        components_.hc_attention_base,
        components_.attention_norm,
        attention_norm_);
    auto branch = attention_hc.reduced;
    detail::profile_eval(
        "layer.attention_hc_pre_norm",
        branch);
    branch = components_.attention(
        branch,
        state,
        pos0);
    auto result = hc_post(
        branch,
        residual,
        attention_hc.post,
        attention_hc.combination);
    detail::profile_eval(
        "layer.attention_hc_post",
        result);

    residual = result;
    auto ffn_hc = hc_pre_norm(
        result,
        components_.hc_ffn_fn,
        components_.hc_ffn_scale,
        components_.hc_ffn_base,
        components_.ffn_norm,
        ffn_norm_);
    branch = ffn_hc.reduced;
    detail::profile_eval(
        "layer.ffn_hc_pre_norm",
        branch);
    auto moe_branches = components_.moe.forward_branches(
        branch,
        token_ids);
    auto output = config_.fast_hyper_connections()
        ? deepseek_v4_hc_post_sum(
              moe_branches.routed,
              moe_branches.shared,
              residual,
              ffn_hc.post,
              ffn_hc.combination)
        : hc_post(
              moe_branches.routed + moe_branches.shared,
              residual,
              ffn_hc.post,
              ffn_hc.combination);
    detail::profile_eval(
        "layer.ffn_hc_post",
        output);
    return output;
}

MlxDeepseekV4CausalLm
MlxDeepseekV4CausalLm::load(
    const MfqContainer& model,
    int max_context,
    std::optional<std::size_t> expert_cache_bytes) {
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
    std::optional<std::size_t> expert_cache_bytes) {
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
    std::shared_ptr<MlxNintMoeOffloadCache>
        expert_offload;
    if (expert_cache_bytes.has_value()) {
        expert_offload =
            std::make_shared<MlxNintMoeOffloadCache>(
                model,
                *expert_cache_bytes,
                checked_int(
                    config.n_experts,
                    "expert count"));
    }
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
                expert_offload));
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
        std::move(expert_offload));
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
    std::shared_ptr<MlxNintMoeOffloadCache>
        expert_offload)
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
      expert_offload_(
          std::move(expert_offload)),
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
    stable_cache_tokens_.clear();
}

void MlxDeepseekV4CausalLm::clear_cache() noexcept {
    states_.clear();
    cache_position_ = 0;
    cache_batch_ = 0;
    stable_cache_tokens_.clear();
}

void MlxDeepseekV4CausalLm::append_state_arrays(
    const MlxDeepseekV4LayerState& state,
    std::vector<array>& arrays) const {
    arrays.push_back(state.local_state());
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
            if (pool.pool_prefix_backup()) {
                arrays.push_back(*pool.pool_prefix_backup());
            }
        };
    if (state.main()) {
        append_pool(*state.main());
    }
    if (state.indexer()) {
        append_pool(*state.indexer());
    }
}

void MlxDeepseekV4CausalLm::materialize_state(
    const MlxDeepseekV4LayerState& state) const {
    std::vector<array> arrays;
    arrays.reserve(11);
    append_state_arrays(state, arrays);
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
    int pos0,
    bool full_logits) {
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
    detail::profile_eval(
        "model.embedding_broadcast",
        hidden_values);
    const bool bounded_prefill = tokens > 1;
    for (std::size_t index = 0;
         index < layers_.size();
         ++index) {
        hidden_values = layers_[index](
            hidden_values,
            token_ids,
            states_[index],
            pos0);
        if (bounded_prefill) {
            // Hidden and cache branches share the layer projections. Submit
            // them together so prefill needs one GPU synchronization per
            // layer instead of separately waiting for hidden and state.
            std::vector<array> layer_outputs{
                hidden_values,
            };
            layer_outputs.reserve(12);
            append_state_arrays(
                states_[index],
                layer_outputs);
            detail::eval_with_timing(
                std::move(layer_outputs));
        }
    }
    auto head_input = full_logits
        ? hidden_values
        : mlx::core::slice(
              hidden_values,
              Shape{0, tokens - 1, 0, 0},
              Shape{
                  batch,
                  tokens,
                  kConnections,
                  hidden,
              });
    auto headed = head(head_input);
    detail::profile_eval(
        "model.final_hc_norm",
        headed);
    auto logits = mlx::core::astype(
        output_(headed),
        mlx::core::float32);
    detail::profile_eval(
        "model.lm_head_cast",
        logits);
    if (!bounded_prefill) {
        // Decode owns one token and a bounded cache update per layer.  Keep
        // the complete 43-layer graph lazy, then materialize logits and every
        // updated cache array together.  Evaluating hidden/state after every
        // layer fragmented one token into roughly 86 Metal synchronizations.
        std::vector<array> outputs{logits};
        const auto append_pool =
            [&outputs](const MlxDeepseekV4PoolState& pool) {
                outputs.push_back(pool.pool());
                outputs.push_back(pool.state_kv());
                outputs.push_back(pool.state_gate());
                if (pool.prev_kv()) {
                    outputs.push_back(*pool.prev_kv());
                }
                if (pool.prev_gate()) {
                    outputs.push_back(*pool.prev_gate());
                }
            };
        for (const auto& state : states_) {
            outputs.push_back(state.local_state());
            if (state.main()) {
                append_pool(*state.main());
            }
            if (state.indexer()) {
                append_pool(*state.indexer());
            }
        }
        if (detail::component_profile_active()) {
            detail::profile_eval(
                "model.finalize_cache",
                std::move(outputs));
        } else {
            detail::eval_with_timing(
                std::move(outputs));
        }
    }
    return logits;
}

array MlxDeepseekV4CausalLm::forward(
    const array& token_ids,
    bool use_cache) {
    stable_cache_tokens_.clear();
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
    auto logits = forward_chunk(ids, start, true);
    cache_position_ += tokens;
    return logits;
}

array MlxDeepseekV4CausalLm::prefill(
    const array& token_ids,
    int chunk_size,
    bool full_logits) {
    stable_cache_tokens_.clear();
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
    const int initial_position = reset ? 0 : cache_position_;
    if (initial_position < 0 ||
        tokens > max_context_ - initial_position) {
        throw std::invalid_argument(
            "DeepSeek-V4 prefill exceeds max_context");
    }
    if (reset) {
        reset_cache(batch);
    } else if (cache_batch_ != batch ||
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
            initial_position + start,
            full_logits);
        cache_position_ = initial_position + end;
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
    stable_cache_tokens_.clear();
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
    return expert_offload_
        ? expert_offload_->cache_limit_bytes()
        : 0;
}

std::size_t
MlxDeepseekV4CausalLm::expert_resident_packed_bytes()
    const {
    return expert_offload_
        ? expert_offload_->resident_packed_bytes()
        : 0;
}

std::size_t
MlxDeepseekV4CausalLm::cached_expert_count() const {
    return expert_offload_
        ? expert_offload_->cached_expert_count()
        : 0;
}

void MlxDeepseekV4CausalLm::clear_expert_cache() {
    if (expert_offload_) {
        expert_offload_->clear();
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
        prefill_callback,
    std::optional<std::size_t> stable_prefix_tokens,
    const MfqTokenConstraintPtr& token_constraint) {
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
    const bool retain_stable_prefix =
        stable_prefix_tokens.has_value() &&
        *stable_prefix_tokens > 0 &&
        *stable_prefix_tokens < prompt.size();
    const std::size_t stable_count = retain_stable_prefix
        ? *stable_prefix_tokens
        : 0;
    std::size_t reused_tokens = 0;
    if (retain_stable_prefix &&
        cache_batch_ == 1 &&
        cache_position_ ==
            static_cast<int>(stable_cache_tokens_.size()) &&
        !stable_cache_tokens_.empty() &&
        stable_cache_tokens_.size() <= stable_count &&
        stable_cache_tokens_.size() <= prompt.size() &&
        std::equal(
            stable_cache_tokens_.begin(),
            stable_cache_tokens_.end(),
            prompt.begin())) {
        reused_tokens = stable_cache_tokens_.size();
    } else {
        // State allocation/zeroing is request setup. Keep it in TTFT while
        // materializing it before the model-evaluation prefill metric starts.
        reset_cache(1);
        for (const auto& state : states_) {
            materialize_state(state);
        }
    }

    struct StableCacheRestore {
        std::vector<MlxDeepseekV4LayerState>& target_states;
        int& target_position;
        int& target_batch;
        std::vector<std::int64_t>& target_tokens;
        std::optional<std::vector<MlxDeepseekV4LayerState>> saved_states;
        std::vector<std::int64_t> saved_tokens;
        int saved_position = 0;
        int saved_batch = 0;

        StableCacheRestore(
            std::vector<MlxDeepseekV4LayerState>& states,
            int& position,
            int& batch,
            std::vector<std::int64_t>& tokens)
            : target_states(states),
              target_position(position),
              target_batch(batch),
              target_tokens(tokens) {}

        void capture(
            const std::vector<std::int64_t>& prompt_tokens,
            std::size_t count) {
            saved_states.emplace();
            saved_states->reserve(target_states.size());
            for (const auto& state : target_states) {
                saved_states->push_back(
                    state.snapshot());
            }
            saved_tokens.assign(
                prompt_tokens.begin(),
                prompt_tokens.begin() +
                    static_cast<std::ptrdiff_t>(count));
            saved_position = static_cast<int>(count);
            saved_batch = target_batch;
        }

        bool active() const noexcept {
            return saved_states.has_value();
        }

        const std::vector<MlxDeepseekV4LayerState>&
        states() const {
            return *saved_states;
        }

        ~StableCacheRestore() noexcept {
            if (!saved_states) return;
            try {
                if (target_states.size() !=
                    saved_states->size()) {
                    throw std::runtime_error(
                        "DeepSeek-V4 stable cache layer count changed");
                }
                for (std::size_t index = 0;
                     index < target_states.size();
                     ++index) {
                    target_states[index].restore_snapshot(
                        std::move((*saved_states)[index]));
                }
                target_position = saved_position;
                target_batch = saved_batch;
                target_tokens = std::move(saved_tokens);
            } catch (...) {
                target_states.clear();
                target_position = 0;
                target_batch = 0;
                target_tokens.clear();
            }
        }
    } stable_restore(
        states_,
        cache_position_,
        cache_batch_,
        stable_cache_tokens_);

    const std::size_t evaluated_prompt_tokens =
        prompt.size() - reused_tokens;
    double prefill_evaluation_ms = 0.0;
    const bool profile_prefill =
        detail::component_profile_requested();
    detail::ComponentProfile prefill_component_profile;
    const auto prefill_component_started =
        std::chrono::steady_clock::now();
    auto logits = [&]() {
        detail::ScopedComponentProfile component_scope(
            profile_prefill
                ? &prefill_component_profile
                : nullptr);
        detail::ScopedMlxEvaluationTiming timing(
            prefill_callback
                ? &prefill_evaluation_ms
                : nullptr);
        const auto prefill_range = [&](std::size_t begin, std::size_t end) {
            auto ids = mlx::core::slice(
                prompt_ids,
                Shape{0, static_cast<int>(begin)},
                Shape{1, static_cast<int>(end)});
            return prefill_impl(
                ids,
                chunk_size,
                false,
                false);
        };

        std::optional<array> stable_logits;
        array value = [&]() {
            if (!retain_stable_prefix) {
                return prefill_range(0, prompt.size());
            }
            if (reused_tokens < stable_count) {
                stable_logits = prefill_range(
                    reused_tokens,
                    stable_count);
            }
            for (const auto& state : states_) {
                materialize_state(state);
            }
            stable_restore.capture(prompt, stable_count);
            // Materialize every copy before evaluating the suffix. Otherwise
            // the lazy copy graph would still read arrays after the suffix or
            // decode kernels had modified them in place.
            for (const auto& state : stable_restore.states()) {
                materialize_state(state);
            }
            if (stable_count < prompt.size()) {
                return prefill_range(
                    stable_count,
                    prompt.size());
            }
            if (!stable_logits) {
                throw std::runtime_error(
                    "DeepSeek-V4 stable cache has no logits for sampling");
            }
            return std::move(*stable_logits);
        }();
        if (prefill_callback) {
            // DeepSeek evaluates layer/state chunks eagerly. Their wrapped
            // eval calls accumulate above; this final eval adds only the
            // remaining output-head graph.
            detail::eval_with_timing(value);
        }
        return value;
    }();
    if (profile_prefill) {
        const double wall_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now()
                - prefill_component_started)
                .count();
        const double evaluated_ms =
            prefill_component_profile.evaluated_ms();
        std::cout
            << "component_profile phase=prefill"
            << " tokens=" << evaluated_prompt_tokens
            << " reused_tokens=" << reused_tokens
            << " wall_ms=" << std::fixed
            << std::setprecision(3) << wall_ms
            << " evaluated_ms=" << evaluated_ms
            << " unscoped_ms="
            << std::max(0.0, wall_ms - evaluated_ms)
            << std::endl;
        for (const auto& [name, timing] :
             prefill_component_profile.timings()) {
            std::cout
                << "component_cost phase=prefill"
                << " name=" << name
                << " ms=" << timing.elapsed_ms
                << " calls=" << timing.evaluations
                << " pct_evaluated="
                << (
                    evaluated_ms > 0.0
                        ? 100.0 * timing.elapsed_ms
                            / evaluated_ms
                        : 0.0
                )
                << std::endl;
        }
    }
    if (prefill_callback) {
        prefill_callback(
            evaluated_prompt_tokens,
            prefill_evaluation_ms);
    }
    MlxSampler sampler(sampling);

    std::int32_t generated = 0;
    while (generated < max_tokens) {
        const int decode_step = generated;
        const int profile_skip =
            detail::component_profile_skip_steps();
        const bool profile_this_step =
            detail::component_profile_requested()
            && decode_step >= profile_skip
            && decode_step - profile_skip
                < detail::component_profile_steps();
        detail::ComponentProfile component_profile;
        detail::ScopedComponentProfile profile_scope(
            profile_this_step
                ? &component_profile
                : nullptr);
        const auto component_started =
            std::chrono::steady_clock::now();
        const auto report_components = [&] {
            if (!profile_this_step) {
                return;
            }
            const double wall_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now()
                    - component_started)
                    .count();
            const double evaluated_ms =
                component_profile.evaluated_ms();
            std::cout
                << "component_profile"
                << " step=" << decode_step
                << " cache_position=" << cache_position_
                << " wall_ms=" << std::fixed
                << std::setprecision(3) << wall_ms
                << " evaluated_ms=" << evaluated_ms
                << " unscoped_ms="
                << std::max(0.0, wall_ms - evaluated_ms)
                << std::endl;
            for (const auto& [name, timing] :
                 component_profile.timings()) {
                std::cout
                    << "component_cost"
                    << " step=" << decode_step
                    << " name=" << name
                    << " ms=" << timing.elapsed_ms
                    << " calls=" << timing.evaluations
                    << " pct_evaluated="
                    << (
                        evaluated_ms > 0.0
                            ? 100.0 * timing.elapsed_ms
                                / evaluated_ms
                            : 0.0
                    )
                    << std::endl;
            }
        };
        auto sampled = sampler.sample(
            logits,
            counts);
        if (profile_this_step) {
            detail::profile_eval(
                "model.sampling",
                sampled);
        } else {
            sampled.eval();
        }
        auto token =
            sampled.data<std::int32_t>()[0];
        if (token < 0 || token >= vocab) {
            throw std::runtime_error(
                "DeepSeek-V4 sampler returned an "
                "out-of-range token");
        }
        if (token_constraint && token_constraint->allows &&
            !token_constraint->allows(token)) {
            auto adjusted = mlx::core::contiguous(
                mlx::core::astype(
                    sampler.apply_penalties(logits, counts),
                    mlx::core::float32));
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
                    "DeepSeek-V4 constrained sampler returned an "
                    "invalid token");
            }
        }
        if (token_constraint && token_constraint->accept) {
            token_constraint->accept(token);
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
            report_components();
            break;
        }
        auto decoded = forward_chunk(
            token_ids,
            cache_position_,
            true);
        ++cache_position_;
        logits = last_token_logits(
            decoded,
            vocab);
        report_components();
    }
    return generated;
}

} // namespace mfq::metal
