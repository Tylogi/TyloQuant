#include "mlx_deepseek_v41_moe.h"

#include "mlx_eval_timing.h"
#include "mlx_moe_ops.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

int checked_int(std::int64_t value, const char* name) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            std::string("DeepSeek-V4.1 ") + name + " is out of range");
    }
    return static_cast<int>(value);
}

MlxMoeWeight moe_weight(
    const MfqContainer& model,
    const std::string& name) {
    if (model.record(name).dtype != "NINTM") {
        throw std::runtime_error(
            "DeepSeek-V4.1 routed expert tensor must use NINTM: " + name);
    }
    const auto mapped = model.map_record(name);
    return MlxMoeWeight::from_blob(mapped.view());
}

array limited_swiglu(
    const array& gate,
    const array& up,
    float limit) {
    auto selected_gate = gate;
    auto selected_up = up;
    if (limit > 0.0f) {
        selected_gate = mlx::core::minimum(selected_gate, array(limit));
        selected_up = mlx::core::maximum(
            mlx::core::minimum(selected_up, array(limit)),
            array(-limit));
    }
    return selected_gate * mlx::core::sigmoid(selected_gate) * selected_up;
}

} // namespace

MlxDeepseekV41Moe MlxDeepseekV41Moe::load(
    const MfqContainer& model,
    const DeepseekV41Config& config,
    const std::string& prefix,
    bool predictor,
    std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache,
    std::size_t expert_cache_layer) {
    const int experts = checked_int(
        predictor ? config.dspark_n_experts : config.n_experts,
        "expert count");
    const int top_k = checked_int(
        predictor ? config.dspark_top_k : config.top_k,
        "route count");
    std::optional<MlxRoutedLinear> gate_up;
    std::optional<MlxRoutedLinear> down;
    if (!ssd_expert_cache) {
        gate_up.emplace(load_routed_gate_up_weight(model, prefix));
        down.emplace(moe_weight(model, prefix + ".experts.down.weight"));
    }
    return MlxDeepseekV41Moe(
        checked_int(config.hidden, "hidden size"),
        checked_int(config.moe_inter, "expert intermediate size"),
        experts,
        top_k,
        config.norm_topk_prob,
        static_cast<float>(config.routed_scaling),
        static_cast<float>(config.swiglu_limit),
        MlxLinear::load(model, prefix + ".router.weight"),
        load_dense_array(
            model.record(prefix + ".router.bias").dtype,
            model.map_record(prefix + ".router.bias").view()),
        model.contains(prefix + ".router.vision_bias")
            ? std::optional<array>(load_dense_array(
                  model.record(prefix + ".router.vision_bias").dtype,
                  model.map_record(prefix + ".router.vision_bias").view()))
            : std::nullopt,
        MlxLinear::load(model, prefix + ".shared_expert.gate.weight"),
        MlxLinear::load(model, prefix + ".shared_expert.up.weight"),
        MlxLinear::load(model, prefix + ".shared_expert.down.weight"),
        std::move(gate_up),
        std::move(down),
        std::move(ssd_expert_cache),
        expert_cache_layer);
}

MlxDeepseekV41Moe::MlxDeepseekV41Moe(
    int hidden,
    int intermediate,
    int experts,
    int top_k,
    bool normalize,
    float scale,
    float swiglu_limit,
    MlxLinear router,
    array router_bias,
    std::optional<array> vision_bias,
    MlxLinear shared_gate,
    MlxLinear shared_up,
    MlxLinear shared_down,
    std::optional<MlxRoutedLinear> routed_gate_up,
    std::optional<MlxRoutedLinear> routed_down,
    std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache,
    std::size_t expert_cache_layer)
    : hidden_(hidden),
      intermediate_(intermediate),
      experts_(experts),
      top_k_(top_k),
      normalize_(normalize),
      scale_(scale),
      swiglu_limit_(swiglu_limit),
      router_(std::move(router)),
      router_bias_(mlx::core::reshape(
          mlx::core::astype(router_bias, mlx::core::float32), Shape{experts})),
      vision_bias_(std::move(vision_bias)),
      shared_gate_(std::move(shared_gate)),
      shared_up_(std::move(shared_up)),
      shared_down_(std::move(shared_down)),
      routed_gate_up_(std::move(routed_gate_up)),
      routed_down_(std::move(routed_down)),
      ssd_expert_cache_(std::move(ssd_expert_cache)),
      expert_cache_layer_(expert_cache_layer) {
    if (top_k_ > experts_ || router_.input_size() != hidden_ ||
        router_.output_size() != experts_ ||
        shared_gate_.input_size() != hidden_ ||
        shared_gate_.output_size() != intermediate_ ||
        shared_up_.input_size() != hidden_ ||
        shared_up_.output_size() != intermediate_ ||
        shared_down_.input_size() != intermediate_ ||
        shared_down_.output_size() != hidden_ ||
        routed_gate_up_.has_value() != routed_down_.has_value() ||
        routed_gate_up_.has_value() == static_cast<bool>(ssd_expert_cache_) ||
        (routed_gate_up_ && (
        routed_gate_up_->weight().experts() != experts_ ||
        routed_gate_up_->weight().neuron_len() != hidden_ ||
        !(
            (routed_gate_up_->weight().projections() == 2 &&
             routed_gate_up_->weight().out_per_expert() == intermediate_) ||
            (routed_gate_up_->weight().projections() == 1 &&
             routed_gate_up_->weight().out_per_expert() == 2 * intermediate_)
        ) ||
        routed_down_->weight().experts() != experts_ ||
        routed_down_->weight().neuron_len() != intermediate_ ||
        routed_down_->weight().out_per_expert() != hidden_))) {
        throw std::runtime_error("DeepSeek-V4.1 MoE tensor geometry disagrees");
    }
    if (vision_bias_.has_value()) {
        *vision_bias_ = mlx::core::reshape(
            mlx::core::astype(*vision_bias_, mlx::core::float32), Shape{experts_});
    }
}

array MlxDeepseekV41Moe::forward(
    const array& input,
    const std::optional<array>& image_mask) const {
    if (input.ndim() < 2 || input.shape(-1) != hidden_) {
        throw std::invalid_argument("DeepSeek-V4.1 MoE input shape mismatch");
    }
    const int rows = static_cast<int>(
        input.size() / static_cast<std::size_t>(hidden_));
    auto source = mlx::core::reshape(input, Shape{rows, hidden_});
    auto logits = router_(source);
    auto routes = moe_topk(
        logits,
        top_k_,
        false,
        true,
        normalize_,
        false,
        router_bias_,
        std::nullopt,
        1e-20f,
        scale_);
    if (image_mask.has_value()) {
        if (!vision_bias_.has_value() || image_mask->size() != static_cast<std::size_t>(rows)) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 visual router mask/bias mismatch");
        }
        auto visual = moe_topk(
            logits,
            top_k_,
            false,
            true,
            normalize_,
            false,
            *vision_bias_,
            std::nullopt,
            1e-20f,
            scale_);
        auto mask = mlx::core::expand_dims(
            mlx::core::reshape(
                mlx::core::astype(*image_mask, mlx::core::bool_),
                Shape{rows}),
            -1);
        routes.ids = mlx::core::where(mask, visual.ids, routes.ids);
        routes.weights = mlx::core::where(mask, visual.weights, routes.weights);
    }

    array routed_pairs(0.0f);
    if (ssd_expert_cache_) {
        auto host_ids = mlx::core::contiguous(
            mlx::core::astype(routes.ids, mlx::core::int32));
        detail::eval_with_timing(host_ids);
        std::vector<std::int32_t> active(
            host_ids.data<std::int32_t>(),
            host_ids.data<std::int32_t>() + host_ids.size());
        active.erase(
            std::remove_if(
                active.begin(), active.end(),
                [](std::int32_t expert) { return expert < 0; }),
            active.end());
        std::sort(active.begin(), active.end());
        active.erase(std::unique(active.begin(), active.end()), active.end());
        auto prepared = ssd_expert_cache_->prepare(
            expert_cache_layer_, active);
        std::vector<std::int32_t> slot_ids(host_ids.size(), -1);
        const auto slot_map = prepared.slot_for_expert();
        for (std::size_t index = 0; index < slot_ids.size(); ++index) {
            const auto expert = host_ids.data<std::int32_t>()[index];
            if (expert >= 0) {
                slot_ids[index] = slot_map[static_cast<std::size_t>(expert)];
            }
        }
        const array resident_ids(slot_ids.begin(), routes.ids.shape());
        auto routed_hidden = prepared.weights().gate_up.swiglu(
            source, resident_ids, swiglu_limit_);
        routed_pairs = prepared.weights().down(
            routed_hidden, resident_ids);
    } else {
        auto routed_hidden = routed_gate_up_->swiglu(
            source, routes.ids, swiglu_limit_);
        routed_pairs = (*routed_down_)(routed_hidden, routes.ids);
    }
    auto routed = moe_weighted_reduce(routed_pairs, routes.weights);
    auto shared = shared_down_(limited_swiglu(
        shared_gate_(source), shared_up_(source), swiglu_limit_));
    return mlx::core::reshape(routed + shared, input.shape());
}

} // namespace mfq::metal
