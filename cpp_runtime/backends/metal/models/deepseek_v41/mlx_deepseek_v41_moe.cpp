#include "mlx_deepseek_v41_moe.h"

#include "mlx_eval_timing.h"
#include "mlx_moe_ops.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

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
    if (model.record(name).dtype != "MFE") {
        throw std::runtime_error(
            "DeepSeek-V4.1 routed expert tensor must use MFE: " + name);
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
    std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache,
    std::size_t expert_cache_layer,
    std::shared_ptr<MlxTqExpertCache> tq_expert_cache) {
    const int experts = checked_int(
        predictor ? config.dspark_n_experts : config.n_experts,
        "expert count");
    const int top_k = checked_int(
        predictor ? config.dspark_top_k : config.top_k,
        "route count");
    const auto expert_prefix = prefix + ".experts.";
    const bool has_gate = model.contains(
        expert_prefix + "gate.weight");
    const bool has_up = model.contains(
        expert_prefix + "up.weight");
    if (has_gate != has_up) {
        throw std::runtime_error(
            "DeepSeek-V4.1 split Gate/Up residency is incomplete");
    }
    const bool split_gate_up = has_gate;
    const auto gate_up_name = split_gate_up
        ? expert_prefix + "gate.weight"
        : expert_prefix + "gate_up.weight";
    const auto up_name = split_gate_up
        ? std::optional<std::string>(expert_prefix + "up.weight")
        : std::nullopt;
    const auto down_name = expert_prefix + "down.weight";
    if (mfe_offload_cache &&
        (!mfe_offload_cache->can_group_mfe(gate_up_name)
         || (up_name && !mfe_offload_cache->can_group_mfe(*up_name))
         || !mfe_offload_cache->can_group_mfe(down_name))) {
        throw std::runtime_error(
            "DeepSeek-V4.1 expert projection cannot be offloaded; "
            "refusing to bypass the configured cache budget: " + prefix);
    }
    std::optional<MlxMoeWeight> gate_up;
    std::optional<MlxRoutedLinear> gate;
    std::optional<MlxRoutedLinear> up;
    std::optional<MlxRoutedLinear> down;
    std::optional<TqWeights> tq;
    if (model.record(gate_up_name).dtype == "TQ") {
        if (!up_name || model.record(*up_name).dtype != "TQ" || model.record(down_name).dtype != "TQ")
            throw std::runtime_error("TQ MoE requires separate Gate, Up and Down tensors");
        if (ssd_expert_cache || mfe_offload_cache)
            throw std::runtime_error("TQ requires resident expert tensors");
        if (tq_expert_cache) {
            tq_expert_cache->validate_layer(expert_cache_layer, experts,
                checked_int(config.hidden, "hidden size"), checked_int(config.moe_inter, "expert intermediate size"));
        } else {
            tq.emplace(TqWeights{MlxTqWeight::load(model, gate_up_name),
                MlxTqWeight::load(model, *up_name), MlxTqWeight::load(model, down_name)});
        }
    } else if (!ssd_expert_cache && !mfe_offload_cache) {
        if (tq_expert_cache) throw std::invalid_argument("TQ cache cannot load non-TQ experts");
        if (split_gate_up) {
            gate.emplace(moe_weight(model, gate_up_name));
            up.emplace(moe_weight(model, *up_name));
        } else {
            gate_up.emplace(moe_weight(model, gate_up_name));
        }
        down.emplace(moe_weight(model, down_name));
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
        std::move(gate),
        std::move(up),
        std::move(down),
        std::move(ssd_expert_cache),
        std::move(mfe_offload_cache),
        gate_up_name,
        std::move(up_name),
        down_name,
        expert_cache_layer, std::move(tq), std::move(tq_expert_cache));
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
    std::optional<MlxMoeWeight> routed_gate_up,
    std::optional<MlxRoutedLinear> routed_gate,
    std::optional<MlxRoutedLinear> routed_up,
    std::optional<MlxRoutedLinear> routed_down,
    std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache,
    std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache,
    std::string streamed_gate_up_name,
    std::optional<std::string> streamed_up_name,
    std::string streamed_down_name,
    std::size_t expert_cache_layer,
    std::optional<TqWeights> tq,
    std::shared_ptr<MlxTqExpertCache> tq_expert_cache)
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
      routed_gate_(std::move(routed_gate)),
      routed_up_(std::move(routed_up)),
      routed_down_(std::move(routed_down)),
      tq_(std::move(tq)),
      tq_expert_cache_(std::move(tq_expert_cache)),
      ssd_expert_cache_(std::move(ssd_expert_cache)),
      mfe_offload_cache_(std::move(mfe_offload_cache)),
      streamed_gate_up_name_(std::move(streamed_gate_up_name)),
      streamed_up_name_(std::move(streamed_up_name)),
      streamed_down_name_(std::move(streamed_down_name)),
      expert_cache_layer_(expert_cache_layer) {
    const bool cached = static_cast<bool>(ssd_expert_cache_)
        || static_cast<bool>(mfe_offload_cache_) || static_cast<bool>(tq_expert_cache_);
    const bool split_resident = routed_gate_.has_value()
        && routed_up_.has_value();
    const bool combined_resident = routed_gate_up_.has_value();
    const bool routed_geometry = combined_resident
        ? routed_gate_up_->experts() == experts_
            && routed_gate_up_->neuron_len() == hidden_
            && ((routed_gate_up_->projections() == 2
                 && routed_gate_up_->out_per_expert() == intermediate_)
                || (routed_gate_up_->projections() == 1
                    && routed_gate_up_->out_per_expert()
                        == 2 * intermediate_))
        : split_resident
            ? routed_gate_->weight().experts() == experts_
                && routed_gate_->weight().neuron_len() == hidden_
                && routed_gate_->weight().out_per_expert() == intermediate_
                && routed_up_->weight().experts() == experts_
                && routed_up_->weight().neuron_len() == hidden_
                && routed_up_->weight().out_per_expert() == intermediate_
            : true;
    if (top_k_ > experts_ || router_.input_size() != hidden_ ||
        router_.output_size() != experts_ ||
        shared_gate_.input_size() != hidden_ ||
        shared_gate_.output_size() != intermediate_ ||
        shared_up_.input_size() != hidden_ ||
        shared_up_.output_size() != intermediate_ ||
        shared_down_.input_size() != intermediate_ ||
        shared_down_.output_size() != hidden_ ||
        routed_gate_.has_value() != routed_up_.has_value() ||
        (combined_resident && split_resident) ||
        routed_down_.has_value() !=
            (combined_resident || split_resident) ||
        (combined_resident || split_resident || tq_.has_value()) == cached ||
        (int(bool(ssd_expert_cache_)) + int(bool(mfe_offload_cache_)) + int(bool(tq_expert_cache_)) > 1) ||
        !routed_geometry ||
        (routed_down_ && (
        routed_down_->weight().experts() != experts_ ||
        routed_down_->weight().neuron_len() != intermediate_ ||
        routed_down_->weight().out_per_expert() != hidden_))) {
        throw std::runtime_error("DeepSeek-V4.1 MoE tensor geometry disagrees");
    }
    if (tq_) {
        const auto& gate = tq_->gate;
        const auto& down = tq_->down;
        if (gate.experts() != experts_ || gate.input_size() != hidden_
            || gate.output_size() != intermediate_
            || down.experts() != experts_ || down.input_size() != intermediate_ || down.output_size() != hidden_
            || tq_->up.experts() != experts_ || tq_->up.input_size() != hidden_
            || tq_->up.output_size() != intermediate_)
            throw std::runtime_error("DeepSeek TQ expert geometry differs");
        tq_->prepare(top_k_, swiglu_limit_);
    }
    if (tq_expert_cache_) tq_expert_cache_->prepare_layer(expert_cache_layer_, top_k_, swiglu_limit_);
    if (mfe_offload_cache_) {
        const auto gate = mfe_offload_cache_->projection_info(
            streamed_gate_up_name_);
        const auto down = mfe_offload_cache_->projection_info(
            streamed_down_name_);
        bool gate_matches = gate.experts == experts_
            && gate.neuron_len == hidden_;
        if (streamed_up_name_) {
            const auto up = mfe_offload_cache_->projection_info(
                *streamed_up_name_);
            gate_matches = gate_matches
                && gate.out_per_expert == intermediate_
                && up.experts == experts_
                && up.neuron_len == hidden_
                && up.out_per_expert == intermediate_;
        } else {
            gate_matches = gate_matches
                && gate.out_per_expert == 2 * intermediate_;
        }
        if (!gate_matches
            || down.experts != experts_
            || down.neuron_len != intermediate_
            || down.out_per_expert != hidden_) {
            throw std::runtime_error(
                "DeepSeek-V4.1 streamed MoE geometry disagrees");
        }
        mfe_offload_cache_->activate_record(streamed_gate_up_name_);
        if (streamed_up_name_) mfe_offload_cache_->activate_record(*streamed_up_name_);
        mfe_offload_cache_->activate_record(streamed_down_name_);
        if (streamed_up_name_) mfe_offload_cache_->prepare_swiglu_pair(
            streamed_gate_up_name_, *streamed_up_name_, top_k_);
    }
    if (vision_bias_.has_value()) {
        *vision_bias_ = mlx::core::reshape(
            mlx::core::astype(*vision_bias_, mlx::core::float32), Shape{experts_});
    }
    projection_batch_.emplace(
        std::vector<const MlxLinear*>{
            &router_, &shared_gate_, &shared_up_,
        });
    shared_gate_up_batch_.emplace(
        std::vector<const MlxLinear*>{
            &shared_gate_, &shared_up_,
        });
}

std::size_t MlxDeepseekV41Moe::resident_routed_bytes() const noexcept {
    if (tq_) return tq_->gate.packed_nbytes() + tq_->up.packed_nbytes() + tq_->down.packed_nbytes();
    std::size_t bytes = routed_gate_up_ ? routed_gate_up_->packed_nbytes() : 0;
    for (const auto* projection : {&routed_gate_, &routed_up_, &routed_down_}) {
        if (*projection) bytes += (*projection)->weight().packed_nbytes();
    }
    return bytes;
}

void MlxDeepseekV41Moe::validate_offload(
    const std::shared_ptr<MlxMfeOffloadCache>& cache) const {
    for (const auto& name : {streamed_gate_up_name_,
             streamed_up_name_.value_or(streamed_gate_up_name_),
             streamed_down_name_}) {
        if (!cache->can_group_mfe(name)) {
            throw std::runtime_error("expert projection cannot be offloaded: " + name);
        }
    }
}

void MlxDeepseekV41Moe::use_offload(
    const std::shared_ptr<MlxMfeOffloadCache>& cache) {
    cache->activate_record(streamed_gate_up_name_);
    if (streamed_up_name_) cache->activate_record(*streamed_up_name_);
    cache->activate_record(streamed_down_name_);
    routed_gate_up_.reset();
    routed_gate_.reset();
    routed_up_.reset();
    routed_down_.reset();
    ssd_expert_cache_.reset();
    mfe_offload_cache_ = cache;
}

int MlxDeepseekV41Moe::recommended_prefill_chunk_size() const noexcept {
    if (ssd_expert_cache_ || mfe_offload_cache_ ||
        (!routed_gate_up_ && !routed_gate_) || !routed_down_) {
        return 0;
    }
    const int gate_up = routed_gate_up_
        ? routed_gate_up_->recommended_mxfp4_nax_prefill_tokens(top_k_)
        : std::min(
              routed_gate_->recommended_mxfp4_nax_prefill_tokens(top_k_),
              routed_up_->recommended_mxfp4_nax_prefill_tokens(top_k_));
    const int down =
        routed_down_->recommended_mxfp4_nax_prefill_tokens(top_k_);
    return gate_up > 0 && down > 0 ? std::min(gate_up, down) : 0;
}

void MlxDeepseekV41Moe::validate_tq_offload(const std::shared_ptr<MlxTqExpertCache>& cache) const {
    if (!tq_ && !tq_expert_cache_) throw std::invalid_argument("TQ cache cannot replace non-TQ experts");
    cache->validate_layer(expert_cache_layer_, experts_, hidden_, intermediate_);
}

void MlxDeepseekV41Moe::use_tq_offload(const std::shared_ptr<MlxTqExpertCache>& cache) {
    validate_tq_offload(cache);
    tq_.reset();
    tq_expert_cache_ = cache;
}

std::optional<MlxMfeCachedRoute> MlxDeepseekV41Moe::cached_route() const {
    if (!mfe_offload_cache_) return std::nullopt;
    MlxMfeCachedRoute route{mfe_offload_cache_, {}, std::nullopt};
    std::vector<std::string> names{streamed_gate_up_name_};
    if (streamed_up_name_) names.push_back(*streamed_up_name_);
    names.push_back(streamed_down_name_);
    for (const auto& name : names) {
        auto projection = mfe_offload_cache_->cached_projection(name);
        if (!projection) return std::nullopt;
        route.projections.push_back(std::move(projection));
    }
    return route;
}

std::optional<MlxSsdPrefetchedExpertLayer>
MlxDeepseekV41Moe::prefetch_routed(std::size_t rows) const {
    constexpr std::size_t kFullLayerPrefetchRows = 512;
    if (ssd_expert_cache_ && rows >= kFullLayerPrefetchRows &&
        ssd_expert_cache_->prefill_overlap_enabled()) {
        return ssd_expert_cache_->prefetch_layer(expert_cache_layer_);
    }
    return std::nullopt;
}

MlxDeepseekV41MoeResult MlxDeepseekV41Moe::forward(
    const array& input,
    const std::optional<array>& image_mask,
    MlxSsdPrefetchedExpertLayer* prefetched,
    MlxMfeCachedRoute* deferred_route) const {
    if (input.ndim() < 2 || input.shape(-1) != hidden_) {
        throw std::invalid_argument("DeepSeek-V4.1 MoE input shape mismatch");
    }
    const int rows = static_cast<int>(
        input.size() / static_cast<std::size_t>(hidden_));
    auto source = mlx::core::reshape(input, Shape{rows, hidden_});
    const auto* dense_router = router_.dense_weight_ref();
    const bool fused_router = image_mask == std::nullopt &&
        top_k_ == 6 && normalize_ && dense_router != nullptr &&
        moe_dense_router_topk_supported(source, *dense_router);
    const auto router_logits = [&] {
        if (image_mask && dense_router) {
            return mlx::core::matmul(
                mlx::core::astype(source, mlx::core::float32),
                mlx::core::transpose(mlx::core::astype(*dense_router, mlx::core::float32)));
        }
        return router_(source);
    };

    std::optional<array> logits;
    array shared_hidden(0.0f);
    if (shared_gate_up_batch_ &&
        shared_gate_up_batch_->supports_fused_swiglu(source)) {
        shared_hidden = shared_gate_up_batch_->swiglu(
            source, swiglu_limit_);
        if (!fused_router) {
            logits = router_logits();
        }
    } else if (!fused_router && !image_mask && projection_batch_) {
        auto projections = (*projection_batch_)(source);
        logits = std::move(projections.at(0));
        shared_hidden = limited_swiglu(
            projections.at(1), projections.at(2), swiglu_limit_);
    } else if (shared_gate_up_batch_) {
        auto projections = (*shared_gate_up_batch_)(source);
        shared_hidden = limited_swiglu(
            projections.at(0), projections.at(1), swiglu_limit_);
        if (!fused_router) {
            logits = router_logits();
        }
    } else {
        shared_hidden = limited_swiglu(
            shared_gate_(source), shared_up_(source), swiglu_limit_);
        if (!fused_router) {
            logits = router_logits();
        }
    }

    auto routes = fused_router
        ? moe_dense_router_topk(
              source,
              *dense_router,
              router_bias_,
              std::nullopt,
              1e-20f,
              scale_)
        : moe_topk(
              *logits,
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
            *logits,
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

    auto shared = shared_down_(shared_hidden);
    array routed(0.0f);
    const bool route_transaction = ssd_expert_cache_ &&
        ssd_expert_cache_->route_transaction_active();
    if (prefetched != nullptr) {
        if (!ssd_expert_cache_ ||
            prefetched->layer() != expert_cache_layer_) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 SSD prefetch layer mismatch");
        }
        const auto& weights = prefetched->wait();
        auto routed_hidden = weights.gate_up.swiglu(
            source, routes.ids, swiglu_limit_);
        routed = weights.down.combine(
            routed_hidden,
            routes.ids,
            routes.weights);
        detail::eval_with_timing(routed);
    } else if (route_transaction) {
        auto snapshot = ssd_expert_cache_->snapshot_page_table(
            expert_cache_layer_);
        auto resident_ids = mlx::core::take(
            snapshot.slot_ids(),
            mlx::core::astype(routes.ids, mlx::core::int32));
        auto routed_hidden = snapshot.weights().gate_up.swiglu(
            source, resident_ids, swiglu_limit_);
        routed = snapshot.weights().down.combine(
            routed_hidden,
            resident_ids,
            routes.weights);
        snapshot.defer_transaction_global(routes.ids);
    } else if (ssd_expert_cache_) {
        auto prepared = ssd_expert_cache_->prepare_routes(
            expert_cache_layer_, routes.ids);
        auto routed_hidden = prepared.weights().gate_up.swiglu(
            source, prepared.expert_ids(), swiglu_limit_);
        routed = prepared.weights().down.combine(
            routed_hidden,
            prepared.expert_ids(),
            routes.weights);
    } else if (mfe_offload_cache_) {
        if (deferred_route) {
            auto output = deferred_route->forward(source, routes.ids, swiglu_limit_, &routes.weights);
            return {mlx::core::reshape(std::move(output), input.shape()),
                mlx::core::reshape(std::move(shared), input.shape())};
        }
        if (rows == 1 && !image_mask && !detail::component_profile_active()) {
            if (auto cached = cached_route()) {
                auto output = cached->forward(source, routes.ids, swiglu_limit_, &routes.weights);
                mlx::core::async_eval({output});
                if (cached->resolve()) {
                    return {mlx::core::reshape(std::move(output), input.shape()),
                        mlx::core::reshape(std::move(shared), input.shape())};
                }
            }
        }
        auto global_ids = mlx::core::contiguous(
            mlx::core::astype(routes.ids, mlx::core::int32));
        detail::eval_with_timing(global_ids);
        if (rows <= 6) mlx::core::async_eval({shared});
        const auto* ids = global_ids.data<std::int32_t>();
        std::vector<std::int32_t> active;
        active.reserve(global_ids.size());
        std::vector<std::uint32_t> seen(
            static_cast<std::size_t>(experts_), 0);
        for (std::size_t index = 0; index < global_ids.size(); ++index) {
            const auto expert = ids[index];
            if (expert == -1) {
                continue;
            }
            if (expert < -1 || expert >= experts_) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 routed expert ID is out of range");
            }
            if (seen[static_cast<std::size_t>(expert)] == 0) {
                active.push_back(expert);
            }
            ++seen[static_cast<std::size_t>(expert)];
        }
        if (active.empty()) {
            throw std::runtime_error(
                "DeepSeek-V4.1 routing selected no resident expert");
        }
        std::vector<std::int32_t> global_to_local(
            static_cast<std::size_t>(experts_) + 1, -1);
        for (std::size_t local = 0; local < active.size(); ++local) {
            global_to_local[static_cast<std::size_t>(active[local]) + 1] =
                static_cast<std::int32_t>(local);
        }
        std::vector<std::int32_t> local_routes(global_ids.size(), -1);
        for (std::size_t index = 0; index < global_ids.size(); ++index)
            local_routes[index] = global_to_local[static_cast<std::size_t>(ids[index] + 1)];
        auto local_ids = array(local_routes.begin(), global_ids.shape(), mlx::core::int32);
        const auto groups = rows <= 6
            ? moe_route_groups_host(local_routes, rows, global_ids.shape(1)) : std::nullopt;
        const auto* route_groups = groups ? &*groups : nullptr;
        std::vector<std::uint32_t> activation_counts;
        for (const auto expert : active) activation_counts.push_back(seen[expert]);
        std::vector<std::string> names{streamed_gate_up_name_};
        if (streamed_up_name_) names.push_back(*streamed_up_name_);
        names.push_back(streamed_down_name_);
        const bool prefill_scan = rows > 6;
        auto weights = prefill_scan ? mfe_offload_cache_->layer_mfe(names, active)
            : std::nullopt;
        bool layer_ready = weights.has_value();
        std::optional<array> staged_hidden;
        if (weights) {
            local_ids = global_ids;
        } else if (rows == 1 && streamed_up_name_) {
            weights = mfe_offload_cache_->grouped_mfe(names, active, activation_counts, false,
                [&](const std::vector<MlxMfeWeight>& gu) {
                    staged_hidden = gu[0].routed_swiglu_pair(gu[1], source,
                        local_ids, swiglu_limit_, route_groups);
                    mlx::core::async_eval({*staged_hidden});
                });
            layer_ready = true;
        } else {
            const auto projections = std::span<const std::string>(names);
            weights = mfe_offload_cache_->grouped_mfe(projections.first(projections.size() - 1),
                active, activation_counts, prefill_scan);
        }
        const auto& gate = weights->front();
        const int route_count = static_cast<int>(local_ids.size());
        std::optional<array> sorted_routes;
        std::optional<MlxGroupedMmqPlan> gate_plan;
        if (prefill_scan && rows >= 32 && gate.supports_grouped_mmq() &&
            (!streamed_up_name_ || weights->at(1).supports_grouped_mmq())) {
            sorted_routes = mlx::core::contiguous(mlx::core::astype(
                mlx::core::argsort(mlx::core::reshape(local_ids, Shape{route_count})), mlx::core::int32));
            gate_plan = gate.build_grouped_mmq_plan(local_ids, *sorted_routes,
                gate.recommended_grouped_mmq_block_rows(route_count, !streamed_up_name_));
        }
        array routed_hidden = [&]() {
            if (staged_hidden) return *staged_hidden;
            if (sorted_routes) {
                auto projected = gate.routed_matmul_sorted(source, local_ids, *sorted_routes,
                    false, !streamed_up_name_, swiglu_limit_, &*gate_plan);
                if (!streamed_up_name_) return projected;
                const auto& up = weights->at(1);
                const int block_rows = up.recommended_grouped_mmq_block_rows(route_count, false);
                const auto plan = block_rows == gate_plan->block_rows ? *gate_plan
                    : up.build_grouped_mmq_plan(local_ids, *sorted_routes, block_rows);
                auto up_projected = up.routed_matmul_sorted(
                    source, local_ids, *sorted_routes, false, false, 0.0f, &plan);
                return swiglu_limit_ > 0.0f
                    ? moe_limited_swiglu_pair(std::move(projected), std::move(up_projected), swiglu_limit_)
                    : moe_swiglu_pair(std::move(projected), std::move(up_projected));
            }
            if (streamed_up_name_) {
                return gate.routed_swiglu_pair(
                    weights->at(1), source, local_ids, swiglu_limit_, route_groups);
            }
            return gate.routed_swiglu(
                source, local_ids, swiglu_limit_, route_groups);
        }();
        if (!layer_ready) {
            if (!prefill_scan || sorted_routes) mlx::core::async_eval({routed_hidden});
            weights->clear();
            weights->push_back(mfe_offload_cache_->grouped_mfe(
                streamed_down_name_, active, activation_counts, prefill_scan));
        }
        const auto& down = weights->back();
        if (sorted_routes && down.supports_grouped_mmq()) {
            const int block_rows = down.recommended_grouped_mmq_block_rows(route_count, false);
            const auto plan = block_rows == gate_plan->block_rows ? *gate_plan
                : down.build_grouped_mmq_plan(local_ids, *sorted_routes, block_rows);
            auto sorted_output = down.routed_matmul_sorted(routed_hidden, local_ids,
                *sorted_routes, true, false, 0.0f, &plan);
            routed = moe_weighted_reduce(mlx::core::reshape(
                moe_restore_routed_rows(sorted_output, *sorted_routes, local_ids),
                Shape{rows, top_k_, hidden_}), routes.weights);
        } else {
            if (sorted_routes) routed_hidden = mlx::core::reshape(
                moe_restore_routed_rows(routed_hidden, *sorted_routes, local_ids),
                Shape{rows, top_k_, intermediate_});
            routed = down.routed_matmul_reduce(routed_hidden, local_ids, routes.weights, route_groups);
        }
    } else if (tq_expert_cache_) {
        routed = tq_expert_cache_->apply(expert_cache_layer_, routes.ids, [&](const auto& weights, const array* reuse) {
            return weights.gate.swiglu(weights.up, source, routes.ids, swiglu_limit_, false, reuse);
        }, [&](const auto& weights, const array& hidden, const array* reuse) {
            return weights.down.routed_matmul_reduce(hidden, routes.ids, routes.weights, reuse);
        }, &shared);
    } else if (tq_) {
        auto hidden = tq_->gate.swiglu(tq_->up, source, routes.ids, swiglu_limit_);
        routed = tq_->down.routed_matmul_reduce(hidden, routes.ids, routes.weights);
    } else {
        auto routed_hidden = routed_gate_up_
            ? routed_gate_up_->routed_swiglu(
                  source, routes.ids, swiglu_limit_)
            : routed_gate_->swiglu_pair(
                  *routed_up_,
                  source,
                  routes.ids,
                  swiglu_limit_);
        routed = routed_down_->combine(
            routed_hidden,
            routes.ids,
            routes.weights);
    }
    return {
        mlx::core::reshape(std::move(routed), input.shape()),
        mlx::core::reshape(std::move(shared), input.shape()),
    };
}

} // namespace mfq::metal
