#include "mlx_deepseek_v4_moe.h"

#include "mlx_eval_timing.h"
#include "mlx_moe_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

int checked_int(std::size_t value, const char* name) {
    if (value >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 MoE ") + name
            + " exceeds MLX limits");
    }
    return static_cast<int>(value);
}

array load_dense(
    const MfqContainer& model,
    const std::string& name,
    mlx::core::Dtype dtype) {
    const auto& record = model.record(name);
    const bool integer_target =
        dtype == mlx::core::int32;
    const bool supported = integer_target
        ? (record.dtype == "I32" || record.dtype == "I64")
        : (record.dtype == "F16" || record.dtype == "F32");
    if (!supported) {
        throw std::runtime_error(
            "DeepSeek-V4 dense tensor has incompatible dtype: "
            + name);
    }
    auto value = load_dense_array(
        record.dtype,
        model.read(name));
    if (value.dtype() != dtype) {
        value = mlx::core::astype(value, dtype);
    }
    return mlx::core::contiguous(value);
}

MlxRoutedLinear load_routed(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "NINTM") {
        throw std::runtime_error(
            "DeepSeek-V4 routed expert tensor must use NINTM: "
            + name);
    }
    return MlxRoutedLinear::from_blob(model.read(name));
}

std::optional<MlxGroupedLinear> make_grouped(
    const MlxLinear& router,
    const MlxLinear& shared_gate,
    const MlxLinear& shared_up) {
    const auto router_ref = router.grouped_weight_ref();
    const auto gate_ref = shared_gate.grouped_weight_ref();
    const auto up_ref = shared_up.grouped_weight_ref();
    if (!router_ref || !gate_ref || !up_ref) {
        return std::nullopt;
    }
    try {
        return MlxGroupedLinear({
            *router_ref,
            *gate_ref,
            *up_ref,
        });
    } catch (const MlxGroupedLinearUnsupported&) {
        return std::nullopt;
    }
}

array bool_vector(
    const std::optional<array>& value,
    int experts) {
    if (!value.has_value()) {
        return mlx::core::ones(
            Shape{experts},
            mlx::core::bool_);
    }
    auto result = mlx::core::contiguous(
        mlx::core::reshape(
            mlx::core::astype(
                *value,
                mlx::core::bool_),
            Shape{checked_int(
                value->size(),
                "availability size")}));
    if (result.shape() != Shape{experts}) {
        throw std::invalid_argument(
            "DeepSeek-V4 expert availability shape mismatch");
    }
    return result;
}

int count_available(const array& value) {
    auto bytes = mlx::core::contiguous(
        mlx::core::astype(
            value,
            mlx::core::uint8));
    detail::eval_with_timing(bytes);
    return static_cast<int>(
        std::count(
            bytes.data<std::uint8_t>(),
            bytes.data<std::uint8_t>() + bytes.size(),
        std::uint8_t{1}));
}

array streamed_availability(
    const std::optional<array>& requested,
    MlxCccpExpertResidency& residency,
    const std::string& gate_up_name,
    const std::string& down_name,
    int experts) {
    const auto gate =
        residency.availability(gate_up_name);
    const auto down =
        residency.availability(down_name);
    if (
        gate.size()
            != static_cast<std::size_t>(experts)
        || down.size()
            != static_cast<std::size_t>(experts)
    ) {
        throw std::runtime_error(
            "DeepSeek-V4 streamed expert "
            "availability size mismatch");
    }
    const array gate_array(
        gate.begin(),
        Shape{experts});
    const array down_array(
        down.begin(),
        Shape{experts});
    return mlx::core::logical_and(
        bool_vector(requested, experts),
        mlx::core::logical_and(
            mlx::core::astype(
                gate_array,
                mlx::core::bool_),
            mlx::core::astype(
                down_array,
                mlx::core::bool_)));
}

} // namespace

MlxDeepseekV4Moe MlxDeepseekV4Moe::load(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    std::size_t layer,
    const std::optional<array>& available,
    std::shared_ptr<MlxCccpExpertResidency>
        residency) {
    config.validate();
    if (layer >=
        static_cast<std::size_t>(config.n_layers)) {
        throw std::out_of_range(
            "DeepSeek-V4 MoE layer index is out of range");
    }
    const auto name =
        [layer](const char* suffix) {
            return DeepseekV4TensorNames::layer(
                layer,
                suffix);
        };
    std::optional<array> router_bias;
    std::optional<array> token_experts;
    if (layer <
        static_cast<std::size_t>(
            config.n_hash_layers)) {
        token_experts = load_dense(
            model,
            name("ffn.gate.tid2eid"),
            mlx::core::int32);
    } else {
        router_bias = load_dense(
            model,
            name("ffn.gate.bias"),
            mlx::core::float32);
    }
    auto gate_up_name =
        name("ffn.experts.gate_up.weight");
    auto down_name =
        name("ffn.experts.down.weight");
    bool stream_gate_up = false;
    bool stream_down = false;
    if (residency) {
        stream_gate_up =
            residency->can_stream(gate_up_name);
        try {
            stream_down =
                residency->can_stream(down_name);
        } catch (...) {
            if (stream_gate_up) {
                residency->discard_record(
                    gate_up_name);
            }
            throw;
        }
    }
    if (stream_gate_up && stream_down) {
        try {
            const auto gate_info =
                residency->projection_info(
                    gate_up_name);
            const auto down_info =
                residency->projection_info(
                    down_name);
            const int experts = checked_int(
                static_cast<std::size_t>(
                    config.n_experts),
                "expert count");
            const int hidden = checked_int(
                static_cast<std::size_t>(
                    config.hidden),
                "hidden width");
            const int routed = checked_int(
                static_cast<std::size_t>(
                    config.moe_inter),
                "routed expert width");
            if (
                gate_info.experts != experts
                || gate_info.out_per_expert
                    != 2 * routed
                || gate_info.neuron_len != hidden
                || down_info.experts != experts
                || down_info.out_per_expert != hidden
                || down_info.neuron_len != routed
            ) {
                throw std::runtime_error(
                    "DeepSeek-V4 streamed expert "
                    "projection dimensions mismatch");
            }
            auto effective_available =
                streamed_availability(
                    available,
                    *residency,
                    gate_up_name,
                    down_name,
                    experts);
            return MlxDeepseekV4Moe(
                config,
                MlxLinear::load(
                    model,
                    name("ffn.gate.weight")),
                MlxLinear::load(
                    model,
                    name(
                        "ffn.shared_experts.w1.weight")),
                MlxLinear::load(
                    model,
                    name(
                        "ffn.shared_experts.w3.weight")),
                MlxLinear::load(
                    model,
                    name(
                        "ffn.shared_experts.w2.weight")),
                std::nullopt,
                std::nullopt,
                residency,
                gate_up_name,
                down_name,
                std::move(router_bias),
                std::move(token_experts),
                std::move(effective_available));
        } catch (...) {
            residency->discard_record(
                gate_up_name);
            residency->discard_record(
                down_name);
            throw;
        }
    }
    // A mixed representation must use the eager routed implementation for
    // both projections.  Drop any projection/codebook parsed by can_stream()
    // so a streamable half does not remain resident but unused.
    if (residency) {
        if (stream_gate_up) {
            residency->discard_record(gate_up_name);
        }
        if (stream_down) {
            residency->discard_record(down_name);
        }
    }
    return MlxDeepseekV4Moe(
        config,
        MlxLinear::load(
            model,
            name("ffn.gate.weight")),
        MlxLinear::load(
            model,
            name("ffn.shared_experts.w1.weight")),
        MlxLinear::load(
            model,
            name("ffn.shared_experts.w3.weight")),
        MlxLinear::load(
            model,
            name("ffn.shared_experts.w2.weight")),
        load_routed(model, gate_up_name),
        load_routed(model, down_name),
        std::move(router_bias),
        std::move(token_experts),
        available);
}

MlxDeepseekV4Moe::MlxDeepseekV4Moe(
    DeepseekV4Config config,
    MlxLinear router,
    MlxLinear shared_gate,
    MlxLinear shared_up,
    MlxLinear shared_down,
    MlxRoutedLinear routed_gate_up,
    MlxRoutedLinear routed_down,
    std::optional<array> router_bias,
    std::optional<array> token_experts,
    std::optional<array> available)
    : MlxDeepseekV4Moe(
          std::move(config),
          std::move(router),
          std::move(shared_gate),
          std::move(shared_up),
          std::move(shared_down),
          std::optional<MlxRoutedLinear>(
              std::move(routed_gate_up)),
          std::optional<MlxRoutedLinear>(
              std::move(routed_down)),
          nullptr,
          {},
          {},
          std::move(router_bias),
          std::move(token_experts),
          std::move(available)) {}

MlxDeepseekV4Moe::MlxDeepseekV4Moe(
    DeepseekV4Config config,
    MlxLinear router,
    MlxLinear shared_gate,
    MlxLinear shared_up,
    MlxLinear shared_down,
    std::optional<MlxRoutedLinear>
        routed_gate_up,
    std::optional<MlxRoutedLinear>
        routed_down,
    std::shared_ptr<MlxCccpExpertResidency>
        expert_residency,
    std::string streamed_gate_up_name,
    std::string streamed_down_name,
    std::optional<array> router_bias,
    std::optional<array> token_experts,
    std::optional<array> available)
    : config_(std::move(config)),
      router_(std::move(router)),
      shared_gate_(std::move(shared_gate)),
      shared_up_(std::move(shared_up)),
      shared_down_(std::move(shared_down)),
      routed_gate_up_(std::move(routed_gate_up)),
      routed_down_(std::move(routed_down)),
      expert_residency_(
          std::move(expert_residency)),
      streamed_gate_up_name_(
          std::move(streamed_gate_up_name)),
      streamed_down_name_(
          std::move(streamed_down_name)),
      router_bias_(std::move(router_bias)),
      token_experts_(std::move(token_experts)),
      available_(bool_vector(
          available,
          checked_int(
              static_cast<std::size_t>(config_.n_experts),
              "expert count"))) {
    config_.validate();
    const int hidden = checked_int(
        static_cast<std::size_t>(config_.hidden),
        "hidden width");
    const int experts = checked_int(
        static_cast<std::size_t>(config_.n_experts),
        "expert count");
    const int shared = checked_int(
        static_cast<std::size_t>(
            config_.shared_intermediate_size()),
        "shared expert width");
    const int routed = checked_int(
        static_cast<std::size_t>(config_.moe_inter),
        "routed expert width");
    const int routed_gate_up_width = checked_int(
        static_cast<std::size_t>(config_.moe_inter) * 2,
        "routed gate/up width");
    if (router_.input_size() != hidden ||
        router_.output_size() != experts ||
        shared_gate_.input_size() != hidden ||
        shared_gate_.output_size() != shared ||
        shared_up_.input_size() != hidden ||
        shared_up_.output_size() != shared ||
        shared_down_.input_size() != shared ||
        shared_down_.output_size() != hidden) {
        throw std::invalid_argument(
            "DeepSeek-V4 MoE component dimensions mismatch");
    }
    const bool full_resident =
        routed_gate_up_.has_value()
        && routed_down_.has_value();
    const bool streamed =
        static_cast<bool>(expert_residency_);
    if (
        routed_gate_up_.has_value()
            != routed_down_.has_value()
        || full_resident == streamed
    ) {
        throw std::invalid_argument(
            "DeepSeek-V4 MoE requires exactly one "
            "routed expert residency mode");
    }
    if (full_resident) {
        if (
            routed_gate_up_->weight().experts()
                    != experts
            || routed_gate_up_->weight().neuron_len()
                    != hidden
            || routed_gate_up_->weight()
                    .out_per_expert()
                != routed_gate_up_width
            || routed_down_->weight().experts()
                    != experts
            || routed_down_->weight().neuron_len()
                    != routed
            || routed_down_->weight()
                    .out_per_expert()
                != hidden
        ) {
            throw std::invalid_argument(
                "DeepSeek-V4 MoE routed component "
                "dimensions mismatch");
        }
    } else {
        if (
            streamed_gate_up_name_.empty()
            || streamed_down_name_.empty()
        ) {
            throw std::invalid_argument(
                "DeepSeek-V4 streamed expert record "
                "names cannot be empty");
        }
        const auto gate =
            expert_residency_->projection_info(
                streamed_gate_up_name_);
        const auto down =
            expert_residency_->projection_info(
                streamed_down_name_);
        if (
            gate.experts != experts
            || gate.neuron_len != hidden
            || gate.out_per_expert
                != routed_gate_up_width
            || down.experts != experts
            || down.neuron_len != routed
            || down.out_per_expert != hidden
        ) {
            throw std::invalid_argument(
                "DeepSeek-V4 streamed routed component "
                "dimensions mismatch");
        }
    }
    const auto router_scale =
        static_cast<float>(config_.routed_scaling);
    const auto swiglu_limit =
        static_cast<float>(config_.swiglu_limit);
    if (!std::isfinite(router_scale) ||
        !std::isfinite(swiglu_limit)) {
        throw std::invalid_argument(
            "DeepSeek-V4 MoE parameters exceed float range");
    }
    available_count_ = available.has_value()
        ? count_available(available_)
        : experts;
    if (available_count_ <
        checked_int(
            static_cast<std::size_t>(config_.top_k),
            "route count")) {
        throw std::invalid_argument(
            "DeepSeek-V4 MoE has fewer available experts "
            "than routes");
    }
    if (router_bias_.has_value()) {
        *router_bias_ = mlx::core::contiguous(
            mlx::core::reshape(
                mlx::core::astype(
                    *router_bias_,
                    mlx::core::float32),
                Shape{checked_int(
                    router_bias_->size(),
                    "router bias size")}));
        if (router_bias_->shape() != Shape{experts}) {
            throw std::invalid_argument(
                "DeepSeek-V4 router bias shape mismatch");
        }
    }
    if (token_experts_.has_value()) {
        *token_experts_ = mlx::core::contiguous(
            mlx::core::astype(
                *token_experts_,
                mlx::core::int32));
        if (token_experts_->shape() !=
            Shape{
                checked_int(
                    static_cast<std::size_t>(config_.vocab),
                    "vocabulary size"),
                checked_int(
                    static_cast<std::size_t>(config_.top_k),
                    "route count"),
            }) {
            throw std::invalid_argument(
                "DeepSeek-V4 token-to-expert table shape mismatch");
        }
    }
    if (static_cast<bool>(router_bias_) ==
        static_cast<bool>(token_experts_)) {
        throw std::invalid_argument(
            "DeepSeek-V4 MoE requires exactly one routing mode");
    }
    grouped_projections_ = make_grouped(
        router_,
        shared_gate_,
        shared_up_);
}

std::array<array, 3>
MlxDeepseekV4Moe::project_shared(
    const array& input) const {
    if (grouped_projections_.has_value() &&
        grouped_projections_->supports(input)) {
        auto outputs =
            grouped_projections_->matmul(input);
        return {
            std::move(outputs.at(0)),
            std::move(outputs.at(1)),
            std::move(outputs.at(2)),
        };
    }
    return {
        router_(input),
        shared_gate_(input),
        shared_up_(input),
    };
}

MlxDeepseekV4MoeResult
MlxDeepseekV4Moe::forward_with_routing(
    const array& input,
    const array& token_ids) const {
    const int hidden = checked_int(
        static_cast<std::size_t>(config_.hidden),
        "hidden width");
    if (input.ndim() < 2 ||
        input.shape(-1) != hidden ||
        input.size() == 0) {
        throw std::invalid_argument(
            "DeepSeek-V4 MoE input shape mismatch");
    }
    const int rows = checked_int(
        input.size() / static_cast<std::size_t>(hidden),
        "row count");
    if (token_ids.size() !=
        static_cast<std::size_t>(rows)) {
        throw std::invalid_argument(
            "DeepSeek-V4 token id count mismatch");
    }
    auto source = mlx::core::contiguous(
        mlx::core::reshape(
            input,
            Shape{rows, hidden}));
    auto projections = project_shared(source);
    auto logits = std::move(projections[0]);
    array expert_ids = mlx::core::zeros(
        Shape{
            rows,
            checked_int(
                static_cast<std::size_t>(config_.top_k),
                "route count"),
        },
        mlx::core::int32);
    array expert_weights = mlx::core::zeros(
        expert_ids.shape(),
        mlx::core::float32);
    if (token_experts_.has_value()) {
        auto ids = token_ids;
        if (ids.dtype() != mlx::core::int32 &&
            ids.dtype() != mlx::core::uint32) {
            ids = mlx::core::astype(
                ids,
                mlx::core::int32);
        }
        ids = mlx::core::reshape(ids, Shape{rows});
        expert_ids = mlx::core::take(
            *token_experts_,
            ids,
            0);
        const int candidate_count = std::min(
            {
                16,
                available_count_,
                checked_int(
                    static_cast<std::size_t>(
                        config_.n_experts),
                    "expert count"),
                std::max(
                    checked_int(
                        static_cast<std::size_t>(
                            2 * config_.top_k),
                        "candidate count"),
                    checked_int(
                        static_cast<std::size_t>(
                            config_.top_k),
                        "route count")),
            });
        const auto candidates = moe_topk(
            logits,
            candidate_count,
            false,
            true,
            false,
            false,
            std::nullopt,
            available_);
        expert_ids = moe_repair_hash_ids(
            expert_ids,
            candidates.ids,
            available_);
        expert_weights =
            moe_selected_sqrtsoftplus_weights(
                logits,
                expert_ids,
                config_.norm_topk_prob,
                1e-20f,
                static_cast<float>(
                    config_.routed_scaling));
    } else {
        auto routing = moe_topk(
            logits,
            checked_int(
                static_cast<std::size_t>(config_.top_k),
                "route count"),
            false,
            true,
            config_.norm_topk_prob,
            false,
            router_bias_,
            available_,
            1e-20f,
            static_cast<float>(
                config_.routed_scaling));
        expert_ids = std::move(routing.ids);
        expert_weights = std::move(routing.weights);
    }

    array routed = mlx::core::zeros(
        Shape{rows, hidden},
        source.dtype());
    if (!expert_residency_) {
        auto gate_up =
            routed_gate_up_->forward(
                source,
                expert_ids);
        auto routed_hidden =
            moe_limited_swiglu_split(
                gate_up,
                static_cast<float>(
                    config_.swiglu_limit));
        routed = routed_down_->combine(
            routed_hidden,
            expert_ids,
            expert_weights);
    } else {
        constexpr int kStreamRows = 16;
        auto host_ids =
            mlx::core::contiguous(
                mlx::core::astype(
                    expert_ids,
                    mlx::core::int32));
        detail::eval_with_timing(host_ids);
        const auto* id_values =
            host_ids.data<std::int32_t>();
        const int routes =
            checked_int(
                static_cast<std::size_t>(
                    config_.top_k),
                "route count");
        std::vector<array> chunks;
        chunks.reserve(
            (
                static_cast<std::size_t>(rows)
                + kStreamRows - 1
            ) / kStreamRows);
        for (
            int start = 0;
            start < rows;
            start += kStreamRows
        ) {
            const int end =
                std::min(
                    rows,
                    start + kStreamRows);
            std::vector<std::int32_t> selected;
            selected.reserve(
                static_cast<std::size_t>(
                    end - start)
                * routes);
            for (
                int row = start;
                row < end;
                ++row
            ) {
                for (
                    int route = 0;
                    route < routes;
                    ++route
                ) {
                    const auto expert =
                        id_values[
                            static_cast<std::size_t>(
                                row)
                                * routes
                            + route];
                    if (expert >= 0) {
                        selected.push_back(
                            expert);
                    }
                }
            }
            std::sort(
                selected.begin(),
                selected.end());
            selected.erase(
                std::unique(
                    selected.begin(),
                    selected.end()),
                selected.end());
            auto chunk_source =
                mlx::core::slice(
                    source,
                    Shape{start, 0},
                    Shape{end, hidden});
            auto chunk_ids =
                mlx::core::contiguous(
                    mlx::core::slice(
                        expert_ids,
                        Shape{start, 0},
                        Shape{end, routes}));
            auto chunk_weights =
                mlx::core::slice(
                    expert_weights,
                    Shape{start, 0},
                    Shape{end, routes});
            auto gate_weight =
                expert_residency_->grouped(
                    streamed_gate_up_name_,
                    selected);
            auto gate_up =
                gate_weight.routed_matmul(
                    chunk_source,
                    chunk_ids);
            auto routed_hidden =
                moe_limited_swiglu_split(
                    gate_up,
                    static_cast<float>(
                        config_.swiglu_limit));
            auto down_weight =
                expert_residency_->grouped(
                    streamed_down_name_,
                    selected);
            auto down =
                down_weight.routed_matmul(
                    routed_hidden,
                    chunk_ids);
            auto chunk_output =
                moe_weighted_reduce(
                    down,
                    chunk_weights);
            // Decode/small-M stays one lazy gate->down graph and is
            // materialized by the causal-layer boundary.  For multi-chunk
            // prefill, finish each weighted chunk before advancing so the
            // graph never retains packed inputs for every prior chunk.
            if (rows > kStreamRows) {
                detail::eval_with_timing(chunk_output);
            }
            chunks.push_back(
                std::move(chunk_output));
        }
        routed = chunks.size() == 1
            ? std::move(chunks.front())
            : mlx::core::concatenate(
                std::move(chunks),
                0);
    }
    auto shared_gate_up = mlx::core::concatenate(
        {
            std::move(projections[1]),
            std::move(projections[2]),
        },
        -1);
    auto shared_hidden =
        moe_limited_swiglu_split(
            shared_gate_up,
            static_cast<float>(
                config_.swiglu_limit));
    auto shared = shared_down_(shared_hidden);
    auto output = routed + shared;
    Shape output_shape(
        input.shape().begin(),
        input.shape().end() - 1);
    output_shape.push_back(hidden);
    return {
        mlx::core::reshape(
            std::move(output),
            std::move(output_shape)),
        std::move(expert_ids),
        std::move(expert_weights),
    };
}

array MlxDeepseekV4Moe::forward(
    const array& input,
    const array& token_ids) const {
    return forward_with_routing(
        input,
        token_ids).output;
}

} // namespace mfq::metal
