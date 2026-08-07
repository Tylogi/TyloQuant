#include "mlx_deepseek_v4_moe.h"

#include "mlx_eval_timing.h"
#include "mlx_moe_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
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
        : (record.dtype == "BF16" ||
           record.dtype == "F16" ||
           record.dtype == "F32");
    if (!supported) {
        throw std::runtime_error(
            "DeepSeek-V4 dense tensor has incompatible dtype: "
            + name);
    }
    const auto mapped = model.map_record(name);
    auto value = load_dense_array(
        record.dtype,
        mapped.view());
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
    const auto mapped = model.map_record(name);
    return MlxRoutedLinear::from_blob(mapped.view());
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

std::optional<MlxGroupedLinear> make_grouped_gate_up(
    const MlxLinear& shared_gate,
    const MlxLinear& shared_up) {
    const auto gate_ref = shared_gate.grouped_weight_ref();
    const auto up_ref = shared_up.grouped_weight_ref();
    if (!gate_ref || !up_ref) {
        return std::nullopt;
    }
    try {
        return MlxGroupedLinear({
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
    MlxNintMoeOffloadCache& offload,
    const std::vector<std::string>& projection_names,
    int experts) {
    auto result = bool_vector(requested, experts);
    for (const auto& name : projection_names) {
        const auto available = offload.availability(name);
        if (available.size() !=
            static_cast<std::size_t>(experts)) {
            throw std::runtime_error(
                "DeepSeek-V4 streamed expert "
                "availability size mismatch");
        }
        const array available_array(
            available.begin(),
            Shape{experts});
        result = mlx::core::logical_and(
            std::move(result),
            mlx::core::astype(
                available_array,
                mlx::core::bool_));
    }
    return result;
}

array limited_swiglu_pair(
    array gate,
    array up,
    float limit) {
    return moe_limited_swiglu_split(
        mlx::core::concatenate(
            {
                std::move(gate),
                std::move(up),
            },
            -1),
        limit);
}

} // namespace

MlxDeepseekV4Moe MlxDeepseekV4Moe::load(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    std::size_t layer,
    const std::optional<array>& available,
    std::shared_ptr<MlxNintMoeOffloadCache>
        offload) {
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
    auto gate_name =
        name("ffn.experts.gate.weight");
    auto up_name =
        name("ffn.experts.up.weight");
    auto down_name =
        name("ffn.experts.down.weight");
    const bool has_split_gate =
        model.contains(gate_name);
    const bool has_split_up =
        model.contains(up_name);
    if (has_split_gate != has_split_up) {
        throw std::runtime_error(
            "DeepSeek-V4 split routed Gate/Up records "
            "are incomplete at layer " +
            std::to_string(layer));
    }
    const bool split_gate_up = has_split_gate;
    std::vector<std::string> routed_names =
        split_gate_up
        ? std::vector<std::string>{
              gate_name,
              up_name,
              down_name,
          }
        : std::vector<std::string>{
              gate_up_name,
              down_name,
          };
    std::vector<bool> streamable(
        routed_names.size(),
        false);
    if (offload) {
        try {
            for (std::size_t index = 0;
                 index < routed_names.size();
                 ++index) {
                streamable[index] =
                    offload->can_offload(
                        routed_names[index]);
            }
        } catch (...) {
            for (std::size_t index = 0;
                 index < routed_names.size();
                 ++index) {
                if (streamable[index]) {
                    offload->discard_record(
                        routed_names[index]);
                }
            }
            throw;
        }
    }
    const bool stream_all =
        offload &&
        std::all_of(
            streamable.begin(),
            streamable.end(),
            [](bool value) { return value; });
    if (stream_all) {
        try {
            const auto down_info =
                offload->projection_info(
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
            bool gate_up_matches = false;
            if (split_gate_up) {
                const auto gate_info =
                    offload->projection_info(
                        gate_name);
                const auto up_info =
                    offload->projection_info(
                        up_name);
                gate_up_matches =
                    gate_info.experts == experts
                    && gate_info.out_per_expert == routed
                    && gate_info.neuron_len == hidden
                    && up_info.experts == experts
                    && up_info.out_per_expert == routed
                    && up_info.neuron_len == hidden;
            } else {
                const auto gate_info =
                    offload->projection_info(
                        gate_up_name);
                gate_up_matches =
                    gate_info.experts == experts
                    && gate_info.out_per_expert
                        == 2 * routed
                    && gate_info.neuron_len == hidden;
            }
            if (!gate_up_matches
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
                    *offload,
                    routed_names,
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
                std::nullopt,
                std::nullopt,
                offload,
                split_gate_up
                    ? std::string{}
                    : gate_up_name,
                split_gate_up
                    ? gate_name
                    : std::string{},
                split_gate_up
                    ? up_name
                    : std::string{},
                down_name,
                std::move(router_bias),
                std::move(token_experts),
                std::move(effective_available));
        } catch (...) {
            for (const auto& routed_name :
                 routed_names) {
                offload->discard_record(
                    routed_name);
            }
            throw;
        }
    }
    // A mixed representation must use the eager routed implementation for
    // both projections. Drop any projection/codebook parsed by can_offload()
    // so a streamable half does not remain resident but unused.
    if (offload) {
        for (std::size_t index = 0;
             index < routed_names.size();
             ++index) {
            if (streamable[index]) {
                offload->discard_record(
                    routed_names[index]);
            }
        }
    }
    std::optional<MlxRoutedLinear>
        routed_gate_up;
    std::optional<MlxRoutedLinear> routed_gate;
    std::optional<MlxRoutedLinear> routed_up;
    if (split_gate_up) {
        routed_gate.emplace(
            load_routed(model, gate_name));
        routed_up.emplace(
            load_routed(model, up_name));
    } else {
        routed_gate_up.emplace(
            load_routed(model, gate_up_name));
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
        std::move(routed_gate_up),
        std::move(routed_gate),
        std::move(routed_up),
        std::optional<MlxRoutedLinear>(
            load_routed(model, down_name)),
        nullptr,
        {},
        {},
        {},
        {},
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
          std::nullopt,
          std::nullopt,
          std::optional<MlxRoutedLinear>(
              std::move(routed_down)),
          nullptr,
          {},
          {},
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
        routed_gate,
    std::optional<MlxRoutedLinear>
        routed_up,
    std::optional<MlxRoutedLinear>
        routed_down,
    std::shared_ptr<MlxNintMoeOffloadCache>
        expert_offload,
    std::string streamed_gate_up_name,
    std::string streamed_gate_name,
    std::string streamed_up_name,
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
      routed_gate_(std::move(routed_gate)),
      routed_up_(std::move(routed_up)),
      routed_down_(std::move(routed_down)),
      expert_offload_(
          std::move(expert_offload)),
      streamed_gate_up_name_(
          std::move(streamed_gate_up_name)),
      streamed_gate_name_(
          std::move(streamed_gate_name)),
      streamed_up_name_(
          std::move(streamed_up_name)),
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
    const bool split_resident =
        routed_gate_.has_value()
        && routed_up_.has_value();
    if (routed_gate_.has_value() !=
        routed_up_.has_value()) {
        throw std::invalid_argument(
            "DeepSeek-V4 split routed Gate/Up "
            "residency is incomplete");
    }
    const bool combined_resident =
        routed_gate_up_.has_value();
    const bool full_resident =
        routed_down_.has_value()
        && (combined_resident != split_resident);
    const bool streamed =
        static_cast<bool>(expert_offload_);
    if (
        routed_down_.has_value()
            != (combined_resident || split_resident)
        || (combined_resident && split_resident)
        || full_resident == streamed
    ) {
        throw std::invalid_argument(
            "DeepSeek-V4 MoE requires exactly one "
            "routed expert residency mode");
    }
    if (full_resident) {
        const bool gate_up_matches =
            combined_resident
            ? routed_gate_up_->weight().experts()
                    == experts
                && routed_gate_up_->weight().neuron_len()
                    == hidden
                && routed_gate_up_->weight()
                    .out_per_expert()
                    == routed_gate_up_width
            : routed_gate_->weight().experts()
                    == experts
                && routed_gate_->weight().neuron_len()
                    == hidden
                && routed_gate_->weight()
                    .out_per_expert()
                    == routed
                && routed_up_->weight().experts()
                    == experts
                && routed_up_->weight().neuron_len()
                    == hidden
                && routed_up_->weight()
                    .out_per_expert()
                    == routed;
        if (!gate_up_matches
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
        const bool split_streamed =
            !streamed_gate_name_.empty()
            || !streamed_up_name_.empty();
        if (streamed_gate_name_.empty() !=
            streamed_up_name_.empty()) {
            throw std::invalid_argument(
                "DeepSeek-V4 split streamed Gate/Up "
                "record names are incomplete");
        }
        if (streamed_down_name_.empty()
            || (streamed_gate_up_name_.empty()
                == !split_streamed)) {
            throw std::invalid_argument(
                "DeepSeek-V4 streamed expert record "
                "names cannot be empty");
        }
        const auto down =
            expert_offload_->projection_info(
                streamed_down_name_);
        bool gate_up_matches = false;
        if (split_streamed) {
            const auto gate =
                expert_offload_->projection_info(
                    streamed_gate_name_);
            const auto up =
                expert_offload_->projection_info(
                    streamed_up_name_);
            gate_up_matches =
                gate.experts == experts
                && gate.neuron_len == hidden
                && gate.out_per_expert == routed
                && up.experts == experts
                && up.neuron_len == hidden
                && up.out_per_expert == routed;
        } else {
            const auto gate =
                expert_offload_->projection_info(
                    streamed_gate_up_name_);
            gate_up_matches =
                gate.experts == experts
                && gate.neuron_len == hidden
                && gate.out_per_expert
                    == routed_gate_up_width;
        }
        if (!gate_up_matches
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
    const char* grouped_gate_up_env = std::getenv(
        "MFQ_METAL_DSV4_SHARED_GATE_UP_GROUPED");
    const bool grouped_gate_up_enabled =
        grouped_gate_up_env == nullptr
        || std::string_view(grouped_gate_up_env) != "0";
    const char* fused_swiglu_env = std::getenv(
        "MFQ_METAL_DSV4_SHARED_SWIGLU_FUSED");
    fused_shared_swiglu_ =
        fused_swiglu_env == nullptr
        || std::string_view(fused_swiglu_env) != "0";
    const char* fused_router_env = std::getenv(
        "MFQ_METAL_DSV4_DENSE_ROUTER_FUSED");
    fused_dense_router_ =
        fused_router_env == nullptr
        || std::string_view(fused_router_env) != "0";
    if (grouped_gate_up_enabled) {
        grouped_shared_gate_up_ = make_grouped_gate_up(
            shared_gate_,
            shared_up_);
    }
}

std::vector<array>
MlxDeepseekV4Moe::project_shared(
    const array& input,
    float swiglu_limit,
    bool project_router) const {
    if (
        grouped_shared_gate_up_.has_value()
        && fused_shared_swiglu_
        && grouped_shared_gate_up_
            ->supports_single_row_swiglu(input)
    ) {
        auto shared_hidden =
            grouped_shared_gate_up_->single_row_swiglu(
                input,
                swiglu_limit);
        if (project_router) {
            return {
                router_(input),
                std::move(shared_hidden),
            };
        }
        return {std::move(shared_hidden)};
    }
    if (project_router &&
        grouped_projections_.has_value() &&
        grouped_projections_->supports(input)) {
        auto outputs =
            grouped_projections_->matmul(input);
        return {
            std::move(outputs.at(0)),
            std::move(outputs.at(1)),
            std::move(outputs.at(2)),
        };
    }
    if (
        grouped_shared_gate_up_.has_value()
        && input.size() == static_cast<std::size_t>(
            input.shape(-1))
        && grouped_shared_gate_up_->supports(input)
    ) {
        auto outputs =
            grouped_shared_gate_up_->matmul(input);
        if (project_router) {
            return {
                router_(input),
                std::move(outputs.at(0)),
                std::move(outputs.at(1)),
            };
        }
        return {
            std::move(outputs.at(0)),
            std::move(outputs.at(1)),
        };
    }
    if (project_router) {
        return {
            router_(input),
            shared_gate_(input),
            shared_up_(input),
        };
    }
    return {
        shared_gate_(input),
        shared_up_(input),
    };
}

MlxDeepseekV4MoeBranches
MlxDeepseekV4Moe::forward_branches(
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
    const auto* dense_router = router_.dense_weight_ref();
    const bool use_fused_dense_router =
        fused_dense_router_
        && !token_experts_.has_value()
        && dense_router != nullptr
        && config_.n_experts == 256
        && config_.top_k == 6
        && config_.norm_topk_prob
        && moe_dense_router_topk_supported(
            source,
            *dense_router);
    auto projections = project_shared(
        source,
        static_cast<float>(config_.swiglu_limit),
        !use_fused_dense_router);
    if (detail::component_profile_active()) {
        detail::profile_eval(
            "moe.shared_router_projections",
            projections);
    }
    std::optional<array> logits;
    std::size_t shared_offset = 0;
    if (!use_fused_dense_router) {
        logits.emplace(std::move(projections[0]));
        shared_offset = 1;
    }
    std::optional<array> fused_shared_hidden;
    const auto shared_projection_count =
        projections.size() - shared_offset;
    if (shared_projection_count == 1) {
        fused_shared_hidden.emplace(
            std::move(projections[shared_offset]));
    } else if (shared_projection_count != 2) {
        throw std::logic_error(
            "DeepSeek-V4 shared projection count is invalid");
    }
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
    if (use_fused_dense_router) {
        auto routing = moe_dense_router_topk(
            source,
            *dense_router,
            router_bias_,
            available_,
            1e-20f,
            static_cast<float>(config_.routed_scaling));
        expert_ids = std::move(routing.ids);
        expert_weights = std::move(routing.weights);
    } else if (token_experts_.has_value()) {
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
            *logits,
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
                *logits,
                expert_ids,
                config_.norm_topk_prob,
                1e-20f,
                static_cast<float>(
                    config_.routed_scaling));
    } else {
        auto routing = moe_topk(
            *logits,
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
    if (detail::component_profile_active()) {
        detail::profile_eval(
            "moe.routing",
            std::vector<array>{
                expert_ids,
                expert_weights,
            });
    }

    array routed = mlx::core::zeros(
        Shape{rows, hidden},
        source.dtype());
    if (!expert_offload_) {
        const bool split_resident =
            routed_gate_.has_value();
        const bool gate_up_grouped =
            split_resident
            ? routed_gate_->supports_grouped_vq_mmq()
                && routed_up_->supports_grouped_vq_mmq()
            : routed_gate_up_->supports_grouped_vq_mmq();
        const bool grouped_prefill =
            rows >= 32
            && gate_up_grouped
            && routed_down_->supports_grouped_vq_mmq();
        if (grouped_prefill) {
            const int routes = expert_ids.shape(1);
            auto route_order = mlx::core::contiguous(
                mlx::core::astype(
                    mlx::core::argsort(
                        mlx::core::reshape(
                            expert_ids,
                            Shape{rows * routes})),
                    mlx::core::int32));
            auto block_plan =
                split_resident
                ? routed_gate_->build_grouped_vq_mmq_plan(
                      expert_ids,
                      route_order)
                : routed_gate_up_->build_grouped_vq_mmq_plan(
                      expert_ids,
                      route_order);
            array routed_hidden = [&]() {
                if (split_resident) {
                    auto gate = routed_gate_->forward_sorted(
                        source,
                        expert_ids,
                        route_order,
                        false,
                        &block_plan);
                    auto up = routed_up_->forward_sorted(
                        source,
                        expert_ids,
                        route_order,
                        false,
                        &block_plan);
                    return limited_swiglu_pair(
                        std::move(gate),
                        std::move(up),
                        static_cast<float>(
                            config_.swiglu_limit));
                }
                return moe_limited_swiglu_split(
                    routed_gate_up_->forward_sorted(
                        source,
                        expert_ids,
                        route_order,
                        false,
                        &block_plan),
                    static_cast<float>(
                        config_.swiglu_limit));
            }();
            if (detail::component_profile_active()) {
                detail::profile_eval(
                    "moe.routed_gate_up_swiglu",
                    routed_hidden);
            }
            auto down_sorted =
                routed_down_->forward_sorted(
                    routed_hidden,
                    expert_ids,
                    route_order,
                    true,
                    &block_plan);
            if (detail::component_profile_active()) {
                detail::profile_eval(
                    "moe.routed_down",
                    down_sorted);
            }
            auto routed_pairs = mlx::core::reshape(
                mlx::core::take(
                    std::move(down_sorted),
                    mlx::core::argsort(route_order),
                    0),
                Shape{rows, routes, hidden});
            routed = moe_weighted_reduce(
                routed_pairs,
                expert_weights);
            if (detail::component_profile_active()) {
                detail::profile_eval(
                    "moe.route_reduce",
                    routed);
            }
        } else {
            array routed_hidden = [&]() {
                if (split_resident) {
                    return limited_swiglu_pair(
                        routed_gate_->forward(
                            source,
                            expert_ids),
                        routed_up_->forward(
                            source,
                            expert_ids),
                        static_cast<float>(
                            config_.swiglu_limit));
                }
                return routed_gate_up_->swiglu(
                    source,
                    expert_ids,
                    static_cast<float>(
                        config_.swiglu_limit));
            }();
            if (detail::component_profile_active()) {
                detail::profile_eval(
                    "moe.routed_gate_up_swiglu",
                    routed_hidden);
                auto routed_pairs =
                    routed_down_->forward(
                        routed_hidden,
                        expert_ids);
                detail::profile_eval(
                    "moe.routed_down",
                    routed_pairs);
                routed = moe_weighted_reduce(
                    routed_pairs,
                    expert_weights);
                detail::profile_eval(
                    "moe.route_reduce",
                    routed);
            } else {
                routed = routed_down_->combine(
                    routed_hidden,
                    expert_ids,
                    expert_weights);
            }
        }
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
            auto routed_hidden = [&]() {
                if (!streamed_gate_name_.empty()) {
                    auto gate_weight =
                        expert_offload_->grouped(
                            streamed_gate_name_,
                            selected);
                    auto up_weight =
                        expert_offload_->grouped(
                            streamed_up_name_,
                            selected);
                    return limited_swiglu_pair(
                        gate_weight.routed_matmul(
                            chunk_source,
                            chunk_ids),
                        up_weight.routed_matmul(
                            chunk_source,
                            chunk_ids),
                        static_cast<float>(
                            config_.swiglu_limit));
                }
                auto gate_weight =
                    expert_offload_->grouped(
                        streamed_gate_up_name_,
                        selected);
                return moe_limited_swiglu_split(
                    gate_weight.routed_matmul(
                        chunk_source,
                        chunk_ids),
                    static_cast<float>(
                        config_.swiglu_limit));
            }();
            auto down_weight =
                expert_offload_->grouped(
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
    auto shared_hidden = [&]() {
        if (fused_shared_hidden.has_value()) {
            return std::move(*fused_shared_hidden);
        }
        auto shared_gate_up = mlx::core::concatenate(
            {
                std::move(projections[shared_offset]),
                std::move(projections[shared_offset + 1]),
            },
            -1);
        auto result = moe_limited_swiglu_split(
            shared_gate_up,
            static_cast<float>(
                config_.swiglu_limit));
        detail::profile_eval(
            "moe.shared_swiglu",
            result);
        return result;
    }();
    auto shared = shared_down_(shared_hidden);
    detail::profile_eval(
        "moe.shared_down",
        shared);
    Shape output_shape(
        input.shape().begin(),
        input.shape().end() - 1);
    output_shape.push_back(hidden);
    return {
        mlx::core::reshape(
            std::move(routed),
            output_shape),
        mlx::core::reshape(
            std::move(shared),
            std::move(output_shape)),
        std::move(expert_ids),
        std::move(expert_weights),
    };
}

MlxDeepseekV4MoeResult
MlxDeepseekV4Moe::forward_with_routing(
    const array& input,
    const array& token_ids) const {
    auto branches = forward_branches(
        input,
        token_ids);
    auto output = branches.routed + branches.shared;
    detail::profile_eval(
        "moe.output_add",
        output);
    return {
        std::move(output),
        std::move(branches.expert_ids),
        std::move(branches.expert_weights),
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
