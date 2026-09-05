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
#include <time.h>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

double current_thread_cpu_seconds() noexcept {
    timespec value{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) {
        return 0.0;
    }
    return static_cast<double>(value.tv_sec) +
        static_cast<double>(value.tv_nsec) * 1.0e-9;
}

bool ssd_device_route_enabled() noexcept {
    const char* value = std::getenv("MFQ_SSD_DEVICE_ROUTE");
    return value != nullptr && std::string_view(value) != "0";
}

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
    std::optional<array> visual_router_bias;
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
    if (config.has_vision()) {
        visual_router_bias = load_dense(
            model,
            name("ffn.gate.bias_vl"),
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
                nullptr,
                layer,
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
                std::move(effective_available),
                visual_router_bias);
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
        nullptr,
        layer,
        {},
        {},
        {},
        {},
        std::move(router_bias),
        std::move(token_experts),
        available,
        std::move(visual_router_bias));
}

MlxDeepseekV4Moe MlxDeepseekV4Moe::load_named(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    const std::string& prefix,
    const std::optional<array>& available,
    std::shared_ptr<MlxNintMoeOffloadCache> offload,
    std::size_t expert_cache_layer) {
    config.validate();
    if (prefix.empty()) {
        throw std::invalid_argument(
            "DeepSeek-V4 named MoE prefix cannot be empty");
    }
    const auto name = [&prefix](std::string_view suffix) {
        return prefix + "." + std::string(suffix);
    };
    auto gate_up_name = name("ffn.experts.gate_up.weight");
    auto gate_name = name("ffn.experts.gate.weight");
    auto up_name = name("ffn.experts.up.weight");
    auto down_name = name("ffn.experts.down.weight");
    const bool split = model.contains(gate_name);
    if (split != model.contains(up_name)) {
        throw std::runtime_error(
            "DeepSeek-V4 named MoE has an incomplete split Gate/Up pair");
    }
    const std::vector<std::string> routed_names = split
        ? std::vector<std::string>{gate_name, up_name, down_name}
        : std::vector<std::string>{gate_up_name, down_name};
    std::vector<bool> streamable(routed_names.size(), false);
    if (offload) {
        try {
            for (std::size_t index = 0; index < routed_names.size(); ++index) {
                streamable[index] = offload->can_offload(routed_names[index]);
            }
        } catch (...) {
            for (std::size_t index = 0; index < routed_names.size(); ++index) {
                if (streamable[index]) {
                    offload->discard_record(routed_names[index]);
                }
            }
            throw;
        }
    }
    const bool stream_all = offload && std::all_of(
        streamable.begin(),
        streamable.end(),
        [](bool value) { return value; });
    std::optional<MlxRoutedLinear> gate_up;
    std::optional<MlxRoutedLinear> gate;
    std::optional<MlxRoutedLinear> up;
    if (!stream_all && split) {
        gate.emplace(load_routed(model, gate_name));
        up.emplace(load_routed(model, up_name));
    } else if (!stream_all) {
        gate_up.emplace(load_routed(model, gate_up_name));
    }
    std::optional<array> visual_bias;
    if (config.has_vision() &&
        model.contains(name("ffn.gate.bias_vl"))) {
        visual_bias = load_dense(
            model, name("ffn.gate.bias_vl"), mlx::core::float32);
    }
    if (stream_all) {
        try {
            auto effective_available = streamed_availability(
                available,
                *offload,
                routed_names,
                checked_int(
                    static_cast<std::size_t>(config.n_experts),
                    "expert count"));
            return MlxDeepseekV4Moe(
                config,
                MlxLinear::load(model, name("ffn.gate.weight")),
                MlxLinear::load(model, name("ffn.shared_experts.w1.weight")),
                MlxLinear::load(model, name("ffn.shared_experts.w3.weight")),
                MlxLinear::load(model, name("ffn.shared_experts.w2.weight")),
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                offload,
                nullptr,
                expert_cache_layer,
                split ? std::string{} : gate_up_name,
                split ? gate_name : std::string{},
                split ? up_name : std::string{},
                down_name,
                load_dense(model, name("ffn.gate.bias"), mlx::core::float32),
                std::nullopt,
                std::move(effective_available),
                std::move(visual_bias));
        } catch (...) {
            for (const auto& routed_name : routed_names) {
                offload->discard_record(routed_name);
            }
            throw;
        }
    }
    if (offload) {
        for (std::size_t index = 0; index < routed_names.size(); ++index) {
            if (streamable[index]) {
                offload->discard_record(routed_names[index]);
            }
        }
    }
    return MlxDeepseekV4Moe(
        config,
        MlxLinear::load(model, name("ffn.gate.weight")),
        MlxLinear::load(model, name("ffn.shared_experts.w1.weight")),
        MlxLinear::load(model, name("ffn.shared_experts.w3.weight")),
        MlxLinear::load(model, name("ffn.shared_experts.w2.weight")),
        std::move(gate_up),
        std::move(gate),
        std::move(up),
        std::optional<MlxRoutedLinear>(load_routed(model, down_name)),
        nullptr,
        nullptr,
        0,
        {},
        {},
        {},
        {},
        load_dense(model, name("ffn.gate.bias"), mlx::core::float32),
        std::nullopt,
        available,
        std::move(visual_bias));
}

MlxDeepseekV4Moe MlxDeepseekV4Moe::load_named(
    const MlxHfTensorStore& model,
    const DeepseekV4Config& config,
    const std::string& prefix,
    std::shared_ptr<MlxDeepseekV4SsdExpertCache> expert_cache,
    std::size_t expert_cache_layer,
    const std::optional<array>& available) {
    config.validate();
    if (prefix.empty() || !expert_cache) {
        throw std::invalid_argument(
            "DeepSeek-V4 named HF MoE requires a prefix and SSD cache");
    }
    const auto name = [&prefix](std::string_view suffix) {
        return prefix + "." + std::string(suffix);
    };
    std::optional<array> visual_bias;
    const auto visual_name = name("ffn.gate.bias_vl");
    if (model.checkpoint().tensors().contains(visual_name)) {
        visual_bias = model.load_dense(visual_name);
    }
    return MlxDeepseekV4Moe(
        config,
        model.load_linear(name("ffn.gate.weight")),
        model.load_linear(name("ffn.shared_experts.w1.weight")),
        model.load_linear(name("ffn.shared_experts.w3.weight")),
        model.load_linear(name("ffn.shared_experts.w2.weight")),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        nullptr,
        std::move(expert_cache),
        expert_cache_layer,
        {},
        {},
        {},
        {},
        model.load_dense(name("ffn.gate.bias")),
        std::nullopt,
        available,
        std::move(visual_bias));
}

MlxDeepseekV4Moe MlxDeepseekV4Moe::load(
    const MlxHfTensorStore& model,
    const DeepseekV4Config& config,
    std::size_t layer,
    std::shared_ptr<MlxDeepseekV4SsdExpertCache>
        expert_cache,
    const std::optional<array>& available) {
    config.validate();
    if (layer >= static_cast<std::size_t>(config.n_layers)) {
        throw std::out_of_range(
            "DeepSeek-V4 HF MoE layer index is out of range");
    }
    if (!expert_cache) {
        throw std::invalid_argument(
            "DeepSeek-V4 HF MoE requires an SSD expert cache");
    }
    const auto name = [layer](std::string_view suffix) {
        return DeepseekV4TensorNames::layer(layer, suffix);
    };
    std::optional<array> router_bias;
    std::optional<array> token_experts;
    std::optional<array> visual_router_bias;
    if (layer < static_cast<std::size_t>(config.n_hash_layers)) {
        token_experts = model.load_dense(name("ffn.gate.tid2eid"));
    } else {
        router_bias = model.load_dense(name("ffn.gate.bias"));
    }
    if (config.has_vision()) {
        visual_router_bias = model.load_dense(name("ffn.gate.bias_vl"));
    }
    return MlxDeepseekV4Moe(
        config,
        model.load_linear(name("ffn.gate.weight")),
        model.load_linear(name("ffn.shared_experts.w1.weight")),
        model.load_linear(name("ffn.shared_experts.w3.weight")),
        model.load_linear(name("ffn.shared_experts.w2.weight")),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        nullptr,
        std::move(expert_cache),
        layer,
        {},
        {},
        {},
        {},
        std::move(router_bias),
        std::move(token_experts),
        available,
        std::move(visual_router_bias));
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
    std::optional<array> available,
    std::optional<array> visual_router_bias)
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
          nullptr,
          0,
          {},
          {},
          {},
          {},
          std::move(router_bias),
          std::move(token_experts),
          std::move(available),
          std::move(visual_router_bias)) {}

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
    std::shared_ptr<MlxDeepseekV4SsdExpertCache>
        ssd_expert_cache,
    std::size_t layer,
    std::string streamed_gate_up_name,
    std::string streamed_gate_name,
    std::string streamed_up_name,
    std::string streamed_down_name,
    std::optional<array> router_bias,
    std::optional<array> token_experts,
    std::optional<array> available,
    std::optional<array> visual_router_bias)
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
      ssd_expert_cache_(
          std::move(ssd_expert_cache)),
      layer_(layer),
      streamed_gate_up_name_(
          std::move(streamed_gate_up_name)),
      streamed_gate_name_(
          std::move(streamed_gate_name)),
      streamed_up_name_(
          std::move(streamed_up_name)),
      streamed_down_name_(
          std::move(streamed_down_name)),
      router_bias_(std::move(router_bias)),
      visual_router_bias_(std::move(visual_router_bias)),
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
    const bool legacy_streamed =
        static_cast<bool>(expert_offload_);
    const bool ssd_streamed =
        static_cast<bool>(ssd_expert_cache_);
    const int residency_modes =
        static_cast<int>(full_resident)
        + static_cast<int>(legacy_streamed)
        + static_cast<int>(ssd_streamed);
    if (
        routed_down_.has_value()
            != (combined_resident || split_resident)
        || (combined_resident && split_resident)
        || residency_modes != 1
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
    } else if (legacy_streamed) {
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
    } else {
        if (config_.hidden != 4096 || config_.moe_inter != 2048 ||
            config_.n_experts != 256 ||
            layer_ >= static_cast<std::size_t>(
                config_.n_layers + config_.n_mtp_layers)) {
            throw std::invalid_argument(
                "DeepSeek-V4 SSD experts require official V4F geometry");
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
    if (visual_router_bias_.has_value()) {
        *visual_router_bias_ = mlx::core::contiguous(
            mlx::core::reshape(
                mlx::core::astype(
                    *visual_router_bias_,
                    mlx::core::float32),
                Shape{checked_int(
                    visual_router_bias_->size(),
                    "visual router bias size")}));
        if (visual_router_bias_->shape() != Shape{experts}) {
            throw std::invalid_argument(
                "DeepSeek-V4 visual router bias shape mismatch");
        }
    }
    if (config_.has_vision() != visual_router_bias_.has_value()) {
        throw std::invalid_argument(
            "DeepSeek-V4 visual routing bias does not match model config");
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
    return forward_branches(input, token_ids, nullptr);
}

std::optional<MlxDeepseekV4SsdPrefetchedLayer>
MlxDeepseekV4Moe::prefetch_routed(std::size_t rows) const {
    constexpr std::size_t kFullLayerPrefetchRows = 512;
    if (ssd_expert_cache_ && rows >= kFullLayerPrefetchRows &&
        ssd_expert_cache_->prefill_overlap_enabled()) {
        return ssd_expert_cache_->prefetch_layer(layer_);
    }
    return std::nullopt;
}

MlxDeepseekV4MoeBranches
MlxDeepseekV4Moe::forward_branches(
    const array& input,
    const array& token_ids,
    MlxDeepseekV4SsdPrefetchedLayer* prefetched) const {
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
    auto flat_token_ids = token_ids;
    if (flat_token_ids.dtype() != mlx::core::int32 &&
        flat_token_ids.dtype() != mlx::core::uint32) {
        flat_token_ids = mlx::core::astype(
            flat_token_ids,
            mlx::core::int32);
    }
    flat_token_ids = mlx::core::reshape(
        flat_token_ids, Shape{rows});
    std::optional<array> image_mask;
    if (visual_router_bias_.has_value()) {
        image_mask = mlx::core::greater_equal(
            flat_token_ids,
            array(
                checked_int(
                    static_cast<std::size_t>(config_.vocab),
                    "vocabulary size"),
                mlx::core::int32));
    }
    if (prefetched != nullptr && prefetched->layer() != layer_) {
        throw std::invalid_argument(
            "DeepSeek-V4 SSD prefetch layer mismatch");
    }
    const auto* dense_router = router_.dense_weight_ref();
    const bool use_fused_dense_router =
        fused_dense_router_
        && !token_experts_.has_value()
        && !visual_router_bias_.has_value()
        && dense_router != nullptr
        && config_.n_experts == 256
        && config_.top_k == 6
        && config_.norm_topk_prob
        && moe_dense_router_topk_supported(
            source,
            *dense_router);
    std::optional<MlxDeepseekV4SsdPageTableSnapshot>
        device_route_snapshot;
    const bool route_transaction =
        ssd_expert_cache_ &&
        ssd_expert_cache_->route_transaction_active();
    if (ssd_expert_cache_ && prefetched == nullptr && rows == 1 &&
        (route_transaction || ssd_device_route_enabled()) &&
        (use_fused_dense_router || route_transaction)) {
        device_route_snapshot.emplace(
            ssd_expert_cache_->snapshot_page_table(layer_));
    }
    auto projections = project_shared(
        source,
        static_cast<float>(config_.swiglu_limit),
        !use_fused_dense_router);
    if (prefetched != nullptr) {
        // The SSD workers are already reading the routed bank. Submit the
        // independent shared/router projections while those reads are in
        // flight, then wait only for the residual storage latency below.
        detail::eval_with_timing(projections);
    }
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
    bool packed_expert_ids = false;
    if (use_fused_dense_router) {
        if (device_route_snapshot.has_value()) {
            auto routing = moe_dense_router_topk_packed(
                source,
                *dense_router,
                device_route_snapshot->slot_ids(),
                router_bias_,
                available_,
                1e-20f,
                static_cast<float>(config_.routed_scaling));
            expert_ids = std::move(routing.ids);
            expert_weights = std::move(routing.weights);
            packed_expert_ids = true;
        } else {
            auto routing = moe_dense_router_topk(
                source,
                *dense_router,
                router_bias_,
                available_,
                1e-20f,
                static_cast<float>(config_.routed_scaling));
            expert_ids = std::move(routing.ids);
            expert_weights = std::move(routing.weights);
        }
    } else if (token_experts_.has_value()) {
        auto ids = flat_token_ids;
        if (image_mask.has_value()) {
            ids = mlx::core::where(
                *image_mask,
                mlx::core::zeros_like(ids),
                ids);
        }
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
        if (visual_router_bias_.has_value()) {
            auto visual = moe_topk(
                *logits,
                checked_int(
                    static_cast<std::size_t>(config_.top_k),
                    "route count"),
                false,
                true,
                config_.norm_topk_prob,
                false,
                visual_router_bias_,
                available_,
                1e-20f,
                static_cast<float>(config_.routed_scaling));
            expert_ids = mlx::core::where(
                mlx::core::expand_dims(*image_mask, -1),
                visual.ids,
                expert_ids);
        }
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
        if (visual_router_bias_.has_value()) {
            auto visual = moe_topk(
                *logits,
                checked_int(
                    static_cast<std::size_t>(config_.top_k),
                    "route count"),
                false,
                true,
                config_.norm_topk_prob,
                false,
                visual_router_bias_,
                available_,
                1e-20f,
                static_cast<float>(config_.routed_scaling));
            const auto route_mask =
                mlx::core::expand_dims(*image_mask, -1);
            expert_ids = mlx::core::where(
                route_mask, visual.ids, expert_ids);
            expert_weights = mlx::core::where(
                route_mask, visual.weights, expert_weights);
        }
    }
    if (device_route_snapshot.has_value() && !packed_expert_ids) {
        const auto slots = mlx::core::take(
            device_route_snapshot->slot_ids(),
            expert_ids,
            0);
        expert_ids = mlx::core::bitwise_or(
            expert_ids,
            mlx::core::left_shift(
                mlx::core::add(
                    slots,
                    array(1, mlx::core::int32)),
                array(8, mlx::core::int32)));
        packed_expert_ids = true;
    }
    if (detail::component_profile_active()) {
        detail::profile_eval(
            "moe.routing",
            std::vector<array>{
                expert_ids,
                expert_weights,
            });
    }

    const auto build_shared = [&]() {
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
                static_cast<float>(config_.swiglu_limit));
            detail::profile_eval(
                "moe.shared_swiglu",
                result);
            return result;
        }();
        auto result = shared_down_(shared_hidden);
        detail::profile_eval(
            "moe.shared_down",
            result);
        return result;
    };
    std::optional<array> shared;

    array routed = mlx::core::zeros(
        Shape{rows, hidden},
        source.dtype());
    const auto run_resident = [
        &source,
        &expert_ids,
        &expert_weights,
        rows,
        hidden,
        this
    ](
        const MlxRoutedLinear* gate_up,
        const MlxRoutedLinear* gate,
        const MlxRoutedLinear* up,
        const MlxRoutedLinear* down,
        const array* selected_ids = nullptr,
        const array* selected_weights = nullptr,
        const array* precomputed_hidden = nullptr,
        const array* expert_map = nullptr,
        bool packed_expert_ids = false) {
        const auto& route_ids = selected_ids != nullptr
            ? *selected_ids
            : expert_ids;
        const auto& route_weights = selected_weights != nullptr
            ? *selected_weights
            : expert_weights;
        const bool split_resident =
            gate != nullptr;
        if ((gate == nullptr) != (up == nullptr) || down == nullptr ||
            (split_resident == (gate_up != nullptr))) {
            throw std::logic_error(
                "DeepSeek-V4 routed weight set is incomplete");
        }
        const bool gate_up_grouped =
            split_resident
            ? gate->supports_grouped_vq_mmq()
                && up->supports_grouped_vq_mmq()
            : gate_up->supports_grouped_vq_mmq();
        const bool grouped_prefill =
            rows >= 32
            && gate_up_grouped
            && down->supports_grouped_vq_mmq();
        const bool smallm_nax =
            !split_resident
            && precomputed_hidden == nullptr
            && expert_map == nullptr
            && !packed_expert_ids
            && gate_up->prefers_mxfp4_smallm_nax(route_ids)
            && down->prefers_mxfp4_smallm_nax(route_ids);
        if (smallm_nax) {
            const int routes = route_ids.shape(1);
            auto route_order = mlx::core::contiguous(
                mlx::core::astype(
                    mlx::core::argsort(
                        mlx::core::reshape(
                            route_ids,
                            Shape{rows * routes})),
                    mlx::core::int32));
            auto routed_hidden = gate_up->swiglu_sorted(
                source,
                route_ids,
                route_order,
                static_cast<float>(config_.swiglu_limit),
                true);
            if (detail::component_profile_active()) {
                detail::profile_eval(
                    "moe.routed_gate_up_swiglu",
                    routed_hidden);
            }
            auto down_sorted = down->forward_sorted(
                routed_hidden,
                route_ids,
                route_order,
                true,
                nullptr,
                true);
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
            auto output = moe_weighted_reduce(
                routed_pairs,
                route_weights);
            if (detail::component_profile_active()) {
                detail::profile_eval(
                    "moe.route_reduce",
                    output);
            }
            return output;
        } else if (grouped_prefill) {
            const int routes = route_ids.shape(1);
            auto route_order = mlx::core::contiguous(
                mlx::core::astype(
                    mlx::core::argsort(
                        mlx::core::reshape(
                            route_ids,
                            Shape{rows * routes})),
                    mlx::core::int32));
            auto block_plan =
                split_resident
                ? gate->build_grouped_vq_mmq_plan(
                      route_ids,
                      route_order)
                : gate_up->build_grouped_vq_mmq_plan(
                      route_ids,
                      route_order);
            array routed_hidden = [&]() {
                if (split_resident) {
                    auto gate_output = gate->forward_sorted(
                        source,
                        route_ids,
                        route_order,
                        false,
                        &block_plan);
                    auto up_output = up->forward_sorted(
                        source,
                        route_ids,
                        route_order,
                        false,
                        &block_plan);
                    return limited_swiglu_pair(
                        std::move(gate_output),
                        std::move(up_output),
                        static_cast<float>(
                            config_.swiglu_limit));
                }
                return moe_limited_swiglu_split(
                    gate_up->forward_sorted(
                        source,
                        route_ids,
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
                down->forward_sorted(
                    routed_hidden,
                    route_ids,
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
            auto output = moe_weighted_reduce(
                routed_pairs,
                route_weights);
            if (detail::component_profile_active()) {
                detail::profile_eval(
                    "moe.route_reduce",
                    output);
            }
            return output;
        } else {
            array routed_hidden = precomputed_hidden != nullptr
                ? *precomputed_hidden
                : [&]() {
                    if (split_resident) {
                        return limited_swiglu_pair(
                            expert_map != nullptr
                                ? gate->forward_mapped(
                                      source,
                                      route_ids,
                                      *expert_map)
                                : packed_expert_ids
                                    ? gate->forward_packed(
                                          source,
                                          route_ids)
                                    : gate->forward(source, route_ids),
                            expert_map != nullptr
                                ? up->forward_mapped(
                                      source,
                                      route_ids,
                                      *expert_map)
                                : packed_expert_ids
                                    ? up->forward_packed(
                                          source,
                                          route_ids)
                                    : up->forward(source, route_ids),
                            static_cast<float>(
                                config_.swiglu_limit));
                    }
                    return expert_map != nullptr
                        ? gate_up->swiglu_mapped(
                              source,
                              route_ids,
                              *expert_map,
                              static_cast<float>(
                                  config_.swiglu_limit))
                        : packed_expert_ids
                            ? gate_up->swiglu_packed(
                                  source,
                                  route_ids,
                                  static_cast<float>(
                                      config_.swiglu_limit))
                            : gate_up->swiglu(
                                  source,
                                  route_ids,
                                  static_cast<float>(
                                      config_.swiglu_limit));
                }();
            if (detail::component_profile_active()) {
                detail::profile_eval(
                    "moe.routed_gate_up_swiglu",
                    routed_hidden);
                auto routed_pairs = expert_map != nullptr
                    ? down->forward_mapped(
                          routed_hidden,
                          route_ids,
                          *expert_map)
                    : packed_expert_ids
                        ? down->forward_packed(
                              routed_hidden,
                              route_ids)
                        : down->forward(routed_hidden, route_ids);
                detail::profile_eval(
                    "moe.routed_down",
                    routed_pairs);
                auto output = moe_weighted_reduce(
                    routed_pairs,
                    route_weights);
                detail::profile_eval(
                    "moe.route_reduce",
                    output);
                return output;
            } else {
                return expert_map != nullptr
                    ? down->combine_mapped(
                          routed_hidden,
                          route_ids,
                          *expert_map,
                          route_weights)
                    : packed_expert_ids
                        ? down->combine_packed(
                              routed_hidden,
                              route_ids,
                              route_weights)
                        : down->combine(
                              routed_hidden,
                              route_ids,
                              route_weights);
            }
        }
    };
    if (ssd_expert_cache_) {
        if (prefetched != nullptr) {
            const auto& weights = prefetched->wait();
            routed = run_resident(
                &weights.gate_up,
                nullptr,
                nullptr,
                &weights.down);
            detail::eval_with_timing(routed);
        } else {
            bool device_route_hit = false;
            std::optional<array> evaluated_host_ids;
            std::optional<std::vector<std::int32_t>> evaluated_active;
            if (device_route_snapshot.has_value() &&
                packed_expert_ids && route_transaction) {
                auto& snapshot = *device_route_snapshot;
                shared.emplace(build_shared());
                routed = run_resident(
                    &snapshot.weights().gate_up,
                    nullptr,
                    nullptr,
                    &snapshot.weights().down,
                    &expert_ids,
                    &expert_weights,
                    nullptr,
                    nullptr,
                    true);
                snapshot.defer_transaction(expert_ids);
                device_route_hit = true;
            }
            if (device_route_snapshot.has_value() &&
                packed_expert_ids && !device_route_hit) {
                auto& snapshot = *device_route_snapshot;
                auto host_ids = mlx::core::contiguous(
                    mlx::core::astype(expert_ids, mlx::core::int32));
                const auto eval_begin =
                    std::chrono::steady_clock::now();
                detail::eval_with_timing(
                    std::vector<array>{
                        host_ids,
                        expert_weights,
                    });
                const double eval_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        eval_begin).count();
                const auto host_begin =
                    std::chrono::steady_clock::now();
                std::vector<std::int32_t> global_ids(host_ids.size());
                std::transform(
                    host_ids.data<std::int32_t>(),
                    host_ids.data<std::int32_t>() + host_ids.size(),
                    global_ids.begin(),
                    [](std::int32_t encoded) { return encoded & 0xff; });
                std::vector<std::int32_t> active(global_ids);
                active.erase(
                    std::remove_if(
                        active.begin(),
                        active.end(),
                        [](std::int32_t expert) { return expert < 0; }),
                    active.end());
                std::sort(active.begin(), active.end());
                active.erase(
                    std::unique(active.begin(), active.end()),
                    active.end());
                device_route_hit = std::all_of(
                    host_ids.data<std::int32_t>(),
                    host_ids.data<std::int32_t>() + host_ids.size(),
                    [](std::int32_t encoded) {
                        return (encoded >> 8) - 1 >= 0;
                    });
                const double host_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        host_begin).count();
                if (device_route_hit) {
                    shared.emplace(build_shared());
                    routed = run_resident(
                        &snapshot.weights().gate_up,
                        nullptr,
                        nullptr,
                        &snapshot.weights().down,
                        &expert_ids,
                        &expert_weights,
                        nullptr,
                        nullptr,
                        true);
                    snapshot.defer_finish(
                        active,
                        eval_seconds,
                        host_seconds);
                } else {
                    snapshot.finish(
                        active,
                        false,
                        eval_seconds,
                        host_seconds);
                    evaluated_host_ids.emplace(
                        array(global_ids.begin(), expert_ids.shape()));
                    evaluated_active.emplace(std::move(active));
                }
            }
            if (!device_route_hit) {
            auto host_ids = evaluated_host_ids.has_value()
                ? std::move(*evaluated_host_ids)
                : mlx::core::contiguous(
                      mlx::core::astype(expert_ids, mlx::core::int32));
            std::vector<std::int32_t> active;
            if (evaluated_active.has_value()) {
                active = std::move(*evaluated_active);
            } else {
                const auto route_sync_begin =
                    std::chrono::steady_clock::now();
                const double route_sync_cpu_begin =
                    current_thread_cpu_seconds();
                detail::eval_with_timing(host_ids);
                const double route_sync_cpu_seconds =
                    current_thread_cpu_seconds() - route_sync_cpu_begin;
                const double route_sync_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        route_sync_begin).count();
                const auto route_host_begin =
                    std::chrono::steady_clock::now();
                active.assign(
                    host_ids.data<std::int32_t>(),
                    host_ids.data<std::int32_t>() + host_ids.size());
                active.erase(
                    std::remove_if(
                        active.begin(),
                        active.end(),
                        [](std::int32_t expert) { return expert < 0; }),
                    active.end());
                std::sort(active.begin(), active.end());
                active.erase(
                    std::unique(active.begin(), active.end()),
                    active.end());
                const double route_host_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        route_host_begin).count();
                ssd_expert_cache_->record_route_timing(
                    route_sync_seconds,
                    route_sync_cpu_seconds,
                    route_host_seconds);
            }
            if (!shared.has_value()) {
                shared.emplace(build_shared());
            }
            std::optional<array> ready_gate_up;
            std::optional<array> ready_down;
            std::optional<array> pending_gate_up;
            std::optional<array> pending_slot_ids;
            std::vector<std::int32_t> ready_positions;
            std::vector<std::int32_t> pending_positions;
            auto prepared = ssd_expert_cache_->prepare(
                layer_,
                active,
                [
                    &active,
                    &host_ids,
                    &pending_positions,
                    &ready_gate_up,
                    &ready_down,
                    &ready_positions,
                    &shared,
                    &source,
                    rows,
                    this
                ](
                    const MlxDeepseekV4SsdExpertWeights& weights,
                    std::span<const std::int32_t> ready_experts,
                    std::span<const std::int32_t> slot_for_expert) {
                    std::vector<array> overlap_values;
                    overlap_values.push_back(*shared);
                    if (rows == 1 &&
                        ready_experts.size() < active.size()) {
                        const auto* ids = host_ids.data<std::int32_t>();
                        const auto routes = host_ids.size();
                        ready_positions.reserve(routes);
                        pending_positions.reserve(routes);
                        for (std::size_t position = 0;
                             position < routes;
                             ++position) {
                            const auto expert = ids[position];
                            if (expert < 0 || std::binary_search(
                                    ready_experts.begin(),
                                    ready_experts.end(),
                                    expert)) {
                                ready_positions.push_back(
                                    static_cast<std::int32_t>(position));
                            } else {
                                pending_positions.push_back(
                                    static_cast<std::int32_t>(position));
                            }
                        }
                        if (!ready_positions.empty()) {
                            std::vector<std::int32_t> ready_slots;
                            ready_slots.reserve(ready_positions.size());
                            for (const auto position : ready_positions) {
                                const auto expert = ids[position];
                                ready_slots.push_back(expert < 0
                                    ? -1
                                    : slot_for_expert[
                                          static_cast<std::size_t>(expert)]);
                            }
                            const array ready_ids(
                                ready_slots.begin(),
                                Shape{
                                    1,
                                    static_cast<int>(ready_slots.size()),
                                });
                            ready_gate_up.emplace(
                                weights.gate_up.swiglu(
                                    source,
                                    ready_ids,
                                    static_cast<float>(
                                        config_.swiglu_limit)));
                            ready_down.emplace(
                                weights.down.forward(
                                    *ready_gate_up,
                                    ready_ids));
                            overlap_values.push_back(*ready_down);
                        }
                    }
                    detail::eval_with_timing(
                        std::move(overlap_values));
                },
                [
                    &host_ids,
                    &pending_gate_up,
                    &pending_positions,
                    &pending_slot_ids,
                    &source,
                    rows,
                    this
                ](
                    const MlxDeepseekV4SsdExpertWeights& weights,
                    std::span<const std::int32_t>,
                    std::span<const std::int32_t> slot_for_expert) {
                    if (rows != 1 || pending_positions.empty()) {
                        return;
                    }
                    const auto* ids = host_ids.data<std::int32_t>();
                    std::vector<std::int32_t> pending_slots;
                    pending_slots.reserve(pending_positions.size());
                    for (const auto position : pending_positions) {
                        pending_slots.push_back(slot_for_expert[
                            static_cast<std::size_t>(ids[position])]);
                    }
                    pending_slot_ids.emplace(
                        pending_slots.begin(),
                        Shape{
                            1,
                            static_cast<int>(pending_slots.size()),
                        });
                    pending_gate_up.emplace(
                        weights.gate_up.swiglu(
                            source,
                            *pending_slot_ids,
                            static_cast<float>(config_.swiglu_limit)));
                    detail::eval_with_timing(*pending_gate_up);
                });
            const auto& weights = prepared.weights();
            const auto slot_for_expert = prepared.slot_for_expert();
            const auto* global_ids = host_ids.data<std::int32_t>();
            std::vector<std::int32_t> resident_slot_values(host_ids.size());
            for (std::size_t index = 0; index < host_ids.size(); ++index) {
                const auto expert = global_ids[index];
                resident_slot_values[index] = expert < 0
                    ? -1
                    : slot_for_expert[static_cast<std::size_t>(expert)];
            }
            const array resident_ids(
                resident_slot_values.begin(),
                expert_ids.shape());
            if (ready_down.has_value() &&
                pending_gate_up.has_value()) {
                auto pending_down = weights.down.forward(
                    *pending_gate_up,
                    *pending_slot_ids);
                std::vector<std::int32_t> reorder(
                    ready_positions.size() + pending_positions.size());
                for (std::size_t index = 0;
                     index < ready_positions.size();
                     ++index) {
                    reorder[static_cast<std::size_t>(
                        ready_positions[index])] =
                            static_cast<std::int32_t>(index);
                }
                for (std::size_t index = 0;
                     index < pending_positions.size();
                     ++index) {
                    reorder[static_cast<std::size_t>(
                        pending_positions[index])] =
                            static_cast<std::int32_t>(
                                ready_positions.size() + index);
                }
                auto routed_pairs = mlx::core::take(
                    mlx::core::concatenate(
                        {
                            std::move(*ready_down),
                            std::move(pending_down),
                        },
                        1),
                    array(
                        reorder.begin(),
                        Shape{static_cast<int>(reorder.size())}),
                    1);
                routed = moe_weighted_reduce(
                    routed_pairs,
                    expert_weights);
            } else if (pending_gate_up.has_value()) {
                routed = run_resident(
                    &weights.gate_up,
                    nullptr,
                    nullptr,
                    &weights.down,
                    &resident_ids,
                    &expert_weights,
                    &*pending_gate_up);
            } else {
                routed = run_resident(
                    &weights.gate_up,
                    nullptr,
                    nullptr,
                    &weights.down,
                    &resident_ids,
                    &expert_weights);
            }
            }
        }
    } else if (!expert_offload_) {
        routed = run_resident(
            routed_gate_up_ ? &*routed_gate_up_ : nullptr,
            routed_gate_ ? &*routed_gate_ : nullptr,
            routed_up_ ? &*routed_up_ : nullptr,
            routed_down_ ? &*routed_down_ : nullptr);
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
    if (!shared.has_value()) {
        shared.emplace(build_shared());
    }
    Shape output_shape(
        input.shape().begin(),
        input.shape().end() - 1);
    output_shape.push_back(hidden);
    return {
        mlx::core::reshape(
            std::move(routed),
            output_shape),
        mlx::core::reshape(
            std::move(*shared),
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
