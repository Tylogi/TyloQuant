#pragma once

#include "deepseek_v4_model.h"
#include "mlx_grouped_linear.h"
#include "mlx_moe.h"
#include "mlx_tensor.h"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV4MoeResult {
    mlx::core::array output;
    mlx::core::array expert_ids;
    mlx::core::array expert_weights;
};

// Complete DeepSeek-V4 shared plus routed expert subgraph. Ordinary router,
// shared gate and shared up projections use one grouped packed dispatch when
// their formats support it. Routed gate/up and down remain NINTM-native.
class MlxDeepseekV4Moe {
public:
    static MlxDeepseekV4Moe load(
        const MfqContainer& model,
        const DeepseekV4Config& config,
        std::size_t layer,
        const std::optional<mlx::core::array>& available =
            std::nullopt,
        std::shared_ptr<MlxNintMoeOffloadCache> offload =
            nullptr);

    MlxDeepseekV4Moe(
        DeepseekV4Config config,
        MlxLinear router,
        MlxLinear shared_gate,
        MlxLinear shared_up,
        MlxLinear shared_down,
        MlxRoutedLinear routed_gate_up,
        MlxRoutedLinear routed_down,
        std::optional<mlx::core::array> router_bias =
            std::nullopt,
        std::optional<mlx::core::array> token_experts =
            std::nullopt,
        std::optional<mlx::core::array> available =
            std::nullopt);

    MlxDeepseekV4MoeResult forward_with_routing(
        const mlx::core::array& input,
        const mlx::core::array& token_ids) const;

    mlx::core::array forward(
        const mlx::core::array& input,
        const mlx::core::array& token_ids) const;

    mlx::core::array operator()(
        const mlx::core::array& input,
        const mlx::core::array& token_ids) const {
        return forward(input, token_ids);
    }

    bool uses_grouped_shared_projection() const noexcept {
        return grouped_projections_.has_value();
    }
    bool uses_streamed_experts() const noexcept {
        return static_cast<bool>(expert_offload_);
    }

private:
    MlxDeepseekV4Moe(
        DeepseekV4Config config,
        MlxLinear router,
        MlxLinear shared_gate,
        MlxLinear shared_up,
        MlxLinear shared_down,
        std::optional<MlxRoutedLinear> routed_gate_up,
        std::optional<MlxRoutedLinear> routed_down,
        std::shared_ptr<MlxNintMoeOffloadCache>
            expert_offload,
        std::string streamed_gate_up_name,
        std::string streamed_down_name,
        std::optional<mlx::core::array> router_bias,
        std::optional<mlx::core::array> token_experts,
        std::optional<mlx::core::array> available);

    std::array<mlx::core::array, 3> project_shared(
        const mlx::core::array& input) const;

    DeepseekV4Config config_;
    MlxLinear router_;
    MlxLinear shared_gate_;
    MlxLinear shared_up_;
    MlxLinear shared_down_;
    std::optional<MlxRoutedLinear> routed_gate_up_;
    std::optional<MlxRoutedLinear> routed_down_;
    std::shared_ptr<MlxNintMoeOffloadCache>
        expert_offload_;
    std::string streamed_gate_up_name_;
    std::string streamed_down_name_;
    std::optional<MlxGroupedLinear> grouped_projections_;
    std::optional<mlx::core::array> router_bias_;
    std::optional<mlx::core::array> token_experts_;
    mlx::core::array available_;
    int available_count_ = 0;
};

} // namespace mfq::metal
