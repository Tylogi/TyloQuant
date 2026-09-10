#pragma once

#include "deepseek_v41_model.h"
#include "mlx_moe.h"
#include "mlx_tensor.h"

#include <optional>
#include <string>

#include <mlx/mlx.h>

namespace mfq::metal {

// DeepSeek-V4.1 routed + shared expert block.  The architecture owns router
// semantics; only the heterogeneous NINTM projection primitive is shared.
class MlxDeepseekV41Moe {
public:
    static MlxDeepseekV41Moe load(
        const MfqContainer& model,
        const DeepseekV41Config& config,
        const std::string& prefix,
        bool predictor = false);

    mlx::core::array forward(
        const mlx::core::array& input,
        const std::optional<mlx::core::array>& image_mask = std::nullopt) const;

private:
    MlxDeepseekV41Moe(
        int hidden,
        int intermediate,
        int experts,
        int top_k,
        bool normalize,
        float scale,
        float swiglu_limit,
        MlxLinear router,
        mlx::core::array router_bias,
        std::optional<mlx::core::array> vision_bias,
        MlxLinear shared_gate,
        MlxLinear shared_up,
        MlxLinear shared_down,
        MlxRoutedLinear routed_gate_up,
        MlxRoutedLinear routed_down);

    int hidden_;
    int intermediate_;
    int experts_;
    int top_k_;
    bool normalize_;
    float scale_;
    float swiglu_limit_;
    MlxLinear router_;
    mlx::core::array router_bias_;
    std::optional<mlx::core::array> vision_bias_;
    MlxLinear shared_gate_;
    MlxLinear shared_up_;
    MlxLinear shared_down_;
    MlxRoutedLinear routed_gate_up_;
    MlxRoutedLinear routed_down_;
};

} // namespace mfq::metal
