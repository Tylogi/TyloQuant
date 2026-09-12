#pragma once

#include "deepseek_v41_model.h"
#include "mlx_deepseek_hc.h"
#include "mlx_tensor.h"
#include "mlx_transformer.h"

#include <string>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV41MhcResult {
    mlx::core::array branch;
    mlx::core::array next_pre;
    MlxDeepseekV4HcPreResult expansion;
};

// Single-pass Mega-mHC adapter. V4.1 deliberately carries the pre-mix from
// the preceding sublayer; this differs from the older V4 layer lifecycle.
class MlxDeepseekV41Mhc {
public:
    static MlxDeepseekV41Mhc load(
        const MfqContainer& model,
        const DeepseekV41Config& config,
        const std::string& prefix,
        const std::string& norm_name);

    MlxDeepseekV41MhcResult collapse(
        const mlx::core::array& residual,
        const mlx::core::array& previous_pre) const;

    mlx::core::array expand(
        const mlx::core::array& branch,
        const mlx::core::array& residual,
        const MlxDeepseekV4HcPreResult& expansion) const;

    static mlx::core::array identity_pre(
        int batch,
        int tokens);

private:
    MlxDeepseekV41Mhc(
        DeepseekV41Config config,
        MlxLinear function,
        mlx::core::array base,
        mlx::core::array scale,
        mlx::core::array norm);

    DeepseekV41Config config_;
    MlxLinear function_;
    mlx::core::array base_;
    mlx::core::array scale_;
    MlxRmsNorm norm_;
};

} // namespace mfq::metal
