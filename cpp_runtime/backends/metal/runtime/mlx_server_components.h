#pragma once

#include "mlx_deepseek_v4_causal_lm.h"
#include "mlx_minicpmo45.h"
#include "mlx_qwen35_causal_lm.h"

#include "mfq_model_graph.h"
#include "mfq/server.h"

#include <memory>
#include <mutex>
#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

// Backend-specific payload translation lives behind this adapter. The HTTP
// service consumes only these common callbacks and the capabilities recovered
// from model_graph.json; it never selects a component by model name.
struct MlxServerComponentCallbacks {
    MfqMultimodalGenerateFn multimodal_generate;
    MfqDuplexBackend duplex;
    bool mtp_available = false;
};

MlxServerComponentCallbacks make_mlx_server_components(
    const MfqModelGraph* graph,
    std::shared_ptr<std::mutex> runtime_mutex,
    std::shared_ptr<std::optional<MlxQwen35CausalLm>> runtime,
    mlx::core::Stream runtime_stream);

MlxServerComponentCallbacks make_mlx_server_components(
    const MfqModelGraph* graph,
    std::shared_ptr<std::mutex> runtime_mutex,
    std::shared_ptr<std::optional<MlxMiniCPMO45Runtime>> runtime,
    mlx::core::Stream runtime_stream);

MlxServerComponentCallbacks make_mlx_server_components(
    const MfqModelGraph* graph,
    std::shared_ptr<std::mutex> runtime_mutex,
    std::shared_ptr<std::optional<MlxDeepseekV4CausalLm>> runtime,
    mlx::core::Stream runtime_stream);

} // namespace mfq::metal
