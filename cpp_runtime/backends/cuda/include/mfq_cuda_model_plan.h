#pragma once

#include "mfq_model_graph.h"

#include <string>
#include <string_view>
#include <utility>

namespace mfq::cuda {

// CUDA resolves semantic implementation IDs from model_graph.json.  These are
// execution implementations, not checkpoint/model aliases.
enum class MfqCudaBackbone {
    generic_qwen,
    minicpmo45,
    minicpmo_tts,
    gemma4,
    glm_dsa,
    deepseek_v4,
    unsupported,
};

enum class MfqCudaVisionAdapter {
    none,
    minicpmo45,
};

enum class MfqCudaPredictorAdapter {
    none,
};

struct MfqCudaModelPlan {
    MfqCudaBackbone backbone = MfqCudaBackbone::unsupported;
    MfqCudaVisionAdapter vision = MfqCudaVisionAdapter::none;
    MfqCudaPredictorAdapter predictor = MfqCudaPredictorAdapter::none;
};

struct MfqCudaComponentState {
    bool vision_declared = false;
    bool vision_supported = false;
    bool vision_available = false;
    bool vision_enabled = false;
    bool mtp_declared = false;
    bool mtp_supported = false;
    bool mtp_available = false;
    bool mtp_enabled = false;
};

inline constexpr MfqCudaBackbone mfq_cuda_backbone(
        std::string_view implementation) noexcept {
    if (implementation == "qwen3_5" ||
        implementation == "generic_qwen") {
        return MfqCudaBackbone::generic_qwen;
    }
    if (implementation == "minicpmo45") {
        return MfqCudaBackbone::minicpmo45;
    }
    if (implementation == "minicpmo_tts") {
        return MfqCudaBackbone::minicpmo_tts;
    }
    if (implementation == "gemma4") return MfqCudaBackbone::gemma4;
    if (implementation == "glm_dsa") return MfqCudaBackbone::glm_dsa;
    if (implementation == "deepseek_v4") {
        return MfqCudaBackbone::deepseek_v4;
    }
    return MfqCudaBackbone::unsupported;
}

inline MfqCudaModelPlan mfq_cuda_model_plan(
        const MfqModelGraph& graph) noexcept {
    MfqCudaModelPlan result;
    result.backbone = mfq_cuda_backbone(graph.backbone);
    const auto* vision = graph.component("vision");
    const auto* audio = graph.component("audio_input");
    const auto* tts = graph.component("audio_output");
    // The current CUDA MiniCPM adapter is one composite implementation.  Do
    // not advertise a partially declared graph as supported until those
    // weighted components have independent adapters.
    if (vision != nullptr && audio != nullptr && tts != nullptr &&
        vision->implementation == "minicpmo45_vision" &&
        audio->implementation == "minicpmo45_audio" &&
        tts->implementation == "minicpmo45_tts") {
        result.vision = MfqCudaVisionAdapter::minicpmo45;
    }
    // Predictor implementations are deliberately not inferred from a family
    // name. Add an adapter here only when the CUDA execution path exists.
    return result;
}

inline MfqCudaComponentState mfq_cuda_component_state(
        const MfqModelGraph& graph,
        const MfqCudaModelPlan& plan,
        bool vision_loaded,
        bool mtp_loaded,
        bool enable_vision = true,
        bool enable_mtp = true) noexcept {
    MfqCudaComponentState result;
    result.vision_declared = graph.has_component("vision");
    result.vision_supported = result.vision_declared &&
        plan.vision != MfqCudaVisionAdapter::none;
    result.vision_available = result.vision_supported && vision_loaded;
    result.vision_enabled = result.vision_available && enable_vision;
    result.mtp_declared = graph.has_component("predictor");
    result.mtp_supported = result.mtp_declared &&
        plan.predictor != MfqCudaPredictorAdapter::none;
    result.mtp_available = result.mtp_supported && mtp_loaded;
    result.mtp_enabled = result.mtp_available && enable_mtp;
    return result;
}

} // namespace mfq::cuda
